"""Faithfulness scoring: the judge seam, the fail-closed paths, and the score itself.

Uses a deterministic fake judge through the `Judge` protocol. No test here calls
ollama, and that is enforced rather than trusted: `_no_network` patches `urlopen` for
every test in the file, so a code path that reaches for a model fails the suite
instead of quietly making it slow and machine-dependent. Same reason `test_embed.py`
uses a fake embedder -- the scoring, the label mapping, the store integration and the
evidence window are all arithmetic, and a real model adds nothing to those assertions
except minutes and variance.

Every test names a rule that would otherwise fail silently, and every rule here was
checked by deleting it and confirming the test went red. A test that survives
removing the behaviour it names is not a test.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.eval.faithfulness import (
    LABEL_NOT_SUPPORTED,
    LABEL_SUPPORTED,
    LABEL_UNCERTAIN,
    LABELS,
    Adjudication,
    FaithfulnessReport,
    Judgement,
    JudgeUnavailable,
    OllamaJudge,
    adjudicate,
    adjudicate_assertion,
    build_prompt,
    faithfulness,
    normalise_label,
    parse_judgement,
    render_evidence,
)

SOURCE = (
    'def acquire(parcel_id):\n'
    '    """Take a lease."""\n'
    '    if parcel_id is None:\n'
    '        return False\n'
    '    return True\n'
    '\n'
    '\n'
    'def release(parcel_id):\n'
    '    """Drop a lease, writing a tombstone."""\n'
    '    return False\n'
)

ACQUIRE_END = SOURCE.index("\n\n\n") + 1
RELEASE_START = SOURCE.index("def release")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach a model. Enforced, not assumed.

    An LLM-judged metric is the easiest thing in a repo to accidentally make
    network-dependent: one forgotten default and the suite passes on the machine that
    happens to have ollama running and hangs everywhere else. Patching `urlopen` to
    raise means that mistake fails immediately and names itself.
    """

    def _refuse(*args, **kwargs):
        raise urllib.error.URLError("tests must not reach a model")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


@pytest.fixture
def repo(tmp_path):
    """A one-file repo plus an index bound to it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    conn = db.init_db(tmp_path / "index.db")
    db.bind_repo_root(conn, root)
    return root, conn


class FakeJudge:
    """Supports a claim only when every `backticked` token in it appears in the
    evidence it was shown.

    Deterministic and dependency-free, so faithfulness is asserted against arithmetic
    anyone can verify by reading the test. It records every (subject, claim, evidence)
    triple it was handed, which is how the evidence window gets asserted: the fake
    judge is the only place a test can observe exactly what a real judge would see.

    `forced` overrides the rule for claims containing a given substring, so the
    label-routing and ordering tests can name a verdict directly instead of
    contriving evidence to produce it.
    """

    def __init__(self, forced: dict[str, str] | None = None, name: str = "fake-judge/v1") -> None:
        self._name = name
        self._forced = forced or {}
        self.seen: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def judge(self, *, claim: str, evidence: str, subject: str) -> Judgement:
        self.seen.append((subject, claim, evidence))
        for needle, label in self._forced.items():
            if needle in claim:
                return Judgement(label, f"forced {label} for {needle!r}", self._name)
        tokens = re.findall(r"`([^`]+)`", claim)
        if not tokens:
            return Judgement(
                LABEL_NOT_SUPPORTED, "the claim quotes nothing checkable", self._name
            )
        missing = [t for t in tokens if t not in evidence]
        if missing:
            return Judgement(
                LABEL_NOT_SUPPORTED, f"evidence does not contain {missing}", self._name
            )
        return Judgement(LABEL_SUPPORTED, "every quoted token is in the evidence", self._name)


def _acquire_span(root):
    return store.span_for(root, "leases.py", 0, ACQUIRE_END)


def _release_span(root):
    return store.span_for(root, "leases.py", RELEASE_START, len(SOURCE))


def _admit(conn, spans, claim, *, qualname="leases.acquire"):
    return store.write_assertion(
        conn,
        subject_qualname=qualname,
        kind="purpose",
        claim=claim,
        spans=spans,
        generator="claude/test",
        confidence=0.9,
    )


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------


def test_faithfulness_is_the_supported_fraction(repo):
    """The RAGAS-style metric: supported / adjudicated, and nothing cleverer.

    Two of three claims quote a token that is in their cited span; the third quotes
    `tombstone`, which is in the file but not in the span it cited."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "acquire can `return False`")
    _admit(conn, [_acquire_span(root)], "acquire can `return True`")
    _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")

    report = adjudicate(conn, FakeJudge(), root)

    assert len(report) == 3
    assert report.count(LABEL_SUPPORTED) == 2
    assert report.score == pytest.approx(2 / 3)


def test_the_score_of_an_empty_set_is_none_not_one(repo):
    """'Every claim was supported' is trivially true of no claims. A store with
    nothing in it must report that, not a perfect score -- this repo has already been
    bitten once by a vacuous truth reading as success."""
    root, conn = repo
    report = adjudicate(conn, FakeJudge(), root)
    assert len(report) == 0
    assert report.score is None
    assert "n/a" in report.summary()
    assert faithfulness([]) is None


def test_the_metric_can_be_recomputed_over_a_subset(repo):
    """A single number over a whole store cannot tell one generator, or one kind of
    claim, from another. The free function exists so a subset can be scored without
    re-running a judge over it."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "acquire can `return True`")
    _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")
    report = adjudicate(conn, FakeJudge(), root)

    assert faithfulness(report.adjudications) == report.score
    assert faithfulness([a for a in report.adjudications if a.supported]) == 1.0
    assert faithfulness(report.unfaithful) == 0.0


def test_a_low_score_is_diagnosable_not_just_low(repo):
    """The per-assertion detail is the deliverable, not a courtesy. A faithfulness
    number with nothing attached is exactly as trustworthy as the judge, with no way
    for a reader to form a second opinion about that."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")

    report = adjudicate(conn, FakeJudge(), root)
    failed = report.unfaithful

    assert [a.assertion_id for a in failed] == [aid]
    assert failed[0].claim == "acquire writes a `tombstone`"
    assert failed[0].citations == ("leases.py:1-5",)
    assert "tombstone" in failed[0].judgement.reasoning
    assert "tombstone" in report.format_report()


def test_unfaithful_puts_refusals_before_the_judges_own_failures(repo):
    """'The citation does not carry the claim' is evidence about the generator.
    'I could not tell' is evidence about the judge. Reading them in that order is
    what makes the list a diagnosis rather than a pile."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "claim U is `return True`")
    _admit(conn, [_acquire_span(root)], "claim N is `return True`")
    judge = FakeJudge(forced={"claim U": LABEL_UNCERTAIN, "claim N": LABEL_NOT_SUPPORTED})

    labels = [a.judgement.label for a in adjudicate(conn, judge, root).unfaithful]
    assert labels == [LABEL_NOT_SUPPORTED, LABEL_UNCERTAIN]


# --------------------------------------------------------------------------
# the verdict goes through the store, and nothing else does
# --------------------------------------------------------------------------


def test_a_refused_claim_is_rejected_through_the_store(repo):
    """The gate's policy -- one unsupportive verdict rejects, and rejection is a
    state rather than a delete -- belongs to `record_verdict` and is tested there.
    This asserts the judge routes through it instead of reimplementing it."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")

    adjudicate(conn, FakeJudge(), root)

    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_REJECTED
    verdicts = store.verdicts_for(conn, aid)
    assert [v["verdict"] for v in verdicts] == [store.VERDICT_REFUTED]
    assert verdicts[0]["judge"] == "fake-judge/v1"
    # The reasoning survives into the store. A rejection nobody can read the reason
    # for cannot distinguish a strict judge from a loose generator.
    assert "tombstone" in verdicts[0]["rationale"]
    # And the spans stay: nothing is deleted.
    assert conn.execute(
        "SELECT count(*) c FROM evidence_spans WHERE assertion_id=?", (aid,)
    ).fetchone()["c"] == 1


def test_a_supported_claim_stays_servable_and_keeps_its_verdict(repo):
    """A supported claim must be left alone AND leave a record. Without the record
    there is no way to tell a store that was audited from one that never was."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "acquire can `return True`")

    report = adjudicate(conn, FakeJudge(), root)

    assert report.score == 1.0
    assert [a.id for a in store.servable_assertions(conn, root)] == [aid]
    assert [v["verdict"] for v in store.verdicts_for(conn, aid)] == [store.VERDICT_SUPPORTED]


def test_uncertain_is_recorded_as_unsupported_not_as_refuted(repo):
    """'The evidence says otherwise' and 'the judge could not tell' both stop a claim
    being served, and they are not the same fact. Collapsing them would make a
    malfunctioning judge indistinguishable from a lying generator in the data."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "claim U is `return True`")
    judge = FakeJudge(forced={"claim U": LABEL_UNCERTAIN})

    adjudicate(conn, judge, root)

    assert [v["verdict"] for v in store.verdicts_for(conn, aid)] == [
        store.VERDICT_UNSUPPORTED
    ]
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_REJECTED


def test_every_label_maps_to_a_store_verdict_and_only_one_is_supportive(repo):
    """The judge's vocabulary is three words wide and the store's is three words
    wide, and exactly one word in each is the permissive one. A fourth label added
    without a mapping, or mapped to `supported`, is how the gate quietly opens."""
    assert set(LABELS) == {LABEL_SUPPORTED, LABEL_NOT_SUPPORTED, LABEL_UNCERTAIN}
    supportive = [
        label
        for label in LABELS
        if Judgement(label, "", "j").verdict == store.VERDICT_SUPPORTED
    ]
    assert supportive == [LABEL_SUPPORTED]


def test_record_false_scores_without_touching_the_store(repo):
    """Calibrating a judge, or comparing two on the same claims, must not rewrite the
    store as a side effect of being measured."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")

    report = adjudicate(conn, FakeJudge(), root, record=False)

    assert report.score == 0.0
    assert report.recorded is False
    assert "dry run" in report.summary()
    assert store.verdicts_for(conn, aid) == []
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_ACTIVE
    assert report.adjudications[0].verdict_id is None


# --------------------------------------------------------------------------
# the evidence window -- the judge sees the citation, not the codebase
# --------------------------------------------------------------------------


def test_the_judge_sees_only_the_cited_span(repo):
    """The measurement is worthless if the window is generous. `tombstone` is in this
    file, eight lines below the span the claim cited; a judge shown the enclosing
    file would support the claim, and a later reader following the citation would
    find nothing supporting it there."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "acquire writes a `tombstone`")
    judge = FakeJudge()

    report = adjudicate(conn, judge, root)

    _, _, evidence = judge.seen[0]
    assert "return True" in evidence          # inside the cited span
    assert "tombstone" not in evidence        # in the file, outside the span
    assert "def release" not in evidence
    assert report.score == 0.0


def test_the_judge_is_shown_every_cited_span_with_its_citation(repo):
    """A claim resting on two spans must be judged against both, and each block has
    to name the `path:line-line` a human would go and read. Evidence the judge saw
    that a reader cannot locate is the failure this whole tier exists to prevent."""
    root, conn = repo
    _admit(
        conn,
        [_acquire_span(root), _release_span(root)],
        "the pair `return True` and `tombstone` bracket the lease lifecycle",
    )
    judge = FakeJudge()

    report = adjudicate(conn, judge, root)

    _, _, evidence = judge.seen[0]
    assert "leases.py:1-5" in evidence
    assert "leases.py:8-10" in evidence
    assert "span 1 of 2" in evidence and "span 2 of 2" in evidence
    assert report.score == 1.0


def test_the_subject_qualname_reaches_the_judge_labelled_as_not_evidence(repo):
    """The claim's subject is needed -- a span can hold several symbols -- but a name
    that sounds like the claim proves nothing, and a judge that treats it as evidence
    will support claims from their own titles."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "acquire can `return True`")
    judge = FakeJudge()

    adjudicate(conn, judge, root)

    subject, _, _ = judge.seen[0]
    assert subject == "leases.acquire"
    _, user = build_prompt(claim="c", evidence="e", subject="leases.acquire")
    assert "It is not evidence" in user


def test_render_evidence_reports_an_unreadable_span_instead_of_judging_silence(repo):
    """A span that cannot be read must say so inside the evidence. Rendering it as an
    empty string would hand the judge a claim with nothing under it and let it be
    graded as an ordinary refusal, hiding a broken read behind a plausible score."""
    root, conn = repo
    span = _acquire_span(root)
    (root / "leases.py").unlink()
    assertion = store.Assertion(
        id=1, subject_qualname="leases.acquire", subject_symbol_id=None, kind="purpose",
        claim="c", status=store.STATUS_ACTIVE, generator=None, confidence=None,
        created_at="", spans=(span,),
    )
    rendered = render_evidence(root, assertion)
    assert "could not read this span" in rendered
    assert span.citation in rendered


# --------------------------------------------------------------------------
# staleness is a different failure and must not move this score
# --------------------------------------------------------------------------


def test_a_stale_claim_is_not_scored_as_unfaithful(repo):
    """Faithfulness must not move when the code moves. A claim whose cited bytes have
    been edited is stale -- a different failure with a different repair -- and
    counting it here would make the score measure two things at once with no way to
    tell which one changed."""
    root, conn = repo
    fresh = _admit(conn, [_release_span(root)], "release can `return False`")
    edited = _admit(conn, [_acquire_span(root)], "acquire can `return True`")
    (root / "leases.py").write_text(SOURCE.replace("    return True", "    return None"))

    report = adjudicate(conn, FakeJudge(), root)

    assert [a.assertion_id for a in report.adjudications] == [fresh]
    assert report.score == 1.0
    # The edited claim was expired by the store's own verification, not refuted here.
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (edited,)
    ).fetchone()["status"] == store.STATUS_STALE
    assert store.verdicts_for(conn, edited) == []


# --------------------------------------------------------------------------
# fail closed: every path that is not an explicit 'supported' stops the claim
# --------------------------------------------------------------------------


def test_an_unparseable_response_is_never_supported():
    """A judge that malfunctions must not be able to admit claims. A permissive
    parser turns a broken judge into a store full of claims that were nominally
    adjudicated and actually were not, and it looks exactly like a good run."""
    for text in ("", "   ", "I think it is fine, honestly", "{{{", "[1, 2, 3]", "null"):
        judgement = parse_judgement(text, "j")
        assert judgement.label != LABEL_SUPPORTED
        assert judgement.verdict != store.VERDICT_SUPPORTED


def test_not_supported_is_not_read_as_supported():
    """The substring trap. `"not supported"` CONTAINS `"supported"`, so a parser that
    searches for the word instead of matching the whole token reads the most
    dangerous answer in the set as the most permissive one."""
    for spelling in ("not_supported", "not supported", "NOT SUPPORTED", "Not-Supported",
                     "unsupported", "refuted", "no"):
        assert normalise_label(spelling) == LABEL_NOT_SUPPORTED
        assert parse_judgement(f'{{"verdict": "{spelling}"}}', "j").label == (
            LABEL_NOT_SUPPORTED
        )


def test_an_unrecognised_verdict_word_fails_closed_and_says_so():
    """An answer nobody anticipated is not a verdict. It must not become `supported`,
    and the raw word has to survive into the reasoning or the next person cannot tell
    a prompt that needs fixing from a generator that does."""
    judgement = parse_judgement('{"verdict": "probably fine", "reasoning": "eh"}', "j")
    assert judgement.label == LABEL_UNCERTAIN
    assert "probably fine" in judgement.reasoning


def test_a_verdict_wrapped_in_thinking_and_fences_is_still_read():
    """Thinking models emit `<think>` blocks and chat models add fences. Those are the
    same verdict, and failing them all to `uncertain` would reject a store's worth of
    correctly-judged claims over formatting.

    The `<think>` block here DRAFTS a verdict and then rejects it, which is what a
    reasoning model actually does. That makes the block dangerous rather than merely
    noisy: read as part of the answer, its abandoned first guess is the one recovered,
    and the abandoned guess is the permissive one. The block has to be removed before
    anything looks for a verdict, not merely tolerated."""
    raw = (
        '<think>First guess: {"verdict": "supported"}. But re-reading the span, it '
        "never returns at all, so that was wrong.</think>\n"
        '```json\n{"verdict": "not_supported", "reasoning": "the span never returns"}\n```'
    )
    judgement = parse_judgement(raw, "j")
    assert judgement.label == LABEL_NOT_SUPPORTED
    assert judgement.reasoning == "the span never returns"
    assert judgement.raw == raw


def test_a_stated_verdict_survives_malformed_json(repo):
    """REGRESSION, from a real run. `qwen3.5:9b` judging `embed.serialize` quoted the
    code back at us -- `f"{len(values)}f"` -- leaving an unescaped double quote inside
    its own JSON string. The object does not parse, the verdict was lost to
    `uncertain`, and the bias is systematic: it fires on claims about code containing
    quotes, which in a Python repo means anything involving strings."""
    raw = (
        '{"verdict": "supported", "reasoning": "the format string `f"{len(values)}f"` '
        'confirms it packs floats"}'
    )
    judgement = parse_judgement(raw, "j")
    assert judgement.label == LABEL_SUPPORTED
    assert "recovered from malformed JSON" in judgement.reasoning


def test_recovery_cannot_manufacture_a_verdict_nobody_gave():
    """The recovery above is the one place a permissive parser could have been
    smuggled in. It reads the verdict the judge wrote; it does not decide one. A
    response with no verdict field must still fail closed, and a malformed refusal
    must stay a refusal."""
    assert parse_judgement('{"reasoning": "the span says `x = "y"` and nothing else"}',
                           "j").label == LABEL_UNCERTAIN
    assert parse_judgement('{"verdict": "not_supported", "reasoning": "`a "b"` no"}',
                           "j").label == LABEL_NOT_SUPPORTED
    assert parse_judgement('{"verdict": "vibes", "reasoning": "`a "b"`"}',
                           "j").label == LABEL_UNCERTAIN


def test_a_claim_with_no_spans_fails_closed_without_consulting_the_judge(repo):
    """'Every cited span supports the claim' is trivially true of no spans -- the same
    vacuous truth the store's `no_evidence` guard exists for. Unreachable while the
    write gate holds, and checked anyway, because it is the one shape that would be
    reported as verified while resting on nothing."""
    root, conn = repo
    assertion = store.Assertion(
        id=1, subject_qualname="leases.acquire", subject_symbol_id=None, kind="purpose",
        claim="acquire can `return True`", status=store.STATUS_ACTIVE, generator=None,
        confidence=None, created_at="", spans=(),
    )
    judge = FakeJudge()

    result = adjudicate_assertion(conn, judge, assertion, root, record=False)

    assert result.judgement.label == LABEL_NOT_SUPPORTED
    assert judge.seen == []  # never asked -- there was nothing to ask about
    assert result.citations == ()


def test_a_judge_that_cannot_be_reached_raises_instead_of_rejecting_everything():
    """The one failure that must NOT fail closed into a verdict. Recording
    `uncertain` because ollama was not running would reject every assertion in the
    store and log a reason that blames the claims. No answer is not a verdict."""
    with pytest.raises(JudgeUnavailable) as excinfo:
        OllamaJudge(host="http://localhost:11434").judge(
            claim="c", evidence="e", subject="s"
        )
    assert "No verdict was reached" in str(excinfo.value)


# --------------------------------------------------------------------------
# the prompt carries a rule, so the rule is asserted
# --------------------------------------------------------------------------


def test_the_prompt_tells_the_judge_to_refute_and_where_to_land_when_torn():
    """The single most load-bearing string in the module. An agreeable judge produces
    a high score for a store full of unaccountable claims, which is strictly worse
    than not measuring -- so the instruction to refute, and the tie-break onto
    not_supported, are asserted rather than assumed to still be there."""
    system, _ = build_prompt(claim="c", evidence="e", subject="s")
    assert "REFUTE" in system
    assert "burden of proof is on the claim" in system
    assert "If you are torn, answer not_supported" in system
    # And the escape hatch is narrowed: unreadable evidence only, never mere doubt.
    assert 'Use "uncertain" ONLY when the evidence is unreadable' in system


def test_an_index_with_no_repo_root_refuses_instead_of_guessing_at_the_cwd(tmp_path):
    """The store refuses to verify citations without knowing where the repo is, and
    that refusal has to survive being called from here. Defaulting to the cwd would
    let the judge read 'evidence' from whatever happened to sit at those paths
    relative to wherever the process started, and then score it."""
    conn = db.init_db(tmp_path / "unbound.db")
    with pytest.raises(ValueError, match="not bound to a repo root"):
        adjudicate(conn, FakeJudge())


def test_the_report_names_which_judge_produced_it():
    """Two judges on one store are the point of the seam. A score with no judge
    attached cannot be compared with anything, including itself a month later."""
    report = FaithfulnessReport(judge="ollama/qwen3.5:9b")
    assert "ollama/qwen3.5:9b" in report.summary()
    assert OllamaJudge(model="qwen3.5:9b").name == "ollama/qwen3.5:9b"


def test_an_adjudication_renders_its_own_detail_line():
    judgement = Judgement(LABEL_NOT_SUPPORTED, "the span is silent on locking", "j")
    detail = Adjudication(
        assertion_id=7, subject_qualname="a.b", claim="b locks", citations=("f.py:1-2",),
        judgement=judgement,
    ).detail()
    assert "#7" in detail and "a.b" in detail
    assert "f.py:1-2" in detail
    assert "silent on locking" in detail
