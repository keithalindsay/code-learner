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

import importlib
import json
import re
import urllib.error
import urllib.request

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.eval.faithfulness import (
    CAUSE_FORMAT_FAILURE,
    CAUSE_JUDGED,
    CAUSE_NO_EVIDENCE,
    CAUSE_PARSE_FAILURE,
    LABEL_NOT_SUPPORTED,
    LABEL_SUPPORTED,
    LABEL_UNCERTAIN,
    LABELS,
    PERTURBATIONS,
    REVIEW_FIELDS,
    Adjudication,
    FaithfulnessReport,
    Judgement,
    JudgeMisbehaving,
    JudgeUnavailable,
    OllamaJudge,
    adjudicate,
    adjudicate_assertion,
    build_prompt,
    cluster_correction,
    export_for_review,
    faithfulness,
    format_interval,
    measure_prompt_stability,
    normalise_label,
    parse_judgement,
    render_evidence,
    score_review,
    wilson_interval,
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


class RawFakeJudge:
    """Returns raw judge TEXT through the real parser, so causes are not hand-set.

    The three `uncertain` routes are decisions `parse_judgement` makes, and a fake that
    constructs `Judgement(cause=...)` directly would assert the test's opinion about
    which route a response takes rather than the parser's. This one hands over the
    bytes a model would emit and lets the module classify them.

    `responses` maps a substring of the claim to the raw text; anything unmatched gets
    `default`.
    """

    def __init__(self, responses=None, default='{"verdict": "supported", "reasoning": "ok"}',
                 name="raw-fake/v1"):
        self._responses = responses or {}
        self._default = default
        self.seen: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return "raw-fake/v1"

    def judge(self, *, claim: str, evidence: str, subject: str) -> Judgement:
        self.seen.append((subject, claim, evidence))
        for needle, text in self._responses.items():
            if needle in claim:
                return parse_judgement(text, self.name)
        return parse_judgement(self._default, self.name)


class WhitespaceSensitiveJudge:
    """Supports every claim, unless a line of the evidence ends in whitespace.

    A judge whose verdict depends on a change that carries no information -- the exact
    failure `measure_prompt_stability` exists to put a rate on, made deterministic. It
    also flips on `separator_style` if `also_separators` is set, so a test can assert
    the harness attributes flips to the perturbation that caused them rather than
    pooling them.
    """

    def __init__(self, *, also_separators: bool = False) -> None:
        self.also_separators = also_separators
        self.calls = 0

    @property
    def name(self) -> str:
        return "whitespace-sensitive/v1"

    def judge(self, *, claim: str, evidence: str, subject: str) -> Judgement:
        self.calls += 1
        flipped = any(line != line.rstrip() for line in evidence.split("\n"))
        if self.also_separators and "=== span" in evidence:
            flipped = True
        label = LABEL_NOT_SUPPORTED if flipped else LABEL_SUPPORTED
        return Judgement(label, "whitespace" if flipped else "clean", self.name)


def _adj(assertion_id, label, *, cause=CAUSE_JUDGED, subject=None, claim="a `c`",
         citations=("leases.py:1-5",), evidence="e"):
    """One `Adjudication` with no store behind it, for scoring arithmetic."""
    return Adjudication(
        assertion_id=assertion_id,
        subject_qualname=subject if subject is not None else f"m.s{assertion_id}",
        claim=claim,
        citations=citations,
        judgement=Judgement(label, f"r{assertion_id}", "j", cause=cause),
        evidence=evidence,
    )


def _report_of(*, supported=0, not_supported=0, judge_uncertain=0, parse_failures=0,
               format_failures=0, subject=None):
    """A `FaithfulnessReport` with a stated composition and no judge involved."""
    report = FaithfulnessReport(judge="j")
    spec = (
        [(LABEL_SUPPORTED, CAUSE_JUDGED)] * supported
        + [(LABEL_NOT_SUPPORTED, CAUSE_JUDGED)] * not_supported
        + [(LABEL_UNCERTAIN, CAUSE_JUDGED)] * judge_uncertain
        + [(LABEL_UNCERTAIN, CAUSE_PARSE_FAILURE)] * parse_failures
        + [(LABEL_UNCERTAIN, CAUSE_FORMAT_FAILURE)] * format_failures
    )
    report.adjudications = [
        _adj(i, label, cause=cause, subject=subject)
        for i, (label, cause) in enumerate(spec, start=1)
    ]
    return report


def _acquire_span(root):
    return store.span_for(root, "leases.py", 0, ACQUIRE_END)


def _release_span(root):
    return store.span_for(root, "leases.py", RELEASE_START, len(SOURCE))


def _admit(conn, spans, claim, *, qualname="leases.acquire"):
    """Admit one claim, deliberately without an index behind it.

    `allow_unindexed_subject=True` because the fixture is a repo and a bound DB with
    no symbols in it, and nothing in this file is about the write gate -- the subject
    of these claims exists on disk, which is what the judge is shown. The flag is the
    store's explicit escape rather than a weakened rule: it is here so that a reader
    can see the subject check was skipped on purpose.
    """
    return store.write_assertion(
        conn,
        subject_qualname=qualname,
        kind="purpose",
        claim=claim,
        spans=spans,
        generator="claude/test",
        confidence=0.9,
        allow_unindexed_subject=True,
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


# --------------------------------------------------------------------------
# one denominator held three different events
# --------------------------------------------------------------------------


def test_the_three_uncertains_are_counted_apart():
    """`uncertain` arrives by three routes and only one of them is about the claim.

    The judge saying so is a verdict. `_extract_json` and `_salvage_fields` both
    failing is a harness fact. A verdict token that does not normalise is a
    judge-format fact. Before `cause` existed all three were one value, so a run in
    which ollama emitted garbage was indistinguishable in the data from a run in which
    the judge read every span and declined to decide on two of them -- and both
    reported a low faithfulness that a reader charges to the generator."""
    genuine = parse_judgement('{"verdict": "uncertain", "reasoning": "span truncated"}', "j")
    unparseable = parse_judgement("I think it is fine, honestly", "j")
    misformatted = parse_judgement('{"verdict": "probably fine", "reasoning": "eh"}', "j")

    # All three land on the same label -- that part is deliberate and unchanged.
    assert [j.label for j in (genuine, unparseable, misformatted)] == [LABEL_UNCERTAIN] * 3
    # And they are now three different facts underneath it.
    assert genuine.cause == CAUSE_JUDGED
    assert unparseable.cause == CAUSE_PARSE_FAILURE
    assert misformatted.cause == CAUSE_FORMAT_FAILURE
    assert genuine.instrument_failure is False
    assert unparseable.instrument_failure is True
    assert misformatted.instrument_failure is True


def test_the_report_counts_the_three_causes_so_a_bad_run_reads_as_a_bad_run():
    """Two runs with the same score and opposite meanings. The counters are the only
    thing that separates 'the judge was unsure about three claims' from 'the transport
    ate three responses', and the second is not a fact about the generator."""
    judged = _report_of(supported=5, not_supported=2, judge_uncertain=3)
    broken = _report_of(supported=5, not_supported=2, parse_failures=2, format_failures=1)

    assert judged.score == broken.score == pytest.approx(0.5)
    assert (judged.judge_uncertain, judged.instrument_failures) == (3, 0)
    assert (broken.judge_uncertain, broken.instrument_failures) == (0, 3)
    assert (broken.parse_failures, broken.format_failures) == (2, 1)
    assert broken.count(LABEL_UNCERTAIN) == judged.count(LABEL_UNCERTAIN) == 3

    summary = broken.summary()
    assert "parse_failures=2" in summary and "format_failures=1" in summary
    assert any("harness failing to read a response" in w for w in broken.warnings())
    assert judged.warnings() and all(
        "harness" not in w for w in judged.warnings()
    )


def test_a_claim_that_cited_nothing_is_the_generators_fault_not_the_instruments():
    """The one non-judged label that must NOT be excused as an instrument failure. A
    claim that reached the judge with no spans cited nothing, which is the generator's
    doing, so it counts in both denominators."""
    root = None
    assertion = store.Assertion(
        id=1, subject_qualname="leases.acquire", subject_symbol_id=None, kind="purpose",
        claim="c", status=store.STATUS_ACTIVE, generator=None, confidence=None,
        created_at="", spans=(),
    )
    conn = db.init_db(":memory:")
    result = adjudicate_assertion(conn, FakeJudge(), assertion, root, record=False)

    assert result.judgement.cause == CAUSE_NO_EVIDENCE
    assert result.judgement.instrument_failure is False
    assert result.judgement.label == LABEL_NOT_SUPPORTED


def test_both_scores_are_reported_and_the_headline_keeps_the_full_denominator():
    """`supported/n` and `supported/(supported+not_supported)` answer different
    questions and the report owes both. The headline stays the full denominator: it is
    a lower bound needing no assumption about WHY a claim went undecided, where the
    conditional score assumes the undecided set is a random subset -- known false here,
    since the measured parse failures fire on claims about code containing quotes."""
    report = _report_of(supported=8, not_supported=2, judge_uncertain=2, parse_failures=3)

    assert report.score == pytest.approx(8 / 15)
    assert report.score_decided == pytest.approx(8 / 10)
    assert report.decided == 10
    assert faithfulness(report.adjudications) == report.score
    assert faithfulness(report.adjudications, decided_only=True) == report.score_decided
    # And the conditional score is the more flattering one, which is the reason it is
    # not the headline.
    assert report.score_decided > report.score


def test_a_broken_harness_moves_the_two_scores_apart_instead_of_hiding_in_one(repo):
    """The bug in prose: ollama returns malformed output and faithfulness reads as
    'the claims are bad'. With both numbers reported, the same run says so -- `score`
    falls, `score_decided` does not move at all, and the gap between them is exactly
    the instrument's contribution."""
    root, conn = repo
    for i in range(4):
        _admit(conn, [_acquire_span(root)], f"claim {i} can `return True`")
    clean = adjudicate(conn, RawFakeJudge(), root, record=False)
    broken = adjudicate(
        conn,
        RawFakeJudge(responses={"claim 0": "no json here at all",
                                "claim 1": "not even close"}),
        root,
        record=False,
        max_instrument_failure_rate=None,
    )

    assert clean.score == clean.score_decided == 1.0
    assert broken.score == pytest.approx(0.5)      # charged to the generator
    assert broken.score_decided == 1.0             # not charged to the generator
    assert broken.instrument_failures == 2
    assert broken.judge_uncertain == 0


def test_score_decided_is_none_when_nothing_was_decided():
    """A run in which every response was unreadable decided nothing, and 'faithfulness
    among the decided claims' has no value -- not 0.0 and not 1.0. The same vacuous
    truth the empty-set guard exists for, one level down."""
    report = _report_of(parse_failures=4)
    assert report.score == 0.0
    assert report.score_decided is None
    assert report.interval_decided is None
    assert faithfulness(report.adjudications, decided_only=True) is None


# --------------------------------------------------------------------------
# the interval -- three decimals were a resolution nobody measured
# --------------------------------------------------------------------------


def test_the_published_number_carries_its_interval_and_loses_a_decimal():
    """0.544 on n=147 is `0.54 [0.46, 0.62]`. The third decimal implied a resolution
    two orders of magnitude finer than the interval, and every reader who did not stop
    to do this arithmetic read it as precision."""
    low, high = wilson_interval(80, 147)
    assert (round(low, 3), round(high, 3)) == (0.464, 0.623)
    assert format_interval(80 / 147, (low, high)) == "0.54 [0.46, 0.62]"

    report = _report_of(supported=80, not_supported=65, judge_uncertain=2)
    assert report.score == pytest.approx(0.544, abs=5e-4)
    assert format_interval(report.score, report.interval) == "0.54 [0.46, 0.62]"
    assert "0.54 [0.46, 0.62]" in report.summary()
    assert "0.544" not in report.summary()


def test_the_judges_own_calibration_interval_is_the_one_that_matters():
    """15/16 reads as 94% and is consistent with a judge that is right 72% of the time.
    That is the whole argument for reporting intervals, applied to the number this
    module's credibility rests on."""
    low, high = wilson_interval(15, 16)
    assert format_interval(15 / 16, (low, high)) == "0.94 [0.72, 0.99]"


def test_wilson_stays_inside_the_unit_interval_and_is_never_zero_width():
    """The reason it is Wilson and not `p +/- z*sqrt(p(1-p)/n)`. The normal
    approximation runs past 1, and at p=1 it collapses to `[1.0, 1.0]` -- a gate that
    refused every attack it was shown would report certainty it never measured."""
    low, high = wilson_interval(8, 8)
    assert 0.0 < low < 1.0 and high == 1.0
    low, high = wilson_interval(0, 8)
    assert low == 0.0 and 0.0 < high < 1.0
    for successes, total in ((1, 3), (50, 100), (147, 147)):
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= successes / total <= high <= 1.0
    assert wilson_interval(0, 0) is None
    assert _report_of().interval is None


def test_clustering_widens_the_interval_wilson_would_have_reported():
    """147 claims about one repo are not 147 independent draws: several per symbol,
    many per file, some about `sample_repo/` toys. Hand-checkable fixture -- four
    subjects, two claims each, perfect agreement inside every subject. That is an
    intra-cluster correlation of 1 and a design effect equal to the cluster size, so
    eight rows are worth four observations and the interval is the one for n=4."""
    outcomes = [("a", True), ("a", True), ("b", True), ("b", True),
                ("c", False), ("c", False), ("d", False), ("d", False)]
    correction = cluster_correction(outcomes)

    assert (correction.n, correction.clusters) == (8, 4)
    assert correction.icc == pytest.approx(1.0)
    assert correction.design_effect == pytest.approx(2.0)
    assert correction.effective_n == pytest.approx(4.0)

    report = FaithfulnessReport(judge="j")
    report.adjudications = [
        _adj(i, LABEL_SUPPORTED if outcome else LABEL_NOT_SUPPORTED, subject=cluster)
        for i, (cluster, outcome) in enumerate(outcomes, start=1)
    ]
    plain = report.interval
    clustered = report.clustered_interval()
    assert clustered[0] < plain[0] and clustered[1] > plain[1]
    assert clustered == wilson_interval(2.0, 4.0)
    assert "clustered by subject" in report.summary()


def test_uncorrelated_clusters_leave_the_interval_alone():
    """The correction has to be able to find nothing. A design effect that is always
    above 1 is not a measurement of clustering, it is a fudge factor."""
    outcomes = [(c, i % 2 == 0) for c in "abcd" for i in range(2)]
    correction = cluster_correction(outcomes)
    assert correction.icc == pytest.approx(0.0)
    assert correction.design_effect == pytest.approx(1.0)


def test_the_correction_declines_rather_than_guessing_at_one_point_zero():
    """A design effect of 1.0 asserts 'measured, and there is no clustering'. None says
    'this sample cannot answer that'. One claim per symbol is the second, and reporting
    it as the first would put a fabricated independence claim under the interval."""
    assert cluster_correction([]) is None
    assert cluster_correction([("a", True), ("b", False)]) is None   # every cluster size 1
    assert cluster_correction([("a", True), ("a", False)]) is None   # one cluster
    # And a report of singleton subjects prints no clustered line rather than a false one.
    assert _report_of(supported=3, not_supported=2).clustered_interval() is None
    assert "clustered by" not in _report_of(supported=3, not_supported=2).summary()


# --------------------------------------------------------------------------
# a run that stops being readable is not a low score
# --------------------------------------------------------------------------


def test_a_run_whose_responses_stop_parsing_aborts_instead_of_rejecting_the_store(repo):
    """The judgement call, argued in `JudgeMisbehaving`. An individual parse failure is
    survivable and honestly recorded. A run of them is not a low faithfulness score --
    it is `record_verdict` demoting live claims to rejected for a formatting reason,
    with a rationale that blames the claim. Counters make that visible afterwards; only
    stopping makes it not happen."""
    root, conn = repo
    ids = [_admit(conn, [_acquire_span(root)], f"claim {i} can `return True`")
           for i in range(12)]

    with pytest.raises(JudgeMisbehaving) as excinfo:
        adjudicate(conn, RawFakeJudge(default="no json anywhere"), root)

    assert "could not be read" in str(excinfo.value)
    assert "not about the claims" in str(excinfo.value)
    # The partial report is attached, not lost, and it stopped at the sample floor.
    assert len(excinfo.value.report) == 10
    assert excinfo.value.report.parse_failures == 10
    # Verdicts already written stay written -- `adjudicate` records as it goes so an
    # interrupted run keeps what it reached -- and the rest of the store is untouched.
    assert store.verdicts_for(conn, ids[0]) != []
    assert store.verdicts_for(conn, ids[11]) == []
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (ids[11],)
    ).fetchone()["status"] == store.STATUS_ACTIVE


def test_a_handful_of_parse_failures_is_noise_and_does_not_stop_the_run(repo):
    """The other half of the same judgement. 1-in-16 was measured on a real run and the
    number it produced was still a measurement; aborting there would throw away eleven
    completed adjudications over one unescaped quote."""
    root, conn = repo
    for i in range(12):
        _admit(conn, [_acquire_span(root)], f"claim {i} can `return True`")

    report = adjudicate(conn, RawFakeJudge(responses={"claim 3": "no json"}), root)

    assert len(report) == 12
    assert report.parse_failures == 1
    assert report.score == pytest.approx(11 / 12)


def test_the_floor_keeps_one_bad_first_response_from_killing_a_whole_run(repo):
    """Without a sample floor the guard fires at 1/1 = 100%, so a single quote
    character in the first claim's code is fatal to every claim behind it."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "claim 0 can `return True`")

    report = adjudicate(conn, RawFakeJudge(default="no json"), root)

    assert report.parse_failures == 1
    assert report.score == 0.0


def test_the_abort_can_be_switched_off_to_measure_a_judges_output_shape(repo):
    """Deliberately characterising a judge that emits malformed JSON is a legitimate
    experiment, and it must not be the default -- `None` is spelled by the caller."""
    root, conn = repo
    for i in range(12):
        _admit(conn, [_acquire_span(root)], f"claim {i} can `return True`")

    report = adjudicate(conn, RawFakeJudge(default="no json"), root, record=False,
                        max_instrument_failure_rate=None)

    assert len(report) == 12
    assert report.parse_failures == 12


def test_a_misbehaving_judge_is_not_an_unreachable_one():
    """A caller that retries on a transport fault must not retry a judge that is
    answering in a shape the parser cannot read: running it again changes nothing."""
    assert not issubclass(JudgeMisbehaving, JudgeUnavailable)
    assert not issubclass(JudgeUnavailable, JudgeMisbehaving)


# --------------------------------------------------------------------------
# the instrument: prompt stability, the largest unmeasured uncertainty
# --------------------------------------------------------------------------


def test_the_perturbation_harness_puts_a_rate_on_a_judge_that_flips_on_whitespace(repo):
    """The measurement the README admitted was missing. Trailing whitespace changes no
    information a verdict could depend on, so a label that moves under it was never
    determined by the evidence. This judge flips on exactly that and on nothing else,
    and the harness has to attribute it to the named perturbation rather than reporting
    'the judge is unstable'."""
    root, conn = repo
    for i in range(4):
        _admit(conn, [_acquire_span(root)], f"claim {i} can `return True`")

    report = measure_prompt_stability(conn, WhitespaceSensitiveJudge(), root)

    assert report.rate_for("trailing_whitespace").flips == 4
    assert report.rate_for("trailing_whitespace").rate == 1.0
    assert report.rate_for("identity").flips == 0
    assert report.rate_for("span_order").flips == 0
    assert report.rate_for("separator_style").flips == 0
    assert {f.perturbation for f in report.flips} == {"trailing_whitespace"}
    assert report.flips[0].baseline_label == LABEL_SUPPORTED
    assert report.flips[0].perturbed_label == LABEL_NOT_SUPPORTED
    # Pooled across the non-baseline perturbations, with an interval, because a flip
    # rate is also an estimate.
    assert report.pooled.n == 16 and report.pooled.flips == 4
    assert "trailing_whitespace" in report.format_report()
    assert report.rate_for("trailing_whitespace").interval[0] > 0.0


def test_the_identity_control_separates_prompt_sensitivity_from_decoding_noise(repo):
    """Without a no-op run in the set, a flip rate conflates 'the prompt decided it'
    with 'the sampler did', and those have opposite repairs. Whatever the control flips
    is the floor the rest are read against."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "claim 0 can `return True`")

    report = measure_prompt_stability(conn, WhitespaceSensitiveJudge(also_separators=True), root)

    assert report.baseline == "identity"
    assert report.rate_for("identity").flips == 0
    assert {f.perturbation for f in report.flips} == {
        "trailing_whitespace", "separator_style"
    }
    assert "control" in report.summary()


def test_perturbations_are_individually_selectable(repo):
    """Attribution is the point. 'The judge is unstable' cannot be acted on; 'it is
    stable under reordering and flips on trailing whitespace' can."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "claim 0 can `return True`")
    judge = WhitespaceSensitiveJudge()

    report = measure_prompt_stability(
        conn, judge, root, perturbations=("identity", "span_order")
    )

    assert [r.perturbation for r in report.rates] == ["identity", "span_order"]
    assert report.flips == []
    with pytest.raises(ValueError, match="unknown perturbation"):
        measure_prompt_stability(conn, judge, root, perturbations=("bold_it",))


def test_every_perturbation_preserves_what_a_verdict_could_legitimately_depend_on(repo):
    """A perturbation that changed the evidence would measure the judge being right,
    not the judge being unstable. Each one has to leave the spans, their citations and
    their non-whitespace content exactly where they were."""
    root, conn = repo
    _admit(
        conn,
        [_acquire_span(root), _release_span(root)],
        "the pair `return True` and `tombstone` bracket the lease lifecycle",
    )
    assertion = store.servable_assertions(conn, root)[0]
    rendered = render_evidence(root, assertion)

    for name, perturb in PERTURBATIONS.items():
        perturbed = perturb(rendered)
        squashed = re.sub(r"[^a-zA-Z0-9.:_-]+", "", perturbed)
        for citation in ("leases.py:1-5", "leases.py:8-10"):
            assert citation in perturbed, name
        assert "returnTrue" in squashed, name
        assert "tombstone" in squashed, name
        assert perturbed.count("span 1 of 2") == 1, name
        assert perturbed.count("span 2 of 2") == 1, name
    assert PERTURBATIONS["identity"](rendered) == rendered
    # Reordering reorders and does nothing else: the same characters, differently
    # arranged. A splitter that stripped greedily would smuggle a whitespace change in
    # here and make the two flip rates inseparable.
    assert sorted(PERTURBATIONS["span_order"](rendered)) == sorted(rendered)


def test_the_stability_harness_cannot_record_a_verdict(repo):
    """It deliberately malforms the prompt. Letting an experiment about the judge write
    to the store would let it reject live claims on evidence it garbled itself."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)], "claim 0 can `return True`")

    measure_prompt_stability(conn, WhitespaceSensitiveJudge(), root)

    assert store.verdicts_for(conn, aid) == []
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_ACTIVE


# --------------------------------------------------------------------------
# the instrument: human calibration, made cheap enough to happen
# --------------------------------------------------------------------------


def test_the_review_export_is_blind_to_the_verdict_it_is_checking(tmp_path):
    """The design decision in the function. A reviewer who can see the label being
    checked is anchored by it, and the resulting agreement measures deference rather
    than accuracy -- which is worse than the current state, because it would look like
    the missing calibration had been done."""
    adjudications = [_adj(i, LABEL_SUPPORTED if i % 2 else LABEL_NOT_SUPPORTED,
                          claim=f"claim {i}", evidence=f"span {i}") for i in range(1, 9)]

    export = export_for_review(adjudications, tmp_path / "review.jsonl", n=4)

    rows = [json.loads(line) for line in export.review_path.read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert set(row) == set(REVIEW_FIELDS)
        assert row["human"] == ""       # empty, never pre-filled: a default is a vote
        assert row["evidence"].startswith("span ")
    assert "not_supported" not in export.review_path.read_text()
    # The labels went to the key file, which the reviewer does not open.
    key_rows = [json.loads(line) for line in export.key_path.read_text().splitlines()]
    assert {r["label"] for r in key_rows} == {LABEL_SUPPORTED, LABEL_NOT_SUPPORTED}


def test_an_adjudication_keeps_exactly_the_evidence_the_judge_was_shown(repo):
    """The export and the flip harness are both meaningless unless what they show a
    human, or perturb, is what the judge actually read. `render_evidence` reads the
    working tree, so re-deriving it an hour later can produce a different string for
    the same citation -- and a reviewer told 'this is what the judge saw' would be
    calibrating against a span that had moved."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)], "acquire can `return True`")
    judge = FakeJudge()

    report = adjudicate(conn, judge, root, record=False)

    _, _, shown = judge.seen[0]
    assert report.adjudications[0].evidence == shown
    assert "return True" in report.adjudications[0].evidence
    assert "tombstone" not in report.adjudications[0].evidence

    export = export_for_review(report.adjudications, tmp_path_of(root) / "r.jsonl", n=1)
    row = json.loads(export.review_path.read_text().splitlines()[0])
    assert row["evidence"] == shown


def tmp_path_of(root):
    """The fixture repo's parent, for a test that needs somewhere to write."""
    return root.parent


def test_the_review_sample_is_balanced_and_reproducible(tmp_path):
    """Precision and recall each rest on one half of the sample, and an unstratified 30
    of a 0.54 store gives whichever split luck hands out. Reproducible because a
    calibration sample that cannot be regenerated cannot be extended or re-reviewed."""
    adjudications = (
        [_adj(i, LABEL_SUPPORTED) for i in range(1, 21)]
        + [_adj(i, LABEL_NOT_SUPPORTED) for i in range(21, 27)]
        + [_adj(i, LABEL_UNCERTAIN) for i in range(27, 31)]
    )

    export = export_for_review(adjudications, tmp_path / "a.jsonl", n=8, seed=7)
    again = export_for_review(adjudications, tmp_path / "b.jsonl", n=8, seed=7)

    assert export.per_label == {LABEL_SUPPORTED: 4, LABEL_NOT_SUPPORTED: 4}
    assert export.n == 8
    assert export.review_path.read_text() == again.review_path.read_text()
    # `uncertain` is out by default -- the target is the judge's error on the claims it
    # decided -- and reachable when that is the question being asked.
    with_uncertain = export_for_review(
        adjudications, tmp_path / "c.jsonl", n=9,
        labels=(LABEL_SUPPORTED, LABEL_NOT_SUPPORTED, LABEL_UNCERTAIN),
    )
    assert with_uncertain.per_label[LABEL_UNCERTAIN] == 3


def test_a_short_stratum_fills_from_the_others_and_reports_what_it_got(tmp_path):
    """A store with four refusals cannot supply fifteen. The sample says what it
    actually contains instead of quietly returning half the requested size."""
    adjudications = (
        [_adj(i, LABEL_SUPPORTED) for i in range(1, 11)]
        + [_adj(i, LABEL_NOT_SUPPORTED) for i in range(11, 13)]
    )
    export = export_for_review(adjudications, tmp_path / "r.jsonl", n=8)
    assert export.n == 8
    assert export.per_label == {LABEL_SUPPORTED: 6, LABEL_NOT_SUPPORTED: 2}


def test_a_filled_review_scores_judge_against_human(tmp_path):
    """The round trip, with arithmetic anyone can check by reading it. Four rows: the
    judge said supported twice and not_supported twice; the human agreed on three.
    `supported` is the positive class because the errors are not symmetric -- a false
    `supported` admits an unaccountable claim, which is what this tier exists to
    prevent."""
    adjudications = [_adj(1, LABEL_SUPPORTED), _adj(2, LABEL_SUPPORTED),
                     _adj(3, LABEL_NOT_SUPPORTED), _adj(4, LABEL_NOT_SUPPORTED)]
    export = export_for_review(adjudications, tmp_path / "review.jsonl", n=4)

    human = {1: "supported", 2: "not_supported", 3: "not_supported", 4: "no"}
    rows = [json.loads(line) for line in export.review_path.read_text().splitlines()]
    for row in rows:
        row["human"] = human[row["assertion_id"]]
    export.review_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = score_review(export.review_path, export.key_path)

    assert report.n == 4 and report.agreements == 3
    assert report.agreement == 0.75
    assert report.precision == 0.5      # 1 of the 2 the judge called supported
    assert report.recall == 1.0         # 1 of the 1 the human called supported
    assert report.kappa == pytest.approx(0.5)
    assert report.agreement_interval[0] < 0.75 < report.agreement_interval[1]
    assert "precision(supported) 0.50" in report.summary()
    # The key can come from whatever the caller still has.
    assert score_review(export.review_path, adjudications).confusion == report.confusion
    assert score_review(export.review_path, {1: LABEL_SUPPORTED, 2: LABEL_SUPPORTED,
                                             3: LABEL_NOT_SUPPORTED,
                                             4: LABEL_NOT_SUPPORTED}).n == 4


def test_kappa_is_reported_because_a_balanced_sample_inflates_raw_agreement(tmp_path):
    """The sample is deliberately half supported, so two labellers guessing agree half
    the time. A judge that agrees 50% of the time on it has learned nothing, and only
    kappa says so."""
    adjudications = [_adj(1, LABEL_SUPPORTED), _adj(2, LABEL_SUPPORTED),
                     _adj(3, LABEL_NOT_SUPPORTED), _adj(4, LABEL_NOT_SUPPORTED)]
    export = export_for_review(adjudications, tmp_path / "review.jsonl", n=4)
    human = {1: "supported", 2: "not_supported", 3: "supported", 4: "not_supported"}
    rows = [json.loads(line) for line in export.review_path.read_text().splitlines()]
    for row in rows:
        row["human"] = human[row["assertion_id"]]
    export.review_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = score_review(export.review_path, export.key_path)

    assert report.agreement == 0.5
    assert report.kappa == pytest.approx(0.0)


def test_an_unreviewed_row_is_counted_not_dropped(tmp_path):
    """A reviewer who labelled 18 of 30 measured 18. An agreement rate over a
    denominator of 18 while implying 30 is the same over-reading the interval exists to
    stop."""
    adjudications = [_adj(1, LABEL_SUPPORTED), _adj(2, LABEL_NOT_SUPPORTED)]
    export = export_for_review(adjudications, tmp_path / "review.jsonl", n=2)
    rows = [json.loads(line) for line in export.review_path.read_text().splitlines()]
    rows[0]["human"] = "supported"
    export.review_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = score_review(export.review_path, export.key_path)

    assert (report.n, report.agreements, report.skipped) == (1, 1, 1)
    assert "+1 unreviewed" in report.summary()


def test_a_human_label_nobody_recognises_raises_instead_of_vanishing(tmp_path):
    """Silently discarding a typo biases the sample in whatever direction the typo was,
    and leaves no trace that it happened."""
    adjudications = [_adj(1, LABEL_SUPPORTED)]
    export = export_for_review(adjudications, tmp_path / "review.jsonl", n=1)
    rows = [json.loads(line) for line in export.review_path.read_text().splitlines()]
    rows[0]["human"] = "sort of?"
    export.review_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    with pytest.raises(ValueError, match="unrecognised human label"):
        score_review(export.review_path, export.key_path)


def test_a_review_row_with_no_judge_label_behind_it_raises(tmp_path):
    """Scoring a row against a key that does not contain it would silently drop the
    row, or worse, compare it against the wrong claim."""
    review = tmp_path / "review.jsonl"
    review.write_text(json.dumps({"assertion_id": 99, "human": "supported"}) + "\n")
    with pytest.raises(ValueError, match="not in the key"):
        score_review(review, {1: LABEL_SUPPORTED})


# --------------------------------------------------------------------------
# what the module claims about its own judge
# --------------------------------------------------------------------------


def test_the_module_does_not_claim_cross_family_independence_it_cannot_show():
    """`model_family` is a string-prefix test on a published tag: a Qwen-distilled
    model shipped as `deepseek-r1` passes it, and this repo's own Qwen3-based reranker
    would too. The docstring has to say what the pairing is -- different weights and a
    different tokenizer, a proxy -- and name the measurement that would establish the
    thing it used to assert.

    `import_module` rather than `from codelearner.eval import faithfulness`: the
    package re-exports the FUNCTION under that name, so the attribute lookup returns
    it and this test would read a function's docstring while claiming to read the
    module's."""
    module = importlib.import_module("codelearner.eval.faithfulness")

    doc = " ".join(module.__doc__.split())  # unwrapped, so line breaks cannot hide a phrase
    assert "proxy for independence, not a demonstration of it" in doc
    assert "string-prefix test on an ollama tag" in doc
    # The old wording survives only inside the sentence that withdraws it.
    assert '"a different model family, which is the point" -- claimed more' in doc
    assert "second-judge agreement" in doc
    assert "qwen3:14b" in doc
