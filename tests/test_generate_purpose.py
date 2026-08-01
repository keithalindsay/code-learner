"""The LLM purpose adapter: the seam it satisfies, and the boundary it must not open.

Same standard as `test_gold_from_history.py`, which is the file next door and the one
this adapter has to be worthy of: every test names a rule, and deleting the rule has
to turn the test red. Two rules here are the kind that pass vacuously if you are not
careful, so each gets an explicit counterpart:

* **"the model only ever sees a `SourceView`."** A test that runs the adapter and
  finds no leak passes just as happily when the gate is unwired.
  `test_a_view_carrying_leaked_prose_is_refused_before_the_model_is_called` and
  `test_a_richer_object_at_the_seam_is_refused` exist to show the gate CAN fire, so
  the clean runs beside them mean something.

* **"the cache cannot serve a docstring-informed answer to the doc-blind row."** The
  failure is silent by construction -- a contaminated blind row is not an error, it is
  a number that looks like a finding. It is tested by making the fake's answer reveal
  what it was shown, so contamination becomes visible in the output rather than only
  in the score.

**No test here calls a model or touches the network.** That is the house rule stated
in `eval/faithfulness.py`'s `Judge` docstring and in `generate/types.py`'s
`ClaimGenerator`, and it is why the backend seam is a protocol: every fake below is
deterministic and answers from the prompt it was handed.

The fixture is a plain directory rather than a git repo, unlike the fixture next door.
Nothing here exercises mining -- attribution, the funnel and the reject reasons are
already tested against real history in `test_gold_from_history.py` -- so the labels are
hand-built, which keeps every test in this file about the adapter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from codelearner.eval.gold_from_history import (
    DEFAULT_CONDITIONS,
    LeakDetected,
    MinedLabel,
    SourceView,
    docstring_purpose,
    score_purposes,
    source_view,
    suspect_tokens,
    token_f1,
)
from codelearner.generate.purpose import (
    MAX_PURPOSE_WORDS,
    NORMALISATION_RULE,
    GeneratorUnavailable,
    LLMPurposeGenerator,
    assert_source_only,
    build_prompt,
    llm_condition,
    llm_conditions,
    normalise_purpose,
)
from codelearner.ingest.python_extract import extract_file

# A phrase that exists ONLY in held-out label prose, never in any fixture source file.
# Every boundary test keys off it: if it turns up on the model's side of the seam, the
# boundary is not holding and no score computed afterwards means anything.
SENTINEL = "the parcel lock is taken with a compare and swap insert"

# A word that exists only in a fixture DOCSTRING. The doc-blind tests key off it: if a
# blind-condition prompt or answer contains it, the blinding did not hold.
DOC_ONLY = "custodian"


@pytest.fixture
def repo(tmp_path):
    """Three labellable symbols in two files. No git -- see the module docstring."""
    root = tmp_path / "fixture"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "leases.py").write_text(
        '"""Leases."""\n'
        "\n"
        "\n"
        "def acquire_lease(parcel, ttl):\n"
        '    """Take the lease for a parcel on behalf of one ' + DOC_ONLY + '.\n'
        "\n"
        "    Only one holder at a time, enforced by the store.\n"
        '    """\n'
        "    return True\n"
        "\n"
        "\n"
        "def release_lease(parcel):\n"
        '    """Hand a parcel back to the pool."""\n'
        "    return True\n"
    )
    (root / "pkg" / "events.py").write_text(
        '"""Events."""\n'
        "\n"
        "\n"
        "def tail(conn, since):\n"
        '    """Read rows newer than a cursor."""\n'
        "    return []\n"
    )
    return root


@pytest.fixture
def labels(repo):
    """Held-out prose, in the shape `mine_labels` would have produced it.

    Written the way real commit bodies are: about the change and the reason for it, in
    the author's own vocabulary, sharing a word or two with the code and nothing like a
    copied clause -- so `find_leaks` stays quiet and the gates under test here are the
    ones actually doing the work.
    """
    return [
        _label(
            "pkg.leases.acquire_lease",
            "pkg/leases.py",
            "acquire_lease refuses a second holder while the first is alive, because "
            + SENTINEL
            + " and two agents must never both believe they hold one parcel.",
        ),
        _label(
            "pkg.leases.release_lease",
            "pkg/leases.py",
            "release_lease drops the row immediately so a well behaved agent hands the "
            "parcel back the moment it finishes rather than blocking the next one.",
        ),
        _label(
            "pkg.events.tail",
            "pkg/events.py",
            "`tail(conn, since)` returns events newer than a caller's cursor so the "
            "read-the-world step no longer pages the whole log from sequence zero.",
        ),
    ]


def _label(qualname: str, path: str, prose: str) -> MinedLabel:
    return MinedLabel(
        qualname=qualname,
        kind="function",
        path=path,
        prose=prose,
        commit="0" * 40,
        subject="Fixture commit",
        method="line-log",
        files_touched=1,
        units=1,
    )


def _view(repo, qualname: str, path: str) -> SourceView:
    extract = extract_file(repo / path, repo)
    sym = next(s for s in extract.symbols if s.qualname == qualname)
    return source_view(repo, path, sym)


def _acquire(repo) -> SourceView:
    return _view(repo, "pkg.leases.acquire_lease", "pkg/leases.py")


# --------------------------------------------------------------------------------
# Deterministic fakes. No test in this repo may call a model.
# --------------------------------------------------------------------------------


class FakeModel:
    """A `PurposeModel` that answers from the prompt it was handed, and records it.

    Recording the prompts is what makes the doc-blind tests direct rather than
    inferential: "the blind condition received a stripped view" is checked by reading
    what the backend was actually sent, not by watching a score go down.
    """

    def __init__(self, reply, name: str = "fake/deterministic") -> None:
        self._reply = reply
        self._name = name
        self.prompts: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, *, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self._reply(user) if callable(self._reply) else self._reply

    @property
    def users(self) -> list[str]:
        return [user for _, user in self.prompts]


class OutageModel:
    """A backend that is down. It answers nothing, ever -- it does not answer badly."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "fake/unreachable"

    def complete(self, *, system: str, user: str) -> str:
        self.attempts += 1
        raise GeneratorUnavailable("could not reach the fake backend")


_DOCSTRING_BLOCK = re.compile(r"^DOCSTRING:\n(.*?)\nSOURCE:", re.DOTALL | re.MULTILINE)

NO_DOCSTRING_ANSWER = "no documentation was shown to me"


def _echo_docstring(user: str) -> str:
    """The docstring copier, as a model: it relays what it was shown and infers nothing.

    Legitimate by the rules of this eval -- `docstring_purpose` is a shipped baseline
    and the stated upper reference -- and it is the fake that makes the doc-blind
    condition observable, because with the docstring gone it has nothing to relay.
    """
    match = _DOCSTRING_BLOCK.search(user)
    return match.group(1).strip() if match else NO_DOCSTRING_ANSWER


# --------------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------------


def test_the_adapter_is_a_generator_and_scores_through_the_existing_harness(repo, labels):
    """The point of the module: a model drops into `score_purposes` unchanged.

    No new scoring code, no new leak checks in the harness, no branch in
    `score_purposes` for "this one is a model". If this test needs anything in
    `gold_from_history` to change, the adapter has failed at its only job.
    """
    model = FakeModel("Refuses a second holder while the first lease is alive.")
    adapter = LLMPurposeGenerator(model, repo)

    card = score_purposes(repo, labels, adapter, "LLM fake")

    assert card.n == len(labels)
    assert card.empty_output == 0
    assert len(model.prompts) == len(labels)
    assert card.gold > 0.0


def test_the_backend_receives_two_strings_and_no_view(repo, labels):
    """The transport layer never holds a pointer back to the file.

    `PurposeModel.complete` takes a system prompt and a user prompt. It is not handed
    the `SourceView`, and the prompt deliberately omits `view.path`, so a backend
    cannot re-read the file whose docstring the blind condition just removed. This is
    the reason the seam is not `ClaimGenerator`, whose `Offer.span` is a path and a
    byte range.
    """
    model = FakeModel("Takes a lease.")
    LLMPurposeGenerator(model, repo)(_acquire(repo))

    system, user = model.prompts[0]
    assert isinstance(system, str) and isinstance(user, str)
    assert "pkg/leases.py" not in user
    assert "pkg/leases.py" not in system
    assert "acquire_lease" in user  # the symbol itself is fair game; the file is not


# --------------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------------


def test_a_view_carrying_leaked_prose_is_refused_before_the_model_is_called(repo, labels):
    """The gate fires, and it fires BEFORE a prompt exists.

    Proof the detector can fail, without which every clean run in this file is
    vacuous. And the model-call assertion is the half that matters operationally: a
    gate that raises after the backend has already been asked has still shown the
    answer to the model.
    """
    honest = _acquire(repo)
    prose = labels[0].prose
    model = FakeModel("anything at all")
    adapter = LLMPurposeGenerator(model, repo)

    forged_doc = SourceView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature,
        docstring=prose,           # the answer, handed to the generator
        source=honest.source,
    )
    with pytest.raises(LeakDetected):
        adapter(forged_doc)

    forged_source = SourceView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature, docstring=honest.docstring,
        source="# " + prose + "\n",
    )
    with pytest.raises(LeakDetected, match="not a substring"):
        adapter(forged_source)

    assert model.prompts == [], "the model must not be asked before the gate passes"
    # And the honest view goes through, so the exceptions above are the gate firing
    # rather than the adapter being broken.
    assert adapter(honest)
    assert SENTINEL not in model.users[0]


def test_a_richer_object_at_the_seam_is_refused(repo, labels):
    """The type check, and the failure it is really for.

    `SourceView` has no field for a commit message -- that absence is the structural
    argument of the whole eval. A subclass has room, and so does any object with the
    same eight attributes. The failure being defended against is not today's harness;
    it is a caller who later decides the generator would do better with "a bit more
    context" and passes a richer object through the same seam, where every other check
    would pass and the score would simply improve.
    """
    honest = _acquire(repo)

    @dataclass(frozen=True)
    class RicherView(SourceView):
        commit_message: str = ""

    richer = RicherView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature, docstring=honest.docstring, source=honest.source,
        commit_message=labels[0].prose,
    )
    model = FakeModel("anything at all")
    with pytest.raises(LeakDetected, match="not a SourceView"):
        LLMPurposeGenerator(model, repo)(richer)
    assert model.prompts == []


def test_the_gate_accepts_an_honest_doc_blind_view_and_still_refuses_a_forged_one(repo, labels):
    """The blinded view breaks the strict gate by design, so the fallback must be real.

    `without_docstring` rewrites the source specifically so the docstring is no longer
    in it, which means `assert_view_is_source_only` cannot hold on a blind view. The
    line-wise fallback is what replaces it -- and a fallback that accepted everything
    would be worse than no fallback, so the second half plants prose in a blinded view
    and requires it to be caught.
    """
    blinded = _acquire(repo).without_docstring()
    assert_source_only(repo, blinded)  # does not raise

    forged = SourceView(
        qualname=blinded.qualname, kind=blinded.kind, path=blinded.path,
        line_start=blinded.line_start, line_end=blinded.line_end,
        signature=blinded.signature, docstring=None,
        source=blinded.source + "\n    # " + labels[0].prose + "\n",
    )
    with pytest.raises(LeakDetected, match="not in pkg/leases.py"):
        assert_source_only(repo, forged)


def test_the_doc_blind_condition_really_receives_a_stripped_view(repo, labels):
    """Read what the backend was sent, not what the score did.

    A doc-blind row that is not blind is the single most expensive bug available here:
    it reports an inference score for a copy. Checking the prompts directly is the only
    way to know, since a contaminated row does not look wrong from the outside.
    """
    model = FakeModel(_echo_docstring)
    adapter = LLMPurposeGenerator(model, repo)

    score_purposes(repo, labels, adapter, "LLM fake, doc-blind", docstring_blind=True)

    assert len(model.users) == len(labels)
    for user in model.users:
        assert "DOCSTRING:" not in user
        assert DOC_ONLY not in user
        assert SENTINEL not in user


def test_the_cache_cannot_serve_a_docstring_informed_answer_to_the_blind_row(repo, labels):
    """THE hazard. A cache keyed on the symbol would fabricate the blind result.

    The two conditions differ only in the view, so a cache keyed on `qualname` -- the
    obvious key, and the one a reasonable person writes -- would hand the blind row the
    answer generated with the docstring in hand. Nothing would raise. The blind row
    would simply come out level with the sighted one, and "the model infers purpose
    without documentation" would be reported as a finding on the strength of a
    dictionary key.

    The fake's answer reveals what it was shown, so contamination is visible in the
    output rather than only in the number.
    """
    model = FakeModel(_echo_docstring)
    adapter = LLMPurposeGenerator(model, repo)

    sighted = score_purposes(repo, labels, adapter, "LLM fake")
    blind = score_purposes(repo, labels, adapter, "LLM fake, doc-blind", docstring_blind=True)

    # Every symbol was generated twice: once sighted, once blind. A cross-condition
    # cache hit would show up here as a smaller number.
    assert adapter.calls == 2 * len(labels)
    sighted_answers = [_echo_docstring(u) for u in model.users[: len(labels)]]
    blind_answers = [_echo_docstring(u) for u in model.users[len(labels) :]]
    assert all(a != NO_DOCSTRING_ANSWER for a in sighted_answers)
    assert blind_answers == [NO_DOCSTRING_ANSWER] * len(labels)
    assert sighted.gold > blind.gold == 0.0


def test_the_cache_works_and_the_blinded_view_is_a_different_key(repo):
    """Both halves of the caching claim, on one view.

    The saving is real (the second identical call does not reach the backend) and the
    key is the whole frozen view (the blinded form is a different entry). Either half
    alone is a half-truth: a cache that never hits is pointless, and a cache that hits
    across conditions is a fabricated result.
    """
    view = _acquire(repo)
    adapter = LLMPurposeGenerator(FakeModel(_echo_docstring), repo)

    sighted = adapter(view)
    blinded = adapter(view.without_docstring())
    assert adapter.calls == 2
    assert sighted != blinded
    assert DOC_ONLY in sighted and DOC_ONLY not in blinded

    assert adapter(view) == sighted
    assert adapter(view.without_docstring()) == blinded
    assert adapter.calls == 2, "identical views must not be re-generated"
    assert adapter.cached == 2


def test_caching_can_be_turned_off(repo):
    view = _acquire(repo)
    adapter = LLMPurposeGenerator(FakeModel("Takes a lease."), repo, cache=False)
    adapter(view)
    adapter(view)
    assert adapter.calls == 2
    assert adapter.cached == 0


def test_a_model_that_relays_the_docstring_behaves_as_the_leak_rules_say(repo, labels):
    """A docstring copier is legal, undetectable, and exactly why the blind row exists.

    The rules in `gold_from_history` are specific about this. Relaying the docstring is
    not a leak -- `docstring_purpose` is a shipped baseline and the stated upper
    reference -- and `suspect_tokens` correctly stays silent, because every word came
    from the input. The thing that costs a copier its score is the doc-blind condition,
    not the leak detector.

    The second half is the vacuity guard: a fake that returns the HELD-OUT prose does
    light `suspect_tokens` up, so the silence above is the detector working rather than
    the detector being broken.
    """
    view = _acquire(repo)
    label = labels[0]
    adapter = LLMPurposeGenerator(FakeModel(_echo_docstring), repo)

    relayed = adapter(view)
    assert DOC_ONLY in relayed
    assert SENTINEL not in relayed
    assert suspect_tokens(relayed, label.prose, view) == []
    # It relays the same first sentence the shipped docstring baseline does, which is
    # what makes the two rows comparable.
    assert token_f1(relayed, docstring_purpose(view)) == 1.0

    cheating = LLMPurposeGenerator(FakeModel(label.prose), repo)(view)
    assert suspect_tokens(cheating, label.prose, view), (
        "a generator echoing the held-out label must show suspect tokens"
    )


# --------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "This function appears to close the WAL connection before the process exits.",
            "close the WAL connection before the process exits.",
        ),
        (
            "Sure! Here's a one-sentence summary: Closes the WAL connection before exit.",
            "Closes the WAL connection before exit.",
        ),
        (
            "The purpose of this function is to acquire the lease for a parcel.",
            "acquire the lease for a parcel.",
        ),
        (
            "It is responsible for reaping expired leases so a crashed agent cannot hold one.",
            "reaping expired leases so a crashed agent cannot hold one.",
        ),
        ("Purpose:\nRefuses a second holder.", "Refuses a second holder."),
        ("- Refuses a second holder.", "Refuses a second holder."),
        ("**Refuses a second holder.**", "Refuses a second holder."),
        ("```\nRefuses a second holder.\n```", "Refuses a second holder."),
        (
            "<think>let me read the body</think>Refuses a second holder.",
            "Refuses a second holder.",
        ),
        (
            "Refuses a second holder. It does this with a compare-and-swap insert.",
            "Refuses a second holder.",
        ),
        (
            "Refuses a second\nholder while the first is alive.",
            "Refuses a second holder while the first is alive.",
        ),
        (
            "Based on the code, this method drops the row immediately.",
            "drops the row immediately.",
        ),
    ],
)
def test_normalisation_reduces_output_to_one_purpose_statement(raw, expected):
    """The documented rule, case by case. See `NORMALISATION_RULE`.

    Every case here is a shape a chat-tuned model actually emits instead of answering.
    Left in, they cost token-F1 precision for a reason unrelated to understanding --
    and the three baselines this row is compared against are terse by construction.
    """
    assert normalise_purpose(raw) == expected


def test_normalisation_raises_the_score_and_the_docstring_says_which_way(repo, labels):
    """The bias is real, it is upward, and it is written down rather than implied.

    Stated as a test because the honest thing to do with a thumb on the scale is to
    measure it. Preamble is non-label vocabulary, precision is overlap / |output
    tokens|, so removing it can only raise F1 -- which is why the same reduction is
    applied before the shuffled control and why the harness asks you to read `lift`.
    """
    label = labels[0].prose
    verbose = "Sure! Here's a summary. This function appears to refuse a second holder."
    assert token_f1(normalise_purpose(verbose), label) > token_f1(verbose, label)

    for phrase in ("precision", "UP", "lift", "40 words"):
        assert phrase in NORMALISATION_RULE


def test_normalisation_never_manufactures_an_empty_answer():
    """Stripping must not turn a verbose model into a silent one.

    An output that is nothing but scaffolding would otherwise reduce to `""`, which
    `token_f1` scores 0.0 and `PurposeScorecard` counts as `empty_output` -- reporting
    the model as having said nothing when it said the wrong thing. Those are different
    findings with different repairs.
    """
    assert normalise_purpose("Sure! Here's a summary:").strip()
    assert normalise_purpose("This function.").strip()
    # A model that genuinely said nothing still reads as nothing.
    assert normalise_purpose("   \n\n  ") == ""


def test_normalisation_caps_a_run_on_answer():
    """The backstop: one sentence with no boundary in it cannot score by volume alone."""
    long_answer = " ".join(["parcel"] * 200)
    assert len(normalise_purpose(long_answer).split()) == MAX_PURPOSE_WORDS


# --------------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------------


def test_a_backend_outage_raises_instead_of_scoring_an_empty_answer(repo, labels):
    """An outage must stop the run, not become a low score.

    The exception travels all the way out of `score_purposes`, so a run made while the
    backend was down produces no scorecard at all. The second half shows what the
    alternative would have looked like: a fake returning `""` scores a perfectly
    presentable 0.000 with a full `n`, which reads as "the model is bad" rather than
    "there was no run" -- and nothing in the report distinguishes them.
    """
    model = OutageModel()
    adapter = LLMPurposeGenerator(model, repo)
    with pytest.raises(GeneratorUnavailable):
        score_purposes(repo, labels, adapter, "LLM fake")
    assert adapter.cached == 0, "a failure must never be cached as an answer"

    # Retried rather than served from cache, because nothing was stored.
    with pytest.raises(GeneratorUnavailable):
        adapter(_acquire(repo))
    assert model.attempts >= 2

    silent = score_purposes(repo, labels, LLMPurposeGenerator(FakeModel(""), repo), "silent")
    assert silent.n == len(labels)
    assert silent.gold == 0.0
    assert silent.empty_output == len(labels)


# --------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------


def test_llm_conditions_have_the_shape_default_conditions_has(repo, labels):
    """The factory output must be passable to `run_purpose_eval(conditions=...)` as is.

    Shape checked against `DEFAULT_CONDITIONS` itself rather than against a literal, so
    a change to the harness's tuple breaks this test instead of silently making the
    factory produce something the harness no longer accepts.
    """
    conditions = llm_conditions(FakeModel("Takes a lease."), repo, name="LLM fake")

    assert len(conditions) == 2
    for condition in conditions:
        assert len(condition) == len(DEFAULT_CONDITIONS[0])
        name, generator, blind = condition
        assert isinstance(name, str) and callable(generator) and isinstance(blind, bool)

    assert [(name, blind) for name, _, blind in conditions] == [
        ("LLM fake", False),
        ("LLM fake, doc-blind", True),
    ]
    # Suffixed the way the baselines suffix theirs, so a report reads consistently.
    assert "body identifiers, doc-blind" in {name for name, _, _ in DEFAULT_CONDITIONS}
    # And unambiguous beside them.
    assert not {name for name, _, _ in conditions} & {
        name for name, _, _ in DEFAULT_CONDITIONS
    }


def test_the_paired_conditions_share_one_adapter_and_therefore_one_cache(repo):
    """One cache across the pair is the arrangement that puts the hazard under test.

    Giving each row its own adapter would make cross-contamination impossible by
    accident rather than by design, and the day someone consolidated them the property
    would be untested. They share, and
    `test_the_cache_cannot_serve_a_docstring_informed_answer_to_the_blind_row` is what
    holds the sharing safe.
    """
    conditions = llm_conditions(FakeModel("Takes a lease."), repo)
    assert conditions[0][1] is conditions[1][1]


def test_a_single_condition_can_be_named_and_paired_by_hand(repo):
    model = FakeModel("Takes a lease.", name="ollama/qwen3.5:9b")
    name, generator, blind = llm_condition(model, repo)
    assert name == "LLM ollama/qwen3.5:9b"
    assert blind is False
    assert isinstance(generator, LLMPurposeGenerator)

    name, _, blind = llm_condition(model, repo, name="LLM qwen", docstring_blind=True)
    assert (name, blind) == ("LLM qwen, doc-blind", True)


def test_the_conditions_run_through_the_loop_run_purpose_eval_uses(repo, labels):
    """End to end at the seam: the factory's tuples, scored by the harness untouched.

    This is the loop `run_purpose_eval` performs over its `conditions` argument. It is
    reproduced rather than called because `run_purpose_eval` mines git history first,
    which is tested against real history next door -- what is under test here is that
    the tuples this module produces are consumed by the scoring code without
    modification.
    """
    conditions = llm_conditions(FakeModel(_echo_docstring), repo, name="LLM fake")
    cards = [
        score_purposes(repo, labels, generator, name, docstring_blind=blind)
        for name, generator, blind in conditions
    ]

    assert [card.name for card in cards] == ["LLM fake", "LLM fake, doc-blind"]
    assert all(card.n == len(labels) for card in cards)
    assert all(card.suspect == 0 for card in cards)
    assert cards[0].gold > cards[1].gold  # the copier loses its crib in the blind row


def test_build_prompt_differs_between_conditions_only_by_the_docstring_block(repo):
    """The instructions are byte-identical across conditions; only the evidence changes.

    A prompt that reworded itself for the blind condition would make the two rows
    differ for a reason other than the blinding, and nothing in the scorecard could
    separate the two causes.
    """
    view = _acquire(repo)
    sighted_system, sighted_user = build_prompt(view)
    blind_system, blind_user = build_prompt(view.without_docstring())

    assert sighted_system == blind_system
    assert "DOCSTRING:" in sighted_user and "DOCSTRING:" not in blind_user
    assert DOC_ONLY in sighted_user and DOC_ONLY not in blind_user
    # Nothing announces that a docstring was removed.
    assert "removed" not in blind_user.lower()
