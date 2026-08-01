"""The claim generator: the citation channel, the refusals, and the outage boundary.

No test here calls a model, and that is enforced rather than trusted -- `_no_network`
patches `urlopen` for every test in the file, so a code path that reaches for ollama
fails the suite instead of quietly making it slow and machine-dependent. Same rule and
same fixture as `tests/test_faithfulness.py`; the `Judge` protocol's docstring states
it for the scoring half, and it is not weaker here just because this half is the one
writing the claims.

What is asserted is mostly the shape of the seam rather than the quality of the output:
that the model can only cite by number, that everything it can get wrong lands on a
draft the pipeline refuses, and that a backend which never answered is distinguishable
from one that answered badly. Claim quality is not testable here by construction -- it
is what `eval.faithfulness` measures, with a model of a different family, at runtime.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import pytest

from codelearner.assertions.store import EvidenceSpan
from codelearner.eval.faithfulness import DEFAULT_JUDGE_MODEL
from codelearner.generate.llm import (
    DEFAULT_GENERATOR_MODEL,
    JUDGE_FAMILY,
    OllamaClaimGenerator,
    build_generation_prompt,
    collides_with_judge,
    model_family,
    parse_draft,
    render_menu,
)
from codelearner.generate.types import Draft, GeneratorUnavailable, Offer

ACQUIRE = 'def acquire(parcel_id):\n    if parcel_id is None:\n        return False\n'
RELEASE = 'def release(parcel_id):\n    return False\n'


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach a model. Enforced, not assumed.

    A generator is even easier to make accidentally network-dependent than a judge:
    one forgotten fake and the suite passes on the machine with ollama running,
    hangs everywhere else, and asserts against whatever a 9B model felt like saying.
    """

    def _refuse(*args, **kwargs):
        raise urllib.error.URLError("tests must not reach a model")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


def _span(path: str = "leases.py", line_start: int = 1, line_end: int = 3) -> EvidenceSpan:
    return EvidenceSpan(
        path=path,
        line_start=line_start,
        line_end=line_end,
        byte_start=0,
        byte_end=len(ACQUIRE),
        content_hash="0" * 64,
    )


def _offers() -> tuple[Offer, ...]:
    return (
        Offer(ref=1, span=_span(), text=ACQUIRE, label="leases.acquire"),
        Offer(
            ref=2,
            span=_span("leases.py", 5, 6),
            text=RELEASE,
            label="leases.release",
        ),
    )


class FakeOllama:
    """A scripted `_post`, standing in for the backend.

    Records every payload it was sent, which is how the request-shape assertions are
    made: the fake transport is the only place a test can see exactly what a real
    backend would have received.
    """

    def __init__(self, content: str = "", *, thinking: str = "") -> None:
        self.content = content
        self.thinking = thinking
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, payload: dict) -> dict:
        self.calls.append((path, payload))
        return {"message": {"content": self.content, "thinking": self.thinking}}


def _generator(fake: FakeOllama, monkeypatch, **kwargs) -> OllamaClaimGenerator:
    generator = OllamaClaimGenerator(**kwargs)
    monkeypatch.setattr(generator, "_post", fake.post)
    return generator


# --------------------------------------------------------------------------
# the menu: the only way a model is allowed to see, or name, evidence
# --------------------------------------------------------------------------


def test_the_menu_is_numbered_from_the_offers_own_refs():
    """1-based, in the caller's numbering, and stable across calls.

    Numbering by position instead of by `Offer.ref` would produce drafts that cite the
    wrong spans while looking perfectly well-formed -- `Draft.resolve` maps answers
    back through the caller's refs, so a menu that renumbers is a silent mis-citation
    rather than an error. The offers here are numbered 5 and 6 precisely so that a
    positional implementation prints 1 and 2 and fails."""
    offers = (
        Offer(ref=5, span=_span(), text=ACQUIRE, label="leases.acquire"),
        Offer(ref=6, span=_span(), text=RELEASE, label="leases.release"),
    )
    menu = render_menu(offers)

    assert "[5] leases.acquire" in menu
    assert "[6] leases.release" in menu
    assert "[1]" not in menu
    assert menu.index("[5]") < menu.index("[6]")
    assert render_menu(offers) == menu

    ordinary = render_menu(_offers())
    assert ordinary.startswith("[1] leases.acquire")
    assert "[2] leases.release" in ordinary


def test_the_menu_shows_the_bytes_but_never_a_path_or_an_offset():
    """The model is not shown a citation, so a path in its answer cannot be an echo.

    This is stricter than the parser needs -- references are parsed as integers and a
    path would be discarded anyway -- and it is the difference between a fabricated
    location and a near-miss on something real. It costs the model the ability to know
    two spans share a file, which is accepted deliberately."""
    menu = render_menu(_offers())

    assert "return False" in menu           # the bytes are all the way through
    assert "leases.py" not in menu          # the path is not
    assert ":1-3" not in menu               # nor the line range
    assert "0" * 64 not in menu             # nor the hash


def test_an_unlabelled_offer_still_gets_a_number():
    """A missing label must not collapse two menu entries into an ambiguous one."""
    menu = render_menu((Offer(ref=1, span=_span(), text=ACQUIRE, label="  "),))
    assert menu.startswith("[1] (unlabelled)")


# --------------------------------------------------------------------------
# the prompt carries the rules, so the rules are asserted
# --------------------------------------------------------------------------


def test_the_prompt_pushes_against_confident_overclaiming():
    """The load-bearing string. A generator that always produces a confident sentence
    is the failure the tier-2 gate exists to catch, and these four instructions are
    the only thing in this module standing against it -- so their presence is asserted
    rather than assumed to still be there after the next edit."""
    system, _ = build_generation_prompt(subject="leases.acquire", offered=_offers())

    # (a) only what the spans establish
    assert "Say only what the shown spans establish" in system
    # (b) cite what you used, and nothing else
    assert "Cite every span you relied on" in system
    assert "cite no others" in system
    # (c) narrow over broad
    assert "Prefer a narrow claim you can support to a broad one you cannot" in system
    # (d) abstention is allowed, and is preferred to a guess
    assert "answer with an empty claim and an empty list of references" in system
    assert "It is a correct and useful answer" in system
    # and the citation channel is closed in prose as well as in the parser
    assert "Never write a file path, a line number or a byte offset" in system


def test_the_prompt_asks_for_purpose_and_rejects_signature_restatement():
    """REGRESSION, from a real run. Told only to be narrow and supportable,
    `llama3.1:8b` answered "the function `_free_port` returns an integer" and "the
    `main` function requires a `--base-url` argument" -- perfect compliance, and
    worthless. Restating a signature is the global optimum of narrow-and-supportable:
    the span entails it, so the faithfulness judge supports every one and a store full
    of them scores ~1.0 while saying nothing about any symbol in the repo.

    Three things fix it and all three are asserted, because a later edit tightening the
    prompt for length would take the examples out first and nothing else in the suite
    would notice: the question is stated, the bad shape is named WITH an instance of
    it, and narrowness is re-scoped so it can no longer be satisfied by descending to
    syntax."""
    system, _ = build_generation_prompt(subject="leases.acquire", offered=_offers())

    assert "what job does this code do for the rest of the program" in system
    assert "Not what it is called, not what it takes, not what it returns" in system
    assert "NOT AN ANSWER" in system
    assert "BAD:" in system and "GOOD:" in system
    assert "returns an integer" in system  # the measured failure, shown as the bad form
    assert "Restating a signature is not a narrow claim" in system
    assert "narrow means claiming less of the job, not retreating to syntax" in system
    # The abstention path stays honest rather than being loosened into speculation:
    # the way out of a symbol with no visible purpose is still silence.
    assert "If the only thing you can write is the signature, write nothing" in system


def test_the_prompt_points_the_model_at_the_caller_spans():
    """A symbol's job is usually invisible from inside it, so without this the
    purpose question and the evidence-bound rule genuinely trade off -- and the model
    resolves that conflict by retreating to the signature, which is exactly what it
    did. The role words are duplicated into the prompt as prose rather than imported,
    because the generator must not depend on the pipeline that drives it; this is the
    pin that stops the two vocabularies drifting apart in silence."""
    from codelearner.generate.pipeline import ROLE_CALLEE, ROLE_CALLER, ROLE_SUBJECT

    system, user = build_generation_prompt(subject="leases.acquire", offered=_offers())

    assert f"`{ROLE_SUBJECT}` are the code being described" in system
    assert f"`{ROLE_CALLER}` are code that calls it" in system
    assert f"`{ROLE_CALLEE}` are code it calls" in system
    assert "what a caller passes in, what a caller does with the result" in system
    # Citing the caller is stated as a duty, not left implied: a purpose read off a
    # caller and then cited only to the subject is an unsupported claim.
    assert "If a caller span is what showed you the purpose, cite that caller span" in system
    assert "including any caller span that is what showed you the purpose" in user


def test_the_prompt_labels_the_subject_as_not_evidence():
    """A qualname that sounds like a purpose establishes nothing, and a model handed
    one unlabelled will happily restate it as a finding about the code."""
    _, user = build_generation_prompt(subject="leases.acquire", offered=_offers())
    assert "leases.acquire" in user
    assert "It is not evidence" in user


def test_the_prompt_is_pure_and_deterministic():
    """Two runs whose claims differ must be able to establish whether the instruction
    was the same. A prompt that varies with anything but its arguments makes that
    question unanswerable, so it is a function of (subject, offers) and nothing else --
    including the offers themselves, which it must not mutate."""
    offers = _offers()
    first = build_generation_prompt(subject="s", offered=offers)
    second = build_generation_prompt(subject="s", offered=offers)

    assert first == second
    assert offers == _offers()  # unchanged by having been rendered
    assert build_generation_prompt(subject="other", offered=offers) != first


# --------------------------------------------------------------------------
# parsing: the model can be wrong, it cannot be creative about citations
# --------------------------------------------------------------------------


def test_a_well_formed_response_parses():
    raw = '{"claim": "`acquire` returns False when `parcel_id` is None", "cited_refs": [1]}'
    draft = parse_draft(raw)

    assert draft == Draft(
        claim="`acquire` returns False when `parcel_id` is None",
        cited_refs=(1,),
        kind="purpose",
    )
    spans, invalid = draft.resolve(_offers())
    assert [s.path for s in spans] == ["leases.py"]
    assert invalid == []


def test_a_response_wrapped_in_thinking_and_fences_is_still_read():
    """Reasoning models emit `<think>` blocks and chat models add fences, and failing
    both to an empty draft would refuse a store's worth of good claims over
    formatting. The block here drafts a broad claim and then abandons it, which is what
    a reasoning model actually does -- and the abandoned draft is the confident one, so
    it must be removed before anything looks for an object rather than merely
    tolerated."""
    raw = (
        '<think>First attempt: {"claim": "manages the whole lease lifecycle", '
        '"cited_refs": [1, 2]}. But span 2 only shows a return, so that is too '
        "broad.</think>\n"
        '```json\n{"claim": "`release` returns False", "cited_refs": [2]}\n```'
    )
    draft = parse_draft(raw)

    assert draft.claim == "`release` returns False"
    assert draft.cited_refs == (2,)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "I had a look and it seems to acquire a lease.",
        "{{{",
        "[1, 2, 3]",
        "null",
        '{"claim": "the format string `f"{n}f"` packs floats", "cited_refs": [1]}',
    ],
)
def test_an_unreadable_response_is_an_empty_draft_and_not_an_exception(raw):
    """A model that answered badly has answered: the run records a refusal for this
    subject and carries on. It must not raise (that is reserved for an outage) and it
    must not produce a partial claim.

    The last case is the one that matters and it is not hypothetical -- an unescaped
    double quote inside a sentence quoting code is a measured failure of a local model
    on this machine (see `_salvage_fields` in `eval/faithfulness.py`). The judge
    reconstructs its verdict from that wreckage because a verdict is one token from a
    closed set. A claim is prose, and half a sentence is not half a claim, it is a
    different claim nobody made -- so this one is lost on purpose."""
    draft = parse_draft(raw)
    assert draft.claim == ""
    assert draft.cited_refs == ()


def test_references_that_are_not_integers_are_discarded():
    """Everything that is not an integer in this field is a path, an object, or a
    hallucinated location. `true` is checked explicitly because `True` is an `int` in
    Python and would otherwise become a citation of offer 1 -- a real span, cited by a
    model that answered `[true]`."""
    raw = json.dumps(
        {
            "claim": "`acquire` guards on a null id",
            "cited_refs": [
                "leases.py:1-3",
                "codelearner/db.py[4120:4380]",
                {"path": "leases.py", "byte_start": 0},
                None,
                True,
                1.5,
                "",
                1,
            ],
        }
    )
    draft = parse_draft(raw)
    assert draft.cited_refs == (1,)


def test_an_integer_written_as_a_string_is_still_a_reference():
    """Schema-constrained decoding is a feature of this runtime, not a property of a
    model, and `"2"` is unambiguous. Nothing else stringy is: `"1, 2"` would need a
    splitter, and a splitter is what eventually reads `"db.py:1-2"` as two
    references."""
    assert parse_draft('{"claim": "c", "cited_refs": [" 2 "]}').cited_refs == (2,)
    assert parse_draft('{"claim": "c", "cited_refs": ["1, 2"]}').cited_refs == ()


def test_off_menu_references_survive_parsing_so_resolve_can_count_them():
    """The generator does not repair its own mistakes. `Draft.resolve` discards
    references that were never offered AND reports them, and how often a model cites
    off the menu is the main number this seam produces -- filtering them here would
    hand the pipeline a clean-looking draft and no idea the model missed."""
    draft = parse_draft('{"claim": "c", "cited_refs": [1, 0, -1, 99, 4120]}')
    assert draft.cited_refs == (1, 0, -1, 99, 4120)

    spans, invalid = draft.resolve(_offers())
    assert len(spans) == 1
    assert invalid == [0, -1, 99, 4120]


def test_duplicate_references_collapse_to_one():
    """`resolve` collapses them too, so nothing downstream changes -- but a
    `cited_refs` that repeats itself makes a one-span claim look like a three-span
    claim to anything counting references before resolving them, including a human
    reading a report."""
    draft = parse_draft('{"claim": "c", "cited_refs": [2, 1, 2, "2", 1]}')
    assert draft.cited_refs == (2, 1)


def test_a_smuggled_path_or_byte_offset_is_ignored_rather_than_honoured():
    """The failure the whole package is shaped around. A model that answers with a
    path and a byte range must get nothing for it: the location keys are never read,
    the `citations` key is not one of the aliases (it is the key a model reaches for
    when it is about to answer with paths), and a location in the reference list is not
    an integer. What comes back is a claim with no citations, which the pipeline
    refuses."""
    raw = json.dumps(
        {
            "claim": "`acquire` guards on a null id",
            "citations": [{"path": "codelearner/db.py", "byte_start": 4120, "byte_end": 4380}],
            "path": "codelearner/db.py",
            "byte_start": 4120,
            "line_start": 12,
        }
    )
    draft = parse_draft(raw)

    assert draft.cited_refs == ()
    assert draft.resolve(_offers()) == ([], [])
    # Nothing on a Draft can carry a location: the dataclass has no field for one.
    assert set(vars(draft)) == {"claim", "cited_refs", "kind", "confidence"}


def test_a_claimless_answer_normalises_to_a_full_abstention():
    """'These spans establish nothing' is a correct answer and must be representable.
    An empty claim carrying references is not a shape anything downstream has a use
    for, and leaving them attached invites a caller that checks `cited_refs` first
    into treating it as a claim with evidence."""
    assert parse_draft('{"claim": "", "cited_refs": [1, 2]}') == Draft("", (), "purpose")
    assert parse_draft('{"claim": "   ", "cited_refs": [1]}').cited_refs == ()
    assert parse_draft('{"cited_refs": [1]}').claim == ""


def test_the_kind_travels_with_the_draft():
    assert parse_draft('{"claim": "c", "cited_refs": [1]}', kind="invariant").kind == "invariant"


# --------------------------------------------------------------------------
# the backend seam: an outage is not a result
# --------------------------------------------------------------------------


def test_a_backend_that_cannot_be_reached_raises_instead_of_refusing_everything():
    """The one failure that must NOT fail closed into a draft. Returning an empty
    draft because ollama was not running would record 'these spans establish nothing'
    against every symbol in the repo, and the run report would read like a careful
    generator rather than a stopped one."""
    generator = OllamaClaimGenerator(host="http://localhost:11434")

    with pytest.raises(GeneratorUnavailable) as excinfo:
        generator.draft(subject="leases.acquire", offered=_offers())

    assert "No claim was drafted" in str(excinfo.value)


@pytest.mark.parametrize("body", [b"<html>not json</html>", b'"a string"', b"[1, 2]"])
def test_a_body_that_is_not_a_json_object_is_an_outage_not_a_refusal(monkeypatch, body):
    """A proxy's error page and a bare JSON scalar are both 'the backend did not
    answer', and neither is a model declining to make a claim."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())
    generator = OllamaClaimGenerator()

    with pytest.raises(GeneratorUnavailable):
        generator.draft(subject="s", offered=_offers())


def test_a_model_that_answered_unreadably_returns_an_empty_draft(monkeypatch):
    """The other side of the same boundary: the backend answered, so the run
    continues. Refusing this subject is a result and is recorded as one."""
    fake = FakeOllama(content="honestly it looks like a lease manager to me")
    generator = _generator(fake, monkeypatch)

    draft = generator.draft(subject="leases.acquire", offered=_offers())

    assert draft == Draft("", (), "purpose")
    assert len(fake.calls) == 1


def test_no_offers_means_no_model_call_at_all(monkeypatch):
    """Every claim from an empty menu would be uncited, so the only honest answer is
    the empty one -- and asking anyway spends a GPU-second to invite a guess."""
    fake = FakeOllama(content='{"claim": "it manages leases", "cited_refs": [1]}')
    generator = _generator(fake, monkeypatch)

    draft = generator.draft(subject="leases.acquire", offered=())

    assert draft == Draft("", (), "purpose")
    assert fake.calls == []


def test_the_request_constrains_decoding_and_removes_free_variance(monkeypatch):
    """Temperature 0 because two runs over an unchanged repo that disagree cannot be
    compared, and comparison is what `assertions.generator` exists for. The schema
    because `cited_refs` should arrive as integers rather than as prose to be
    salvaged. Thinking off because a reasoning model spends the whole budget in the
    think block and returns empty content, which here is a refused claim for a
    formatting reason. `keep_alive` because the card is shared."""
    fake = FakeOllama(content='{"claim": "c", "cited_refs": [1]}')
    generator = _generator(fake, monkeypatch)

    generator.draft(subject="leases.acquire", offered=_offers())
    path, payload = fake.calls[0]

    assert path == "/api/chat"
    assert payload["options"]["temperature"] == 0.0
    assert payload["think"] is False
    assert payload["keep_alive"] == "5m"
    schema = payload["format"]
    assert schema["properties"]["cited_refs"] == {"type": "array", "items": {"type": "integer"}}
    assert sorted(schema["required"]) == ["cited_refs", "claim"]
    # And the prompt that went out is the one `build_generation_prompt` produces, not
    # an f-string that happens to look like it.
    system, user = build_generation_prompt(subject="leases.acquire", offered=_offers())
    assert [m["content"] for m in payload["messages"]] == [system, user]


def test_a_thinking_models_reasoning_is_parsed_only_when_content_is_empty(monkeypatch):
    """When the budget went to the think block, that block is the only record of what
    the model was doing instead of answering. It is read rather than dropped -- and if
    it holds no object, the result is still an empty draft, never a salvaged one."""
    fake = FakeOllama(content="", thinking='{"claim": "`release` returns False", "cited_refs": [2]}')
    generator = _generator(fake, monkeypatch)
    assert generator.draft(subject="s", offered=_offers()).cited_refs == (2,)

    quiet = FakeOllama(content="", thinking="hmm, let me think about what this does")
    assert _generator(quiet, monkeypatch).draft(subject="s", offered=_offers()).claim == ""


def test_name_is_the_runtime_and_the_model(monkeypatch):
    """`assertions.generator` is filled from this, and it is the only thing that lets a
    store holding claims from two generators tell them apart -- without it, re-running
    with a better model destroys the evidence that the worse one was worse. Prefixed
    because the same weights behind another runtime are not guaranteed to be the same
    generator."""
    assert OllamaClaimGenerator(model="minicpm-v:8b").name == "ollama/minicpm-v:8b"
    assert OllamaClaimGenerator().name == f"ollama/{DEFAULT_GENERATOR_MODEL}"


def test_release_asks_for_an_unload_and_swallows_a_failure(monkeypatch):
    """The card holds 10GB and the judge that follows a generation run needs 6.6GB of
    it. Releasing is mandatory; failing to release is not a reason to fail a run that
    already produced its claims."""
    fake = FakeOllama()
    generator = _generator(fake, monkeypatch)
    generator.release()

    _, payload = fake.calls[0]
    assert payload["keep_alive"] == 0
    assert payload["messages"] == []

    # And an unreachable backend at unload time is logged, not raised: `_no_network`
    # is still in force for this instance.
    OllamaClaimGenerator().release()


# --------------------------------------------------------------------------
# the family collision -- the hazard the faithfulness number depends on
# --------------------------------------------------------------------------


def test_the_default_generator_is_not_the_judges_family():
    """The faithfulness score is only worth reading because the judge did not write the
    claim. A Qwen generator under a Qwen judge breaks that silently -- the pipeline
    runs, the gate hashes, the judge answers, and the number is now two models with the
    same blind spots agreeing. This test is the pin: it fails if the default moves into
    the judge's family, and it fails if the judge moves into the default's."""
    assert collides_with_judge(DEFAULT_JUDGE_MODEL)
    assert model_family(DEFAULT_JUDGE_MODEL) == JUDGE_FAMILY
    assert not collides_with_judge(DEFAULT_GENERATOR_MODEL)


def test_the_family_of_a_model_tag_survives_namespaces_tags_and_case():
    """`JUDGE_FAMILY` is a duplicated fact (importing the judge's model would make
    `generate` depend on `eval`), so the function that consumes it has to be blunt
    enough that the duplication cannot be defeated by how a model happens to be
    tagged."""
    assert model_family("qwen3.5:9b") == "qwen"
    assert model_family("qwen3:14b") == "qwen"
    assert model_family("hf.co/user/Qwen2.5-7B-GGUF:Q4_K_M") == "qwen"
    assert model_family("openbmb/minicpm-o4.5:latest") == "minicpm"
    assert model_family("minicpm-v:8b") == "minicpm"
    assert model_family("bakllava:latest") == "bakllava"


def test_a_colliding_model_is_allowed_but_says_so(caplog):
    """Not forbidden: comparing the two families deliberately is a legitimate
    experiment, and an experiment that cannot be run is not a control. But it must not
    be possible to do by accident, and the warning has to name the property being given
    up rather than just the model."""
    with caplog.at_level(logging.WARNING, logger="codelearner.generate.llm"):
        OllamaClaimGenerator(model="qwen3:14b")

    assert "same family" in caplog.text
    assert "faithfulness" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codelearner.generate.llm"):
        OllamaClaimGenerator()
    assert caplog.text == ""


def test_the_prompt_examples_name_no_symbol_a_real_repo_could_contain():
    """REGRESSION, from a real run, and the subtler of the two prompt failures.

    The first bad/good pair reused the actual symbols from the run that motivated it --
    `main` and `_free_port`, out of swarm-sync's demo package. On the next run the model
    answered for `demo.run_demo.main` with the GOOD example almost word for word, and
    that symbol is not the one the example was written about: the example described
    `demo._crash_agent.main`. So the claim was fluent, purpose-shaped, cited a real span,
    and was about a different function.

    That is the worst failure available to a few-shot example, because nothing
    downstream can see it. The gate hashes the citation and it verifies. The judge is
    asked whether the spans entail the claim, and a menu that includes a plausible
    `main` can entail it. It would have landed in the store as a good claim.

    An example is retrieved by name, so the only durable fix is that its names cannot
    collide with anything in a repo under index. This asserts the shape survives and the
    leaked wording does not."""
    system, _ = build_generation_prompt(subject="leases.acquire", offered=_offers())

    # The pair still exists and still teaches the contrast.
    assert "BAD:" in system and "GOOD:" in system
    assert "_next_ticket" in system and "_drain_outbox" in system

    # The specific prose that leaked is gone, in both its example and its subject form.
    assert "crash demo" not in system
    assert "already-running gateway" not in system
    assert "_free_port" not in system
    assert "--base-url" not in system

    # And the model is told what the examples are for, since it treated them as content.
    assert "do not exist" in system
    assert "never reach for the wording of an example" in system
