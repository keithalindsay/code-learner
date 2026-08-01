"""The other half of the tier-2 measurement: a local model that writes the claims.

`eval.faithfulness` scores claims. This writes them, and it is deliberately built to
be bad at the thing a language model is best at. Asked what a function is for, a model
will produce a fluent, confident sentence about it whether or not the bytes it was
shown establish anything at all -- and a generator that always answers is exactly the
failure the tier-2 gate and the faithfulness judge exist to catch. Every design
decision below gives up some claim volume to make an unsupported claim harder to emit
and cheaper to refuse.

**It cites by number, and there is no other channel.** `build_generation_prompt`
renders the offers as a numbered menu and the model answers with integers. It is
never shown a path or a byte offset and it is never parsed for one; see the module
docstring of `types.py` for why a model-supplied offset is the one bad citation that
verifies forever while pointing at nothing. The menu here goes further than that
argument requires: it does not print the citations either, so a path in a response
cannot even be an echo of something the model read. The cost is real and is accepted
-- the model cannot say "these two spans are in the same file", so cross-file claims
have to rest on what the labels convey -- and it buys the property that a
location-shaped string in an answer is always something the model made up, and always
discarded.

**Not answering is a first-class answer.** A `Draft` with an empty claim, or with no
references, is what this returns when the model declines or when it produced something
unreadable. The pipeline refuses such a draft and no row is written. That is the point:
"these spans establish nothing about this symbol" is information, and it is the
information a fluent generator destroys by producing a plausible sentence instead.
The prompt says so explicitly, because a model that is not told abstention is allowed
will not abstain.

**The prompt asks for purpose, and names signature restatement as a wrong answer.**
This is the correction of a MEASURED failure, not a precaution. The first version of
this prompt said only "prefer a narrow claim you can support to a broad one you
cannot", and `llama3.1:8b` obeyed it exactly, over real symbols:

    demo._crash_agent.main    -> "The `main` function requires a `--base-url` argument."
    demo.run_demo._free_port  -> "The function `_free_port` returns an integer."
    demo.run_demo._setup_repo -> "`_setup_repo` takes a `workdir` parameter of type `Path`."

Restating the signature is the global optimum of "narrow and supportable": it is
trivially entailed by the span, so a faithfulness judge supports every one of them and
the run scores ~1.0. That is the worst number this project can produce -- a high score
carried entirely by claims that say nothing, indistinguishable from a high score
carried by good ones, and it would collapse against the purpose gold mined from commit
prose, which is about *why* a symbol was written. Narrowness was never the goal; it was
a hedge against over-claiming, and unhedged it degenerates into answering a different
question. So the prompt now fixes the question first ("what job does this do for its
callers, and why does it exist"), rejects the signature shape by name with a
bad/good pair, and re-scopes narrowness to mean claiming less of the JOB rather than
retreating to syntax. The anti-over-claim rules are untouched, and the escape route
from a symbol whose purpose really is not visible is still abstention, not a guess --
"if the only thing you can write is the signature, write nothing".

**Caller spans are what make a purpose claim supportable at all.** A function's job is
usually invisible from inside it: what it is for is what its callers do with it. The
pipeline puts caller and callee spans on the menu beside the subject, and the prompt
tells the model to read them for exactly that and to cite the caller span when the
caller is what showed it the purpose. Without that instruction the evidence-bound rule
and the purpose question genuinely do trade off, and the model resolves the conflict by
retreating to the signature -- which is what it did.

**An outage is not a result.** A backend that cannot be reached raises
`GeneratorUnavailable`; a backend that answered with rubbish returns an empty `Draft`.
Collapsing those would let ollama being down write "no describable purpose" against
every symbol in the repo, which is the same trap `JudgeUnavailable` exists to avoid on
the scoring side.

**The default model is not a Qwen model, and that is load-bearing.** The faithfulness
judge is `qwen3.5:9b`, and the entire argument for trusting its score is that it did
not write the claim -- different family, different training distribution, different way
of being confidently wrong. Running a Qwen generator against a Qwen judge does not
break any test in this repo; it silently converts the faithfulness number from a
measurement into an agreement statistic, and nothing downstream would notice. So the
default is `llama3.1:8b`, the model stays a constructor argument so the
collision can be measured on purpose, and `collides_with_judge` makes the hazard
something a caller or a test can check rather than something a docstring merely warns
about.

**No salvage of malformed JSON, unlike the judge.** `parse_judgement` reconstructs a
verdict out of broken JSON, because the field it is recovering is one short token from
a fixed set and a mis-transcribed `not_supported` is still unambiguously a refusal.
The field here is free prose. Half of a sentence is not half of a claim, it is a
different claim that nobody made, and storing one with real citations attached is
worse than losing the call. So a response that does not parse becomes an empty draft
and a warning in the log, and the prompt tries to prevent the failure at the source by
asking for backticks instead of double quotes -- the escaping bug that cost one
judgement in sixteen on this machine (see `_salvage_fields` in `eval/faithfulness.py`).

This module talks to ollama over `urllib` and duplicates a little of `OllamaJudge`'s
transport and JSON extraction rather than importing it. That is intentional: a shared
private helper spanning `generate` and `eval` would tie the thing being measured to
the thing measuring it, which is the one coupling this design cannot afford, and it is
a worse trade than thirty duplicated lines.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Sequence

from .types import Draft, GeneratorUnavailable, Offer

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_KEEP_ALIVE",
    "DEFAULT_KIND",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_TIMEOUT_S",
    "JUDGE_FAMILY",
    "OllamaClaimGenerator",
    "OllamaPurposeModel",
    "build_generation_prompt",
    "collides_with_judge",
    "model_family",
    "parse_draft",
    "render_menu",
]

# The generator's default. NOT a Qwen model, and not an arbitrary preference: see
# `collides_with_judge` below and the module docstring.
#
# Chosen by elimination and then by measurement. Of the models already on this machine
# (`qwen3.5:9b`, `qwen3:14b`, `openbmb/minicpm-o4.5:latest`, `minicpm-v:8b`,
# `bakllava:latest`) the two Qwens collide with the judge's family, and the rest are
# vision models. `openbmb/minicpm-o4.5` was the only survivor on paper and it does not
# survive contact: asked this exact task, schema-constrained, it did not answer inside
# a 300s timeout, which is not a slow model but an unusable one when the run is one
# call per symbol. `llama3.1:8b` was pulled for the purpose and answers the same probe
# in 2.9s with schema-clean output.
#
# So the default is a Meta model judged by a Qwen model, and the cross-family property
# the faithfulness number rests on holds by construction rather than by hope. The
# probe's claim was also subtly WRONG about the code it was shown -- it inverted the
# hash comparison -- which is the honest reason to keep it: a generator that never errs
# would leave the judge with nothing to catch and the measurement with nothing to say.
DEFAULT_GENERATOR_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# The `kind` written onto every draft unless a caller says otherwise. `purpose` is the
# only kind the prompt below actually asks for -- a caller that wants a different kind
# of claim needs a different prompt, not just a different string in the column.
DEFAULT_KIND = "purpose"

# One draft is a sentence and a short list of integers, but a model can spend a long
# time before it emits any of it. Generous rather than tight, for the same reason as
# the judge: a timeout that fires mid-generation costs the whole call and returns
# nothing, which is worse than waiting.
DEFAULT_TIMEOUT_S = 180.0

# The 10GB card is shared with the judge (~6.6GB) and the embedder (~1.2GB), and they
# do not co-reside. `keep_alive` is short so a finished generation run stops holding
# VRAM; `release()` makes it immediate for a caller that knows it is done.
DEFAULT_KEEP_ALIVE = "5m"

# The family the faithfulness judge belongs to, named here rather than imported from
# `eval.faithfulness.DEFAULT_JUDGE_MODEL`.
#
# This looks like the wrong call -- it is a duplicated fact, and duplicated facts drift.
# Importing it would make `generate` depend on `eval`, which is precisely the direction
# that must stay empty: the package that writes claims must not be able to reach into
# the package that grades them, or "the generator and the judge are independent" stops
# being structurally true. The duplication is pinned by a test that imports both and
# asserts `collides_with_judge(DEFAULT_JUDGE_MODEL)`, so a judge swapped to another
# family fails the suite here instead of drifting quietly.
JUDGE_FAMILY = "qwen"

_FAMILY_PREFIX = re.compile(r"[a-z]+")


def model_family(model: str) -> str:
    """The coarse family of an ollama model tag -- `qwen3.5:9b` -> `qwen`.

    A heuristic, and deliberately a blunt one: registry namespace dropped, tag
    dropped, then the leading run of letters. `openbmb/minicpm-o4.5:latest` ->
    `minicpm`, `hf.co/user/Qwen2.5-7B-GGUF` -> `qwen`. It over-matches by design,
    because the two error directions are not symmetric -- a spurious warning about
    two unrelated `llama` derivatives costs a reader ten seconds, while a missed
    Qwen-on-Qwen pairing costs the faithfulness number its meaning with no other
    symptom.
    """
    name = model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]
    match = _FAMILY_PREFIX.match(name)
    return match.group(0) if match else name


def collides_with_judge(model: str) -> bool:
    """True when this generator shares a model family with the faithfulness judge.

    The hazard made checkable. A Qwen generator judged by a Qwen judge produces a
    faithfulness score that measures agreement between two models with the same blind
    spots, and reads exactly like a score that measured truth. Nothing in the code can
    forbid it -- comparing the two families deliberately is a legitimate experiment --
    so the least this module can do is let a caller, a report, or a test ask.
    """
    return model_family(model) == JUDGE_FAMILY


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

# The instruction that decides what this module is worth. Written to UNDER-claim.
#
# The judge prompt in `eval/faithfulness.py` is written to refute; this is the other
# half of the same measurement and it is written against the same failure, approached
# from the other side. A model asked what a function is for will answer. It will answer
# when the span is a decorator, when the span is an import block, and when the span was
# truncated mid-line -- fluently, and in the register of something that read the code.
# Every one of those answers passes the store's gate (the citations are real spans, so
# they hash), and only the judge catches them, one at a time, after the fact.
#
# So the rules that matter are: only what the spans show, cite what you used and nothing
# else, and -- the one a model will not do unprompted -- answering with nothing is
# allowed and is preferred to a guess.
#
# The other half of the instruction is the correction of a measured failure, and it is
# the reason this prompt is longer than the judge's. Told only to be narrow and
# supportable, `llama3.1:8b` answered "the function `_free_port` returns an integer" and
# "the `main` function requires a `--base-url` argument" -- perfect compliance, and
# worthless. Restating a signature is the global optimum of narrow-and-supportable, it
# is entailed by the span so the judge supports it, and a store full of it scores ~1.0
# on faithfulness while saying nothing about any symbol in the repo.
#
# The fix is three moves and each one is load-bearing:
#
#   - the QUESTION is stated before the rules, because "describe this code" and "what is
#     this code for" have different right answers and the model was answering the first;
#   - the signature shape is named as a REJECTED answer with a bad/good pair, because the
#     model does not experience it as a failure -- it is being maximally careful, which
#     is what it was asked for, and only an explicit example distinguishes the two;
#   - narrowness is re-scoped to "claim less of the job", so that hedging can no longer
#     be satisfied by descending to syntax.
#
# The bad/good pair uses INVENTED symbols (`_next_ticket`, `_drain_outbox`), and that is
# the correction of a second measured failure rather than a stylistic choice. The first
# version of the pair reused the real symbols from the run that motivated it -- `main`
# and `_free_port` out of swarm-sync's demo -- and on the very next run the model
# returned the GOOD example almost verbatim for `demo.run_demo.main`, a DIFFERENT symbol
# from the one the example described. A few-shot example is retrieved by name, and a
# generator that meets the name again will relay the example instead of reading the
# spans. The output was fluent, purpose-shaped, and about the wrong function; against a
# judge scoring entailment from a menu that also contained the real `main`, it could
# plausibly have passed. So examples must never name a symbol that could appear in a repo
# being indexed, and the prompt says out loud that they are illustrations of shape rather
# than content.
#
# What is deliberately NOT done: nothing here licenses speculation. Rule 1 is unchanged,
# and the escape hatch from a symbol whose purpose is not visible is still an empty
# claim -- rule 5 closes the loop by saying that in the signature's own terms, so the
# model's fallback when it cannot find a purpose is silence rather than the shape it
# fell into last time.
_SYSTEM_PROMPT = """\
You are writing one claim about what a piece of source code is FOR, for an index whose \
claims are audited. An adversarial verifier will afterwards be shown your claim and \
the spans you cited, and nothing else, and asked to refute it.

A plain claim it cannot refute is worth more than an impressive one it can. But a claim \
that merely restates what the code obviously IS -- its name, its parameters, its return \
type -- is worth nothing at all: it is irrefutable because it says nothing.

THE QUESTION you are answering: what job does this code do for the rest of the program, \
and why does it exist? Not what it is called, not what it takes, not what it returns.

NOT AN ANSWER. These are the shapes to avoid, and they are what comes out if you stop \
reading too early:
  BAD:  The function `_next_ticket` returns an integer.
  BAD:  The `_drain_outbox` function requires a `queue` argument.
  GOOD: `_next_ticket` hands out the sequence number a writer stamps on a record, so \
two writers appending at once cannot be given the same slot.
  GOOD: `_drain_outbox` is the only place queued messages are actually delivered -- \
its callers just enqueue and return, which is what keeps them off the network path.
Restating a signature is not a narrow claim. It is a different question, answered.

The two symbols above do not exist. They are illustrations of the SHAPE of an answer, \
never of its content: say what the spans in front of you establish about the subject \
you were given, and never reach for the wording of an example.

You are shown a numbered menu of evidence spans. That menu is the only thing that \
exists.

Rules:
1. Say only what the shown spans establish. Do not use what you know about how code \
like this usually behaves, what the names suggest, or what is obviously true in \
general. If something is probably the case but the spans do not show it, it is not \
part of your claim.
2. Read the spans around the subject, not only the subject. What a symbol is FOR is \
usually invisible from inside it: what a caller passes in, what a caller does with the \
result, and what the subject calls in turn are the strongest evidence of purpose you \
will be given, and they are on the menu for that reason. Entries labelled `the \
subject` are the code being described; entries labelled `caller` are code that calls \
it; entries labelled `callee` are code it calls.
3. Cite every span you relied on, by its number, and cite no others. If a caller span \
is what showed you the purpose, cite that caller span. A span you did not use is not a \
citation, and padding the list makes a thin claim look well evidenced.
4. Prefer a narrow claim you can support to a broad one you cannot -- but narrow means \
claiming less of the job, not retreating to syntax. If the spans show it validating one \
field, say it validates that field: do not promote it to the validation layer, and do \
not demote it to taking a dict and returning a bool.
5. If the spans do not establish a purpose, answer with an empty claim and an empty \
list of references. That includes the case where you can see exactly what the code does \
mechanically and still cannot see what it is for, which is the usual case for \
boilerplate, for plain accessors, and for a subject shown with no callers. It is a \
correct and useful answer. A guess is not: it will be refused later anyway, and the only \
difference is whose time it wastes first. If the only thing you can write is the \
signature, write nothing.
6. Never write a file path, a line number or a byte offset. References are numbers \
from the menu, and there is no other way to cite here -- a location written instead of \
a number is discarded and cites nothing.
7. Write one or two plain sentences. Put identifiers in `backticks`, and do not use \
double quotes anywhere in your answer: a double quote inside the text has broken the \
JSON of a local model on this machine before.

Answer with a single JSON object and nothing else:
{"claim": "<one or two sentences saying what this is for, or an empty string>", \
"cited_refs": [<span numbers>]}\
"""

_USER_TEMPLATE = """\
SUBJECT: {subject}
(the subject is a label for the code you are describing. It is not evidence -- a name \
that sounds like a purpose establishes nothing about the code.)

EVIDENCE -- the numbered menu, and all you may use:
{menu}

What is {subject} FOR? Answer from the spans above, and list the numbers of the spans \
that support your answer -- including any caller span that is what showed you the \
purpose. If the spans show only what it does mechanically, and nothing about why it \
exists, answer with an empty claim and an empty list.\
"""


def render_menu(offered: Sequence[Offer]) -> str:
    """The offers as the model will see them: a number, a label, and the bytes.

    Each block is headed with the offer's own `ref` rather than with its position in
    the sequence. Those are normally the same and must never be assumed to be: the
    caller owns the numbering, `Draft.resolve` maps answers back through it, and a
    menu that renumbered its entries would produce drafts that cite the wrong spans
    while looking perfectly well-formed.

    No path, no line range, no byte offset -- deliberately, and this is the one place
    it costs something. The model cannot tell that two spans come from the same file,
    so a claim that depends on that relationship cannot be made here. In exchange, a
    path in a response is never an echo of anything the model was shown, which makes
    every location-shaped string in an answer a fabrication and none of them a
    near-miss worth trying to honour.

    The label is therefore doing more work than it looks like it is. It is the only
    structure the model gets: the pipeline writes it as `<role>: <kind> <qualname>`,
    and the role is what tells the model which block is the subject and which blocks
    are its callers -- which is what makes a claim about PURPOSE supportable at all,
    since a function's job is usually only visible from the code that uses it. The
    prompt names those role words in prose rather than importing them, because the
    generator must not depend on the pipeline that drives it; a test pins the two
    vocabularies together instead.

    The span text is passed through whole. Truncating it here would mean the model
    read something different from what a reader following the citation will read, and
    `Offer` carries `span` and `text` together specifically so those cannot drift; a
    menu too large for the context window is the caller's problem to size, and a loud
    one, rather than a silent shortening of evidence.
    """
    blocks = []
    for offer in offered:
        label = offer.label.strip() or "(unlabelled)"
        blocks.append(f"[{offer.ref}] {label}\n{offer.text}")
    return "\n\n".join(blocks)


def build_generation_prompt(*, subject: str, offered: Sequence[Offer]) -> tuple[str, str]:
    """The (system, user) pair a generator is sent. Exposed so it can be read and diffed.

    Mirrors `eval.faithfulness.build_prompt`, for the same reason: the prompt is the
    largest uncontrolled variable in anything a model produces, and two runs whose
    claims differ should be able to establish whether the instruction was the same.
    Pure -- it reads `offered` and returns strings, touches no clock, no network and no
    global, so the pair is a function of its arguments and a diff between two runs is
    evidence about the prompt rather than about when it was built.
    """
    return _SYSTEM_PROMPT, _USER_TEMPLATE.format(subject=subject, menu=render_menu(offered))


# --------------------------------------------------------------------------
# parsing, defensively
# --------------------------------------------------------------------------

# Constrained decoding, so `cited_refs` arrives as integers instead of prose to be
# salvaged. ollama grammar-constrains generation to this schema.
#
# The parser below is not redundant given it, for the reasons `_RESPONSE_SCHEMA` in
# `eval/faithfulness.py` records: an older ollama ignores `format` silently, another
# backend has no equivalent, and a thinking model emits its reasoning outside the
# constrained span. The schema removes variance; the parser still decides.
_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "cited_refs": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["claim", "cited_refs"],
}

# Adapted from `eval.faithfulness` rather than imported: `_THINK_BLOCK`, `_FENCE` and
# `_extract_json` are private to that module, and a shared helper reaching across the
# generator/judge boundary is a worse coupling than these few duplicated lines.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# A reference is an integer written as one. Not `int(text)`: that accepts `"  4_2 "`
# and unicode digits, and this is the one field where being liberal about what is
# accepted means accepting a citation that was never offered.
_INT_TEXT = re.compile(r"^[+-]?\d+$")

# Keys a model actually uses for the reference list. `cited_refs` is what the schema
# asks for and `refs` is the obvious abbreviation of it.
#
# `citations` is deliberately NOT here. That is the key a model reaches for when it is
# about to answer with paths and offsets, and reading it would be the first step toward
# honouring one -- the shape this whole package exists to make unrepresentable. A
# response that puts its references there loses them, which is the correct outcome:
# the draft comes back with nothing cited and the pipeline refuses it.
_REF_KEYS = ("cited_refs", "refs")

_EMPTY_DRAFT_REASON = "the generator answered, but nothing citeable could be read from it"


def _extract_json(text: str) -> dict[str, object] | None:
    """Pull the model's JSON object out of whatever it wrapped it in.

    Copied in spirit from `eval.faithfulness._extract_json`. Thinking models emit
    `<think>` blocks, chat models add fences and a sentence of preamble, and a
    schema-constrained model emits the bare object; all three are the same answer. A
    response with no object in it at all returns None so the caller can produce an
    empty draft rather than guess at one.

    The `<think>` block is stripped before anything looks for an object, not merely
    tolerated. A reasoning model drafts an answer inside the block and then abandons
    it, and the abandoned draft is the confident one -- reading it as the answer would
    recover exactly the claim the model decided it could not support.
    """
    cleaned = _FENCE.sub("", _THINK_BLOCK.sub("", text)).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _as_ref(value: object) -> int | None:
    """One element of the model's reference list as an int, or None if it is not one.

    `bool` is excluded before `int` because `True` is an `int` in Python and would
    otherwise become a citation of offer 1 -- a real span, cited by accident, by a
    model that answered `[true]`.

    Anything path-shaped lands here and returns None. `"codelearner/db.py[4120:4380]"`
    is not an integer, `{"path": ..., "byte_start": ...}` is not an integer, and there
    is no branch that tries to make either of them into one. A bare `4120` IS an
    integer and is kept, because an invented byte offset and a mis-typed reference are
    indistinguishable at this layer -- both are handled by not being on the menu.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INT_TEXT.match(value.strip()):
        return int(value.strip())
    return None


def parse_draft(text: str, *, kind: str = DEFAULT_KIND) -> Draft:
    """Turn a raw model response into a `Draft`, inventing nothing.

    Every failure route ends at an empty `Draft`, which the pipeline refuses. It never
    ends at an exception, because a model that answered badly has answered -- the run
    should record a refusal for this subject and carry on -- and it never ends at a
    partial claim, because there is no such thing: a claim recovered from half a
    response is a sentence the model did not write, carrying citations it did not
    choose. That is the one difference from `parse_judgement`, which does reconstruct a
    mangled verdict; a verdict is one token from a closed set, and this is prose.

    References that are not integers are dropped, because `Draft.cited_refs` cannot
    hold them and because everything that is not an integer here is a path, an object
    or a hallucinated location. References that ARE integers are kept even when they
    are outside the menu -- 0, -1, 4120, 99 -- and that is deliberate. `Draft.resolve`
    already discards and REPORTS them, and how often a generator cites off the menu is
    the main number this seam exists to produce; filtering them here would leave the
    pipeline with a clean-looking draft and no idea the model missed.

    Duplicates collapse to their first occurrence. `resolve` collapses them too, so
    nothing downstream changes -- but a `cited_refs` that repeats itself makes a
    one-span claim look like a three-span claim to anything that counts references
    before resolving them, including a human reading a report.
    """
    parsed = _extract_json(text)
    if parsed is None:
        logger.warning("%s: %r", _EMPTY_DRAFT_REASON, text[:400])
        return Draft(claim="", cited_refs=(), kind=kind)

    raw_claim = parsed.get("claim", parsed.get("statement", ""))
    claim = raw_claim.strip() if isinstance(raw_claim, str) else ""

    raw_refs: object = ()
    for key in _REF_KEYS:
        if key in parsed:
            raw_refs = parsed[key]
            break
    # A lone integer where a list was asked for is a shape a model produces and there
    # is nothing ambiguous about it. A string is NOT unwrapped the same way: `"1, 2"`
    # would have to be split, and a splitter is the thing that eventually reads
    # `"db.py:1-2"` as a pair of references.
    if isinstance(raw_refs, int) and not isinstance(raw_refs, bool):
        raw_refs = [raw_refs]
    candidates = raw_refs if isinstance(raw_refs, list | tuple) else ()

    refs: list[int] = []
    for value in candidates:
        ref = _as_ref(value)
        if ref is None:
            logger.warning("discarding a reference that is not an integer: %r", value)
        elif ref not in refs:
            refs.append(ref)

    if not claim:
        # An abstention is normalised to empty-and-empty. A model that returns no claim
        # but a list of references has cited spans in support of nothing, and leaving
        # those references attached invites a caller that checks `cited_refs` first
        # into treating it as a claim with evidence.
        return Draft(claim="", cited_refs=(), kind=kind)
    return Draft(claim=claim, cited_refs=tuple(refs), kind=kind)


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


def _chat_post(
    *,
    host: str,
    path: str,
    payload: dict[str, object],
    timeout: float,
    model: str,
    role: str,
    outage_detail: str,
) -> dict[str, object]:
    """One POST to ollama for both backends in this module.

    Extracted when the second backend arrived, not before. The duplication this module
    deliberately keeps is the duplication ACROSS packages -- `eval.faithfulness` has its
    own transport so that the thing being measured cannot import the thing measuring it
    -- and that argument says nothing about two classes in one file, where a second copy
    would only be a second place for the outage semantics to drift.

    Those semantics are the reason this is not three lines of `urlopen`. An unreachable
    backend, a timeout, a non-JSON body and a JSON body that is not an object are all
    raised as `GeneratorUnavailable`, because none of them is a model declining to
    answer, and a caller that cannot tell "ollama is down" from "the model had nothing
    to say" will write the second into its report every time the first happens.
    `role` and `outage_detail` exist so each backend can say what was lost in its own
    terms rather than share a message that fits neither.
    """
    request = urllib.request.Request(  # noqa: S310 - fixed http(s) localhost URL
        f"{host}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeneratorUnavailable(
            f"could not reach the {role} at {host} ({exc}). {outage_detail} "
            f"Start it (`ollama serve`), pull {model!r}, and re-run."
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneratorUnavailable(
            f"the {role} at {host} returned a body that is not JSON ({exc})."
        ) from exc
    if not isinstance(body, dict):
        raise GeneratorUnavailable(
            f"the {role} at {host} returned {type(body).__name__}, not an object."
        )
    return body


# --------------------------------------------------------------------------
# the ollama-backed generator
# --------------------------------------------------------------------------


class OllamaClaimGenerator:
    """`ClaimGenerator` backed by a local ollama model -- by default a non-Qwen one.

    **The model choice is a methodological constraint, not a preference.** The
    faithfulness judge in `eval/faithfulness.py` is `qwen3.5:9b`, and the whole reason
    its score is worth reading is that the judge is a different model family from
    whatever wrote the claim. Passing a Qwen model here -- `qwen3:14b` is the obvious
    local candidate, and it is the biggest text model on this machine -- breaks that
    property. It does not break it loudly: the pipeline runs, the gate still hashes
    spans, the judge still answers, and the number that comes out the far end is now
    two models from one family agreeing with each other. No test in this repo can see
    that, which is why the default is `llama3.1:8b`, why the model is
    still a constructor argument (measuring the collision on purpose is a legitimate
    experiment, and an unmeasured one is worthless), and why constructing a colliding
    generator logs a warning naming the property being given up. `collides_with_judge`
    is the programmatic form for a caller that would rather check than remember.

    **VRAM.** The card holds 10GB and is shared with the judge and the embedder, which
    cannot co-reside. `release()` is not optional housekeeping -- see its docstring.

    Talks to ollama over `urllib` from the standard library, matching `OllamaJudge`:
    `httpx` is in this venv only as a transitive dependency of the optional `mcp`
    extra, and depending on it would make claim generation require an ASGI stack for
    one POST.
    """

    def __init__(
        self,
        model: str = DEFAULT_GENERATOR_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        think: bool = False,
        num_ctx: int = 8192,
        num_predict: int = 512,
        kind: str = DEFAULT_KIND,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._think = think
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._kind = kind
        if collides_with_judge(model):
            logger.warning(
                "generator model %r is in the same family (%s) as the faithfulness "
                "judge. The faithfulness score assumes the judge did not write the "
                "claim; with this pairing it measures agreement between two models "
                "that share their blind spots, and nothing downstream will notice.",
                model,
                model_family(model),
            )

    @property
    def name(self) -> str:
        """`ollama/<model>`, which is what lands in `assertions.generator`.

        Prefixed rather than bare so a stored claim says where it was generated as well
        as by which weights: the same weights behind a different runtime, quantisation
        or sampler are not guaranteed to be the same generator, and that column is the
        only thing that lets a before/after comparison survive a re-run.
        """
        return f"ollama/{self._model}"

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """POST to ollama, raising `GeneratorUnavailable` for anything that is not an answer.

        Unreachable backend, timeout, a body that is not JSON, and a body that is JSON
        but not an object all land here rather than downstream: none of them is a model
        declining to make a claim, and the only way to keep that distinction is to
        refuse to produce a `Draft` from any of them.

        Kept as a method, delegating to `_chat_post`, rather than replaced by the free
        function: the tests fake the backend by patching this attribute on an instance,
        and a class whose transport cannot be swapped per-instance would force them
        onto `urlopen` and make every one of them a test of urllib.
        """
        return _chat_post(
            host=self._host,
            path=path,
            payload=payload,
            timeout=self._timeout,
            model=self._model,
            role="generator",
            outage_detail=(
                "No claim was drafted and none was refused -- returning an empty draft "
                "here would record 'these spans establish nothing' against every symbol "
                "in the repo because ollama was not running."
            ),
        )

    def draft(self, *, subject: str, offered: Sequence[Offer]) -> Draft:
        """Draft one claim about `subject` from the offered spans, citing by number.

        Returns an empty `Draft` when the model abstained, when it answered
        unreadably, and when there was nothing to offer it -- all three are refusals,
        and the pipeline treats them alike. Raises `GeneratorUnavailable` when the
        backend did not answer at all, which is not a refusal and must stop the run.

        The returned references are whatever integers the model gave, not a validated
        subset of the menu. Validation belongs to `Draft.resolve`, which drops
        off-menu references and hands the caller the list it dropped; doing it here
        would silently repair the generator's mistakes and delete the only measurement
        of how often it makes them.
        """
        if not offered:
            # Not sent. Every claim would be uncited, so the only answer the model
            # could give that is not a fabrication is the empty one -- and asking it
            # anyway spends a GPU-second to invite a guess.
            logger.debug("no offers for %s; refusing without consulting the model", subject)
            return Draft(claim="", cited_refs=(), kind=self._kind)

        system, user = build_generation_prompt(subject=subject, offered=offered)
        body = self._post(
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Thinking OFF by default, following the judge's measured result: a
                # reasoning model spends the whole token budget in the think block and
                # returns empty `content`, which here means an empty draft and a
                # refused claim for a formatting reason. Left as an argument because a
                # non-thinking model ignores it and a caller with budget may want it.
                "think": self._think,
                "format": _RESPONSE_SCHEMA,
                "keep_alive": self._keep_alive,
                # Temperature 0. Claim generation is not creative writing here: two
                # runs over an unchanged repo that produce different claims cannot be
                # compared, and comparing runs is the entire purpose of storing
                # `generator` alongside each claim. It does not make the model
                # deterministic across ollama or driver versions and is not relied on
                # for that -- it removes the variance that is free to remove.
                "options": {
                    "temperature": 0.0,
                    "num_ctx": self._num_ctx,
                    "num_predict": self._num_predict,
                },
            },
        )
        message = body.get("message")
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")
            # ollama puts a thinking model's reasoning in its own field. It is not the
            # answer, but when `content` comes back empty it is the only record of what
            # the model was doing instead of answering, so it is parsed rather than
            # dropped -- and if it holds no JSON object, the result is still an empty
            # draft.
            thinking = str(message.get("thinking") or "")
            if not content.strip() and thinking:
                content = thinking
        return parse_draft(content, kind=self._kind)

    def release(self) -> None:
        """Ask ollama to unload the generator now, freeing its VRAM.

        Mandatory in practice, not tidiness. The card holds 10GB and this model, the
        `qwen3.5:9b` judge and the embedder do not fit together -- and the natural
        shape of a run is generate, then judge what was generated. A generator sitting
        on `keep_alive` is precisely what makes the judge that follows it either OOM or
        fall back to CPU and take twenty minutes.

        Best-effort: a backend that cannot be reached to unload a model is not a reason
        to fail a generation run that already succeeded, so the failure is logged and
        swallowed.
        """
        try:
            self._post("/api/chat", {"model": self._model, "messages": [], "keep_alive": 0})
        except GeneratorUnavailable as exc:
            logger.warning("could not unload %s: %s", self._model, exc)


# --------------------------------------------------------------------------
# the ollama-backed purpose model
# --------------------------------------------------------------------------


class OllamaPurposeModel:
    """`generate.purpose.PurposeModel` backed by a local ollama model: strings in, string out.

    A second, deliberately dumber seam beside `OllamaClaimGenerator`, and the split is
    not an oversight. The claim generator exists to produce something the tier-2 gate
    can admit, so it cites, it abstains, and it answers in a constrained schema. This
    one exists to be *scored* by `eval.gold_from_history` against purpose labels mined
    from commit prose, where a citation has nothing to attach to -- there is one span,
    the symbol itself -- and a refusal is not an abstention but a missing measurement.
    Sharing one class between the two jobs would mean the eval scored a different
    artefact than the one the pipeline stores, under the same name, which is the
    ambiguity `purpose.py` explicitly refuses to create.

    So this returns free prose and no JSON schema. `normalise_purpose` in `purpose.py`
    is what turns that into something `token_f1` can compare, and it is applied to the
    shuffled control too -- the reason a model is allowed to be chatty here at all.

    **It is never handed a `SourceView`, and that is load-bearing.** The interface is
    two strings because `purpose.py` runs a `docstring_blind` condition that strips the
    docstring out of the source before the prompt is built; a backend holding a path, a
    byte range or an index handle could read the stripped text back and score the blind
    condition on documentation it was not supposed to see, with nothing raising. The
    narrow seam is what makes that reach-around unavailable rather than merely
    discouraged.

    Same VRAM story as its sibling: `release()` before running the judge, or the judge
    lands on a card that has no room for it.
    """

    def __init__(
        self,
        model: str = DEFAULT_GENERATOR_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        think: bool = False,
        num_ctx: int = 8192,
        num_predict: int = 256,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._think = think
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        if collides_with_judge(model):
            logger.warning(
                "purpose model %r is in the judge's family (%s). The purpose eval does "
                "not use the judge, so this is not the faithfulness collision -- but a "
                "run that reports both numbers side by side will be reporting one of "
                "them from a model related to the judge, and the report should say so.",
                model,
                JUDGE_FAMILY,
            )

    @property
    def name(self) -> str:
        """`ollama/<model>`, matching `OllamaClaimGenerator.name` and `OllamaJudge.name`.

        It reaches a scorecard row rather than a database column, so a report naming
        two conditions can say which weights produced each.
        """
        return f"ollama/{self._model}"

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """Patchable per instance, for the same reason as its sibling's: tests fake here."""
        return _chat_post(
            host=self._host,
            path=path,
            payload=payload,
            timeout=self._timeout,
            model=self._model,
            role="purpose model",
            outage_detail=(
                "No purpose was inferred. Returning an empty string here would be "
                "scored by token-F1 as a legitimate wrong answer, and a run where the "
                "backend was down would read as a model that understands nothing."
            ),
        )

    def complete(self, *, system: str, user: str) -> str:
        """One completion. Raises `GeneratorUnavailable` rather than returning `''`.

        The empty string is the one return value this must never invent. `token_f1`
        scores it as a real answer with no overlap, so an outage would come out the far
        end as a confident zero -- a number that looks exactly like a finding about the
        model and is a fact about ollama.
        """
        body = self._post(
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": self._think,
                "keep_alive": self._keep_alive,
                # Temperature 0 for the reason given in `OllamaJudge`: it removes the
                # one source of variance that is free to remove, and does not pretend
                # to make the run reproducible across ollama or driver versions.
                "options": {
                    "temperature": 0.0,
                    "num_ctx": self._num_ctx,
                    "num_predict": self._num_predict,
                },
            },
        )
        message = body.get("message")
        if not isinstance(message, dict):
            return ""
        content = str(message.get("content") or "")
        # A thinking model with a short budget can spend all of it reasoning and return
        # empty content. Its reasoning is not an answer, but `normalise_purpose` takes a
        # first sentence out of whatever it is given, and one sentence of reasoning about
        # the symbol scores closer to the truth than a guaranteed zero.
        if not content.strip():
            content = str(message.get("thinking") or "")
        return content

    def release(self) -> None:
        """Unload now. See `OllamaClaimGenerator.release` -- same card, same 10GB."""
        try:
            self._post("/api/chat", {"model": self._model, "messages": [], "keep_alive": 0})
        except GeneratorUnavailable as exc:
            logger.warning("could not unload %s: %s", self._model, exc)
