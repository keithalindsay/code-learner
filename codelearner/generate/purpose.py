"""A model behind the `SourceView -> str` seam, without opening the leak boundary.

`eval/gold_from_history.py` builds a purpose eval and then measures three
deterministic baselines with it -- a docstring copier, a name echo, a bag of body
identifiers. The README is blunt about what that establishes: *"No generator is
measured here yet."* The harness has resolution; nothing has been put through it.
This module is the adapter that puts a real model through it, and every decision in
here is about doing that without quietly destroying the property that makes the
measurement worth reading.

That property is narrow and easy to lose. The gold labels are commit prose, and they
are gold **only** because the generator is structurally prevented from seeing them --
not because anyone promised not to look. `score_purposes` hands out a `SourceView`,
which has no field for a commit message, and runs both gates (`assert_view_is_source_only`
then `assert_no_leak`) on every symbol before it scores anything. An adapter that
reaches around that -- opens the file itself, re-parses to recover what was stripped,
queries the index -- would not fail. It would score *better*, and a better score is
the one outcome nobody investigates.

## Four things this adapter refuses to do

**It refuses input it cannot prove came from the working tree.** `__call__` runs
`assert_source_only` before it builds a prompt: the object must be exactly a
`SourceView` (not a subclass, not a duck type that happens to have the same
attributes), and its text must be present in the file it claims to come from. The
first half is the one that matters in a year's time -- the failure being defended
against is not today's harness, it is a future caller that decides the generator
would do better with "a bit more context" and passes a richer object through the same
seam. Structurally that object could carry the answer, and without the type check it
would be scored happily.

**It refuses to un-blind the `docstring_blind` condition.** That condition exists
because all 42 usable-labelled symbols on swarm-sync have a docstring, so without it
the eval largely measures how well the author documented their own code. The adapter
gets `view.without_docstring()` and that is all it gets: it does not read
`view.path` off disk, does not re-parse the source to recover the literal that was
stripped, and does not consult an index. The only cost of that discipline shows up in
`assert_source_only`, which cannot use the strict "is this a substring of the file"
check on a blinded view, because a blinded view is deliberately *not* a substring of
the file. The fallback is line-wise and is described at that function.

**It refuses to hand the backend anything but strings.** `PurposeModel.complete`
takes a system prompt and a user prompt and returns text. The backend never sees the
`SourceView`, and the prompt deliberately omits `view.path`, so nothing in the
transport layer is holding a pointer back to the file whose docstring was just
stripped. This is the reason the seam is a new protocol rather than `ClaimGenerator`
(see below).

**It refuses to turn an outage into a score.** A backend that cannot be reached
raises `GeneratorUnavailable` and the exception travels all the way out of
`score_purposes`. Returning `""` would be scored by `token_f1` as 0.0, which is a
legitimate-looking number, and a run made while ollama was down would read as
"the model is bad" rather than "there was no run". A model that answers with an empty
string is a different thing and is *not* an error -- it is counted in
`PurposeScorecard.empty_output`, which is what that field is for.

## Why a new protocol instead of `ClaimGenerator`

`generate/types.py` already defines a generator seam: `draft(subject=..., offered=...)
-> Draft`, where each `Offer` carries an `EvidenceSpan` and the model replies with
reference numbers. That design is right for tier-2 claims and wrong here, for a
reason that is specific rather than aesthetic: `Offer.span` is a path plus a byte
range. Bridging this seam through it would mean handing a purpose backend a live
pointer into the file whose docstring the `docstring_blind` condition has just
removed -- re-opening, at the transport layer, precisely the hole the condition
exists to close. The citation machinery would also be doing no work: there is exactly
one span a purpose statement about a symbol could cite, the symbol itself, so
`cited_refs` would be `(1,)` for every row by construction.

So the backend protocol here is `PurposeModel`, which is two strings in and one
string out. It is still a protocol, for the same reason `Judge` and `ClaimGenerator`
are: every test in this repo runs against a deterministic fake, and no test calls a
model.

## What this does not establish

- **It does not measure the shipped claim generator.** It measures a model's purpose
  inference at this seam, with this prompt. The generator that writes tier-2
  assertions through `pipeline` cites spans and is judged for faithfulness; a good
  number here is not transferable to it, and no adapter from `ClaimGenerator` is
  provided precisely so nobody reads one as the other.
- **It cannot stop a backend that goes to disk on its own.** The adapter hands over
  two strings, but a `PurposeModel` implementation is ordinary Python and could read
  the repo. Nothing short of a sandbox prevents that, and this module does not
  pretend otherwise -- what it guarantees is that the *adapter* never supplies the
  means, and that `suspect_tokens` in the harness will still notice output vocabulary
  that appears in the answer and nowhere in the input.
- **Normalisation moves the number.** See `NORMALISATION_RULE`. Comparing a model row
  against the baseline rows is only fair because the output has been cut down to the
  same shape the baselines emit, and cutting it down raises the score. The direction
  is stated rather than buried, and the shuffled control moves with it -- which is
  the usual argument for reading `lift` rather than `gold`.
- **One model at one temperature is a measurement, not a verdict**, exactly as
  `faithfulness.py` says of its judge.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from ..eval.gold_from_history import (
    Generator,
    LeakDetected,
    SourceView,
    assert_view_is_source_only,
)
from .types import GeneratorUnavailable

__all__ = [
    "MAX_PURPOSE_WORDS",
    "MAX_SOURCE_CHARS",
    "NORMALISATION_RULE",
    # Re-exported so a caller wiring this in does not need a third import to name the
    # exception a failed run raises.
    "GeneratorUnavailable",
    "LLMPurposeGenerator",
    "PurposeModel",
    "assert_source_only",
    "build_prompt",
    "llm_condition",
    "llm_conditions",
    "normalise_purpose",
]


# The output cap, in words. A backstop rather than a working part: the prompt asks for
# one sentence and the sentence split below already cuts at the first boundary, so on
# well-behaved output this never fires. It exists for the model that answers with a
# single 300-word run-on, which no sentence splitter can cut and which would otherwise
# score by sheer vocabulary volume against a 30-word label.
MAX_PURPOSE_WORDS = 40

# How much of a symbol's source goes into the prompt. Generous -- the median symbol is
# far shorter -- because truncation removes vocabulary the model could legitimately
# have used, and a truncated input is a worse measurement than a slow one.
MAX_SOURCE_CHARS = 6000


#: The normalisation rule, published in the same spirit as
#: `gold_from_history.LABELLING_RULE`: a thumb on the scale that is not written down
#: next to the number is a thumb nobody can argue with.
NORMALISATION_RULE = (
    "Model output is reduced to a single purpose statement before scoring: thinking "
    "blocks and code fences are removed, a leading label line ('Purpose:') is dropped, "
    "the first paragraph is taken, list markers and surrounding quotes are stripped, "
    "conversational preamble is removed ('Sure!', \"Here's a summary:\", 'This "
    "function', 'appears to', 'is responsible for'), the first sentence is kept, and "
    "the result is capped at 40 words. The direction of this bias is UP: token-F1 "
    "precision is overlap / |output tokens|, so deleting tokens that do not appear in "
    "the label can only raise the score, and an unnormalised model would be penalised "
    "for preamble rather than for misunderstanding. It raises the shuffled control by "
    "the same mechanism, so `lift` moves far less than `gold` -- read the lift. The "
    "baselines are terse by construction (a docstring's first sentence, a name, an "
    "identifier bag), so normalisation is what makes the model row a comparison rather "
    "than an artefact of answer length."
)


class PurposeModel(Protocol):
    """The swappable backend: two strings in, one string out.

    Narrower than `ClaimGenerator` on purpose, and the narrowness is the safety
    property rather than a simplification -- see the module docstring. A backend that
    never receives the `SourceView` cannot read `view.path` off disk, so the
    `docstring_blind` condition survives contact with the transport layer.

    `name` is a value rather than an assumption for the same reason `Judge.name` is:
    a report holding rows from two models has to be able to tell them apart, and a
    row labelled only "LLM" is a row that cannot be compared to next month's run.
    """

    @property
    def name(self) -> str: ...

    def complete(self, *, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------------
# The boundary check
# --------------------------------------------------------------------------------

# Characters a blinded docstring literal can leave behind on a line of its own once
# `SourceView.without_docstring` has removed the prose from between the quotes.
_QUOTE_RESIDUE = frozenset("\"'")


def assert_source_only(repo: Path | str, view: SourceView) -> None:
    """Raise `LeakDetected` unless `view` is a plain `SourceView` built from the tree.

    Two checks, defending two different failures.

    **The type check.** `type(view) is SourceView`, not `isinstance`. The frozen
    dataclass has no field for a commit message, which is the whole structural
    argument of the eval -- but a *subclass* does have room, and so does any object
    with the same eight attribute names. The failure this defends against is not
    today's harness, which is correct; it is a caller six months from now who decides
    the generator would do better with "a little more context" and passes a richer
    object through the same seam. That object could carry the answer, and every other
    check here would pass. Rejecting it costs nothing and is the only check that
    keeps working when the harness changes.

    **The content check.** `assert_view_is_source_only` is the repo's existing
    structural gate and is used unchanged wherever it applies: `view.source` must be a
    substring of the file it names. It cannot apply to a doc-blind view, because
    `without_docstring` rewrites the source specifically so that the docstring is no
    longer in it -- a blinded view is *supposed* to differ from the file. The obvious
    fix, re-reading the file and re-deriving the blinded form, is exactly the
    reach-around the condition forbids: it would put the stripped docstring in this
    process's memory, one attribute access away from the prompt builder.

    So the fallback is line-wise and never reconstructs anything. Every non-blank line
    of a blinded view's source must either be quote characters only (the residue
    blinding leaves where the literal was) or appear, whitespace-normalised, in the
    file. Commit prose spliced into a blinded view fails that: it is not in the file.
    It is a weaker check than the strict one -- a line could in principle be
    assembled from fragments the file happens to contain -- but every line it admits
    is still source vocabulary, which is the property being defended.
    """
    if type(view) is not SourceView:
        raise LeakDetected(
            f"the purpose adapter was handed {type(view).__name__}, not a SourceView. "
            "The leak boundary of this eval is the absence of a field to put a commit "
            "message in; an object with extra fields has one, so it is refused rather "
            "than scored."
        )
    if view.docstring is not None:
        # A view that still carries its docstring cannot be doc-blind, so the strict
        # gate must hold outright.
        assert_view_is_source_only(Path(repo), view)
        return
    try:
        # Undocumented symbols also have `docstring is None` and their source IS in
        # the file, so the strict gate is tried first and is what normally answers.
        assert_view_is_source_only(Path(repo), view)
    except LeakDetected:
        _assert_blinded_view_is_source_only(Path(repo), view)


def _assert_blinded_view_is_source_only(repo: Path, view: SourceView) -> None:
    """The line-wise gate for a doc-blind view. See `assert_source_only`."""
    try:
        raw = (repo / view.path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LeakDetected(f"{view.qualname}: view cannot be checked -- {exc}") from exc
    flat_file = " ".join(raw.split())
    for line in view.source.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= _QUOTE_RESIDUE:
            continue
        if " ".join(stripped.split()) not in flat_file:
            raise LeakDetected(
                f"{view.qualname}: a line of the doc-blind view is not in {view.path} "
                f"-- {stripped[:60]!r}. Something other than the working tree wrote it."
            )


# --------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------

# The instruction, written to suppress the two behaviours that would make the model's
# row incomparable to the baselines: preamble (which costs token-F1 precision for a
# reason unrelated to understanding) and narration of the implementation (which is
# what the `body identifiers` baseline already measures, and measures better).
#
# Rule 1 is not decoration either. The label is prose from the commit that introduced
# the symbol, so a model that reasons about callers, configuration or history it
# cannot see is inventing a purpose rather than inferring one, and any agreement that
# invention reaches with the label is luck.
_SYSTEM_PROMPT = """\
You are reading one symbol from a Python codebase and stating what it is FOR.

Rules:
1. The code shown to you is the only thing that exists. Do not reason about callers, \
tests, configuration or history you cannot see. If the code does not show a motive, \
describe the most specific thing the code actually does.
2. Answer with ONE sentence and nothing else. No preamble, no "This function...", no \
restating the question, no markdown, no code block, no bullet list, no closing remark.
3. Say what problem it solves or what it guarantees -- the reason it exists -- rather \
than narrating the implementation line by line.
4. Be specific. A sentence that would be true of any function in any codebase is \
worse than a narrow sentence that is only mostly right.\
"""

_USER_TEMPLATE = """\
SYMBOL: {qualname}
KIND: {kind}
SIGNATURE: {signature}
{docstring_block}SOURCE:
{source}

In one sentence: what is this symbol for?\
"""


def build_prompt(view: SourceView, *, max_source_chars: int = MAX_SOURCE_CHARS) -> tuple[str, str]:
    """The (system, user) pair the backend is sent. A named function so it can be diffed.

    A prompt is the largest uncontrolled variable in any model-in-the-loop metric, so
    it is not an f-string buried in a request body -- two runs whose numbers differ
    should be able to establish whether the prompt was one of the things that changed.
    The same argument as `faithfulness.build_prompt`, for the same reason.

    Two properties this function is responsible for:

    **The instructions are byte-identical across conditions.** The only difference
    between the normal and the doc-blind prompt is the presence of the DOCSTRING
    block, because that difference IS the condition. Nothing tells the model that a
    docstring was removed: a prompt that varied its wording between conditions would
    make the two rows differ for a reason other than the blinding.

    **`view.path` is not in the prompt.** It adds nothing to purpose inference, and
    leaving it out means the string handed to the backend does not contain a file
    pointer. That does not stop a determined backend from finding the file anyway
    (see the module docstring), but it does mean nothing here supplies it.
    """
    source = view.source
    if len(source) > max_source_chars:
        source = source[:max_source_chars] + "\n... (source truncated)"
    doc = (view.docstring or "").strip()
    block = f"DOCSTRING:\n{doc}\n" if doc else ""
    return _SYSTEM_PROMPT, _USER_TEMPLATE.format(
        qualname=view.qualname,
        kind=view.kind,
        signature=view.signature or "(none)",
        docstring_block=block,
        source=source,
    )


# --------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*")

# A line that is only a label the model put above its real answer.
_LABEL_ONLY = re.compile(
    r"^\**\s*(?:purpose|summary|answer|description|response|one[- ]sentence(?: summary)?)"
    r"\s*\**\s*:?\s*$",
    re.IGNORECASE,
)

# Leading list markers, block quotes and emphasis, plus wrapping quotes/backticks.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•>]+|\d+[.)])\s+")
_WRAPPERS = "\"'`*_ \t"

# Sentence boundary. Deliberately the same crude rule `gold_from_history` cuts a
# docstring's first sentence with, so the model's answer is cut at the same
# granularity as the baseline it is being compared against. Copied rather than
# imported because that regex is private to its module, and a private import is a
# worse coupling than a duplicated four-token pattern.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(—-])")

# Conversational scaffolding, stripped from the front repeatedly until the text stops
# changing. Every entry here was chosen because it is a phrase a chat-tuned model
# emits INSTEAD of answering, not because it is a phrase that carries no meaning:
# "is responsible for" is fine English, but as the opening of a one-sentence purpose
# statement it is four tokens of shape that the baselines never pay for.
_PREAMBLE = (
    r"(?:sure|certainly|of course|okay|ok|got it|absolutely)\b[\s,.:!-]*",
    r"here(?:['’]s| is)(?:\s+a)?(?:\s+(?:short|brief|concise|one[-\s]sentence))*"
    r"(?:\s+(?:summary|description|purpose|statement|answer|sentence))?\s*[:.\-]*\s*",
    r"(?:in\s+summary|in\s+short|briefly|overall|essentially|basically)\b[\s,.:-]*",
    r"(?:based\s+on|looking\s+at|reading|from)\s+(?:the\s+)?"
    r"(?:code|source|implementation|body|signature|docstring)\b[\s,.:-]*",
    r"the\s+(?:purpose|job|role|intent|goal|point)\s+of\s+(?:this|the)\s+\w+"
    r"(?:\s+\w+)?\s+(?:is|appears\s+to\s+be|seems\s+to\s+be)\s+(?:to\s+)?",
    r"(?:this|the)\s+(?:function|method|class|helper|routine|coroutine|property"
    r"|decorator|symbol|code|snippet)\b\s*",
    r"(?:it|this)\s+",
    r"(?:appears|seems)\s+to\s+(?:be\s+used\s+to\s+|be\s+)?",
    r"is\s+(?:responsible\s+for|used\s+to|intended\s+to|meant\s+to|designed\s+to"
    r"|there\s+to)\s+",
    r"(?:exists\s+to|serves\s+to)\s+",
)
_PREAMBLE_RE = re.compile(r"^\s*(?:" + "|".join(_PREAMBLE) + r")", re.IGNORECASE)

# Bound on the strip loop. Six is far more scaffolding than any real answer carries;
# the bound is here so a pathological pattern interaction cannot spin.
_MAX_PREAMBLE_STRIPS = 6


def normalise_purpose(text: str, *, max_words: int = MAX_PURPOSE_WORDS) -> str:
    """Reduce a model's answer to a single short purpose statement. See `NORMALISATION_RULE`.

    **This function biases the score upward, and the reader is owed that plainly.**
    `token_f1` precision is overlap / |output tokens|, so every token deleted here
    that does not appear in the label raises the score. Three sentences of "This
    function appears to..." would score lower than the same understanding stated
    once, for a reason that has nothing to do with understanding -- and the three
    baselines it is compared against are terse by construction. Normalising is the
    only way the model's row is a comparison rather than a measurement of answer
    length; not normalising would be a different thumb on the scale, pressing the
    other way, and pretending to be neutral.

    Two mitigations, neither of which makes it neutral. The same normalisation is
    applied before the shuffled control is computed, because the control scores the
    same string against a wrong label -- so `lift`, the number the harness asks you to
    read, moves much less than `gold`. And the rule is fixed and published rather
    than tuned per model.

    One failure this specifically avoids: **normalisation must never manufacture an
    empty answer.** An output that is nothing but scaffolding would otherwise reduce
    to `""`, which `token_f1` scores as 0.0 and `PurposeScorecard` counts as
    `empty_output` -- reporting the model as silent when it was merely verbose. If
    stripping empties the string, the flattened original is returned instead.
    """
    cleaned = _FENCE.sub(" ", _THINK_BLOCK.sub(" ", text))
    lines = [ln.strip() for ln in cleaned.splitlines()]
    # Leading label lines ("Purpose:") are dropped, along with any blank lines above
    # the real answer, so the first paragraph is the answer rather than its heading.
    while lines and (not lines[0] or _LABEL_ONLY.match(lines[0])):
        lines.pop(0)
    paragraph: list[str] = []
    for line in lines:
        if not line:
            break
        paragraph.append(line)
    # The first PARAGRAPH, not the first line: a model told to answer in one sentence
    # may still hard-wrap it, and cutting at the newline would truncate mid-clause.
    candidate = " ".join(" ".join(paragraph).split())
    candidate = _LIST_MARKER.sub("", candidate).strip(_WRAPPERS)
    for _ in range(_MAX_PREAMBLE_STRIPS):
        stripped = _PREAMBLE_RE.sub("", candidate, count=1).lstrip(_WRAPPERS)
        if stripped == candidate:
            break
        candidate = stripped
    first = _SENTENCE_END.split(candidate)
    candidate = first[0].strip() if first else candidate.strip()
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
    if not candidate.strip():
        # Everything was scaffolding. Report what the model actually said rather than
        # a manufactured silence -- see the docstring.
        candidate = " ".join(_FENCE.sub(" ", _THINK_BLOCK.sub(" ", text)).split())
        candidate = " ".join(candidate.split()[:max_words])
    return candidate.strip()


# --------------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------------


class LLMPurposeGenerator:
    """A `PurposeModel` behind `Generator = Callable[[SourceView], str]`.

    Callable, so it drops into `score_purposes` and `run_purpose_eval(conditions=...)`
    beside `docstring_purpose` and friends with no change to the scoring code. The
    interesting behaviour is what happens on the way in and on the way out:

    **On the way in**, `assert_source_only` runs on every call, *before* the cache is
    consulted. Checking after the cache would make a poisoned entry a way past the
    gate, and the gate is the only reason the number means anything.

    **On the way out**, the answer is normalised (`NORMALISATION_RULE`) and cached.
    The cache is keyed on the whole frozen `SourceView`, and that is the single most
    important line in this class. A cache keyed on `qualname`, or on
    `(path, qualname)` -- the obvious keys, and the ones a reasonable person writes --
    would hand the `docstring_blind` condition the answer generated from the
    docstring-bearing view, because the two conditions differ ONLY in the view.
    Nothing would fail; the blind row would simply come out level with the sighted
    one, which is exactly the result that gets reported as a finding. `SourceView` is
    a frozen dataclass, so value equality over all eight fields is free, and the
    blinded view differs in two of them.

    **A backend outage is never cached and never scored.** `GeneratorUnavailable`
    propagates out of `__call__`, out of `score_purposes`, and out of the run. See the
    module docstring for why an empty string would be worse than a crash.
    """

    def __init__(
        self,
        model: PurposeModel,
        repo: Path | str,
        *,
        cache: bool = True,
        max_words: int = MAX_PURPOSE_WORDS,
        max_source_chars: int = MAX_SOURCE_CHARS,
    ) -> None:
        """`repo` is required, because the check that needs it is not optional.

        The structural gate reads the file the view claims to come from, so it cannot
        run without a root. Making the root default to `None` would make the gate
        default to off, and a boundary that is enforced only when the caller
        remembered to pass an argument is a boundary that will be off in the run that
        matters.
        """
        self._model = model
        self._repo = Path(repo)
        self._max_words = max_words
        self._max_source_chars = max_source_chars
        self._cache: dict[SourceView, str] | None = {} if cache else None
        self._calls = 0

    @property
    def name(self) -> str:
        """The backend's name, so a scorecard row says which model produced it."""
        return self._model.name

    @property
    def calls(self) -> int:
        """Backend calls attempted, cache misses only. A call that raised still counts.

        Exposed because "did the cache work" and "did the blind condition really
        re-generate" are both questions about this counter, and the second one is a
        correctness question rather than a performance one.
        """
        return self._calls

    @property
    def cached(self) -> int:
        return len(self._cache) if self._cache is not None else 0

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def __call__(self, view: SourceView) -> str:
        assert_source_only(self._repo, view)
        if self._cache is not None:
            hit = self._cache.get(view)
            if hit is not None:
                return hit
        system, user = build_prompt(view, max_source_chars=self._max_source_chars)
        self._calls += 1
        # Not wrapped in try/except. A backend is responsible for raising
        # `GeneratorUnavailable` when it cannot reach its model, exactly as
        # `OllamaJudge` raises `JudgeUnavailable`; catching anything here would turn
        # an outage into a low score for the model.
        answer = normalise_purpose(
            self._model.complete(system=system, user=user), max_words=self._max_words
        )
        if self._cache is not None:
            self._cache[view] = answer
        return answer


# --------------------------------------------------------------------------------
# Conditions, in the shape DEFAULT_CONDITIONS uses
# --------------------------------------------------------------------------------


def _condition_name(base: str, docstring_blind: bool) -> str:
    """`<base>` or `<base>, doc-blind`, matching the baselines' own naming.

    `DEFAULT_CONDITIONS` names its pair "body identifiers" and "body identifiers,
    doc-blind", and a report is only readable if the model's pair is suffixed the same
    way. `PurposeScorecard.row` formats the name in 30 columns, so a long model name
    will push the row wider -- pass a short `name` if the table matters more than the
    provenance.
    """
    return f"{base}, doc-blind" if docstring_blind else base


def llm_condition(
    model: PurposeModel,
    repo: Path | str,
    *,
    name: str | None = None,
    docstring_blind: bool = False,
    generator: LLMPurposeGenerator | None = None,
    cache: bool = True,
) -> tuple[str, Generator, bool]:
    """One condition, in the exact shape `run_purpose_eval(conditions=...)` expects.

    `generator` lets a caller supply an adapter that already exists, which is how
    `llm_conditions` gives its two rows a shared cache. Passing one built against a
    different repo than `repo` is the caller's error and is not detected here -- the
    adapter's own gate will catch it on the first view.
    """
    adapter = generator or LLMPurposeGenerator(model, repo, cache=cache)
    return _condition_name(name or f"LLM {adapter.name}", docstring_blind), adapter, docstring_blind


def llm_conditions(
    model: PurposeModel,
    repo: Path | str,
    *,
    name: str | None = None,
    cache: bool = True,
) -> tuple[tuple[str, Generator, bool], ...]:
    """The sighted and doc-blind pair for one model, sharing one adapter.

    Both rows or neither. A model measured only on documented symbols with their
    docstrings visible is being measured against a corpus where the answer is
    correlated with the input -- one author wrote the docstring and the commit message
    in the same sitting -- and on swarm-sync that is not a corner case, it is all 42
    labelled symbols. The blind row is the one that says whether the model inferred
    anything, so this factory does not offer the option of reporting the flattering
    row alone.

    They share one `LLMPurposeGenerator`, and therefore one cache, deliberately: the
    two conditions hand it views that differ, so a cache that could not tell them
    apart would silently score the blind row with docstring-informed output. Making
    them share is what puts that hazard under test rather than out of sight.
    """
    adapter = LLMPurposeGenerator(model, repo, cache=cache)
    return (
        llm_condition(model, repo, name=name, docstring_blind=False, generator=adapter),
        llm_condition(model, repo, name=name, docstring_blind=True, generator=adapter),
    )
