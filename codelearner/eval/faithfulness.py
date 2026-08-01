"""Faithfulness: does a tier-2 claim actually follow from the spans it cites?

The assertion store already answers a narrower question, and answers it with
arithmetic: the cited spans exist, and their bytes still hash to what was cited. A
model cannot argue with a sha256. But a claim can cite a span that is perfectly
present, perfectly unedited, and says nothing whatsoever about what the claim
asserts. That failure is invisible to every check in `assertions/store.py`, and it is
the one this module measures.

**The judge is a different model family from the generator, on purpose.** Claude
writes the assertions through `submit_assertion`; `qwen3.5:9b` judges them. This is
not a budget compromise -- a local 9B model is not chosen because it is cheaper than
an API call, it is chosen because it did not write the claim. A generator grading its
own output shares its blind spots, its tokenizer, its training distribution and its
particular way of being confidently wrong, so agreement between the two measures
consistency rather than truth. The one number this module produces is only worth
reading because the thing producing it is not the thing being measured. If the judge
is ever swapped for the generator, the score stops meaning anything and no test here
will notice, so it is stated in prose instead.

**The judge is prompted to refute, and defaults to "not supported".** A permissive
judge is worse than no judge: it converts an unmeasured risk into a number that reads
like a guarantee, and everything downstream then trusts the claim *more* than before
anyone looked at it. So every path that is not an explicit, parseable "supported"
lands on a label that stops the claim being served -- an unparseable response, an
empty response, a claim with no spans left to read. Fail-closed is not politeness
here; it is the only direction in which a bug is survivable.

**Only servable claims are scored.** The set under measurement comes from
`store.servable_assertions`, so every span handed to the judge has just been
re-hashed off disk. A stale claim is a different failure with a different repair, and
counting it as unfaithful would make the score move when the code moves -- which is
precisely the thing faithfulness must not be sensitive to. Staleness is already
measured, by hash, in `assertions/stale.py`.

**The verdict goes through `store.record_verdict`.** Nothing here writes to
`assertions` or `verdicts` directly, and nothing here decides what a non-supportive
verdict does to a claim's status. That policy -- one unsupportive verdict rejects,
rejection is a state and never a delete -- already exists and is already tested. The
three labels map onto the store's existing vocabulary rather than extending it:

    supported      -> VERDICT_SUPPORTED    the evidence entails the claim
    not_supported  -> VERDICT_REFUTED      the evidence contradicts or omits it
    uncertain      -> VERDICT_UNSUPPORTED  the judge could not reach a verdict

`uncertain` deliberately does not get its own status. The store's distinction is
already the right one: "the evidence says otherwise" and "the evidence is silent" are
different problems, and neither is a reason to keep answering questions with the
claim.

**The score is RAGAS-style faithfulness**: the fraction of claims a judge could
support from their own retrieved context. RAGAS decomposes a generated answer into
atomic statements first, because an answer is prose; here the decomposition already
happened upstream -- one assertion IS one atomic claim, and its `evidence_spans` ARE
its context -- so the metric applies directly to a row in the store.

The score of an empty set is `None`, not 1.0. "Every claim was supported" is
trivially true of no claims, and this repo has already been bitten by a vacuous
truth reading as success (see the `no_evidence` guard in `store.py`). A run that
adjudicated nothing must report that it adjudicated nothing.

**What this does not measure.** Whether the claim is *true* -- only whether the cited
evidence supports it. A correct claim citing an irrelevant span scores as unfaithful,
which is the intended behaviour: the citation is the only thing a later reader can
check. And the judge is one 9B model at temperature 0; it is a measurement with its
own error rate, not an oracle. `report.unfaithful` exists so that a low score is read
by looking at the claims that failed, not by trusting the number.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .. import db
from ..assertions import store

logger = logging.getLogger(__name__)

# The judge's three answers. Deliberately not the store's three verdicts: this is
# what a judge is asked for, `_STORE_VERDICT` below is what the store is told, and
# keeping them separate is what lets a judge express "I could not tell" without
# inventing a fourth assertion status to hold it.
LABEL_SUPPORTED = "supported"
LABEL_NOT_SUPPORTED = "not_supported"
LABEL_UNCERTAIN = "uncertain"

LABELS = (LABEL_SUPPORTED, LABEL_NOT_SUPPORTED, LABEL_UNCERTAIN)

_STORE_VERDICT = {
    LABEL_SUPPORTED: store.VERDICT_SUPPORTED,
    LABEL_NOT_SUPPORTED: store.VERDICT_REFUTED,
    LABEL_UNCERTAIN: store.VERDICT_UNSUPPORTED,
}

DEFAULT_JUDGE_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# One judgement is one short JSON object, but a thinking model can spend hundreds of
# tokens before it emits any of it. Generous rather than tight: a timeout that fires
# mid-generation costs the whole call and returns no verdict at all, which is a worse
# outcome than waiting.
DEFAULT_TIMEOUT_S = 180.0

# The 10GB card is shared -- the embedder wants ~1.2GB and the reranker ~3.4GB, and a
# 9B judge at ~6.6GB does not fit alongside both. `keep_alive` is short so the judge
# releases VRAM soon after a run instead of holding it until something else OOMs;
# `release()` makes that immediate for a caller that knows it is done. See
# `index/embed.py` for the other half of this: the embedder falls back to CPU rather
# than evicting whatever is resident, because it is not this tool's place to evict.
DEFAULT_KEEP_ALIVE = "5m"


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached, so there is no verdict -- not even a bad one.

    Distinct from an unparseable answer on purpose, and the distinction is the whole
    reason this exception exists. A model that replied with nonsense has adjudicated
    the claim badly, and `uncertain` is an honest record of that. A model that never
    replied has adjudicated nothing, and recording `uncertain` for it would reject
    every assertion in the store because ollama was not running. One is a verdict;
    the other must stop the run.
    """


@dataclass(frozen=True)
class Judgement:
    """One judge's answer about one claim, with the reasoning kept.

    The reasoning is not decoration. A score of 0.6 is only actionable if the four
    claims that failed can be read back with the sentence that failed them -- that is
    what separates "the generator is citing too loosely" from "the judge is being
    obtuse about a correct claim", and those call for opposite repairs.
    """

    label: str
    reasoning: str
    judge: str
    raw: str = ""

    @property
    def verdict(self) -> str:
        """This judgement in the store's vocabulary."""
        return _STORE_VERDICT[self.label]

    @property
    def supported(self) -> bool:
        return self.label == LABEL_SUPPORTED


@dataclass(frozen=True)
class Adjudication:
    """A judged assertion: what was claimed, what it cited, and what came back."""

    assertion_id: int
    subject_qualname: str
    claim: str
    citations: tuple[str, ...]
    judgement: Judgement
    verdict_id: int | None = None  # None when `record=False`

    @property
    def supported(self) -> bool:
        return self.judgement.supported

    def detail(self) -> str:
        """One assertion's outcome, in the form a human diagnoses a low score with."""
        return (
            f"[{self.judgement.label}] #{self.assertion_id} {self.subject_qualname}\n"
            f"  claim:    {self.claim}\n"
            f"  cited:    {', '.join(self.citations) or '(none)'}\n"
            f"  judge:    {self.judgement.reasoning.strip()}"
        )


@dataclass
class FaithfulnessReport:
    """The score and every per-assertion outcome behind it.

    Both, always. A faithfulness number with no attached detail cannot be acted on
    and cannot be checked -- it is exactly as trustworthy as the judge, with no way
    for a reader to form a second opinion about that.
    """

    judge: str
    adjudications: list[Adjudication] = field(default_factory=list)
    recorded: bool = True

    def __len__(self) -> int:
        return len(self.adjudications)

    def count(self, label: str) -> int:
        return sum(1 for a in self.adjudications if a.judgement.label == label)

    @property
    def score(self) -> float | None:
        """RAGAS-style faithfulness: supported / adjudicated. None over an empty set.

        None rather than 1.0. "Every claim was supported" is trivially true of no
        claims, and a vacuous truth reads as success everywhere it is not
        specifically looked for -- the same trap the store's `no_evidence` guard
        exists for. A caller that adjudicated nothing needs to find that out here,
        not from a perfect score it has no reason to distrust.
        """
        if not self.adjudications:
            return None
        return self.count(LABEL_SUPPORTED) / len(self.adjudications)

    @property
    def unfaithful(self) -> list[Adjudication]:
        """Everything the judge would not support, worst first.

        `not_supported` before `uncertain`: the first is the judge saying the
        citation does not carry the claim, the second is the judge failing to say
        anything, and only the first is evidence about the generator.
        """
        order = {LABEL_NOT_SUPPORTED: 0, LABEL_UNCERTAIN: 1}
        return sorted(
            (a for a in self.adjudications if not a.supported),
            key=lambda a: (order.get(a.judgement.label, 2), a.assertion_id),
        )

    def summary(self) -> str:
        score = "n/a (nothing adjudicated)" if self.score is None else f"{self.score:.3f}"
        return (
            f"faithfulness {score}  judge={self.judge}  "
            f"n={len(self.adjudications)} "
            f"supported={self.count(LABEL_SUPPORTED)} "
            f"not_supported={self.count(LABEL_NOT_SUPPORTED)} "
            f"uncertain={self.count(LABEL_UNCERTAIN)}"
            + ("" if self.recorded else "  (dry run -- no verdicts recorded)")
        )

    def format_report(self) -> str:
        """The summary line plus every claim the judge would not support."""
        lines = [self.summary()]
        failed = self.unfaithful
        if failed:
            lines.append("")
            lines.append(f"-- {len(failed)} claim(s) the judge would not support --")
            lines += [a.detail() for a in failed]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the judge seam
# --------------------------------------------------------------------------


class Judge(Protocol):
    """The seam that keeps the judge swappable -- and keeps it out of the tests.

    Same reason `Embedder` exists in `index/embed.py`: everything here depends on
    this protocol rather than on ollama, so the scoring, the label mapping, the
    fail-closed paths and the store integration are all asserted against a
    deterministic fake in milliseconds. No test in this repo may call a model.

    It also makes "which model judged this" a value rather than an assumption. The
    `judge` column in `verdicts` is filled from `name`, so a store holding verdicts
    from two different judges can still tell them apart afterwards.
    """

    @property
    def name(self) -> str: ...

    def judge(self, *, claim: str, evidence: str, subject: str) -> Judgement: ...


# The instruction the whole measurement rests on. Written to REFUTE.
#
# The failure mode being defended against is not a judge that is wrong, it is a judge
# that is agreeable. An LLM asked "does this claim follow from this evidence?" will
# say yes to almost anything plausible, because plausibility is what it is good at --
# and a judge that says yes to a claim citing an unrelated span produces a high
# faithfulness score for a store full of unaccountable claims, which is strictly
# worse than not measuring at all.
#
# So the task is inverted. The judge is told its job is to find the reason the claim
# fails, that the burden of proof is on the claim, and that silence in the evidence is
# a failure rather than a neutral outcome. The three rules named explicitly -- ignore
# your own knowledge of code, ignore what the names suggest, ignore what is obviously
# true -- are the three routes by which a model supports a claim its evidence does not
# carry.
_SYSTEM_PROMPT = """\
You are an adversarial verifier auditing claims made about source code. Your job is \
NOT to decide whether a claim is true. Your job is to decide whether the SPECIFIC \
EVIDENCE shown to you proves it.

You are trying to REFUTE the claim. The burden of proof is on the claim, not on you.

Rules:
1. The evidence below is the only thing that exists. Do not use what you know about \
how code like this usually behaves, what the function or variable names suggest, or \
what is obviously true in general. If the claim is correct but the evidence shown \
does not establish it, that is not_supported.
2. Judge the WHOLE claim. If any part of it -- a condition, a guarantee, a "because", \
a "always", a "never" -- is not established by the evidence, the claim is \
not_supported. A partly-supported claim is not supported.
3. Evidence that is merely consistent with the claim does not support it. Silence is \
not support.
4. If you are torn, answer not_supported. Answering supported when you are unsure is \
the worst outcome available to you.

Answer with a single JSON object and nothing else:
{"verdict": "supported" | "not_supported" | "uncertain", "reasoning": "<one or two \
sentences naming the specific thing that decided it>"}

Use "uncertain" ONLY when the evidence is unreadable or truncated so badly that you \
cannot assess the claim at all. A readable span that simply fails to establish the \
claim is not_supported, not uncertain.\
"""

_USER_TEMPLATE = """\
SUBJECT: {subject}
(the subject is a label for what the claim is about. It is not evidence -- a name \
that sounds like the claim proves nothing.)

EVIDENCE -- the complete set of spans this claim cites, and all you may use:
{evidence}

CLAIM: {claim}

Does that evidence, and nothing else, prove that claim?\
"""


def build_prompt(*, claim: str, evidence: str, subject: str) -> tuple[str, str]:
    """The (system, user) pair a judge is sent. Exposed so it can be read and diffed.

    A prompt is the single largest uncontrolled variable in an LLM-judged metric, so
    it is a named function rather than an f-string buried in a request body. Two runs
    whose scores differ should be able to establish whether the prompt was the same.
    """
    return _SYSTEM_PROMPT, _USER_TEMPLATE.format(
        subject=subject, evidence=evidence, claim=claim
    )


# Answers a model actually produces, mapped to the three labels. Checked as an exact
# match against a normalised key rather than by substring search, because `"not
# supported"` CONTAINS `"supported"` and a substring test would read the most
# dangerous answer in the set as the most permissive one.
_LABEL_ALIASES = {
    "supported": LABEL_SUPPORTED,
    "support": LABEL_SUPPORTED,
    "yes": LABEL_SUPPORTED,
    "entailed": LABEL_SUPPORTED,
    "true": LABEL_SUPPORTED,
    "not_supported": LABEL_NOT_SUPPORTED,
    "unsupported": LABEL_NOT_SUPPORTED,
    "refuted": LABEL_NOT_SUPPORTED,
    "contradicted": LABEL_NOT_SUPPORTED,
    "no": LABEL_NOT_SUPPORTED,
    "false": LABEL_NOT_SUPPORTED,
    "uncertain": LABEL_UNCERTAIN,
    "unknown": LABEL_UNCERTAIN,
    "unclear": LABEL_UNCERTAIN,
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def normalise_label(text: str) -> str | None:
    """Map a judge's word for its verdict onto one of `LABELS`. None if unrecognised.

    None, never a default. A caller deciding what an unrecognised answer means is the
    caller that has to fail closed, and returning `LABEL_UNCERTAIN` from here would
    make "the model said uncertain" and "the model said something I do not
    understand" the same value in the data.
    """
    key = re.sub(r"[^a-z]+", "_", text.strip().lower()).strip("_")
    return _LABEL_ALIASES.get(key)


def _extract_json(text: str) -> dict[str, object] | None:
    """Pull the judge's JSON object out of whatever it wrapped it in.

    Thinking models emit `<think>` blocks, chat models add fences and a sentence of
    preamble, and a schema-constrained model emits the bare object. All three are the
    same verdict and none of them is worth failing over -- but a response with no
    object in it at all is a parse failure, and returns None so the caller can fail
    closed rather than guess.
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


# Field-level recovery from JSON that does not parse. MEASURED, not anticipated: on a
# real run over 16 assertions about this repo, `qwen3.5:9b` judged
# `codelearner.index.embed.serialize` and produced
#
#   {"verdict": "supported", "reasoning": "... the format string `f"{len(values)}f"`
#    confirms it packs floats ..."}
#
# -- an unescaped double quote inside the reasoning, because the code it was quoting
# back contains one. The object is invalid JSON and the whole judgement was lost to
# `uncertain`, which is the correct fail-closed behaviour and also the wrong answer:
# the judge HAD reached a verdict and said so unambiguously. One in sixteen, and
# systematically biased toward claims about code containing quote characters, which in
# a Python repo means claims about anything involving strings.
#
# This does not loosen anything. The recovered token still has to normalise to a
# recognised label, so a garbled response cannot become `supported` unless the judge
# actually wrote `supported` in the verdict field. What it recovers is a verdict that
# was stated and mis-transcribed, not one that was never given.
_VERDICT_FIELD = re.compile(r'"(?:verdict|label)"\s*:\s*"([^"]{1,40})"')
_REASONING_FIELD = re.compile(
    r'"(?:reasoning|rationale)"\s*:\s*"(.+?)"?\s*\}?\s*$', re.DOTALL
)


def _salvage_fields(text: str) -> tuple[str, str] | None:
    """Pull `(verdict, reasoning)` out of malformed JSON. None if no verdict field.

    The verdict field is one short token and survives the escaping bugs that break
    the reasoning field beside it, so the two are recovered independently: a lost
    explanation is a nuisance, a lost verdict is a rejected claim.
    """
    verdict = _VERDICT_FIELD.search(text)
    if verdict is None:
        return None
    reasoning = _REASONING_FIELD.search(text)
    return verdict.group(1), (reasoning.group(1).strip() if reasoning else "")


def parse_judgement(text: str, judge: str) -> Judgement:
    """Turn a judge's raw response into a `Judgement`, failing closed.

    Every route that is not an explicit recognised "supported" ends at a label that
    stops the claim being served. That is the single most important line of code in
    this module: a permissive parser silently converts a judge that is malfunctioning
    into a store full of claims that were nominally adjudicated and actually were
    not, and it looks exactly like a good run.
    """
    parsed = _extract_json(text)
    note = ""
    if parsed is None:
        salvaged = _salvage_fields(text)
        if salvaged is None:
            return Judgement(
                label=LABEL_UNCERTAIN,
                reasoning=(
                    "no verdict field in the judge's response, so no verdict was read "
                    "from it. Failing closed: an unparseable answer is not a supported "
                    "claim."
                ),
                judge=judge,
                raw=text,
            )
        parsed = {"verdict": salvaged[0], "reasoning": salvaged[1]}
        note = " [verdict recovered from malformed JSON]"
    raw_label = parsed.get("verdict", parsed.get("label", ""))
    label = normalise_label(str(raw_label))
    reasoning = str(parsed.get("reasoning", parsed.get("rationale", ""))).strip() + note
    if label is None:
        return Judgement(
            label=LABEL_UNCERTAIN,
            reasoning=(
                f"unrecognised verdict {str(raw_label)!r} from the judge. Failing "
                f"closed. Judge said: {reasoning or '(nothing)'}"
            ),
            judge=judge,
            raw=text,
        )
    return Judgement(
        label=label,
        reasoning=reasoning or "(the judge gave a verdict with no reasoning)",
        judge=judge,
        raw=text,
    )


# Constrained decoding, so the judge cannot answer in a shape the parser has to
# guess at. ollama grammar-constrains generation to this schema, and the `enum`
# means the verdict is one of three exact strings or generation fails outright.
#
# The parser below is NOT redundant given this. Constrained decoding is a feature of
# this runtime rather than a property of a judge: an older ollama ignores `format`
# silently, a different backend has no equivalent, and `qwen3.5:9b` with thinking
# left on emits its reasoning OUTSIDE the constrained span. Measured: with
# `think: true` the reasoning consumed the whole 400-token budget and `content` came
# back empty, schema or no schema. So the schema removes variance and the parser
# still decides.
_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(LABELS)},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}


class OllamaJudge:
    """`Judge` backed by a local ollama model -- by default `qwen3.5:9b`.

    Local and small is a deliberate pairing with the generator, not a saving. See the
    module docstring: the point is that this model did not write the claims, and a
    model running on the same machine as the index is one that can be run on every
    assertion in a repo without a bill or a rate limit deciding how much of the store
    gets audited.

    Talks to ollama over `urllib` from the standard library. `httpx` is present in
    this venv as a transitive dependency of `mcp`, but relying on it would make the
    eval package depend on the optional MCP extra for one POST.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        think: bool = False,
        num_ctx: int = 8192,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._think = think
        self._num_ctx = num_ctx

    @property
    def name(self) -> str:
        """`ollama/<model>`, which is what lands in `verdicts.judge`.

        Prefixed rather than bare, so a stored verdict says where the judge ran as
        well as which weights: `qwen3.5:9b` served by ollama and the same weights
        behind some other runtime are not guaranteed to be the same judge.
        """
        return f"ollama/{self._model}"

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(  # noqa: S310 - fixed http(s) localhost URL
            f"{self._host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JudgeUnavailable(
                f"could not reach the judge at {self._host} ({exc}). No verdict was "
                f"reached, and none was recorded -- recording 'uncertain' here would "
                f"reject every assertion in the store because ollama was not running. "
                f"Start it (`ollama serve`), pull {self._model!r}, and re-run."
            ) from exc
        except json.JSONDecodeError as exc:
            raise JudgeUnavailable(
                f"the judge at {self._host} returned a body that is not JSON ({exc})."
            ) from exc
        if not isinstance(body, dict):
            raise JudgeUnavailable(
                f"the judge at {self._host} returned {type(body).__name__}, not an object."
            )
        return body

    def judge(self, *, claim: str, evidence: str, subject: str) -> Judgement:
        system, user = build_prompt(claim=claim, evidence=evidence, subject=subject)
        body = self._post(
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Thinking OFF, and this is a measured decision rather than a
                # preference. `qwen3.5:9b` reasons by default and its reasoning does
                # not fit in a verdict-sized budget: a 400-token cap was consumed
                # entirely by the think block, leaving `content` empty and the
                # judgement unparseable -- which fails closed to `uncertain` and would
                # have scored a whole store at zero for a formatting reason. Raising
                # the budget instead costs ~20x the wall time per claim, on a shared
                # card, to reach the same verdict.
                "think": self._think,
                "format": _RESPONSE_SCHEMA,
                "keep_alive": self._keep_alive,
                # Temperature 0 because a metric that moves when nothing changed
                # cannot be used to detect that something did. It does not make the
                # judge deterministic across ollama or GPU versions, and is not
                # relied on for that -- it removes the one source of variance that is
                # free to remove.
                "options": {
                    "temperature": 0.0,
                    "num_ctx": self._num_ctx,
                    "num_predict": 512,
                },
            },
        )
        message = body.get("message")
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")
            # ollama returns a thinking model's reasoning in its own field. It is not
            # the verdict, but it is the only explanation of one when `content` turns
            # out to be unparseable, so it is kept rather than dropped.
            thinking = str(message.get("thinking") or "")
            if not content.strip() and thinking:
                content = thinking
        return parse_judgement(content, self.name)

    def release(self) -> None:
        """Ask ollama to unload the judge now, freeing its VRAM.

        The card is shared and holds 10GB. A caller that judges a store and then runs
        the retrieval ablation needs the embedder and the reranker to fit, and a 6.6GB
        judge sitting on `keep_alive` is what makes them not. Best-effort by design:
        failing to free memory is not a reason to fail a completed measurement.
        """
        try:
            self._post("/api/chat", {"model": self._model, "messages": [], "keep_alive": 0})
        except JudgeUnavailable as exc:
            logger.warning("could not unload %s: %s", self._model, exc)


# --------------------------------------------------------------------------
# evidence rendering
# --------------------------------------------------------------------------


def render_evidence(root: Path, assertion: store.Assertion) -> str:
    """The cited spans, and only the cited spans, as the judge will see them.

    The whole measurement depends on this function being stingy. Handing the judge
    the enclosing file, the neighbouring function, or the symbol's docstring when the
    claim cited three lines of its body would let the judge support a claim from
    evidence a later reader cannot follow -- and the citation, not the file, is what
    the store re-hashes and what a human is pointed at. Widening the window here
    would raise the score while making the claims less accountable, which is the
    exact trade this module exists to make impossible.

    Each span is read from disk at its byte range. The caller is expected to have
    verified those hashes already (`adjudicate` goes through
    `store.servable_assertions`), so a read that disagrees with the citation means
    the file moved mid-run; the span is labelled as such rather than silently judged.
    """
    blocks: list[str] = []
    for i, span in enumerate(assertion.spans, start=1):
        target = root / span.path
        # Kept out of the `except OSError` below because this failure does not raise.
        # `read_bytes` on a FIFO blocks until some other process opens the write end,
        # so one cited pipe stalls the whole adjudication -- every later claim
        # unjudged, no traceback, no partial report, and a judging run that looks like
        # a slow model rather than a stopped one. `is_file()` is False for a FIFO, a
        # directory, a socket and a device node, and True for a regular file or a
        # symlink to one. Labelled like any other unreadable span rather than skipped:
        # a span the judge cannot see must still be visible to it as a span it cannot
        # see, or a claim gets judged on the evidence that happened to be readable.
        # It does not close the window between this test and the read; a regular file
        # swapped for a FIFO in between still blocks.
        if not target.is_file():
            text = "<<could not read this span: not a regular file>>"
        else:
            try:
                source = target.read_bytes()
                text = source[span.byte_start : span.byte_end].decode("utf-8", "replace")
            except OSError as exc:
                text = f"<<could not read this span: {exc}>>"
        blocks.append(f"--- span {i} of {len(assertion.spans)}: {span.citation} ---\n{text}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------


_NO_EVIDENCE = Judgement(
    label=LABEL_NOT_SUPPORTED,
    reasoning=(
        "the claim reached the judge with no evidence spans to read, so nothing "
        "could support it. Not sent to the judge and not counted as supported: "
        "'every cited span supports the claim' is trivially true of no spans, which "
        "is the one way a claim resting on nothing gets reported as verified."
    ),
    judge="(not consulted)",
)


def adjudicate_assertion(
    conn: sqlite3.Connection,
    judge: Judge,
    assertion: store.Assertion,
    root: Path,
    *,
    record: bool = True,
) -> Adjudication:
    """Judge one assertion against its own citations, and record the verdict.

    `record=False` is for measuring a judge without letting it change the store --
    calibrating a new one, or comparing two on the same claims. It is not the
    default, because a verdict nobody kept cannot be audited later and the rejected
    set is the only evidence the gate does anything.
    """
    citations = tuple(span.citation for span in assertion.spans)
    if not assertion.spans:
        judgement = _NO_EVIDENCE
    else:
        judgement = judge.judge(
            claim=assertion.claim,
            evidence=render_evidence(root, assertion),
            subject=assertion.subject_qualname,
        )
    verdict_id: int | None = None
    if record:
        # Through the store, never around it. What a non-supportive verdict does to a
        # claim's status is `record_verdict`'s decision and is already tested there;
        # a second implementation of that policy here is how the two come to disagree.
        verdict_id = store.record_verdict(
            conn,
            assertion.id,
            judgement.judge,
            judgement.verdict,
            rationale=judgement.reasoning,
        )
    return Adjudication(
        assertion_id=assertion.id,
        subject_qualname=assertion.subject_qualname,
        claim=assertion.claim,
        citations=citations,
        judgement=judgement,
        verdict_id=verdict_id,
    )


def adjudicate(
    conn: sqlite3.Connection,
    judge: Judge,
    repo_root: db.StrPath | None = None,
    *,
    kind: str | None = None,
    subject_qualname: str | None = None,
    limit: int | None = None,
    record: bool = True,
) -> FaithfulnessReport:
    """Score every servable assertion for faithfulness to its own citations.

    The candidate set is `store.servable_assertions`, which re-hashes every cited
    span off disk before returning it. That is not incidental: a claim whose evidence
    has been edited is stale, and staleness is a different failure with a different
    repair. Scoring it here would make faithfulness fall whenever the repo changed,
    and the number would then be measuring two things at once with no way to tell
    which one moved.

    Verdicts are recorded as each claim is judged rather than in one batch at the
    end, so a run interrupted halfway leaves the verdicts it actually reached instead
    of losing all of them.
    """
    # `repo_root` is passed through unresolved so that `store._repo_root` keeps its
    # refusal to guess. Defaulting to the cwd here would let a judge read "evidence"
    # from whatever happened to be at those paths relative to wherever the process
    # started, and score it -- an unbound index must fail loudly, not plausibly.
    candidates = store.servable_assertions(
        conn, repo_root=repo_root, kind=kind, subject_qualname=subject_qualname
    )
    root = Path(str(repo_root if repo_root is not None else db.stored_repo_root(conn)))
    if limit is not None:
        candidates = candidates[:limit]
    report = FaithfulnessReport(judge=judge.name, recorded=record)
    for assertion in candidates:
        report.adjudications.append(
            adjudicate_assertion(conn, judge, assertion, root, record=record)
        )
    return report


def faithfulness(adjudications: Sequence[Adjudication]) -> float | None:
    """RAGAS-style faithfulness over any set of adjudications. None over an empty set.

    Free function as well as a report property, so the metric can be recomputed over
    a subset -- one `kind` of claim, or one generator's output -- without re-running
    a judge.
    """
    if not adjudications:
        return None
    return sum(1 for a in adjudications if a.supported) / len(adjudications)
