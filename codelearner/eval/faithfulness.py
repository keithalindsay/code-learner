"""Faithfulness: does a tier-2 claim actually follow from the spans it cites?

The assertion store already answers a narrower question, and answers it with
arithmetic: the cited spans exist, and their bytes still hash to what was cited. A
model cannot argue with a sha256. But a claim can cite a span that is perfectly
present, perfectly unedited, and says nothing whatsoever about what the claim
asserts. That failure is invisible to every check in `assertions/store.py`, and it is
the one this module measures.

**The judge is different weights and a different tokenizer from the generator: a
proxy for independence, not a demonstration of it.** Claude writes the assertions
through `submit_assertion`; `qwen3.5:9b` judges them. This is not a budget compromise
-- a local 9B model is not chosen because it is cheaper than an API call, it is chosen
because it did not write the claim. A generator grading its own output shares its
blind spots, its tokenizer, its training distribution and its particular way of being
confidently wrong, so agreement between the two measures consistency rather than
truth. The one number this module produces is only worth reading because the thing
producing it is not the thing being measured. If the judge is ever swapped for the
generator, the score stops meaning anything and no test here will notice, so it is
stated in prose instead.

What that pairing does NOT establish is independence, and the earlier wording here --
"a different model family, which is the point" -- claimed more than the check behind
it can carry. The check is `generate.llm.model_family`, a string-prefix test on an
ollama tag: it lowercases the tag, drops the registry namespace and the tag suffix,
and takes the leading run of letters. A Qwen-distilled model published as `deepseek-r1`
passes it. This repo's own reranker is Qwen3-based and would pass it. A published tag
is a marketing string, not a lineage, and no arrangement of its characters can prove
two models do not share pre-training data. Independence is a measurable property and
it is not measured here. **The measurement that would establish it is second-judge
agreement**: re-adjudicate the same claims with a second judge -- `qwen3:14b` is
already pulled locally, and a non-Qwen judge would be better -- and publish the
disagreement rate. Until that exists, read "cross-family" as the narrower thing that
is actually true: this model did not write the claim.

**One denominator used to hold three different events.** `score` is
`supported / n`, and `n` counts every `uncertain` -- but `uncertain` arrives by three
routes that are not the same fact. The judge can say "uncertain", which is the only
route the prompt sanctions and the only one that is evidence about the claim. Or
`_extract_json` and `_salvage_fields` can both fail, which is a harness/transport
failure and says nothing about the claim. Or a verdict token can fail to normalise,
which is a judge-format failure and also says nothing about the claim. Charging the
last two to the generator makes a run in which ollama returned malformed output report
a low faithfulness that reads as "the claims are bad" -- observed at 1-in-16 on a real
run before `_salvage_fields` existed. So `Judgement.cause` records which route was
taken, `FaithfulnessReport` counts the three apart, and past a threshold the run stops
instead of publishing a number about the generator that is really about the parser
(see `JudgeMisbehaving`).

`score` stays `supported / n` all the same, and `score_decided`
(`supported / (supported + not_supported)`) is reported beside it rather than in place
of it. Two reasons. `supported / n` is a lower bound under the assumption that an
undecided claim is at worst unfaithful; `score_decided` is an estimate that assumes
the undecided set is a random subset of the whole, which is known to be false here --
the parse failures that produced it were measurably biased toward claims about code
containing quote characters. And a headline that divides the instrument's failures out
of its own denominator stays flattering exactly when the instrument is broken, which
is the pathology the rest of this module is written against. Read the pair: a small gap
means the claims, a large gap means the instrument.

**The number carries an interval, because three decimals were a lie.** `0.544` on
n=147 has a Wilson 95% interval of `[0.464, 0.623]` -- the third decimal implies a
resolution two orders of magnitude finer than the measurement has. It is reported as
`0.54 [0.46, 0.62]`. That interval still assumes 147 independent draws, and the claims
are clustered: several per symbol, many per file, some about toy fixtures in
`sample_repo/`. Positive intra-cluster correlation makes the true interval wider, so
`clustered_interval` estimates the design effect from the adjudications themselves and
reports the corrected one alongside.

**The judge itself is uncalibrated on the data it measures, and this module says so.**
Its entire calibration is 15/16 on a *different* 16-claim set that was pre-labelled by
the same model family that authored those claims -- Wilson `[0.72, 0.99]`, which is
consistent with a judge that is right 72% of the time. "Self-consistency" measured as
three runs of an identical prompt at temperature 0 measures decoding determinism, not
judge stability. Label flips under whitespace-only prompt changes have been observed
with no rate attached, and that flip rate is the largest single uncertainty in the
number. `measure_prompt_stability` is the harness for measuring it and
`export_for_review` / `score_review` are the scaffold for the human calibration that
would bound the judge's error on the set the number is actually computed over. Neither
is a measurement; both are the apparatus. They are here so the missing measurements are
cheap enough to actually happen.

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
import math
import random
import re
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
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

# WHY a label arrived, which is a different question from what it is.
#
# Three of these produce `uncertain` and only one of them is evidence about the claim.
# Before this existed they were one value in the data, so a run in which the judge was
# fine and the transport was broken was indistinguishable from a run in which the judge
# read 147 spans and could not make up its mind about two of them -- and both reported a
# low faithfulness that a reader charges to the generator. The cause is recorded at the
# point the parser makes the decision, because that is the only place the information
# still exists.
CAUSE_JUDGED = "judged"  # the label is the one the judge stated
CAUSE_PARSE_FAILURE = "parse_failure"  # no verdict field anywhere in the response
CAUSE_FORMAT_FAILURE = "format_failure"  # a verdict token that does not normalise
CAUSE_NO_EVIDENCE = "no_evidence"  # not_supported, decided without asking a judge

CAUSES = (CAUSE_JUDGED, CAUSE_PARSE_FAILURE, CAUSE_FORMAT_FAILURE, CAUSE_NO_EVIDENCE)

# The two causes that are facts about the instrument rather than about the claim. Named
# as a pair because every threshold, counter and warning below is about the pair: a
# response with no verdict in it and a response with a verdict nobody recognises are
# the same failure wearing different hats, and neither was produced by the generator.
INSTRUMENT_CAUSES = (CAUSE_PARSE_FAILURE, CAUSE_FORMAT_FAILURE)

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

# 1.959963985..., the two-sided normal quantile for 95%. Spelled to full precision
# rather than as 1.96, because the rounded constant moves the third decimal of an
# interval whose whole purpose is to say the third decimal is not real.
Z_95 = 1.959963984540054

# Above this share of INSTRUMENT failures, `adjudicate` stops rather than reporting.
#
# The judgement being made: a handful of unparseable responses is noise and the run is
# still a measurement; a third of them means the harness is broken and every number the
# run produces is about the parser rather than about the claims. The threshold is ~3x
# the worst rate ever observed here (1-in-16 = 0.0625, before `_salvage_fields`), so a
# normal bad day does not trip it and the 30% case cannot avoid it.
#
# The floor matters as much as the rate: without it a run whose first response is
# malformed aborts at 1/1 = 1.0, which would make a single quote character in the first
# claim's code fatal to the whole store.
DEFAULT_MAX_INSTRUMENT_FAILURE_RATE = 0.2
MIN_INSTRUMENT_FAILURE_SAMPLE = 10


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached, so there is no verdict -- not even a bad one.

    Distinct from an unparseable answer on purpose, and the distinction is the whole
    reason this exception exists. A model that replied with nonsense has adjudicated
    the claim badly, and `uncertain` is an honest record of that. A model that never
    replied has adjudicated nothing, and recording `uncertain` for it would reject
    every assertion in the store because ollama was not running. One is a verdict;
    the other must stop the run.
    """


class JudgeMisbehaving(RuntimeError):
    """The judge is answering, and enough of the answers are unreadable that the run
    is no longer measuring the claims.

    The third case, between `JudgeUnavailable` (no answers at all) and a normal run.
    Deliberately NOT a subclass of `JudgeUnavailable`: a caller that retries on a
    transport fault should not retry this, because nothing about running it again
    changes a model that is emitting a shape the parser cannot read.

    Why it aborts at all, when a parse failure is individually survivable and recorded
    honestly as `uncertain`: the harm is not the missing verdict, it is that
    `record=True` turns each one into `VERDICT_UNSUPPORTED` and `STATUS_REJECTED` in
    the store. At a 30% instrument-failure rate, a run rejects 30% of a store's claims
    for a formatting reason, writes a rationale that blames the claim, and reports a
    faithfulness number that a reader charges to the generator. Counters make that
    visible after the fact; only stopping makes it not happen.

    The partial report is attached rather than lost. Verdicts already written stay
    written -- `adjudicate` records as it goes precisely so an interrupted run keeps
    what it reached -- and `report` is how a caller sees which claims those were.
    """

    def __init__(self, message: str, report: FaithfulnessReport | None = None) -> None:
        super().__init__(message)
        self.report = report


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
    cause: str = CAUSE_JUDGED

    @property
    def verdict(self) -> str:
        """This judgement in the store's vocabulary."""
        return _STORE_VERDICT[self.label]

    @property
    def supported(self) -> bool:
        return self.label == LABEL_SUPPORTED

    @property
    def instrument_failure(self) -> bool:
        """True when this label came from the harness failing, not from a judge.

        The distinction the counters, the abort threshold and the two scores are all
        built on. It is a property of the cause and never of the label: `uncertain` is
        the label all three routes land on, which is exactly why the label cannot be
        asked.
        """
        return self.cause in INSTRUMENT_CAUSES


@dataclass(frozen=True)
class Adjudication:
    """A judged assertion: what was claimed, what it cited, and what came back."""

    assertion_id: int
    subject_qualname: str
    claim: str
    citations: tuple[str, ...]
    judgement: Judgement
    verdict_id: int | None = None  # None when `record=False`
    # Exactly the bytes the judge was shown, kept rather than re-derived.
    #
    # `render_evidence` reads the working tree, so re-rendering an hour later can
    # produce a different string for the same citation -- and the two things that
    # consume this (`export_for_review`, `measure_prompt_stability`) are both
    # meaningless unless what they show a human, or perturb, is what the judge
    # actually read. Empty for a claim decided without consulting a judge.
    evidence: str = ""

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


# --------------------------------------------------------------------------
# how much of the number is real
# --------------------------------------------------------------------------


def wilson_interval(
    successes: float, total: float, *, z: float = Z_95
) -> tuple[float, float] | None:
    """Wilson score interval for a proportion. None when `total` is not positive.

    Wilson rather than the textbook normal approximation, and the reason is not
    fussiness. `p ± z·sqrt(p(1-p)/n)` produces intervals that run past 0 and 1, has
    zero width at p=0 and p=1 -- so a gate that refused 6,091 of 6,091 attacks would
    report `[1.0, 1.0]`, a certainty nobody measured -- and under-covers badly at the
    sample sizes this repo actually has. Wilson does none of those, needs no
    dependency, and is four lines.

    `successes` is a float rather than an int so a cluster-corrected effective sample
    can be passed straight through: `wilson_interval(p * n_eff, n_eff)`.
    """
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def format_interval(
    point: float | None, interval: tuple[float, float] | None, *, places: int = 2
) -> str:
    """`0.54 [0.46, 0.62]` -- the only form this number should be quoted in.

    Two decimals, because 0.544 on n=147 claims a resolution the interval says is not
    there by two orders of magnitude, and a third decimal is read as precision by
    every reader who does not stop to do the arithmetic.

    Rounded to nearest rather than outward. Outward rounding is the more conservative
    habit and it is wrong at this width: it would print `[0.46, 0.63]` for a bound of
    0.6226, widening the interval by 0.0074 to protect against a rounding error of
    0.0026.
    """
    if point is None:
        return "n/a"
    text = f"{point:.{places}f}"
    if interval is None:
        return text
    return f"{text} [{interval[0]:.{places}f}, {interval[1]:.{places}f}]"


@dataclass(frozen=True)
class ClusterCorrection:
    """The design effect of clustering, estimated from the outcomes themselves.

    Wilson assumes `n` independent draws. These are not: 147 claims about one repo are
    several claims per symbol and many per file, sharing a subject, a generator pass,
    an evidence window and -- for the `sample_repo/` fixtures -- a toy that is not
    representative of anything. Claims in one cluster fail together, so the effective
    number of independent observations is below the number of rows, and an interval
    computed from the row count is too narrow.

    Kish's design effect with the one-way ANOVA estimator of the intra-cluster
    correlation. It is an estimate with real limits, stated rather than buried: it
    assumes a common ICC across clusters, it is noisy when the number of clusters is
    small, and it is clamped at 0 because a negative ICC here means "no clustering
    signal in this sample", not "narrower than binomial".

    Not a substitute for the honest version, which is a cluster bootstrap or a
    hierarchical model over more than one repo. It is the correction that can be
    computed from what is already stored, and it is reported beside the uncorrected
    interval rather than instead of it, so a reader can see what the assumption bought.
    """

    key: str
    n: int
    clusters: int
    icc: float
    design_effect: float

    @property
    def effective_n(self) -> float:
        """The number of independent observations this sample is worth."""
        return self.n / self.design_effect if self.design_effect > 0 else float(self.n)


def cluster_correction(
    outcomes: Sequence[tuple[str, bool]], *, key: str = "cluster"
) -> ClusterCorrection | None:
    """Estimate the design effect from `(cluster_id, outcome)` pairs. None if it cannot.

    None rather than 1.0 when there is nothing to estimate from -- one cluster, or one
    observation per cluster. A design effect of 1.0 asserts "measured, and there is no
    clustering"; None says "this sample cannot answer that", and the two must not be
    the same value in a report about how much of a number is real.
    """
    if not outcomes:
        return None
    sizes: dict[str, int] = {}
    successes: dict[str, int] = {}
    for cluster_id, outcome in outcomes:
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
        successes[cluster_id] = successes.get(cluster_id, 0) + (1 if outcome else 0)
    n = sum(sizes.values())
    k = len(sizes)
    if k < 2 or k >= n:
        return None
    overall = sum(successes.values()) / n
    rates = {c: successes[c] / sizes[c] for c in sizes}
    ss_within = sum(sizes[c] * rates[c] * (1.0 - rates[c]) for c in sizes)
    ss_between = sum(sizes[c] * (rates[c] - overall) ** 2 for c in sizes)
    mean_within = ss_within / (n - k)
    mean_between = ss_between / (k - 1)
    # Kish's m0: the average cluster size adjusted for how uneven the sizes are. Using
    # the plain mean would understate the design effect whenever a few large clusters
    # carry most of the rows, which is the shape a real repo has.
    m0 = (n - sum(m * m for m in sizes.values()) / n) / (k - 1)
    denominator = mean_between + (m0 - 1.0) * mean_within
    icc = 0.0 if denominator <= 0 else (mean_between - mean_within) / denominator
    icc = min(max(icc, 0.0), 1.0)
    return ClusterCorrection(
        key=key, n=n, clusters=k, icc=icc, design_effect=1.0 + (m0 - 1.0) * icc
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

    def count_cause(self, cause: str) -> int:
        return sum(1 for a in self.adjudications if a.judgement.cause == cause)

    @property
    def parse_failures(self) -> int:
        """`uncertain` because nothing in the response was a verdict field.

        A harness/transport fact. The judge may well have reached a verdict and said
        so; what failed is the round trip between it and this parser.
        """
        return self.count_cause(CAUSE_PARSE_FAILURE)

    @property
    def format_failures(self) -> int:
        """`uncertain` because the verdict token did not normalise to a known label."""
        return self.count_cause(CAUSE_FORMAT_FAILURE)

    @property
    def instrument_failures(self) -> int:
        """The two causes that are evidence about the instrument, not the claims."""
        return self.parse_failures + self.format_failures

    @property
    def judge_uncertain(self) -> int:
        """`uncertain` because the judge said so -- the only sanctioned route.

        The one uncertain the prompt asks for, and the only one of the three that is
        evidence about the claim: the judge read the evidence and reported that it was
        unreadable or truncated past assessment.
        """
        return sum(
            1
            for a in self.adjudications
            if a.judgement.label == LABEL_UNCERTAIN
            and a.judgement.cause == CAUSE_JUDGED
        )

    @property
    def decided(self) -> int:
        """Claims the judge actually reached a supportive or refuting verdict on."""
        return self.count(LABEL_SUPPORTED) + self.count(LABEL_NOT_SUPPORTED)

    @property
    def score(self) -> float | None:
        """RAGAS-style faithfulness: supported / adjudicated. None over an empty set.

        None rather than 1.0. "Every claim was supported" is trivially true of no
        claims, and a vacuous truth reads as success everywhere it is not
        specifically looked for -- the same trap the store's `no_evidence` guard
        exists for. A caller that adjudicated nothing needs to find that out here,
        not from a perfect score it has no reason to distrust.

        Still the headline, and still over the full denominator, after the audit that
        added `score_decided`. It is a lower bound: it holds if an undecided claim is
        at worst unfaithful, which needs no assumption about why it was undecided.
        `score_decided` needs one -- that the undecided set is a random subset -- and
        that assumption is known to be false here. And a headline that divides the
        instrument's own failures out of its denominator is at its most flattering
        exactly when the instrument is broken. The bad run is made visible by the
        counters and stopped by `JudgeMisbehaving`, not by choosing a kinder quotient.
        """
        if not self.adjudications:
            return None
        return self.count(LABEL_SUPPORTED) / len(self.adjudications)

    @property
    def score_decided(self) -> float | None:
        """supported / (supported + not_supported). None when nothing was decided.

        The other question: of the claims this judge could read at all, what fraction
        did their citations carry? Unlike `score` it does not move when the transport
        breaks, which is what makes it worth reporting -- and it also does not move
        when the transport breaks in a way that is *correlated with the claims*, which
        is what makes it not worth reporting alone. The measured example: unescaped
        quotes in the judge's own reasoning, which fire on claims about code containing
        string literals.

        Read as a pair with `score`. A small gap means the two agree about the
        generator. A large gap means most of what `score` is reporting is the
        instrument, and `instrument_failures` says whether that is the harness or the
        judge's honest doubt.
        """
        if self.decided == 0:
            return None
        return self.count(LABEL_SUPPORTED) / self.decided

    @property
    def interval(self) -> tuple[float, float] | None:
        """Wilson 95% on `score`, assuming 147 independent draws. They are not."""
        if not self.adjudications:
            return None
        return wilson_interval(self.count(LABEL_SUPPORTED), len(self.adjudications))

    @property
    def interval_decided(self) -> tuple[float, float] | None:
        """Wilson 95% on `score_decided`."""
        if self.decided == 0:
            return None
        return wilson_interval(self.count(LABEL_SUPPORTED), self.decided)

    def cluster_correction(self, *, key: str = "subject") -> ClusterCorrection | None:
        """Design effect for `score`, clustering by `subject` or by `file`.

        `subject` groups the claims about one symbol; `file` groups by the path of the
        first citation, which is the coarser and usually the stronger clustering --
        one badly-cited file produces a run of correlated failures. Both are available
        because which one dominates is an empirical question about a given repo, and
        the answer is not knowable from here.
        """
        outcomes = [(_cluster_id(a, key), a.supported) for a in self.adjudications]
        return cluster_correction(outcomes, key=key)

    def clustered_interval(self, *, key: str = "subject") -> tuple[float, float] | None:
        """`interval` widened by the estimated design effect. None if not estimable."""
        correction = self.cluster_correction(key=key)
        if correction is None or self.score is None:
            return None
        effective = correction.effective_n
        return wilson_interval(self.score * effective, effective)

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

    def warnings(self) -> list[str]:
        """What a reader must know before quoting the number. Empty on a clean run.

        Returned as data rather than logged, so the surface that prints the score is
        structurally unable to print it without them.
        """
        notes: list[str] = []
        n = len(self.adjudications)
        if n and self.instrument_failures:
            rate = self.instrument_failures / n
            notes.append(
                f"{self.instrument_failures}/{n} ({rate:.0%}) of these labels came from "
                f"the harness failing to read a response, not from a judge. They are "
                f"counted in the denominator of `score` and excluded from "
                f"`score_decided`; the gap between the two is what they cost."
            )
        if n and self.judge_uncertain:
            notes.append(
                f"{self.judge_uncertain}/{n} claims the judge declined to decide. "
                f"`score` charges those to the generator; `score_decided` drops them."
            )
        return notes

    def summary(self) -> str:
        if self.score is None:
            headline = "faithfulness n/a (nothing adjudicated)"
        else:
            headline = f"faithfulness {format_interval(self.score, self.interval)}"
        lines = [
            f"{headline}  judge={self.judge}"
            + ("" if self.recorded else "  (dry run -- no verdicts recorded)"),
            f"  n={len(self.adjudications)} "
            f"supported={self.count(LABEL_SUPPORTED)} "
            f"not_supported={self.count(LABEL_NOT_SUPPORTED)} "
            f"uncertain={self.count(LABEL_UNCERTAIN)}"
            f" (judge={self.judge_uncertain}"
            f" parse_failures={self.parse_failures}"
            f" format_failures={self.format_failures})",
        ]
        if self.score_decided is not None:
            lines.append(
                "  supported/(supported+not_supported) = "
                f"{format_interval(self.score_decided, self.interval_decided)}"
                f"  on {self.decided}/{len(self.adjudications)} decided"
            )
        correction = self.cluster_correction()
        clustered = self.clustered_interval()
        if correction is not None and clustered is not None:
            lines.append(
                f"  clustered by {correction.key} "
                f"(k={correction.clusters}, icc={correction.icc:.2f}, "
                f"deff={correction.design_effect:.2f}, "
                f"n_eff={correction.effective_n:.0f}): "
                f"{format_interval(self.score, clustered)}"
            )
        lines += [f"  ! {note}" for note in self.warnings()]
        return "\n".join(lines)

    def format_report(self) -> str:
        """The summary block plus every claim the judge would not support."""
        lines = [self.summary()]
        failed = self.unfaithful
        if failed:
            lines.append("")
            lines.append(f"-- {len(failed)} claim(s) the judge would not support --")
            lines += [a.detail() for a in failed]
        return "\n".join(lines)


def _cluster_id(adjudication: Adjudication, key: str) -> str:
    """The cluster an adjudication belongs to, under `subject` or `file` grouping."""
    if key == "subject":
        return adjudication.subject_qualname
    if key == "file":
        first = adjudication.citations[0] if adjudication.citations else ""
        return first.rsplit(":", 1)[0] if ":" in first else first
    raise ValueError(f"unknown cluster key {key!r}; expected 'subject' or 'file'")


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
                    "claim. This is a harness failure and not evidence about the "
                    "claim -- see `parse_failures` on the report."
                ),
                judge=judge,
                raw=text,
                cause=CAUSE_PARSE_FAILURE,
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
            cause=CAUSE_FORMAT_FAILURE,
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
    # Decided without a judge, and still a fact about the generator rather than about
    # the instrument: a claim that cited nothing is the generator's doing. So it counts
    # in BOTH denominators, and is not an instrument failure.
    cause=CAUSE_NO_EVIDENCE,
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
    evidence = ""
    if not assertion.spans:
        judgement = _NO_EVIDENCE
    else:
        evidence = render_evidence(root, assertion)
        judgement = judge.judge(
            claim=assertion.claim,
            evidence=evidence,
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
        evidence=evidence,
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
    max_instrument_failure_rate: float | None = DEFAULT_MAX_INSTRUMENT_FAILURE_RATE,
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

    `max_instrument_failure_rate` stops the run when too much of what is coming back
    cannot be read -- see `JudgeMisbehaving` for why that is worth an abort when an
    individual parse failure is not. `None` disables it, which is the right setting
    for deliberately measuring a judge's output shape and the wrong one for producing
    a number anybody quotes.
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
        _check_instrument(report, max_instrument_failure_rate)
    return report


def _check_instrument(
    report: FaithfulnessReport, max_rate: float | None
) -> None:
    """Stop a run whose responses have stopped being readable. See `JudgeMisbehaving`.

    Checked after every claim rather than at the end, because the harm being prevented
    accrues per claim: each instrument failure recorded is one more live assertion
    demoted to rejected for a formatting reason.
    """
    if max_rate is None:
        return
    n = len(report.adjudications)
    failures = report.instrument_failures
    if n < MIN_INSTRUMENT_FAILURE_SAMPLE or failures / n <= max_rate:
        return
    raise JudgeMisbehaving(
        f"{failures} of the first {n} responses from {report.judge} could not be read "
        f"({failures / n:.0%} > {max_rate:.0%}): "
        f"{report.parse_failures} with no verdict field, {report.format_failures} with "
        f"an unrecognised verdict. That is a fact about the harness, not about the "
        f"claims -- continuing would reject live assertions for a formatting reason "
        f"and report a faithfulness number that is really a parse rate. Verdicts "
        f"already recorded are kept and are on the attached `report`.",
        report,
    )


def faithfulness(
    adjudications: Sequence[Adjudication], *, decided_only: bool = False
) -> float | None:
    """RAGAS-style faithfulness over any set of adjudications. None over an empty set.

    Free function as well as a report property, so the metric can be recomputed over
    a subset -- one `kind` of claim, or one generator's output -- without re-running
    a judge.

    `decided_only=True` is `FaithfulnessReport.score_decided`: the supported fraction
    of the claims the judge reached a verdict on. Default False, matching the headline,
    for the reasons on `score` -- the full denominator is the one that does not get
    more flattering as the instrument gets worse.
    """
    pool = (
        [a for a in adjudications if a.judgement.label != LABEL_UNCERTAIN]
        if decided_only
        else list(adjudications)
    )
    if not pool:
        return None
    return sum(1 for a in pool if a.supported) / len(pool)


# --------------------------------------------------------------------------
# measuring the instrument: prompt stability
# --------------------------------------------------------------------------
#
# The largest unmeasured uncertainty in the faithfulness number. The judge's whole
# calibration is 15/16 on a DIFFERENT 16-claim set -- Wilson [0.72, 0.99], consistent
# with a judge that is right about three quarters of the time -- and the
# "self-consistency" figure beside it is three runs of an identical prompt at
# temperature 0, which measures whether decoding is deterministic and not whether the
# judge is stable. Label flips under whitespace-only prompt changes have been observed
# here and no rate was ever attached to them.
#
# This is the apparatus for attaching one. It re-adjudicates a stored set under
# transformations of `render_evidence` that change no information a verdict could
# legitimately depend on, and reports how often the label moved anyway. A flip is not a
# wrong answer -- neither run is ground truth -- it is a demonstration that the answer
# was not determined by the evidence. A 10% flip rate means roughly a tenth of every
# faithfulness number this repo has published is the prompt's formatting.
#
# Deliberately NOT run by any test and NOT run here: this calls a model once per
# (claim x perturbation), so 147 claims and the five default perturbations is 735 judge
# calls on a shared card.


def _perturb_identity(evidence: str) -> str:
    """No change. The control, and the reason the rest of the numbers mean anything.

    Without it a flip rate conflates prompt sensitivity with decoding nondeterminism,
    and the two have opposite repairs: one is fixed by constraining the prompt, the
    other by pinning the runtime. Whatever this perturbation flips is the floor every
    other perturbation is measured against.
    """
    return evidence


def _perturb_span_order(evidence: str) -> str:
    """Reverse the span blocks, renumbering so the text stays well-formed.

    The spans are an unordered set of citations -- each block names its own
    `path:line-line` -- so the order carries no information about whether the evidence
    entails the claim. A judge whose verdict depends on it is reading position.
    """
    blocks = _split_spans(evidence)
    if len(blocks) < 2:
        return evidence
    reversed_blocks = list(reversed(blocks))
    out: list[str] = []
    total = len(reversed_blocks)
    for i, (header, body) in enumerate(reversed_blocks, start=1):
        citation = header.split(": ", 1)[-1].rsplit(" ---", 1)[0]
        out.append(f"--- span {i} of {total}: {citation} ---\n{body}")
    return "\n".join(out)


def _perturb_trailing_whitespace(evidence: str) -> str:
    """Two spaces at the end of every line. Whitespace only, changing nothing.

    The direct test of the thing the README admitted and never measured. Source code
    is whitespace-sensitive in ways that make a model attend to it, and trailing
    spaces are the one whitespace change that alters no Python semantics at all.
    """
    return "\n".join(line + "  " for line in evidence.split("\n"))


def _perturb_separator_style(evidence: str) -> str:
    """`--- span 1 of 2: f.py:1-2 ---` becomes `=== span 1 of 2: f.py:1-2 ===`."""
    return re.sub(r"^--- (span .*?) ---$", r"=== \1 ===", evidence, flags=re.MULTILINE)


def _perturb_blank_lines(evidence: str) -> str:
    """A blank line between span blocks."""
    return re.sub(r"\n(--- span )", r"\n\n\1", evidence)


# Named and individually selectable, so a flip rate can be attributed to a specific
# transformation rather than to "perturbation" in general. A judge that is stable under
# reordering and unstable under whitespace is a different finding, with a different
# repair, from the reverse.
PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "identity": _perturb_identity,
    "span_order": _perturb_span_order,
    "trailing_whitespace": _perturb_trailing_whitespace,
    "separator_style": _perturb_separator_style,
    "blank_lines": _perturb_blank_lines,
}

DEFAULT_PERTURBATIONS = tuple(PERTURBATIONS)
BASELINE_PERTURBATION = "identity"


def _split_spans(evidence: str) -> list[tuple[str, str]]:
    """`render_evidence` output back into `(header, body)` pairs.

    Exactly reversible: `"\\n".join(f"{h}\\n{b}")` reconstructs the input. That matters
    more than it looks. A span's own bytes can end in a newline, and `render_evidence`
    also joins blocks with one, so a splitter that strips greedily turns a reordering
    perturbation into a reordering-plus-whitespace perturbation and the two flip rates
    stop being separable -- which is the entire point of naming them individually.
    """
    parts = re.split(r"^(--- span \d+ of \d+: .*? ---)$", evidence, flags=re.MULTILINE)
    blocks: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]
        if body.startswith("\n"):
            body = body[1:]
        # Every body but the last is followed by the newline `render_evidence` joined
        # with. That newline is the separator, not the span's content.
        if i + 2 < len(parts) - 1 and body.endswith("\n"):
            body = body[:-1]
        blocks.append((parts[i], body))
    return blocks


@dataclass(frozen=True)
class LabelFlip:
    """One claim whose label moved when the evidence formatting did."""

    assertion_id: int
    subject_qualname: str
    perturbation: str
    baseline_label: str
    perturbed_label: str

    def detail(self) -> str:
        return (
            f"[{self.perturbation}] #{self.assertion_id} {self.subject_qualname}: "
            f"{self.baseline_label} -> {self.perturbed_label}"
        )


@dataclass(frozen=True)
class FlipRate:
    """A flip rate with its interval, because a flip rate is also an estimate.

    5 flips in 147 is 0.034 with a Wilson interval of roughly [0.015, 0.077] -- the
    upper end is five times the lower, and a number quoted without it invites exactly
    the over-reading this whole module is being repaired for.
    """

    perturbation: str
    n: int
    flips: int
    correction: ClusterCorrection | None = None

    @property
    def rate(self) -> float | None:
        return self.flips / self.n if self.n else None

    @property
    def interval(self) -> tuple[float, float] | None:
        """Wilson, widened by the design effect when one was estimated."""
        if not self.n:
            return None
        rate = self.flips / self.n
        if self.correction is None:
            return wilson_interval(self.flips, self.n)
        effective = self.correction.effective_n
        return wilson_interval(rate * effective, effective)

    def format_rate(self) -> str:
        return format_interval(self.rate, self.interval)


@dataclass
class StabilityReport:
    """What the judge did when the evidence was reformatted and nothing else changed."""

    judge: str
    baseline: str
    # (assertion_id, perturbation, flipped) for every re-judgement performed. Kept
    # rather than reduced to counts because the pooled interval below needs to know
    # which trials came from the same claim, and counts have thrown that away.
    trials: list[tuple[int, str, bool]] = field(default_factory=list)
    flips: list[LabelFlip] = field(default_factory=list)

    @property
    def rates(self) -> list[FlipRate]:
        """One rate per perturbation, in the order they were run."""
        order: list[str] = []
        counts: dict[str, list[int]] = {}
        for _, perturbation, flipped in self.trials:
            if perturbation not in counts:
                counts[perturbation] = [0, 0]
                order.append(perturbation)
            counts[perturbation][0] += 1
            counts[perturbation][1] += 1 if flipped else 0
        return [FlipRate(p, counts[p][0], counts[p][1]) for p in order]

    @property
    def pooled(self) -> FlipRate | None:
        """Every non-baseline perturbation together, clustered by assertion.

        Pooling is the whole point -- "how much of this number is formatting" is one
        question -- and pooling naively would overstate its own precision, because one
        claim contributes several correlated trials. A claim the judge is genuinely
        torn about flips under everything; a claim it is sure about flips under
        nothing. So the pooled interval is corrected by the design effect estimated
        over the assertion, which is the cluster.
        """
        pooled = [t for t in self.trials if t[1] != self.baseline]
        if not pooled:
            return None
        flips = sum(1 for t in pooled if t[2])
        correction = cluster_correction(
            [(str(aid), flipped) for aid, _, flipped in pooled], key="assertion"
        )
        return FlipRate("pooled", len(pooled), flips, correction)

    def rate_for(self, perturbation: str) -> FlipRate | None:
        for rate in self.rates:
            if rate.perturbation == perturbation:
                return rate
        return None

    def summary(self) -> str:
        lines = [f"prompt stability  judge={self.judge}  baseline={self.baseline}"]
        pooled = self.pooled
        if pooled is not None:
            lines.append(
                f"  pooled flip rate {pooled.format_rate()} "
                f"({pooled.flips}/{pooled.n} re-judgements changed label)"
            )
        for rate in self.rates:
            marker = "  (control -- decoding nondeterminism, not sensitivity)" if (
                rate.perturbation == self.baseline
            ) else ""
            lines.append(
                f"    {rate.perturbation:<20} {rate.format_rate()} "
                f"({rate.flips}/{rate.n}){marker}"
            )
        return "\n".join(lines)

    def format_report(self) -> str:
        lines = [self.summary()]
        if self.flips:
            lines.append("")
            lines.append(f"-- {len(self.flips)} label flip(s) --")
            lines += [f"  {f.detail()}" for f in self.flips]
        return "\n".join(lines)


def measure_prompt_stability(
    conn: sqlite3.Connection,
    judge: Judge,
    repo_root: db.StrPath | None = None,
    *,
    perturbations: Sequence[str] = DEFAULT_PERTURBATIONS,
    baseline: str = BASELINE_PERTURBATION,
    kind: str | None = None,
    subject_qualname: str | None = None,
    limit: int | None = None,
) -> StabilityReport:
    """Re-judge a stored set under formatting-only perturbations; report the flip rate.

    Mirrors `adjudicate`'s signature and candidate set on purpose: the number this
    produces is only about the faithfulness score if it is measured over the same
    claims, with the same judge, through the same evidence window.

    **Never records.** There is no `record=` parameter and no way to reach one. This
    measures the instrument, and letting a run that deliberately malformed the prompt
    write verdicts would let an experiment about the judge reject live claims.

    Cost, since the caller is the one paying it: `len(perturbations) x n` judge calls.
    Over the 147 servable claims with the five defaults, 735.

    `perturbations` names keys of `PERTURBATIONS` and is ordered; `baseline` names the
    one every other is compared against and should stay `identity` unless the question
    has changed. Selecting a subset is supported because attribution matters -- a
    single "the judge is unstable" number cannot be acted on, and "it is stable under
    reordering and flips 8% of the time on trailing whitespace" can.
    """
    unknown = [p for p in (*perturbations, baseline) if p not in PERTURBATIONS]
    if unknown:
        raise ValueError(
            f"unknown perturbation(s) {unknown}; known: {sorted(PERTURBATIONS)}"
        )
    candidates = store.servable_assertions(
        conn, repo_root=repo_root, kind=kind, subject_qualname=subject_qualname
    )
    root = Path(str(repo_root if repo_root is not None else db.stored_repo_root(conn)))
    if limit is not None:
        candidates = candidates[:limit]

    report = StabilityReport(judge=judge.name, baseline=baseline)
    for assertion in candidates:
        if not assertion.spans:
            continue  # never sent to a judge, so it cannot flip
        rendered = render_evidence(root, assertion)
        baseline_label = judge.judge(
            claim=assertion.claim,
            evidence=PERTURBATIONS[baseline](rendered),
            subject=assertion.subject_qualname,
        ).label
        for name in perturbations:
            label = judge.judge(
                claim=assertion.claim,
                evidence=PERTURBATIONS[name](rendered),
                subject=assertion.subject_qualname,
            ).label
            flipped = label != baseline_label
            report.trials.append((assertion.id, name, flipped))
            if flipped:
                report.flips.append(
                    LabelFlip(
                        assertion_id=assertion.id,
                        subject_qualname=assertion.subject_qualname,
                        perturbation=name,
                        baseline_label=baseline_label,
                        perturbed_label=label,
                    )
                )
    return report


# --------------------------------------------------------------------------
# measuring the instrument: human calibration
# --------------------------------------------------------------------------
#
# The other missing measurement. The judge has never been scored against a human on the
# set the faithfulness number is computed over -- its calibration is 15/16 on a
# different set, pre-labelled by the same model family that wrote those claims, which
# is a measurement of agreement between two Qwen-adjacent processes.
#
# The obstacle is human time, so the obstacle this code removes is everything else:
# pick a stratified sample, write it out blind, read it back, and compute the
# agreement. Roughly thirty claims is an evening.


REVIEW_FIELDS = ("assertion_id", "subject", "claim", "citations", "evidence", "human")


@dataclass(frozen=True)
class ReviewExport:
    """Where the two files went and what ended up in them."""

    review_path: Path
    key_path: Path
    n: int
    per_label: dict[str, int]


@dataclass(frozen=True)
class CalibrationReport:
    """Judge against human on the claims the number is actually computed over.

    `supported` is the positive class, because the errors are not symmetric. A judge
    that calls a supported claim `not_supported` costs a claim that would have been
    served; a judge that calls an unsupported claim `supported` admits an unaccountable
    claim to the store, which is the failure this whole tier exists to prevent. So
    precision on `supported` is the number to read first.
    """

    n: int
    agreements: int
    confusion: dict[tuple[str, str], int]  # (judge, human) -> count
    skipped: int = 0

    @property
    def agreement(self) -> float | None:
        return self.agreements / self.n if self.n else None

    @property
    def agreement_interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.agreements, self.n) if self.n else None

    def _cell(self, judge_label: str, human_label: str) -> int:
        return self.confusion.get((judge_label, human_label), 0)

    @property
    def precision(self) -> float | None:
        """Of the claims this judge called supported, how many a human agreed with."""
        called = sum(v for (j, _), v in self.confusion.items() if j == LABEL_SUPPORTED)
        if not called:
            return None
        return self._cell(LABEL_SUPPORTED, LABEL_SUPPORTED) / called

    @property
    def recall(self) -> float | None:
        """Of the claims a human called supported, how many the judge also did."""
        actual = sum(v for (_, h), v in self.confusion.items() if h == LABEL_SUPPORTED)
        if not actual:
            return None
        return self._cell(LABEL_SUPPORTED, LABEL_SUPPORTED) / actual

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa: agreement above what two labellers get by guessing.

        Raw agreement on a set stratified to be half supported is inflated by chance
        alone -- two labellers flipping coins agree half the time -- and this sample is
        deliberately balanced, so the raw number is the most misleading form available.
        """
        if not self.n:
            return None
        labels = {label for pair in self.confusion for label in pair}
        expected = 0.0
        for label in labels:
            judge_marginal = sum(v for (j, _), v in self.confusion.items() if j == label)
            human_marginal = sum(v for (_, h), v in self.confusion.items() if h == label)
            expected += (judge_marginal / self.n) * (human_marginal / self.n)
        if expected >= 1.0:
            return None
        observed = self.agreements / self.n
        return (observed - expected) / (1.0 - expected)

    def summary(self) -> str:
        parts = [
            f"judge vs human  n={self.n}"
            + (f" (+{self.skipped} unreviewed)" if self.skipped else ""),
            f"  agreement {format_interval(self.agreement, self.agreement_interval)}",
            f"  precision(supported) "
            f"{'n/a' if self.precision is None else f'{self.precision:.2f}'}"
            f"  recall(supported) "
            f"{'n/a' if self.recall is None else f'{self.recall:.2f}'}"
            f"  kappa {'n/a' if self.kappa is None else f'{self.kappa:.2f}'}",
        ]
        for (judge_label, human_label), count in sorted(self.confusion.items()):
            if judge_label != human_label:
                parts.append(
                    f"    judge={judge_label} human={human_label}: {count}"
                )
        return "\n".join(parts)


def export_for_review(
    adjudications: Sequence[Adjudication],
    review_path: db.StrPath,
    *,
    n: int = 30,
    seed: int = 0,
    labels: Sequence[str] = (LABEL_SUPPORTED, LABEL_NOT_SUPPORTED),
    key_path: db.StrPath | None = None,
) -> ReviewExport:
    """Write a stratified, blind sample of adjudications for a human to label.

    **Blind on purpose, and this is the design decision in the function.** The review
    file carries the claim, its citations and exactly the evidence the judge was shown,
    and does NOT carry the judge's verdict. A reviewer who can see the label being
    checked is anchored by it, and the resulting agreement number measures deference
    rather than accuracy -- which would be a worse outcome than the current state,
    because it would look like the missing calibration had been done. The labels go to
    a separate key file that `score_review` reads and the reviewer does not open.

    Stratified and balanced across `labels` (by default `supported` and
    `not_supported`) because an unstratified 30 of a 0.54 store yields ~16 supported
    and ~14 not_supported by luck, and precision and recall each rest on one of those
    halves. `uncertain` is excluded by default: the target is the judge's error on the
    claims it decided, and there were two of them in 147. Pass `labels` to include it.

    Deterministic given `seed` -- a calibration sample that cannot be regenerated
    cannot be extended or re-reviewed, and a second reviewer on a different sample
    measures a different thing.
    """
    review = Path(str(review_path))
    key = Path(str(key_path)) if key_path is not None else review.with_suffix(".key.jsonl")
    strata: dict[str, list[Adjudication]] = {label: [] for label in labels}
    for adjudication in adjudications:
        bucket = strata.get(adjudication.judgement.label)
        if bucket is not None:
            bucket.append(adjudication)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    for bucket in strata.values():
        bucket.sort(key=lambda a: a.assertion_id)
        rng.shuffle(bucket)

    # Round-robin rather than n//len(labels) each: when one stratum is short the sample
    # fills up from the others instead of silently returning fewer than n, and the
    # actual composition is reported rather than assumed.
    chosen: list[Adjudication] = []
    while len(chosen) < n and any(strata.values()):
        for label in labels:
            if len(chosen) >= n:
                break
            if strata[label]:
                chosen.append(strata[label].pop())
    chosen.sort(key=lambda a: a.assertion_id)

    with review.open("w", encoding="utf-8") as handle:
        for adjudication in chosen:
            handle.write(
                json.dumps(
                    {
                        "assertion_id": adjudication.assertion_id,
                        "subject": adjudication.subject_qualname,
                        "claim": adjudication.claim,
                        "citations": list(adjudication.citations),
                        "evidence": adjudication.evidence,
                        # The reviewer fills this in with `supported`, `not_supported`
                        # or `uncertain`. Left empty rather than pre-filled: a default
                        # is a vote, and it would be cast for every row nobody read.
                        "human": "",
                    }
                )
                + "\n"
            )
    with key.open("w", encoding="utf-8") as handle:
        for adjudication in chosen:
            handle.write(
                json.dumps(
                    {
                        "assertion_id": adjudication.assertion_id,
                        "judge": adjudication.judgement.judge,
                        "label": adjudication.judgement.label,
                        "cause": adjudication.judgement.cause,
                    }
                )
                + "\n"
            )
    per_label: dict[str, int] = {}
    for adjudication in chosen:
        label = adjudication.judgement.label
        per_label[label] = per_label.get(label, 0) + 1
    return ReviewExport(review_path=review, key_path=key, n=len(chosen), per_label=per_label)


def score_review(
    review_path: db.StrPath,
    key: db.StrPath | Sequence[Adjudication] | Mapping[int, str],
) -> CalibrationReport:
    """Score a filled-in review file against the judge's labels.

    `key` is the key file `export_for_review` wrote, or the adjudications themselves,
    or a plain `{assertion_id: label}` -- whichever the caller still has. The review
    file never carried the judge's labels (see `export_for_review`), so one of these
    is required rather than optional.

    Rows with an empty `human` are counted as `skipped` and reported, not dropped: a
    reviewer who labelled 18 of 30 has measured 18, and an agreement rate quoted over
    a denominator of 18 while implying 30 is the same over-reading this module is being
    repaired for. A row with a human label nobody recognises raises, because silently
    discarding a typo biases the sample in whatever direction the typo was.
    """
    judge_labels = _key_labels(key)
    confusion: dict[tuple[str, str], int] = {}
    agreements = 0
    scored = 0
    skipped = 0
    for lineno, line in enumerate(
        Path(str(review_path)).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        assertion_id = int(row["assertion_id"])
        raw = str(row.get("human", "")).strip()
        if not raw:
            skipped += 1
            continue
        human = normalise_label(raw)
        if human is None:
            raise ValueError(
                f"{review_path}:{lineno}: unrecognised human label {raw!r} for "
                f"assertion {assertion_id}. Expected one of {list(LABELS)}. Not "
                f"skipped: a discarded row is a silent change to the sample."
            )
        if assertion_id not in judge_labels:
            raise ValueError(
                f"{review_path}:{lineno}: assertion {assertion_id} is not in the key, "
                f"so there is no judge label to compare it against."
            )
        judged = judge_labels[assertion_id]
        confusion[(judged, human)] = confusion.get((judged, human), 0) + 1
        scored += 1
        if judged == human:
            agreements += 1
    return CalibrationReport(
        n=scored, agreements=agreements, confusion=confusion, skipped=skipped
    )


def _key_labels(
    key: db.StrPath | Sequence[Adjudication] | Mapping[int, str],
) -> dict[int, str]:
    """`{assertion_id: judge_label}` from a key file, adjudications, or a mapping."""
    if isinstance(key, Mapping):
        return {int(k): str(v) for k, v in key.items()}
    if isinstance(key, (str, Path)):
        labels: dict[int, str] = {}
        for line in Path(str(key)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[int(row["assertion_id"])] = str(row["label"])
        return labels
    if isinstance(key, Iterable):
        return {a.assertion_id: a.judgement.label for a in key}
    raise TypeError(f"cannot read judge labels from {type(key).__name__}")
