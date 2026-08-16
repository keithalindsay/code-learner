"""Per-modality retrieval ablation against a hand-labelled gold set.

This exists so that fusion parameters are chosen by measurement rather than by
whichever configuration happened to look good on the query someone was staring at.
Tuning weights against a single demo query is how a retrieval system becomes
excellent at one question and mediocre at the rest.

Metrics are the standard three, computed the standard way:

  - **recall@k** -- fraction of a query's relevant symbols that appear in the top k.
    The headline for retrieval: if the right code is not in the candidate set, no
    amount of reranking downstream can recover it.
  - **MRR** -- 1/rank of the FIRST relevant hit, averaged. Captures "is a good
    answer at the top", which recall alone does not.
  - **hit@k** -- fraction of queries with at least one relevant hit in the top k.

## How small 16 queries actually is

The gold set is 16 hand-labelled queries. WITHDRAWN: the band this module used to
quote here -- one or two points -- sat below the instrument's own quantum, which
made it worse than offering no band at all, because it invited exactly the
fine-grained reading the set cannot support. The arithmetic it should have been:

  - `hit@5` is the mean of 16 booleans, so **one query flipping moves it 6.25
    points**. A difference of "one or two points" is not a small difference on this
    instrument; it is a difference this instrument cannot express.
  - 11 of the 16 queries carry exactly ONE relevant symbol, so `recall@5` has the
    same 6.25-point quantum on two thirds of the set.
  - The sampling sd is not the quantum. On n=16 the binomial sd of a rate is
    `sqrt(p(1-p)/16)`: **12.5 points at p=0.5, 10.8 at p=0.75, 7.5 at p=0.9** --
    the range the measured rows sit in. So the honest band is around **10 points,
    not one or two**, and `Scorecard.ci` computes it per row rather than leaving it
    to a sentence.

Two consequences, both load-bearing. Nothing under ~10 points separates two rows on
this set, so the sweep should be read for its SHAPE and not its ordering. And the
comparison worth making is paired: `Scorecard.delta_ci` resamples the same query
indices for both rows, which cancels the "some queries are just hard" variance that
dominates two independent intervals and is the only way a 6-point difference on 16
queries can be resolved at all.

## How many queries the instrument actually needs

Measured, not derived. The 16 queries were scored under 11 model-free configurations
against the real swarm-sync index, giving 55 real paired comparisons and therefore 55
real per-query difference vectors. `paired_sd`, `ci_half_width` and `required_n` below
are the sizing tools; these are the numbers they produced.

**The scaling law holds and its constant is measured.** Subsampling the real 16 down
to n=8 and resampling them up to n=512, the paired 95% half-width tracks `c/sqrt(n)`
with `c ~= 0.60` for MRR, stable to within 5% across that whole range. So the curve is
`half_width(n) ~= 0.60 / sqrt(n)`, and 0.60 is close to `1.96 * 0.322`, the median
per-query sd of a real paired difference on this corpus.

**sd is a property of the corpus, not of the effect being measured.** Across the 55
real pairs the correlation between `|mean diff|` and `sd(diff)` is 0.15, so a
comparison's noise cannot be predicted down from its effect being small. Size from sd.

**The current 16-query interval is not a 95% interval.** Under a true null built from
the real per-query noise, the percentile paired bootstrap rejects at **11.9%** at
n=16, against a nominal 5%. It reaches 7.9% at n=32, 6.0% at n=128 and 5.4% at n=256.
The per-query MRR difference is skewed (median skew -0.25) and mostly zeros (19-81% of
queries tie), which is exactly the shape the percentile bootstrap handles worst at
small n. The old set's "95% CI" was closer to an 88% CI, so the enlargement buys
calibration before it buys resolution.

Putting those together, at the median comparison (`sd(diff) = 0.322` for MRR,
`0.267` for nDCG@10):

    delta   half-width < delta   80% power (MRR)   80% power (nDCG@10)
    0.15            n =  18           n =  37            n =  25
    0.10            n =  40           n =  82            n =  56
    0.05            n = 160           n = 326            n = 224

**Read the middle column as the weak bar and the right two as the real one.** A CI
whose half-width merely equals the effect is a coin flip on whether any given run
excludes zero -- it is 50% power, not a resolution.

The whole table assumes new queries resemble the existing 16 in difficulty and
variance. A deliberately HARDER enlarged set violates that assumption in the
unfavourable direction: harder queries produce more ties at the bottom and more
rank churn, which raises `sd(diff)` and raises every n above. Treat these as floors,
re-run `required_n` on the enlarged set once it exists, and do not carry these
constants forward as if they were properties of the metric rather than of this corpus.

### What the enlarged set then showed

520 queries across swarm-sync, kalshi-bot and TradingAgents were scored the same way,
and two things came out of it.

**The extrapolation held.** Measured `sd(diff)` on 520 queries is 0.333 for MRR and
0.275 for nDCG@10, against the 0.322 and 0.267 predicted from the 16. Sizing from a
small pilot worked for these queries, which is the assumption above surviving its
first real test.

**The numbers above are EFFECTIVE n, and the multiplier is about five.** With three
repos to measure across, the design effect of repo clustering is 5.4 on nDCG@10 -- so
520 queries carry the evidence of roughly 90. `required_n` takes `deff` for this
reason. And because effective n saturates at `repos / ICC`, three repos cannot exceed
about 116 effective queries however much gold is written against them: **repos buy
power, extra queries within a repo buy progressively less.** `design_effect` carries
the tradeoff table.

**The pooled row hides reversals, and there is one.** `prefer_implementation` is worth
+0.062 nDCG@10 on kalshi-bot with an interval excluding zero, -0.071 on TradingAgents,
and +0.018 on swarm-sync. Pooled it is +0.029 and not significant -- the average of a
win and a loss, reported as an absence of effect. Read `format_delta_report`, not the
pooled row.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import db
from ..assertions.search_index import assertion_search_structures_present
from ..index.embed import Embedder
from ..retrieve.dense import search_dense
from ..retrieve.fuse import reciprocal_rank_fusion
from ..retrieve.graph import expand
from ..retrieve.lexical import Hit, search_lexical
from ..retrieve.mixed import search_candidates
from ..retrieve.rerank import Reranker
from ..retrieve.search import search
from ..retrieve.types import AssertionCandidate, SourceCandidate
from .goldset import (
    GOLD_DIR,
    GoldQuery,
    GoldSchemaError,
    load_gold_set,
    validate_against_index,
)

#: Bootstrap settings, PRINTED by `format_table` rather than only living here. An
#: interval whose resample count and seed are not next to the number is not something
#: the reader can reproduce, and this table's whole job is to be checkable.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20250801


@dataclass
class QueryResult:
    query: str
    relevant: list[str]
    retrieved: list[str]
    first_relevant_rank: int | None = None
    #: Provenance, carried from the gold file so a result can be routed to the right
    #: stratum. Empty on results built by hand in tests, which have no provenance.
    source: str = ""
    repo: str = ""
    query_id: str = ""

    def recall_at(self, k: int) -> float:
        if not self.relevant:
            return 0.0
        found = sum(1 for r in self.relevant if r in self.retrieved[:k])
        return found / len(self.relevant)

    def hit_at(self, k: int) -> bool:
        return any(r in self.retrieved[:k] for r in self.relevant)

    @property
    def is_single_relevant(self) -> bool:
        return len(self.relevant) == 1

    def ndcg_at(self, k: int) -> float:
        """Binary-relevance nDCG@k -- the whole ranking of relevant symbols, discounted.

        MRR reads ONE position, the first relevant hit, and is blind to everything
        after it: a configuration that puts the one remaining relevant symbol at rank 2
        and one that loses it entirely score identically. nDCG credits every relevant
        symbol at `1/log2(rank+1)` and normalises by the best achievable arrangement, so
        it moves when the tail moves.

        Relevance is BINARY because the gold labels are binary -- a symbol is listed or
        it is not. Graded relevance would let the ideal ranking distinguish "the
        function that answers this" from "the caller you would read next", and the
        schema in `goldset` has room for it, but inventing grades from an unordered
        list would be reading structure into labels that do not carry any.
        """
        if not self.relevant or k <= 0:
            return 0.0
        relevant = set(self.relevant)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, qualname in enumerate(self.retrieved[:k])
            if qualname in relevant
        )
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / ideal if ideal else 0.0

    def ap_at(self, k: int) -> float:
        """Average precision@k -- precision recomputed at every relevant hit.

        Like nDCG it reads the whole cut rather than the first hit, but it discounts
        linearly in rank rather than logarithmically, so it punishes a relevant symbol
        at rank 9 harder than nDCG does. The denominator is `min(len(relevant), k)` so
        a query with more relevant symbols than `k` can still reach 1.0; dividing by
        `len(relevant)` would cap such a query below 1 no matter what any retriever did.
        """
        if not self.relevant or k <= 0:
            return 0.0
        relevant = set(self.relevant)
        hits = 0
        total = 0.0
        for rank, qualname in enumerate(self.retrieved[:k], start=1):
            if qualname in relevant:
                hits += 1
                total += hits / rank
        return total / min(len(relevant), k)


#: The label a row carries when it averages every stratum together. Spelled out in the
#: row itself rather than left to the reader, because a pooled mean over sources with
#: different biases moves when the MIX changes and not only when the retriever does.
POOLED = "POOLED"


@dataclass
class Scorecard:
    name: str
    results: list[QueryResult] = field(default_factory=list)
    #: Which slice of the gold set this row covers: `POOLED`, `source=...`, `repo=...`.
    scope: str = POOLED
    #: How many tier-2 candidates the configuration returned across every query.
    #: A COUNT, deliberately, and not a metric: this gold set labels symbols, so
    #: there is nothing here to score a claim against. Phase 2.5's semantic gold set
    #: is what turns this into a measurement; until then it says only that the
    #: modality fired, which is the difference between "no claims matched" and "the
    #: modality never ran".
    assertions_returned: int = 0

    def recall_at(self, k: int) -> float:
        return _mean([r.recall_at(k) for r in self.results])

    def hit_at(self, k: int) -> float:
        return _mean([1.0 if r.hit_at(k) else 0.0 for r in self.results])

    def ndcg_at(self, k: int = 10) -> float:
        return _mean([r.ndcg_at(k) for r in self.results])

    def map_at(self, k: int = 10) -> float:
        return _mean([r.ap_at(k) for r in self.results])

    @property
    def mrr(self) -> float:
        return _mean(
            [
                1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0
                for r in self.results
            ]
        )

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def single_relevant(self) -> int:
        """How many of these queries carry exactly one relevant symbol.

        The diagnostic that explains a table rather than decorating it. On a query with
        one relevant symbol `recall@k` can only be 0 or 1, which IS `hit@k`; when this
        count approaches `n` the two recall columns and the hit column are one
        measurement printed three times, and `ndcg`/`map` collapse toward `mrr` too.
        """
        return sum(1 for r in self.results if r.is_single_relevant)

    def per_query(self, metric: str, k: int = 5) -> list[float]:
        """One value per query, so the bootstrap has something to resample.

        The unit of resampling is a QUERY, not a retrieved symbol: two symbols
        relevant to the same query are not two independent trials of the retriever.
        """
        if metric == "recall":
            return [r.recall_at(k) for r in self.results]
        if metric == "hit":
            return [1.0 if r.hit_at(k) else 0.0 for r in self.results]
        if metric == "mrr":
            return [
                1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0
                for r in self.results
            ]
        if metric == "ndcg":
            return [r.ndcg_at(k) for r in self.results]
        if metric == "map":
            return [r.ap_at(k) for r in self.results]
        raise ValueError(f"unknown metric {metric!r}")

    def cluster_labels(self) -> list[str]:
        return [r.repo for r in self.results]

    def ci(
        self,
        metric: str = "hit",
        k: int = 5,
        resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED,
        cluster: bool = False,
    ) -> tuple[float, float]:
        """95% bootstrap interval over the queries. See the module docstring.

        `cluster=True` resamples REPOS rather than queries; see `_resample_clusters`.
        """
        values = self.per_query(metric, k)
        if not values:
            return (0.0, 0.0)
        draws = self._draws(len(values), resamples, seed, cluster)
        return _percentile_ci([_mean([values[i] for i in idx]) for idx in draws])

    def delta_ci(
        self,
        other: Scorecard,
        metric: str = "hit",
        k: int = 5,
        resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED,
        cluster: bool = False,
    ) -> tuple[float, float]:
        """PAIRED interval for `self - other`, resampling the same queries for both.

        This is the comparison the table is for, and it is not the same thing as
        looking at whether two `ci()` intervals overlap. Both rows answer the same
        queries, so most of the variance in either interval is "this query is hard",
        which cancels in the difference. Two overlapping marginal intervals routinely
        sit either side of a difference whose paired interval excludes zero.

        `cluster=True` makes the resampling unit the repo rather than the query.
        """
        mine = self.per_query(metric, k)
        theirs = other.per_query(metric, k)
        if not mine or len(mine) != len(theirs):
            return (0.0, 0.0)
        diffs = [a - b for a, b in zip(mine, theirs, strict=True)]
        draws = self._draws(len(diffs), resamples, seed, cluster)
        return _percentile_ci([_mean([diffs[i] for i in idx]) for idx in draws])

    def _draws(self, n: int, resamples: int, seed: int, cluster: bool) -> list[list[int]]:
        if not cluster:
            return _resample_indices(n, resamples, seed)
        return _resample_clusters(self.cluster_labels(), resamples, seed)

    def row(self) -> str:
        lo, hi = self.ci("hit", 5)
        return (
            f"{self.name:<30} {self.scope:<18} {self.n:>4} "
            f"{self.recall_at(5):>8.3f} {self.recall_at(10):>9.3f} "
            f"{self.hit_at(5):>7.3f} [{lo:>5.3f},{hi:>5.3f}] "
            f"{self.mrr:>7.3f} {self.ndcg_at(10):>8.3f} {self.map_at(10):>7.3f}"
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resample_indices(n: int, resamples: int, seed: int) -> list[list[int]]:
    """Bootstrap index draws, a pure function of (n, resamples, seed).

    Deliberately not a method: every scorecard over the same gold set must get the
    IDENTICAL draws, because that is what makes `delta_ci` paired. A per-card RNG
    would silently un-pair the comparison while still printing intervals.
    """
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def _resample_clusters(labels: Sequence[str], resamples: int, seed: int) -> list[list[int]]:
    """Bootstrap draws whose unit is the CLUSTER (the repo), not the query.

    ## Why pooling across repos has to cluster

    The per-query bootstrap assumes queries are exchangeable draws from the population
    we want to generalise to. Across repos they are not. Every query labelled against
    one repo shares that repo's index, its naming conventions, its docstring density,
    its call-graph shape, and the one person who labelled it. If graph expansion is
    weak on a repo whose modules barely import each other, it is weak on ALL forty of
    that repo's queries at once -- their errors move together, so forty queries carry
    much less than forty queries' worth of independent evidence.

    Resampling queries independently would treat that shared error as forty separate
    confirmations and return an interval too narrow by roughly `sqrt(1 + (m-1) * ICC)`,
    where `m` is the queries per repo and `ICC` the intra-repo correlation. At 40
    queries per repo, an ICC as small as 0.05 nearly doubles the true standard error.
    That understatement points the wrong way -- it manufactures significance -- which is
    the failure mode this whole exercise exists to remove.

    The claim we want to publish is "this configuration is better", not "this
    configuration is better on swarm-sync". That generalisation is over REPOS, so the
    repo is the sampling unit, and this draws whole repos with replacement.

    ## Its limits, which are severe at small repo counts

    A cluster bootstrap has as many effective observations as there are CLUSTERS. With
    two or three repos it is resampling two or three things and its interval is
    unstable however many queries sit inside them -- `format_delta_report` says so
    rather than printing it silently. Below about five repos, per-repo rows are the
    honest reporting and the pooled clustered interval is a placeholder for when there
    are enough repos to earn it. `design_effect` measures the ICC once there is more
    than one repo to measure it from.

    Draws remain a pure function of `(labels, resamples, seed)`, so two scorecards over
    the same gold set get IDENTICAL draws and `delta_ci` stays paired.
    """
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    keys = sorted(groups)
    if len(keys) <= 1:
        # One cluster: clustering is a no-op that would otherwise resample the SAME
        # single group every time and return a zero-width interval reading as certainty.
        return _resample_indices(len(labels), resamples, seed)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    draws: list[list[int]] = []
    for _ in range(resamples):
        picked: list[int] = []
        for _ in range(len(keys)):
            picked.extend(groups[keys[rng.randrange(len(keys))]])
        draws.append(picked)
    return draws


def design_effect(values: Sequence[float], labels: Sequence[str]) -> float:
    """`1 + (mean cluster size - 1) * ICC` -- how much a cluster costs in effective n.

    Multiply `required_n` by this to size a multi-repo gold set. It returns 1.0 when
    there is nothing to cluster (one repo, or no within-repo correlation), and cannot
    return less than 1.0: a negative ICC estimate means the data show no clustering,
    not that clustering buys precision.

    MEASURED, once there were repos to measure across. On 520 queries spanning
    swarm-sync, kalshi-bot and TradingAgents, the per-query nDCG@10 difference between
    `lexical only` and `hybrid + prefer_impl` has a design effect of **5.4** (5.8 for
    MRR) -- so those 520 queries carry the evidence of about 90 independent ones. The
    ICC behind it is small, about 0.026, and the design effect is large anyway because
    the clusters are: `1 + (173 - 1) * 0.026`.

    That arithmetic has a consequence worth more than the number. Effective n is
    `m * q / (1 + (q - 1) * ICC)` for `m` repos of `q` queries, which SATURATES at
    `m / ICC` however many queries are labelled -- three repos cannot exceed about 116
    effective queries no matter how much gold is written against them. Repos buy power;
    queries within a repo buy progressively less. To reach the 56 effective queries
    that `delta=0.10` needs at 80% power on nDCG@10:

        2 repos x 98 queries = 196      6 repos x 12 = 72
        3 repos x 36 queries = 108     10 repos x  7 = 70

    An ICC estimated from three clusters is itself noisy, so treat the shape -- add
    repos before adding queries -- as the finding, and the exact multiplier as
    provisional until there are more repos to re-estimate it from.
    """
    groups: dict[str, list[float]] = {}
    for value, label in zip(values, labels, strict=True):
        groups.setdefault(label, []).append(value)
    sized = [g for g in groups.values() if len(g) > 1]
    if len(groups) < 2 or not sized:
        return 1.0
    grand = _mean(list(values))
    between_df = len(sized) - 1
    within_df = sum(len(g) for g in sized) - len(sized)
    if between_df <= 0 or within_df <= 0:
        return 1.0
    ms_between = sum(len(g) * (_mean(g) - grand) ** 2 for g in sized) / between_df
    ms_within = sum(sum((v - _mean(g)) ** 2 for v in g) for g in sized) / within_df
    mean_size = sum(len(g) for g in sized) / len(sized)
    denom = ms_between + (mean_size - 1) * ms_within
    if denom <= 0:
        return 1.0
    icc = (ms_between - ms_within) / denom
    return max(1.0, 1.0 + (mean_size - 1) * icc)


# --------------------------------------------------------------------------------
# Sizing: how many queries the next gold set needs
# --------------------------------------------------------------------------------

#: z at 97.5% and at 80% power. Named so the arithmetic below is readable rather than
#: two magic constants that look like they were tuned.
_Z_ALPHA = 1.959964
_Z_POWER = 0.841621


def paired_sd(a: Scorecard, b: Scorecard, metric: str = "mrr", k: int = 5) -> float:
    """Sample sd of the per-query difference `a - b`. The input to every sizing answer.

    Measured on this corpus across 55 real config pairs: median 0.322 for MRR, 0.267
    for nDCG@10. It is a property of the CORPUS rather than of the effect -- the
    correlation between `|mean diff|` and this sd across those pairs is 0.15 -- so a
    comparison whose effect looks small cannot be assumed to be quiet.
    """
    mine = a.per_query(metric, k)
    theirs = b.per_query(metric, k)
    if len(mine) != len(theirs) or len(mine) < 2:
        return 0.0
    diffs = [x - y for x, y in zip(mine, theirs, strict=True)]
    mean = _mean(diffs)
    return math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))


def ci_half_width(sd: float, n: int) -> float:
    """The measured curve: `1.96 * sd / sqrt(n)`.

    Not an assumption -- checked against the real data by subsampling the 16 queries
    down to n=8 and resampling them up to n=512, where the empirical half-width tracked
    `0.60/sqrt(n)` to within 5% and `1.96 * 0.322 = 0.63`.
    """
    if n <= 0:
        return float("inf")
    return _Z_ALPHA * sd / math.sqrt(n)


def required_n(
    sd: float, delta: float, power: float = 0.80, deff: float = 1.0
) -> int:
    """Queries needed to resolve a true difference of `delta`, given per-query `sd`.

    `power=0.50` is the weak bar -- "the half-width is smaller than the effect" -- and
    is a coin flip on whether any single run's interval excludes zero. `power=0.80` is
    the bar worth sizing to. On this corpus, at the median comparison:

        delta   50% (MRR)  80% (MRR)  80% (nDCG@10)
        0.15         18         37         25
        0.10         40         82         56
        0.05        160        326        224

    `deff` is the design effect from `design_effect`, and it is NOT a refinement: on
    the three-repo corpus it measured 5.4, so a pooled multi-repo answer that ignores
    it under-orders gold fivefold. Those columns are the EFFECTIVE n; multiply by
    `deff` for the number of queries actually to label.

    Also observe the calibration floor described in `format_delta_report`: below
    n~128 the percentile bootstrap's real false-positive rate exceeds its nominal 5%,
    so an n large enough for RESOLUTION can still be too small for the interval to mean
    what it says.
    """
    if delta <= 0 or sd <= 0:
        return 0
    z_power = _Z_POWER if power >= 0.80 else 0.0
    return math.ceil(max(1.0, deff) * ((_Z_ALPHA + z_power) * sd / delta) ** 2)


def power_curve(
    sd: float,
    deltas: Sequence[float] = (0.05, 0.10, 0.15),
    sizes: Sequence[int] = (16, 32, 64, 128, 256, 512),
) -> str:
    """The sizing table, printed with the sd it was computed from.

    A required-n quoted without its sd is not checkable, and the sd is the only part of
    it that belongs to this corpus rather than to arithmetic.
    """
    lines = [
        f"Sizing at per-query sd(diff) = {sd:.3f}",
        "  half-width by n:  " + "  ".join(f"n={n}:{ci_half_width(sd, n):.3f}" for n in sizes),
        f"  {'delta':>7} {'n @50% power':>13} {'n @80% power':>13}",
    ]
    for delta in deltas:
        lines.append(
            f"  {delta:>7.3f} {required_n(sd, delta, 0.50):>13} "
            f"{required_n(sd, delta, 0.80):>13}"
        )
    lines.append(
        "  Assumes new queries match the existing ones in difficulty and variance. A "
        "HARDER set raises sd and every n above."
    )
    return "\n".join(lines)


def _percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    lo = ordered[min(len(ordered) - 1, int((alpha / 2) * len(ordered)))]
    hi = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return (lo, hi)


def load_gold(name: str = "swarm_sync") -> dict:
    """The raw JSON of one gold file. Kept for callers that want the file as written.

    `load_gold_set` is what scoring uses: it carries provenance, validates the schema,
    and can pool several files. This returns the unparsed dict and validates nothing.
    """
    return json.loads((GOLD_DIR / f"{name}.json").read_text())


def _score(
    name: str,
    per_query: Sequence[tuple[GoldQuery, Sequence[Hit | SourceCandidate]]],
    scope: str = POOLED,
) -> Scorecard:
    card = Scorecard(name=name, scope=scope)
    for spec, hits in per_query:
        retrieved = [h.qualname for h in hits]
        result = QueryResult(
            query=spec.query,
            relevant=list(spec.relevant),
            retrieved=retrieved,
            source=spec.source,
            repo=spec.repo,
            query_id=spec.query_id,
        )
        for rank, qualname in enumerate(retrieved, start=1):
            if qualname in spec.relevant:
                result.first_relevant_rank = rank
                break
        card.results.append(result)
    return card


def stratify(card: Scorecard) -> list[Scorecard]:
    """Split one pooled row into the rows it is actually an average of.

    Returns the pooled row first, then one row per source and one per repo. A source
    or repo with only one value present is not re-emitted, because a stratum identical
    to the pool is the pool with a more specific-looking label on it, and that reads as
    corroboration it is not.
    """
    rows = [card]
    for attr, prefix in (("source", "source"), ("repo", "repo")):
        values = sorted({getattr(r, attr) for r in card.results if getattr(r, attr)})
        if len(values) < 2:
            continue
        for value in values:
            picked = [r for r in card.results if getattr(r, attr) == value]
            rows.append(
                Scorecard(name=card.name, results=picked, scope=f"{prefix}={value}")
            )
    return rows


def stratified_cards(cards: Sequence[Scorecard]) -> list[Scorecard]:
    """`stratify` over a whole table, keeping configurations grouped together."""
    return [row for card in cards for row in stratify(card)]


def run_ablation(
    conn: sqlite3.Connection,
    embedder: Embedder | None = None,
    gold_name: str | Sequence[str] = "swarm_sync",
    k: int = 10,
    reranker: Reranker | None = None,
    validate: bool = True,
) -> list[Scorecard]:
    """Score every modality alone and every combination that matters.

    `reranker` is optional and adds three rows when supplied. It is a separate
    argument rather than always-on because the reranker costs a forward pass per
    candidate -- roughly 640 of them per row -- and the modality rows must stay
    runnable in seconds on a machine with no GPU.

    `gold_name` accepts several names, which pools their queries into one run. The
    rows returned are pooled; pass them through `stratified_cards` to get the per-source
    and per-repo rows, which is what you should read when the sources differ in bias.

    `validate` checks every gold qualname against the index BEFORE scoring and raises
    `GoldIndexMismatch` if any is absent. This function used to score whatever it was
    handed: a gold set that did not match the index returned a full table of 0.000 with
    `[0.000, 0.000]` intervals -- output shaped exactly like a finished measurement and
    containing none. No retriever can rank a symbol that is not in the index, so that
    was never a result about retrieval, and it now fails instead of printing. Turn it
    off only to score a deliberately partial gold set, and know that every unmatched
    name silently costs the configuration a query it could not have won.
    """
    gold = load_gold_set(gold_name)
    if validate:
        validate_against_index(gold, conn)
    return _run_configs(conn, gold.queries, embedder, k, reranker)


def run_ablation_multi(
    conns: dict[str, sqlite3.Connection],
    embedder: Embedder | None = None,
    gold_name: str | Sequence[str] = "swarm_sync",
    k: int = 10,
    reranker: Reranker | None = None,
    validate: bool = True,
) -> list[Scorecard]:
    """Score a gold set that spans repos, each repo against ITS OWN index.

    `run_ablation` takes one connection, which is all a single-repo gold set needs and
    is silently wrong for a multi-repo one: every query from another repo would name
    symbols this index does not contain. That is precisely the failure `validate`
    now catches, so the multi-repo case needs a connection per repo rather than an
    exemption from the check.

    Returns pooled rows whose `results` carry each query's repo, so `stratified_cards`
    can split them and `delta_ci(cluster=True)` can resample repos. `conns` must cover
    every repo the gold set names -- a missing one raises rather than quietly dropping
    that repo's queries, which would change `n` and every mean computed from it.
    """
    gold = load_gold_set(gold_name)
    missing = sorted(set(gold.repos()) - set(conns))
    if missing:
        raise GoldSchemaError(
            f"no index supplied for repo(s) {missing}. Scoring them against another "
            "repo's index would score zeros; dropping them would change n silently."
        )
    subsets = {repo: gold.subset(repo=repo) for repo in gold.repos()}
    # EVERY repo is validated before ANY repo is scored. Validating lazily would spend
    # a full GPU pass on the first repo before discovering the third one's gold is
    # stale, and the point of the check is to fail before the expensive part.
    if validate:
        for repo, subset in subsets.items():
            validate_against_index(subset, conns[repo])

    merged: dict[str, Scorecard] = {}
    order: list[str] = []
    for repo, subset in subsets.items():
        for card in _run_configs(conns[repo], subset.queries, embedder, k, reranker):
            if card.name not in merged:
                merged[card.name] = Scorecard(name=card.name)
                order.append(card.name)
            merged[card.name].results.extend(card.results)
    return [merged[name] for name in order]


def _semantic_card(
    conn: sqlite3.Connection,
    queries: Sequence[GoldQuery],
    embedder: Embedder | None,
    k: int,
) -> Scorecard | None:
    """The tier-2 row: plumbing for Phase 2.5, and NOT a lift measurement.

    Read this row for one thing only -- whether semantic retrieval ran and how many
    claims it returned. Its metrics are NOT comparable with the rows above it, and
    the reason is structural rather than a caveat that might be lifted by running it
    on more data. The gold set labels SYMBOLS. A claim is not a symbol, so a claim
    occupying a slot can only ever cost this row recall against a symbol-labelled
    answer key, however good the claim is. Scoring it as if the two were the same
    thing is precisely the coercion this row exists not to perform: only the source
    candidates are scored, and the claims are counted beside them.

    What would make it a measurement is Phase 2.5's semantic gold set -- questions
    whose answer IS a claim, with hard negatives, over at least five repositories.
    Until that exists, no number here supports a statement about whether the
    semantic layer helps.
    """
    root = db.stored_repo_root(conn)
    if root is None or not assertion_search_structures_present(conn):
        # An index built before schema v7, or one not bound to a repository, cannot
        # verify a citation. Skipped rather than scored as zeros, which would look
        # like a modality that ran and found nothing.
        return None
    per_query: list[tuple[GoldQuery, Sequence[Hit | SourceCandidate]]] = []
    returned = 0
    for spec in queries:
        result = search_candidates(
            conn,
            Path(root),
            spec.query,
            k=k,
            embedder=embedder,
            use_dense=embedder is not None,
        )
        returned += sum(
            1 for c in result.candidates if isinstance(c, AssertionCandidate)
        )
        per_query.append(
            (spec, [c for c in result.candidates if isinstance(c, SourceCandidate)])
        )
    card = _score("hybrid + assertions (plumbing)", per_query)
    card.assertions_returned = returned
    return card


def _run_configs(
    conn: sqlite3.Connection,
    queries: Sequence[GoldQuery],
    embedder: Embedder | None,
    k: int,
    reranker: Reranker | None,
) -> list[Scorecard]:
    """Every configuration, scored against one index. The body `run_ablation` used to be."""
    cards: list[Scorecard] = []

    # --- single modalities -------------------------------------------------
    cards.append(
        _score("lexical only", [(q, search_lexical(conn, q.query, k=k)) for q in queries])
    )
    if embedder is not None:
        cards.append(
            _score("dense only", [(q, search_dense(conn, q.query, embedder, k=k)) for q in queries])
        )

    # Graph alone is meaningless -- it has no query representation and must be
    # seeded. Scored here seeded by dense, to isolate what expansion ADDS.
    if embedder is not None:
        graph_only = []
        for q in queries:
            seeds = search_dense(conn, q.query, embedder, k=5)
            graph_only.append((q, expand(conn, seeds, k=k)))
        cards.append(_score("graph only (dense-seeded)", graph_only))

    # --- combinations ------------------------------------------------------
    if embedder is not None:
        cards.append(
            _score(
                "lexical + dense",
                [
                    (
                        q,
                        reciprocal_rank_fusion(
                            {
                                "lexical": search_lexical(conn, q.query, k=k * 4),
                                "dense": search_dense(conn, q.query, embedder, k=k * 4),
                            },
                            k=k,
                        ),
                    )
                    for q in queries
                ],
            )
        )

    # Isolates whether `prefer_implementation` is rescuing the graph modality or is
    # simply good on its own. Without this pairing, "hybrid + prefer_impl wins"
    # cannot distinguish the two, and the graph modality would get credit it may
    # not have earned.
    if embedder is not None:
        cards.append(
            _score(
                "lexical + dense + pref_impl",
                [
                    (
                        q,
                        reciprocal_rank_fusion(
                            {
                                "lexical": search_lexical(conn, q.query, k=k * 4),
                                "dense": search_dense(conn, q.query, embedder, k=k * 4),
                            },
                            k=k,
                            prefer_implementation=True,
                        ),
                    )
                    for q in queries
                ],
            )
        )

    # Both flags are passed EXPLICITLY. `prefer_implementation` now defaults to True
    # in `search()` because this ablation showed it should, and a row that inherits
    # the default it was used to justify measures nothing.
    cards.append(
        _score(
            "hybrid, no pref_impl",
            [
                (
                    q,
                    search(
                        conn, q.query, k=k, embedder=embedder,
                        prefer_implementation=False,
                    ).hits,
                )
                for q in queries
            ],
        )
    )
    cards.append(
        _score(
            "hybrid + prefer_impl (default)",
            [
                (
                    q,
                    search(
                        conn, q.query, k=k, embedder=embedder,
                        prefer_implementation=True,
                    ).hits,
                )
                for q in queries
            ],
        )
    )

    # --- reranking ---------------------------------------------------------
    #
    # Three rows, each PAIRED with a row above that differs by exactly one thing, so
    # the lift is attributable. The claim under test is specific: graph expansion
    # buys recall and costs MRR because it has no query representation, and a
    # cross-encoder -- which does -- should be able to keep the recall and give the
    # MRR back.
    #
    #   hybrid + rerank             vs  hybrid + prefer_impl   -> does it undo the
    #                                                             graph's dilution?
    #   lex+dense pref_impl+rerank  vs  lexical + dense + pref_impl
    #                                                          -> or does it just
    #                                                             help everywhere,
    #                                                             graph or no graph?
    #   hybrid + rerank, no p_i     vs  hybrid, no pref_impl   -> does a model that
    #                                                             reads the query
    #                                                             make the test
    #                                                             demotion redundant?
    #
    # Without the second row, "reranking fixed the graph modality" and "reranking is
    # good" are indistinguishable. Without the third, the largest lever measured so
    # far (test demotion) never gets asked whether it is still needed.
    #
    # NO NUMBERS ARE RECORDED HERE, and their absence is the point.
    #
    # This comment used to carry four measured reranking rows under the heading
    # "MEASURED", plus three conclusions drawn from them -- including an MRR swing
    # quoted to three decimals. Those figures were RETRACTED in the README (the live
    # swing is smaller), and two of the three conclusions did not survive the re-run,
    # but the retraction never reached this file. So the module that PRODUCES the
    # table went on shipping the withdrawn version of it, in the most authoritative
    # place a reader would look, for as long as nobody re-read the comment.
    #
    # A source comment cannot be re-measured when the reranker, the index, or the
    # repo changes, so it will always drift toward exactly that failure. Reranked
    # rows need a GPU pass; run them and read the output:
    #
    #     print(format_table(run_ablation(conn, embedder, reranker=reranker)))
    #
    # and publish that output stamped `repo@sha`, beside the `hit@5` intervals the
    # table now prints -- which, at 16 queries, are wide enough that a four-row
    # comparison quoted to three decimals was never the right shape for this result.
    # `Scorecard.delta_ci` is the paired test for the three questions above.
    if reranker is not None:
        rerank_rows = [
            ("hybrid + rerank", {"use_graph": True, "prefer_implementation": True}),
            ("lex+dense pref_impl+rerank", {"use_graph": False, "prefer_implementation": True}),
            ("hybrid + rerank no pref_impl", {"use_graph": True, "prefer_implementation": False}),
        ]
        for name, kwargs in rerank_rows:
            cards.append(
                _score(
                    name,
                    [
                        (
                            q,
                            search(
                                conn, q.query, k=k, embedder=embedder,
                                reranker=reranker, **kwargs,
                            ).hits,
                        )
                        for q in queries
                    ],
                )
            )

    # Graph weight sweep, both with the test demotion on, so the comparison is
    # against the best non-graph configuration rather than a strawman.
    for weight in (0.3, 1.0, 1.5):
        cards.append(
            _score(
                f"hybrid pref_impl gw={weight}",
                [
                    (q, _weighted_hybrid(conn, q.query, embedder, k, weight))
                    for q in queries
                ],
            )
        )

    # Last, and skipped entirely on an index that cannot serve claims. Its metrics
    # are not comparable with the rows above -- see `_semantic_card`.
    semantic = _semantic_card(conn, queries, embedder, k)
    if semantic is not None:
        cards.append(semantic)
    return cards


def _weighted_hybrid(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder | None,
    k: int,
    graph_weight: float,
) -> list[Hit]:
    """Hybrid retrieval with an explicit graph weight, for the sweep.

    NOTE: this reimplements the pipeline rather than calling `search()`, because
    `search()` takes no weights argument. It seeds graph expansion from the top 5 of
    each modality instead of `search`'s interleaved merge, so sweep rows are
    comparable with EACH OTHER but differ slightly from the `search()` rows above
    (observed: MRR 0.463 here vs 0.453 there at the same weight). Read the sweep for
    its shape -- monotonic decline above 0.3 -- not for absolute values.
    """
    depth = k * 4
    per_modality: dict[str, list[Hit]] = {
        "lexical": search_lexical(conn, query, k=depth),
    }
    if embedder is not None:
        per_modality["dense"] = search_dense(conn, query, embedder, k=depth)
    seeds: list[Hit] = []
    seen: set[int] = set()
    for hits in per_modality.values():
        for hit in hits[:5]:
            if hit.symbol_id not in seen:
                seen.add(hit.symbol_id)
                seeds.append(hit)
    if seeds:
        per_modality["graph"] = expand(conn, seeds, k=depth)
    return reciprocal_rank_fusion(
        per_modality,
        k=k,
        weights={"lexical": 1.0, "dense": 1.0, "graph": graph_weight},
        prefer_implementation=True,
    )


#: Below this many queries the percentile bootstrap's real false-positive rate exceeds
#: its nominal 5% (measured on this corpus: 11.9% at n=16, 7.9% at n=32, 6.0% at n=128).
#: A table under this floor may be readable for its shape and is not a 95% interval.
CALIBRATION_FLOOR = 128


def _diagnostics(cards: Sequence[Scorecard]) -> list[str]:
    """The facts that decide how the numbers above may be read."""
    card = cards[0]
    n = card.n
    if not n:
        return []
    single = card.single_relevant
    sources = sorted({r.source for r in card.results if r.source})
    repos = sorted({r.repo for r in card.results if r.repo})
    lines = [
        f"n = {n} queries. hit@5 moves in steps of {1 / n:.4f} -- one query.",
        # THE diagnostic. recall@k and hit@k are not two measurements on a set whose
        # queries carry one relevant symbol each; on those queries recall@k is 0 or 1,
        # which is the definition of hit@k. Stating the count is what lets a reader see
        # whether the two columns are independent evidence or the same column twice.
        f"single-relevant queries: {single}/{n} ({single / n:.0%}); "
        f"mean relevant per query {_mean([float(len(r.relevant)) for r in card.results]):.2f}. "
        "On a single-relevant query recall@k IS hit@k, and nDCG/MAP collapse toward MRR.",
    ]
    if single == n:
        lines.append(
            "  ALL queries are single-relevant: recall@k, hit@k, nDCG and MAP are one "
            "measurement printed four times. Only the k at which they are cut differs."
        )
    if sources:
        lines.append(f"sources: {', '.join(sources)}")
    if len(sources) > 1:
        lines.append(
            "  POOLED across sources with different biases (hand-written questions "
            "borrow the code's vocabulary; mined prose describes the change instead). "
            "A pooled mean moves when the MIX changes and not only when the retriever "
            "does -- read the per-source rows from stratified_cards()."
        )
    if repos:
        lines.append(f"repos: {', '.join(repos)}")
    if len(repos) > 1:
        lines.append(
            f"  POOLED across {len(repos)} repos. Queries within one repo share an "
            "index, a naming convention and a labeller, so their errors correlate; "
            "pass cluster=True to ci()/delta_ci() to resample REPOS rather than "
            "queries. With fewer than ~5 repos that interval is itself unstable and "
            "the per-repo rows are the honest reporting."
        )
    lines.append(
        f"CI: bootstrap over queries, {BOOTSTRAP_RESAMPLES} resamples, "
        f"seed {BOOTSTRAP_SEED}. Marginal, so do NOT read row differences off it:"
        " use Scorecard.delta_ci, which resamples the same queries for both rows."
    )
    if n < CALIBRATION_FLOOR:
        lines.append(
            f"CALIBRATION: n={n} is below the {CALIBRATION_FLOOR}-query floor at which "
            "this bootstrap's false-positive rate reaches its nominal 5% on this "
            "corpus (11.9% at n=16, 7.9% at n=32). These are not 95% intervals; they "
            "are narrower than 95% intervals, in the direction that invents findings."
        )
    return lines


def format_table(cards: list[Scorecard]) -> str:
    header = (
        f"{'configuration':<30} {'scope':<18} {'n':>4} "
        f"{'recall@5':>8} {'recall@10':>9} {'hit@5':>7} "
        f"{'95% CI':>13} {'MRR':>7} {'nDCG@10':>8} {'MAP@10':>7}"
    )
    lines = [header, "-" * len(header)]
    lines += [c.row() for c in cards]
    diagnostics = _diagnostics(cards) if cards else []
    if diagnostics:
        # The caption travels with the table. Detached from these facts the rows read
        # as if a 0.02 difference meant something, which on this n it does not.
        lines += ["-" * len(header), *diagnostics]
    return "\n".join(lines)


def format_delta_report(
    cards: Sequence[Scorecard],
    baseline: str,
    metric: str = "ndcg",
    k: int = 10,
    cluster: bool | None = None,
) -> str:
    """Paired deltas against one baseline configuration, per stratum.

    One row per (configuration, stratum) rather than one per configuration, because a
    difference that holds on hand-written questions and reverses on mined prose is the
    single most important thing an enlarged gold set can tell us, and a pooled delta
    hides it by construction -- it is the average of the two.

    `cluster` defaults to ON whenever the stratum spans more than one repo, which is
    the case in which the per-query bootstrap is anti-conservative. Pass it explicitly
    to override.
    """
    by_name: dict[str, list[Scorecard]] = {}
    for card in cards:
        by_name.setdefault(card.scope, []).append(card)

    lines = [
        f"Paired 95% deltas vs {baseline!r} on {metric}@{k}, "
        f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.",
        f"{'scope':<20} {'configuration':<30} {'n':>4} {'delta':>8} {'95% CI':>17} {'':>4}",
        "-" * 86,
    ]
    for scope in sorted(by_name):
        rows = by_name[scope]
        base = next((c for c in rows if c.name == baseline), None)
        if base is None:
            lines.append(f"{scope:<20} (no {baseline!r} row in this stratum -- skipped)")
            continue
        repos = {r.repo for r in base.results if r.repo}
        use_cluster = (len(repos) > 1) if cluster is None else cluster
        base_mean = _mean(base.per_query(metric, k))
        for card in rows:
            if card.name == baseline:
                continue
            delta = _mean(card.per_query(metric, k)) - base_mean
            lo, hi = card.delta_ci(base, metric, k, cluster=use_cluster)
            mark = "*" if (lo > 0 or hi < 0) else ""
            lines.append(
                f"{scope:<20} {card.name:<30} {card.n:>4} {delta:>8.3f} "
                f"[{lo:>6.3f},{hi:>6.3f}] {mark:>4}"
            )
        if use_cluster:
            lines.append(
                f"{'':<20} (clustered by repo: {len(repos)} clusters"
                + ("; fewer than 5, so this interval is unstable)" if len(repos) < 5 else ")")
            )
    lines.append("* = interval excludes zero.")
    n = cards[0].n if cards else 0
    if n and n < CALIBRATION_FLOOR:
        lines.append(
            f"At n={n} a '*' is not a 5% claim: this bootstrap's measured "
            f"false-positive rate is above nominal below n={CALIBRATION_FLOOR}."
        )
    return "\n".join(lines)
