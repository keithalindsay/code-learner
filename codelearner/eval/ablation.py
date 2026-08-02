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
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..index.embed import Embedder
from ..retrieve.dense import search_dense
from ..retrieve.fuse import reciprocal_rank_fusion
from ..retrieve.graph import expand
from ..retrieve.lexical import Hit, search_lexical
from ..retrieve.rerank import Reranker
from ..retrieve.search import search

GOLD_DIR = Path(__file__).parent / "gold"

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

    def recall_at(self, k: int) -> float:
        if not self.relevant:
            return 0.0
        found = sum(1 for r in self.relevant if r in self.retrieved[:k])
        return found / len(self.relevant)

    def hit_at(self, k: int) -> bool:
        return any(r in self.retrieved[:k] for r in self.relevant)


@dataclass
class Scorecard:
    name: str
    results: list[QueryResult] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        return _mean([r.recall_at(k) for r in self.results])

    def hit_at(self, k: int) -> float:
        return _mean([1.0 if r.hit_at(k) else 0.0 for r in self.results])

    @property
    def mrr(self) -> float:
        return _mean(
            [
                1.0 / r.first_relevant_rank if r.first_relevant_rank else 0.0
                for r in self.results
            ]
        )

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
        raise ValueError(f"unknown metric {metric!r}")

    def ci(
        self,
        metric: str = "hit",
        k: int = 5,
        resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED,
    ) -> tuple[float, float]:
        """95% bootstrap interval over the 16 queries. See the module docstring."""
        values = self.per_query(metric, k)
        if not values:
            return (0.0, 0.0)
        draws = _resample_indices(len(values), resamples, seed)
        return _percentile_ci([_mean([values[i] for i in idx]) for idx in draws])

    def delta_ci(
        self,
        other: Scorecard,
        metric: str = "hit",
        k: int = 5,
        resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED,
    ) -> tuple[float, float]:
        """PAIRED interval for `self - other`, resampling the same queries for both.

        This is the comparison the table is for, and it is not the same thing as
        looking at whether two `ci()` intervals overlap. Both rows answer the same 16
        queries, so most of the variance in either interval is "this query is hard",
        which cancels in the difference. Two overlapping marginal intervals routinely
        sit either side of a difference whose paired interval excludes zero.
        """
        mine = self.per_query(metric, k)
        theirs = other.per_query(metric, k)
        if not mine or len(mine) != len(theirs):
            return (0.0, 0.0)
        diffs = [a - b for a, b in zip(mine, theirs, strict=True)]
        draws = _resample_indices(len(diffs), resamples, seed)
        return _percentile_ci([_mean([diffs[i] for i in idx]) for idx in draws])

    def row(self) -> str:
        lo, hi = self.ci("hit", 5)
        return (
            f"{self.name:<30} {self.recall_at(5):>8.3f} {self.recall_at(10):>9.3f} "
            f"{self.hit_at(5):>7.3f} [{lo:>5.3f},{hi:>5.3f}] {self.mrr:>7.3f}"
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


def _percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    lo = ordered[min(len(ordered) - 1, int((alpha / 2) * len(ordered)))]
    hi = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return (lo, hi)


def load_gold(name: str = "swarm_sync") -> dict:
    return json.loads((GOLD_DIR / f"{name}.json").read_text())


def _score(name: str, per_query: list[tuple[dict, list[Hit]]]) -> Scorecard:
    card = Scorecard(name=name)
    for spec, hits in per_query:
        retrieved = [h.qualname for h in hits]
        result = QueryResult(
            query=spec["query"], relevant=spec["relevant"], retrieved=retrieved
        )
        for rank, qualname in enumerate(retrieved, start=1):
            if qualname in spec["relevant"]:
                result.first_relevant_rank = rank
                break
        card.results.append(result)
    return card


def run_ablation(
    conn: sqlite3.Connection,
    embedder: Embedder | None = None,
    gold_name: str = "swarm_sync",
    k: int = 10,
    reranker: Reranker | None = None,
) -> list[Scorecard]:
    """Score every modality alone and every combination that matters.

    `reranker` is optional and adds three rows when supplied. It is a separate
    argument rather than always-on because the reranker costs a forward pass per
    candidate -- roughly 640 of them per row -- and the modality rows must stay
    runnable in seconds on a machine with no GPU.
    """
    gold = load_gold(gold_name)
    queries = gold["queries"]
    cards: list[Scorecard] = []

    # --- single modalities -------------------------------------------------
    cards.append(
        _score("lexical only", [(q, search_lexical(conn, q["query"], k=k)) for q in queries])
    )
    if embedder is not None:
        cards.append(
            _score("dense only", [(q, search_dense(conn, q["query"], embedder, k=k)) for q in queries])
        )

    # Graph alone is meaningless -- it has no query representation and must be
    # seeded. Scored here seeded by dense, to isolate what expansion ADDS.
    if embedder is not None:
        graph_only = []
        for q in queries:
            seeds = search_dense(conn, q["query"], embedder, k=5)
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
                                "lexical": search_lexical(conn, q["query"], k=k * 4),
                                "dense": search_dense(conn, q["query"], embedder, k=k * 4),
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
                                "lexical": search_lexical(conn, q["query"], k=k * 4),
                                "dense": search_dense(conn, q["query"], embedder, k=k * 4),
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
                        conn, q["query"], k=k, embedder=embedder,
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
                        conn, q["query"], k=k, embedder=embedder,
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
                                conn, q["query"], k=k, embedder=embedder,
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
                    (q, _weighted_hybrid(conn, q["query"], embedder, k, weight))
                    for q in queries
                ],
            )
        )
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


def format_table(cards: list[Scorecard]) -> str:
    n = len(cards[0].results) if cards else 0
    header = (
        f"{'configuration':<30} {'recall@5':>8} {'recall@10':>9} {'hit@5':>7} "
        f"{'95% CI':>13} {'MRR':>7}"
    )
    lines = [header, "-" * len(header)]
    lines += [c.row() for c in cards]
    if n:
        # The caption travels with the table. Detached from these three facts the
        # rows read as if a 0.02 difference meant something, which on 16 queries it
        # does not.
        lines += [
            "-" * len(header),
            f"n = {n} queries. hit@5 moves in steps of {1 / n:.4f} -- one query.",
            f"CI: bootstrap over queries, {BOOTSTRAP_RESAMPLES} resamples, "
            f"seed {BOOTSTRAP_SEED}. Marginal, so do NOT read row differences off it:"
            " use Scorecard.delta_ci, which resamples the same queries for both rows.",
        ]
    return "\n".join(lines)
