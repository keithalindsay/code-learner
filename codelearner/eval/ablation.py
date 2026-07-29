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

The gold set is small (16 queries) and hand-labelled, so treat differences of one or
two points as noise. It is enough to tell a modality that works from one that does
not, and not enough to justify fine-grained tuning -- a limit worth respecting
rather than working around.
"""
from __future__ import annotations

import json
import sqlite3
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

    def row(self) -> str:
        return (
            f"{self.name:<28} {self.recall_at(5):>8.3f} {self.recall_at(10):>9.3f} "
            f"{self.hit_at(5):>7.3f} {self.mrr:>7.3f}"
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    # MEASURED, with `zerank-1-small-reranker` on the swarm-sync gold set:
    #
    #   hybrid + prefer_impl          0.646  0.802  0.750  0.453   <- the baseline
    #   hybrid + rerank               0.750  0.781  0.875  0.679
    #   lex+dense pref_impl+rerank    0.750  0.781  0.875  0.679
    #   hybrid + rerank no pref_impl  0.688  0.781  0.812  0.677
    #
    # Answers, in order. Yes -- MRR +0.226, past even the 0.516 that lexical+dense
    # +prefer_impl managed without the graph modality diluting it. No -- rows two and
    # three tie exactly, so nothing here attributes any of that gain to graph
    # expansion. And mostly yes on the third: the demotion no longer moves MRR (0.679
    # vs 0.677) but still moves recall@5 (0.750 vs 0.688), so it stays on.
    #
    # The row that is easy to skip: recall@10 FELL, 0.802 -> 0.781, in every reranked
    # configuration. Reranking reorders a fixed candidate set and cannot add recall;
    # here it traded a gold symbol sitting at rank 9-10 for better answers above it.
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
    header = f"{'configuration':<28} {'recall@5':>8} {'recall@10':>9} {'hit@5':>7} {'MRR':>7}"
    lines = [header, "-" * len(header)]
    lines += [c.row() for c in cards]
    return "\n".join(lines)
