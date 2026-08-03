"""The hybrid retrieval pipeline: three modalities, fused, optionally reranked.

    lexical (BM25) ─┐
                    ├─> RRF fusion ─> [rerank] ─> results
    dense (vector) ─┤
                    │
    graph expansion ┘   (seeded by the text modalities' output)

Graph expansion runs *after* the text modalities rather than beside them, because it
has no query representation of its own -- it needs somewhere to start. That ordering
is the whole reason it can rescue an answer the text signals missed: the seeds carry
the query's meaning, and the graph carries the structure that meaning implies.

Every stage is individually switchable, which is not a convenience feature. Phase 8's
per-modality ablation has to be able to run lexical-only, dense-only, graph-only, and
every combination, or the question "which modality actually carries retrieval" cannot
be answered.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..index.embed import Embedder
from .dense import search_dense
from .fuse import RESERVED_TEST_SLOTS, reciprocal_rank_fusion
from .graph import expand
from .lexical import Hit, search_lexical
from .rerank import Reranker

# Each modality retrieves deeper than the final k, so fusion has room to promote a
# symbol that several modalities agree on but none ranked first.
CANDIDATE_MULTIPLIER = 4


@dataclass
class SearchResult:
    hits: list[Hit]
    # Per-modality candidate lists, kept for the ablation and for explaining a
    # result. Retrieval that cannot show its work is hard to improve.
    per_modality: dict[str, list[Hit]]
    reranked: bool = False


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    embedder: Embedder | None = None,
    use_lexical: bool = True,
    use_dense: bool = True,
    use_graph: bool = True,
    reranker: Reranker | None = None,
    prefer_implementation: bool = True,
    reserved_test_slots: int = RESERVED_TEST_SLOTS,
) -> SearchResult:
    """Run the hybrid pipeline and return the top `k` symbols.

    `embedder` is required for the dense modality; without one, dense is skipped
    rather than erroring, so an index built without embeddings still answers.

    `reranker` is optional and `None` means "skip the stage" -- a machine with no
    model retrieves exactly as this pipeline did before Phase 3b. When one IS given,
    fusion deliberately returns `k * CANDIDATE_MULTIPLIER` candidates instead of `k`,
    because the whole point of the stage is to reorder a set wider than the answer:
    truncating to `k` first would let RRF discard the recall graph expansion bought
    before the cross-encoder ever saw it. See `rerank.py` for the measured effect.

    `prefer_implementation` defaults ON: measured on 170 hand gold queries across three
    repos it is the largest single quality lever (nDCG@10 +0.181 on swarm-sync). Turn
    it off for questions genuinely about test code. See `fuse.reciprocal_rank_fusion`
    for the caveat about what that measurement does and does not establish.

    `reserved_test_slots` is what keeps that demotion from becoming a filter when tests
    lose a modality's vote -- which is not hypothetical, it is what this function does
    RIGHT HERE whenever `embedder is None`. See `fuse.RESERVED_TEST_SLOTS`.

    The reserve is applied at whatever depth fusion runs, which means it does two
    different jobs on the two paths, both of them the right one. With no reranker it
    guarantees the answer is in the results. With a reranker it guarantees the
    cross-encoder is SHOWN a test -- and that guarantee is the load-bearing one, because
    without it a demoted test is absent from all `k * CANDIDATE_MULTIPLIER` candidates
    (measured: zero tests in the top 40 on 100% of queries once the dense vote is gone),
    and a stage that reads the query cannot rerank what it was never given.

    What is NOT measured is the reranked end-to-end number -- whether the cross-encoder
    then promotes a reserved test or buries it. The GPU was contended and the run fell
    back to unreranked order often enough to make the result meaningless, so it was
    thrown away rather than reported. The reserve is justified on the fusion numbers
    alone; how much the reranker adds to or subtracts from that is still open.
    """
    depth = k * CANDIDATE_MULTIPLIER
    per_modality: dict[str, list[Hit]] = {}

    if use_lexical:
        per_modality["lexical"] = search_lexical(conn, query, k=depth)
    if use_dense and embedder is not None:
        per_modality["dense"] = search_dense(conn, query, embedder, k=depth)

    if use_graph:
        # Seed from whatever the text modalities found, best-first and deduplicated.
        seeds = _merge_seeds(per_modality, limit=k)
        if seeds:
            per_modality["graph"] = expand(conn, seeds, k=depth)

    fused = reciprocal_rank_fusion(
        per_modality,
        k=depth if reranker is not None else k,
        prefer_implementation=prefer_implementation,
        reserved_test_slots=reserved_test_slots,
    )

    if reranker is not None and fused:
        fused = reranker.rerank(query, fused, k=k)
        return SearchResult(hits=fused, per_modality=per_modality, reranked=True)

    return SearchResult(hits=fused[:k], per_modality=per_modality)


def _merge_seeds(per_modality: dict[str, list[Hit]], limit: int) -> list[Hit]:
    """Interleave modality results into one seed list, best-first, deduplicated.

    Interleaved rather than concatenated so that no single modality supplies every
    seed -- if lexical returns junk for this query, dense still gets to steer the
    expansion, and vice versa.
    """
    seeds: list[Hit] = []
    seen: set[int] = set()
    lists = [hits for hits in per_modality.values() if hits]
    for rank in range(max((len(h) for h in lists), default=0)):
        for hits in lists:
            if rank < len(hits) and hits[rank].symbol_id not in seen:
                seen.add(hits[rank].symbol_id)
                seeds.append(hits[rank])
                if len(seeds) >= limit:
                    return seeds
    return seeds
