"""Reciprocal Rank Fusion across retrieval modalities.

**Why rank-based and not score-based.** The three modalities produce numbers on
wildly incompatible scales: BM25 lands around 10-25, cosine similarity in 0-1, graph
activation in roughly 0-2 with a long tail. Normalising them (min-max, z-score)
requires assuming a distribution shape that none of them actually has, and the
result silently depends on how many candidates each modality happened to return.

RRF sidesteps the problem by discarding magnitudes entirely and using only position:

    RRF(d) = sum over modalities m of  weight[m] / (K + rank_m(d))

A document ranked 1st by one modality and absent from the others scores
`1/(60+1) = 0.0164`. A document ranked 3rd by all three scores `3/63 = 0.0476` and
wins. That is the desired behaviour -- agreement across independent signals beats a
single confident vote -- and it falls out of the formula rather than being tuned in.

`K = 60` is the value from the original RRF paper (Cormack et al., 2009). It is
large enough that the difference between rank 1 and rank 2 is modest, which keeps
one modality's top hit from dominating.
"""
from __future__ import annotations

from dataclasses import replace

from .lexical import Hit

# From Cormack, Clarke & Buettcher (2009). Damps the gap between adjacent ranks so
# that consensus matters more than any single modality's first place.
RRF_K = 60

# Per-modality weight. Lexical and dense are peers. The graph weight is 0.3 because
# it was MEASURED, not because it felt right -- the first guess was 0.6 and it made
# retrieval worse.
#
# Weight sweep on the 16-query swarm-sync gold set (recall@5, test demotion on):
#     0.3 -> 0.646    0.6 -> 0.615    1.0 -> 0.552    1.5 -> 0.354
# Monotonically harmful above ~0.3, and at the original unweighted default the whole
# hybrid scored 0.385 against 0.573 for lexical+dense alone -- the graph modality was
# actively destroying retrieval quality.
#
# The reason is structural, not a tuning accident: graph expansion has no query
# representation. It contributes symbols that text retrieval missed, which raises
# recall (0.604 -> 0.646 @5, 0.781 -> 0.802 @10), but every vote it casts is
# evidence about the CODE rather than about the QUESTION, so at higher weights those
# votes displace better-matched answers at the top. MRR falls 0.516 -> 0.453 even at
# the weight that maximises recall. That trade is real and is not tuned away.
#
# Phase 3b answered the "so fix it downstream" half of that. A cross-encoder, which
# DOES read the query, recovers the lost ranking and then some: MRR 0.453 -> 0.679
# with graph still on at 0.3. What it did NOT do is vindicate the graph modality --
# with reranking enabled, turning graph off scores identically on all four metrics.
# See `rerank.py`. The weight stays at 0.3 because these numbers do not argue for
# moving it in either direction, not because anything here re-measured it.
DEFAULT_WEIGHTS = {"lexical": 1.0, "dense": 1.0, "graph": 0.3}


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Hit]],
    k: int = 10,
    weights: dict[str, float] | None = None,
    prefer_implementation: bool = False,
) -> list[Hit]:
    """Fuse per-modality ranked lists into one, best first.

    `prefer_implementation` demotes test code, and it is the single largest quality
    lever measured so far: on lexical+dense it moves recall@10 from 0.635 to 0.781
    and MRR from 0.331 to 0.516. Bigger than adding a whole retrieval modality.

    **The honest caveat about that number.** All 16 gold queries are of the form
    "how does X work", and that question shape is precisely the one the demotion
    helps. A gold set asking "how is X tested" would show the opposite. The default
    is on because finding implementation is the dominant use, but the measurement
    validates it against a question distribution this author chose, and that is a
    real limit on how much the number proves.
    """
    weights = weights or DEFAULT_WEIGHTS
    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}
    contributors: dict[int, list[str]] = {}

    for modality, hits in ranked_lists.items():
        weight = weights.get(modality, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.symbol_id] = scores.get(hit.symbol_id, 0.0) + weight / (
                RRF_K + rank
            )
            contributors.setdefault(hit.symbol_id, []).append(modality)
            # Keep the richest representation of the symbol. A graph hit carries a
            # `via` explanation the text modalities do not have, and losing it would
            # throw away the only account of why the symbol surfaced.
            existing = best.get(hit.symbol_id)
            if existing is None or (not existing.via and hit.via):
                best[hit.symbol_id] = hit

    if prefer_implementation:
        # A demotion, not a filter. Tests still rank -- they are often genuinely the
        # best answer -- but they no longer outrank the code they exercise purely
        # because they describe it in more words.
        for symbol_id, hit in best.items():
            if hit.is_test:
                scores[symbol_id] *= 0.5

    ordered = sorted(scores.items(), key=lambda item: -item[1])[:k]
    return [
        replace(
            best[symbol_id],
            score=score,
            modality="+".join(sorted(set(contributors[symbol_id]))),
        )
        for symbol_id, score in ordered
    ]
