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

# How hard a test symbol's fused score is scaled when `prefer_implementation` is on.
#
# **0.5 is not a measured optimum, it is an unadjudicated trade-off**, and until there
# were two gold sets it was impossible to see that. The factor was swept against 170
# implementation queries in which 0 of 978 relevant labels is a test symbol, so half of
# what it does was invisible to the instrument that set it. Swept against BOTH sets --
# nDCG@10, 2000-resample paired bootstrap, seed 20250801, shipping index, per repo
# because TradingAgents has no test files and would dilute any average:
#
#   factor    impl swarm  impl kalshi  |  test swarm  test kalshi
#     1.0        0.264       0.196     |     0.654       0.230
#     0.8        0.360       0.229     |     0.538       0.177
#     0.7        0.372       0.235     |     0.500       0.156
#     0.6        0.416       0.245     |     0.463       0.143
#     0.5        0.450       0.249     |     0.308       0.101   <- here
#     0.4        0.455       0.249     |     0.074       0.021
#    0.25        0.457       0.249     |     0.032       0.003
#
# The last 0.005 of implementation gain costs three quarters of test-seeking retrieval.
# 0.5 sits one step from the edge of that: 0.6 would give back 0.034 of implementation
# nDCG on swarm-sync (0.004 on kalshi-bot) and buy 0.155 of test-seeking nDCG, a trade
# of roughly 5:1 in favour of the questions this file used to be unable to measure.
#
# It stays at 0.5 anyway, because choosing between those columns needs the one number
# nobody has: how often real users ask each kind of question. Moving it would be
# swapping one unmeasured assumption for another. `RESERVED_TEST_SLOTS` is the part of
# the problem that does NOT require that number, because it improves the right-hand
# columns while measuring +0.000 on the left.
TEST_DEMOTION_FACTOR = 0.5

# How many of the top `k` are held open for test symbols that the demotion pushed out.
#
# **Why a reserved slot exists at all.** The demotion is a multiplicative factor on an
# RRF score, and a multiplicative factor is not a demotion -- it is a demotion or a
# filter depending on how many modalities happened to vote for the symbol. A test that
# wins both a lexical and a dense vote is merely reordered by it. A test that wins only
# a lexical vote is annihilated by it, because every implementation symbol it competes
# with still has two votes. Which of those two things `0.5` does is therefore a property
# of the CORPUS, not of the ranking policy, and it changes underfoot. Same code, same
# factor, same k, test-seeking gold nDCG@10 on swarm-sync:
#
#     shipping index, embedder present ...................... 0.308
#     index built WITHOUT embeddings (a supported config) .... 0.023   hit@10 0.041
#     tests outside the embedding corpus ..................... 0.026
#
# In those last two the factor removes tests from the top FORTY, not just the top ten --
# zero tests survive on 100% of queries -- so the cross-encoder is never shown one either
# and cannot rescue what fusion already deleted.
#
# The reserve makes the outcome a property of the policy instead: after demotion, if
# fewer than this many tests survive into the top `k`, the best remaining ones are
# promoted into the lowest non-test slots. They enter at the bottom -- present, not
# dominant -- and the demotion still decides everything above them.
#
# **Why 2.** It is the largest floor that is free on the implementation gold. Measured at
# factor 0.5 on 170 implementation queries and 123 test-seeking queries, per repo:
#
#   reserve | impl swarm  impl kalshi  impl trading | test swarm  test kalshi
#      0    |   0.450       0.249        0.184      |   0.308       0.101
#      1    |   0.450       0.249        0.184      |   0.312       0.107
#      2    |   0.450       0.249        0.184      |   0.327       0.114   <- free, and
#      3    |   0.443       0.241        0.184      |   0.336       0.115      strictly
#      5    |   0.428       0.232        0.184      |   0.360       0.135      positive
#
# At 2 the implementation interval is [+0.000,+0.000] on all three repos -- it changes
# the top 10 of 106/170 implementation queries and displaces a relevant symbol on ZERO
# of them -- while test-seeking gains +0.019 [+0.002,+0.038] on swarm-sync and
# +0.013 [+0.000,+0.030] on kalshi-bot. 3 is where the left-hand columns start paying
# (-0.005 [-0.011,-0.001]), so 2 is the last value that costs nothing.
#
# The gain is far larger where the factor is a filter rather than a nudge: in the
# no-embedder configuration, swarm-sync test-seeking nDCG@10 goes 0.023 -> 0.184
# (+0.161 [+0.128,+0.195]) and hit@10 0.041 -> 0.603, with the implementation gold
# unmoved at 0.393/0.121/0.068. That is the case the reserve exists for.
RESERVED_TEST_SLOTS = 2


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Hit]],
    k: int = 10,
    weights: dict[str, float] | None = None,
    prefer_implementation: bool = False,
    *,
    reserved_test_slots: int = RESERVED_TEST_SLOTS,
) -> list[Hit]:
    """Fuse per-modality ranked lists into one, best first.

    `prefer_implementation` demotes test code, and it is the single largest quality
    lever measured so far: on 170 hand gold queries across three repos it is worth
    nDCG@10 +0.181 [+0.142,+0.225] on swarm-sync and +0.051 [+0.024,+0.083] on
    kalshi-bot, and exactly +0.000 on TradingAgents, which has no test files for the
    mechanism to fire on. Bigger than adding a whole retrieval modality.

    **What that number leaves out, and what it cost.** Every one of those gold queries
    is of the form "how does X work", and 0 of their 978 relevant labels is a test
    symbol -- so that gold set is structurally incapable of noticing "how is X tested"
    getting worse, and for as long as it was the only instrument, it did not. Measured
    against `hand_tests_*` (123 test-seeking queries, no name overlap, the word "test"
    banned because a pytest function names its own answer), the same +0.181 costs
    swarm-sync 0.654 -> 0.308 nDCG@10 on the questions the first gold set could not see.
    That is a trade, not a free win, and `TEST_DEMOTION_FACTOR` records where on the
    curve it sits. `reserved_test_slots` is the part of it that is not a trade.
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
                scores[symbol_id] *= TEST_DEMOTION_FACTOR

    ordered = sorted(scores.items(), key=lambda item: -item[1])

    if prefer_implementation:
        # "A demotion, not a filter" is a claim about the factor that the factor cannot
        # keep on its own: whether it demotes or deletes depends on how many modalities
        # voted, which is a fact about the index and not about this policy. The reserve
        # is what makes the comment above true in every configuration.
        ordered = _reserve_test_slots(ordered, best, k, reserved_test_slots)

    return [
        replace(
            best[symbol_id],
            score=score,
            modality="+".join(sorted(set(contributors[symbol_id]))),
        )
        for symbol_id, score in ordered[:k]
    ]


def _reserve_test_slots(
    ordered: list[tuple[int, float]],
    best: dict[int, Hit],
    k: int,
    reserved: int,
) -> list[tuple[int, float]]:
    """Hold `reserved` of the top `k` open for tests the demotion pushed out.

    The promoted tests are the highest-scoring ones available, and they enter at the
    bottom of the top `k`, displacing the lowest-scoring NON-tests. Nothing above them
    moves: the demotion still decides the ordering of everything a test-seeking user
    did not ask for, and a run where enough tests already survived is untouched.

    Returns the full ordering rather than a truncated one, so the caller's `[:k]` is
    the only place the cut happens.
    """
    # The reserve may never claim more than half the results. The bound is inactive at
    # both depths this was measured at (2 <= 10//2 and 2 <= 40//2), so it changes none
    # of the numbers in `RESERVED_TEST_SLOTS`; it exists for the small-k callers those
    # numbers say nothing about. `search(k=1)` asks for the single best answer, and a
    # reserve that answered it with a test would have turned a floor into a ceiling.
    reserved = min(reserved, k // 2)
    if reserved <= 0 or k <= 0:
        return ordered
    head, tail = ordered[:k], ordered[k:]
    surviving = sum(1 for symbol_id, _ in head if best[symbol_id].is_test)
    if surviving >= reserved:
        return ordered

    available = [item for item in tail if best[item[0]].is_test]
    # Never evict a test to make room for another one, and never grow past `k`: the
    # number promoted is bounded by what exists below AND by what can move aside above.
    displaceable = [item for item in head if not best[item[0]].is_test]
    room = min(reserved - surviving, len(available), len(displaceable))
    if room == 0:
        return ordered

    promoted = available[:room]
    # `displaceable` is still in score order, so its last `room` entries are the
    # lowest-scoring non-tests in the top k -- the cheapest thing to give up.
    evicted = {symbol_id for symbol_id, _ in displaceable[-room:]}
    new_head = sorted(
        [item for item in head if item[0] not in evicted] + promoted,
        key=lambda item: -item[1],
    )
    head_ids = {symbol_id for symbol_id, _ in new_head}
    return new_head + [item for item in ordered if item[0] not in head_ids]
