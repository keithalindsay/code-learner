"""The ablation's metrics, hand-computed.

This module had **zero tests** and 12 of 12 functions were never entered by any test,
while producing the table the fusion design rests on. Two mutations survived a
mutation run as a direct consequence:

* `recall_at` ignoring its `k` -- so every recall@5 was silently a recall@10, and the
  two columns of the published table were the same measurement printed twice.
* `mrr` returning the raw rank instead of its reciprocal -- which inverts the metric
  (bigger is worse) and puts it outside [0, 1], and still scored a table that read
  plausibly because nobody compared it to a number worked out by hand.

Both are killed below by fixtures whose expected value is arithmetic in the docstring
rather than whatever the code happens to return. Nothing here builds an index or
touches a repo: `QueryResult` is a plain record of "what was asked, what is relevant,
what came back", so the metrics are testable at that level and should be tested there.
"""
from __future__ import annotations

import pytest

from codelearner.eval import ablation
from codelearner.eval.ablation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    QueryResult,
    Scorecard,
    format_table,
    load_gold,
)


def _result(relevant, retrieved):
    """A `QueryResult` with `first_relevant_rank` filled the way `_score` fills it."""
    out = QueryResult(query="q", relevant=list(relevant), retrieved=list(retrieved))
    for rank, qualname in enumerate(out.retrieved, start=1):
        if qualname in out.relevant:
            out.first_relevant_rank = rank
            break
    return out


# --------------------------------------------------------------------------------
# recall@k -- the mutation that survived was ignoring k
# --------------------------------------------------------------------------------


def test_recall_at_counts_only_the_top_k():
    """THE surviving mutation: a `recall_at` that ignores `k` returns 1.0 here.

    Three relevant symbols, retrieved at ranks 1, 6 and 7. recall@5 is 1/3 -- only
    `a` is inside the top five. recall@10 is 3/3. A `recall_at` that sliced the whole
    list would report 1.0 for both and the two columns of the table would be one
    measurement printed twice.
    """
    result = _result(["a", "b", "c"], ["a", "x", "y", "z", "w", "b", "c"])
    assert result.recall_at(5) == pytest.approx(1 / 3)
    assert result.recall_at(10) == pytest.approx(1.0)
    assert result.recall_at(1) == pytest.approx(1 / 3)
    assert result.recall_at(0) == 0.0


def test_recall_is_a_fraction_of_the_relevant_set_not_of_the_retrieved_set():
    """Two relevant, one found in the top 5: 0.5, whatever else came back.

    The denominator is what the query SHOULD find. A recall that divided by `k` would
    read 0.2 here and would fall as the candidate set grew, which is a precision.
    """
    result = _result(["a", "b"], ["a", "x", "y", "z", "w"])
    assert result.recall_at(5) == pytest.approx(0.5)


def test_a_query_with_no_relevant_symbols_scores_zero_rather_than_dividing_by_zero():
    assert _result([], ["a", "b"]).recall_at(5) == 0.0


# --------------------------------------------------------------------------------
# hit@k
# --------------------------------------------------------------------------------


def test_hit_at_k_is_any_relevant_symbol_inside_the_cut():
    """hit@k asks a different question from recall@k: ANY, not HOW MANY.

    The same result is hit@5 True (b is at rank 5) and recall@5 only 1/2.
    """
    result = _result(["b", "c"], ["x", "y", "z", "w", "b", "c"])
    assert result.hit_at(5) is True
    assert result.recall_at(5) == pytest.approx(0.5)
    assert result.hit_at(4) is False
    assert result.hit_at(6) is True


def test_hit_at_k_is_false_when_nothing_relevant_was_retrieved():
    assert _result(["a"], ["x", "y"]).hit_at(5) is False
    assert _result([], ["x"]).hit_at(5) is False


# --------------------------------------------------------------------------------
# MRR -- the mutation that survived was returning the raw rank
# --------------------------------------------------------------------------------


def test_mrr_is_the_reciprocal_of_the_rank_not_the_rank():
    """THE other surviving mutation. Ranks 1 and 4 give (1/1 + 1/4)/2 = 0.625.

    Returning the raw rank gives (1 + 4)/2 = 2.5 -- outside [0, 1] and ordered the
    wrong way, since a bigger number would then mean a worse retriever.
    """
    card = Scorecard(name="c")
    card.results.append(_result(["a"], ["a", "x"]))
    card.results.append(_result(["b"], ["x", "y", "z", "b"]))
    assert card.mrr == pytest.approx(0.625)
    assert 0.0 <= card.mrr <= 1.0


def test_mrr_scores_a_query_that_found_nothing_as_zero_not_as_infinity():
    """`first_relevant_rank is None` is a miss, and a miss contributes 0.

    Dropping it from the average instead would make MRR rise as retrieval got worse,
    because the queries it failed would stop counting.
    """
    card = Scorecard(name="c")
    card.results.append(_result(["a"], ["a"]))
    card.results.append(_result(["b"], ["x", "y"]))
    assert card.results[1].first_relevant_rank is None
    assert card.mrr == pytest.approx(0.5)


def test_only_the_first_relevant_hit_sets_the_rank():
    """MRR is about the top answer. A second relevant symbol lower down cannot help it."""
    result = _result(["a", "b"], ["x", "a", "b"])
    assert result.first_relevant_rank == 2


# --------------------------------------------------------------------------------
# _mean, including the empty case
# --------------------------------------------------------------------------------


def test_mean_of_nothing_is_zero_rather_than_a_zero_division():
    """An empty scorecard is a legitimate state -- a modality that was skipped -- and
    it has to print a row rather than abort the table."""
    assert ablation._mean([]) == 0.0
    assert ablation._mean([1.0, 2.0]) == pytest.approx(1.5)
    empty = Scorecard(name="skipped")
    assert empty.recall_at(5) == 0.0
    assert empty.hit_at(5) == 0.0
    assert empty.mrr == 0.0
    assert empty.row().startswith("skipped")


# --------------------------------------------------------------------------------
# The gold set's own arithmetic -- WP13.4
# --------------------------------------------------------------------------------


def test_the_gold_set_is_too_small_for_the_noise_band_it_used_to_quote():
    """The caption said "treat one or two points as noise". The quantum is 6.25.

    11 of the 16 queries carry exactly one relevant symbol, so on two thirds of the
    set recall@5 is a coin flip worth 1/16 of the mean, and hit@5 has that quantum on
    all of it. A stated band below the instrument's resolution is worse than no band,
    because it invites exactly the fine-grained reading the set cannot support.
    """
    gold = load_gold("swarm_sync")
    queries = gold["queries"]
    assert len(queries) == 16
    assert sum(1 for q in queries if len(q["relevant"]) == 1) == 11
    assert 1 / len(queries) == pytest.approx(0.0625)


def test_the_table_caption_states_the_quantum_and_the_seed():
    """The three facts every row depends on travel WITH the rows.

    A seed and a resample count that live only in the source are not reproducible by
    the person reading the number, and a quantum stated nowhere gets read as zero.
    """
    card = Scorecard(name="lexical only")
    card.results.append(_result(["a"], ["a"]))
    card.results.append(_result(["b"], ["x"]))
    text = format_table([card])
    assert "n = 2 queries" in text
    assert "0.5000" in text  # the quantum: one query out of two
    assert f"seed {BOOTSTRAP_SEED}" in text
    assert f"{BOOTSTRAP_RESAMPLES} resamples" in text
    assert "delta_ci" in text


# --------------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------------


def _card(name, hits):
    """A scorecard whose hit@5 pattern is exactly `hits` (1 = found at rank 1)."""
    card = Scorecard(name=name)
    for i, hit in enumerate(hits):
        card.results.append(_result([f"r{i}"], [f"r{i}"] if hit else ["miss"]))
    return card


def test_per_query_returns_one_value_per_query_for_each_metric():
    """The bootstrap resamples QUERIES. Two symbols relevant to one query are not two
    independent trials of the retriever, so the unit has to be the query."""
    card = Scorecard(name="c")
    card.results.append(_result(["a", "b"], ["a", "b"]))
    card.results.append(_result(["c"], ["x", "c"]))
    assert card.per_query("hit", 5) == [1.0, 1.0]
    assert card.per_query("recall", 5) == pytest.approx([1.0, 1.0])
    assert card.per_query("mrr") == pytest.approx([1.0, 0.5])
    with pytest.raises(ValueError, match="unknown metric"):
        card.per_query("f1")


def test_the_interval_brackets_the_point_estimate_and_is_reproducible():
    """And the endpoints are themselves quantised to 1/16, which is the finding.

    Reseeding does not move this interval: a mean of 16 booleans can only land on a
    16-point grid, so the 2.5th and 97.5th percentiles of 2000 resamples fall on the
    same two grid points under any seed. The interval is honest and it is also coarse,
    for the same reason the caption now states -- 16 queries. `_resample_indices` is
    where seed sensitivity is pinned, because there it is not hidden by the grid.
    """
    card = _card("c", [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    lo, hi = card.ci("hit", 5)
    assert lo <= card.hit_at(5) <= hi
    assert card.ci("hit", 5) == card.ci("hit", 5)
    assert (lo * 16) == pytest.approx(round(lo * 16))
    assert (hi * 16) == pytest.approx(round(hi * 16))


def test_the_interval_on_sixteen_queries_is_about_ten_points_wide_each_side():
    """The number that makes the old caption wrong by an order of magnitude.

    Binomial sd at n=16 is `sqrt(p(1-p)/16)`: 12.5 points at p=0.5, 10.8 at p=0.75.
    A 95% interval is roughly +/-2sd, so 12 of 16 hits resolves to something like
    +/-20 points -- against a caption that offered "one or two".
    """
    card = _card("c", [1] * 12 + [0] * 4)
    lo, hi = card.ci("hit", 5)
    assert card.hit_at(5) == pytest.approx(0.75)
    assert (hi - lo) > 0.25, "an interval this narrow would be understating n=16"


def test_a_perfect_row_has_a_degenerate_interval_rather_than_a_wrong_one():
    """16 of 16 resamples to 16 of 16 every time. The interval is a point, and that is
    a true statement about the bootstrap rather than evidence of certainty."""
    card = _card("c", [1] * 16)
    assert card.ci("hit", 5) == (1.0, 1.0)


def test_the_paired_delta_resolves_a_difference_two_marginal_intervals_cannot():
    """Why `delta_ci` exists and why reading two `ci()` intervals is not a substitute.

    Two rows answer the same 16 queries. Row B wins exactly the four queries row A
    loses and they agree everywhere else, so the difference is 0.25 with no
    disagreement about which queries are hard. Marginally the two intervals overlap
    heavily; paired, the interval excludes zero.
    """
    a = _card("a", [1] * 10 + [0] * 6)
    b = _card("b", [1] * 14 + [0] * 2)
    assert b.hit_at(5) - a.hit_at(5) == pytest.approx(0.25)

    a_lo, a_hi = a.ci("hit", 5)
    b_lo, b_hi = b.ci("hit", 5)
    assert a_hi >= b_lo, "the marginal intervals must overlap, or this proves nothing"

    lo, hi = b.delta_ci(a, "hit", 5)
    assert lo > 0.0, "the paired interval must exclude zero"
    assert lo <= 0.25 <= hi


def test_the_paired_delta_is_zero_when_two_rows_are_the_same_row():
    """Identical per-query outcomes have no difference to resample. A `delta_ci` that
    drew independent indices for the two sides would return a non-zero interval here,
    which is precisely the un-pairing this is guarding against."""
    a = _card("a", [1, 0, 1, 1, 0, 1, 0, 0])
    b = _card("b", [1, 0, 1, 1, 0, 1, 0, 0])
    assert a.delta_ci(b, "hit", 5) == (0.0, 0.0)


def test_every_scorecard_over_the_same_gold_set_gets_the_same_resample_draws():
    """The mechanism behind pairing, pinned directly.

    The draws are a pure function of (n, resamples, seed) rather than of a per-card
    RNG. A per-card RNG would print intervals that look identical and silently un-pair
    every comparison drawn from them.
    """
    first = ablation._resample_indices(16, 50, BOOTSTRAP_SEED)
    second = ablation._resample_indices(16, 50, BOOTSTRAP_SEED)
    assert first == second
    assert first != ablation._resample_indices(16, 50, BOOTSTRAP_SEED + 1)
    assert all(len(draw) == 16 for draw in first)
    assert all(0 <= i < 16 for draw in first for i in draw)


def test_an_empty_scorecard_has_no_interval_rather_than_an_invented_one():
    assert Scorecard(name="skipped").ci("hit", 5) == (0.0, 0.0)
    assert Scorecard(name="a").delta_ci(Scorecard(name="b")) == (0.0, 0.0)


# --------------------------------------------------------------------------------
# The retraction -- WP13, ablation.py:246-261
# --------------------------------------------------------------------------------


def test_the_retracted_reranking_numbers_are_not_in_the_source_any_more():
    """A comment headed "MEASURED" carried four reranking rows the README retracted,
    plus three conclusions drawn from them, and the retraction never reached this
    file. A source comment cannot be re-measured, so it will always drift back toward
    this failure; the fix is that the numbers are not written down here at all."""
    from pathlib import Path

    text = Path(ablation.__file__).read_text()
    for retracted in ("0.750", "0.781", "0.875", "0.679", "+0.226"):
        assert retracted not in text, f"retracted figure {retracted} still in ablation.py"
