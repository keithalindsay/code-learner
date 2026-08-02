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

import json
import sqlite3

import pytest

from codelearner.eval import ablation, goldset
from codelearner.eval.ablation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_FLOOR,
    QueryResult,
    Scorecard,
    ci_half_width,
    design_effect,
    format_delta_report,
    format_table,
    load_gold,
    paired_sd,
    power_curve,
    required_n,
    stratify,
)
from codelearner.eval.goldset import GoldQuery, load_gold_set


def _result(relevant, retrieved, source="", repo=""):
    """A `QueryResult` with `first_relevant_rank` filled the way `_score` fills it."""
    out = QueryResult(
        query="q",
        relevant=list(relevant),
        retrieved=list(retrieved),
        source=source,
        repo=repo,
    )
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


# --------------------------------------------------------------------------------
# Provenance: sources and repos that must not be silently pooled
# --------------------------------------------------------------------------------


def _gold_file(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_gold_file_without_a_source_loads_as_unspecified_rather_than_guessed():
    """`swarm_sync.json` predates the provenance schema and must keep working.

    It gets `UNSPECIFIED_SOURCE`, not `"handwritten"`. Guessing the provenance of
    someone else's labels is exactly how a bias stops being visible: a mined file
    mislabelled as hand-written would be averaged into the hand-written row and its
    vocabulary gap would be attributed to the retriever.
    """
    gold = load_gold_set("swarm_sync")
    assert len(gold) == 16
    assert gold.sources() == [goldset.UNSPECIFIED_SOURCE]
    assert gold.repos() == ["swarm-sync"]
    assert all(q.repo == "swarm-sync" for q in gold)
    assert all(q.query_id for q in gold), "every query needs a stable id to join on"


def test_provenance_is_carried_per_query_and_file_level_keys_are_only_defaults(tmp_path):
    """Per-query `source`/`repo` override the file's, because one file can legitimately
    hold both -- a mined set with three hand-corrected entries is still one artifact."""
    queries = goldset.parse_gold(
        {
            "repo": "swarm-sync",
            "source": "commit_prose",
            "queries": [
                {"query": "mined one", "relevant": ["a.b"]},
                {"query": "fixed one", "relevant": ["a.c"], "source": "handwritten"},
                {"query": "other repo", "relevant": ["a.d"], "repo": "code-learner"},
            ],
        },
        filename="x.json",
    )
    assert [q.source for q in queries] == ["commit_prose", "handwritten", "commit_prose"]
    assert [q.repo for q in queries] == ["swarm-sync", "swarm-sync", "code-learner"]


def test_loading_several_files_pools_them_but_keeps_them_distinguishable(tmp_path):
    _gold_file(tmp_path, "hand", {"repo": "r1", "source": "handwritten",
                                  "queries": [{"query": "h1", "relevant": ["a.b"]}]})
    _gold_file(tmp_path, "mined", {"repo": "r2", "source": "commit_prose",
                                   "queries": [{"query": "m1", "relevant": ["a.c"]}]})
    gold = load_gold_set(["hand", "mined"], gold_dir=tmp_path)
    assert len(gold) == 2
    assert gold.sources() == ["commit_prose", "handwritten"]
    assert gold.repos() == ["r1", "r2"]
    assert len(gold.subset(source="handwritten")) == 1
    assert len(gold.subset(repo="r2")) == 1
    assert len(gold.subset(source="handwritten", repo="r2")) == 0


def test_two_files_labelling_the_same_question_is_an_error_not_a_double_weight(tmp_path):
    """The same query in two files would count twice in every mean, weighting one
    question double for no reason a reader could see."""
    for name in ("a", "b"):
        _gold_file(tmp_path, name, {"repo": "r", "source": "s",
                                    "queries": [{"query": "same question", "relevant": ["a.b"]}]})
    with pytest.raises(goldset.GoldSchemaError, match="collide"):
        load_gold_set(["a", "b"], gold_dir=tmp_path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"queries": []}, "non-empty list"),
        ({"queries": [{"query": "q"}]}, "'relevant' must be a non-empty list"),
        ({"queries": [{"query": "q", "relevant": []}]}, "measures nothing"),
        ({"queries": [{"query": "", "relevant": ["a.b"]}]}, "'query' must be"),
        ({"queries": [{"query": "q", "relevant": ["a.b", "a.b"]}]}, "repeats a qualname"),
        ({"queries": [{"query": "q", "relevant": ["a.b"], "id": "x"},
                      {"query": "r", "relevant": ["a.c"], "id": "x"}]}, "duplicate query id"),
    ],
)
def test_a_malformed_gold_file_raises_rather_than_dropping_the_bad_entry(payload, match):
    """Skipping a malformed query would change `n`, and `n` is the denominator of every
    number the table prints -- including the quantum the caption quotes."""
    with pytest.raises(goldset.GoldSchemaError, match=match):
        goldset.parse_gold(payload, filename="x.json")


def test_asking_for_a_gold_file_that_does_not_exist_lists_the_ones_that_do(tmp_path):
    _gold_file(tmp_path, "real", {"queries": [{"query": "q", "relevant": ["a.b"]}]})
    with pytest.raises(goldset.GoldSchemaError, match=r"Available: \['real'\]"):
        load_gold_set("imaginary", gold_dir=tmp_path)


# --------------------------------------------------------------------------------
# The silent all-zeros failure -- a gold set that does not match the index
# --------------------------------------------------------------------------------


def _index(qualnames):
    """A connection with just enough schema for validation: the symbols table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE symbols (id INTEGER PRIMARY KEY, qualname TEXT)")
    conn.executemany("INSERT INTO symbols (qualname) VALUES (?)", [(q,) for q in qualnames])
    return conn


def _set(pairs, source="s", repo="r"):
    return goldset.GoldSet(
        name="t",
        queries=[
            GoldQuery(query=q, relevant=tuple(rel), source=source, repo=repo, query_id=q)
            for q, rel in pairs
        ],
    )


def test_gold_naming_symbols_the_index_lacks_raises_instead_of_scoring_zeros():
    """THE audit finding. `run_ablation` used to score whatever it was handed.

    A gold set whose symbols are absent from the index produced a full table of 0.000
    with `[0.000, 0.000]` intervals -- output shaped exactly like a finished
    measurement, containing none, and reading as *certainty* because a degenerate
    bootstrap interval is a point. No retriever can rank a symbol that is not indexed,
    so this was never a result about retrieval.
    """
    conn = _index(["pkg.mod.present"])
    gold = _set([("q1", ["pkg.mod.present"]), ("q2", ["pkg.mod.renamed_away"])])
    with pytest.raises(goldset.GoldIndexMismatch) as excinfo:
        goldset.validate_against_index(gold, conn)
    message = str(excinfo.value)
    assert "pkg.mod.renamed_away" in message, "the message must name what is missing"
    assert "pkg.mod.present" not in message, "and must not name what is fine"
    assert "1 of 2" in message
    assert "zeros" in message


def test_the_error_lists_the_missing_names_but_truncates_a_whole_stale_gold_file():
    """Enough to see the pattern -- a stale module prefix -- without printing 200 names."""
    conn = _index(["pkg.kept"])
    gold = _set([(f"q{i}", [f"old.pkg.gone{i}"]) for i in range(40)])
    with pytest.raises(goldset.GoldIndexMismatch) as excinfo:
        goldset.validate_against_index(gold, conn)
    assert "+28 more" in str(excinfo.value)
    assert len(excinfo.value.missing) == 40


def test_an_index_with_no_symbols_at_all_is_the_same_failure_not_a_pass():
    """An empty index matches nothing, which is not "the gold happens to be fine"."""
    with pytest.raises(goldset.GoldIndexMismatch, match="0 symbols"):
        goldset.validate_against_index(_set([("q", ["a.b"])]), _index([]))


def test_a_gold_set_that_matches_the_index_validates_silently():
    conn = _index(["a.b", "a.c", "a.d"])
    goldset.validate_against_index(_set([("q1", ["a.b", "a.c"]), ("q2", ["a.d"])]), conn)


def test_run_ablation_validates_BEFORE_it_scores_anything(tmp_path, monkeypatch):
    """The fix has to run first or it is only a nicer message on a wasted GPU pass.

    This index has a `symbols` table and no FTS table at all, so any attempt to
    retrieve would raise `OperationalError`. Getting `GoldIndexMismatch` proves
    validation happened before the first query was issued.
    """
    monkeypatch.setattr(ablation, "load_gold_set", lambda *a, **kw: _set([("q", ["not.in.index"])]))
    with pytest.raises(goldset.GoldIndexMismatch):
        ablation.run_ablation(_index(["a.b"]), embedder=None)


def test_validation_can_be_turned_off_but_is_on_by_default(monkeypatch):
    """Scoring a deliberately partial gold set is legitimate; doing it by accident is
    the failure. With `validate=False` the run proceeds and fails at retrieval instead,
    which is a loud error rather than a table of zeros."""
    monkeypatch.setattr(ablation, "load_gold_set", lambda *a, **kw: _set([("q", ["not.in.index"])]))
    with pytest.raises(sqlite3.OperationalError):
        ablation.run_ablation(_index(["a.b"]), embedder=None, validate=False)


# --------------------------------------------------------------------------------
# nDCG and MAP -- what MRR cannot see
# --------------------------------------------------------------------------------


def test_mrr_cannot_tell_two_rankings_apart_that_ndcg_and_map_can():
    """The concrete case for adding a rank-aware metric, worked by hand.

    Three relevant symbols. Config A returns them at ranks 1, 2, 3; config B returns
    the first at rank 1 and loses the other two entirely. MRR is 1.0 for BOTH -- it
    reads only the first relevant hit and is blind to everything after it.

    nDCG@10 for A: DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 =
    2.1309, which is also the ideal, so 1.0. For B: DCG = 1.0, ideal unchanged, so
    1/2.1309 = 0.4693.

    MAP@10 for A: (1/1 + 2/2 + 3/3)/3 = 1.0. For B: (1/1 + 0 + 0)/3 = 0.3333.
    """
    good = _result(["a", "b", "c"], ["a", "b", "c"])
    bad = _result(["a", "b", "c"], ["a", "x", "y", "z"])
    assert good.first_relevant_rank == bad.first_relevant_rank == 1
    assert Scorecard(name="g", results=[good]).mrr == Scorecard(name="b", results=[bad]).mrr

    assert good.ndcg_at(10) == pytest.approx(1.0)
    assert bad.ndcg_at(10) == pytest.approx(1.0 / 2.13092975, rel=1e-6)
    assert good.ap_at(10) == pytest.approx(1.0)
    assert bad.ap_at(10) == pytest.approx(1 / 3)


def test_ndcg_discounts_by_position_rather_than_only_counting():
    """Same two relevant symbols found, different order: recall@10 cannot tell them
    apart and nDCG can.

    Ideal DCG for two relevant symbols is 1/log2(2) + 1/log2(3) = 1 + 0.630930 =
    1.630930 in both cases. Found at ranks 1,2 that is exactly the DCG, so nDCG = 1.0.
    Found at ranks 3,4 the DCG is 1/log2(4) + 1/log2(5) = 0.500000 + 0.430677 =
    0.930677, so nDCG = 0.930677/1.630930 = 0.570642.
    """
    early = _result(["a", "b"], ["a", "b", "x", "y"])
    late = _result(["a", "b"], ["x", "y", "a", "b"])
    assert early.recall_at(10) == late.recall_at(10) == 1.0
    assert early.ndcg_at(10) == pytest.approx(1.0)
    assert late.ndcg_at(10) == pytest.approx(0.930677 / 1.630930, rel=1e-5)
    assert late.ndcg_at(10) < early.ndcg_at(10)


def test_on_a_single_relevant_query_ndcg_is_the_reciprocal_log_of_the_rank():
    """Which is why the single-relevant COUNT is the diagnostic that matters.

    With one relevant symbol the ideal DCG is 1, so nDCG collapses to 1/log2(rank+1)
    -- a monotone transform of MRR's 1/rank, ordering queries identically. On a set
    that is entirely single-relevant, nDCG buys nothing over MRR, and the count in the
    caption is what tells a reader that.
    """
    import math

    for rank in (1, 2, 3, 7):
        retrieved = ["x"] * (rank - 1) + ["a"]
        assert _result(["a"], retrieved).ndcg_at(10) == pytest.approx(1 / math.log2(rank + 1))
    assert _result(["a"], ["a"]).ndcg_at(10) == pytest.approx(1.0)


def test_ndcg_and_map_are_bounded_and_zero_when_nothing_relevant_came_back():
    miss = _result(["a", "b"], ["x", "y", "z"])
    assert miss.ndcg_at(10) == 0.0
    assert miss.ap_at(10) == 0.0
    assert _result([], ["a"]).ndcg_at(10) == 0.0
    assert _result([], ["a"]).ap_at(10) == 0.0
    assert _result(["a"], ["a"]).ndcg_at(0) == 0.0


def test_map_can_still_reach_one_when_there_are_more_relevant_symbols_than_k():
    """The denominator is `min(len(relevant), k)`. Dividing by `len(relevant)` would
    cap a query with 5 relevant symbols at 0.6 for k=3 no matter what any retriever
    did, which would make the metric report the gold set's shape as retriever error."""
    result = _result(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"])
    assert result.ap_at(3) == pytest.approx(1.0)
    assert result.ndcg_at(3) == pytest.approx(1.0)


def test_per_query_exposes_the_new_metrics_to_the_bootstrap():
    card = Scorecard(name="c")
    card.results.append(_result(["a", "b"], ["a", "b"]))
    card.results.append(_result(["c"], ["x", "c"]))
    assert card.per_query("ndcg", 10) == pytest.approx([1.0, 0.6309297535714575])
    assert card.per_query("map", 10) == pytest.approx([1.0, 0.5])
    assert card.ndcg_at(10) == pytest.approx((1.0 + 0.6309297535714575) / 2)
    assert card.map_at(10) == pytest.approx(0.75)


# --------------------------------------------------------------------------------
# Strata: per-source and per-repo rows
# --------------------------------------------------------------------------------


def _mixed_card():
    card = Scorecard(name="cfg")
    card.results = [
        _result(["a"], ["a"], source="handwritten", repo="r1"),
        _result(["b"], ["x"], source="handwritten", repo="r1"),
        _result(["c"], ["c"], source="commit_prose", repo="r2"),
        _result(["d"], ["x"], source="commit_prose", repo="r2"),
    ]
    return card


def test_stratify_emits_the_pooled_row_and_one_row_per_source_and_repo():
    rows = stratify(_mixed_card())
    scopes = [r.scope for r in rows]
    assert scopes[0] == ablation.POOLED
    assert set(scopes[1:]) == {
        "source=handwritten", "source=commit_prose", "repo=r1", "repo=r2",
    }
    assert all(r.name == "cfg" for r in rows), "the configuration name must survive"
    assert all(r.n == 2 for r in rows[1:])


def test_a_stratum_that_is_the_whole_set_is_not_re_emitted_as_if_it_corroborated():
    """One source and one repo means the per-stratum row IS the pooled row. Printing it
    twice under a more specific label reads as two agreeing measurements."""
    card = Scorecard(name="cfg")
    card.results = [_result(["a"], ["a"], source="handwritten", repo="r1")] * 3
    assert [r.scope for r in stratify(card)] == [ablation.POOLED]


def test_a_pooled_mean_moves_when_the_MIX_changes_and_not_only_the_retriever():
    """Why the pooled row is labelled POOLED rather than presented as the answer.

    Hand-written questions score 0.5 here and mined ones 0.0. The retriever is
    identical in both halves; adding two more mined queries drops the pooled number
    from 0.25 to 0.167 without anything about retrieval having changed.
    """
    card = _mixed_card()
    by_scope = {r.scope: r for r in stratify(card)}
    assert by_scope["source=handwritten"].hit_at(5) == pytest.approx(0.5)
    assert by_scope["source=commit_prose"].hit_at(5) == pytest.approx(0.5)

    skewed = Scorecard(name="cfg", results=list(card.results))
    skewed.results[2] = _result(["c"], ["x"], source="commit_prose", repo="r2")
    assert card.hit_at(5) == pytest.approx(0.5)
    assert skewed.hit_at(5) == pytest.approx(0.25)


# --------------------------------------------------------------------------------
# Clustering by repo
# --------------------------------------------------------------------------------


def test_the_cluster_bootstrap_resamples_repos_not_queries():
    """The unit of the draw is the repo, so a draw contains whole repos or none of them.

    Each repo arrives as a COMPLETE block, 0 or more times over -- never partially,
    which is what per-query resampling would do. Three repos of sizes 3, 2 and 1 are
    drawn three times with replacement, so a draw holds between 3 and 9 queries and its
    size varies, unlike the fixed-n per-query bootstrap.
    """
    labels = ["r1", "r1", "r1", "r2", "r2", "r3"]
    draws = ablation._resample_clusters(labels, 200, BOOTSTRAP_SEED)
    assert all(3 <= len(d) <= 9 for d in draws)
    assert len({len(d) for d in draws}) > 1, "cluster draws have varying size by design"
    for draw in draws[:50]:
        counts = {label: 0 for label in set(labels)}
        for i in draw:
            counts[labels[i]] += 1
        # each repo's queries arrive as a complete block, 0 or more times over
        assert counts["r1"] % 3 == 0
        assert counts["r2"] % 2 == 0


def test_clustered_draws_stay_a_pure_function_of_labels_and_seed_so_delta_ci_stays_paired():
    """The same guarantee `_resample_indices` gives, extended to clusters. A per-card
    RNG here would silently un-pair every multi-repo comparison while still printing
    intervals that looked right."""
    labels = ["r1", "r1", "r2", "r2"]
    first = ablation._resample_clusters(labels, 50, BOOTSTRAP_SEED)
    assert first == ablation._resample_clusters(labels, 50, BOOTSTRAP_SEED)
    assert first != ablation._resample_clusters(labels, 50, BOOTSTRAP_SEED + 1)


def test_clustering_with_one_repo_falls_back_rather_than_returning_certainty():
    """Resampling one cluster with replacement always redraws the same single cluster,
    giving a zero-width interval that reads as certainty. With nothing to cluster, the
    query is the right unit."""
    labels = ["r1"] * 8
    assert ablation._resample_clusters(labels, 20, 7) == ablation._resample_indices(8, 20, 7)

    card = Scorecard(name="c")
    card.results = [_result(["a"], ["a"] if i % 2 else ["x"], repo="r1") for i in range(8)]
    assert card.ci("hit", 5, cluster=True) == card.ci("hit", 5, cluster=False)
    assert card.ci("hit", 5, cluster=True)[0] < card.ci("hit", 5, cluster=True)[1]


def test_clustering_widens_the_interval_when_a_repo_moves_as_a_block():
    """The reason pooling across repos must cluster, made concrete.

    Two repos of eight queries each. Every query in r1 is a hit, every query in r2 is a
    miss -- the extreme of "errors correlate within a repo". Resampling queries treats
    that as sixteen independent coin flips and reports a narrow interval around 0.5.
    Resampling repos sees what it actually is: two observations, one 1.0 and one 0.0,
    so the interval must span nearly the whole range.
    """
    card = Scorecard(name="c")
    card.results = [_result(["a"], ["a"], repo="r1") for _ in range(8)]
    card.results += [_result(["b"], ["x"], repo="r2") for _ in range(8)]
    assert card.hit_at(5) == pytest.approx(0.5)

    naive_lo, naive_hi = card.ci("hit", 5, cluster=False)
    clustered_lo, clustered_hi = card.ci("hit", 5, cluster=True)
    assert (clustered_hi - clustered_lo) > (naive_hi - naive_lo), (
        "the per-query bootstrap understates the interval when a repo moves as a block, "
        "which manufactures significance -- the failure this whole exercise removes"
    )
    assert clustered_lo == pytest.approx(0.0) and clustered_hi == pytest.approx(1.0)


def test_design_effect_is_one_when_there_is_no_clustering_and_never_below_one():
    """A negative ICC means the data show no clustering, not that clustering buys
    precision. Returning <1 would let a sizing calculation ask for FEWER queries
    because of a structure it could not detect."""
    values = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    alternating = ["r1", "r2"] * 3
    assert design_effect(values, alternating) >= 1.0
    assert design_effect([1.0, 1.0], ["r1", "r1"]) == 1.0  # one cluster: nothing to measure

    blocked = ["r1", "r1", "r1", "r2", "r2", "r2"]
    assert design_effect([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], blocked) > 1.5


# --------------------------------------------------------------------------------
# Sizing: how many queries the enlarged set needs
# --------------------------------------------------------------------------------


def test_paired_sd_is_the_sd_of_the_per_query_DIFFERENCE_not_of_either_row():
    """The quantity that sizes the set. Two rows can each be highly variable across
    queries and still differ by a near-constant amount; it is the difference's spread
    that decides how many queries are needed, which is the same reason `delta_ci` is
    paired rather than two marginal intervals."""
    a = Scorecard(name="a", results=[_result(["x"], ["x"]), _result(["y"], ["q", "y"])])
    b = Scorecard(name="b", results=[_result(["x"], ["q", "x"]), _result(["y"], ["q", "q", "y"])])
    # per-query MRR: a = [1.0, 0.5], b = [0.5, 0.3333]; diffs = [0.5, 0.16667]
    assert a.per_query("mrr") == pytest.approx([1.0, 0.5])
    assert b.per_query("mrr") == pytest.approx([0.5, 1 / 3])
    # sd of [0.5, 0.166667] with ddof=1 is |0.5-0.166667|/sqrt(2) = 0.235702
    assert paired_sd(a, b, "mrr") == pytest.approx(0.2357022, rel=1e-5)
    assert paired_sd(a, a, "mrr") == 0.0, "a row against itself has no spread to size from"


def test_the_half_width_curve_falls_as_one_over_root_n():
    """The measured law. Quadrupling n halves the half-width; it does not quarter it,
    which is the arithmetic that makes 0.05 so much more expensive than 0.10."""
    sd = 0.322
    assert ci_half_width(sd, 16) == pytest.approx(1.959964 * sd / 4, rel=1e-6)
    assert ci_half_width(sd, 64) == pytest.approx(ci_half_width(sd, 16) / 2, rel=1e-9)
    assert ci_half_width(sd, 0) == float("inf")


def test_the_measured_constant_reproduces_the_empirical_curve_on_this_corpus():
    """`half_width(n) ~= 0.60/sqrt(n)` for MRR, the constant measured by subsampling
    the real 16 queries down to n=8 and resampling them up to n=512. The analytic
    `1.96 * 0.322 = 0.631` sits within 5% of it -- the percentile bootstrap runs
    slightly narrow at small n, which is the same bias the calibration floor is about.
    """
    for n in (16, 64, 128, 512):
        assert ci_half_width(0.322, n) == pytest.approx(0.631 / (n**0.5), rel=0.01)


def test_required_n_at_eighty_percent_power_is_roughly_double_the_fifty_percent_answer():
    """(1.96 + 0.84)^2 / 1.96^2 = 2.04. "The half-width is smaller than the effect" is
    a coin flip on whether any single run's interval excludes zero, and quoting it as
    the requirement is how a sizing exercise under-orders gold by half."""
    weak = required_n(0.322, 0.10, power=0.50)
    strong = required_n(0.322, 0.10, power=0.80)
    assert strong / weak == pytest.approx(2.04, rel=0.05)


def test_the_sizing_numbers_this_corpus_actually_produced():
    """Pinned so they cannot drift silently, and so the report can be checked.

    sd(diff) = 0.322 is the median over the 55 real paired comparisons available from
    the 16 queries under 11 model-free configurations; 0.267 is the same statistic for
    nDCG@10. This is the table the enlargement is being sized from.
    """
    assert required_n(0.322, 0.15, 0.50) == 18
    assert required_n(0.322, 0.10, 0.50) == 40
    assert required_n(0.322, 0.05, 0.50) == 160

    assert required_n(0.322, 0.15, 0.80) == 37
    assert required_n(0.322, 0.10, 0.80) == 82
    assert required_n(0.322, 0.05, 0.80) == 326

    # nDCG@10 is quieter per query on this corpus, so it needs ~30% fewer queries
    assert required_n(0.267, 0.15, 0.80) == 25
    assert required_n(0.267, 0.10, 0.80) == 56
    assert required_n(0.267, 0.05, 0.80) == 224
    assert required_n(0.267, 0.10, 0.80) / required_n(0.322, 0.10, 0.80) == pytest.approx(
        0.68, rel=0.05
    )


def test_required_n_refuses_to_answer_a_question_with_no_answer():
    assert required_n(0.322, 0.0) == 0
    assert required_n(0.0, 0.10) == 0, "a comparison with no per-query spread needs no n"


def test_the_power_curve_prints_the_sd_it_was_computed_from_and_its_assumption():
    """A required-n quoted without its sd is not checkable, and the sd is the only part
    of it that belongs to this corpus rather than to arithmetic."""
    text = power_curve(0.322)
    assert "0.322" in text
    assert "80% power" in text
    assert "50% power" in text
    assert "HARDER" in text, "the assumption a harder set violates must travel with it"


# --------------------------------------------------------------------------------
# The diagnostics that say what the metrics can and cannot separate
# --------------------------------------------------------------------------------


def test_the_caption_reports_how_many_queries_are_single_relevant():
    """THE quantity that made recall@k and hit@k duplicates. On a query with one
    relevant symbol recall@k can only be 0 or 1, which IS hit@k."""
    card = Scorecard(name="c")
    card.results = [
        _result(["a"], ["a"]),
        _result(["b"], ["x"]),
        _result(["c", "d"], ["c", "x"]),
        _result(["e", "f"], ["e", "f"]),
    ]
    assert card.single_relevant == 2
    text = format_table([card])
    assert "single-relevant queries: 2/4 (50%)" in text
    assert "mean relevant per query 1.50" in text
    assert "recall@k IS hit@k" in text


def test_an_entirely_single_relevant_set_is_told_that_its_metrics_are_one_metric():
    card = Scorecard(name="c", results=[_result(["a"], ["a"]), _result(["b"], ["x"])])
    assert card.single_relevant == card.n == 2
    text = format_table([card])
    assert "ALL queries are single-relevant" in text

    mixed = Scorecard(name="c", results=[_result(["a"], ["a"]), _result(["b", "c"], ["b"])])
    assert "ALL queries are single-relevant" not in format_table([mixed])


def test_the_old_sixteen_query_set_is_two_thirds_single_relevant_and_says_so():
    """The real number behind the finding, read off the real file.

    11 queries carry one relevant symbol, three carry two and two carry three:
    11 + 6 + 6 = 23 relevant symbols over 16 queries, so 1.4375 each. The set is
    nominally multi-relevant and effectively is not.
    """
    gold = load_gold_set("swarm_sync")
    assert len(gold) == 16
    assert gold.single_relevant() == 11
    assert sum(len(q.relevant) for q in gold) == 23
    assert gold.mean_relevant() == pytest.approx(23 / 16)


def test_pooling_across_sources_and_repos_is_flagged_in_the_caption():
    text = format_table([_mixed_card()])
    assert "sources: commit_prose, handwritten" in text
    assert "POOLED across sources" in text
    assert "read the per-source rows" in text
    assert "POOLED across 2 repos" in text
    assert "cluster=True" in text


def test_a_single_source_single_repo_table_is_not_warned_about_pooling():
    """A warning that fires on every table is a warning nobody reads."""
    card = Scorecard(name="c")
    card.results = [_result(["a"], ["a"], source="handwritten", repo="r1")] * 4
    text = format_table([card])
    assert "sources: handwritten" in text
    assert "POOLED across sources" not in text
    assert "POOLED across" not in text


def test_a_table_below_the_calibration_floor_says_its_intervals_are_not_95_percent():
    """Measured on this corpus: the percentile bootstrap's real false-positive rate is
    11.9% at n=16 against a nominal 5%, reaching 6.0% only near n=128. An interval
    that is narrower than it claims errs in the direction that invents findings, so
    the table has to say so where the numbers are."""
    assert CALIBRATION_FLOOR == 128
    small = Scorecard(name="c", results=[_result(["a"], ["a"])] * 16)
    assert "CALIBRATION" in format_table([small])
    assert "not 95% intervals" in format_table([small])

    big = Scorecard(name="c", results=[_result(["a"], ["a"])] * CALIBRATION_FLOOR)
    assert "CALIBRATION" not in format_table([big])


def test_the_table_still_carries_the_quantum_the_seed_and_the_resample_count():
    """The three facts WP13.4 put in the caption must survive the new columns."""
    card = Scorecard(name="lexical only", results=[_result(["a"], ["a"]), _result(["b"], ["x"])])
    text = format_table([card])
    assert "n = 2 queries" in text
    assert "0.5000" in text
    assert f"seed {BOOTSTRAP_SEED}" in text
    assert f"{BOOTSTRAP_RESAMPLES} resamples" in text
    assert "delta_ci" in text


def test_an_empty_table_formats_without_inventing_a_caption():
    assert "n = " not in format_table([])


# --------------------------------------------------------------------------------
# The per-source delta report
# --------------------------------------------------------------------------------


def _two_config_rows():
    """A baseline and a challenger that WINS on hand-written and LOSES on mined prose.

    Pooled they cancel to zero, which is exactly the result a single pooled row would
    report as "no difference" while hiding the most interesting thing in the data.
    """
    base = Scorecard(name="baseline")
    better = Scorecard(name="challenger")
    for _ in range(6):
        base.results.append(_result(["a"], ["x", "a"], source="handwritten", repo="r1"))
        better.results.append(_result(["a"], ["a"], source="handwritten", repo="r1"))
    for _ in range(6):
        base.results.append(_result(["b"], ["b"], source="commit_prose", repo="r1"))
        better.results.append(_result(["b"], ["x", "b"], source="commit_prose", repo="r1"))
    return base, better


def test_the_delta_report_shows_a_reversal_that_the_pooled_delta_cancels_to_zero():
    """The single most important thing a multi-source gold set can tell us, and the
    one thing a pooled mean is structurally unable to say."""
    base, better = _two_config_rows()
    assert better.mrr == pytest.approx(base.mrr), "pooled, the two rows are identical"

    rows = ablation.stratified_cards([base, better])
    text = format_delta_report(rows, baseline="baseline", metric="mrr", k=10)
    lines = [line for line in text.splitlines() if "challenger" in line]
    pooled = next(line for line in lines if line.startswith(ablation.POOLED))
    hand = next(line for line in lines if line.startswith("source=handwritten"))
    mined = next(line for line in lines if line.startswith("source=commit_prose"))
    assert "0.000" in pooled
    assert "0.500" in hand and "*" in hand, "the win must be visible and significant"
    assert "-0.500" in mined and "*" in mined, "and so must the loss"


def test_the_delta_report_marks_only_intervals_that_exclude_zero():
    base, better = _two_config_rows()
    text = format_delta_report([base, better], baseline="baseline", metric="mrr", k=10)
    pooled = next(line for line in text.splitlines() if "challenger" in line)
    assert "*" not in pooled, "a pooled delta of exactly zero cannot be significant"
    assert "* = interval excludes zero." in text


def test_the_delta_report_clusters_by_repo_when_the_stratum_spans_repos():
    base, better = _two_config_rows()
    for card in (base, better):
        for i, result in enumerate(card.results):
            result.repo = "r1" if i % 2 else "r2"
    text = format_delta_report([base, better], baseline="baseline", metric="mrr", k=10)
    assert "clustered by repo: 2 clusters" in text
    assert "fewer than 5, so this interval is unstable" in text


def test_the_delta_report_says_when_a_stratum_has_no_baseline_to_compare_against():
    """Silently omitting the stratum would make a table that looks complete and is
    missing a row nobody can see is missing."""
    base = Scorecard(name="baseline", scope="source=handwritten",
                     results=[_result(["a"], ["a"], source="handwritten")])
    orphan = Scorecard(name="challenger", scope="source=commit_prose",
                       results=[_result(["b"], ["b"], source="commit_prose")])
    text = format_delta_report([base, orphan], baseline="baseline")
    assert "no 'baseline' row in this stratum -- skipped" in text


def test_a_delta_report_below_the_calibration_floor_qualifies_its_own_stars():
    base, better = _two_config_rows()
    text = format_delta_report([base, better], baseline="baseline", metric="mrr", k=10)
    assert "is not a 5% claim" in text


# --------------------------------------------------------------------------------
# Multi-repo scoring: one index per repo
# --------------------------------------------------------------------------------


def _two_repo_gold():
    return goldset.GoldSet(
        name="t",
        queries=[
            GoldQuery(query="q1", relevant=("a.b",), source="hand", repo="r1", query_id="1"),
            GoldQuery(query="q2", relevant=("c.d",), source="hand", repo="r2", query_id="2"),
        ],
    )


def test_a_multi_repo_gold_set_needs_an_index_per_repo_and_says_which_is_missing(monkeypatch):
    """Scoring repo B's queries against repo A's index is the all-zeros bug wearing a
    different hat; dropping them instead would change `n` with nothing to see it."""
    monkeypatch.setattr(ablation, "load_gold_set", lambda *a, **kw: _two_repo_gold())
    with pytest.raises(goldset.GoldSchemaError, match=r"no index supplied for repo\(s\) \['r2'\]"):
        ablation.run_ablation_multi({"r1": _index(["a.b"])}, embedder=None)


def test_multi_repo_scoring_validates_each_repo_against_its_OWN_index(monkeypatch):
    """r1's gold is fine against r1's index; r2's is not against r2's. The error must
    be about r2, not about r1's names being absent from r2's index."""
    monkeypatch.setattr(ablation, "load_gold_set", lambda *a, **kw: _two_repo_gold())
    conns = {"r1": _index(["a.b"]), "r2": _index(["something.else"])}
    with pytest.raises(goldset.GoldIndexMismatch) as excinfo:
        ablation.run_ablation_multi(conns, embedder=None)
    assert "c.d" in str(excinfo.value)
    assert "a.b" not in str(excinfo.value), "r1's gold matched its own index and is not at fault"


def test_every_repo_is_validated_before_any_repo_is_scored(monkeypatch):
    """Validating lazily would spend a full GPU pass on repo 1 before discovering that
    repo 2's gold is stale. These indexes have a symbols table and no FTS table, so any
    retrieval raises OperationalError; getting GoldIndexMismatch proves nothing was
    scored before the last repo had been checked."""
    monkeypatch.setattr(ablation, "load_gold_set", lambda *a, **kw: _two_repo_gold())
    conns = {"r1": _index(["a.b"]), "r2": _index(["nope"])}
    with pytest.raises(goldset.GoldIndexMismatch):
        ablation.run_ablation_multi(conns, embedder=None)


def test_required_n_multiplies_by_the_design_effect_rather_than_ignoring_it():
    """Clustering is not a refinement at this corpus's numbers. The measured design
    effect over three repos is 5.4, so a pooled answer that ignores it under-orders
    gold by a factor of five."""
    assert required_n(0.267, 0.10, 0.80) == 56
    assert required_n(0.267, 0.10, 0.80, deff=5.44) == 305
    assert required_n(0.267, 0.10, 0.80, deff=0.5) == 56, "a deff below 1 cannot shrink n"


def test_design_effect_grows_with_cluster_SIZE_not_only_with_correlation():
    """`1 + (mean cluster size - 1) * ICC`. This is why a small ICC of ~0.026 still
    produced a design effect of 5.4: the clusters held ~173 queries each. It is also
    why adding queries inside an existing repo buys progressively less."""
    small = ["r1", "r1", "r2", "r2"]
    big = [f"r{i // 20 + 1}" for i in range(80)]
    values_small = [1.0, 0.9, 0.1, 0.0]
    values_big = [1.0 if i < 40 else 0.0 for i in range(80)]
    assert design_effect(values_big, big) > design_effect(values_small, small)
