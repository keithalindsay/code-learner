"""Graph expansion, RRF fusion, and the hybrid pipeline."""
from __future__ import annotations

import subprocess

import pytest

from codelearner.ingest import index_repo
from codelearner.ingest.indexer import is_test_path
from codelearner.retrieve import expand, reciprocal_rank_fusion, search, search_lexical
from codelearner.retrieve.fuse import (
    DEFAULT_WEIGHTS,
    RESERVED_TEST_SLOTS,
    RRF_K,
    TEST_DEMOTION_FACTOR,
)
from codelearner.retrieve.lexical import Hit


def _mkrepo(root, files: dict[str, str]):
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


def _hit(symbol_id: int, qualname: str, score: float = 1.0, modality: str = "lexical",
         is_test: bool = False, via: str = "") -> Hit:
    return Hit(
        symbol_id=symbol_id, qualname=qualname, kind="function", path="m.py",
        line_start=1, line_end=2, score=score, modality=modality, header="",
        is_test=is_test, via=via,
    )


# --------------------------------------------------------------------------
# test-path classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_leases.py", True),
        ("test/helpers.py", True),
        ("pkg/test_thing.py", True),
        ("pkg/thing_test.py", True),
        ("conftest.py", True),
        ("swarmsync/blackboard/leases.py", False),
        ("src/contest.py", False),          # not a test despite containing "test"
        ("latest/module.py", False),        # directory merely ends in "test"
    ],
)
def test_is_test_path_recognises_conventions(path, expected):
    assert is_test_path(path) is expected


def test_index_records_which_files_are_tests(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "src.py": "def impl():\n    return 1\n",
        "tests/test_src.py": "def test_impl():\n    return 1\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    flags = {r["path"]: r["is_test"] for r in conn.execute("SELECT path, is_test FROM files")}
    assert flags["src.py"] == 0
    assert flags["tests/test_src.py"] == 1


# --------------------------------------------------------------------------
# RRF fusion
# --------------------------------------------------------------------------

def test_rrf_prefers_consensus_over_a_single_first_place():
    """The property that makes RRF worth using: agreement across independent
    signals beats one confident vote. Ranked 3rd by three modalities (3/63) must
    beat ranked 1st by one (1/61)."""
    consensus = _hit(1, "agreed")
    solo = _hit(2, "solo")
    fused = reciprocal_rank_fusion(
        {
            "lexical": [solo, _hit(3, "x"), consensus],
            "dense": [_hit(4, "y"), _hit(5, "z"), consensus],
            "graph": [_hit(6, "w"), _hit(7, "v"), consensus],
        },
        k=5,
        weights={"lexical": 1.0, "dense": 1.0, "graph": 1.0},
    )
    assert fused[0].qualname == "agreed"


def test_rrf_uses_rank_not_score():
    """Scores across modalities are on incompatible scales (BM25 ~20, cosine ~0.7).
    A huge raw score at a worse rank must not win."""
    fused = reciprocal_rank_fusion(
        {"lexical": [_hit(1, "first", score=0.01), _hit(2, "second", score=9999.0)]},
        k=2,
    )
    assert [h.qualname for h in fused] == ["first", "second"]


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion({"lexical": [_hit(1, "only")]}, k=1,
                                   weights={"lexical": 1.0})
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_records_every_contributing_modality():
    fused = reciprocal_rank_fusion(
        {"lexical": [_hit(1, "both")], "dense": [_hit(1, "both", modality="dense")]},
        k=1,
    )
    assert fused[0].modality == "dense+lexical"


def test_rrf_keeps_the_graph_explanation_when_merging():
    """A graph hit carries a `via` account of why it surfaced. Merging must not
    discard the only explanation available for that symbol."""
    fused = reciprocal_rank_fusion(
        {
            "lexical": [_hit(1, "sym")],
            "graph": [_hit(1, "sym", modality="graph", via="calls tests.test_thing")],
        },
        k=1,
    )
    assert fused[0].via == "calls tests.test_thing"


def test_prefer_implementation_demotes_but_does_not_remove_tests():
    """A demotion, not a filter -- tests are often genuinely the best answer."""
    lists = {"lexical": [_hit(1, "a_test", is_test=True), _hit(2, "impl")]}
    plain = reciprocal_rank_fusion(lists, k=2)
    demoted = reciprocal_rank_fusion(lists, k=2, prefer_implementation=True)
    assert [h.qualname for h in plain] == ["a_test", "impl"]
    assert [h.qualname for h in demoted] == ["impl", "a_test"]
    assert len(demoted) == 2


# --------------------------------------------------------------------------
# the reserved test slot
#
# The demotion above is a multiplicative factor on an RRF score, so what it DOES
# depends on how many modalities voted for the symbol rather than on the policy: a
# test with two votes is reordered by it, a test with one vote is erased by it. These
# tests pin the floor that makes the outcome a property of the ranking instead.
# --------------------------------------------------------------------------

def _one_vote_test_against(n_impls: int, n_tests: int = 1) -> dict[str, list[Hit]]:
    """The situation that turns the demotion into a filter.

    Tests ranked FIRST by lexical and absent from dense -- which is what happens
    whenever tests are outside the embedding corpus, including every index built with
    no embedder at all. Every implementation they compete with still has two votes, so
    halving a test's single vote drops it below all of them.

    The first test is always `the_test`, so a caller reserving one slot knows which
    symbol the floor is supposed to save.
    """
    tests = [_hit(1, "the_test", is_test=True)] + [
        _hit(1 + i, f"other_test{i}", is_test=True) for i in range(1, n_tests)
    ]
    return {
        "lexical": tests + [_hit(10 + i, f"impl{i}") for i in range(n_impls)],
        "dense": [_hit(10 + i, f"impl{i}") for i in range(n_impls)],
    }


def test_the_factor_alone_erases_a_test_that_lost_a_modality():
    """The defect the reserve exists for, stated as a test rather than as a comment.

    Measured consequence on the 123-query test-seeking gold: in the no-embedder
    configuration -- which `search()` supports on purpose -- the factor takes swarm-sync
    from nDCG@10 0.413 undemoted to 0.023, and hit@10 to 0.041. Not out of the top ten:
    out of the top FORTY, so the cross-encoder is never shown one either.
    """
    lists = _one_vote_test_against(12)
    erased = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=0
    )
    assert "the_test" not in [h.qualname for h in erased]


def test_a_reserved_slot_keeps_a_demoted_test_reachable():
    lists = _one_vote_test_against(12)
    fused = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=1
    )
    names = [h.qualname for h in fused]
    assert "the_test" in names
    assert len(fused) == 10


def test_reserved_tests_enter_at_the_bottom_not_the_top():
    """Present, not dominant. The demotion still decides everything above them, so a
    reserve costs the rank-1 answer nothing -- which is why it measured free."""
    lists = _one_vote_test_against(12, n_tests=2)
    fused = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=2
    )
    names = [h.qualname for h in fused]
    assert names[:8] == [f"impl{i}" for i in range(8)]
    assert [h.is_test for h in fused[-2:]] == [True, True]


def test_the_reserve_promotes_the_best_available_test():
    lists = {
        "lexical": [
            _hit(1, "best_test", is_test=True),
            _hit(2, "worse_test", is_test=True),
        ]
        + [_hit(10 + i, f"impl{i}") for i in range(12)],
        "dense": [_hit(10 + i, f"impl{i}") for i in range(12)],
    }
    fused = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=1
    )
    names = [h.qualname for h in fused]
    assert "best_test" in names
    assert "worse_test" not in names


def test_the_reserve_is_a_floor_and_never_a_quota():
    """A run where enough tests already survived the demotion is returned untouched.

    This is the whole reason the reserve measured at +0.000 on the gold set: on the
    shipping index the demotion is already only a nudge, so the floor almost never
    binds and cannot displace an answer that was ranking on merit.
    """
    lists = {
        "lexical": [_hit(1, "t1", is_test=True), _hit(2, "t2", is_test=True)],
        "dense": [_hit(1, "t1", is_test=True), _hit(2, "t2", is_test=True)],
    }
    with_reserve = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=2
    )
    without = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=0
    )
    assert [h.qualname for h in with_reserve] == [h.qualname for h in without]


def test_the_reserve_does_nothing_when_the_demotion_is_off():
    """Nothing to undo. Reserving slots against a ranking that never penalised tests
    would be injecting them, not restoring them."""
    lists = _one_vote_test_against(12)
    assert [
        h.qualname
        for h in reciprocal_rank_fusion(lists, k=10, reserved_test_slots=5)
    ] == [h.qualname for h in reciprocal_rank_fusion(lists, k=10, reserved_test_slots=0)]


def test_the_reserve_never_claims_more_than_half_the_results():
    """`search(k=1)` asks for the single best answer; a floor that answered it with a
    test would be a ceiling. Inactive at the k=10 and k=40 the value was measured at."""
    lists = _one_vote_test_against(12, n_tests=3)
    top1 = reciprocal_rank_fusion(
        lists, k=1, prefer_implementation=True, reserved_test_slots=2
    )
    assert [h.is_test for h in top1] == [False]
    top2 = reciprocal_rank_fusion(
        lists, k=2, prefer_implementation=True, reserved_test_slots=2
    )
    assert sum(h.is_test for h in top2) == 1
    top10 = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=2
    )
    assert sum(h.is_test for h in top10) == 2


def test_the_reserve_cannot_promote_a_test_that_does_not_exist():
    lists = {"lexical": [_hit(10 + i, f"impl{i}") for i in range(12)]}
    fused = reciprocal_rank_fusion(
        lists, k=10, prefer_implementation=True, reserved_test_slots=3
    )
    assert len(fused) == 10
    assert not any(h.is_test for h in fused)


def test_the_reserve_returns_k_results_without_duplicates():
    lists = _one_vote_test_against(12)
    for reserved in range(0, 6):
        fused = reciprocal_rank_fusion(
            lists, k=10, prefer_implementation=True, reserved_test_slots=reserved
        )
        ids = [h.symbol_id for h in fused]
        assert len(ids) == 10
        assert len(set(ids)) == 10


def test_reserved_test_slots_default_is_the_measured_value():
    """REGRESSION on two measured constants, like the graph weight below.

    Both swept on 170 implementation queries AND 123 test-seeking queries (nDCG@10,
    paired bootstrap, 2000 resamples, seed 20250801), per repo.

    The reserve is 2 because that is the largest floor that is free on the
    implementation gold: reserve-2-minus-reserve-0 is [+0.000,+0.000] on all three
    repos, displacing a relevant symbol on 0 of the 106 queries whose top 10 it
    changes, while test-seeking gains +0.019 [+0.002,+0.038] on swarm-sync. 3 costs
    -0.005 [-0.011,-0.001]. Raising it should break a test, not quietly cost recall.

    The factor is 0.5 because moving it is a TRADE, not an improvement, and nobody has
    measured the query mix that would settle it: 0.6 gives back 0.034 implementation
    nDCG on swarm-sync to buy 0.155 test-seeking nDCG. Changing it should force whoever
    does so to look at both columns in `fuse.TEST_DEMOTION_FACTOR`.
    """
    assert RESERVED_TEST_SLOTS == 2
    assert TEST_DEMOTION_FACTOR == 0.5


def test_search_reserves_slots_without_an_embedder(tmp_path):
    """End to end on the configuration where the factor is a filter TODAY.

    No embedder means tests get one vote and implementations get one vote too -- but
    the demotion halves only the test's, so with enough implementations around a test
    cannot place. This is a supported configuration, not a hypothetical one.
    """
    root = _mkrepo(tmp_path / "repo", {
        "impl.py": "".join(
            f"def reclaim_lease_{i}():\n    '''reclaim an expired lease'''\n    return {i}\n\n"
            for i in range(12)
        ),
        "tests/test_impl.py": (
            "def test_reclaim_lease_expired():\n"
            "    '''reclaim an expired lease'''\n"
            "    return 1\n"
        ),
    })
    conn, _ = index_repo(root, index_path=tmp_path / "i.db")
    query = "reclaim an expired lease"
    without = search(conn, query, k=6, embedder=None, reserved_test_slots=0)
    with_floor = search(conn, query, k=6, embedder=None, reserved_test_slots=2)
    assert not any(h.is_test for h in without.hits)
    assert any(h.is_test for h in with_floor.hits)
    assert len(with_floor.hits) == len(without.hits)
    # The floor changes only the tail: everything the demotion ranked above the
    # reserved slots is untouched.
    assert [h.qualname for h in with_floor.hits[:4]] == [
        h.qualname for h in without.hits[:4]
    ]


def test_graph_weight_default_is_the_measured_value():
    """REGRESSION on a measured constant. The first guess of 0.6 made the whole
    hybrid score WORSE than lexical+dense alone (0.385 vs 0.573 recall@5); the
    sweep showed monotonic decline above 0.3. Changing this without re-running
    the ablation should break a test, not silently degrade retrieval."""
    assert DEFAULT_WEIGHTS["graph"] == 0.3
    assert DEFAULT_WEIGHTS["lexical"] == DEFAULT_WEIGHTS["dense"] == 1.0


# --------------------------------------------------------------------------
# graph expansion
# --------------------------------------------------------------------------

GRAPH_REPO = {
    "impl.py": "def reclaim():\n    return 1\n",
    "tests/test_impl.py": (
        "from impl import reclaim\n\n"
        "def test_reclaim_expires():\n    return reclaim()\n\n"
        "def test_reclaim_again():\n    return reclaim()\n"
    ),
}


def test_expansion_reaches_the_implementation_from_its_tests(tmp_path):
    """The mechanism this modality exists for: a test is an edge to its subject.
    Text retrieval ranks tests above implementations because tests describe
    behaviour in words; the call graph points back at the code."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    seeds = [h for h in search_lexical(conn, "reclaim expires", k=5) if h.is_test]
    assert seeds, "expected the test to be found by text search"

    found = expand(conn, seeds, k=5)
    assert "impl.reclaim" in {h.qualname for h in found}


def test_expansion_explains_how_it_got_there(tmp_path):
    """Retrieval that cannot say why it returned something is hard to trust."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    seeds = [h for h in search_lexical(conn, "reclaim", k=5) if h.is_test]
    hit = next(h for h in expand(conn, seeds, k=5) if h.qualname == "impl.reclaim")
    assert hit.via.startswith("calls ")
    assert hit.modality == "graph"


def test_expansion_excludes_its_own_seeds(tmp_path):
    """Seeds are already in the fusion input. Returning them lets one modality
    vote twice for the same symbol."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    seeds = search_lexical(conn, "reclaim", k=5)
    seed_ids = {h.symbol_id for h in seeds}
    assert not ({h.symbol_id for h in expand(conn, seeds, k=10)} & seed_ids)


def test_expansion_accumulates_across_multiple_seeds(tmp_path):
    """Convergence is the signal: something several seeds point at should outrank
    something reached from only one.

    The seed ORDER here is doing deliberate work. Seed activation falls off by rank
    (1/(1+rank)), so the single seed pointing at `lonely` is placed FIRST -- giving
    it the strongest possible individual contribution -- while the three pointing at
    `shared` are ranked below it. `shared` can then only win by summing:

        lonely  = 1.00 * decay                        = 0.450
        shared  = (0.50 + 0.33 + 0.25) * decay        = 0.487

    An earlier version of this test seeded in natural search order and passed even
    when `+=` was replaced with `max()` -- it was measuring seed rank, not
    accumulation. Mutation-checked: replacing `+=` with `max()` now fails it."""
    repo = _mkrepo(tmp_path / "r", {
        "impl.py": "def shared():\n    return 1\n\ndef lonely():\n    return 2\n",
        "tests/test_a.py": "from impl import shared\n\ndef test_a():\n    return shared()\n",
        "tests/test_b.py": "from impl import shared\n\ndef test_b():\n    return shared()\n",
        "tests/test_d.py": "from impl import shared\n\ndef test_d():\n    return shared()\n",
        "tests/test_c.py": "from impl import lonely\n\ndef test_c():\n    return lonely()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    by_qualname = {
        r["qualname"]: r["id"]
        for r in conn.execute("SELECT id, qualname FROM symbols")
    }
    order = [
        "tests.test_c.test_c",   # -> lonely, ranked first (strongest single seed)
        "tests.test_a.test_a",   # -> shared
        "tests.test_b.test_b",   # -> shared
        "tests.test_d.test_d",   # -> shared
    ]
    seeds = [_hit(by_qualname[q], q, is_test=True) for q in order]

    scores = {h.qualname: h.score for h in expand(conn, seeds, k=10)}
    assert scores["impl.shared"] > scores["impl.lonely"]


def test_expansion_of_nothing_is_nothing(tmp_path):
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert expand(conn, [], k=5) == []


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

def test_search_works_without_an_embedder(tmp_path):
    """An index built without embeddings must still answer, not raise."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    result = search(conn, "reclaim", k=5)
    assert result.hits
    assert "dense" not in result.per_modality


def test_search_reports_each_modality_separately(tmp_path):
    """Phase 8's ablation needs per-modality candidates, and a result that cannot
    show its working cannot be improved."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    result = search(conn, "reclaim", k=5)
    assert "lexical" in result.per_modality
    assert "graph" in result.per_modality


def test_every_modality_can_be_switched_off(tmp_path):
    """Not a convenience: without this the ablation cannot isolate a modality."""
    repo = _mkrepo(tmp_path / "r", GRAPH_REPO)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    lexical_only = search(conn, "reclaim", k=5, use_graph=False)
    assert set(lexical_only.per_modality) == {"lexical"}
    assert search(conn, "reclaim", k=5, use_lexical=False, use_graph=False).hits == []
