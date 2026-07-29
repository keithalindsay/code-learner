"""Graph expansion, RRF fusion, and the hybrid pipeline."""
from __future__ import annotations

import subprocess

import pytest

from codelearner.ingest import index_repo
from codelearner.ingest.indexer import is_test_path
from codelearner.retrieve import expand, reciprocal_rank_fusion, search, search_lexical
from codelearner.retrieve.fuse import DEFAULT_WEIGHTS, RRF_K
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
