"""The CLI surface: argument parsing, `--json` shapes, and failing without a traceback.

Every test here is written so that deleting the rule it names makes it fail. Several
of the rules are about what happens when something is *missing* -- no index, no
embeddings, no text modality -- because those are the paths a user actually hits and
the ones a happy-path test suite never covers.

No test loads a real embedding model. `main` takes an embedder factory for exactly
this reason: `Qwen3-Embedding-0.6B` costs tens of seconds and ~1.2GB of VRAM that
another process may be holding, and a fake returning three floats proves the wiring
just as well.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from codelearner.cli import build_parser, main
from codelearner.cli.commands import INDEX_RELPATH, resolve_index_path
from codelearner.cli.render import facts_only, tier_of
from codelearner.retrieve import Hit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Deterministic, dependency-free `Embedder`.

    Defined here rather than imported from `test_embed` on purpose: that module
    skips itself entirely when sqlite-vec is absent, and importing it would drag
    this file's parsing and error-handling tests into that skip for no reason.
    """

    MARKERS = ("frobnicate", "widget", "plumbing")

    def __init__(self, name: str = "fake/v1") -> None:
        self._name = name

    @property
    def dim(self) -> int:
        return len(self.MARKERS)

    @property
    def name(self) -> str:
        return self._name

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [float(lowered.count(m)) for m in self.MARKERS]
        norm = sum(v * v for v in raw) ** 0.5
        return [v / norm for v in raw] if norm else [0.0] * len(raw)

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


def fake_factory(model_name: str) -> FakeEmbedder:
    """Stands in for the real model loader; echoes back the name it was asked for."""
    return FakeEmbedder(model_name)


REPO_FILES = {
    # `frobnicate_widgets` is reachable by text. `_plumbing` deliberately shares no
    # vocabulary with the query, so the ONLY route to it is the call edge -- which
    # is what makes the --no-graph test mean something.
    "core.py": (
        'def frobnicate_widgets():\n'
        '    """Frobnicate every widget on the tray."""\n'
        '    return _plumbing()\n'
        '\n'
        '\n'
        'def _plumbing():\n'
        '    """Detail."""\n'
        '    return 42\n'
    ),
}

QUERY = "frobnicate widgets"


def _mkrepo(root: Path, files: dict[str, str] = REPO_FILES) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


def _indexed(tmp_path: Path, capsys, embed: bool = False) -> tuple[Path, Path]:
    """A repo with an index, returned as (repo, index_path).

    Drains capsys on the way out. Otherwise the indexer's own report is still in
    the buffer when the test under examination reads it, and every `--json`
    assertion tries to parse a table with a JSON document stapled to the end.
    """
    repo = _mkrepo(tmp_path / "repo")
    argv = ["index", str(repo)]
    if embed:
        argv += ["--embed", "--model", "fake/v1"]
    assert main(argv, embedder_factory=fake_factory) == 0
    capsys.readouterr()
    return repo, repo / INDEX_RELPATH


def _search_json(capsys, argv: list[str], factory=fake_factory) -> dict:
    code = main(argv, embedder_factory=factory)
    out = capsys.readouterr().out
    assert code == 0
    return json.loads(out)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def test_index_writes_the_default_path_inside_the_repo(tmp_path, capsys):
    """The default must agree with `index_repo`'s own default. One file per repo is
    what makes cross-repo contamination structurally impossible; a CLI that invented
    a second location would quietly undo that."""
    repo = _mkrepo(tmp_path / "repo")
    assert main(["index", str(repo)], embedder_factory=fake_factory) == 0
    assert (repo / ".codelearner" / "index.db").exists()
    assert resolve_index_path(repo, None) == repo / ".codelearner" / "index.db"
    assert "indexed" in capsys.readouterr().out


def test_index_path_override_is_honoured(tmp_path):
    repo = _mkrepo(tmp_path / "repo")
    elsewhere = tmp_path / "other" / "custom.db"
    assert main(
        ["index", str(repo), "--index-path", str(elsewhere)], embedder_factory=fake_factory
    ) == 0
    assert elsewhere.exists()
    assert not (repo / ".codelearner" / "index.db").exists()


def test_index_json_reports_every_count_and_the_resolution_rate(tmp_path, capsys):
    repo = _mkrepo(tmp_path / "repo")
    assert main(["index", str(repo), "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == 1
    assert payload["symbols"] == 3  # module + two functions
    assert payload["edges"] >= 1
    assert payload["chunks"] == 3
    # The honest denominator: resolution measured against in-repo references only.
    assert payload["resolution"]["rate_of_internal"] == pytest.approx(1.0)
    assert payload["resolution"]["by_resolver"]
    assert payload["embeddings"] is None


def test_reindexing_refuses_rather_than_clobbering(tmp_path, capsys):
    """There is no incremental update, so a second index would hit a UNIQUE
    constraint on files.path. Refusing with a remedy beats an IntegrityError
    traceback -- and beats silently deleting embeddings that cost minutes."""
    repo, _ = _indexed(tmp_path, capsys)
    assert main(["index", str(repo)], embedder_factory=fake_factory) == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    assert "--force" in err
    assert "Traceback" not in err


def test_force_rebuilds_an_existing_index(tmp_path, capsys):
    repo, index_path = _indexed(tmp_path, capsys)
    before = index_path.stat().st_mtime_ns
    assert main(["index", str(repo), "--force"], embedder_factory=fake_factory) == 0
    assert index_path.exists()
    assert index_path.stat().st_mtime_ns != before


def test_indexing_a_missing_directory_explains_itself(tmp_path, capsys):
    missing = tmp_path / "nope"
    assert main(["index", str(missing)], embedder_factory=fake_factory) == 1
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_embed_uses_the_injected_embedder_and_records_the_model(tmp_path, capsys):
    pytest.importorskip("sqlite_vec", reason="embedding storage requires sqlite-vec")
    repo = _mkrepo(tmp_path / "repo")
    assert main(
        ["index", str(repo), "--embed", "--model", "fake/v1", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["embeddings"]["model"] == "fake/v1"
    assert payload["embeddings"]["embedded"] == payload["chunks"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_shows_qualname_and_file_line(tmp_path, capsys):
    repo, _ = _indexed(tmp_path, capsys)
    assert main(["search", QUERY, "--repo", str(repo)], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "core.frobnicate_widgets" in out
    assert "core.py:1-3" in out  # location is a place in a file, not just a file


def test_search_json_is_one_parseable_document_with_a_stable_hit_shape(tmp_path, capsys):
    repo, index_path = _indexed(tmp_path, capsys)
    payload = _search_json(capsys, ["search", QUERY, "--repo", str(repo), "--json"])
    assert payload["query"] == QUERY
    assert payload["index"] == str(index_path)
    assert payload["count"] == len(payload["hits"])
    assert set(payload["modalities"]) == {"lexical", "dense", "graph"}
    assert set(payload["hits"][0]) == {
        "rank", "tier", "tier_n", "symbol_id", "qualname", "kind", "path",
        "line_start", "line_end", "score", "modality", "is_test", "via",
    }
    assert payload["hits"][0]["rank"] == 1
    assert payload["hits"][0]["tier"] == "T0"


def test_k_limits_the_result_count(tmp_path, capsys):
    repo, _ = _indexed(tmp_path, capsys)
    payload = _search_json(capsys, ["search", QUERY, "--repo", str(repo), "--k", "1", "--json"])
    assert payload["count"] == 1


def test_k_below_one_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["search", QUERY, "--k", "0"], embedder_factory=fake_factory)
    assert exc.value.code == 2


def test_no_graph_removes_the_symbols_only_the_graph_could_reach(tmp_path, capsys):
    """`_plumbing` shares no vocabulary with the query; its only route is the call
    edge. With graph expansion on it appears WITH a `via`; with --no-graph it is gone
    entirely. A test that only checked the flag parsed would pass with the switch
    unwired."""
    repo, _ = _indexed(tmp_path, capsys)
    with_graph = _search_json(capsys, ["search", QUERY, "--repo", str(repo), "--json"])
    names = [h["qualname"] for h in with_graph["hits"]]
    assert "core._plumbing" in names
    reached = next(h for h in with_graph["hits"] if h["qualname"] == "core._plumbing")
    assert reached["via"] == "calls core.frobnicate_widgets"
    assert reached["tier"] == "T1"  # reached through a RESOLVED edge, not parsed text

    without = _search_json(
        capsys, ["search", QUERY, "--repo", str(repo), "--no-graph", "--json"]
    )
    assert without["modalities"]["graph"] is False
    assert "core._plumbing" not in [h["qualname"] for h in without["hits"]]
    assert all(h["via"] == "" for h in without["hits"])


def test_search_without_embeddings_degrades_instead_of_failing(tmp_path, capsys):
    """An index built without the embedding step must still answer. Dense is
    reported unavailable on stderr so stdout stays a clean document, and the message
    says how to fix it."""
    repo, _ = _indexed(tmp_path, capsys)
    code = main(["search", QUERY, "--repo", str(repo), "--json"], embedder_factory=fake_factory)
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)  # stdout is JSON and nothing else
    assert payload["modalities"]["dense"] is False
    assert payload["count"] > 0
    assert "no embeddings" in captured.err
    assert "--embed" in captured.err


def test_no_dense_never_touches_the_embedder(tmp_path, capsys):
    """--no-dense must short-circuit BEFORE model construction, not merely drop the
    results afterwards. Loading a model costs tens of seconds and 1.2GB of VRAM, so
    "it still returns the right hits" is not evidence the flag did its job.

    The index here deliberately HAS embeddings: against an index without them the
    embedder is never built anyway, and the test would pass with the flag unwired."""
    pytest.importorskip("sqlite_vec", reason="an embedded index requires sqlite-vec")
    repo, _ = _indexed(tmp_path, capsys, embed=True)

    def exploding_factory(model_name: str):
        raise AssertionError(f"the embedder was built despite --no-dense: {model_name}")

    payload = _search_json(
        capsys,
        ["search", QUERY, "--repo", str(repo), "--no-dense", "--json"],
        factory=exploding_factory,
    )
    assert payload["modalities"]["dense"] is False


def test_search_with_no_text_modality_is_an_error_not_an_empty_result(tmp_path, capsys):
    """Graph expansion has no query representation -- it is seeded by the text
    modalities. With lexical off and dense unavailable it would return nothing for
    every query, forever, silently. That is a failure and must exit non-zero."""
    repo, _ = _indexed(tmp_path, capsys)
    code = main(
        ["search", QUERY, "--repo", str(repo), "--no-lexical"], embedder_factory=fake_factory
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "nothing to search with" in captured.err
    assert "Traceback" not in captured.err


def test_search_on_a_missing_index_exits_nonzero_with_a_remedy(tmp_path, capsys):
    """`db.connect` will happily create an empty database at any path, which turns a
    typo into '0 results' instead of an error. The existence check is what stops
    that, so the message must name the path and the command that builds one."""
    repo = _mkrepo(tmp_path / "repo")
    code = main(["search", QUERY, "--repo", str(repo)], embedder_factory=fake_factory)
    captured = capsys.readouterr()
    assert code == 1
    assert str(repo / INDEX_RELPATH) in captured.err
    assert "codelearner index" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_a_query_matching_nothing_is_not_an_error(tmp_path, capsys):
    repo, _ = _indexed(tmp_path, capsys)
    code = main(
        ["search", "zzzqqq nonexistent", "--repo", str(repo)], embedder_factory=fake_factory
    )
    assert code == 0
    assert "no results" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dense, faked
# ---------------------------------------------------------------------------

def test_dense_runs_when_the_index_has_matching_vectors(tmp_path, capsys):
    pytest.importorskip("sqlite_vec", reason="dense retrieval requires sqlite-vec")
    repo, _ = _indexed(tmp_path, capsys, embed=True)
    payload = _search_json(capsys, ["search", QUERY, "--repo", str(repo), "--json"])
    assert payload["modalities"]["dense"] is True
    assert any("dense" in h["modality"] for h in payload["hits"])


def test_dense_is_disabled_when_the_model_does_not_match_the_vectors(tmp_path, capsys):
    """Vectors from two models are not comparable. Querying anyway returns results
    that look plausible and mean nothing -- strictly worse than returning none."""
    pytest.importorskip("sqlite_vec", reason="dense retrieval requires sqlite-vec")
    repo, _ = _indexed(tmp_path, capsys, embed=True)
    code = main(
        ["search", QUERY, "--repo", str(repo), "--json"],
        embedder_factory=lambda _name: FakeEmbedder("some-other-model/v9"),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["modalities"]["dense"] is False
    assert "not comparable" in captured.err


# ---------------------------------------------------------------------------
# tiers and --facts-only
# ---------------------------------------------------------------------------

def _hit(modality: str) -> Hit:
    return Hit(
        symbol_id=1, qualname="a.b", kind="function", path="a.py",
        line_start=1, line_end=2, score=1.0, modality=modality, header="",
    )


def test_tier_follows_the_strongest_evidence_that_reached_the_hit():
    """Text match is T0 -- nothing had to be decided. Graph is T1, because expansion
    only walks RESOLVED edges and a resolution can be wrong. A symbol found by both
    is T0: the text modality reached it without needing the binding."""
    assert tier_of(_hit("lexical")) == 0
    assert tier_of(_hit("dense")) == 0
    assert tier_of(_hit("graph")) == 1
    assert tier_of(_hit("dense+graph")) == 0


def test_an_unrecognised_modality_is_treated_as_inferred():
    """--facts-only is a promise about provenance, so it fails closed. An unknown
    source is exactly what a caller asking for facts is trying to exclude."""
    assert tier_of(_hit("mystery")) == 2
    assert tier_of(_hit("")) == 2
    assert facts_only([_hit("mystery")]) == []


def test_facts_only_drops_inferred_hits_and_keeps_t0_and_t1():
    fact, resolved, inferred = _hit("lexical"), _hit("graph"), _hit("inferred")
    assert facts_only([fact, resolved, inferred]) == [fact, resolved]


def test_facts_only_flag_reaches_the_filter(tmp_path, capsys):
    repo, _ = _indexed(tmp_path, capsys)
    payload = _search_json(
        capsys, ["search", QUERY, "--repo", str(repo), "--facts-only", "--json"]
    )
    assert payload["facts_only"] is True
    assert all(h["tier_n"] <= 1 for h in payload["hits"])


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_reports_tiers_kinds_resolvers_and_absent_embeddings(tmp_path, capsys):
    repo, _ = _indexed(tmp_path, capsys)
    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "T0 FACT" in out and "T1 RESOLVED" in out and "T2 INFERRED" in out
    assert "function" in out and "module" in out
    assert "module_local/v1" in out  # resolution broken down by resolver
    assert "none." in out  # embeddings absent, said plainly


def test_stats_json_shape(tmp_path, capsys):
    repo, index_path = _indexed(tmp_path, capsys)
    assert main(["stats", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["index"] == str(index_path)
    assert payload["repo_root"] == str(repo)
    assert payload["counts"] == {"files": 1, "symbols": 3, "edges": payload["counts"]["edges"],
                                 "chunks": 3}
    assert set(payload["tiers"]) == {"T0", "T1", "T2"}
    assert payload["tiers"]["T2"] == 0  # the inference layer is not built yet
    assert payload["symbol_kinds"]["function"] == 2
    assert payload["resolution"]["by_resolver"]["module_local/v1"]["count"] == 1
    assert payload["resolution"]["by_resolver"]["module_local/v1"]["confidence"] == pytest.approx(0.9)
    assert payload["embeddings"] == {"present": False, "model": None, "dim": None, "vectors": 0}


def test_stats_reports_which_model_produced_the_vectors(tmp_path, capsys):
    """Which model matters as much as whether: vectors from two models are not
    comparable, so 'has embeddings' alone is not enough to decide anything."""
    pytest.importorskip("sqlite_vec", reason="dense retrieval requires sqlite-vec")
    repo, _ = _indexed(tmp_path, capsys, embed=True)
    assert main(["stats", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    embeddings = json.loads(capsys.readouterr().out)["embeddings"]
    assert embeddings["present"] is True
    assert embeddings["model"] == "fake/v1"
    assert embeddings["dim"] == 3
    assert embeddings["vectors"] == 3


def test_stats_on_a_missing_index_exits_nonzero(tmp_path, capsys):
    repo = _mkrepo(tmp_path / "repo")
    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 1
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# parsing and packaging
# ---------------------------------------------------------------------------

def test_every_command_is_registered():
    parser = build_parser()
    for argv in (["index", "."], ["search", "q"], ["stats"]):
        assert parser.parse_args(argv).command == argv[0]


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_console_script_is_registered():
    """Requirement, not decoration: without this entry point `codelearner` is not a
    command, only a module."""
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["codelearner"] == "codelearner.cli:main"


def test_module_entry_point_runs_out_of_the_source_tree():
    """`python -m codelearner.cli` must work before anything is pip-installed, which
    is the state every checkout starts in."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "codelearner.cli", "--help"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "index" in proc.stdout and "search" in proc.stdout and "stats" in proc.stdout
