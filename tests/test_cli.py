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
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from codelearner import db
from codelearner.assertions import store
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


def _indexed(
    tmp_path: Path, capsys, embed: bool = False, files: dict[str, str] = REPO_FILES
) -> tuple[Path, Path]:
    """A repo with an index, returned as (repo, index_path).

    Drains capsys on the way out. Otherwise the indexer's own report is still in
    the buffer when the test under examination reads it, and every `--json`
    assertion tries to parse a table with a JSON document stapled to the end.
    """
    repo = _mkrepo(tmp_path / "repo", files)
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


# ---------------------------------------------------------------------------
# --force and the tier-2 store
# ---------------------------------------------------------------------------
#
# `--force` used to unlink the DB file and call it "discards its embeddings". Every
# test in this section names something that was destroyed by that sentence and is
# not re-derivable from source: a verdict, a rejection, an expiry event, the claim
# itself. The schema's `ON DELETE SET NULL` on `subject_symbol_id` was written for
# exactly this moment and had never once executed, because nothing deleted a symbol
# row -- the whole file went instead.


def _git_add(repo: Path) -> None:
    """Re-stage the repo, because `iter_python_files` asks git what is tracked."""
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)  # noqa: S603, S607


def _admit(
    index_path: Path,
    repo: Path,
    subject: str,
    claim: str,
    *,
    cite: str | None = None,
) -> int:
    """Admit one claim through the real gate. Returns its id.

    `cite` defaults to the subject; passing a different qualname is how a test
    separates "the subject vanished" from "the evidence moved", which the carry path
    has to answer differently.
    """
    conn = db.connect(index_path)
    try:
        subject_row = conn.execute(
            "SELECT id FROM symbols WHERE qualname = ?", (subject,)
        ).fetchone()
        assert subject_row is not None, f"fixture has no symbol {subject!r}"
        cited = conn.execute(
            "SELECT f.path, s.byte_start, s.byte_end FROM symbols s "
            "JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
            (cite or subject,),
        ).fetchone()
        assert cited is not None, f"fixture has no symbol {cite!r}"
        span = store.span_for(repo, cited["path"], cited["byte_start"], cited["byte_end"])
        return store.write_assertion(
            conn,
            subject_qualname=subject,
            kind="purpose",
            claim=claim,
            spans=[span],
            subject_symbol_id=subject_row["id"],
            generator="test-agent/v1",
            confidence=0.7,
            repo_root=repo,
        )
    finally:
        conn.close()


def _rows(index_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = db.connect(index_path, check_schema=False)
    try:
        return list(conn.execute(sql, params))
    finally:
        conn.close()


def test_force_refuses_a_non_empty_tier_2_store_and_names_the_counts(tmp_path, capsys):
    """The counts are the argument. "Discards its embeddings" is true and beside the
    point -- embeddings are minutes of GPU time, and a verdict is a judgement that was
    made once and cannot be made again from source."""
    repo, index_path = _indexed(tmp_path, capsys)
    first = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    second = _admit(index_path, repo, "core._plumbing", "returns the answer")
    third = _admit(index_path, repo, "core", "the module")
    conn = db.connect(index_path)
    store.record_verdict(conn, second, "judge/v1", "refuted", "it does not")
    store.mark_stale(conn, third, store.REASON_HASH_MISMATCH)
    conn.close()

    assert main(["index", str(repo), "--force"], embedder_factory=fake_factory) == 1
    err = capsys.readouterr().err
    assert "3 assertions" in err
    assert "1 verdict" in err
    assert "1 staleness event" in err
    assert "--carry-assertions" in err
    assert "--discard-assertions" in err
    assert "Traceback" not in err
    # Refused means untouched, not partially applied.
    assert [r["id"] for r in _rows(index_path, "SELECT id FROM assertions ORDER BY id")] == [
        first,
        second,
        third,
    ]


def test_carry_assertions_preserves_the_store_and_re_resolves_its_subjects(tmp_path, capsys):
    """`subject_qualname` is `NOT NULL` precisely so a rebuild can re-find the symbol
    after every row id in the file has been replaced. A new file shifts those ids, so
    a carry that merely copied the old integer would come back pointing at the wrong
    symbol -- or at a row that no longer exists."""
    repo, index_path = _indexed(tmp_path, capsys)
    admitted = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    before = _rows(index_path, "SELECT * FROM assertions WHERE id = ?", (admitted,))[0]

    # Sorts before core.py in `git ls-files`, so every symbol id after it moves.
    (repo / "aardvark.py").write_text("def dig():\n    return 1\n\n\ndef burrow():\n    return 2\n")
    _git_add(repo)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    after = _rows(index_path, "SELECT * FROM assertions ORDER BY id")
    assert [r["id"] for r in after] == [admitted]
    row = after[0]
    assert row["claim"] == before["claim"]
    assert row["status"] == "active"
    assert row["kind"] == before["kind"]
    assert row["generator"] == before["generator"]
    assert row["confidence"] == before["confidence"]
    # A rebuilt `created_at` would turn "we served that for three months" into "we
    # wrote it today".
    assert row["created_at"] == before["created_at"]

    symbol = _rows(
        index_path, "SELECT id FROM symbols WHERE qualname = 'core.frobnicate_widgets'"
    )[0]
    assert symbol["id"] != before["subject_symbol_id"], "fixture failed to move the ids"
    assert row["subject_symbol_id"] == symbol["id"]

    spans = _rows(index_path, "SELECT * FROM evidence_spans WHERE assertion_id = ? ORDER BY id", (admitted,))
    assert len(spans) == 1
    assert spans[0]["path"] == "core.py"
    assert payload["tier2"]["assertions"] == 1
    assert payload["tier2"]["subjects_resolved"] == 1
    assert payload["tier2"]["expired_by_rebuild"] == 0


def test_a_carried_claim_whose_evidence_moved_comes_back_stale_with_a_log_row(tmp_path, capsys):
    """The honest outcome, and the one the staleness engine exists for. Deleting the
    claim would destroy the record of what was believed and why it stopped being
    true; keeping it active would serve a claim about bytes that are gone."""
    repo, index_path = _indexed(tmp_path, capsys)
    admitted = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")

    (repo / "core.py").write_text(
        REPO_FILES["core.py"].replace("every widget on the tray", "nothing whatsoever")
    )
    _git_add(repo)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    row = _rows(index_path, "SELECT * FROM assertions ORDER BY id")[0]
    assert row["id"] == admitted
    assert row["status"] == "stale"
    assert row["claim"] == "frobnicates widgets"
    events = _rows(index_path, "SELECT * FROM staleness_log WHERE assertion_id = ? ORDER BY id", (admitted,))
    assert [e["reason"] for e in events] == ["hash_mismatch"]
    assert events[0]["expected_hash"] and events[0]["observed_hash"]
    assert payload["tier2"]["expired_by_rebuild"] == 1


def test_a_carried_claim_whose_subject_vanished_keeps_a_null_link(tmp_path, capsys):
    """The case `ON DELETE SET NULL` was written for, finally reachable. The claim
    outlives the symbol it was about: the name is durable, the id was only ever a
    convenience, and dropping the row would delete evidence of what the repository
    used to contain."""
    repo, index_path = _indexed(
        tmp_path,
        capsys,
        files={
            **REPO_FILES,
            "helper.py": 'def only_here():\n    """Doomed."""\n    return 1\n',
        },
    )
    # Subject in the file about to disappear, evidence in the file that stays: this
    # separates "the subject vanished" from "the evidence moved".
    admitted = _admit(
        index_path, repo, "helper.only_here", "helps", cite="core.frobnicate_widgets"
    )
    (repo / "helper.py").unlink()
    _git_add(repo)

    assert main(
        ["index", str(repo), "--force", "--carry-assertions", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    row = _rows(index_path, "SELECT * FROM assertions ORDER BY id")[0]
    assert row["id"] == admitted
    assert row["subject_qualname"] == "helper.only_here"
    assert row["subject_symbol_id"] is None
    # Its evidence never moved, so nothing about it went stale.
    assert row["status"] == "active"
    assert payload["tier2"]["subjects_unresolved"] == 1
    assert payload["tier2"]["subjects_resolved"] == 0


def test_verdicts_and_the_rejected_set_survive_a_rebuild(tmp_path, capsys):
    """The rejected set is the only evidence the gate does anything. A rebuild that
    took it away would leave the pass rate free to be whatever the last run says."""
    repo, index_path = _indexed(tmp_path, capsys)
    good = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    bad = _admit(index_path, repo, "core._plumbing", "reticulates splines")
    conn = db.connect(index_path)
    store.record_verdict(conn, good, "judge/v1", "supported", "the span says so")
    store.record_verdict(conn, bad, "judge/v1", "refuted", "it returns 42")
    conn.close()

    assert main(
        ["index", str(repo), "--force", "--carry-assertions"], embedder_factory=fake_factory
    ) == 0
    capsys.readouterr()

    conn = db.connect(index_path)
    try:
        assert [a.id for a in store.assertions_with_status(conn, store.STATUS_REJECTED)] == [bad]
        assert [a.id for a in store.assertions_with_status(conn, store.STATUS_ACTIVE)] == [good]
        verdicts = store.verdicts_for(conn, bad)
        assert [v["verdict"] for v in verdicts] == ["refuted"]
        assert verdicts[0]["judge"] == "judge/v1"
        assert verdicts[0]["rationale"] == "it returns 42"
    finally:
        conn.close()


def test_discard_assertions_is_the_only_way_to_lose_the_store(tmp_path, capsys):
    """Destruction stays possible and stays deliberate. What changed is that it now
    has to be typed out."""
    repo, index_path = _indexed(tmp_path, capsys)
    _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")

    assert main(
        ["index", str(repo), "--force", "--discard-assertions", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier2"] is None
    assert _rows(index_path, "SELECT * FROM assertions ORDER BY id") == []


def test_carry_and_discard_cannot_both_be_asked_for():
    """Two opposite answers to one question, so a command line asserting both is one
    whose author believed something untrue about at least one of them. argparse
    refuses it as a usage error -- a 2, not the 1 that means the world was wrong."""
    parser = build_parser()
    assert parser.parse_args(["index", "r", "--force", "--carry-assertions"]).carry_assertions
    assert parser.parse_args(["index", "r", "--force", "--discard-assertions"]).discard_assertions
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            ["index", "r", "--force", "--carry-assertions", "--discard-assertions"]
        )
    assert excinfo.value.code == 2


def test_a_crash_after_the_delete_leaves_the_store_on_disk_and_recoverable(
    tmp_path, capsys, monkeypatch
):
    """The failure mode the whole package exists for. `index_repo` raising after
    `_delete_index` used to end with the index gone and the store gone with it -- and
    an in-memory carry would do exactly the same, because the interpreter is what
    died. The sidecar outlives the process, and the plain no-flags `index` an operator
    reaches for next is what puts the store back."""
    from codelearner.cli import commands

    repo, index_path = _indexed(tmp_path, capsys)
    admitted = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    carry = commands.carry_path(index_path)

    def die(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(commands, "index_repo", die)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions"], embedder_factory=fake_factory
    ) == 1
    capsys.readouterr()
    assert not index_path.exists(), "the delete is what makes this the interesting case"
    assert carry.exists(), "the store died with the process"

    monkeypatch.undo()
    assert main(["index", str(repo), "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier2"]["recovered"] is True
    assert payload["tier2"]["assertions"] == 1
    row = _rows(index_path, "SELECT * FROM assertions ORDER BY id")[0]
    assert row["id"] == admitted
    assert row["claim"] == "frobnicates widgets"
    assert row["subject_symbol_id"] is not None
    # Removed only once the rows are back, so the window in which a second crash
    # loses everything does not exist.
    assert not carry.exists()


def test_a_crash_during_the_restore_is_recovered_from_the_sidecar(tmp_path, capsys, monkeypatch):
    """The other half of the window. The index is back and the store is not, which
    looks from the outside like a successful rebuild -- so the sidecar is removed
    only after the restore has committed, and until then it outranks whatever the
    rebuilt file holds. `index_repo` never writes an assertion, which is what makes
    the sidecar the superset rather than merely another opinion."""
    from codelearner.cli import commands

    repo, index_path = _indexed(tmp_path, capsys)
    admitted = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    carry = commands.carry_path(index_path)

    def die(*args, **kwargs):
        raise sqlite3.OperationalError("killed mid-restore")

    monkeypatch.setattr(commands, "_restore_store", die)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions"], embedder_factory=fake_factory
    ) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert index_path.exists()
    assert carry.exists(), "removed before the rows were back"
    assert _rows(index_path, "SELECT * FROM assertions ORDER BY id") == []

    monkeypatch.undo()
    assert main(["index", str(repo), "--force", "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier2"]["recovered"] is True
    assert [r["id"] for r in _rows(index_path, "SELECT id FROM assertions ORDER BY id")] == [
        admitted
    ]
    assert not carry.exists()


def test_a_carry_file_from_another_repo_is_refused_rather_than_restored(tmp_path, capsys):
    """A carry file names claims by qualname and by repo-relative path. Restoring one
    into an index of a different tree would attach real citations to unrelated bytes,
    and every later verification would report that as an edit nobody made."""
    from codelearner.cli import commands

    repo, index_path = _indexed(tmp_path, capsys)
    _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    dump = commands._dump_store(index_path)

    other = _mkrepo(tmp_path / "other")
    assert main(["index", str(other)], embedder_factory=fake_factory) == 0
    capsys.readouterr()
    other_index = other / INDEX_RELPATH
    commands._write_carry_file(
        commands.carry_path(other_index), dump, repo=repo, index_path=index_path
    )

    assert main(["index", str(other), "--force"], embedder_factory=fake_factory) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert str(repo) in err
    assert commands.carry_path(other_index).exists(), "a refusal must not consume it"


# ---------------------------------------------------------------------------
# a schema stamp this code cannot read -- one line, never a traceback
# ---------------------------------------------------------------------------

def _stamp_schema(index_path: Path, version: str) -> None:
    conn = db.connect(index_path, check_schema=False)
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (version,))
    conn.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["stats"],
        ["search", "frobnicate"],
        ["learn"],
    ],
)
def test_an_index_from_another_schema_is_one_line_not_a_traceback(tmp_path, capsys, argv):
    """`db.connect` refuses a stale stamp on every READ, which is the point -- and
    `SchemaVersionError` is a RuntimeError, so catching only `sqlite3.Error` sent the
    single most predicted failure in this design out as a traceback. The stamp has
    moved five times."""
    repo, index_path = _indexed(tmp_path, capsys)
    _stamp_schema(index_path, "4")

    assert main([*argv, "--repo", str(repo)], embedder_factory=fake_factory) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err + captured.out
    assert "v4" in captured.err
    assert "--carry-assertions" in captured.err


def test_the_remedy_for_a_schema_mismatch_no_longer_costs_the_assertions(tmp_path, capsys):
    """The two work packages meet here. `--force` is the documented remedy for a
    stamp this code cannot read, and until the store could survive a rebuild that
    remedy WAS the data loss -- the upgrade path and the destruction path were the
    same command. The dump therefore has to read a database whose version check would
    refuse it, which is why it opens with `check_schema=False`."""
    repo, index_path = _indexed(tmp_path, capsys)
    admitted = _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    _stamp_schema(index_path, "4")

    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 1
    assert "--carry-assertions" in capsys.readouterr().err

    assert main(
        ["index", str(repo), "--force", "--carry-assertions"], embedder_factory=fake_factory
    ) == 0
    capsys.readouterr()
    assert [r["id"] for r in _rows(index_path, "SELECT id FROM assertions ORDER BY id")] == [
        admitted
    ]
    stamp = _rows(index_path, "SELECT value FROM meta WHERE key = 'schema_version'")[0]
    assert stamp["value"] == str(db.SCHEMA_VERSION)
    # And the read that refused a moment ago now answers.
    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 0


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


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------


class FakeClaimGenerator:
    """A `ClaimGenerator` that cites ref 1 and nothing else.

    Stands in for ollama for the same reason `FakeEmbedder` stands in for
    `Qwen3-Embedding-0.6B`: no test in this repo may call a model, and the wiring
    being checked here -- that the command reaches the pipeline, that refusals reach
    the report, that an outage becomes one line instead of a traceback -- is not
    wiring a model has anything to say about.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.released = False

    @property
    def name(self) -> str:
        return "fake/claims"

    def draft(self, *, subject, offered):
        from codelearner.generate.types import Draft

        return Draft(claim=f"{subject} does the thing.", cited_refs=(1,), kind="purpose")

    def release(self) -> None:
        self.released = True


def _patch_generator(monkeypatch, cls) -> None:
    """Swap the generator the command builds.

    `cmd_learn` imports `OllamaClaimGenerator` inside the function body, so patching
    the attribute on the package is enough and no injection seam is needed. That
    import is deliberate for a second reason: `codelearner search` must not pay the
    import cost of the generation stack.
    """
    import codelearner.generate as generate

    monkeypatch.setattr(generate, "OllamaClaimGenerator", cls)


def test_learn_admits_claims_and_reports_what_it_refused(tmp_path, capsys, monkeypatch):
    """The happy path, and the shape of its report.

    `admitted` alone cannot be the headline: a generator that cites whatever is in
    front of it produces the same number as one that understood the repo. So the
    refusal counters have to survive into the output, and this pins that they do."""
    _patch_generator(monkeypatch, FakeClaimGenerator)
    repo, _ = _indexed(tmp_path, capsys)

    assert main(["learn", "--repo", str(repo)], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "admitted" in out


def test_learn_json_carries_every_counter(tmp_path, capsys, monkeypatch):
    """`--json` is the machine surface, so a counter missing from it is a counter that
    silently stops being auditable. The refusal breakdown is the point of the document,
    not an appendix to it."""
    _patch_generator(monkeypatch, FakeClaimGenerator)
    repo, _ = _indexed(tmp_path, capsys)

    assert main(["learn", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    doc = json.loads(capsys.readouterr().out)
    for key in (
        "generator",
        "considered",
        "drafts_requested",
        "admitted",
        "refused_empty_claim",
        "refused_no_citation",
        "invalid_refs",
        "generator_errors",
    ):
        assert key in doc, key
    assert doc["generator"] == "fake/claims"


def test_learn_without_an_index_says_so_instead_of_creating_one(tmp_path, monkeypatch, capsys):
    """`db.connect` will happily make an empty file at any path, which is how a typo
    becomes "0 symbols considered" rather than an error. Same rule as every other
    read-side command."""
    _patch_generator(monkeypatch, FakeClaimGenerator)
    empty = tmp_path / "not-a-repo"
    empty.mkdir()

    assert main(["learn", "--repo", str(empty)], embedder_factory=fake_factory) == 1
    assert "no index" in capsys.readouterr().err


def test_a_backend_outage_is_one_line_and_not_a_traceback(tmp_path, capsys, monkeypatch):
    """The failure this command will actually hit: ollama is not running.

    It must not surface as a stack trace, and -- more importantly -- must not surface
    as a completed run with nothing admitted, which is what absorbing the exception
    would produce and what a reader would mistake for a model that had nothing to say.
    """

    class Down(FakeClaimGenerator):
        def draft(self, *, subject, offered):
            from codelearner.generate.types import GeneratorUnavailable

            raise GeneratorUnavailable("could not reach the generator at http://x (down)")

    _patch_generator(monkeypatch, Down)
    repo, _ = _indexed(tmp_path, capsys)

    assert main(["learn", "--repo", str(repo)], embedder_factory=fake_factory) == 1
    captured = capsys.readouterr()
    assert "could not reach the generator" in captured.err
    assert "Traceback" not in captured.err
    assert "admitted" not in captured.out


def test_a_second_run_skips_what_the_first_admitted(tmp_path, capsys, monkeypatch):
    """The store never deletes, so a re-run that re-drafted everything would double it
    permanently and re-weight every rate computed over it afterwards."""
    _patch_generator(monkeypatch, FakeClaimGenerator)
    repo, _ = _indexed(tmp_path, capsys)

    assert main(["learn", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["learn", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["admitted"] > 0
    assert second["skipped_existing"] == first["admitted"]
    assert second["drafts_requested"] == 0
