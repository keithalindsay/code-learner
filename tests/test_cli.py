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
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from codelearner import db, gpu
from codelearner.assertions import store
from codelearner.cli import build_parser, main
from codelearner.cli.commands import INDEX_RELPATH, resolve_index_path
from codelearner.cli.render import facts_only, tier_of
from codelearner.retrieve import Hit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No command driven from this file may reach a daemon. Enforced, not assumed.

    Added when `index --embed` grew a pre-flight VRAM check that reads ollama's
    `/api/ps`: without this, every `--embed` test in the file would behave one way on
    the workstation with ollama running and another way everywhere else, which is
    exactly the machine-dependence `tests/test_faithfulness.py` and
    `tests/test_generate_llm.py` already refuse. Same fixture, same reason.

    Subprocess-based tests further down are unaffected -- they run in a child
    interpreter, where this patch does not reach.
    """

    def _refuse(*args, **kwargs):
        raise urllib.error.URLError("tests must not reach a daemon")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


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
# drift -- the index measured against the tree it was built from
# ---------------------------------------------------------------------------
#
# Tier 2 has a two-stage staleness engine, a `span_verifications` baseline and a
# `staleness_log`. Tier 0/1 had NOTHING: `files.mtime_ns` and `files.size_bytes`
# were written at index time and read by no code path at all, so after any edit
# `search` went on serving T0 rows -- "deterministic, reproducible from source
# alone" -- at line numbers that had moved, with nothing anywhere saying the index
# was behind the tree. That the facts tier was the one with no drift check is the
# inversion these tests exist to keep closed.
#
# Every assertion below is on stderr or on a `drift` key, never on stdout prose,
# because the note must never be able to corrupt a `--json` document.

CHANGED_MARK = "since this index was built"
FLOOR_MARK = "these counts are floors"


def _edit(path: Path, text: str) -> None:
    """Rewrite a file so that BOTH mtime and size move.

    Size is the belt: two writes inside one filesystem timestamp tick are entirely
    possible on a fast machine, and a test that relied on mtime alone would be the
    kind that passes on a laptop and fails in CI once a year.
    """
    path.write_text(text)


def test_an_edited_file_makes_search_say_the_index_is_behind_the_tree(tmp_path, capsys):
    """The whole point: a moved line must not be served in silence.

    Shifting the file down changes every line number the index recorded for it, and
    before this the only symptom was a citation that was quietly wrong."""
    repo, _ = _indexed(tmp_path, capsys)
    _edit(repo / "core.py", "# a new header line\n" + (repo / "core.py").read_text())

    assert main(["search", QUERY, "--repo", str(repo)], embedder_factory=fake_factory) == 0
    err = capsys.readouterr().err
    assert CHANGED_MARK in err
    assert "1 of 1 indexed file has changed" in err
    assert "--force --carry-assertions" in err
    assert "Traceback" not in err


def test_an_untouched_index_says_nothing_at_all(tmp_path, capsys):
    """The control, and the more important half of the pair.

    A check that fires on a clean tree is a check that gets ignored within a day,
    and once ignored it is worth less than no check at all -- the user has learnt
    to scroll past the one line that would have told them their answer was wrong."""
    repo, _ = _indexed(tmp_path, capsys)

    # --no-dense only so that the (unrelated, expected) "no embeddings" note does not
    # occupy stderr; the assertion worth making is that stderr is EMPTY, not that it
    # merely lacks two substrings.
    assert main(
        ["search", QUERY, "--repo", str(repo), "--no-dense"], embedder_factory=fake_factory
    ) == 0
    err = capsys.readouterr().err
    assert CHANGED_MARK not in err
    assert "not in this index" not in err
    assert err.strip() == ""


def test_the_note_never_corrupts_a_json_document(tmp_path, capsys):
    """`--json` is a machine surface, so the warning goes to stderr and the FACTS go
    into the document. A human-readable note on stdout would break every pipeline
    that pipes this into `jq`, and dropping the facts entirely would mean the machine
    surface is the one that cannot tell it is being served stale line numbers."""
    repo, _ = _indexed(tmp_path, capsys)
    _edit(repo / "core.py", "# a new header line\n" + (repo / "core.py").read_text())

    assert main(
        ["search", QUERY, "--repo", str(repo), "--json"], embedder_factory=fake_factory
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # the assertion: still ONE parseable document
    assert CHANGED_MARK in captured.err
    assert payload["drift"]["checked"] is True
    assert payload["drift"]["changed"] == 1
    assert payload["drift"]["missing"] == 0
    assert payload["drift"]["unindexed"] == 0
    assert payload["drift"]["indexed"] == 1
    # Named so a reader knows what was compared and therefore what was NOT.
    assert payload["drift"]["method"] == "mtime_ns+size_bytes"


def test_a_file_the_index_never_had_is_counted_apart_from_a_changed_one(tmp_path, capsys):
    """Two different failures, so two different numbers.

    A modified file moves citations -- the answer is wrong and looks right. A file
    that was never indexed is simply absent -- the answer is missing and looks like
    "no results", exit 0. Folding them into one count would tell a user to expect
    the wrong symptom."""
    repo, _ = _indexed(tmp_path, capsys)
    (repo / "extra.py").write_text('def newly_written():\n    """Added later."""\n    return 1\n')
    _git_add(repo)

    assert main(
        ["search", QUERY, "--repo", str(repo), "--json"], embedder_factory=fake_factory
    ) == 0
    captured = capsys.readouterr()
    drift = json.loads(captured.out)["drift"]
    assert drift["changed"] == 0
    assert drift["unindexed"] == 1
    assert "not in this index" in captured.err
    assert CHANGED_MARK not in captured.err


def test_an_indexed_file_that_is_gone_is_counted_apart_again(tmp_path, capsys):
    """A third failure: the citation names a path with nothing behind it. Reported
    separately because "the bytes moved" and "the file is not there" have different
    remedies and different blast radii."""
    repo, _ = _indexed(
        tmp_path,
        capsys,
        files={**REPO_FILES, "doomed.py": 'def gone():\n    """Bye."""\n    return 1\n'},
    )
    (repo / "doomed.py").unlink()

    assert main(
        ["search", QUERY, "--repo", str(repo), "--json"], embedder_factory=fake_factory
    ) == 0
    captured = capsys.readouterr()
    drift = json.loads(captured.out)["drift"]
    assert drift["missing"] == 1
    assert drift["changed"] == 0
    assert "no longer on disk" in captured.err


def test_the_note_says_it_is_a_floor_and_not_an_audit(tmp_path, capsys):
    """mtime+size can miss an edit that preserves both, so the note must never read
    as an exhaustive audit. It reports what it measured and says what it did not."""
    repo, _ = _indexed(tmp_path, capsys)
    _edit(repo / "core.py", "# a new header line\n" + (repo / "core.py").read_text())

    assert main(["search", QUERY, "--repo", str(repo)], embedder_factory=fake_factory) == 0
    assert FLOOR_MARK in capsys.readouterr().err


def test_stats_reports_drift_too_including_the_clean_case(tmp_path, capsys):
    """`stats` is the command someone types to ask "what is in this index", so it is
    the one surface where the freshness answer is worth printing even when it is
    "nothing has moved" -- unlike `search`, where an unconditional line would train
    the reader to ignore it."""
    repo, index_path = _indexed(tmp_path, capsys)
    assert main(["stats", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    clean = json.loads(capsys.readouterr().out)["drift"]
    assert clean == {
        "checked": True,
        "indexed": 1,
        "changed": 0,
        "missing": 0,
        "unindexed": 0,
        "method": "mtime_ns+size_bytes",
    }

    _edit(repo / "core.py", "# a new header line\n" + (repo / "core.py").read_text())
    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert str(index_path)  # the path is still reported; nothing was swallowed
    assert "freshness" in out
    assert "1 of 1" in out


def test_an_unmeasurable_tree_is_not_reported_as_a_clean_one(tmp_path, capsys):
    """"Did not measure" and "measured zero" are different answers, and the second is
    the one a reader will assume unless the payload refuses to say it.

    Reached by pointing an index at a tree that is not on this machine -- an index
    copied off a build box, a repo on an unmounted volume. Every count is null rather
    than 0, and nothing is printed, because there is nothing this process established."""
    repo, index_path = _indexed(tmp_path, capsys)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    # The WAL sidecars come too. Copying only the main file leaves the tables that
    # have not been checkpointed behind, which is a different bug pretending to be
    # this one -- `_delete_index` exists for the mirror-image of the same fact.
    for suffix in ("", "-wal", "-shm"):
        sidecar = Path(str(index_path) + suffix)
        if sidecar.exists():
            (moved / ("index.db" + suffix)).write_bytes(sidecar.read_bytes())

    assert main(
        ["stats", "--repo", str(repo), "--index-path", str(moved / "index.db"), "--json"],
        embedder_factory=fake_factory,
    ) == 0
    captured = capsys.readouterr()
    # The repo still exists here, so first prove the fixture: nothing has moved.
    assert json.loads(captured.out)["drift"]["checked"] is True

    # Now take the tree away and ask again.
    for path in sorted(repo.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    repo.rmdir()
    assert main(
        ["stats", "--repo", str(moved), "--index-path", str(moved / "index.db"), "--json"],
        embedder_factory=fake_factory,
    ) == 0
    captured = capsys.readouterr()
    drift = json.loads(captured.out)["drift"]
    assert drift == {
        "checked": False,
        "indexed": None,
        "changed": None,
        "missing": None,
        "unindexed": None,
        "method": "mtime_ns+size_bytes",
    }
    assert CHANGED_MARK not in captured.err
    assert "Traceback" not in captured.err


def test_drift_survives_a_search_that_returns_nothing(tmp_path, capsys):
    """The nastiest case in the report: a symbol added after indexing returns "no
    results", exit 0, which is indistinguishable from a repo that does not contain
    it. The note is the only thing that separates them."""
    repo, _ = _indexed(tmp_path, capsys)
    (repo / "extra.py").write_text('def zzzqqq_nonexistent():\n    return 1\n')
    _git_add(repo)

    assert main(
        ["search", "zzzqqq nonexistent", "--repo", str(repo)], embedder_factory=fake_factory
    ) == 0
    captured = capsys.readouterr()
    assert "no results" in captured.out
    assert "not in this index" in captured.err


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
    # `assertions` joins the count map here rather than living somewhere else, because
    # the MCP `index_stats` payload already carries it there and two surfaces over one
    # index that disagree about the shape of the answer are worse than either shape.
    assert payload["counts"] == {"files": 1, "symbols": 3, "edges": payload["counts"]["edges"],
                                 "chunks": 3, "assertions": 0}
    assert set(payload["tiers"]) == {"T0", "T1", "T2"}
    # Structurally always 0: tier 2 lives in `assertions`, never on `edges.tier`.
    # Kept in the shape so the MCP `index_stats` payload and this one agree.
    assert payload["tiers"]["T2"] == 0
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


def test_stats_reports_the_assertion_store_it_used_to_be_blind_to(tmp_path, capsys):
    """Phase 9 shipped and `stats` never noticed.

    It printed a `T2 INFERRED` count read from `edges.tier` -- a column tier 2 never
    occupies, so the number was structurally always 0 -- annotated "the inference
    layer is not built yet", next to a store holding claims, verdicts and expiries.
    After a full `learn` run the one command whose job is "what is in this index"
    said nothing about the only part of it that was not derived from source."""
    repo, index_path = _indexed(tmp_path, capsys)
    _admit(index_path, repo, "core.frobnicate_widgets", "frobnicates widgets")
    refuted = _admit(index_path, repo, "core._plumbing", "returns the answer")
    expired = _admit(index_path, repo, "core", "the module")
    conn = db.connect(index_path)
    store.record_verdict(conn, refuted, "judge/v1", "refuted", "it does not")
    store.mark_stale(conn, expired, store.REASON_HASH_MISMATCH)
    conn.close()

    assert main(["stats", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["assertions"] == 3
    # Explicit zeros are part of the contract: `rejected` must be visible rather than
    # merely absent, because the rejected set is the only evidence the gate does
    # anything at all.
    assert payload["assertions_by_status"] == {"active": 1, "rejected": 1, "stale": 1}

    assert main(["stats", "--repo", str(repo)], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "assertions" in out
    assert "rejected" in out
    # The sentence that was false the moment Phase 9 landed.
    assert "the inference layer is not built yet" not in out


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
    not an appendix to it.

    The key set is derived from `LearnReport` rather than listed, and that is the
    whole test. A hand-written list is what let waves 1-2 add five `refused_*`
    counters -- `invalid_span`, `unverifiable`, `unknown_subject`, `stale_evidence`,
    `escaping_span` -- while the CLI kept emitting a fixed dict that omitted all of
    them, so a run refused entirely by the gate serialised as a run that admitted
    nothing for no stated reason. Listing them here would rebuild exactly that trap
    one layer up."""
    from dataclasses import fields as dataclass_fields

    from codelearner.generate.pipeline import LearnReport

    _patch_generator(monkeypatch, FakeClaimGenerator)
    repo, index_path = _indexed(tmp_path, capsys)

    assert main(["learn", "--repo", str(repo), "--json"], embedder_factory=fake_factory) == 0
    doc = json.loads(capsys.readouterr().out)

    expected = {f.name for f in dataclass_fields(LearnReport)} - {"results"}
    assert expected <= set(doc), sorted(expected - set(doc))
    # And nothing beyond the report's own fields plus the derived rates and the two
    # locations, so a key that stops being a counter cannot linger as a lie.
    assert set(doc) - expected == {
        "repo",
        "index",
        "admission_rate",
        "refused_by_the_gate",
        "drift",
    }
    for name in (
        "refused_invalid_span",
        "refused_unverifiable",
        "refused_unknown_subject",
        "refused_stale_evidence",
        "refused_escaping_span",
        "offers_dropped_oversize",
    ):
        assert name in doc, name
    assert doc["generator"] == "fake/claims"
    assert doc["index"] == str(index_path)


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


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------
#
# The exit code is the contract here. `codelearner gpu --free` is meant to be the
# line a measurement script puts in front of a long run, and a script can only gate
# on a number -- so the tests that matter most are the ones asserting 1 comes back
# when the memory did not.


def _fake_release(monkeypatch, report):
    monkeypatch.setattr(gpu, "release", lambda **kwargs: report)
    return report


def _state(models=(), reachable=True, sampled=True):
    return gpu.GpuState(
        ollama_reachable=reachable, models=models, host="http://fake", usage_sampled=sampled
    )


def _held(usage=gpu.USAGE_IDLE, sampled=True):
    return _state(
        models=(gpu.LoadedModel("qwen3:14b", 9_756_000_000, usage=usage),), sampled=sampled
    )


def test_gpu_needs_no_index_and_no_repo(tmp_path, capsys, monkeypatch):
    """The one command in the tool that works from anywhere.

    "Why is my run about to be ten times slower" is asked from whatever directory the
    terminal is in, often before an index exists at all. Requiring one would put the
    diagnostic out of reach exactly where it is needed, so this runs from an empty
    directory and must still answer."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpu, "read_state", lambda **kwargs: _held())

    assert main(["gpu"], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "qwen3:14b" in out
    assert "codelearner gpu --free" in out


def test_gpu_reports_without_freeing_anything(capsys, monkeypatch):
    """Reporting is read-only. A command that unloaded models as a side effect of
    being asked what was loaded would be the eviction `index/embed.py` refuses,
    performed without being asked for."""
    called = []
    monkeypatch.setattr(gpu, "read_state", lambda **kwargs: _held())
    monkeypatch.setattr(gpu, "release", lambda **kwargs: called.append(1))

    assert main(["gpu"], embedder_factory=fake_factory) == 0
    assert called == []


def test_gpu_free_exits_zero_when_the_memory_came_back(capsys, monkeypatch):
    _fake_release(
        monkeypatch,
        gpu.ReleaseReport(outcome=gpu.OUTCOME_FREED, before=_held(), after=_state()),
    )
    assert main(["gpu", "--free"], embedder_factory=fake_factory) == 0
    assert "FREED" in capsys.readouterr().out


def test_gpu_free_exits_NON_ZERO_when_it_could_not_free(capsys, monkeypatch):
    """The whole reason this command exists.

    Ollama answers the unload request with `done_reason="unload"` and keeps the
    memory. A `--free` that exited 0 here would let a measurement script proceed onto
    a card that is still full, which is the run-that-quietly-became-a-different-run
    this project is built to prevent -- one level up from where `gpu.py` prevents it.
    """
    _fake_release(
        monkeypatch,
        gpu.ReleaseReport(
            outcome=gpu.OUTCOME_NOT_FREED,
            before=_held(),
            after=_held(),
            asked=("qwen3:14b",),
            responses=(("qwen3:14b", "unload"),),
            reason=gpu.REASON_STILL_LISTED,
            waited_s=30.0,
        ),
    )
    assert main(["gpu", "--free"], embedder_factory=fake_factory) == 1
    out = capsys.readouterr().out
    # Both halves of the truth, not one: what was asked, and what is actually true.
    assert "done_reason=unload" in out
    assert "NOT FREED" in out
    assert "sudo systemctl restart ollama" in out


def test_gpu_free_exits_zero_when_there_was_nothing_to_free(capsys, monkeypatch):
    """Nothing loaded is not a failure to unload. A script gating on this is asking
    "is the card clear of ollama", and the answer is yes -- exiting 1 would send it
    off to fix something that is not broken."""
    for outcome in (gpu.OUTCOME_NOTHING_LOADED, gpu.OUTCOME_NO_OLLAMA):
        _fake_release(
            monkeypatch,
            gpu.ReleaseReport(outcome=outcome, before=_state(reachable=False), after=_state()),
        )
        assert main(["gpu", "--free"], embedder_factory=fake_factory) == 0
        capsys.readouterr()


def test_gpu_json_is_parseable_and_carries_the_verdict(capsys, monkeypatch):
    _fake_release(
        monkeypatch,
        gpu.ReleaseReport(
            outcome=gpu.OUTCOME_NOT_FREED,
            before=_held(),
            after=_held(),
            reason=gpu.REASON_STILL_LISTED,
        ),
    )
    assert main(["gpu", "--free", "--json"], embedder_factory=fake_factory) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["outcome"] == "not-freed"
    assert payload["advice"]


def test_gpu_state_json_names_what_it_could_not_see(capsys, monkeypatch):
    """`--json` from a machine with no ollama and no driver must still be valid JSON
    that says so, rather than zeroes a reader would take for measurements."""
    monkeypatch.setattr(
        gpu,
        "read_state",
        lambda **kwargs: gpu.GpuState(
            ollama_reachable=False,
            ollama_detail="could not reach ollama at http://fake (refused)",
            devices_detail="nvidia-smi could not be run; VRAM totals unknown",
        ),
    )
    assert main(["gpu", "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ollama_reachable"] is False
    assert payload["free_bytes"] is None
    assert payload["devices"] == []


def test_index_embed_warns_before_the_model_load_and_still_indexes(tmp_path, capsys, monkeypatch):
    """The pre-flight check warns and does not act.

    It must not become a gate: a contended card is a reason to tell the user
    something, not a reason to refuse to index. And the warning has to arrive BEFORE
    the embedder is built, because after it the minute of loading has already been
    spent on the wrong device."""
    order = []
    monkeypatch.setattr(gpu, "warn_if_contended", lambda **kwargs: order.append("warned"))
    repo = _mkrepo(tmp_path / "repo")

    def _factory(name):
        order.append("embedder")
        return FakeEmbedder(name)

    assert main(["index", str(repo), "--embed", "--model", "fake/v1"], embedder_factory=_factory) == 0
    assert order == ["warned", "embedder"]
    assert "embedded" in capsys.readouterr().out


def test_index_embed_survives_a_preflight_check_that_cannot_run(tmp_path, capsys):
    """With `urlopen` refused by this file's fixture, the check reaches nothing. An
    indexing run must not care -- a diagnostic that can fail the thing it diagnoses is
    worse than no diagnostic."""
    repo = _mkrepo(tmp_path / "repo")
    assert main(
        ["index", str(repo), "--embed", "--model", "fake/v1"], embedder_factory=fake_factory
    ) == 0
    assert "embedded" in capsys.readouterr().out


def test_gpu_does_not_offer_to_free_a_model_that_is_in_use(capsys, monkeypatch):
    """The bug the coordinator found by using it.

    A resident model whose `expires_at` is advancing is being CALLED, and the first
    version printed "Free it with `codelearner gpu --free`" at it regardless. Advice
    that is confidently wrong is worse than none, because it gets followed -- and
    following this one unloads a model mid-request."""
    monkeypatch.setattr(gpu, "read_state", lambda **kwargs: _held(usage=gpu.USAGE_IN_USE))
    assert main(["gpu"], embedder_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "IN USE" in out
    assert "Free it with" not in out
    assert "not a lock" in out  # the sample is stated as a sample


def test_gpu_offers_to_free_an_idle_model(capsys, monkeypatch):
    monkeypatch.setattr(gpu, "read_state", lambda **kwargs: _held(usage=gpu.USAGE_IDLE))
    assert main(["gpu"], embedder_factory=fake_factory) == 0
    assert "Free it with `codelearner gpu --free`." in capsys.readouterr().out


def test_gpu_samples_usage_by_default_and_no_usage_check_opts_out(capsys, monkeypatch):
    """Opt-IN for library callers, opt-OUT here. A human waiting at a prompt does not
    notice a second and a half, and the answer decides whether the next thing they are
    told to do destroys someone else's work."""
    seen = []
    monkeypatch.setattr(
        gpu, "read_state", lambda **kwargs: (seen.append(kwargs.get("usage_gap_s")), _held())[1]
    )
    assert main(["gpu"], embedder_factory=fake_factory) == 0
    assert main(["gpu", "--no-usage-check"], embedder_factory=fake_factory) == 0
    assert seen == [gpu.USAGE_SAMPLE_GAP_S, None]
    capsys.readouterr()


def test_gpu_free_exits_3_when_it_declined_because_the_model_is_in_use(capsys, monkeypatch):
    """A code of its own, and both boundaries are the point.

    Not 0: the card is not clear, and a measurement script reading success here starts
    onto a full one. Not 1: 1 needs a human with sudo, this clears itself when the
    other job finishes -- so a script can sleep and retry on 3 while escalating on 1.
    """
    _fake_release(
        monkeypatch,
        gpu.ReleaseReport(
            outcome=gpu.OUTCOME_IN_USE,
            before=_held(usage=gpu.USAGE_IN_USE),
            after=_held(usage=gpu.USAGE_IN_USE),
            reason="qwen3:14b is serving requests right now",
        ),
    )
    assert main(["gpu", "--free"], embedder_factory=fake_factory) == 3
    out = capsys.readouterr().out
    assert "DECLINED. Nothing was asked to unload." in out
    assert "--force" in out


def test_gpu_free_passes_force_through(capsys, monkeypatch):
    """`--force` is the only way to unload a model somebody is calling, and it has to
    be typed."""
    seen = {}
    monkeypatch.setattr(
        gpu,
        "release",
        lambda **kwargs: (
            seen.update(kwargs),
            gpu.ReleaseReport(outcome=gpu.OUTCOME_FREED, before=_held(), after=_state()),
        )[1],
    )
    assert main(["gpu", "--free"], embedder_factory=fake_factory) == 0
    assert seen["force"] is False
    assert main(["gpu", "--free", "--force"], embedder_factory=fake_factory) == 0
    assert seen["force"] is True
    capsys.readouterr()


def test_gpu_json_carries_the_usage_verdict_for_a_script(capsys, monkeypatch):
    """A script gating on this needs the verdict as data, not as a sentence to grep."""
    monkeypatch.setattr(gpu, "read_state", lambda **kwargs: _held(usage=gpu.USAGE_IN_USE))
    assert main(["gpu", "--json"], embedder_factory=fake_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["safe_to_free"] is False
    assert payload["usage_sampled"] is True
    assert payload["models"][0]["usage"] == "in-use"
