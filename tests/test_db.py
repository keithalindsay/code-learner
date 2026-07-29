"""Index store: schema versioning, transaction discipline, repo binding."""
from __future__ import annotations

import sqlite3

import pytest

from codelearner import db


def test_init_db_creates_the_expected_tables(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(db.EXPECTED_TABLES) <= names
    assert "meta" in names


def test_init_db_is_idempotent(tmp_path):
    path = tmp_path / "i.db"
    db.init_db(path).close()
    db.init_db(path).close()  # must not raise or duplicate


def test_init_db_creates_parent_directories(tmp_path):
    conn = db.init_db(tmp_path / "nested" / "deep" / "i.db")
    assert conn.execute("SELECT 1").fetchone() is not None


def test_foreign_keys_are_on_so_cascades_actually_fire(tmp_path):
    """PRAGMA foreign_keys defaults OFF per-connection; without it the schema's
    ON DELETE CASCADE relationships would silently never run."""
    conn = db.init_db(tmp_path / "i.db")
    conn.execute("INSERT INTO files (path,lang,content_hash,size_bytes,mtime_ns) VALUES ('a.py','python','h',1,1)")
    fid = conn.execute("SELECT id FROM files").fetchone()["id"]
    conn.execute(
        "INSERT INTO symbols (file_id,kind,name,qualname,line_start,line_end,"
        "byte_start,byte_end,content_hash) VALUES (?,'function','f','m.f',1,1,0,1,'h')", (fid,)
    )
    conn.execute("DELETE FROM files WHERE id = ?", (fid,))
    assert conn.execute("SELECT count(*) c FROM symbols").fetchone()["c"] == 0


def test_schema_version_mismatch_is_refused(tmp_path):
    path = tmp_path / "i.db"
    conn = db.init_db(path)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.close()
    with pytest.raises(db.SchemaVersionError):
        db.init_db(path)


def test_unstamped_legacy_db_is_refused_before_any_ddl(tmp_path):
    """A DB with application tables but no version stamp must be caught BEFORE
    `CREATE ... IF NOT EXISTS` runs -- afterwards it is indistinguishable from a
    fresh one, which is exactly how a stale index gets stranded silently."""
    path = tmp_path / "i.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()
    with pytest.raises(db.SchemaVersionError):
        db.init_db(path)


def test_transaction_commits_on_success(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    with db.transaction(conn):
        conn.execute("INSERT INTO files (path,lang,content_hash,size_bytes,mtime_ns) VALUES ('a.py','python','h',1,1)")
    assert conn.execute("SELECT count(*) c FROM files").fetchone()["c"] == 1


def test_transaction_rolls_back_on_error(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            conn.execute("INSERT INTO files (path,lang,content_hash,size_bytes,mtime_ns) VALUES ('a.py','python','h',1,1)")
            raise RuntimeError("boom")
    assert conn.execute("SELECT count(*) c FROM files").fetchone()["c"] == 0


def test_transaction_refuses_to_nest(tmp_path):
    """Nesting would silently make an outer writer's fate depend on an inner
    block's rollback. One transaction per connection."""
    conn = db.init_db(tmp_path / "i.db")
    with db.transaction(conn):
        with pytest.raises(sqlite3.ProgrammingError):
            with db.transaction(conn):
                pass


def test_bind_repo_root_is_sticky_and_rejects_a_second_root(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    db.bind_repo_root(conn, a)
    db.bind_repo_root(conn, a)  # same root -- no-op
    assert db.stored_repo_root(conn) == str(a.resolve())
    with pytest.raises(db.RepoRootMismatchError):
        db.bind_repo_root(conn, b)


def test_stored_repo_root_is_none_before_binding(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    assert db.stored_repo_root(conn) is None


def test_connect_refuses_a_schema_mismatched_index(tmp_path):
    """REGRESSION -- the READ paths are where a stale index does its damage.

    `init_db` has always enforced refuse-and-rotate, but it only runs when an index
    is built. Every query afterwards goes through `connect`: the CLI, the MCP server,
    `run_ablation`. Found by measuring a retrieval ablation against a v3 index with
    v5 code -- it opened without complaint, the columns that query named happened to
    exist, and it produced a complete results table that was simply not about the
    current schema. An error is recoverable; plausible wrong numbers are not.
    """
    path = tmp_path / "i.db"
    db.init_db(path).close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()

    with pytest.raises(db.SchemaVersionError):
        db.connect(path)


def test_connect_can_be_told_not_to_check(tmp_path):
    """`init_db` must open a DB *before* it can judge the schema, and tooling that
    deliberately inspects an old file needs the same escape hatch."""
    path = tmp_path / "i.db"
    db.init_db(path).close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()

    conn = db.connect(path, check_schema=False)
    assert conn.execute("SELECT 1").fetchone() is not None


def test_connect_accepts_a_file_with_nothing_written_yet(tmp_path):
    """An empty file is fresh, not stale -- it is what `init_db` is about to fill."""
    conn = db.connect(tmp_path / "brand-new.db")
    assert conn.execute("SELECT 1").fetchone() is not None


def test_connect_refuses_an_unstamped_legacy_index(tmp_path):
    """Application tables but no stamp: pre-versioning, and just as unreadable."""
    path = tmp_path / "i.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
    raw.commit()
    raw.close()
    with pytest.raises(db.SchemaVersionError):
        db.connect(path)
