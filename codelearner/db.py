"""SQLite (WAL) connection + schema init for a code-learner index.

One DB file per repository. The connection model, `BEGIN IMMEDIATE` transaction
helper, and "refuse + rotate" schema policy follow the patterns proven in
swarm-sync's blackboard (`swarmsync/blackboard/db.py`) rather than being
re-derived -- they exist because each one has already been paid for once.

**Repo isolation.** `bind_repo_root` pins a DB file to the one repo root it
indexes. Symbol qualnames and file paths are root-relative, so pointing an index
at a second repo would silently merge two codebases into one graph. Isolation is
therefore structural (a separate file per repo) *and* enforced (a mismatched root
raises before any write).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Union

StrPath = Union[str, Path]  # noqa: UP007 - a type alias, not an annotation

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA = SCHEMA_PATH.read_text()

# Version of the schema this code reads and writes, stamped into `meta` on create.
# The migration policy is "refuse + rotate", NOT a migration framework: a DB whose
# stamp is missing or different is refused before any DDL runs, and the operator
# re-indexes. Re-indexing a repo is cheap (seconds); a silently half-migrated index
# that answers questions wrongly is not.
#
# History:
#   v1 -- files, symbols, edges. The tier-0/tier-1 split lives in `edges`
#         (`dst_name` always set, `dst_symbol_id` only once resolved).
#   v2 -- `chunks` (symbol-boundary retrieval units) plus the `chunks_fts` external-
#         content FTS5 index and its three sync triggers. Additive, but a v1 DB is
#         refused and re-indexed per the policy above rather than migrated.
#   v3 -- `files.is_test`, so retrieval can distinguish a test from the code it
#         tests. Both modalities were measured ranking tests above implementations.
#   v4 -- the tier-2 assertion store: `assertions`, `evidence_spans`, `verdicts`,
#         `staleness_log`. An inferred claim is admitted only with citations, and is
#         served only while those citations still hash to what was cited.
SCHEMA_VERSION = 4

EXPECTED_TABLES = ("files", "symbols", "edges", "chunks", "assertions")

# How long a write that loses a brief lock race waits for the winner instead of
# failing with "database is locked".
BUSY_TIMEOUT_SECONDS = 5.0


class SchemaVersionError(RuntimeError):
    """The DB file's schema version is missing (legacy) or does not match
    `SCHEMA_VERSION`. Raised by `init_db` BEFORE any DDL touches the file."""


class RepoRootMismatchError(RuntimeError):
    """The DB file is already bound to a DIFFERENT repo root.

    Symbol qualnames and file paths are root-relative, so proceeding would merge
    two unrelated codebases into one graph. Raised by `bind_repo_root`."""


def load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec on `conn`. Returns True on success.

    Optional by design: the lexical and graph modalities work without it, and a
    missing vector extension should degrade retrieval rather than prevent the tool
    from opening an index at all.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (AttributeError, sqlite3.OperationalError):
        # Python built without extension support, or a host SQLite that refuses.
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except AttributeError:
            pass


def _configure(conn: sqlite3.Connection) -> None:
    """Apply the pragmas + row factory every connection to an index must use."""
    conn.row_factory = sqlite3.Row
    # journal_mode is database-level but must be re-asserted per handle to take
    # effect on it; foreign_keys is per-connection and defaults OFF, so without
    # this the ON DELETE CASCADE relationships in the schema would silently not fire.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")


def connect(path: StrPath) -> sqlite3.Connection:
    """Open a connection to the index at `path` with WAL + FKs enabled.

    Does NOT create the schema -- callers needing a guaranteed-initialized DB
    should call `init_db`.
    """
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    _configure(conn)
    # Vector search lives in the same file, so the extension has to be present on
    # every handle that might query it -- a vec0 table is invisible to a connection
    # that did not load it.
    load_vec_extension(conn)
    return conn


def init_db(path: StrPath) -> sqlite3.Connection:
    """Idempotently create the index schema at `path` and return a connection.

    Safe to call repeatedly: the DDL is `CREATE ... IF NOT EXISTS` throughout.

    Schema versioning is "refuse + rotate" (see `SCHEMA_VERSION`). `sqlite_master`
    is inspected BEFORE the schema script runs, because once `CREATE ... IF NOT
    EXISTS` has executed, a legacy DB and a fresh one are indistinguishable --
    which is exactly how a stale index would get stranded silently.
    """
    parent = Path(str(path)).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    version: str | None = None
    if "meta" in names:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None:
            version = row["value"]
    if version is None and any(table in names for table in EXPECTED_TABLES):
        conn.close()
        raise SchemaVersionError(
            f"index {path} has application tables but no schema_version stamp "
            f"(pre-versioning). This code requires schema v{SCHEMA_VERSION} and "
            "has no migration system. Remedy: delete the index file and re-index."
        )
    if version is not None and version != str(SCHEMA_VERSION):
        conn.close()
        raise SchemaVersionError(
            f"index {path} is schema v{version}, but this code requires "
            f"v{SCHEMA_VERSION} (the stamp may come from older or newer "
            "code-learner; there is no migration system). Remedy: delete the "
            "index file and re-index."
        )
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a multi-statement write batch as ONE transaction on `conn`.

    `BEGIN IMMEDIATE` (not the default deferred `BEGIN`) takes the write lock up
    front, so two writers can never each hold a read lock and then deadlock trying
    to upgrade under WAL -- the loser simply waits on `busy_timeout`.

    Refuses to nest: entering while a transaction is already open on `conn` would
    silently make the outer writer's fate depend on this block's rollback.
    """
    if conn.in_transaction:
        raise sqlite3.ProgrammingError(
            "db.transaction(): connection is already inside a transaction; "
            "one transaction per connection (open a separate connection instead)"
        )
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def bind_repo_root(conn: sqlite3.Connection, root: StrPath) -> None:
    """Pin the index behind `conn` to the ONE repo root it indexes.

    First call stores the root; later calls with the same root are a no-op; a
    different root raises `RepoRootMismatchError` naming both. The write is
    `INSERT OR IGNORE` + read-back, so two concurrent first binds race safely:
    exactly one wins, and the loser either no-ops or raises.
    """
    resolved = str(Path(str(root)).resolve())
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('repo_root', ?)",
        (resolved,),
    )
    stored = stored_repo_root(conn)
    if stored != resolved:
        raise RepoRootMismatchError(
            f"index is bound to repo root {stored!r}, but this run is indexing "
            f"{resolved!r}. Symbol qualnames are root-relative, so reusing one "
            "index across repos would merge two codebases into one graph. "
            "Remedy: use a separate index file per repo (the default), or delete "
            "this one and re-index."
        )


def stored_repo_root(conn: sqlite3.Connection) -> str | None:
    """Return the repo root this index is bound to, or None if never bound."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'repo_root'").fetchone()
    return None if row is None else str(row["value"])


def reset(path: StrPath) -> None:
    """Delete the index file (and any WAL/SHM sidecars). Test helper only."""
    p = Path(str(path))
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(p) + suffix)
        if candidate.exists():
            candidate.unlink()
