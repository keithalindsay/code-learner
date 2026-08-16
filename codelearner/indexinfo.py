"""Where an index lives, and the facts read straight off its tables.

A leaf, deliberately, the same way `tier.py` is: it imports nothing from
`codelearner.cli`, `codelearner.server`, or `codelearner.eval`, so both the CLI and
the MCP server can depend on it without either depending on the other. It was written
in `cli/commands.py`, which is where a person's `codelearner stats` reads it from --
but `server/app.py` needed the same path-resolution logic and the same table counts
to answer the MCP `get_stats` tool, and reaching them meant importing *upward* into
the module a person types commands into. That inversion is what moving them here
closes: `resolve_index_path`, `INDEX_RELPATH` and the `stats` helpers below now sit in
a leaf both surfaces import as peers, and `cli/commands.py` re-exports every name so
existing importers keep working unchanged.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .retrieve import stored_embed_model

# Kept in step with `indexer.index_repo`'s own default. One file per repo is what
# makes cross-repo contamination structurally impossible, so the CLI must not
# invent a second convention for where that file lives.
INDEX_RELPATH = Path(".codelearner") / "index.db"


def resolve_index_path(repo: Path, index_path: Path | None) -> Path:
    """Where this invocation's index lives: explicit if given, else the default."""
    if index_path is not None:
        return index_path.expanduser()
    return repo / INDEX_RELPATH


REBUILD_ADVICE = (
    "Since --carry-assertions that no longer costs the assertions: rebuild with "
    "`codelearner index <repo> --force --carry-assertions` and the tier-2 store "
    "(assertions, verdicts, staleness log) is carried across; only the embeddings "
    "have to be re-derived."
)


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return 0 if row is None else int(row[0])


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def _classify_unresolved(conn: sqlite3.Connection) -> tuple[int, int]:
    """Split unresolved edges into (external, ambiguous), mirroring `resolve_all`.

    Recomputed from the stored graph rather than remembered from index time, so
    `stats` tells the truth about the file in front of it even if the resolver was
    re-run since. External means "no symbol in this repo even shares the basename",
    which is the only honest way to say that a call to `json.dumps` is not a
    resolution failure.
    """
    names = {row["name"] for row in conn.execute("SELECT DISTINCT name FROM symbols")}
    external = ambiguous = 0
    for row in conn.execute("SELECT dst_name FROM edges WHERE dst_symbol_id IS NULL"):
        base = str(row["dst_name"]).rsplit(".", 1)[-1]
        if base in names:
            ambiguous += 1
        else:
            external += 1
    return external, ambiguous


def _embedding_info(conn: sqlite3.Connection) -> dict[str, Any]:
    """What vectors this index holds, if any, and from which model."""
    model = stored_embed_model(conn)
    dim = _meta(conn, "embed_dim")
    try:
        vectors = _scalar(conn, "SELECT count(*) FROM vec_chunks")
    except sqlite3.OperationalError:
        # No vec table, or sqlite-vec is not loadable on this handle. Either way
        # there is nothing to report and nothing to fail about.
        vectors = 0
    return {
        "present": bool(model and vectors),
        "model": model,
        "dim": int(dim) if dim is not None else None,
        "vectors": vectors,
    }
