"""Dense retrieval over the sqlite-vec chunk vectors.

**The `k = ?` form is mandatory here, not stylistic.** This host's SQLite is 3.37.2
(2021), which does not push `LIMIT` down into a virtual table's query planner, so
the documented `ORDER BY distance LIMIT n` form fails outright with *"A LIMIT or
'k = ?' constraint is required on vec0 knn queries"*. Measured in the Phase 0 spike;
see docs/PHASE0-FINDINGS.md.

Scores are returned as similarity (higher is better) rather than distance, matching
the lexical modality's convention. sqlite-vec returns L2 distance over vectors that
were normalized at encode time, so `1 - d^2/2` recovers cosine similarity exactly.
"""
from __future__ import annotations

import sqlite3

from ..index.embed import Embedder, serialize
from .lexical import Hit


def search_dense(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder,
    k: int = 10,
) -> list[Hit]:
    """Return the top `k` chunks for `query` by vector similarity, best first."""
    if not _has_vectors(conn):
        return []
    query_vec = serialize(embedder.encode_query(query))
    rows = conn.execute(
        """
        SELECT v.chunk_id, v.distance,
               s.id AS symbol_id, s.qualname, s.kind, s.line_start, s.line_end,
               f.path, c.header
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        JOIN symbols s ON s.id = c.symbol_id
        JOIN files f ON f.id = s.file_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_vec, k),
    ).fetchall()
    return [
        Hit(
            symbol_id=r["symbol_id"],
            qualname=r["qualname"],
            kind=r["kind"],
            path=r["path"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            # L2 over unit vectors -> cosine similarity.
            score=1.0 - (float(r["distance"]) ** 2) / 2.0,
            modality="dense",
            header=r["header"],
        )
        for r in rows
    ]


def _has_vectors(conn: sqlite3.Connection) -> bool:
    """True if this index has a populated vec table this connection can see."""
    try:
        row = conn.execute("SELECT count(*) AS c FROM vec_chunks").fetchone()
    except sqlite3.OperationalError:
        # No vec table, or the extension is not loaded on this handle.
        return False
    return bool(row and row["c"])


def stored_embed_model(conn: sqlite3.Connection) -> str | None:
    """Which model produced the vectors in this index, if any.

    Callers must check this before querying: vectors from two different models are
    not comparable, and querying with a mismatched embedder returns results that
    look plausible and are meaningless.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    return None if row is None else str(row["value"])
