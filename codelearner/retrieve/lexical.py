"""Lexical retrieval over the FTS5 chunk index.

The unglamorous modality, and the one the eval exists to keep honest. BM25 over
identifiers is a genuinely strong baseline for code search -- people searching a
codebase usually know a name, and an exact name match beats semantic similarity for
that query shape every time. If the per-modality ablation in Phase 8 shows dense
retrieval failing to beat this, that is a real finding and not an embarrassment.

SQLite's FTS5 `bm25()` returns a score where **more negative is better**. It is
negated here so every modality in this package agrees that higher means better --
mixing the two conventions in a fusion step is a silent, plausible-looking bug.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    symbol_id: int
    qualname: str
    kind: str
    path: str
    line_start: int
    line_end: int
    score: float
    modality: str
    header: str
    is_test: bool = False
    # How this hit was reached when it came from graph expansion, e.g.
    # "called by tests.test_git_ops.test_remove_worktree". Empty for direct hits.
    # Retrieval that cannot say why it returned something is hard to debug and
    # harder to trust.
    via: str = ""


# FTS5 treats these as query syntax. A user typing `db.init_db(` means it literally.
#
# The C0 range (and DEL) is here for a different failure than the punctuation is, and
# a worse one. Quoting a term containing a NUL byte does not produce a quoted term:
# SQLite's string handling stops at the NUL, so the closing quote this function
# appended is never seen and FTS5 raises `unterminated string` -- an `sqlite3.Error`
# that `server.app._guard` cannot distinguish from a damaged index, so the server
# answers `index_unreadable` and tells the caller to re-index a database that is
# perfectly healthy. The rest of C0 cannot appear in an identifier either, so
# stripping the whole range costs no query anyone could have meant.
_FTS_SPECIAL = re.compile(r'[\x00-\x1f\x7f"*():^-]')


def escape_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Every term is quoted, which disables operator interpretation entirely. Passing
    raw input to MATCH otherwise turns `foo(` into a syntax error and `a-b` into a
    NOT query -- both of which look like "no results" rather than like a bug.

    Quoting is necessary and not sufficient: a control byte inside a term escapes the
    quoting itself (see `_FTS_SPECIAL`), which is why the substitution runs first and
    why its character class covers bytes no code search could contain rather than
    only the ones FTS5 documents as operators.
    """
    terms = [t for t in _FTS_SPECIAL.sub(" ", query).split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


def search_lexical(conn: sqlite3.Connection, query: str, k: int = 10) -> list[Hit]:
    """Return the top `k` chunks for `query` by BM25, best first."""
    match = escape_fts_query(query)
    if match == '""':
        return []
    rows = conn.execute(
        """
        SELECT s.id AS symbol_id, s.qualname, s.kind, s.line_start, s.line_end,
               f.path, f.is_test, c.header, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN symbols s ON s.id = c.symbol_id
        JOIN files f ON f.id = s.file_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, k),
    ).fetchall()
    return [
        Hit(
            symbol_id=r["symbol_id"],
            qualname=r["qualname"],
            kind=r["kind"],
            path=r["path"],
            line_start=r["line_start"],
            line_end=r["line_end"],
            # Negated: FTS5's bm25() is more-negative-is-better.
            score=-float(r["score"]),
            modality="lexical",
            header=r["header"],
            is_test=bool(r["is_test"]),
        )
        for r in rows
    ]
