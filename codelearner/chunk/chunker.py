"""Cut a repository into retrieval units on symbol boundaries.

**Why not a fixed window.** The default RAG move is to slice text every N tokens
with some overlap. On prose that is merely lossy; on code it is wrong. Half a
function is not a smaller fact about that function -- it is a fragment that reads as
if it were complete, and its embedding is the embedding of nothing anyone would ever
ask about. A retrieved chunk ending mid-`if` actively misleads.

So the unit here is the symbol: one chunk per function, method, class, or module.
The parser already knows exactly where each one starts and ends, which is the whole
argument for parsing before embedding.

**Why a generated header.** A method body retrieved on its own is not
self-describing. `def acquire(self, parcel_id, mode):` tells a reader -- and an
embedding model -- nothing about which class or module it belongs to, and that is
usually the part the question was actually about. Every chunk therefore opens with a
short generated header naming its file, its qualname, and its enclosing scope. The
header is stored separately as well, so a caller can show provenance without having
to parse it back out.

**Classes and modules are headers, not bodies.** A class chunk containing every
method would duplicate text already covered by the per-method chunks and would blur
the class's own identity across everything it contains. A class chunk is therefore
its declaration, docstring, and the list of its members -- which is what "what is
this class for" actually wants.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .. import db
from ..ingest.types import KIND_CLASS, KIND_MODULE, content_hash

# Symbols whose chunk is a summary of what they contain rather than their full
# source, because their bodies are already covered by their members' own chunks.
_SUMMARY_KINDS = frozenset({KIND_CLASS, KIND_MODULE})

# Cap on a single chunk's source slice. Generous on purpose: the embedding model
# chosen for Phase 2 (C2LLM-0.5B) takes 32K tokens, so this only ever trips on
# genuinely pathological generated code. Truncation is recorded, never silent.
MAX_CHUNK_CHARS = 24_000

TRUNCATION_MARKER = "\n# ... [truncated by code-learner: symbol exceeds MAX_CHUNK_CHARS]"


@dataclass
class ChunkStats:
    chunks: int = 0
    truncated: int = 0
    skipped_empty: int = 0


def _header_for(
    path: str,
    kind: str,
    qualname: str,
    signature: str | None,
    docstring: str | None,
) -> str:
    """Build the context header that opens every chunk.

    Deliberately plain text rather than a structured preamble: it is read by an
    embedding model and by humans, and both do better with a sentence than with a
    key-value block.
    """
    lines = [f"# {path} -- {kind} {qualname}"]
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else None
    if parent and parent != qualname:
        lines.append(f"# defined in: {parent}")
    if signature:
        lines.append(f"# signature: {signature}")
    if docstring:
        first = docstring.strip().splitlines()[0].strip()
        if first:
            lines.append(f"# purpose (from docstring): {first[:200]}")
    return "\n".join(lines)


def chunk_for_symbol(
    source: bytes,
    path: str,
    kind: str,
    qualname: str,
    byte_start: int,
    byte_end: int,
    signature: str | None = None,
    docstring: str | None = None,
    members: list[str] | None = None,
) -> tuple[str, str, bool]:
    """Return `(text, header, was_truncated)` for one symbol.

    `members` is used only for class/module summaries -- the names a container holds,
    which is what makes a container chunk worth retrieving at all.
    """
    header = _header_for(path, kind, qualname, signature, docstring)
    truncated = False

    if kind in _SUMMARY_KINDS:
        body_lines = []
        if docstring:
            body_lines.append(f'"""{docstring.strip()}"""')
        if members:
            label = "methods" if kind == KIND_CLASS else "defines"
            body_lines.append(f"# {label}: {', '.join(members)}")
        body = "\n".join(body_lines)
    else:
        body = source[byte_start:byte_end].decode("utf-8", errors="replace")
        if len(body) > MAX_CHUNK_CHARS:
            body = body[:MAX_CHUNK_CHARS] + TRUNCATION_MARKER
            truncated = True

    return f"{header}\n{body}".strip(), header, truncated


def build_chunks(conn: sqlite3.Connection, repo_root: Path) -> ChunkStats:
    """Build one chunk per symbol in the index.

    Idempotent: existing chunks are cleared first, so re-running after a chunking
    change rebuilds cleanly rather than accumulating. The FTS index follows via the
    schema's triggers.
    """
    stats = ChunkStats()

    members: dict[int, list[str]] = {}
    for row in conn.execute(
        "SELECT parent_id, name FROM symbols WHERE parent_id IS NOT NULL ORDER BY line_start"
    ):
        members.setdefault(row["parent_id"], []).append(row["name"])

    rows = list(
        conn.execute(
            "SELECT s.id, s.kind, s.qualname, s.byte_start, s.byte_end, "
            "       s.signature, s.docstring, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "ORDER BY f.path, s.line_start"
        )
    )

    # Read each file once rather than once per symbol.
    sources: dict[str, bytes] = {}
    pending: list[tuple[int, str, str, int]] = []

    for row in rows:
        path = row["path"]
        if path not in sources:
            target = repo_root / path
            # Not something `except OSError` can catch: a FIFO does not fail this
            # read, it blocks it, until some other process opens the write end. An
            # index build over a repo containing one would hang with no traceback and
            # no log line, which reads as a slow index rather than a stopped one.
            # `is_file()` is False for a FIFO, a directory, a socket and a device
            # node, so one test covers the class. The empty-source fallback is the
            # same disposition an unreadable file already gets: every symbol the
            # index recorded in that file yields a header-only chunk and is counted
            # in `skipped_empty` rather than indexed. Sibling guards live in
            # `assertions.store._read_source`, `assertions.stale._read_file` and
            # `generate.pipeline._read_source`; duplicated rather than shared,
            # because a private cross-package import would couple four packages to
            # save four lines. It does not close the test-then-read window -- a
            # regular file swapped for a FIFO in between still blocks.
            if not target.is_file():
                sources[path] = b""
            else:
                try:
                    sources[path] = target.read_bytes()
                except OSError:
                    sources[path] = b""
        source = sources[path]

        text, header, truncated = chunk_for_symbol(
            source=source,
            path=path,
            kind=row["kind"],
            qualname=row["qualname"],
            byte_start=row["byte_start"],
            byte_end=row["byte_end"],
            signature=row["signature"],
            docstring=row["docstring"],
            members=members.get(row["id"]),
        )
        if truncated:
            stats.truncated += 1
        # A chunk that is nothing but its own header carries no retrievable content
        # -- an empty `__init__.py` is the common case. Indexing it would put a row
        # in the retrieval set that can only ever be a false positive.
        if len(text) <= len(header) + 1:
            stats.skipped_empty += 1
            continue
        pending.append((row["id"], text, header, len(text)))

    with db.transaction(conn):
        conn.execute("DELETE FROM chunks")
        conn.executemany(
            "INSERT INTO chunks (symbol_id, text, header, char_count, text_hash) "
            "VALUES (?,?,?,?,?)",
            [
                (sid, text, header, count, content_hash(text.encode("utf-8")))
                for sid, text, header, count in pending
            ],
        )
    stats.chunks = len(pending)
    return stats
