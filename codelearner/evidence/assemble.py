"""Hydrate retrieved symbols into source evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from ..ingest.types import content_hash
from ..retrieve.lexical import Hit
from .render import content_bytes, number_source
from .types import EvidenceBundle, EvidenceSection

MAX_EVIDENCE_BYTES = 65_536
MAX_SOURCE_FILE_BYTES = 2_000_000


class EvidenceError(RuntimeError):
    """A source file could not be safely assembled into evidence."""

    def __init__(self, code: str, symbol_id: int) -> None:
        self.code = code
        self.symbol_id = symbol_id
        self.message = "Source evidence could not be assembled."
        super().__init__(self.message)


def assemble_evidence(
    conn: sqlite3.Connection,
    repo_root: Path,
    hits: Iterable[Hit],
    *,
    budget_bytes: int,
) -> EvidenceBundle:
    """Load whole indexed symbols in retrieval order."""
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be >= 0")
    budget_bytes = min(budget_bytes, MAX_EVIDENCE_BYTES)

    hit_ids = list(dict.fromkeys(hit.symbol_id for hit in hits))
    if not hit_ids:
        return EvidenceBundle((), budget_bytes, 0, 0, ())
    if budget_bytes == 0:
        return EvidenceBundle((), budget_bytes, 0, len(hit_ids), tuple(hit_ids))

    placeholders = ", ".join("?" for _ in hit_ids)
    query = (
        "SELECT s.id, s.qualname, s.line_start, s.line_end, s.byte_start, s.byte_end, "  # noqa: S608
        "s.content_hash, f.path FROM symbols s JOIN files f ON f.id = s.file_id "
        f"WHERE s.id IN ({placeholders})"
    )
    rows = conn.execute(query, hit_ids).fetchall()
    by_id = {int(row["id"]): row for row in rows}

    sections: list[EvidenceSection] = []
    omitted: list[int] = []
    used_bytes = 0
    root = repo_root.resolve()
    for symbol_id in hit_ids:
        row = by_id.get(symbol_id)
        if row is None:
            omitted.append(symbol_id)
            continue
        path = Path(str(row["path"]))
        if path.is_absolute():
            raise EvidenceError("path_escapes_repo", symbol_id)
        candidate = root / path
        try:
            if candidate.is_symlink():
                raise EvidenceError("file_not_regular", symbol_id)
        except OSError:
            raise EvidenceError("file_not_regular", symbol_id) from None
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            raise EvidenceError("path_escapes_repo", symbol_id) from None
        if not resolved.is_relative_to(root):
            raise EvidenceError("path_escapes_repo", symbol_id)
        try:
            if not candidate.exists():
                raise EvidenceError("file_missing", symbol_id)
            if not candidate.is_file():
                raise EvidenceError("file_not_regular", symbol_id)
            if candidate.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise EvidenceError("file_too_large", symbol_id)
            raw = candidate.read_bytes()
        except FileNotFoundError:
            raise EvidenceError("file_missing", symbol_id) from None
        except OSError:
            raise EvidenceError("file_not_regular", symbol_id) from None

        try:
            byte_start = int(row["byte_start"])
            byte_end = int(row["byte_end"])
            line_start = int(row["line_start"])
            line_end = int(row["line_end"])
        except (OverflowError, TypeError, ValueError):
            raise EvidenceError("invalid_span", symbol_id) from None
        if (
            byte_start < 0
            or byte_end <= byte_start
            or byte_end > len(raw)
            or line_start < 1
            or line_end < line_start
        ):
            raise EvidenceError("invalid_span", symbol_id)
        symbol_source = raw[byte_start:byte_end]
        if content_hash(symbol_source) != str(row["content_hash"]):
            raise EvidenceError("source_changed", symbol_id)

        source = number_source(symbol_source.decode("utf-8", "replace"), line_start)
        section = EvidenceSection(
            symbol_id=symbol_id,
            qualname=str(row["qualname"]),
            path=str(path),
            line_start=line_start,
            line_end=line_end,
            content_hash=str(row["content_hash"]),
            source=source,
            content_bytes=content_bytes(source),
        )
        if section.content_bytes <= budget_bytes - used_bytes:
            sections.append(section)
            used_bytes += section.content_bytes
        else:
            omitted.append(symbol_id)
    return EvidenceBundle(
        tuple(sections), budget_bytes, used_bytes, len(omitted), tuple(omitted)
    )
