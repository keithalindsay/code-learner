"""Hydrate retrieved symbols into source evidence."""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
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


def _read_source_file(root: Path, path: Path, *, symbol_id: int) -> bytes:
    """Read a repo-relative regular file through one rooted descriptor chain.

    Every path component is opened relative to its already-open parent and with
    symlink following disabled. The final descriptor supplies both the metadata and
    the bounded bytes, so a pathname replacement cannot change what is checked.
    Platforms without these primitives fail closed instead of weakening the check.
    """
    if not os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceError("file_not_regular", symbol_id)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise EvidenceError("path_escapes_repo", symbol_id)

    descriptors: list[int] = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptors.append(os.open(root, directory_flags))
        for part in parts[:-1]:
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        descriptors.append(os.open(parts[-1], file_flags, dir_fd=descriptors[-1]))
        file_stat = os.fstat(descriptors[-1])
        if not stat.S_ISREG(file_stat.st_mode):
            raise EvidenceError("file_not_regular", symbol_id)
        if file_stat.st_size > MAX_SOURCE_FILE_BYTES:
            raise EvidenceError("file_too_large", symbol_id)

        chunks: list[bytes] = []
        remaining = MAX_SOURCE_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptors[-1], min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_SOURCE_FILE_BYTES:
            raise EvidenceError("file_too_large", symbol_id)
        return raw
    except EvidenceError:
        raise
    except FileNotFoundError:
        raise EvidenceError("file_missing", symbol_id) from None
    except OSError as exc:
        code = "file_missing" if exc.errno == errno.ENOENT else "file_not_regular"
        raise EvidenceError(code, symbol_id) from None
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


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
        raw = _read_source_file(root, path, symbol_id=symbol_id)

        coordinates = (
            row["byte_start"],
            row["byte_end"],
            row["line_start"],
            row["line_end"],
        )
        if any(type(coordinate) is not int for coordinate in coordinates):
            raise EvidenceError("invalid_span", symbol_id) from None
        byte_start, byte_end, indexed_line_start, indexed_line_end = coordinates
        if (
            byte_start < 0
            or byte_end <= byte_start
            or byte_end > len(raw)
            or indexed_line_start < 1
            or indexed_line_end < indexed_line_start
        ):
            raise EvidenceError("invalid_span", symbol_id)
        symbol_source = raw[byte_start:byte_end]
        if content_hash(symbol_source) != str(row["content_hash"]):
            raise EvidenceError("source_changed", symbol_id)

        line_start = raw[:byte_start].count(b"\n") + 1
        line_end = raw[:byte_end].count(b"\n") + 1

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
