"""Hydrate retrieved symbols into source evidence."""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ..ingest.types import content_hash
from ..retrieve.lexical import Hit
from ..retrieve.types import AssertionCandidate, Candidate, SourceCandidate
from .render import content_bytes, number_source
from .types import (
    AssertionEvidence,
    CandidateEvidence,
    CandidateEvidenceBundle,
    Citation,
    EvidenceBundle,
    EvidenceSection,
    OmittedCandidate,
    RelatedSymbol,
)

if TYPE_CHECKING:
    from ..assertions.store import EvidenceSpan

MAX_EVIDENCE_BYTES = 65_536
MAX_SOURCE_FILE_BYTES = 2_000_000
MAX_RELATED_SYMBOLS = 10
_SQL_BATCH = 500

_SYMBOL_SQL = (
    "SELECT s.id, s.qualname, s.line_start, s.line_end, s.byte_start, s.byte_end, "
    "s.content_hash, f.path FROM symbols s JOIN files f ON f.id = s.file_id "
)


class EvidenceError(RuntimeError):
    """A source file could not be safely assembled into evidence."""

    def __init__(self, code: str, symbol_id: int) -> None:
        self.code = code
        self.symbol_id = symbol_id
        self.message = "Source evidence could not be assembled."
        super().__init__(self.message)


class _ReadError(Exception):
    """A rooted read failed; the caller decides whether that is fatal or a reason."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _read_repo_file(root: Path, path: Path) -> bytes:
    """Read a repo-relative regular file through one rooted descriptor chain.

    Every path component is opened relative to its already-open parent and with
    symlink following disabled. The final descriptor supplies both the metadata and
    the bounded bytes, so a pathname replacement cannot change what is checked.
    Platforms without these primitives fail closed instead of weakening the check.

    Shared by symbol hydration and assertion citations so that neither surface can
    drift into a weaker read: a citation is read exactly the way indexed source is.
    The failure is raised as a code rather than an ``EvidenceError`` because the two
    callers differ in what a failure means -- a source hit raises, a semantic
    candidate is withheld with a reason.
    """
    if not os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise _ReadError("file_not_regular")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _ReadError("path_escapes_repo")

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
            raise _ReadError("file_not_regular")
        if file_stat.st_size > MAX_SOURCE_FILE_BYTES:
            raise _ReadError("file_too_large")

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
            raise _ReadError("file_too_large")
        return raw
    except _ReadError:
        raise
    except FileNotFoundError:
        raise _ReadError("file_missing") from None
    except OSError as exc:
        code = "file_missing" if exc.errno == errno.ENOENT else "file_not_regular"
        raise _ReadError(code) from None
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

    by_id = _symbol_rows(conn, hit_ids)

    sections: list[EvidenceSection] = []
    omitted: list[int] = []
    used_bytes = 0
    root = repo_root.resolve()
    cache: dict[str, bytes | _ReadError] = {}
    for symbol_id in hit_ids:
        row = by_id.get(symbol_id)
        if row is None:
            omitted.append(symbol_id)
            continue
        section = _section_from_row(root, row, cache)
        if section.content_bytes <= budget_bytes - used_bytes:
            sections.append(section)
            used_bytes += section.content_bytes
        else:
            omitted.append(symbol_id)
    return EvidenceBundle(
        tuple(sections), budget_bytes, used_bytes, len(omitted), tuple(omitted)
    )


def _cached_file(root: Path, path: str, cache: dict[str, bytes | _ReadError]) -> bytes:
    """Read one repository file at most once per assembly pass."""
    if path not in cache:
        candidate = Path(path)
        if candidate.is_absolute():
            cache[path] = _ReadError("path_escapes_repo")
        else:
            try:
                cache[path] = _read_repo_file(root, candidate)
            except _ReadError as error:
                cache[path] = error
    found = cache[path]
    if isinstance(found, _ReadError):
        raise found
    return found


def _live_lines(raw: bytes, byte_start: int, byte_end: int) -> tuple[int, int]:
    return raw[:byte_start].count(b"\n") + 1, raw[:byte_end].count(b"\n") + 1


def _symbol_rows(
    conn: sqlite3.Connection, symbol_ids: Iterable[int]
) -> dict[int, sqlite3.Row]:
    """Batch-load the indexed coordinates for every symbol this page can need."""
    ordered = list(dict.fromkeys(symbol_ids))
    rows: dict[int, sqlite3.Row] = {}
    for start in range(0, len(ordered), _SQL_BATCH):
        batch = ordered[start : start + _SQL_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        for row in conn.execute(
            f"{_SYMBOL_SQL}WHERE s.id IN ({placeholders})",  # noqa: S608
            batch,
        ):
            rows[int(row["id"])] = row
    return rows


def _section_from_row(
    root: Path, row: sqlite3.Row, cache: dict[str, bytes | _ReadError]
) -> EvidenceSection:
    """Hydrate one indexed symbol, refusing anything that no longer matches."""
    symbol_id = int(row["id"])
    path = str(row["path"])
    try:
        raw = _cached_file(root, path, cache)
    except _ReadError as error:
        raise EvidenceError(error.code, symbol_id) from None

    coordinates = (
        row["byte_start"],
        row["byte_end"],
        row["line_start"],
        row["line_end"],
    )
    if any(type(coordinate) is not int for coordinate in coordinates):
        raise EvidenceError("invalid_span", symbol_id)
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

    line_start, line_end = _live_lines(raw, byte_start, byte_end)
    source = number_source(symbol_source.decode("utf-8", "replace"), line_start)
    return EvidenceSection(
        symbol_id=symbol_id,
        qualname=str(row["qualname"]),
        path=path,
        line_start=line_start,
        line_end=line_end,
        content_hash=str(row["content_hash"]),
        source=source,
        content_bytes=content_bytes(source),
    )


def _related_symbols(
    conn: sqlite3.Connection, subject_ids: Iterable[int]
) -> dict[int, tuple[RelatedSymbol, ...]]:
    """Batch-load bounded caller and callee context for every subject on the page."""
    ordered = list(dict.fromkeys(subject_ids))
    found: dict[int, list[RelatedSymbol]] = {subject: [] for subject in ordered}
    queries = (
        (
            "caller",
            "SELECT e.dst_symbol_id AS subject_id, e.line, s.id AS other_id, "  # noqa: S608
            "s.qualname, f.path FROM edges e JOIN symbols s ON s.id = e.src_symbol_id "
            "JOIN files f ON f.id = s.file_id WHERE e.dst_symbol_id IN ({placeholders}) "
            "ORDER BY s.qualname, e.line",
        ),
        (
            "callee",
            "SELECT e.src_symbol_id AS subject_id, e.line, d.id AS other_id, "  # noqa: S608
            "d.qualname, f.path FROM edges e JOIN symbols d ON d.id = e.dst_symbol_id "
            "JOIN files f ON f.id = d.file_id WHERE e.src_symbol_id IN ({placeholders}) "
            "AND e.dst_symbol_id IS NOT NULL ORDER BY d.qualname, e.line",
        ),
    )
    for relation, template in queries:
        counts: dict[int, int] = {}
        for start in range(0, len(ordered), _SQL_BATCH):
            batch = ordered[start : start + _SQL_BATCH]
            placeholders = ", ".join("?" for _ in batch)
            for row in conn.execute(template.format(placeholders=placeholders), batch):
                subject_id = int(row["subject_id"])
                if counts.get(subject_id, 0) >= MAX_RELATED_SYMBOLS:
                    continue
                counts[subject_id] = counts.get(subject_id, 0) + 1
                found[subject_id].append(
                    RelatedSymbol(
                        relation=relation,
                        symbol_id=int(row["other_id"]),
                        qualname=str(row["qualname"]),
                        path=str(row["path"]),
                        line=int(row["line"]),
                    )
                )
    return {subject: tuple(related) for subject, related in found.items()}


def _citation_order(span: EvidenceSpan) -> tuple[str, int, int, int]:
    """The order `canonical_document` writes citations in, so both agree."""
    return (
        span.path,
        span.byte_start,
        span.byte_end,
        span.id if span.id is not None else -1,
    )


def _assertion_citations(
    root: Path,
    candidate: AssertionCandidate,
    cache: dict[str, bytes | _ReadError],
) -> tuple[list[Citation], str | None]:
    """Re-read every cited range, or report the first reason to withhold the claim.

    Ranges are laid out largest-first per file so that a citation already contained
    in a wider one is recognised as a duplicate rather than rendered twice. The
    listed order stays canonical -- the order the retrieval document was built in --
    so a caller sees citations in the same sequence every time.
    """
    ordered = sorted(candidate.spans, key=_citation_order)
    rendered: dict[int, tuple[str, int]] = {}
    kept: dict[str, list[tuple[int, int]]] = {}
    lines: dict[int, tuple[int, int]] = {}

    containment_order = sorted(
        range(len(ordered)),
        key=lambda position: (
            ordered[position].path,
            ordered[position].byte_start,
            -ordered[position].byte_end,
        ),
    )
    for position in containment_order:
        span = ordered[position]
        try:
            raw = _cached_file(root, span.path, cache)
        except _ReadError as error:
            return [], f"citation_{error.code}"
        if (
            span.byte_start < 0
            or span.byte_end <= span.byte_start
            or span.byte_end > len(raw)
        ):
            return [], "citation_changed"
        cited = raw[span.byte_start : span.byte_end]
        if content_hash(cited) != span.content_hash:
            return [], "citation_changed"

        line_start, line_end = _live_lines(raw, span.byte_start, span.byte_end)
        lines[position] = (line_start, line_end)
        covered = any(
            start <= span.byte_start and span.byte_end <= end
            for start, end in kept.get(span.path, ())
        )
        if covered:
            continue
        kept.setdefault(span.path, []).append((span.byte_start, span.byte_end))
        source = number_source(cited.decode("utf-8", "replace"), line_start)
        rendered[position] = (source, content_bytes(source))

    citations = []
    for position, span in enumerate(ordered):
        line_start, line_end = lines[position]
        source_and_bytes = rendered.get(position)
        citations.append(
            Citation(
                path=span.path,
                line_start=line_start,
                line_end=line_end,
                byte_start=span.byte_start,
                byte_end=span.byte_end,
                content_hash=span.content_hash,
                source=None if source_and_bytes is None else source_and_bytes[0],
                content_bytes=0 if source_and_bytes is None else source_and_bytes[1],
                duplicate=source_and_bytes is None,
            )
        )
    return citations, None


def _assemble_assertion(
    candidate: AssertionCandidate,
    root: Path,
    subject_row: sqlite3.Row | None,
    related: tuple[RelatedSymbol, ...],
    cache: dict[str, bytes | _ReadError],
    remaining_bytes: int,
) -> tuple[AssertionEvidence | None, str]:
    """Build one semantic result, or say why the whole claim is being withheld."""
    if not candidate.spans:
        return None, "no_citations"
    citations, reason = _assertion_citations(root, candidate, cache)
    if reason is not None:
        return None, reason

    citation_bytes = sum(citation.content_bytes for citation in citations)
    if citation_bytes > remaining_bytes:
        return None, "budget"

    subject: EvidenceSection | None = None
    subject_reason: str | None = None
    subject_range: tuple[int, int] | None = None
    if subject_row is None:
        subject_reason = "subject_not_indexed"
    else:
        try:
            subject = _section_from_row(root, subject_row, cache)
            subject_range = (
                int(subject_row["byte_start"]),
                int(subject_row["byte_end"]),
            )
        except EvidenceError as error:
            subject, subject_reason = None, error.code
    if subject is not None and subject_range is not None:
        covered = any(
            citation.source is not None
            and citation.path == subject.path
            and citation.byte_start <= subject_range[0]
            and subject_range[1] <= citation.byte_end
            for citation in citations
        )
        if covered:
            subject, subject_reason = None, "covered_by_citation"
        elif subject.content_bytes > remaining_bytes - citation_bytes:
            subject, subject_reason = None, "budget"

    used = citation_bytes + (0 if subject is None else subject.content_bytes)
    return (
        AssertionEvidence(
            assertion_id=candidate.assertion_id,
            kind=candidate.kind,
            claim=candidate.claim,
            generator=candidate.generator,
            subject_qualname=candidate.subject_qualname,
            subject_symbol_id=candidate.subject_symbol_id,
            verdicts=candidate.verdicts,
            freshness=candidate.freshness,
            conflict=candidate.conflict,
            citations=tuple(citations),
            subject=subject,
            subject_reason=subject_reason,
            related=related,
            content_bytes=used,
        ),
        "served",
    )


def assemble_candidate_evidence(
    conn: sqlite3.Connection,
    repo_root: Path,
    candidates: Iterable[Candidate],
    *,
    budget_bytes: int,
) -> CandidateEvidenceBundle:
    """Hydrate mixed source and semantic candidates under one byte budget.

    A source candidate keeps Phase 1 behaviour exactly: indexed source that no
    longer matches its recorded hash raises, because a source hit IS its source.
    A semantic candidate is all-or-nothing instead -- a claim whose citations no
    longer read back as cited is withheld with a reason, never shown with a
    partial basis -- because the claim is not the evidence, and half a basis is
    worse than none.
    """
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be >= 0")
    budget_bytes = min(budget_bytes, MAX_EVIDENCE_BYTES)
    root = Path(repo_root).resolve()

    ordered: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.key not in seen:
            seen.add(candidate.key)
            ordered.append(candidate)
    if not ordered:
        return CandidateEvidenceBundle((), budget_bytes, 0, ())

    subject_ids = [
        candidate.subject_symbol_id
        for candidate in ordered
        if isinstance(candidate, AssertionCandidate)
        and candidate.subject_symbol_id is not None
    ]
    rows = _symbol_rows(
        conn,
        [
            candidate.symbol_id
            for candidate in ordered
            if isinstance(candidate, SourceCandidate)
        ]
        + subject_ids,
    )
    related_by_subject = _related_symbols(conn, subject_ids)

    cache: dict[str, bytes | _ReadError] = {}
    results: list[CandidateEvidence] = []
    omitted: list[OmittedCandidate] = []
    used_bytes = 0
    for candidate in ordered:
        rank = len(results) + len(omitted) + 1
        if isinstance(candidate, SourceCandidate):
            row = rows.get(candidate.symbol_id)
            if row is None:
                omitted.append(OmittedCandidate(candidate.key, "not_indexed"))
                continue
            section = _section_from_row(root, row, cache)
            if section.content_bytes > budget_bytes - used_bytes:
                omitted.append(OmittedCandidate(candidate.key, "budget"))
                continue
            used_bytes += section.content_bytes
            results.append(
                CandidateEvidence(
                    rank=rank,
                    candidate_type="source",
                    candidate_key=candidate.key,
                    tier=candidate.tier,
                    modality=candidate.modality,
                    score=candidate.score,
                    section=section,
                )
            )
            continue

        evidence, reason = _assemble_assertion(
            candidate,
            root,
            rows.get(candidate.subject_symbol_id)
            if candidate.subject_symbol_id is not None
            else None,
            related_by_subject.get(candidate.subject_symbol_id or -1, ()),
            cache,
            budget_bytes - used_bytes,
        )
        if evidence is None:
            omitted.append(OmittedCandidate(candidate.key, reason))
            continue
        used_bytes += evidence.content_bytes
        results.append(
            CandidateEvidence(
                rank=rank,
                candidate_type="assertion",
                candidate_key=candidate.key,
                tier=candidate.tier,
                modality=candidate.modality,
                score=candidate.score,
                assertion=evidence,
            )
        )
    return CandidateEvidenceBundle(
        tuple(results), budget_bytes, used_bytes, tuple(omitted)
    )
