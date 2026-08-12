"""Immutable source evidence response types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..tier import TIER_LABELS

if TYPE_CHECKING:
    from ..retrieve.types import Freshness, VerdictSummary


@dataclass(frozen=True)
class EvidenceSection:
    symbol_id: int
    qualname: str
    path: str
    line_start: int
    line_end: int
    content_hash: str
    source: str
    content_bytes: int


@dataclass(frozen=True)
class EvidenceBundle:
    sections: tuple[EvidenceSection, ...]
    budget_bytes: int
    used_bytes: int
    sections_omitted: int
    omitted_symbol_ids: tuple[int, ...]

    @property
    def truncated(self) -> bool:
        return self.sections_omitted > 0

    def as_json(self) -> dict[str, object]:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "truncated": self.truncated,
            "sections_omitted": self.sections_omitted,
            "omitted_symbol_ids": list(self.omitted_symbol_ids),
            "sections": [
                {
                    "symbol_id": section.symbol_id,
                    "qualname": section.qualname,
                    "path": section.path,
                    "line_start": section.line_start,
                    "line_end": section.line_end,
                    "content_hash": section.content_hash,
                    "content_bytes": section.content_bytes,
                    "source": section.source,
                }
                for section in self.sections
            ],
        }


def _section_json(section: EvidenceSection) -> dict[str, object]:
    return {
        "symbol_id": section.symbol_id,
        "qualname": section.qualname,
        "path": section.path,
        "line_start": section.line_start,
        "line_end": section.line_end,
        "content_hash": section.content_hash,
        "content_bytes": section.content_bytes,
        "source": section.source,
    }


@dataclass(frozen=True)
class Citation:
    """One cited byte range, re-read and re-numbered against the current file.

    ``source`` is ``None`` when this exact range is already rendered by a wider
    citation in the same result. The citation is still listed -- what a claim rests
    on is part of the claim -- but its bytes are neither repeated nor charged twice.
    """

    path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    content_hash: str
    source: str | None
    content_bytes: int
    duplicate: bool

    def as_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "content_hash": self.content_hash,
            "content_bytes": self.content_bytes,
            "duplicate": self.duplicate,
            "source": self.source,
        }


@dataclass(frozen=True)
class RelatedSymbol:
    """A resolved caller or callee of the subject, for orientation only."""

    relation: str
    symbol_id: int
    qualname: str
    path: str
    line: int

    def as_json(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "symbol_id": self.symbol_id,
            "qualname": self.qualname,
            "path": self.path,
            "line": self.line,
            "tier": TIER_LABELS[1],
        }


@dataclass(frozen=True)
class AssertionEvidence:
    """A semantic claim with everything needed to check it, or nothing at all.

    Assembled all-or-nothing: if any cited range no longer reads back exactly as
    cited, the whole result is withheld rather than shown with a partial basis.
    ``subject`` is the one best-effort part -- context, not grounds -- so its
    absence is explained by ``subject_reason`` instead of dropping the claim.
    """

    assertion_id: int
    kind: str
    claim: str
    generator: str | None
    subject_qualname: str
    subject_symbol_id: int | None
    verdicts: tuple[VerdictSummary, ...]
    freshness: Freshness
    conflict: bool
    citations: tuple[Citation, ...]
    subject: EvidenceSection | None
    subject_reason: str | None
    related: tuple[RelatedSymbol, ...]
    content_bytes: int

    def as_json(self) -> dict[str, object]:
        return {
            "assertion_id": self.assertion_id,
            "assertion_kind": self.kind,
            "claim": self.claim,
            "generator": self.generator,
            "subject_qualname": self.subject_qualname,
            "subject_symbol_id": self.subject_symbol_id,
            "conflict": self.conflict,
            "freshness": {
                "verified": self.freshness.verified,
                "method": self.freshness.method,
            },
            "verdicts": [
                {
                    "judge": verdict.judge,
                    "verdict": verdict.verdict,
                    "rationale": verdict.rationale,
                }
                for verdict in self.verdicts
            ],
            "citations": [citation.as_json() for citation in self.citations],
            "subject": None if self.subject is None else _section_json(self.subject),
            "subject_reason": self.subject_reason,
            "related": [related.as_json() for related in self.related],
            "content_bytes": self.content_bytes,
        }


@dataclass(frozen=True)
class CandidateEvidence:
    """One ranked result, tagged by what kind of thing it is."""

    rank: int
    candidate_type: str
    candidate_key: str
    tier: int
    modality: str
    score: float
    section: EvidenceSection | None = None
    assertion: AssertionEvidence | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "candidate_type": self.candidate_type,
            "candidate_key": self.candidate_key,
            "tier": TIER_LABELS[self.tier],
            "tier_n": self.tier,
            "modality": self.modality,
            "score": round(float(self.score), 6),
            "section": None if self.section is None else _section_json(self.section),
            "assertion": None if self.assertion is None else self.assertion.as_json(),
        }


@dataclass(frozen=True)
class OmittedCandidate:
    """A candidate that was retrieved but not served, and why."""

    key: str
    reason: str

    def as_json(self) -> dict[str, object]:
        return {"candidate_key": self.key, "reason": self.reason}


@dataclass(frozen=True)
class CandidateEvidenceBundle:
    """Ordered candidate evidence under one byte budget."""

    results: tuple[CandidateEvidence, ...]
    budget_bytes: int
    used_bytes: int
    omitted: tuple[OmittedCandidate, ...]

    @property
    def truncated(self) -> bool:
        return bool(self.omitted)

    def as_json(self) -> dict[str, object]:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "truncated": self.truncated,
            "results": [result.as_json() for result in self.results],
            "omitted": [omitted.as_json() for omitted in self.omitted],
        }
