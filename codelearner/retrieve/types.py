"""Tagged retrieval candidates that preserve source and assertion semantics."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..ingest.types import TIER_INFERRED
from ..tier import tier_of
from .lexical import Hit

if TYPE_CHECKING:
    from ..assertions.store import EvidenceSpan


@dataclass(frozen=True)
class VerdictSummary:
    judge: str
    verdict: str
    rationale: str | None


@dataclass(frozen=True)
class Freshness:
    verified: bool
    method: str


@dataclass(frozen=True)
class ScoreContribution:
    modality: str
    rank: int
    weight: float
    value: float


@dataclass(frozen=True)
class SourceCandidate:
    """A source search hit with its tier made explicit for mixed retrieval."""

    symbol_id: int
    qualname: str
    kind: str
    path: str
    line_start: int
    line_end: int
    score: float
    modality: str
    header: str
    is_test: bool
    via: str
    tier: int
    contributions: tuple[ScoreContribution, ...] = ()

    @classmethod
    def from_hit(cls, hit: Hit) -> SourceCandidate:
        """Copy a source-only hit without changing its stable public shape."""
        return cls(
            symbol_id=hit.symbol_id,
            qualname=hit.qualname,
            kind=hit.kind,
            path=hit.path,
            line_start=hit.line_start,
            line_end=hit.line_end,
            score=hit.score,
            modality=hit.modality,
            header=hit.header,
            is_test=hit.is_test,
            via=hit.via,
            tier=tier_of(hit),
        )

    @property
    def key(self) -> str:
        return f"source:{self.symbol_id}"


@dataclass(frozen=True)
class AssertionCandidate:
    """A semantic claim that remains distinct from its subject source symbol."""

    assertion_id: int
    subject_symbol_id: int | None
    subject_qualname: str
    kind: str
    claim: str
    generator: str | None
    status: str
    verdicts: tuple[VerdictSummary, ...]
    freshness: Freshness
    spans: tuple[EvidenceSpan, ...]
    score: float
    modality: str
    conflict: bool
    contributions: tuple[ScoreContribution, ...]

    @property
    def key(self) -> str:
        return f"assertion:{self.assertion_id}"

    @property
    def tier(self) -> int:
        return TIER_INFERRED


Candidate = SourceCandidate | AssertionCandidate


@dataclass(frozen=True)
class CandidateSearchResult:
    """The ordered candidates and their immutable modality-level explanations."""

    candidates: tuple[Candidate, ...]
    per_modality: Mapping[str, tuple[Candidate, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self,
            "per_modality",
            MappingProxyType(
                {
                    modality: tuple(candidates)
                    for modality, candidates in self.per_modality.items()
                }
            ),
        )
