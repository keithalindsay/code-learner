"""Source evidence response types and rendering helpers."""

from .assemble import (
    MAX_EVIDENCE_BYTES,
    MAX_RELATED_SYMBOLS,
    MAX_SOURCE_FILE_BYTES,
    EvidenceError,
    assemble_candidate_evidence,
    assemble_evidence,
)
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

__all__ = [
    "AssertionEvidence",
    "CandidateEvidence",
    "CandidateEvidenceBundle",
    "Citation",
    "EvidenceBundle",
    "EvidenceError",
    "EvidenceSection",
    "MAX_EVIDENCE_BYTES",
    "MAX_RELATED_SYMBOLS",
    "MAX_SOURCE_FILE_BYTES",
    "OmittedCandidate",
    "RelatedSymbol",
    "assemble_candidate_evidence",
    "assemble_evidence",
    "content_bytes",
    "number_source",
]
