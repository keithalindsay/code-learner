"""Tier-2 assertion storage: evidence-bound, hash-bound, and adjudicated.

The pipeline that generates claims is not here. This package is the store that
decides which of them are admissible and which are still true, which is the part
that has to hold whether or not the generator is any good.
"""

from .store import (
    REASON_FILE_MISSING,
    REASON_HASH_MISMATCH,
    REASON_NO_EVIDENCE,
    REASON_SPAN_TRUNCATED,
    STATUS_ACTIVE,
    STATUS_REJECTED,
    STATUS_STALE,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    VERDICT_UNSUPPORTED,
    Assertion,
    EvidenceRequired,
    EvidenceSpan,
    assertions_with_status,
    is_servable,
    mark_stale,
    record_verdict,
    servable_assertions,
    span_for,
    span_for_symbol,
    staleness_events,
    verdicts_for,
    write_assertion,
)

__all__ = [
    "REASON_FILE_MISSING",
    "REASON_HASH_MISMATCH",
    "REASON_NO_EVIDENCE",
    "REASON_SPAN_TRUNCATED",
    "STATUS_ACTIVE",
    "STATUS_REJECTED",
    "STATUS_STALE",
    "VERDICT_REFUTED",
    "VERDICT_SUPPORTED",
    "VERDICT_UNSUPPORTED",
    "Assertion",
    "EvidenceRequired",
    "EvidenceSpan",
    "assertions_with_status",
    "is_servable",
    "mark_stale",
    "record_verdict",
    "servable_assertions",
    "span_for",
    "span_for_symbol",
    "staleness_events",
    "verdicts_for",
    "write_assertion",
]
