"""Tier-2 assertion storage: evidence-bound, hash-bound, and adjudicated.

The pipeline that generates claims is not here. This package is the store that
decides which of them are admissible and which are still true, which is the part
that has to hold whether or not the generator is any good.

Two verifiers, deliberately: `store.servable_assertions` re-hashes every cited byte
range unconditionally and is the reference, and `stale.serve_assertions` reaches the
same verdicts through a stat() fast path and reports which stage reached them. The
second is an optimisation of the first, so any repo state where they disagree is a
bug in the second -- which is why a test asserts they agree across every failure mode.
"""

from .stale import (
    METHOD_HASH,
    METHOD_STAT,
    RefreshReport,
    ServedAssertion,
    SpanCheck,
    refresh_staleness,
    serve_assertions,
    verification_state,
)
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
    "METHOD_HASH",
    "METHOD_STAT",
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
    "RefreshReport",
    "ServedAssertion",
    "SpanCheck",
    "assertions_with_status",
    "is_servable",
    "mark_stale",
    "record_verdict",
    "refresh_staleness",
    "serve_assertions",
    "servable_assertions",
    "span_for",
    "span_for_symbol",
    "staleness_events",
    "verdicts_for",
    "verification_state",
    "write_assertion",
]
