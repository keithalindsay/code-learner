"""Pure serving-policy decisions for assertion metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ingest.types import TIER_FACT, TIER_INFERRED, TIER_RESOLVED
from . import store

_KNOWN_TIERS = frozenset({TIER_FACT, TIER_RESOLVED, TIER_INFERRED})
_VETO_VERDICTS = frozenset({store.VERDICT_REFUTED, store.VERDICT_UNSUPPORTED})


class _Verdict(Protocol):
    """The verdict metadata policy needs, without a dependency on retrieval."""

    judge: str
    verdict: str
    rationale: str | None


@dataclass(frozen=True)
class ServingPolicy:
    max_tier: int = TIER_INFERRED
    require_verdict: bool = True
    accepted_verdicts: frozenset[str] = frozenset({store.VERDICT_SUPPORTED})
    allow_pending: bool = False

    def __post_init__(self) -> None:
        if self.max_tier not in _KNOWN_TIERS:
            raise ValueError("max_tier must be one of the known retrieval tiers")
        if self.require_verdict == self.allow_pending:
            raise ValueError("a policy must require verdicts or allow pending assertions")
        if not self.accepted_verdicts:
            raise ValueError("accepted_verdicts must not be empty")
        if self.accepted_verdicts & _VETO_VERDICTS:
            raise ValueError("veto verdicts cannot be accepted")


PRODUCTION_POLICY = ServingPolicy()
RESEARCH_PENDING_POLICY = ServingPolicy(require_verdict=False, allow_pending=True)


@dataclass(frozen=True)
class PolicyDecision:
    eligible: bool
    reason: str
    accepted: tuple[_Verdict, ...] = ()


def evaluate_metadata(
    assertion: store.Assertion,
    verdicts: tuple[_Verdict, ...],
    policy: ServingPolicy,
) -> PolicyDecision:
    """Decide eligibility from loaded metadata without touching SQL or the filesystem."""
    if assertion.status != store.STATUS_ACTIVE:
        return PolicyDecision(False, "status")
    if any(verdict.verdict in _VETO_VERDICTS for verdict in verdicts):
        return PolicyDecision(False, "vetoed")

    accepted = tuple(
        verdict for verdict in verdicts if verdict.verdict in policy.accepted_verdicts
    )
    if accepted:
        return PolicyDecision(True, "eligible", accepted)
    if policy.require_verdict:
        return PolicyDecision(False, "verdict_required")
    return PolicyDecision(True, "eligible_pending")
