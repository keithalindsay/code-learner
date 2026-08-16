"""Serving policy is a pure metadata decision, independent of repository I/O."""
import pytest

from codelearner.assertions.policy import (
    PRODUCTION_POLICY,
    RESEARCH_PENDING_POLICY,
    ServingPolicy,
    evaluate_metadata,
)
from codelearner.assertions.store import Assertion, EvidenceSpan
from codelearner.ingest.types import TIER_FACT, TIER_INFERRED, TIER_RESOLVED
from codelearner.retrieve.types import VerdictSummary


def _assertion(*, status: str = "active") -> Assertion:
    return Assertion(
        id=1,
        subject_qualname="leases.acquire",
        subject_symbol_id=3,
        kind="purpose",
        claim="Coordinates lease renewal.",
        status=status,
        generator="test-generator",
        confidence=0.9,
        created_at="2026-08-11T00:00:00Z",
        spans=(
            EvidenceSpan(
                path="leases.py",
                line_start=10,
                line_end=18,
                byte_start=0,
                byte_end=10,
                content_hash="hash",
            ),
        ),
    )


def _verdicts(verdicts: tuple[str, ...]) -> tuple[VerdictSummary, ...]:
    return tuple(VerdictSummary("judge-1", verdict, None) for verdict in verdicts)


@pytest.mark.parametrize(
    ("status", "verdicts", "policy", "eligible", "reason"),
    [
        ("active", ("supported",), PRODUCTION_POLICY, True, "eligible"),
        ("active", (), PRODUCTION_POLICY, False, "verdict_required"),
        ("active", (), RESEARCH_PENDING_POLICY, True, "eligible_pending"),
        ("active", ("supported", "refuted"), PRODUCTION_POLICY, False, "vetoed"),
        ("active", ("supported", "unsupported"), PRODUCTION_POLICY, False, "vetoed"),
        ("rejected", ("supported",), RESEARCH_PENDING_POLICY, False, "status"),
        ("stale", ("supported",), RESEARCH_PENDING_POLICY, False, "status"),
    ],
)
def test_serving_policy_matrix(status, verdicts, policy, eligible, reason):
    """Lifecycle state wins over verdicts, and adverse verdicts veto support."""
    decision = evaluate_metadata(_assertion(status=status), _verdicts(verdicts), policy)

    assert (decision.eligible, decision.reason) == (eligible, reason)


def test_policy_returns_only_accepted_supporting_verdicts():
    """Renderers can show exactly the evidence that made a claim eligible."""
    support = VerdictSummary("supporting-judge", "supported", "Matches the source.")
    decision = evaluate_metadata(_assertion(), (support,), PRODUCTION_POLICY)

    assert decision.accepted == (support,)


@pytest.mark.parametrize(
    ("max_tier", "eligible", "reason"),
    [
        (TIER_FACT, False, "tier"),
        (TIER_RESOLVED, False, "tier"),
        (TIER_INFERRED, True, "eligible"),
    ],
)
def test_policy_enforces_max_tier_for_assertions(max_tier, eligible, reason):
    """A policy below T2 must reject assertion candidates even when supported."""
    support = VerdictSummary("supporting-judge", "supported", "Matches the source.")
    decision = evaluate_metadata(
        _assertion(),
        (support,),
        ServingPolicy(max_tier=max_tier),
    )

    assert (decision.eligible, decision.reason) == (eligible, reason)


@pytest.mark.parametrize("max_tier", (-1, 3))
def test_policy_rejects_tiers_outside_the_known_model(max_tier):
    """A caller cannot accidentally create a policy beyond T0 through T2."""
    with pytest.raises(ValueError):
        ServingPolicy(max_tier=max_tier)


def test_policy_rejects_pending_and_required_verdicts_together():
    """Pending cannot be allowed by a policy that still requires adjudication."""
    with pytest.raises(ValueError):
        ServingPolicy(allow_pending=True)
