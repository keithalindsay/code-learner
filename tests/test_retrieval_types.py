"""The typed candidate boundary keeps source and semantic identities distinct."""
from dataclasses import FrozenInstanceError

import pytest

from codelearner.assertions.store import EvidenceSpan
from codelearner.retrieve.lexical import Hit
from codelearner.retrieve.types import (
    AssertionCandidate,
    CandidateSearchResult,
    Freshness,
    ScoreContribution,
    SourceCandidate,
    VerdictSummary,
)


def _hit(*, symbol_id: int) -> Hit:
    return Hit(
        symbol_id=symbol_id,
        qualname="leases.acquire",
        kind="function",
        path="leases.py",
        line_start=10,
        line_end=18,
        score=0.75,
        modality="lexical",
        header="def acquire():",
        is_test=False,
        via="",
    )


def _assertion_candidate(*, assertion_id: int) -> AssertionCandidate:
    return AssertionCandidate(
        assertion_id=assertion_id,
        subject_symbol_id=3,
        subject_qualname="leases.acquire",
        kind="purpose",
        claim="Coordinates lease renewal.",
        generator="test-generator",
        status="active",
        verdicts=(VerdictSummary("judge-1", "supported", "Cited code supports it."),),
        freshness=Freshness(verified=True, method="hash"),
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
        score=1.0,
        modality="assertion_lexical",
        conflict=False,
        contributions=(
            ScoreContribution("assertion_lexical", rank=1, weight=1.0, value=0.5),
        ),
    )


def test_candidate_keys_are_type_qualified():
    """A numeric source ID cannot collide with an assertion ID in mixed fusion."""
    source = SourceCandidate.from_hit(_hit(symbol_id=7))
    assertion = _assertion_candidate(assertion_id=7)

    assert source.key == "source:7"
    assert assertion.key == "assertion:7"
    assert source.tier in (0, 1)
    assert assertion.tier == 2


def test_source_candidate_copies_hit_fields_without_changing_the_hit():
    """Candidate conversion preserves source-only search data for later consumers."""
    hit = _hit(symbol_id=4)

    candidate = SourceCandidate.from_hit(hit)

    assert (
        candidate.symbol_id,
        candidate.qualname,
        candidate.kind,
        candidate.path,
        candidate.line_start,
        candidate.line_end,
        candidate.score,
        candidate.modality,
        candidate.header,
        candidate.is_test,
        candidate.via,
    ) == (
        hit.symbol_id,
        hit.qualname,
        hit.kind,
        hit.path,
        hit.line_start,
        hit.line_end,
        hit.score,
        hit.modality,
        hit.header,
        hit.is_test,
        hit.via,
    )


def test_candidates_are_frozen():
    """Fusion cannot silently rewrite a candidate returned by another modality."""
    candidate = _assertion_candidate(assertion_id=1)

    with pytest.raises(FrozenInstanceError):
        candidate.score = 0.0  # type: ignore[misc]


def test_candidate_result_keeps_per_modality_explanations():
    """A mixed result retains the immutable candidate tuples behind its ranking."""
    candidate = SourceCandidate.from_hit(_hit(symbol_id=1))
    result = CandidateSearchResult(
        candidates=(candidate,), per_modality={"source_lexical": (candidate,)}
    )

    assert result.per_modality["source_lexical"] == (candidate,)
