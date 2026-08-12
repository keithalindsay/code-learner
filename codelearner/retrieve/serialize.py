"""One JSON shape for retrieval candidates, shared by every surface.

The CLI and the MCP server must not build result objects of their own. When they
did, the two drifted -- a tier meant one thing in a terminal and another over
stdio -- and nothing in the codebase said which was right. Both call
:func:`candidate_json`, so a change to the published shape is one edit and lands
on both surfaces at once.

Source keys are exactly the ones `tier.hit_json` has always published, so an
existing consumer keeps parsing. `candidate_type` and `candidate_key` are the two
additions, and they are what a consumer switches on: a semantic result is not a
symbol with extra fields, and pretending otherwise is how a claim gets read as a
fact.
"""
from __future__ import annotations

from typing import Any

from ..tier import TIER_LABELS
from .types import AssertionCandidate, Candidate, ScoreContribution, SourceCandidate


def _contributions_json(
    contributions: tuple[ScoreContribution, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "modality": contribution.modality,
            "rank": contribution.rank,
            "weight": contribution.weight,
            "value": contribution.value,
        }
        for contribution in contributions
    ]


def candidate_json(
    candidate: Candidate, rank: int, *, debug: bool = False
) -> dict[str, Any]:
    """One ranked candidate as a stable JSON object.

    `score` is rounded, because the exact float is an artefact of the fusion
    constants rather than a published number. The per-modality contributions that
    produced it are omitted unless `debug` asks for them: they are an explanation
    of the ranker, not part of the answer, and shipping them by default would make
    every internal weight a compatibility promise.
    """
    payload: dict[str, Any] = {
        "rank": rank,
        "tier": TIER_LABELS[candidate.tier],
        "tier_n": candidate.tier,
        "candidate_type": "source" if isinstance(candidate, SourceCandidate) else "assertion",
        "candidate_key": candidate.key,
    }
    if isinstance(candidate, SourceCandidate):
        payload.update(
            {
                "symbol_id": candidate.symbol_id,
                "qualname": candidate.qualname,
                "kind": candidate.kind,
                "path": candidate.path,
                "line_start": candidate.line_start,
                "line_end": candidate.line_end,
                "score": round(float(candidate.score), 6),
                "modality": candidate.modality,
                "is_test": candidate.is_test,
                "via": candidate.via,
            }
        )
    else:
        assertion: AssertionCandidate = candidate
        payload.update(
            {
                "assertion_id": assertion.assertion_id,
                "claim": assertion.claim,
                "assertion_kind": assertion.kind,
                "subject_qualname": assertion.subject_qualname,
                "subject_symbol_id": assertion.subject_symbol_id,
                "generator": assertion.generator,
                "status": assertion.status,
                "score": round(float(assertion.score), 6),
                "modality": assertion.modality,
                "conflict": assertion.conflict,
                "freshness": {
                    "verified": assertion.freshness.verified,
                    "method": assertion.freshness.method,
                },
                "verdicts": [
                    {
                        "judge": verdict.judge,
                        "verdict": verdict.verdict,
                        "rationale": verdict.rationale,
                    }
                    for verdict in assertion.verdicts
                ],
                # As CITED, not as currently located. The bytes are re-verified
                # before a claim is served, so these ranges still hash to what was
                # cited -- but nothing re-numbers the lines, and an edit above them
                # moves where they sit. Ask for evidence to get live coordinates;
                # the names here say which kind of number this is.
                "citations": [
                    {
                        "path": span.path,
                        "cited_line_start": span.line_start,
                        "cited_line_end": span.line_end,
                        "byte_start": span.byte_start,
                        "byte_end": span.byte_end,
                        "content_hash": span.content_hash,
                    }
                    for span in assertion.spans
                ],
            }
        )
    if debug:
        payload["contributions"] = _contributions_json(candidate.contributions)
    return payload
