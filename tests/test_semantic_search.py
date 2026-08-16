"""Phase 2 exit: a question about WHY, answered by a claim that had to earn it.

The repository below is written so that no amount of text search can answer the
query. The symbols are named `op_7a` and `_tbl`, the docstrings say nothing, and the
invariant a reader actually wants -- that the index handed to the table is never
negative -- exists only as a stored, cited, adjudicated claim. That is the whole
point of tier 2, and until this file passed there was no test that made retrieval
prove it end to end.

Every control is here too, because a semantic result that cannot be turned off is
not a measurable one: source-only mode must not reach the claim, `facts_only` must
remove it and refill the page, and pending, refuted and stale claims must be absent
under the production policy no matter how well they match the query.
"""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from codelearner.assertions import store
from codelearner.assertions.policy import (
    PRODUCTION_POLICY,
    RESEARCH_PENDING_POLICY,
    ServingPolicy,
    evaluate_metadata,
)
from codelearner.evidence import assemble_candidate_evidence
from codelearner.ingest import index_repo
from codelearner.ingest.types import TIER_RESOLVED
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import (
    AssertionCandidate,
    SourceCandidate,
    VerdictSummary,
)

OPAQUE = '''def op_7a(value):
    """Step 7a."""
    if value < 0:
        value = 0
    return _tbl(value)


def _tbl(value):
    """Table."""
    return value * 2
'''

CLAIM = (
    "op_7a clamps its argument to zero before the table lookup, so _tbl never "
    "receives a negative index."
)
QUESTION = "what guarantees the table never receives a negative index"


@pytest.fixture()
def indexed(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pipeline.py").write_text(OPAQUE)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _admit(
    root: Path,
    conn,
    *,
    claim: str = CLAIM,
    verdict: str | None = store.VERDICT_SUPPORTED,
    status: str | None = None,
) -> int:
    row = conn.execute(
        "SELECT s.id, s.byte_start, s.byte_end, f.path FROM symbols s "
        "JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
        ("pipeline.op_7a",),
    ).fetchone()
    assertion_id = store.write_assertion(
        conn,
        subject_qualname="pipeline.op_7a",
        subject_symbol_id=int(row["id"]),
        kind="invariant",
        claim=claim,
        spans=[store.span_for(root, row["path"], row["byte_start"], row["byte_end"])],
        generator="test/v1",
        repo_root=root,
    )
    if verdict is not None:
        store.record_verdict(conn, assertion_id, "judge/v1", verdict, "the clamp is there")
    if status == store.STATUS_STALE:
        store.mark_stale(conn, assertion_id, store.REASON_HASH_MISMATCH)
    return assertion_id


def test_a_why_question_returns_the_claim_its_subject_and_current_evidence(indexed):
    root, conn = indexed
    assertion_id = _admit(root, conn)

    result = search_candidates(conn, root, QUESTION, k=5)
    bundle = assemble_candidate_evidence(
        conn, root, result.candidates, budget_bytes=16_384
    )

    claim = next(
        candidate
        for candidate in result.candidates
        if isinstance(candidate, AssertionCandidate)
    )
    assert claim.assertion_id == assertion_id
    assert claim.claim == CLAIM
    assert claim.tier == 2
    assert [verdict.verdict for verdict in claim.verdicts] == ["supported"]
    assert claim.freshness.verified is True

    # The subject is a separate result, not a field of the claim: a claim and the
    # code it is about are two different things to have retrieved.
    assert "source:" in " ".join(
        candidate.key
        for candidate in result.candidates
        if isinstance(candidate, SourceCandidate)
        and candidate.qualname == "pipeline.op_7a"
    )

    served = next(
        entry
        for entry in bundle.results
        if entry.candidate_key == f"assertion:{assertion_id}"
    )
    evidence = served.assertion
    assert evidence is not None
    citation = evidence.citations[0]
    assert citation.path == "pipeline.py"
    assert citation.source is not None
    assert "if value < 0:" in citation.source
    assert citation.source.startswith("1 | def op_7a(value):")
    # Graph context, so a reader can see who is protected by the invariant.
    assert ("callee", "pipeline._tbl") in [
        (related.relation, related.qualname) for related in evidence.related
    ]
    assert bundle.used_bytes <= 16_384


def test_source_only_retrieval_cannot_reach_the_claim(indexed):
    """The control the whole phase is measured against. If text search could answer
    this question, the semantic layer would be buying nothing here."""
    root, conn = indexed
    _admit(root, conn)

    result = search_candidates(conn, root, QUESTION, k=5, use_assertions=False)

    assert all(isinstance(c, SourceCandidate) for c in result.candidates)
    assert not any(
        "negative" in getattr(c, "qualname", "") for c in result.candidates
    )


def test_facts_only_removes_the_claim_and_refills_the_page_with_source(indexed):
    root, conn = indexed
    _admit(root, conn)
    facts_policy = replace(PRODUCTION_POLICY, max_tier=TIER_RESOLVED)

    everything = search_candidates(conn, root, "table value clamps", k=2)
    facts = search_candidates(conn, root, "table value clamps", k=2, policy=facts_policy)

    assert any(isinstance(c, AssertionCandidate) for c in everything.candidates)
    assert all(isinstance(c, SourceCandidate) for c in facts.candidates)
    assert len(facts.candidates) == 2


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        (None, None),
        (store.VERDICT_UNSUPPORTED, None),
        (store.VERDICT_REFUTED, None),
        (store.VERDICT_SUPPORTED, store.STATUS_STALE),
    ],
)
def test_no_claim_without_an_independent_supporting_verdict_is_ever_served(
    indexed, verdict, status
):
    root, conn = indexed
    _admit(root, conn, verdict=verdict, status=status)

    result = search_candidates(conn, root, QUESTION, k=5)

    assert not [c for c in result.candidates if isinstance(c, AssertionCandidate)]


def test_the_research_policy_is_a_library_decision_and_reaches_pending_claims(indexed):
    """Pending claims exist and are findable -- by a caller who names the policy.

    This is the seam the CLI and MCP deliberately do not expose. A generator's
    unjudged output is a research artefact; serving it through the same shape as an
    adjudicated claim is how a pipeline starts publishing what it merely produced.
    """
    root, conn = indexed
    assertion_id = _admit(root, conn, verdict=None)

    production = search_candidates(conn, root, QUESTION, k=5)
    research = search_candidates(conn, root, QUESTION, k=5, policy=RESEARCH_PENDING_POLICY)

    assert not [c for c in production.candidates if isinstance(c, AssertionCandidate)]
    assert [
        c.assertion_id
        for c in research.candidates
        if isinstance(c, AssertionCandidate)
    ] == [assertion_id]


def _decision(status: str, policy: ServingPolicy):
    assertion = store.Assertion(
        id=1,
        subject_qualname="pipeline.op_7a",
        subject_symbol_id=1,
        kind="invariant",
        claim=CLAIM,
        status=status,
        generator="test/v1",
        confidence=None,
        created_at="2026-08-11T00:00:00Z",
        spans=(),
    )
    return evaluate_metadata(
        assertion,
        (VerdictSummary("judge/v1", store.VERDICT_SUPPORTED, None),),
        policy,
    )


def test_a_policy_cannot_be_built_that_serves_rejected_or_stale_claims():
    """There is no dial for this. Status is not a tunable, so a claim a judge refused
    and a claim whose evidence moved stay out of every policy, including the research
    one -- their rejection is a finding, not a strictness setting."""
    for policy in (PRODUCTION_POLICY, RESEARCH_PENDING_POLICY, ServingPolicy()):
        for status in (store.STATUS_REJECTED, store.STATUS_STALE):
            decision = _decision(status, policy)
            assert decision.eligible is False
            assert decision.reason == "status"
