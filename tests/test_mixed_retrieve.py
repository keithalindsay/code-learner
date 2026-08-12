"""Tagged fusion and policy-gated promotion for mixed retrieval."""
from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from codelearner.assertions import store
from codelearner.assertions.policy import PRODUCTION_POLICY, RESEARCH_PENDING_POLICY, ServingPolicy
from codelearner.ingest import index_repo
from codelearner.retrieve.fuse import RRF_K
from codelearner.retrieve.lexical import Hit
from codelearner.retrieve.mixed import mixed_rank_fusion, search_candidates
from codelearner.retrieve.search import search
from codelearner.retrieve.types import (
    AssertionCandidate,
    Freshness,
    SourceCandidate,
)


def _source(symbol_id: int, *, modality: str = "lexical") -> SourceCandidate:
    return SourceCandidate.from_hit(
        Hit(
            symbol_id=symbol_id,
            qualname=f"pkg.symbol_{symbol_id}",
            kind="function",
            path="pkg.py",
            line_start=symbol_id,
            line_end=symbol_id + 1,
            score=1.0,
            modality=modality,
            header=f"def symbol_{symbol_id}():",
        )
    )


def _claim(
    assertion_id: int,
    *,
    claim: str | None = None,
    subject_symbol_id: int | None = 10,
    subject_qualname: str = "pkg.subject",
    kind: str = "purpose",
) -> AssertionCandidate:
    return AssertionCandidate(
        assertion_id=assertion_id,
        subject_symbol_id=subject_symbol_id,
        subject_qualname=subject_qualname,
        kind=kind,
        claim=claim or f"Claim {assertion_id}",
        generator="test/v1",
        status="active",
        verdicts=(),
        freshness=Freshness(verified=True, method="hash"),
        spans=(),
        score=1.0,
        modality="assertion_lexical",
        conflict=False,
        contributions=(),
    )


def test_source_and_assertion_numeric_ids_are_independent_slots():
    result = mixed_rank_fusion(
        {"source_lexical": (_source(7),), "assertion_lexical": (_claim(7),)},
        k=2,
    )

    assert [candidate.key for candidate in result] == ["source:7", "assertion:7"]


def test_weighted_rrf_records_debug_contributions():
    source = _source(1)
    result = mixed_rank_fusion(
        {
            "source_lexical": (_source(8), source),
            "source_graph": (source,),
        },
        k=1,
        weights={"source_lexical": 2.0, "source_graph": 0.5},
        debug=True,
    )

    assert result[0].key == "source:1"
    assert result[0].score == pytest.approx(
        2.0 / (RRF_K + 2) + 0.5 / (RRF_K + 1)
    )
    assert [
        (item.modality, item.rank, item.weight, item.value)
        for item in result[0].contributions
    ] == [
        ("source_lexical", 2, 2.0, 2.0 / (RRF_K + 2)),
        ("source_graph", 1, 0.5, 0.5 / (RRF_K + 1)),
    ]


def test_debug_off_does_not_retain_score_contributions():
    result = mixed_rank_fusion(
        {"source_lexical": (_source(1),)}, k=1, debug=False
    )

    assert result[0].contributions == ()


def test_ties_are_source_first_then_numeric_id():
    result = mixed_rank_fusion(
        {
            "source_lexical": (_source(2),),
            "source_dense": (_source(1),),
            "assertion_lexical": (_claim(1),),
        },
        k=3,
        weights={
            "source_lexical": 1.0,
            "source_dense": 1.0,
            "assertion_lexical": 1.0,
        },
    )

    assert [candidate.key for candidate in result] == [
        "source:1",
        "source:2",
        "assertion:1",
    ]


def test_duplicate_keys_appear_once_and_vote_once_per_modality():
    source = _source(4)
    result = mixed_rank_fusion(
        {
            "source_lexical": (source, replace(source, score=99.0)),
            "source_dense": (source,),
        },
        k=5,
        weights={"source_lexical": 1.0, "source_dense": 1.0},
        debug=True,
    )

    assert [candidate.key for candidate in result] == ["source:4"]
    assert result[0].score == pytest.approx(2.0 / (RRF_K + 1))
    assert len(result[0].contributions) == 2


def test_promoted_source_provenance_survives_duplicate_fusion_order():
    direct = _source(4)
    promoted = replace(
        direct, modality="source_assertions", via="assertion:9"
    )

    direct_first = mixed_rank_fusion(
        {
            "source_lexical": (direct,),
            "source_assertions": (promoted,),
        },
        k=1,
    )
    promoted_first = mixed_rank_fusion(
        {
            "source_assertions": (promoted,),
            "source_lexical": (direct,),
        },
        k=1,
    )

    assert direct_first[0].via == promoted_first[0].via == "assertion:9"


def test_distinct_normalized_claims_on_one_subject_and_kind_are_conflicts():
    first = _claim(1, claim="Coordinates renewal.")
    second = _claim(2, claim="Cancels renewal.")

    result = mixed_rank_fusion(
        {"assertion_lexical": (first, second)}, k=2
    )

    assert [candidate.conflict for candidate in result] == [True, True]
    assert [candidate.claim for candidate in result] == [
        "Coordinates renewal.",
        "Cancels renewal.",
    ]


def test_nfkc_casefold_and_whitespace_duplicates_are_not_conflicts():
    first_text = "KEEP\t  Lease\nAlive"
    second_text = "keep lease alive"
    result = mixed_rank_fusion(
        {
            "assertion_lexical": (
                _claim(1, claim=first_text),
                _claim(2, claim=second_text),
            )
        },
        k=2,
    )

    assert [candidate.conflict for candidate in result] == [False, False]
    assert [candidate.claim for candidate in result] == [first_text, second_text]


def test_same_text_on_different_claim_kind_is_not_a_conflict():
    result = mixed_rank_fusion(
        {
            "assertion_lexical": (
                _claim(1, claim="Lease stays alive", kind="purpose"),
                _claim(2, claim="Lease stays alive", kind="invariant"),
            )
        },
        k=2,
    )

    assert [candidate.conflict for candidate in result] == [False, False]


def test_facts_only_filters_before_cut_and_refills():
    ranked = {
        "source_lexical": (_source(1), _source(2)),
        "assertion_lexical": (_claim(3),),
    }

    result = mixed_rank_fusion(ranked, k=2, max_tier=1)

    assert [candidate.key for candidate in result] == ["source:1", "source:2"]


def _repository(tmp_path: Path, *, count: int = 3) -> tuple[Path, object]:
    root = tmp_path / "repo"
    root.mkdir()
    source = "".join(
        f"def operation_{number}():\n"
        f"    '''sourceword{number} orchestrationneedle'''\n"
        f"    return {number}\n\n"
        for number in range(count)
    )
    (root / "leases.py").write_text(source)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _add_assertion(
    root: Path,
    conn,
    *,
    subject: str = "leases.operation_0",
    claim: str,
    verdict: str | None = store.VERDICT_SUPPORTED,
    status: str = store.STATUS_ACTIVE,
) -> int:
    source = (root / "leases.py").read_bytes()
    subject_row = conn.execute(
        "SELECT id FROM symbols WHERE qualname = ?", (subject,)
    ).fetchone()
    assertion_id = store.write_assertion(
        conn,
        subject_qualname=subject,
        subject_symbol_id=int(subject_row["id"]),
        kind="purpose",
        claim=claim,
        spans=(store.span_for(root, "leases.py", 0, len(source)),),
        generator="test/v1",
        status=status,
    )
    if verdict is not None:
        store.record_verdict(conn, assertion_id, "judge/v1", verdict)
    return assertion_id


def test_assertion_match_promotes_its_subject_source(tmp_path):
    root, conn = _repository(tmp_path)
    assertion_id = _add_assertion(
        root, conn, claim="semanticneedle coordinates the lease"
    )
    subject_id = int(
        conn.execute(
            "SELECT id FROM symbols WHERE qualname = ?", ("leases.operation_0",)
        ).fetchone()["id"]
    )

    result = search_candidates(
        conn,
        root,
        "semanticneedle",
        k=5,
        embedder=None,
        use_graph=False,
    )

    assert f"assertion:{assertion_id}" in [candidate.key for candidate in result.candidates]
    assert f"source:{subject_id}" in [candidate.key for candidate in result.candidates]
    assert [candidate.key for candidate in result.per_modality["source_assertions"]] == [
        f"source:{subject_id}"
    ]


def test_source_match_promotes_attached_assertion(tmp_path):
    root, conn = _repository(tmp_path)
    assertion_id = _add_assertion(root, conn, claim="Keeps the owner safe.")

    result = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=10,
        embedder=None,
        use_graph=False,
    )

    assert f"assertion:{assertion_id}" in [candidate.key for candidate in result.candidates]
    assert f"assertion:{assertion_id}" not in [
        candidate.key
        for candidate in result.per_modality.get("assertion_lexical", ())
    ]
    assert f"assertion:{assertion_id}" in [
        candidate.key for candidate in result.per_modality["assertion_subject"]
    ]


def test_subject_promotion_obeys_status_and_verdict_policy(tmp_path):
    root, conn = _repository(tmp_path)
    pending = _add_assertion(root, conn, claim="Pending claim", verdict=None)
    rejected = _add_assertion(
        root, conn, claim="Rejected claim", status=store.STATUS_REJECTED
    )
    stale = _add_assertion(
        root, conn, claim="Stale claim", status=store.STATUS_STALE
    )

    production = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=10,
        policy=PRODUCTION_POLICY,
        embedder=None,
        use_graph=False,
    )
    research = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=10,
        policy=RESEARCH_PENDING_POLICY,
        embedder=None,
        use_graph=False,
    )

    assert not any(
        isinstance(candidate, AssertionCandidate) for candidate in production.candidates
    )
    assert [
        candidate.assertion_id
        for candidate in research.candidates
        if isinstance(candidate, AssertionCandidate)
    ] == [pending]
    assert rejected not in [
        candidate.assertion_id
        for candidate in research.candidates
        if isinstance(candidate, AssertionCandidate)
    ]
    assert stale not in [
        candidate.assertion_id
        for candidate in research.candidates
        if isinstance(candidate, AssertionCandidate)
    ]


def test_assertions_disabled_preserves_source_only_search(tmp_path):
    root, conn = _repository(tmp_path, count=8)
    expected = search(
        conn,
        "orchestrationneedle",
        k=4,
        embedder=None,
        use_graph=False,
    )

    result = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=4,
        embedder=None,
        use_graph=False,
        use_assertions=False,
    )

    assert [candidate.key for candidate in result.candidates] == [
        f"source:{hit.symbol_id}" for hit in expected.hits
    ]
    assert [candidate.score for candidate in result.candidates] == [
        hit.score for hit in expected.hits
    ]


def test_facts_policy_skips_assertions_and_returns_a_full_source_page(tmp_path, monkeypatch):
    root, conn = _repository(tmp_path, count=8)

    def unexpected_assertion_search(*args, **kwargs):
        raise AssertionError("tier-2 retrieval ran under a tier-1 policy")

    monkeypatch.setattr(
        "codelearner.retrieve.mixed.search_assertions", unexpected_assertion_search
    )
    result = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=4,
        policy=ServingPolicy(max_tier=1),
        embedder=None,
        use_graph=False,
    )

    assert len(result.candidates) == 4
    assert all(isinstance(candidate, SourceCandidate) for candidate in result.candidates)


def test_missing_source_embedder_does_not_disable_assertion_fts(tmp_path):
    root, conn = _repository(tmp_path)
    assertion_id = _add_assertion(root, conn, claim="semanticneedle only in claim")

    result = search_candidates(
        conn,
        root,
        "semanticneedle",
        k=3,
        embedder=None,
        use_lexical=False,
        use_dense=True,
        use_graph=False,
    )

    assert f"assertion:{assertion_id}" in [candidate.key for candidate in result.candidates]


class _RecordingReranker:
    def __init__(self) -> None:
        self.received: list[Hit] = []
        self.returned: list[Hit] = []

    def rerank(self, query: str, hits, k: int = 10) -> list[Hit]:
        self.received = list(hits)
        assert all(isinstance(hit, Hit) for hit in hits)
        self.returned = list(reversed(hits))[:k]
        return self.returned


def test_reranker_sees_only_source_hits_before_semantic_fusion(tmp_path):
    root, conn = _repository(tmp_path)
    assertion_id = _add_assertion(root, conn, claim="Attached semantic assertion")
    reranker = _RecordingReranker()

    result = search_candidates(
        conn,
        root,
        "orchestrationneedle",
        k=5,
        embedder=None,
        reranker=reranker,
        use_graph=False,
    )

    assert reranker.received
    assert f"assertion:{assertion_id}" in [candidate.key for candidate in result.candidates]
    assert [
        candidate.symbol_id
        for candidate in result.candidates
        if isinstance(candidate, SourceCandidate)
    ] == [hit.symbol_id for hit in reranker.returned]
