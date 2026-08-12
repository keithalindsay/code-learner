"""Rank fusion across source and evidence-bound semantic candidates."""
from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from ..assertions import store
from ..assertions.policy import (
    PRODUCTION_POLICY,
    ServingPolicy,
    evaluate_metadata,
)
from ..index.embed import Embedder
from .assertions import (
    load_assertions_by_ids,
    search_assertions,
    verdict_summaries,
    verify_assertions,
)
from .fuse import RRF_K
from .lexical import Hit
from .rerank import Reranker
from .search import CANDIDATE_MULTIPLIER, SearchResult, search
from .types import (
    AssertionCandidate,
    Candidate,
    CandidateSearchResult,
    Freshness,
    ScoreContribution,
    SourceCandidate,
    VerdictSummary,
)

SEMANTIC_WEIGHTS: Mapping[str, float] = {
    "source_lexical": 1.0,
    "source_dense": 1.0,
    "source_graph": 0.3,
    "assertion_lexical": 1.0,
    "assertion_subject": 0.5,
    "source_assertions": 0.5,
}

_SQL_BATCH = 500


def _normalized_claim(claim: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", claim).casefold().split())


def _numeric_id(candidate: Candidate) -> int:
    if isinstance(candidate, SourceCandidate):
        return candidate.symbol_id
    return candidate.assertion_id


def _source_provenance(candidate: SourceCandidate) -> tuple[int, str]:
    """Prefer semantic provenance, then any explanation, independent of list order."""
    return (2 if candidate.via.startswith("assertion:") else bool(candidate.via), candidate.via)


def mixed_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Candidate]],
    *,
    k: int,
    max_tier: int = 2,
    weights: Mapping[str, float] = SEMANTIC_WEIGHTS,
    debug: bool = False,
) -> list[Candidate]:
    """Fuse tagged candidates without allowing their numeric IDs to collide."""
    if k <= 0:
        return []

    scores: dict[str, float] = {}
    candidates: dict[str, Candidate] = {}
    tiers: dict[str, int] = {}
    modalities: dict[str, list[str]] = {}
    contributions: dict[str, list[ScoreContribution]] = {}

    for modality, ranked in ranked_lists.items():
        weight = weights.get(modality, 1.0)
        seen: set[str] = set()
        for rank, candidate in enumerate(ranked, start=1):
            if candidate.key in seen:
                continue
            seen.add(candidate.key)
            value = weight / (RRF_K + rank)
            scores[candidate.key] = scores.get(candidate.key, 0.0) + value
            previous = candidates.setdefault(candidate.key, candidate)
            if (
                isinstance(previous, SourceCandidate)
                and isinstance(candidate, SourceCandidate)
                and _source_provenance(candidate) > _source_provenance(previous)
            ):
                candidates[candidate.key] = candidate
            tiers[candidate.key] = min(
                tiers.get(candidate.key, candidate.tier), candidate.tier
            )
            modalities.setdefault(candidate.key, []).append(modality)
            if debug:
                contributions.setdefault(candidate.key, []).append(
                    ScoreContribution(modality, rank, weight, value)
                )

    eligible = {
        key: candidate
        for key, candidate in candidates.items()
        if tiers[key] <= max_tier
    }
    normalized_claims: dict[tuple[str, str], set[str]] = {}
    for candidate in eligible.values():
        if isinstance(candidate, AssertionCandidate):
            group = (candidate.subject_qualname, candidate.kind)
            normalized_claims.setdefault(group, set()).add(
                _normalized_claim(candidate.claim)
            )

    ordered = sorted(
        eligible.values(),
        key=lambda candidate: (
            -scores[candidate.key],
            0 if isinstance(candidate, SourceCandidate) else 1,
            _numeric_id(candidate),
        ),
    )
    fused: list[Candidate] = []
    for candidate in ordered[:k]:
        score = scores[candidate.key]
        modality = "+".join(sorted(modalities[candidate.key]))
        candidate_contributions = tuple(contributions.get(candidate.key, ()))
        if isinstance(candidate, SourceCandidate):
            fused.append(
                replace(
                    candidate,
                    score=score,
                    modality=modality,
                    tier=tiers[candidate.key],
                    contributions=candidate_contributions,
                )
            )
        else:
            group = (candidate.subject_qualname, candidate.kind)
            fused.append(
                replace(
                    candidate,
                    score=score,
                    modality=modality,
                    conflict=len(normalized_claims[group]) > 1,
                    contributions=candidate_contributions,
                )
            )
    return fused


def _source_modalities(result: SearchResult) -> dict[str, tuple[SourceCandidate, ...]]:
    """Keep source modality membership, ordered by the fused/reranked source pool."""
    pool_order = {hit.symbol_id: rank for rank, hit in enumerate(result.hits)}
    converted: dict[str, tuple[SourceCandidate, ...]] = {}
    for modality, hits in result.per_modality.items():
        in_pool = [hit for hit in hits if hit.symbol_id in pool_order]
        in_pool.sort(key=lambda hit: pool_order[hit.symbol_id])
        converted[f"source_{modality}"] = tuple(
            SourceCandidate.from_hit(hit) for hit in in_pool
        )
    return converted


def _load_source_candidates(
    conn: sqlite3.Connection, symbol_ids: Sequence[int]
) -> list[SourceCandidate]:
    """Batch-load a bounded source set in first-occurrence input order."""
    ordered_ids = list(dict.fromkeys(symbol_ids))
    loaded: dict[int, SourceCandidate] = {}
    for start in range(0, len(ordered_ids), _SQL_BATCH):
        batch = ordered_ids[start : start + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            "SELECT s.id AS symbol_id, s.qualname, s.kind, s.line_start, "  # noqa: S608
            "s.line_end, f.path, f.is_test, COALESCE(c.header, '') AS header "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "LEFT JOIN chunks c ON c.symbol_id = s.id "
            f"WHERE s.id IN ({placeholders})",
            tuple(batch),
        )
        for row in rows:
            hit = Hit(
                symbol_id=int(row["symbol_id"]),
                qualname=str(row["qualname"]),
                kind=str(row["kind"]),
                path=str(row["path"]),
                line_start=int(row["line_start"]),
                line_end=int(row["line_end"]),
                score=0.0,
                modality="source_assertions",
                header=str(row["header"]),
                is_test=bool(row["is_test"]),
            )
            loaded[hit.symbol_id] = SourceCandidate.from_hit(hit)
    return [loaded[symbol_id] for symbol_id in ordered_ids if symbol_id in loaded]


def _attached_assertion_ids(
    conn: sqlite3.Connection,
    subject_ids: Sequence[int],
    *,
    max_candidates: int,
) -> list[int]:
    """Find attached active assertions with chunked IN queries and a hard cap."""
    ordered_ids = list(dict.fromkeys(subject_ids))
    found: list[int] = []
    for start in range(0, len(ordered_ids), _SQL_BATCH):
        remaining = max_candidates - len(found)
        if remaining <= 0:
            break
        batch = ordered_ids[start : start + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            "SELECT id FROM assertions WHERE status = ? "  # noqa: S608
            f"AND subject_symbol_id IN ({placeholders}) "
            "ORDER BY id LIMIT ?",
            (store.STATUS_ACTIVE, *batch, remaining),
        )
        found.extend(int(row["id"]) for row in rows)
    return found


def _attached_assertions(
    conn: sqlite3.Connection,
    repo_root: Path,
    source_pool: Sequence[SourceCandidate],
    *,
    policy: ServingPolicy,
    max_candidates: int,
) -> list[AssertionCandidate]:
    source_rank = {
        candidate.symbol_id: rank for rank, candidate in enumerate(source_pool)
    }
    assertion_ids = _attached_assertion_ids(
        conn, list(source_rank), max_candidates=max_candidates
    )
    if not assertion_ids:
        return []

    loaded = load_assertions_by_ids(conn, assertion_ids)
    summaries = verdict_summaries(conn, assertion_ids)
    accepted: dict[int, tuple[VerdictSummary, ...]] = {}
    eligible_ids: list[int] = []
    for assertion in loaded:
        decision = evaluate_metadata(
            assertion, cast(tuple, summaries.get(assertion.id, ())), policy
        )
        if decision.eligible:
            eligible_ids.append(assertion.id)
            accepted[assertion.id] = cast(
                tuple[VerdictSummary, ...], decision.accepted
            )

    verified = verify_assertions(conn, repo_root, eligible_ids)
    candidates = [
        AssertionCandidate(
            assertion_id=assertion.id,
            subject_symbol_id=assertion.subject_symbol_id,
            subject_qualname=assertion.subject_qualname,
            kind=assertion.kind,
            claim=assertion.claim,
            generator=assertion.generator,
            status=assertion.status,
            verdicts=accepted[assertion.id],
            freshness=Freshness(verified=True, method="hash"),
            spans=assertion.spans,
            score=0.0,
            modality="assertion_subject",
            conflict=False,
            contributions=(),
        )
        for assertion in verified
        if assertion.subject_symbol_id in source_rank
    ]
    candidates.sort(
        key=lambda candidate: (
            source_rank[cast(int, candidate.subject_symbol_id)],
            candidate.assertion_id,
        )
    )
    return candidates


def _promoted_sources(
    conn: sqlite3.Connection,
    assertions: Sequence[AssertionCandidate],
    source_pool: Sequence[SourceCandidate],
) -> list[SourceCandidate]:
    source_by_id = {candidate.symbol_id: candidate for candidate in source_pool}
    subject_ids = [
        assertion.subject_symbol_id
        for assertion in assertions
        if assertion.subject_symbol_id is not None
    ]
    missing = [symbol_id for symbol_id in subject_ids if symbol_id not in source_by_id]
    source_by_id.update(
        (candidate.symbol_id, candidate)
        for candidate in _load_source_candidates(conn, missing)
    )

    promoted: list[SourceCandidate] = []
    seen: set[int] = set()
    for assertion in assertions:
        symbol_id = assertion.subject_symbol_id
        if symbol_id is None or symbol_id in seen or symbol_id not in source_by_id:
            continue
        seen.add(symbol_id)
        promoted.append(
            replace(
                source_by_id[symbol_id],
                score=0.0,
                modality="source_assertions",
                via=f"assertion:{assertion.assertion_id}",
                contributions=(),
            )
        )
    return promoted


def _source_only_result(result: SearchResult) -> CandidateSearchResult:
    return CandidateSearchResult(
        candidates=tuple(SourceCandidate.from_hit(hit) for hit in result.hits),
        per_modality={
            f"source_{modality}": tuple(
                SourceCandidate.from_hit(hit) for hit in hits
            )
            for modality, hits in result.per_modality.items()
        },
    )


def search_candidates(
    conn: sqlite3.Connection,
    repo_root: Path,
    query: str,
    *,
    k: int = 10,
    policy: ServingPolicy = PRODUCTION_POLICY,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    use_lexical: bool = True,
    use_dense: bool = True,
    use_graph: bool = True,
    use_assertions: bool = True,
    debug: bool = False,
) -> CandidateSearchResult:
    """Retrieve source and eligible assertions, then fuse their ranking votes."""
    if k <= 0:
        return CandidateSearchResult(candidates=(), per_modality={})

    source_only = not use_assertions or policy.max_tier < 2
    source_k = k if source_only else k * CANDIDATE_MULTIPLIER
    source_result = search(
        conn,
        query,
        k=source_k,
        embedder=embedder,
        reranker=reranker,
        use_lexical=use_lexical,
        use_dense=use_dense,
        use_graph=use_graph,
    )
    if source_only:
        return _source_only_result(source_result)

    source_pool = tuple(SourceCandidate.from_hit(hit) for hit in source_result.hits)
    per_modality: dict[str, tuple[Candidate, ...]] = dict(
        _source_modalities(source_result)
    )
    assertion_lexical = tuple(
        search_assertions(
            conn,
            Path(repo_root),
            query,
            policy=policy,
            k=source_k,
        )
    )
    assertion_subject = tuple(
        _attached_assertions(
            conn,
            Path(repo_root),
            source_pool,
            policy=policy,
            max_candidates=source_k * CANDIDATE_MULTIPLIER,
        )
    )
    source_assertions = tuple(
        _promoted_sources(conn, assertion_lexical, source_pool)
    )
    per_modality["assertion_lexical"] = assertion_lexical
    per_modality["assertion_subject"] = assertion_subject
    per_modality["source_assertions"] = source_assertions

    candidates = mixed_rank_fusion(
        per_modality,
        k=k,
        max_tier=policy.max_tier,
        debug=debug,
    )
    return CandidateSearchResult(
        candidates=tuple(candidates), per_modality=per_modality
    )
