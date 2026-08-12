"""Bounded lexical retrieval of policy-eligible, live semantic assertions."""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..assertions import search_index, store
from ..assertions.policy import (
    PRODUCTION_POLICY,
    ServingPolicy,
    evaluate_metadata,
)
from .lexical import escape_fts_query
from .types import AssertionCandidate, Freshness, VerdictSummary


class AssertionSearchUnavailable(RuntimeError):
    """The index lacks the derived structures required for assertion search."""


@dataclass(frozen=True)
class _SearchRow:
    assertion_id: int
    score: float


def load_assertions_by_ids(
    conn: sqlite3.Connection, assertion_ids: Sequence[int]
) -> list[store.Assertion]:
    """Load a bounded assertion set in first-occurrence input order."""
    return store.load_assertions_by_ids(conn, assertion_ids)


def verdict_summaries(
    conn: sqlite3.Connection, assertion_ids: Sequence[int]
) -> dict[int, tuple[VerdictSummary, ...]]:
    """Batch-load verdicts for supplied IDs only, preserving ID and verdict order."""
    ordered_ids = list(dict.fromkeys(assertion_ids))
    summaries: dict[int, list[VerdictSummary]] = {
        assertion_id: [] for assertion_id in ordered_ids
    }
    for batch in store._chunks(ordered_ids):
        placeholders = ",".join("?" * len(batch))
        for row in conn.execute(
            "SELECT assertion_id, judge, verdict, rationale FROM verdicts "  # noqa: S608
            f"WHERE assertion_id IN ({placeholders}) ORDER BY id",
            tuple(batch),
        ):
            summaries[int(row["assertion_id"])].append(
                VerdictSummary(
                    judge=str(row["judge"]),
                    verdict=str(row["verdict"]),
                    rationale=(
                        None if row["rationale"] is None else str(row["rationale"])
                    ),
                )
            )
    return {
        assertion_id: tuple(summaries[assertion_id])
        for assertion_id in ordered_ids
        if summaries[assertion_id]
    }


def verify_assertions(
    conn: sqlite3.Connection,
    repo_root: Path,
    assertion_ids: Sequence[int],
) -> list[store.Assertion]:
    """Verify only supplied IDs through the authoritative store transition."""
    return store.verify_assertions(conn, repo_root, assertion_ids)


def _verify_loaded_assertions(
    conn: sqlite3.Connection,
    repo_root: Path,
    assertions: Sequence[store.Assertion],
) -> list[store.Assertion]:
    """Keep retrieval's metadata-before-I/O ordering explicit at the seam."""
    return verify_assertions(conn, repo_root, [assertion.id for assertion in assertions])


def _search_page(
    conn: sqlite3.Connection,
    match: str,
    *,
    limit: int,
    offset: int,
) -> list[_SearchRow]:
    rows = conn.execute(
        "SELECT rowid AS assertion_id, bm25(assertions_fts) AS score "
        "FROM assertions_fts WHERE assertions_fts MATCH ? "
        "ORDER BY bm25(assertions_fts), assertion_id LIMIT ? OFFSET ?",
        (match, limit, offset),
    ).fetchall()
    return [
        _SearchRow(assertion_id=int(row["assertion_id"]), score=float(row["score"]))
        for row in rows
    ]


def _present_document_ids(
    conn: sqlite3.Connection, assertion_ids: Sequence[int]
) -> set[int]:
    present: set[int] = set()
    for batch in store._chunks(list(assertion_ids)):
        placeholders = ",".join("?" * len(batch))
        for row in conn.execute(
            "SELECT assertion_id FROM assertion_documents "  # noqa: S608
            f"WHERE assertion_id IN ({placeholders})",
            tuple(batch),
        ):
            present.add(int(row["assertion_id"]))
    return present


def _validate_bounds(k: int, page_size: int, max_candidates: int) -> None:
    for name, value in (
        ("k", k),
        ("page_size", page_size),
        ("max_candidates", max_candidates),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def search_assertions(
    conn: sqlite3.Connection,
    repo_root: Path,
    query: str,
    *,
    policy: ServingPolicy = PRODUCTION_POLICY,
    k: int = 10,
    page_size: int = 40,
    max_candidates: int = 400,
) -> list[AssertionCandidate]:
    """Return up to ``k`` verified assertion hits via bounded deterministic refill."""
    _validate_bounds(k, page_size, max_candidates)
    if not search_index.assertion_search_structures_present(conn):
        raise AssertionSearchUnavailable(
            "assertion search is unavailable because this index lacks its derived "
            "assertion documents or FTS structures; rebuild the index with this "
            "version of code-learner"
        )

    match = escape_fts_query(query)
    if match == '""':
        return []

    results: list[AssertionCandidate] = []
    seen_ids: set[int] = set()
    offset = 0
    examined = 0
    while len(results) < k and examined < max_candidates:
        limit = min(page_size, max_candidates - examined)
        page = _search_page(conn, match, limit=limit, offset=offset)
        if not page:
            break
        offset += len(page)
        examined += len(page)

        new_rows = [row for row in page if row.assertion_id not in seen_ids]
        seen_ids.update(row.assertion_id for row in new_rows)
        if not new_rows:
            if len(page) < limit:
                break
            continue

        ids = [row.assertion_id for row in new_rows]
        scores = {row.assertion_id: row.score for row in new_rows}
        loaded = load_assertions_by_ids(conn, ids)
        verdicts = verdict_summaries(conn, ids)

        eligible: list[store.Assertion] = []
        accepted: dict[int, tuple[VerdictSummary, ...]] = {}
        for assertion in loaded:
            decision = evaluate_metadata(
                assertion, cast(tuple, verdicts.get(assertion.id, ())), policy
            )
            if decision.eligible:
                eligible.append(assertion)
                accepted[assertion.id] = cast(
                    tuple[VerdictSummary, ...], decision.accepted
                )

        verified = (
            _verify_loaded_assertions(conn, repo_root, eligible) if eligible else []
        )
        for assertion in verified:
            results.append(
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
                    score=-scores[assertion.id],
                    modality="assertion_lexical",
                    conflict=False,
                    contributions=(),
                )
            )
            if len(results) == k:
                break
        # A terminal verification failure removes its derived document in the same
        # authoritative transition. Compensate the OFFSET for those rows or the
        # shrunken result set moves the next candidate behind the cursor.
        present_ids = _present_document_ids(conn, ids)
        offset -= sum(assertion_id not in present_ids for assertion_id in ids)
        if len(page) < limit:
            break

    return results
