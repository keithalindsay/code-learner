"""Bounded lexical retrieval for evidence-bound semantic assertions."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.assertions.policy import RESEARCH_PENDING_POLICY
from codelearner.retrieve.assertions import (
    AssertionSearchUnavailable,
    load_assertions_by_ids,
    search_assertions,
    verdict_summaries,
    verify_assertions,
)

SOURCE = "def renew():\n    return 'owner'\n"


@dataclass
class AssertionIndex:
    root: Path
    conn: sqlite3.Connection

    def add(
        self,
        claim: str,
        *,
        path: str = "leases.py",
        verdict: str | None = store.VERDICT_SUPPORTED,
        status: str = store.STATUS_ACTIVE,
        restore_document: bool = False,
    ) -> int:
        assertion_id = store.write_assertion(
            self.conn,
            subject_qualname="leases.renew",
            kind="invariant",
            claim=claim,
            spans=(store.span_for(self.root, path, 0, len(SOURCE)),),
            generator="test/v1",
            status=status,
            allow_unindexed_subject=True,
        )
        if verdict is not None:
            store.record_verdict(self.conn, assertion_id, "judge/v1", verdict)
        if restore_document:
            # Retrieval must defend against an out-of-date derived index. The
            # authoritative metadata remains the policy source of truth.
            self.conn.execute(
                "INSERT INTO assertion_documents(assertion_id, text, text_hash) "
                "VALUES (?, ?, ?) ON CONFLICT(assertion_id) DO NOTHING",
                (assertion_id, f"claim: {claim}", f"test-{assertion_id}"),
            )
        return assertion_id


@pytest.fixture
def assertion_index(tmp_path) -> AssertionIndex:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    (root / "stable.py").write_text(SOURCE)
    (root / "stale.py").write_text(SOURCE)
    conn = db.init_db(tmp_path / "index.db")
    db.bind_repo_root(conn, root)
    return AssertionIndex(root, conn)


def _status(index: AssertionIndex, assertion_id: int) -> str:
    row = index.conn.execute(
        "SELECT status FROM assertions WHERE id = ?", (assertion_id,)
    ).fetchone()
    return str(row["status"])


def test_supported_claim_is_retrieved_with_verdict_and_freshness(assertion_index):
    assertion_index.add("Renews ownership during long work.")

    hits = search_assertions(
        assertion_index.conn, assertion_index.root, "lease ownership", k=1
    )

    assert [(h.claim, h.verdicts[0].verdict, h.freshness.verified) for h in hits] == [
        ("Renews ownership during long work.", "supported", True)
    ]
    assert hits[0].freshness.method == "hash"
    assert hits[0].modality == "assertion_lexical"


def test_ineligible_top_rows_are_refilled(assertion_index):
    assertion_index.add("lease pending", verdict=None)
    assertion_index.add(
        "lease stale", status=store.STATUS_STALE, restore_document=True
    )
    assertion_index.add(
        "lease refuted", verdict=store.VERDICT_REFUTED, restore_document=True
    )
    assertion_index.add("lease supported")

    hits = search_assertions(
        assertion_index.conn, assertion_index.root, "lease", k=1, page_size=2
    )

    assert [h.claim for h in hits] == ["lease supported"]


@pytest.mark.parametrize(
    "query",
    ["", "   \n\t", '" OR (lease* - owner)^ : \x00\x1f'],
)
def test_empty_and_hostile_queries_are_safe(assertion_index, query):
    assertion_index.add("Renews ownership during long work.")

    hits = search_assertions(assertion_index.conn, assertion_index.root, query)

    assert isinstance(hits, list)


def test_research_policy_can_retrieve_pending_claims(assertion_index):
    assertion_index.add("lease pending", verdict=None)

    production = search_assertions(assertion_index.conn, assertion_index.root, "lease")
    research = search_assertions(
        assertion_index.conn,
        assertion_index.root,
        "lease",
        policy=RESEARCH_PENDING_POLICY,
    )

    assert production == []
    assert [hit.claim for hit in research] == ["lease pending"]
    assert research[0].verdicts == ()


def test_metadata_policy_runs_before_filesystem_verification(assertion_index, monkeypatch):
    assertion_index.add("lease pending", verdict=None)

    def unexpected_verification(*args, **kwargs):
        raise AssertionError("ineligible metadata reached filesystem verification")

    monkeypatch.setattr(
        "codelearner.retrieve.assertions._verify_loaded_assertions",
        unexpected_verification,
    )

    assert search_assertions(assertion_index.conn, assertion_index.root, "lease") == []


def test_duplicate_ids_across_pages_are_verified_once(assertion_index, monkeypatch):
    from codelearner.retrieve import assertions as retrieval

    first = assertion_index.add("lease first")
    second = assertion_index.add("lease second")
    snapshot = [
        retrieval._SearchRow(first, -2.0),
        retrieval._SearchRow(first, -1.5),
        retrieval._SearchRow(second, -1.0),
    ]
    verified: list[tuple[int, ...]] = []
    original = retrieval._verify_loaded_assertions

    monkeypatch.setattr(retrieval, "_search_page", lambda *args, **kwargs: snapshot)

    def recording_verify(conn, repo_root, assertions):
        verified.append(tuple(assertion.id for assertion in assertions))
        return original(conn, repo_root, assertions)

    monkeypatch.setattr(retrieval, "_verify_loaded_assertions", recording_verify)

    hits = search_assertions(
        assertion_index.conn, assertion_index.root, "lease", k=2, page_size=2
    )

    assert [hit.assertion_id for hit in hits] == [first, second]
    assert verified == [(first,), (second,)]


def test_transient_unreadability_withholds_without_mutation(assertion_index, monkeypatch):
    assertion_id = assertion_index.add("lease ownership")
    monkeypatch.setattr(store, "_read_source", lambda *args, **kwargs: store._UNREADABLE)

    hits = search_assertions(assertion_index.conn, assertion_index.root, "lease")

    assert hits == []
    assert _status(assertion_index, assertion_id) == store.STATUS_ACTIVE
    assert store.staleness_events(assertion_index.conn, assertion_id) == []
    assert assertion_index.conn.execute(
        "SELECT 1 FROM assertion_documents WHERE assertion_id = ?", (assertion_id,)
    ).fetchone()


@pytest.mark.parametrize("change", ["edit", "missing"])
def test_changed_or_missing_evidence_expires_and_removes_document(assertion_index, change):
    assertion_id = assertion_index.add("lease ownership")
    if change == "edit":
        (assertion_index.root / "leases.py").write_text(SOURCE.replace("owner", "other"))
    else:
        (assertion_index.root / "leases.py").unlink()

    hits = search_assertions(assertion_index.conn, assertion_index.root, "lease")

    assert hits == []
    assert _status(assertion_index, assertion_id) == store.STATUS_STALE
    assert assertion_index.conn.execute(
        "SELECT 1 FROM assertion_documents WHERE assertion_id = ?", (assertion_id,)
    ).fetchone() is None
    assert len(store.staleness_events(assertion_index.conn, assertion_id)) == 1


def test_expired_top_row_does_not_shift_refill_past_next_candidate(assertion_index):
    expired = assertion_index.add("lease first")
    supported = assertion_index.add("lease second", path="stable.py")
    (assertion_index.root / "leases.py").write_text(SOURCE.replace("owner", "other"))

    hits = search_assertions(
        assertion_index.conn,
        assertion_index.root,
        "lease",
        k=1,
        page_size=1,
    )

    assert [hit.assertion_id for hit in hits] == [supported]
    assert _status(assertion_index, expired) == store.STATUS_STALE


def test_bm25_reordering_after_expiry_does_not_skip_snapshot_candidate(assertion_index):
    supported = assertion_index.add("a a a a b b", path="stable.py")
    assertion_index.add(
        # Nineteen fillers is the smallest count that makes this fixture's full
        # canonical documents reorder survivors from 2,1 to 1,2 after ID 3 leaves
        # the corpus. The reviewer's standalone construction used fourteen; our
        # fixture also includes its canonical kind/subject/evidence fields.
        "a a a a b b b b b b b " + "x " * 19,
        path="stable.py",
        verdict=None,
    )
    expired = assertion_index.add(
        "a a a a a a a a b b b b b b b b b " + "x " * 15,
        path="stale.py",
    )
    (assertion_index.root / "stale.py").write_text(
        SOURCE.replace("owner", "other")
    )

    hits = search_assertions(
        assertion_index.conn,
        assertion_index.root,
        "a b",
        k=1,
        page_size=2,
        max_candidates=10,
    )

    assert [hit.assertion_id for hit in hits] == [supported]
    assert _status(assertion_index, expired) == store.STATUS_STALE


def test_equal_scores_use_assertion_id_order(assertion_index):
    ids = [
        assertion_index.add("lease alpha"),
        assertion_index.add("lease bravo"),
        assertion_index.add("lease delta"),
    ]

    hits = search_assertions(assertion_index.conn, assertion_index.root, "lease", k=3)

    assert [hit.assertion_id for hit in hits] == ids


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"k": 0}, "k"),
        ({"k": -1}, "k"),
        ({"page_size": 0}, "page_size"),
        ({"page_size": -1}, "page_size"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_candidates": -1}, "max_candidates"),
    ],
)
def test_non_positive_bounds_are_rejected(assertion_index, kwargs, name):
    with pytest.raises(ValueError, match=name):
        search_assertions(assertion_index.conn, assertion_index.root, "lease", **kwargs)


def test_max_candidate_cap_stops_refill(assertion_index):
    for number in range(3):
        assertion_index.add(f"lease pending{number}", verdict=None)
    assertion_index.add("lease supported")

    capped = search_assertions(
        assertion_index.conn,
        assertion_index.root,
        "lease",
        k=1,
        page_size=2,
        max_candidates=3,
    )
    enough = search_assertions(
        assertion_index.conn,
        assertion_index.root,
        "lease",
        k=1,
        page_size=2,
        max_candidates=4,
    )

    assert capped == []
    assert [hit.claim for hit in enough] == ["lease supported"]


def test_missing_assertion_search_structures_raise_domain_error(assertion_index):
    assertion_index.conn.execute("DROP TABLE assertions_fts")

    with pytest.raises(AssertionSearchUnavailable, match="assertion search"):
        search_assertions(assertion_index.conn, assertion_index.root, "lease")


def test_id_scoped_readers_preserve_requested_order_and_batch_verdicts(assertion_index):
    first = assertion_index.add("lease first")
    second = assertion_index.add("lease second")

    loaded = load_assertions_by_ids(
        assertion_index.conn, [second, 999_999, first, second]
    )
    summaries = verdict_summaries(
        assertion_index.conn, [second, 999_999, first, second]
    )

    assert [assertion.id for assertion in loaded] == [second, first]
    assert list(summaries) == [second, first]
    assert [summary.verdict for summary in summaries[first]] == ["supported"]


def test_id_scoped_verifier_never_scans_unrequested_assertions(assertion_index):
    requested = assertion_index.add("lease requested")
    untouched = assertion_index.add("lease untouched")
    (assertion_index.root / "leases.py").unlink()

    verified = verify_assertions(
        assertion_index.conn, assertion_index.root, [requested, requested]
    )

    assert verified == []
    assert _status(assertion_index, requested) == store.STATUS_STALE
    assert _status(assertion_index, untouched) == store.STATUS_ACTIVE
