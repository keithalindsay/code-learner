"""Derived lexical documents for authoritative tier-2 assertions."""
from __future__ import annotations

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.assertions.store import Assertion, EvidenceSpan


def _assertion(*spans: EvidenceSpan) -> Assertion:
    return Assertion(
        id=17,
        subject_qualname="leases.acquire",
        subject_symbol_id=3,
        kind="purpose",
        claim="Coordinates lease renewal.",
        status="active",
        generator="test/v1",
        confidence=0.9,
        created_at="2026-08-11T00:00:00Z",
        spans=spans,
    )


def test_v7_schema_contains_assertion_documents_and_fts(tmp_path):
    index = db.init_db(tmp_path / "index.db")
    names = {row[0] for row in index.execute("SELECT name FROM sqlite_master")}

    assert {"assertion_documents", "assertions_fts"} <= names
    assert db.SCHEMA_VERSION == 7


def test_canonical_document_is_deterministic():
    from codelearner.assertions.search_index import canonical_document

    assertion = _assertion(
        EvidenceSpan("leases.py", 10, 18, 100, 180, "hash", id=5)
    )

    assert canonical_document(assertion) == (
        "kind: purpose\n"
        "subject: leases.acquire\n"
        "claim: Coordinates lease renewal.\n"
        "evidence: leases.py:10-18"
    )


def test_canonical_document_sorts_evidence_by_stable_source_coordinates():
    from codelearner.assertions.search_index import canonical_document

    assertion = _assertion(
        EvidenceSpan("z.py", 1, 2, 0, 10, "z", id=1),
        EvidenceSpan("a.py", 8, 9, 80, 90, "a2", id=3),
        EvidenceSpan("a.py", 3, 4, 20, 40, "a1", id=2),
    )

    assert canonical_document(assertion).endswith(
        "evidence: a.py:3-4, a.py:8-9, z.py:1-2"
    )


def _stored_assertion(index, *, status: str = store.STATUS_ACTIVE) -> int:
    return store.write_assertion(
        index,
        subject_qualname="leases.acquire",
        kind="purpose",
        claim="Coordinates lease renewal.",
        spans=(EvidenceSpan("leases.py", 10, 18, 100, 180, "hash"),),
        status=status,
        verify=False,
        allow_unindexed_subject=True,
    )


def test_sync_indexes_an_active_pending_assertion_and_updates_fts(tmp_path):
    from codelearner.assertions.search_index import sync_assertion_document

    index = db.init_db(tmp_path / "index.db")
    assertion_id = _stored_assertion(index)

    sync_assertion_document(index, assertion_id)

    document = index.execute(
        "SELECT text, text_hash FROM assertion_documents WHERE assertion_id = ?",
        (assertion_id,),
    ).fetchone()
    assert document is not None
    assert document[0].startswith("kind: purpose\nsubject: leases.acquire\n")
    assert len(document[1]) == 64
    assert [
        row[0]
        for row in index.execute(
            "SELECT rowid FROM assertions_fts WHERE assertions_fts MATCH 'renewal'"
        )
    ] == [assertion_id]


def test_sync_removes_rejected_and_stale_documents(tmp_path):
    from codelearner.assertions.search_index import sync_assertion_document

    index = db.init_db(tmp_path / "index.db")
    rejected = _stored_assertion(index)
    stale = _stored_assertion(index)
    sync_assertion_document(index, rejected)
    sync_assertion_document(index, stale)
    index.execute("UPDATE assertions SET status = 'rejected' WHERE id = ?", (rejected,))
    index.execute("UPDATE assertions SET status = 'stale' WHERE id = ?", (stale,))

    sync_assertion_document(index, rejected)
    sync_assertion_document(index, stale)

    assert index.execute("SELECT count(*) FROM assertion_documents").fetchone()[0] == 0
    assert index.execute(
        "SELECT count(*) FROM assertions_fts WHERE assertions_fts MATCH 'renewal'"
    ).fetchone()[0] == 0


def test_remove_deletes_only_the_derived_document(tmp_path):
    from codelearner.assertions.search_index import (
        remove_assertion_document,
        sync_assertion_document,
    )

    index = db.init_db(tmp_path / "index.db")
    assertion_id = _stored_assertion(index)
    sync_assertion_document(index, assertion_id)

    remove_assertion_document(index, assertion_id)

    assert index.execute("SELECT count(*) FROM assertion_documents").fetchone()[0] == 0
    assert index.execute("SELECT count(*) FROM assertions").fetchone()[0] == 1


def test_rebuild_reconstructs_only_active_documents(tmp_path):
    from codelearner.assertions.search_index import rebuild_assertion_documents

    index = db.init_db(tmp_path / "index.db")
    active = _stored_assertion(index)
    _stored_assertion(index, status=store.STATUS_REJECTED)
    _stored_assertion(index, status=store.STATUS_STALE)

    rebuild_assertion_documents(index)

    assert [
        row[0]
        for row in index.execute(
            "SELECT assertion_id FROM assertion_documents ORDER BY assertion_id"
        )
    ] == [active]


def test_rebuild_recovers_from_cleared_fts_tokens_without_losing_documents(tmp_path):
    from codelearner.assertions.search_index import rebuild_assertion_documents

    index = db.init_db(tmp_path / "index.db")
    active = _stored_assertion(index)
    _stored_assertion(index, status=store.STATUS_REJECTED)
    index.execute("INSERT INTO assertions_fts(assertions_fts) VALUES ('delete-all')")
    assert index.execute("SELECT count(*) FROM assertion_documents").fetchone()[0] == 1
    assert index.execute(
        "SELECT count(*) FROM assertions_fts WHERE assertions_fts MATCH 'renewal'"
    ).fetchone()[0] == 0

    rebuild_assertion_documents(index)

    assert [
        row[0]
        for row in index.execute(
            "SELECT rowid FROM assertions_fts WHERE assertions_fts MATCH 'renewal'"
        )
    ] == [active]


def test_structure_probe_requires_both_derived_tables(tmp_path):
    from codelearner.assertions.search_index import assertion_search_structures_present

    index = db.init_db(tmp_path / "index.db")
    assert assertion_search_structures_present(index) is True

    index.execute("DROP TABLE assertions_fts")
    assert assertion_search_structures_present(index) is False


def test_structure_probe_rejects_a_view_named_like_the_fts_table(tmp_path):
    from codelearner.assertions.search_index import assertion_search_structures_present

    index = db.init_db(tmp_path / "index.db")
    index.execute("DROP TABLE assertions_fts")
    index.execute(
        "CREATE VIEW assertions_fts AS "
        "SELECT assertion_id AS rowid, text FROM assertion_documents"
    )

    assert assertion_search_structures_present(index) is False


def test_structure_probe_requires_assertion_documents_to_be_a_table(tmp_path):
    from codelearner.assertions.search_index import assertion_search_structures_present

    index = db.init_db(tmp_path / "index.db")
    index.execute("DROP TABLE assertion_documents")
    index.execute(
        "CREATE VIEW assertion_documents AS "
        "SELECT id AS assertion_id, claim AS text, claim AS text_hash FROM assertions"
    )
    for operation in ("INSERT", "DELETE", "UPDATE"):
        index.execute(
            f"CREATE TRIGGER assertions_fts_{operation.casefold()} "
            f"INSTEAD OF {operation} ON assertion_documents BEGIN SELECT 1; END"
        )

    assert assertion_search_structures_present(index) is False


def test_structure_probe_rejects_a_non_fts_table_with_the_expected_name(tmp_path):
    from codelearner.assertions.search_index import assertion_search_structures_present

    index = db.init_db(tmp_path / "index.db")
    index.execute("DROP TABLE assertions_fts")
    index.execute("CREATE TABLE assertions_fts (text TEXT)")

    assert assertion_search_structures_present(index) is False


@pytest.mark.parametrize(
    "trigger",
    [
        "assertions_fts_insert",
        "assertions_fts_delete",
        "assertions_fts_update",
    ],
)
def test_structure_probe_requires_each_fts_sync_trigger(tmp_path, trigger):
    from codelearner.assertions.search_index import assertion_search_structures_present

    index = db.init_db(tmp_path / "index.db")
    index.execute(f"DROP TRIGGER {trigger}")

    assert assertion_search_structures_present(index) is False
