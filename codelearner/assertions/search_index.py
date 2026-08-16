"""Disposable lexical documents derived from authoritative assertion history."""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from ..ingest.types import content_hash

if TYPE_CHECKING:
    from .store import Assertion


def canonical_document(assertion: Assertion) -> str:
    """Return the stable retrieval document for one assertion."""
    spans = sorted(
        assertion.spans,
        key=lambda span: (
            span.path,
            span.byte_start,
            span.byte_end,
            span.id if span.id is not None else -1,
        ),
    )
    evidence = ", ".join(span.citation for span in spans)
    return (
        f"kind: {assertion.kind}\n"
        f"subject: {assertion.subject_qualname}\n"
        f"claim: {assertion.claim}\n"
        f"evidence: {evidence}"
    )


def remove_assertion_document(conn: sqlite3.Connection, assertion_id: int) -> None:
    """Remove one derived document without touching its authoritative assertion."""
    conn.execute(
        "DELETE FROM assertion_documents WHERE assertion_id = ?", (assertion_id,)
    )


def sync_assertion_document(conn: sqlite3.Connection, assertion_id: int) -> None:
    """Make one derived document reflect the assertion's current status."""
    from .store import STATUS_ACTIVE, _load_assertions

    found = _load_assertions(conn, "id = ?", (assertion_id,))
    if not found or found[0].status != STATUS_ACTIVE:
        remove_assertion_document(conn, assertion_id)
        return

    document = canonical_document(found[0])
    text_hash = content_hash(document.encode("utf-8"))
    conn.execute(
        "INSERT INTO assertion_documents (assertion_id, text, text_hash) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(assertion_id) DO UPDATE SET "
        "text = excluded.text, text_hash = excluded.text_hash",
        (assertion_id, document, text_hash),
    )


def rebuild_assertion_documents(conn: sqlite3.Connection) -> None:
    """Reconstruct every live document from authoritative assertion rows."""
    # An external-content token index may have diverged from its content table.
    # Repair that relationship before DELETE triggers try to remove tokens that are
    # no longer there, then rebuild once more from the freshly derived documents.
    conn.execute("INSERT INTO assertions_fts(assertions_fts) VALUES ('rebuild')")
    conn.execute("DELETE FROM assertion_documents")
    assertion_ids = [
        int(row[0]) for row in conn.execute("SELECT id FROM assertions ORDER BY id")
    ]
    for assertion_id in assertion_ids:
        sync_assertion_document(conn, assertion_id)
    conn.execute("INSERT INTO assertions_fts(assertions_fts) VALUES ('rebuild')")


def assertion_search_structures_present(conn: sqlite3.Connection) -> bool:
    """Return whether the derived tables and their synchronization triggers exist."""
    required_triggers = {
        "assertions_fts_insert",
        "assertions_fts_delete",
        "assertions_fts_update",
    }
    objects = {
        str(row[0]): (str(row[1]), str(row[2] or ""))
        for row in conn.execute(
            "SELECT name, type, sql FROM sqlite_master WHERE name IN (?,?,?,?,?)",
            ("assertion_documents", "assertions_fts", *sorted(required_triggers)),
        )
    }
    documents = objects.get("assertion_documents")
    fts = objects.get("assertions_fts")
    if documents is None or documents[0] != "table" or fts is None or fts[0] != "table":
        return False
    fts_sql = " ".join(fts[1].casefold().split())
    if "create virtual table" not in fts_sql or "using fts5" not in fts_sql:
        return False
    return all(name in objects and objects[name][0] == "trigger" for name in required_triggers)
