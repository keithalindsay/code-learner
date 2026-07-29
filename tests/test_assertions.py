"""The tier-2 assertion store: the write gate, the hash binding, the rejection log.

Every test here names a rule that would otherwise fail silently. The standard the
project holds itself to is that deleting the rule has to turn the test red -- a test
that survives removing the behaviour it names is not a test.
"""
from __future__ import annotations

import sqlite3

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.ingest import index_repo

SOURCE = (
    'def acquire(parcel_id):\n'
    '    """Take a lease."""\n'
    '    return True\n'
    '\n'
    '\n'
    'def release(parcel_id):\n'
    '    return False\n'
)

# Byte range of `acquire` alone, up to and including its trailing newline.
ACQUIRE_END = SOURCE.index("\n\n\n") + 1
RELEASE_START = SOURCE.index("def release")


@pytest.fixture
def repo(tmp_path):
    """A one-file repo plus an index bound to it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    conn = db.init_db(tmp_path / "index.db")
    db.bind_repo_root(conn, root)
    return root, conn


def _acquire_span(root):
    return store.span_for(root, "leases.py", 0, ACQUIRE_END)


def _release_span(root):
    return store.span_for(root, "leases.py", RELEASE_START, len(SOURCE))


def _admit(conn, spans, *, claim="acquire returns True when the lease is taken"):
    return store.write_assertion(
        conn,
        subject_qualname="leases.acquire",
        kind="purpose",
        claim=claim,
        spans=spans,
        generator="test-model/v1",
        confidence=0.9,
    )


# --------------------------------------------------------------------------
# rule 1 -- no citation, no entry
# --------------------------------------------------------------------------


def test_an_assertion_with_no_evidence_is_refused(repo):
    """The cheap rule and the one that matters most: an uncited claim cannot be
    adjudicated, cannot expire, and cannot be checked by a reader, so it is
    indistinguishable from a good one at every stage after the door."""
    _, conn = repo
    with pytest.raises(store.EvidenceRequired):
        _admit(conn, [])


def test_a_refused_assertion_leaves_no_row_behind(repo):
    """Refusal has to happen BEFORE the insert. An uncited claim stored as
    'rejected' would sit in the same table as the citeable ones, one status update
    away from being served."""
    _, conn = repo
    with pytest.raises(store.EvidenceRequired):
        _admit(conn, [])
    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0
    assert conn.execute("SELECT count(*) c FROM evidence_spans").fetchone()["c"] == 0


def test_an_admitted_claim_is_servable_and_carries_its_citation(repo):
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    servable = store.servable_assertions(conn)
    assert [a.id for a in servable] == [aid]
    assert servable[0].spans[0].citation == "leases.py:1-3"


# --------------------------------------------------------------------------
# rule 2 -- servable means re-verified, not merely stored
# --------------------------------------------------------------------------


def test_editing_a_cited_span_expires_the_claim_instead_of_serving_it(repo):
    """The core of the tier. The row still says 'active' on disk; the bytes it
    cites do not hash to what was cited, so it must not be served."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))

    assert store.servable_assertions(conn) == []
    status = conn.execute("SELECT status FROM assertions WHERE id=?", (aid,)).fetchone()
    assert status["status"] == store.STATUS_STALE


def test_expiry_records_which_citation_moved_and_what_it_became(repo):
    root, conn = repo
    span = _acquire_span(root)
    aid = _admit(conn, [span])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))
    store.servable_assertions(conn)

    events = store.staleness_events(conn, aid)
    assert len(events) == 1
    assert events[0]["reason"] == store.REASON_HASH_MISMATCH
    assert events[0]["expected_hash"] == span.content_hash
    assert events[0]["observed_hash"] not in (None, span.content_hash)


def test_a_status_of_active_alone_is_never_enough_to_serve(repo):
    """Guards the specific mistake of trusting the stored status. Flipping a stale
    claim back to 'active' by hand must not make it servable again -- the status is
    a record of the last time somebody looked, not a statement about the code."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))
    store.servable_assertions(conn)
    conn.execute("UPDATE assertions SET status='active' WHERE id=?", (aid,))

    assert store.servable_assertions(conn) == []
    assert not store.is_servable(conn, aid)


def test_an_unrelated_edit_elsewhere_in_the_file_does_not_expire_the_claim(repo):
    """Spans are hashed, not files. Hashing the whole file would expire every claim
    about a 2,000-line module on any edit to it, and staleness that fires on
    everything is staleness nobody reads."""
    root, conn = repo
    _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return False", "return None"))

    assert len(store.servable_assertions(conn)) == 1
    assert store.staleness_events(conn) == []


def test_every_cited_span_must_still_verify_not_merely_one(repo):
    """A claim resting on two spans is a claim about both. Verifying only the first
    would let the evidence that actually carried the claim rot unnoticed."""
    root, conn = repo
    first, second = _acquire_span(root), _release_span(root)
    aid = _admit(conn, [first, second])
    (root / "leases.py").write_text(SOURCE.replace("return False", "return None"))

    assert store.servable_assertions(conn) == []
    events = store.staleness_events(conn, aid)
    assert len(events) == 1
    assert events[0]["expected_hash"] == second.content_hash


def test_a_deleted_file_is_an_expiry_with_a_reason_not_a_crash(repo):
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").unlink()

    assert store.servable_assertions(conn) == []
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_FILE_MISSING


def test_a_truncated_file_is_distinguished_from_an_edit(repo):
    """Slicing past the end of a bytes object returns a short result rather than
    failing, so a deleted tail would otherwise be logged as an ordinary edit."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text("def acquire")

    assert store.servable_assertions(conn) == []
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_SPAN_TRUNCATED


def test_an_assertion_with_no_spans_left_is_not_vacuously_servable(repo):
    """"Every cited span still matches" is trivially TRUE of no spans. Without an
    explicit check, an assertion that lost its evidence would be promoted by the
    same code path that verifies a well-cited one, and reported as verified."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    conn.execute("DELETE FROM evidence_spans WHERE assertion_id=?", (aid,))

    assert store.servable_assertions(conn) == []
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_NO_EVIDENCE


def test_expiry_is_logged_once_per_event_not_once_per_read(repo):
    """The log's growth rate is meant to measure how fast this repo invalidates its
    own inferences. One row per read would make it measure traffic instead."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))
    store.servable_assertions(conn)
    store.servable_assertions(conn)
    store.servable_assertions(conn)

    assert len(store.staleness_events(conn, aid)) == 1


def test_serving_without_a_known_repo_root_is_refused(tmp_path):
    """Verification means re-reading the cited bytes. Not knowing where the repo is
    means serving claims that nothing checked."""
    conn = db.init_db(tmp_path / "unbound.db")
    with pytest.raises(ValueError, match="repo root"):
        store.servable_assertions(conn)


# --------------------------------------------------------------------------
# rule 3 -- adjudicated, and nothing is deleted
# --------------------------------------------------------------------------


def test_a_refuted_claim_is_retained_with_its_evidence_and_its_verdict(repo):
    """The rejected set IS the measurement of whether the gate works. A store that
    deletes what it rejected can report any pass rate it likes."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(
        conn, aid, "judge/v1", store.VERDICT_REFUTED, "the cited span returns True"
    )

    assert store.servable_assertions(conn) == []
    rejected = store.assertions_with_status(conn, store.STATUS_REJECTED)
    assert [a.id for a in rejected] == [aid]
    assert rejected[0].claim
    assert len(rejected[0].spans) == 1  # its citations are kept too
    verdicts = store.verdicts_for(conn, aid)
    assert [v["verdict"] for v in verdicts] == [store.VERDICT_REFUTED]
    assert verdicts[0]["rationale"] == "the cited span returns True"


def test_an_unsupported_verdict_also_stops_it_being_served(repo):
    """"The evidence is silent" is a different problem from "the evidence says
    otherwise", and both are reasons to stop answering questions with the claim."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(conn, aid, "judge/v1", store.VERDICT_UNSUPPORTED)

    assert store.servable_assertions(conn) == []
    assert store.assertions_with_status(conn, store.STATUS_REJECTED)[0].id == aid


def test_a_supported_verdict_leaves_the_claim_servable(repo):
    """The control. Without it, a store that rejected everything would pass every
    other test in this file."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(conn, aid, "judge/v1", store.VERDICT_SUPPORTED)

    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_one_unsupportive_judge_is_enough(repo):
    """A consensus rule would make the gate exactly as strong as its most
    permissive judge, which is the opposite of what more judges are for."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(conn, aid, "judge-a", store.VERDICT_SUPPORTED)
    store.record_verdict(conn, aid, "judge-b", store.VERDICT_REFUTED)

    assert store.servable_assertions(conn) == []
    assert len(store.verdicts_for(conn, aid)) == 2


def test_a_rejected_claim_is_not_downgraded_to_merely_stale(repo):
    """Rejected is the stronger statement -- a judge refuted it, and that does not
    stop being true because the file changed afterwards."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(conn, aid, "judge/v1", store.VERDICT_REFUTED)

    assert store.mark_stale(conn, aid, store.REASON_HASH_MISMATCH) is False
    row = conn.execute("SELECT status FROM assertions WHERE id=?", (aid,)).fetchone()
    assert row["status"] == store.STATUS_REJECTED
    assert store.staleness_events(conn, aid) == []


def test_a_status_the_schema_does_not_know_is_refused(repo):
    """The CHECK lives in the schema because a typo'd status is otherwise
    indistinguishable from a rejected one -- both merely fail to be 'active', so the
    claim stops being served with no record of why."""
    _, conn = repo
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO assertions (subject_qualname, kind, claim, status) "
            "VALUES ('leases.acquire','purpose','c','probably_fine')"
        )


# --------------------------------------------------------------------------
# citations, and surviving a re-index
# --------------------------------------------------------------------------


def test_span_for_derives_line_numbers_from_the_byte_range(repo):
    """A citation whose lines and bytes disagree points a human at one place and the
    verifier at another, and nothing about it would ever look wrong."""
    root, _ = repo
    span = _release_span(root)
    assert (span.line_start, span.line_end) == (6, 7)
    assert SOURCE.splitlines()[span.line_start - 1].startswith("def release")


def test_span_for_refuses_an_empty_or_out_of_range_citation(repo):
    """An empty slice hashes to a stable value and would verify forever while
    pointing at nothing."""
    root, _ = repo
    with pytest.raises(ValueError):
        store.span_for(root, "leases.py", 5, 5)
    with pytest.raises(ValueError):
        store.span_for(root, "leases.py", 0, len(SOURCE) + 100)


def test_a_symbol_citation_uses_the_hash_the_index_already_stored(tmp_path):
    """The evidence unit for a claim about code is a symbol, and `symbols` already
    holds the sha256 of exactly its bytes. Re-deriving it would introduce a way for
    the citation and the index to disagree about where the symbol is."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    sym = conn.execute(
        "SELECT id FROM symbols WHERE qualname = 'leases.acquire'"
    ).fetchone()

    span = store.span_for_symbol(conn, sym["id"])
    aid = store.write_assertion(
        conn, subject_qualname="leases.acquire", kind="purpose",
        claim="takes a lease", spans=[span], subject_symbol_id=sym["id"],
    )
    assert [a.id for a in store.servable_assertions(conn, root)] == [aid]

    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))
    assert store.servable_assertions(conn, root) == []


def test_re_indexing_away_the_subject_symbol_does_not_delete_the_claim(repo):
    """The reason `subject_symbol_id` is SET NULL and `subject_qualname` is NOT
    NULL. A CASCADE here would mean a routine re-index silently empties the entire
    assertion store, because re-indexing replaces symbol rows wholesale."""
    root, conn = repo
    conn.execute(
        "INSERT INTO files (path,lang,content_hash,size_bytes,mtime_ns) "
        "VALUES ('leases.py','python','h',1,1)"
    )
    fid = conn.execute("SELECT id FROM files").fetchone()["id"]
    conn.execute(
        "INSERT INTO symbols (file_id,kind,name,qualname,line_start,line_end,"
        "byte_start,byte_end,content_hash) VALUES (?,'function','acquire',"
        "'leases.acquire',1,3,0,?,'h')", (fid, ACQUIRE_END)
    )
    sid = conn.execute("SELECT id FROM symbols").fetchone()["id"]
    aid = store.write_assertion(
        conn, subject_qualname="leases.acquire", kind="purpose",
        claim="takes a lease", spans=[_acquire_span(root)], subject_symbol_id=sid,
    )

    conn.execute("DELETE FROM files WHERE id = ?", (fid,))

    row = conn.execute("SELECT * FROM assertions WHERE id=?", (aid,)).fetchone()
    assert row is not None
    assert row["subject_symbol_id"] is None
    assert row["subject_qualname"] == "leases.acquire"
    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_a_write_joins_the_callers_transaction_rather_than_refusing_to_nest(repo):
    """A pipeline admits a batch of claims as one unit. If the batch rolls back,
    none of them may survive -- a half-written batch is an index that claims things
    the run that produced them decided not to keep."""
    root, conn = repo
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            _admit(conn, [_acquire_span(root)])
            _admit(conn, [_release_span(root)])
            raise RuntimeError("batch failed")

    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0
    assert conn.execute("SELECT count(*) c FROM evidence_spans").fetchone()["c"] == 0


def test_the_assertion_tables_exist_in_a_fresh_index(tmp_path):
    conn = db.init_db(tmp_path / "i.db")
    names = {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"assertions", "evidence_spans", "verdicts", "staleness_log"} <= names
