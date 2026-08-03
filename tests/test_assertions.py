"""The tier-2 assertion store: the write gate, the hash binding, the rejection log.

Every test here names a rule that would otherwise fail silently. The standard the
project holds itself to is that deleting the rule has to turn the test red -- a test
that survives removing the behaviour it names is not a test.
"""
from __future__ import annotations

import contextlib
import dataclasses
import inspect
import os
import sqlite3

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.ingest import index_repo
from codelearner.ingest.types import content_hash

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
    """A one-file repo, INDEXED, plus the index bound to it.

    Indexed rather than merely initialised, because the write gate checks that a
    claim's subject is a symbol this index parsed. A fixture that skipped the index
    would have to pass `allow_unindexed_subject=True` on every admission in this file,
    and the store's own tests would then be the only ones in the project exercising
    the gate with one of its rules turned off.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _acquire_span(root):
    return store.span_for(root, "leases.py", 0, ACQUIRE_END)


def _release_span(root):
    return store.span_for(root, "leases.py", RELEASE_START, len(SOURCE))


def _admit(conn, spans, *, claim="acquire returns True when the lease is taken", **kwargs):
    kwargs.setdefault("subject_qualname", "leases.acquire")
    return store.write_assertion(
        conn,
        kind="purpose",
        claim=claim,
        spans=spans,
        generator="test-model/v1",
        confidence=0.9,
        **kwargs,
    )


# --------------------------------------------------------------------------
# rule 1 -- nothing is admitted that a later reader could not check
#
# Six conditions, one rule. Each of them produces a row that is indistinguishable
# from a good claim at every stage after the door -- it stores, it serves, and most
# of them verify forever -- so each is refused here or not at all. Every test below
# goes through `_refused`, which asserts the database was not touched: a gate that
# says no and writes the row anyway has refused nothing.
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


def _refused(conn, exc_type, spans, *, match=None, **kwargs):
    """Run one inadmissible write and assert it left the database untouched.

    The shared half of every rule below, and the half that is easy to leave out. A
    gate that raises after the INSERT has refused nothing: the row is in the same
    table as the admitted ones, one status update or one buggy query away from being
    served, and the exception the caller saw is no longer evidence of anything.

    `total_changes` and `in_transaction` are checked as well as the row counts,
    because counting rows only proves the tables that were counted are clean --
    `total_changes` covers every table in the file, and an open transaction means the
    refusal left the connection in a state the NEXT writer would silently join.
    """
    before = conn.total_changes
    with pytest.raises(exc_type, match=match):
        _admit(conn, spans, **kwargs)
    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0
    assert conn.execute("SELECT count(*) c FROM evidence_spans").fetchone()["c"] == 0
    assert conn.total_changes == before
    assert conn.in_transaction is False


def test_an_admitted_claim_is_servable_and_carries_its_citation(repo):
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    servable = store.servable_assertions(conn)
    assert [a.id for a in servable] == [aid]
    assert servable[0].spans[0].citation == "leases.py:1-3"


def test_an_empty_claim_is_refused_however_good_its_citations_are(repo):
    """Verified evidence carrying no statement. Every other check passes on this
    submission -- the span exists, it hashes correctly, the subject is real -- so
    without this rule it stores `active`, reports servable, and is handed back beside
    the code it is allegedly about, saying nothing. `generate.pipeline` refuses it as
    OUTCOME_EMPTY_CLAIM, which is the proof the rule belongs in the gate: a rule the
    caller enforces is a rule the next caller does not have."""
    root, conn = repo
    _refused(conn, store.EmptyClaim, [_acquire_span(root)], claim="")
    _refused(conn, store.EmptyClaim, [_acquire_span(root)], claim="   \n\t ")


def test_a_zero_length_span_is_refused(repo):
    """sha256 of nothing is a perfectly stable hash, so an empty citation does not
    merely fail to expire -- it VERIFIES, on every read, against the file as it is,
    as it becomes, and after the symbol it pointed at is deleted. It reports the
    strongest possible evidence that the claim still holds while pointing at nothing.

    `span_for` has always refused this. That is a rule for callers who use `span_for`,
    and the gate does not require anyone to.

    Carrying the hash of nothing, because that is the shape that gets through: every
    other check in the gate passes on it. Verified against 31e6c97, this exact span is
    admitted, served, and still served after the file it cites is replaced wholesale."""
    root, conn = repo
    empty = dataclasses.replace(
        _acquire_span(root), byte_start=4, byte_end=4, content_hash=content_hash(b"")
    )
    _refused(conn, store.InvalidSpan, [empty], match="non-empty byte range")


def test_a_negative_or_inverted_span_is_refused(repo):
    """The same rule's other two ends. A negative start slices from the tail of the
    file, and an inverted range slices to nothing -- both hash to something stable
    and neither points where its line numbers say it does."""
    root, conn = repo
    span = _acquire_span(root)
    _refused(conn, store.InvalidSpan, [dataclasses.replace(span, byte_start=-5)])
    _refused(
        conn,
        store.InvalidSpan,
        [dataclasses.replace(span, byte_start=span.byte_end, byte_end=span.byte_start)],
    )


def test_a_span_whose_lines_a_reader_cannot_open_is_refused(repo):
    """The bytes are what the verifier checks and the lines are what a human follows.
    A citation whose two halves disagree sends them to different places and looks
    wrong to neither, which is why the line range is checked here rather than trusted
    because the byte range happened to be sound."""
    root, conn = repo
    span = _acquire_span(root)
    _refused(conn, store.InvalidSpan, [dataclasses.replace(span, line_start=0)])
    _refused(conn, store.InvalidSpan, [dataclasses.replace(span, line_start=9, line_end=2)])


def test_a_span_with_no_hash_to_check_it_against_is_refused(repo):
    """A location with no assertion about what is there can never be found to be
    wrong. It is the vacuous truth `servable_assertions` guards against for a whole
    assertion, one level down at the span: "every cited span still matches" is
    trivially true of a span that never said what it should match."""
    root, conn = repo
    span = _acquire_span(root)
    _refused(conn, store.EvidenceUnverifiable, [dataclasses.replace(span, content_hash="")])
    _refused(conn, store.EvidenceUnverifiable, [dataclasses.replace(span, content_hash="  ")])


def test_a_claim_about_a_symbol_this_index_never_parsed_is_refused(repo):
    """An inference no reader can reach. The span is genuinely correct, so the claim
    is indistinguishable from a good one by inspection -- and distinguishable from one
    only by the fact that nothing will ever ask for it, because `get_symbol` answers
    `no_such_symbol` for the only name that would find it."""
    root, conn = repo
    _refused(
        conn,
        store.UnknownSubject,
        [_acquire_span(root)],
        subject_qualname="leases.acquire_that_was_never_written",
        match="no symbol named",
    )


def test_the_unindexed_subject_escape_has_to_be_asked_for_by_name(repo):
    """The fixtures that legitimately write claims about unindexed symbols say so in
    the call. A rule that instead turned itself off when `symbols` happened to be
    empty would be off on a fresh index, which is exactly when the first claims are
    written -- so the escape is a parameter, and it is visible at every call site
    that takes it."""
    root, conn = repo
    aid = _admit(
        conn,
        [_acquire_span(root)],
        subject_qualname="leases.acquire_that_was_never_written",
        allow_unindexed_subject=True,
    )
    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_a_citation_that_does_not_match_disk_is_refused_at_the_door(repo):
    """The rule the MCP server used to hold alone. A claim admitted on a hash that
    was already wrong is a row whose FIRST verification is guaranteed to fail -- and
    the store would then record that as `stale`, the word it uses for the repository
    moving under a claim that was once good. It was never good."""
    root, conn = repo
    span = _acquire_span(root)
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))

    _refused(conn, store.EvidenceStale, [span], match=store.REASON_HASH_MISMATCH)
    assert store.staleness_events(conn) == []


def test_a_citation_of_a_file_that_is_gone_is_refused_at_the_door(repo):
    """Same rule, the other two ways a citation stops matching: the file is gone, or
    the tail it cited is. Both are caught by the same `_first_failure` the serve path
    uses, which is the point -- an admission verified by a second implementation
    would be an admission that could disagree with tomorrow's serve."""
    root, conn = repo
    span = _acquire_span(root)
    (root / "leases.py").unlink()
    _refused(conn, store.EvidenceStale, [span], match=store.REASON_FILE_MISSING)

    (root / "leases.py").write_text("def acquire")
    _refused(conn, store.EvidenceStale, [span], match=store.REASON_SPAN_TRUNCATED)


def test_verification_is_on_by_default_and_turning_it_off_is_explicit(repo):
    """The control for the rule above. `verify=False` is a statement that nothing
    checked these citations, and the only honest reason to say it is that the caller
    has just hashed the bytes itself -- so it must be visible in the call rather than
    being the default nobody notices."""
    root, conn = repo
    span = _acquire_span(root)
    (root / "leases.py").write_text(SOURCE.replace("return True", "return False"))

    aid = _admit(conn, [span], verify=False)
    # Admitted, and expired on the very first read -- which is exactly the outcome
    # the default prevents, and why the store cannot tell this claim from one the
    # repository moved under afterwards.
    assert store.servable_assertions(conn) == []
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_HASH_MISMATCH


def test_admission_will_not_verify_against_a_root_the_index_is_not_bound_to(repo):
    """Admission binds a claim to this index, and every later verification of it uses
    the STORED root. A claim admitted against some other tree would be re-checked
    tomorrow against bytes it was never compared to, and that surfaces as an expiry
    naming an edit nobody made."""
    root, conn = repo
    elsewhere = root.parent / "other-checkout"
    elsewhere.mkdir()
    (elsewhere / "leases.py").write_text(SOURCE)

    _refused(conn, store.EvidenceUnverifiable, [_acquire_span(root)],
             repo_root=elsewhere, match="bound to")


def test_admission_without_a_repo_root_anywhere_is_refused_rather_than_assumed(tmp_path):
    """Verification that cannot happen must not be reported as verification that
    passed. The same refusal `servable_assertions` makes on the read path, at the
    write end, so an unbound index cannot accumulate claims nothing ever checked."""
    conn = db.init_db(tmp_path / "unbound.db")
    span = store.EvidenceSpan(
        path="leases.py", line_start=1, line_end=3, byte_start=0,
        byte_end=ACQUIRE_END, content_hash="0" * 64,
    )
    with pytest.raises(store.EvidenceUnverifiable, match="not bound to a repo root"):
        _admit(conn, [span], allow_unindexed_subject=True)
    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0


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


def test_a_stale_claim_is_not_demoted_to_rejected(repo):
    """The other direction of the same rule, and the direction the guard was written
    for. `record_verdict` names it in a comment -- "a claim already stale stays stale,
    and its staleness is the more actionable fact" -- and nothing tested it.

    The two states are not interchangeable labels for "not servable". `stale` says the
    cited bytes moved, which is a repair instruction: re-derive this claim against the
    code as it is now. `rejected` says a judge read the claim against its evidence and
    refused it, which is a verdict on text that, once the file has moved, no longer
    describes the repository -- so overwriting the first with the second replaces a
    fact about the code with an opinion about a version of it nobody can see any more.

    Dropping the `AND status = ?` from `_TOUCH_STATUS` leaves the rejected -> stale
    direction (tested directly above) still refusing, because `mark_stale` checks
    `rowcount`; only this direction goes silently wrong.
    """
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    assert store.mark_stale(conn, aid, store.REASON_HASH_MISMATCH) is True

    store.record_verdict(conn, aid, "judge/v1", store.VERDICT_REFUTED, "it returns True")

    row = conn.execute("SELECT status FROM assertions WHERE id=?", (aid,)).fetchone()
    assert row["status"] == store.STATUS_STALE
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]
    assert store.assertions_with_status(conn, store.STATUS_REJECTED) == []
    # The verdict is still RECORDED. Refusing the demotion is not refusing the
    # adjudication -- the judge's finding is kept in `verdicts`, it just does not
    # overwrite a status that already says the more useful thing.
    assert [v["verdict"] for v in store.verdicts_for(conn, aid)] == [store.VERDICT_REFUTED]


def test_reading_a_status_bucket_does_not_re_check_or_re_promote_anything(repo):
    """`assertions_with_status` is the UN-verified reader, and reviewing what the gate
    threw out must not be an act that changes it.

    The rejected and stale sets are the measurement of whether the gate works, so they
    get read by exactly the code that has the least business writing: a report, a `by
    status` listing, an operator looking at what expired. Nothing in this function's
    signature warns a caller that reading might cost them a status transition, so
    nothing downstream defends against it -- an audit of the rejected set would arrive
    at a different set than the one it was asked to audit, and the difference would
    look like data rather than like a bug.

    Pinned by a snapshot of every status and the whole staleness log taken either side
    of reading all three buckets. Adding any write to this function -- a `mark_stale`,
    a re-verification, a promotion of something that hashes again -- moves one of them.
    """
    root, conn = repo
    active = _admit(conn, [_acquire_span(root)])
    expired = _admit(conn, [_release_span(root)], claim="release returns False")
    refused = _admit(conn, [_acquire_span(root)], claim="acquire takes a lease first")
    store.mark_stale(conn, expired, store.REASON_HASH_MISMATCH)
    store.record_verdict(conn, refused, "judge/v1", store.VERDICT_REFUTED)

    def snapshot():
        return (
            [tuple(r) for r in conn.execute("SELECT id, status FROM assertions ORDER BY id")],
            [tuple(r) for r in store.staleness_events(conn)],
            [tuple(r) for r in conn.execute("SELECT * FROM verdicts ORDER BY id")],
        )

    before = snapshot()
    assert before[0] == [
        (active, store.STATUS_ACTIVE),
        (expired, store.STATUS_STALE),
        (refused, store.STATUS_REJECTED),
    ]

    read = {
        status: store.assertions_with_status(conn, status)
        for status in (store.STATUS_ACTIVE, store.STATUS_STALE, store.STATUS_REJECTED)
    }
    # Every bucket is non-empty, or "nothing changed" would be a claim about nothing.
    assert [a.id for a in read[store.STATUS_ACTIVE]] == [active]
    assert [a.id for a in read[store.STATUS_STALE]] == [expired]
    assert [a.id for a in read[store.STATUS_REJECTED]] == [refused]

    assert snapshot() == before
    # And reading twice returns the same thing, which it would not if the first read
    # had moved anything.
    assert [
        [a.id for a in store.assertions_with_status(conn, s)]
        for s in (store.STATUS_ACTIVE, store.STATUS_STALE, store.STATUS_REJECTED)
    ] == [[active], [expired], [refused]]


# --------------------------------------------------------------------------
# loading at repo scale -- the `IN (...)` that used to have a size limit
# --------------------------------------------------------------------------


def _bulk_rows(conn, count, *, spans_per=1, status=store.STATUS_ACTIVE):
    """Write `count` assertion rows and their spans directly, bypassing the gate.

    Directly, because the point is the SIZE of the id list `_load_assertions` binds,
    and 33,000 real files hashed through `write_assertion` would take minutes to prove
    something about a `?` count. Nothing here is a claim about admission; the rows only
    have to be shaped like admitted ones.
    """
    with db.transaction(conn):
        conn.executemany(
            "INSERT INTO assertions (subject_qualname, kind, claim, status) "
            "VALUES (?,?,?,?)",
            [(f"m.f{i}", "purpose", f"claim {i}", status) for i in range(count)],
        )
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM assertions WHERE status = ? ORDER BY id", (status,)
            )
        ]
        conn.executemany(
            "INSERT INTO evidence_spans (assertion_id, path, line_start, line_end, "
            " byte_start, byte_end, content_hash) VALUES (?,?,?,?,?,?,?)",
            [
                (aid, f"m{aid}.py", 1, 2, 0, 10 + n, f"hash-{aid}-{n}")
                for aid in ids
                for n in range(spans_per)
            ],
        )
    return ids


def test_loading_more_assertions_than_one_batch_returns_all_of_them_with_their_spans(tmp_path):
    """One over the batch size: the smallest input that has to cross a boundary.

    Chunking is the kind of fix that is easy to write so it only works on the first
    chunk, or so spans found in the second one overwrite rather than join the first.
    `_BATCH + 1` catches both -- one assertion lands alone in the second batch, and it
    has to come back carrying its citations like every other.
    """
    conn = db.init_db(tmp_path / "big.db")
    ids = _bulk_rows(conn, store._BATCH + 1, spans_per=2)

    loaded = store.assertions_with_status(conn, store.STATUS_ACTIVE)

    assert [a.id for a in loaded] == ids
    assert len(loaded) == store._BATCH + 1
    # Spans accumulate ACROSS batches rather than the later one replacing the earlier.
    assert {len(a.spans) for a in loaded} == {2}
    assert loaded[-1].spans[0].path == f"m{ids[-1]}.py"


def test_loading_past_sqlites_variable_limit_does_not_raise(tmp_path):
    """The bug this batching exists for, at the size where it fired.

    SQLITE_MAX_VARIABLE_NUMBER is 32,766, and an unbatched `IN (...)` binds one
    variable per assertion, so at 32,767 active claims `_load_assertions` raised
    `too many SQL variables` -- and it sits under `servable_assertions`,
    `assertions_with_status`, `serve_assertions` and `refresh_staleness`, so the whole
    tier-2 surface stopped answering at once, at exactly the index size it was built
    for. 33,000 rows is the smallest round number past the limit.
    """
    conn = db.init_db(tmp_path / "huge.db")
    ids = _bulk_rows(conn, 33_000)
    assert len(ids) > 32_766  # or this test is not past the limit it names

    loaded = store.assertions_with_status(conn, store.STATUS_ACTIVE)

    assert len(loaded) == 33_000
    assert all(len(a.spans) == 1 for a in loaded)


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
    sid = conn.execute(
        "SELECT id FROM symbols WHERE qualname = 'leases.acquire'"
    ).fetchone()["id"]
    fid = conn.execute("SELECT id FROM files WHERE path = 'leases.py'").fetchone()["id"]
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


# --------------------------------------------------------------------------------
# WP10: "we could not look" is not "the evidence changed".
#
# Every test below is written so that reverting the disposition split -- catching
# every OSError back into one `file_missing` expiry -- turns it red. The rule they
# name is that a failure of THIS PROCESS must never be recorded as a finding about
# the REPOSITORY, because the recording is irreversible and the failure is not.
# --------------------------------------------------------------------------------

# `chmod 000` denies nobody when the reader is root, so every test that leans on it
# would pass for the wrong reason in a container that runs as uid 0 -- and would go on
# passing after the behaviour it names was deleted. Skipped rather than xfailed: this
# is a property of the test environment, not a known defect.
skip_if_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="chmod 000 does not deny the superuser, so this proves nothing"
)


@contextlib.contextmanager
def _mode(path, mode):
    """Set a file's mode for the block, and put it back even if the block raises.

    A test that leaves a repo file at 000 poisons every test after it in ways that
    look like unrelated failures, and a bare try/finally per test is the kind of thing
    that gets omitted on the fifth copy.
    """
    original = path.stat().st_mode
    path.chmod(mode)
    try:
        yield
    finally:
        path.chmod(original)


@skip_if_root
def test_an_unreadable_file_withholds_the_claim_without_expiring_it(repo):
    """The defect, end to end. `chmod 000` used to expire every claim citing the file,
    log `file_missing` for a file that was sitting right there, and leave no way back:
    nothing in the store ever moved an assertion towards `active`, so `chmod 644`
    returned nothing. Withholding is right; withholding permanently, on evidence that
    never moved, is data loss with a reassuring log line.

    Deleting the `FileNotFoundError`/`OSError` split in `_read_source` turns this red
    on the last assertion, which is the one that matters."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    assert [a.id for a in store.servable_assertions(conn)] == [aid]

    with _mode(root / "leases.py", 0o000):
        # Withheld: the store cannot check it, so it will not serve it.
        assert store.servable_assertions(conn) == []
        assert not store.is_servable(conn, aid)
        # And NOT expired. The status is untouched and nothing was logged, because
        # nothing was established.
        row = conn.execute("SELECT status FROM assertions WHERE id=?", (aid,)).fetchone()
        assert row["status"] == store.STATUS_ACTIVE
        assert store.staleness_events(conn, aid) == []

    # The whole point: no operator action, no reinstate call, no SQL. The next read
    # that can open the file serves the claim again.
    assert [a.id for a in store.servable_assertions(conn)] == [aid]
    assert store.is_servable(conn, aid)


@skip_if_root
def test_an_unreadable_parent_directory_is_also_withheld_not_expired(repo):
    """The same rule one level up, and the one a moved or unmounted checkout hits.

    `stat()` on the file itself fails with EACCES when the directory above it is
    unsearchable -- the file is untouched and the path is intact, and the old code
    called that `file_missing` and expired on it."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.servable_assertions(conn)

    with _mode(root, 0o000):
        assert store.servable_assertions(conn) == []
        assert store.assertions_with_status(conn, store.STATUS_STALE) == []

    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_a_directory_where_a_cited_file_was_is_withheld_not_expired(repo):
    """Not-a-regular-file is the other half of the split, and it needs no permissions.

    A directory (or a FIFO, or a device node) at a cited path is not an absence:
    something is there, this reader cannot take bytes from it, and putting the file
    back must be enough. Expiring here would mean a mount point appearing over a
    vendored package destroys every claim beneath it."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.servable_assertions(conn)

    (root / "leases.py").unlink()
    (root / "leases.py").mkdir()
    assert store.servable_assertions(conn) == []
    assert store.assertions_with_status(conn, store.STATUS_STALE) == []
    assert store.staleness_events(conn, aid) == []

    (root / "leases.py").rmdir()
    (root / "leases.py").write_text(SOURCE)
    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_a_genuinely_deleted_file_still_expires(repo):
    """The control, and the reason the split is a split rather than a retreat.

    Real absence is still terminal and still logged as `file_missing`. A change that
    made everything non-terminal would pass every test above and would have removed
    the store's only mechanism for noticing a rename nothing followed."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").unlink()

    assert store.servable_assertions(conn) == []
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_FILE_MISSING


def test_a_path_whose_parent_is_a_file_is_a_real_absence(repo):
    """`NotADirectoryError`, the second terminal case. `pkg/mod.py/inner.py` cannot
    ever name a file, so it is an absence rather than a read this process fumbled --
    and it must not be parked in the recoverable bucket where nothing ever revisits
    it."""
    root, conn = repo
    (root / "pkg.py").write_text(SOURCE)
    span = store.span_for(root, "pkg.py", 0, ACQUIRE_END)
    span = dataclasses.replace(span, path="pkg.py/inner.py")
    aid = _admit(conn, [span], verify=False)

    assert store.servable_assertions(conn) == []
    assert store.staleness_events(conn, aid)[0]["reason"] == store.REASON_FILE_MISSING


@skip_if_root
def test_admission_refuses_an_unreadable_citation_as_unverifiable_not_stale(repo):
    """The door tells the two apart too, and with the right word.

    `EvidenceStale` means the bytes disagreed with the citation. Nothing disagreed
    here: nothing was read. Raising `EvidenceStale` would tell a generator its claim
    was wrong about the code and send it off to re-derive a claim that was fine, when
    the repair is to fix the filesystem and submit the identical claim again."""
    root, conn = repo
    span = _acquire_span(root)

    with _mode(root / "leases.py", 0o000):
        with pytest.raises(store.EvidenceUnverifiable, match="could not be read"):
            _admit(conn, [span])

    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0


# --------------------------------------------------------------------------------
# reinstate: the only way back, and it is the evidence that decides.
# --------------------------------------------------------------------------------

def test_reinstate_restores_a_stale_claim_when_every_citation_matches_again(repo):
    """The counterweight to a store with one direction of travel.

    Claims expired before the disposition split -- and claims expired by a revert that
    really did restore the bytes -- have to have a route back that is not hand-edited
    SQL. Deleting the `_TOUCH_STATUS` call leaves this red."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root), _release_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return NotImplemented"))
    assert store.servable_assertions(conn) == []

    (root / "leases.py").write_text(SOURCE)
    assert store.reinstate(conn, aid) is True
    assert [a.id for a in store.servable_assertions(conn)] == [aid]
    # The expiry stays in the log. It happened, and a record that is edited to match
    # the present is not a record.
    assert [e["reason"] for e in store.staleness_events(conn, aid)] == [
        store.REASON_HASH_MISMATCH
    ]


def test_reinstate_refuses_unless_every_cited_span_matches_exactly(repo):
    """Exact match of ALL of them, not the first, and not approximately.

    A claim resting on two spans is a claim about both, so restoring one of them
    restores nothing. This is the same first-failure verifier the serve path uses --
    a second implementation of "does the evidence hold" is a second answer waiting to
    differ from the first, and the direction it would differ in here is promotion."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root), _release_span(root)])
    (root / "leases.py").write_text(
        SOURCE.replace("return True", "return NotImplemented").replace(
            "return False", "return None"
        )
    )
    assert store.servable_assertions(conn) == []

    # Only the first cited span is back.
    partial = SOURCE.replace("return False", "return None")
    (root / "leases.py").write_text(partial)
    assert store.reinstate(conn, aid) is False
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]

    (root / "leases.py").write_text(SOURCE)
    assert store.reinstate(conn, aid) is True


def test_reinstate_refuses_a_rejected_claim(repo):
    """A refuted claim is not an expired one, and the bytes were never the dispute.

    A judge read this claim against evidence that was correct at the time and found it
    unsupported. Re-hashing that evidence cannot overturn the verdict -- and if it
    could, a `git checkout` would be able to. The promoted row would then be
    indistinguishable from one that was never refuted, because `verdicts` is a
    separate table nothing on the serve path reads.

    Raising rather than returning False, because False is this function's word for
    "the evidence still does not match", which is routine and unalarming, and asking
    to revive a refuted claim is neither."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    store.record_verdict(conn, aid, "judge/v1", store.VERDICT_REFUTED, "not supported")
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_REJECTED

    # The evidence is untouched and hashes perfectly. That is exactly the situation in
    # which a hash-driven promotion would be wrong.
    with pytest.raises(store.NotReinstatable, match="refuted claim is not an expired one"):
        store.reinstate(conn, aid)
    assert conn.execute(
        "SELECT status FROM assertions WHERE id=?", (aid,)
    ).fetchone()["status"] == store.STATUS_REJECTED


def test_reinstate_will_not_promote_a_claim_that_lost_its_evidence(repo):
    """The vacuous-truth guard, on the path that WRITES `active`.

    "Every cited span matches" is trivially true of no spans. The serve path already
    refuses to be fooled by that; this path is worse if it is, because it does not
    merely serve the claim, it restores it to the set everything else trusts."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").unlink()
    store.servable_assertions(conn)
    conn.execute("DELETE FROM evidence_spans WHERE assertion_id=?", (aid,))

    assert store.reinstate(conn, aid) is False
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]


@skip_if_root
def test_reinstate_will_not_promote_a_claim_it_cannot_read(repo):
    """"We could not check" is not grounds for promotion any more than for expiry.

    The symmetry is the point. The same unreadable file that must not expire a claim
    must not restore one either -- both would be a status written off a check that
    did not happen, and this direction is the more dangerous of the two because it
    ends with the claim being served."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").write_text(SOURCE.replace("return True", "return NotImplemented"))
    assert store.servable_assertions(conn) == []

    (root / "leases.py").write_text(SOURCE)
    with _mode(root / "leases.py", 0o000):
        assert store.reinstate(conn, aid) is False
        assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]

    assert store.reinstate(conn, aid) is True


def test_reinstate_is_a_no_op_on_a_claim_that_is_already_active(repo):
    """Not an error. Idempotence in the same shape `mark_stale` already has: False
    means this call performed no transition, and a caller sweeping a list of ids
    should not have to know which of them somebody else already fixed."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    assert store.reinstate(conn, aid) is False
    assert [a.id for a in store.servable_assertions(conn)] == [aid]


def test_reinstate_has_no_way_to_promote_a_claim_without_re_reading_it(repo):
    """The absence of an override is a feature, and this is the test that names it.

    A `force=` or `assume_fresh=` parameter would be a cached freshness verdict
    entered by hand -- the single failure the whole tier is built to refuse, and a
    worse one than a cache, because it would carry a human's authority and no
    timestamp. Adding such a parameter later must break this test on purpose."""
    root, conn = repo
    aid = _admit(conn, [_acquire_span(root)])
    (root / "leases.py").unlink()
    store.servable_assertions(conn)

    accepted = set(inspect.signature(store.reinstate).parameters)
    assert accepted == {"conn", "assertion_id", "repo_root"}
    assert store.reinstate(conn, aid) is False


def test_reinstate_on_an_unknown_id_is_a_caller_error(repo):
    _root, conn = repo
    with pytest.raises(KeyError):
        store.reinstate(conn, 4242)


def test_a_citation_that_leaves_the_repository_is_refused_at_the_door(repo, tmp_path):
    """REGRESSION, found by running the adversarial corpus against BOTH doors.

    `path_escapes_repo` lived only in `server.app._verify_span`. So the MCP tool
    refused this attack on every one of 12,803 generated instances, and
    `codelearner learn` -- plus every library caller -- admitted it. Reproduced before
    the fix: a claim citing `../outside_secret.py` with that file's real hash was
    stored `active` and reported `servable`.

    It is the worst shape this store can hold precisely because nothing downstream
    looks wrong. The bytes exist, the hash is correct, and it re-verifies on every
    serve, so `refresh_staleness` reports it fresh for as long as the file is
    untouched. The only thing wrong with it is that a reader of THIS repository
    cannot open what it cites -- which no check after admission can see."""
    root, conn = repo
    outside = tmp_path / "outside_secret.py"
    outside.write_text('SECRET = "hunter2"\n')
    raw = outside.read_bytes()

    escaping = store.EvidenceSpan(
        path="../outside_secret.py",
        line_start=1,
        line_end=1,
        byte_start=0,
        byte_end=len(raw),
        content_hash=content_hash(raw),
    )
    with pytest.raises(store.SpanEscapesRepo):
        store.write_assertion(
            conn,
            subject_qualname="leases.acquire",
            kind="purpose",
            claim="reads the deployment secret",
            spans=[escaping],
            verify=False,
        )
    assert conn.execute("SELECT count(*) c FROM assertions").fetchone()["c"] == 0
    assert conn.execute("SELECT count(*) c FROM evidence_spans").fetchone()["c"] == 0


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside.py",
        "pkg/../../outside.py",
        "a/b/../../../outside.py",
        "with\x00nul.py",
        "",
    ],
    ids=["absolute", "leading-dotdot", "buried-dotdot", "deep-dotdot", "nul", "empty"],
)
def test_every_shape_that_cannot_name_a_file_in_the_repo_is_refused(repo, path):
    """`..` is rejected wherever it appears, not only in the lead.

    `pkg/../../outside.py` escapes while never starting with one, and normalising
    first to find out would mean resolving a path this check deliberately does not
    touch the disk to resolve. A repo-relative citation has no legitimate reason to
    carry one in any position -- every span this package builds comes from
    `span_for_symbol`, whose paths are already normalised against the root."""
    _root, conn = repo
    span = store.EvidenceSpan(
        path=path,
        line_start=1,
        line_end=1,
        byte_start=0,
        byte_end=4,
        content_hash="0" * 64,
    )
    with pytest.raises(store.SpanEscapesRepo):
        store.write_assertion(
            conn,
            subject_qualname="leases.acquire",
            kind="purpose",
            claim="a claim",
            spans=[span],
            verify=False,
        )
