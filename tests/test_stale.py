"""The staleness engine: the two-stage check, the four reasons, the serving policy.

Same standard as `test_assertions.py`: every test here names a rule, and deleting the
rule has to turn the test red. Two rules in this module are unusually easy to write a
test that PASSES without them, and both get an explicit one:

* "the fast path did not read the file" -- an assertion about work NOT done. Tested
  with a spy on the one seam through which stage two touches disk, because a
  performance claim nobody can fail is a comment.
* "a touch is not an edit" -- a system that marked stale on mtime alone would pass
  every staleness test in this file except that one.
"""
from __future__ import annotations

import os
import time

import pytest

from codelearner import db
from codelearner.assertions import stale, store

SOURCE = (
    'def acquire(parcel_id):\n'
    '    """Take a lease."""\n'
    '    return True\n'
    '\n'
    '\n'
    'def release(parcel_id):\n'
    '    return False\n'
)

ACQUIRE_END = SOURCE.index("\n\n\n") + 1
RELEASE_START = SOURCE.index("def release")

# An edit INSIDE `acquire`, LONGER than what it replaces. Long enough that the file
# cannot read as a truncation, so it has to land as `hash_mismatch`.
EDIT_LONGER = SOURCE.replace("return True", "return NotImplemented")

# An edit inside `acquire` of exactly the same LENGTH. Every byte offset after it is
# unmoved, so a claim citing `release` further down the same file is untouched by it --
# which is the whole reason spans are hashed individually rather than per file.
EDIT_SAME_LEN = SOURCE.replace("return True", "return Fal5")
assert len(EDIT_SAME_LEN) == len(SOURCE)
assert EDIT_SAME_LEN != SOURCE


def _build(root_dir, db_path):
    """A one-file repo plus an index bound to it."""
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "leases.py").write_text(SOURCE)
    conn = db.init_db(db_path)
    db.bind_repo_root(conn, root_dir)
    return root_dir, conn


@pytest.fixture
def repo(tmp_path):
    return _build(tmp_path / "repo", tmp_path / "index.db")


def _admit(conn, root, *, start=0, end=ACQUIRE_END, qualname="leases.acquire", path="leases.py"):
    span = store.span_for(root, path, start, end)
    return store.write_assertion(
        conn,
        subject_qualname=qualname,
        kind="purpose",
        claim="acquire returns True when the lease is taken",
        spans=[span],
        generator="test-model/v1",
        confidence=0.9,
    )


def _served_ids(results):
    return [r.assertion.id for r in results]


def _reasons(conn):
    return [row["reason"] for row in store.staleness_events(conn)]


class _ReadSpy:
    """Counts every read that goes through the stage-two seam."""

    def __init__(self, real):
        self.real = real
        self.paths: list[str] = []

    def __call__(self, root, path):
        self.paths.append(path)
        return self.real(root, path)


# --------------------------------------------------------------------------------
# End to end: an edit expires the claim, and the claim is withheld.
# --------------------------------------------------------------------------------

def test_editing_a_cited_span_expires_the_assertion_and_withholds_it(repo):
    """The whole point, start to finish: edit the cited bytes, the claim stops coming back.

    Deleting the hash comparison in `_Pass.check_span` leaves this red -- the claim is
    still served after its evidence changed, which is the failure the tier-2 store
    exists to prevent.
    """
    root, conn = repo
    aid = _admit(conn, root)

    assert _served_ids(stale.serve_assertions(conn)) == [aid]

    # An edit INSIDE the cited range. Longer than what it replaces, so the file cannot
    # be short enough to read as a truncation -- this has to land as `hash_mismatch`.
    (root / "leases.py").write_text(EDIT_LONGER)

    assert stale.serve_assertions(conn) == []
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]
    assert _reasons(conn) == [stale.REASON_HASH_MISMATCH]

    event = store.staleness_events(conn, aid)[0]
    assert event["expected_hash"] is not None
    assert event["observed_hash"] is not None
    assert event["expected_hash"] != event["observed_hash"]


def test_expired_claim_reports_the_status_it_was_given(repo):
    """`stale=True` and `assertion.status` never disagree about the same claim."""
    root, conn = repo
    _admit(conn, root)
    stale.serve_assertions(conn)
    (root / "leases.py").write_text(EDIT_LONGER)

    (served,) = stale.serve_assertions(conn, include_stale=True)
    assert served.stale is True
    assert served.assertion.status == store.STATUS_STALE


# --------------------------------------------------------------------------------
# Stage one: the fast path, and the proof that it is one.
# --------------------------------------------------------------------------------

def test_unchanged_mtime_takes_the_fast_path_and_does_not_reread_the_file(repo, monkeypatch):
    """An unchanged file is confirmed by `stat()` alone -- no read, no hash.

    The assertion is about work NOT done, so it is made with a spy on both the read
    seam and the hash function. Delete the fast-path branch in `check_span` and both
    counters go up and this test fails; the correctness tests around it would not.
    """
    root, conn = repo
    aid = _admit(conn, root)

    first = stale.serve_assertions(conn)  # pays the full hash, records the baseline
    assert _served_ids(first) == [aid]
    assert first[0].method == stale.METHOD_HASH

    spy = _ReadSpy(stale._read_file)
    monkeypatch.setattr(stale, "_read_file", spy)
    hashed: list[bytes] = []
    real_hash = stale.content_hash
    monkeypatch.setattr(stale, "content_hash", lambda b: hashed.append(b) or real_hash(b))

    (served,) = stale.serve_assertions(conn)

    assert spy.paths == [], "the fast path read the file it was supposed to skip"
    assert hashed == [], "the fast path re-hashed the bytes it was supposed to skip"
    assert served.stale is False
    assert served.method == stale.METHOD_STAT


def test_fast_path_reports_the_age_of_the_hash_not_the_age_of_the_check(repo):
    """`verified_at` is when the bytes were last hashed; `checked_at` is when we looked.

    Collapsing the two would make a stat-confirmed claim indistinguishable from a
    freshly hashed one -- which is precisely the cached-verdict lie this engine is
    built to avoid. Delete the distinction and `checked_at > verified_at` fails.
    """
    root, conn = repo
    _admit(conn, root)
    (first,) = stale.serve_assertions(conn)

    time.sleep(0.01)  # SQLite stamps milliseconds; make the two instants distinguishable
    (second,) = stale.serve_assertions(conn)

    assert second.method == stale.METHOD_STAT
    assert second.verified_at == first.checked_at
    assert second.checked_at > second.verified_at


def test_a_span_that_was_never_hashed_cannot_take_the_fast_path(repo):
    """No baseline, no shortcut. The first check of a citation always reads.

    The rule that makes the fast path safe to start from: `span_verifications` has a
    row only where a hash was actually witnessed, so a brand-new claim can never be
    served on a stat alone.
    """
    root, conn = repo
    aid = _admit(conn, root)

    report = stale.refresh_staleness(conn)
    assert (report.spans_hashed, report.spans_fast_pathed) == (1, 0)

    again = stale.refresh_staleness(conn)
    assert (again.spans_hashed, again.spans_fast_pathed) == (0, 1)
    assert again.files_read == 0

    rows = stale.verification_state(conn, aid)
    assert len(rows) == 1
    assert rows[0]["verified_hash"] == rows[0]["cited_hash"]


def test_a_baseline_never_vouches_for_a_hash_it_did_not_witness(repo):
    """The stat baseline must match the hash the span cites NOW, or it is ignored.

    Span rows keep their ids across a re-citation. Without this guard an old witness
    would authorise skipping the read for bytes nobody ever checked -- the fast path
    would be vouching for a hash it never saw.
    """
    root, conn = repo
    aid = _admit(conn, root)
    stale.serve_assertions(conn)  # establishes the baseline

    # Re-cite the same bytes under a different expected hash, leaving the baseline.
    with db.transaction(conn):
        conn.execute(
            "UPDATE evidence_spans SET content_hash = ? WHERE assertion_id = ?",
            ("0" * 64, aid),
        )

    assert stale.serve_assertions(conn) == []
    assert _reasons(conn) == [stale.REASON_HASH_MISMATCH]


# --------------------------------------------------------------------------------
# A touch is not an edit.
# --------------------------------------------------------------------------------

def test_moved_mtime_with_identical_content_does_not_mark_stale(repo):
    """`touch` moves mtime and changes nothing. The claim must survive it.

    An engine that expired on the stat alone would pass every other staleness test
    here. This is the one that catches it. It also checks the re-baseline: the cost of
    a touch is one extra read, once, not forever.
    """
    root, conn = repo
    aid = _admit(conn, root)
    stale.serve_assertions(conn)

    path = root / "leases.py"
    moved = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(moved, moved))

    (served,) = stale.serve_assertions(conn)
    assert served.stale is False
    assert served.assertion.id == aid
    assert served.method == stale.METHOD_HASH, "a moved mtime must fall through to the hash"
    assert store.staleness_events(conn) == []
    assert store.assertions_with_status(conn, store.STATUS_STALE) == []

    # And the new mtime became the baseline, so the next check is cheap again.
    (again,) = stale.serve_assertions(conn)
    assert again.method == stale.METHOD_STAT


# --------------------------------------------------------------------------------
# The four reasons, kept apart.
# --------------------------------------------------------------------------------

def test_deleted_file_marks_file_missing(repo):
    """A cited file that is gone is `file_missing`, not a hash mismatch."""
    root, conn = repo
    aid = _admit(conn, root)
    stale.serve_assertions(conn)

    (root / "leases.py").unlink()

    assert stale.serve_assertions(conn) == []
    assert _reasons(conn) == [stale.REASON_FILE_MISSING]
    event = store.staleness_events(conn, aid)[0]
    # There is no observed hash for a file that is not there, and that absence is the
    # finding rather than a gap in it.
    assert event["observed_hash"] is None


def test_shortened_file_marks_span_truncated(repo):
    """A file now shorter than the cited range is `span_truncated`, not a mismatch.

    Slicing past the end of a bytes object returns a short result that would simply
    hash to "not what was expected", so a deleted tail would be indistinguishable from
    an edit unless it is named separately.
    """
    root, conn = repo
    _admit(conn, root, start=RELEASE_START, end=len(SOURCE), qualname="leases.release")
    stale.serve_assertions(conn)

    (root / "leases.py").write_text(SOURCE[:RELEASE_START])

    assert stale.serve_assertions(conn) == []
    assert _reasons(conn) == [stale.REASON_SPAN_TRUNCATED]


def test_assertion_with_no_spans_is_never_served(repo):
    """The vacuous-truth guard: no evidence is not the same as nothing wrong.

    "Every cited span still matches" is trivially TRUE of no spans, so an assertion
    that lost its evidence would be promoted by the exact code that verifies a
    well-cited one. Delete the empty check and this claim gets served.
    """
    root, conn = repo
    aid = _admit(conn, root)
    with db.transaction(conn):
        conn.execute("DELETE FROM evidence_spans WHERE assertion_id = ?", (aid,))

    assert stale.serve_assertions(conn) == []
    assert _reasons(conn) == [stale.REASON_NO_EVIDENCE]


def test_the_four_reasons_are_not_collapsed(tmp_path):
    """Four different repo states produce four different reasons.

    Named as one test because the rule is about the SET: any change that funnels two
    of these into one flag passes each individual reason test above and fails here.
    """
    seen = []
    cases = {
        "edit": lambda p: p.write_text(EDIT_LONGER),
        "delete": lambda p: p.unlink(),
        "truncate": lambda p: p.write_text(SOURCE[:2]),
    }
    for name, mutate in cases.items():
        root, conn = _build(tmp_path / name, tmp_path / f"{name}.db")
        _admit(conn, root)
        stale.serve_assertions(conn)
        mutate(root / "leases.py")
        stale.serve_assertions(conn)
        seen.extend(_reasons(conn))

    root, conn = _build(tmp_path / "empty", tmp_path / "empty.db")
    aid = _admit(conn, root)
    with db.transaction(conn):
        conn.execute("DELETE FROM evidence_spans WHERE assertion_id = ?", (aid,))
    stale.serve_assertions(conn)
    seen.extend(_reasons(conn))

    assert set(seen) == {
        stale.REASON_HASH_MISMATCH,
        stale.REASON_FILE_MISSING,
        stale.REASON_SPAN_TRUNCATED,
        stale.REASON_NO_EVIDENCE,
    }


# --------------------------------------------------------------------------------
# The documented hole in the fast path, and the way out of it.
# --------------------------------------------------------------------------------

def test_force_hash_catches_an_edit_that_preserved_mtime_and_size(repo):
    """The fast path trusts mtime+size; `force_hash=True` does not.

    This test asserts the LIMITATION as well as the escape, on purpose. An edit that
    restores the timestamp and keeps the length is invisible to `stat()`, the module
    says so in as many words, and a test that pretended otherwise would be the
    documentation drifting from the code. Delete `force_hash` and the second half
    fails; make the fast path silently re-hash and the first half fails.
    """
    root, conn = repo
    _admit(conn, root)
    stale.serve_assertions(conn)

    path = root / "leases.py"
    before = path.stat()
    # Same length, different bytes, timestamp put back exactly.
    path.write_text(EDIT_SAME_LEN)
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_mtime_ns, before.st_mtime_ns))

    (missed,) = stale.serve_assertions(conn)
    assert missed.stale is False
    assert missed.method == stale.METHOD_STAT

    assert stale.serve_assertions(conn, force_hash=True) == []
    assert _reasons(conn) == [stale.REASON_HASH_MISMATCH]


# --------------------------------------------------------------------------------
# Serving policy.
# --------------------------------------------------------------------------------

def test_stale_is_withheld_by_default_and_labelled_when_asked_for(repo):
    """Default serve hides stale claims; `include_stale` returns them, always marked.

    Both halves matter. A caller that did not ask about staleness will not read a flag,
    so the default has to withhold; a caller that did ask needs the reason and the
    citation that moved, or "stale" is just a word.
    """
    root, conn = repo
    other = root / "other.py"
    other.write_text(SOURCE)
    doomed = _admit(conn, root)
    survivor = _admit(conn, root, path="other.py", qualname="other.acquire")
    stale.serve_assertions(conn)

    (root / "leases.py").write_text(EDIT_LONGER)

    # The call that detects the expiry labels it with the citation that actually moved.
    both = stale.serve_assertions(conn, include_stale=True)
    assert _served_ids(both) == [doomed, survivor]
    by_id = {r.assertion.id: r for r in both}
    assert by_id[doomed].stale is True
    assert by_id[doomed].reason == stale.REASON_HASH_MISMATCH
    assert by_id[doomed].failing_span.citation == "leases.py:1-3"
    assert by_id[doomed].observed_hash is not None
    assert by_id[doomed].label.startswith("STALE (hash_mismatch at leases.py:1-3")
    assert by_id[survivor].stale is False
    assert by_id[survivor].label.startswith("fresh (")

    # And a caller that did not ask never sees it at all.
    assert _served_ids(stale.serve_assertions(conn)) == [survivor]


def test_a_previously_stale_claim_is_still_withheld_and_still_labelled(repo):
    """Staleness survives the call that detected it, and is never silently re-checked.

    A claim already stale is not re-verified: a revert or a rename-back would otherwise
    resurrect a claim nothing has re-adjudicated. Its `checked_at` is the moment of
    expiry, not now, because nothing looked at it now.
    """
    root, conn = repo
    aid = _admit(conn, root)
    stale.serve_assertions(conn)
    (root / "leases.py").write_text(EDIT_LONGER)
    (flipped,) = stale.serve_assertions(conn, include_stale=True)

    (root / "leases.py").write_text(SOURCE)  # put it back; the claim stays stale

    assert stale.serve_assertions(conn) == []
    (again,) = stale.serve_assertions(conn, include_stale=True)
    assert again.assertion.id == aid
    assert again.stale is True
    assert again.reason == stale.REASON_HASH_MISMATCH
    assert again.method is None, "nothing re-checked it, so no stage may be claimed"
    assert again.checked_at == flipped.checked_at
    # One expiry event, not one per read.
    assert len(store.staleness_events(conn, aid)) == 1


def test_one_expiry_carries_one_timestamp(repo, monkeypatch):
    """The instant reported as `checked_at` is the instant written to the log.

    This pins deterministically what the test above only catches by luck. Detection
    reads the clock once and reports it as `checked_at`; if `mark_stale` lets the
    column default fire a SECOND read, the two drift by however long the write took
    -- usually nothing, occasionally a millisecond. A later call faithfully reporting
    the logged `detected_at` then contradicts what the detecting call returned, and
    the disagreement appears only when the writes straddle a tick.

    Freezing the clock makes the gap unmissable rather than probabilistic: with one
    timestamp the log holds the frozen value, with two it holds a real one.
    """
    root, conn = repo
    aid = _admit(conn, root)
    stale.serve_assertions(conn)
    (root / "leases.py").write_text(EDIT_LONGER)

    frozen = "2000-01-01T00:00:00.000Z"
    monkeypatch.setattr(stale, "_now", lambda _conn: frozen)

    (flipped,) = stale.serve_assertions(conn, include_stale=True)
    assert flipped.checked_at == frozen

    (event,) = store.staleness_events(conn, aid)
    assert event["detected_at"] == frozen, (
        "the expiry was logged at a different instant than the one reported to the "
        "caller -- one event, two clock reads"
    )

    # And the later read must hand back that same instant.
    (again,) = stale.serve_assertions(conn, include_stale=True)
    assert again.checked_at == frozen


def test_served_claim_carries_the_hashes_it_is_bound_to(repo):
    """The binding travels with the claim, so it cannot get separated from it."""
    root, conn = repo
    _admit(conn, root)
    (served,) = stale.serve_assertions(conn)
    ((citation, bound),) = served.bound_hashes
    assert citation == "leases.py:1-3"
    assert bound == store.span_for(root, "leases.py", 0, ACQUIRE_END).content_hash


def test_serving_filters_apply_to_stale_claims_too(repo):
    """`subject_qualname` means the same thing whichever set a claim landed in."""
    root, conn = repo
    other = root / "other.py"
    other.write_text(SOURCE)
    doomed = _admit(conn, root)
    _admit(conn, root, path="other.py", qualname="other.acquire")
    stale.serve_assertions(conn)
    (root / "leases.py").write_text(EDIT_LONGER)
    stale.serve_assertions(conn)

    scoped = stale.serve_assertions(
        conn, subject_qualname="leases.acquire", include_stale=True
    )
    assert _served_ids(scoped) == [doomed]
    assert scoped[0].stale is True


# --------------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------------

def test_refresh_staleness_expires_what_moved_and_counts_what_it_did(tmp_path):
    """The sweep flips the claims whose evidence moved and reports real numbers.

    The counts are the observable: `spans_hashed` staying at zero over an unchanged
    repo is the only evidence anyone gets that the fast path is working at all.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    (root / "other.py").write_text(SOURCE)
    conn = db.init_db(tmp_path / "index.db")
    db.bind_repo_root(conn, root)

    doomed = _admit(conn, root)
    _admit(conn, root, start=RELEASE_START, end=len(SOURCE), qualname="leases.release")
    _admit(conn, root, path="other.py", qualname="other.acquire")

    first = stale.refresh_staleness(conn)
    assert (first.checked, first.fresh, first.expired) == (3, 3, 0)
    assert first.spans_hashed == 3
    assert first.files_statted == 2, "one stat per distinct cited file, not per span"
    assert first.files_read == 2

    quiet = stale.refresh_staleness(conn)
    assert (quiet.checked, quiet.fresh, quiet.expired) == (3, 3, 0)
    assert (quiet.spans_hashed, quiet.spans_fast_pathed) == (0, 3)
    assert quiet.files_read == 0
    assert quiet.by_reason == {}

    # An edit inside `acquire` only, and of the same length. `release` cites bytes
    # further down the same file and must survive it -- span-level hashing is the
    # difference between staleness that gets acted on and staleness that gets ignored.
    (root / "leases.py").write_text(EDIT_SAME_LEN)

    swept = stale.refresh_staleness(conn)
    assert (swept.checked, swept.fresh, swept.expired) == (3, 2, 1)
    assert swept.by_reason == {stale.REASON_HASH_MISMATCH: 1}
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [doomed]
    assert "1 hash_mismatch" in swept.summary()

    # Already-stale claims leave the active set, so the sweep shrinks.
    after = stale.refresh_staleness(conn)
    assert after.checked == 2


def test_an_unchanged_repo_costs_no_writes_at_all(repo):
    """A serve or sweep that changes nothing takes no write lock.

    The buffered-writes design exists for this. Issuing baseline upserts as they are
    found would mean every read path opened a transaction on every query -- and a read
    path that writes is not a read path, it is a contention source that shows up only
    under concurrency, which is where it is hardest to find.
    """
    root, conn = repo
    _admit(conn, root)
    stale.serve_assertions(conn)  # the first pass writes the baseline; that one is expected

    before = conn.total_changes
    stale.serve_assertions(conn)
    stale.refresh_staleness(conn)
    assert conn.total_changes == before

    # And the writes come back the moment something actually moves.
    (root / "leases.py").write_text(EDIT_LONGER)
    stale.serve_assertions(conn)
    assert conn.total_changes > before


def test_refresh_staleness_reaches_claims_nothing_ever_queries(repo):
    """The sweep exists for the claims no query touches. Those are most of them."""
    root, conn = repo
    _admit(conn, root)
    stale.refresh_staleness(conn)
    (root / "leases.py").unlink()

    report = stale.refresh_staleness(conn)
    assert report.expired == 1
    assert report.by_reason == {stale.REASON_FILE_MISSING: 1}


# --------------------------------------------------------------------------------
# The two verifiers must not disagree.
# --------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("untouched", lambda p: None),
        ("edited", lambda p: p.write_text(EDIT_LONGER)),
        ("deleted", lambda p: p.unlink()),
        ("truncated", lambda p: p.write_text(SOURCE[:2])),
        ("touched", lambda p: os.utime(p, ns=(p.stat().st_mtime_ns + 5_000_000_000,) * 2)),
    ],
)
def test_fast_path_agrees_with_the_unconditional_verifier(tmp_path, name, mutate):
    """The two-stage check and `store.servable_assertions` reach the same verdict.

    Two indexes over ONE repo: the fast one and the one that re-hashes everything on
    every call. The fast path is an optimisation, so any repo state where they differ
    is a bug in this module by definition -- and "we made it fast and it started
    disagreeing" is the failure that would otherwise be found in production.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)

    fast = db.init_db(tmp_path / "fast.db")
    slow = db.init_db(tmp_path / "slow.db")
    for conn in (fast, slow):
        db.bind_repo_root(conn, root)
        _admit(conn, root)
        _admit(conn, root, start=RELEASE_START, end=len(SOURCE), qualname="leases.release")

    # Both establish a baseline before the repo moves under them.
    stale.serve_assertions(fast)
    store.servable_assertions(slow)

    mutate(root / "leases.py")

    fast_ids = _served_ids(stale.serve_assertions(fast))
    slow_ids = [a.id for a in store.servable_assertions(slow)]
    assert fast_ids == slow_ids, f"verifiers disagree on a {name} repo"
    assert _reasons(fast) == _reasons(slow), f"reasons differ on a {name} repo"
