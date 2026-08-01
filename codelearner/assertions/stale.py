"""The staleness engine: a two-stage freshness check that reports how fresh it was.

`store.servable_assertions` is correct and expensive. It re-reads and re-hashes every
cited byte range on every call, so a query touching forty claims about one module does
forty slices and forty sha256s, and a repo-scale index does that on every question.
That cost is the reason a system like this drifts toward caching a freshness verdict,
and a cached freshness verdict is the exact failure the tier-2 store exists to
prevent: it looks identical to a real check right up until it is wrong.

So this module makes the check cheap WITHOUT making it a cache of the answer.

**Two stages.** Stage one is a `stat()`: if the cited file's `st_mtime_ns` and
`st_size` are byte-for-byte what they were when this span was last actually hashed
(`span_verifications`), the bytes are taken to be unchanged and nothing is read.
Stage two is the full re-hash, and it runs whenever the stat differs, whenever a span
has never been hashed at all, and whenever a caller asks for it. Steady state is one
`stat()` per distinct cited FILE per query and zero reads; the first pass over a fresh
index, and every pass after an edit, pays the full read.

**What the fast path actually promises, and what it does not.** It promises the file's
mtime and size are unchanged since the hash was witnessed. It does NOT promise the
bytes are unchanged: a writer that restores mtime, or an edit landing inside the
filesystem's timestamp granularity that also preserves length, defeats it. That is a
real hole and the honest response is not to pretend it is closed. It is to make every
served claim state which stage confirmed it and WHEN the hash was last actually seen,
so a caller reading `method='stat', verified_at=<three days ago>` knows precisely what
it is holding -- and to give anyone who wants the guarantee back a `force_hash=True`
that skips stage one entirely. A freshness check that lies is one that reports a
single boolean; this one reports its own provenance.

**A touch is not an edit.** `touch file.py` moves mtime, so stage one misses and stage
two runs -- and stage two finds the hash unchanged, records the new mtime as the
baseline, and marks nothing stale. The fast path is an accelerator over the hash, never
an authority beside it; only the hash can expire a claim.

**Failure modes stay apart.** `hash_mismatch`, `file_missing`, `span_truncated` and
`no_evidence` are the four terminal reasons `store` names, and this engine preserves all
four rather than collapsing them into "stale". They call for different repairs: an
edited function should be re-derived, a missing file is usually a rename that nothing
followed, a truncated span is a deleted tail, and `no_evidence` is a bug in the write
gate rather than a fact about the repo. One flag would make those four indistinguishable
in the log, and the log is the only place the difference is visible.

**`unreadable` is the fifth reason and the only non-terminal one.** A cited file that
is present but cannot be opened -- a permission bit, `EMFILE`, `EIO`, an NFS blip, a
FIFO -- establishes nothing about the bytes, so the claim is WITHHELD for this call and
its status is left exactly as it was found. It is not served: this module's contract is
that a result outside the stale bucket has been checked, and an unchecked claim sitting
in it is the cached-freshness-verdict failure with extra steps. It is not expired
either: expiring on a permission bit is how `chmod 000` used to destroy every claim
citing a file, irreversibly, while logging that the file was missing. A caller who
wants to see them asks for them by name (`include_unverifiable=True`), and a sweep
counts them separately (`RefreshReport.unverifiable`), because a number that quietly
lands in `fresh` is a report claiming a check it did not perform.

**The fast path must never serve what `force_hash` would withhold.** That is why
`_stat_file` tests readability rather than only existence. `stat()` succeeds on a
`chmod 000` file, and its mtime and size are unchanged, so the stat baseline matched
and the claim was served as `fresh, method='stat'` -- while `store.servable_assertions`,
which has to actually open the file, withheld it. Two verifiers, one repo state, two
answers; and worse, `serve_assertions()` and `serve_assertions(force_hash=True)`
disagreed with each other on the same index in the same second. An accelerator that can
outrun the authority it accelerates is not an accelerator. The readability test costs
one `faccessat` per distinct cited FILE per pass, cached beside the `stat`, and it
restores the invariant that stage one only ever reaches conclusions stage two would
also reach. It inherits `os.access`'s known limits -- real rather than effective ids,
and ACL or NFS setups where the client's answer is not the server's -- so the residual
is that stage one can still be optimistic about an exotic filesystem, and `force_hash`
remains the way to buy the exact answer.

**Serving.** `serve_assertions` withholds anything stale by default. `include_stale=True`
returns them, and everything it returns is labelled with its status, its reason, the
stage that checked it and the age of its hash binding -- there is no way to receive a
stale claim from this module without also receiving the word "stale". Unreadable claims
need a SECOND opt-in, `include_unverifiable=True`, and are withheld from
`include_stale=True` as well: the natural way to consume that flag is
`if r.stale: ... else: <treat as fresh>`, so a record with `stale=False` and nothing
verified would land in the `else` and be presented as checked -- the failure this whole
module exists to prevent, arriving through the flag added to prevent it.

**What it actually bought, measured.** Not what was expected, and the number belongs
here rather than buried. Serving every claim in an index, median of interleaved A/B
blocks, warm page cache:

    index                  spans   cited bytes   always-rehash   two-stage   ratio
    code-learner             383      0.28 MiB        4.97 ms     6.91 ms    0.72x
    swarm-sync             1,100      0.95 MiB       13.99 ms    20.09 ms    0.70x
    synthetic 8 x 128 KiB    168      1.01 MiB        3.21 ms     3.53 ms    0.91x
    synthetic 8 x 256 KiB    168      2.01 MiB        4.35 ms     3.41 ms    1.28x
    synthetic 8 x 512 KiB    168      4.01 MiB        5.03 ms     2.62 ms    1.92x
    synthetic 8 x 1 MiB      168      8.01 MiB        8.65 ms     2.60 ms    3.33x

On both REAL repositories the fast path is about 1.4x SLOWER. sha256 over a page-cached
Python file is far cheaper than the premise assumed -- re-hashing every cited byte in
swarm-sync costs under a millisecond -- while the extra `span_verifications` lookup and
the per-span record-keeping cost a few microseconds per span whether or not anything
moved. Below roughly 1.5 MiB of cited bytes per query the unconditional re-hash wins.

What the table shows is the shape, not the constant: the two-stage column is FLAT in
file size (2.6-3.5 ms across an 8x range) because it is O(spans); the re-hash column
grows with bytes because it is O(bytes). So this is the path that holds up as cited
volume grows, and the only one whose cost does not depend on how the filesystem feels
that day -- a page-cache miss, an NFS mount or an encrypted volume moves the re-hash
column and leaves this one alone. On a small warm repo it is a real loss, and
`store.servable_assertions` remains the better call there; saying so here is cheaper
than someone discovering it later.
"""
from __future__ import annotations

import os
import sqlite3
import stat as stat_module
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import db
from ..ingest.types import content_hash
from .store import (
    _ABSENT,
    _UNREADABLE,
    # Re-exported, not used directly any more: `_stat_file` and `_read_file` now
    # return `_ABSENT`, which already carries this reason. The five reasons must stay
    # readable off this module as one set -- the docstring above enumerates them, and
    # a caller that gets its verdicts from `stale` should not have to import `store`
    # to name what it just received.
    REASON_FILE_MISSING,  # noqa: F401
    REASON_HASH_MISMATCH,
    REASON_NO_EVIDENCE,
    REASON_SPAN_TRUNCATED,
    REASON_UNREADABLE,
    STATUS_ACTIVE,
    STATUS_STALE,
    Assertion,
    EvidenceSpan,
    _atomic,
    _load_assertions,
    _repo_root,
    _Unread,
    mark_stale,
)

# Which stage established a result. Reported on every served claim, because "we
# checked" and "we checked how" are different statements and only the second one lets
# a caller judge the answer.
METHOD_STAT = "stat"
METHOD_HASH = "hash"

# SQLite's own timestamp expression, reused verbatim from the schema defaults. Every
# timestamp this engine writes or reports comes from the DB's clock rather than
# Python's -- two clocks stamping rows in one table is how "verified before it was
# created" gets into the data.
_NOW_SQL = "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

# SQLite's variable limit is in the tens of thousands, and a repo-scale sweep can have
# more spans than that. Batching the `IN` lookups keeps the sweep from failing at
# exactly the size it was built for.
_BATCH = 500


@dataclass(frozen=True, slots=True)
class _Stat:
    """The cheap half of the check: what `stat()` says about a cited file."""

    mtime_ns: int
    size: int


@dataclass(frozen=True, slots=True)
class _Baseline:
    """What was witnessed the last time a span's bytes were actually hashed.

    `verified_hash` is stored alongside the stat rather than inferred from the span,
    so the fast path can require that the baseline vouches for *this* hash. Without
    it, re-citing a span with a different hash would leave a stat baseline that
    authorises skipping the read for bytes nobody ever checked.
    """

    mtime_ns: int
    size_bytes: int
    verified_hash: str
    verified_at: str


@dataclass(frozen=True, slots=True)
class SpanCheck:
    """The result of checking one citation, including which stage produced it.

    `slots=True` because these are allocated two per served claim and a serve is the
    hot path: on this repo's own index (383 claims) the record-keeping was measured at
    twice the cost of the file reads it saves, so the record-keeping is what needs to
    be cheap.
    """

    span: EvidenceSpan
    method: str
    ok: bool
    # When these exact bytes were last read and hashed. None means never -- which can
    # only happen on a span whose first check is the one that just failed.
    verified_at: str | None = None
    reason: str | None = None
    observed_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ServedAssertion:
    """A claim handed back with the provenance of its own freshness check.

    Three timestamps-worth of honesty, deliberately not collapsed into one:

    * `checked_at` -- when this pass looked. For anything re-checked that is now; for
      a claim that was ALREADY stale it is when the expiry was detected, because
      nothing re-checks a stale claim and pretending otherwise would be the lie.
    * `verified_at` -- when the cited bytes were last actually hashed. On a fast-path
      hit this is OLDER than `checked_at`, and that gap is the number worth showing.
    * `method` -- `'stat'` or `'hash'`, whichever stage the weakest of its citations
      relied on. Weakest link: one stat-confirmed span makes the whole claim
      stat-confirmed, since the claim is only as verified as its least-verified
      evidence. None if this pass did not check at all.
    """

    assertion: Assertion
    stale: bool
    checked_at: str
    verified_at: str | None
    method: str | None
    checks: tuple[SpanCheck, ...] = ()
    reason: str | None = None
    failing_span: EvidenceSpan | None = None
    observed_hash: str | None = None
    # A cited file could not be opened, so nothing was established either way. Kept as
    # its OWN field rather than folded into `stale`, because `stale` drives `_expire`
    # and means "the evidence moved" -- a claim nobody could check has not moved, and
    # writing it into the status is the irreversible mistake WP10 exists to undo. The
    # two are mutually exclusive and both are False on a claim that verified.
    unreadable: bool = False

    @property
    def bound_hashes(self) -> tuple[tuple[str, str], ...]:
        """`(citation, content_hash)` for every span, i.e. what this claim is bound to.

        Travels with the claim so a caller can print the binding it was served under
        without a second query -- and so a claim and the hashes it rests on cannot get
        separated on the way to whoever has to trust it.
        """
        return tuple((s.citation, s.content_hash) for s in self.assertion.spans)

    @property
    def label(self) -> str:
        """One line naming the status and the strength of the check behind it.

        Every path out of this module that can emit a stale or unchecked claim goes
        through here, so a caller cannot end up holding one that is not visibly marked.

        The unreadable line says UNVERIFIED rather than STALE and names the withholding
        explicitly. A reader skimming labels must not be able to mistake it for either
        neighbour: it is not `fresh` (nothing was checked) and it is not `STALE`
        (nothing was found wrong, and no status changed).
        """
        if self.unreadable:
            where = f" at {self.failing_span.citation}" if self.failing_span else ""
            return f"UNVERIFIED (could not read{where}) -- withheld, status unchanged"
        if self.stale:
            where = f" at {self.failing_span.citation}" if self.failing_span else ""
            return f"STALE ({self.reason or 'unknown'}{where})"
        return f"fresh (checked by {self.method}, hash verified {self.verified_at})"


@dataclass(frozen=True)
class RefreshReport:
    """What a sweep did, in the terms that say whether the fast path is working.

    `spans_fast_pathed` vs `spans_hashed` is the measurement the whole module exists
    to make: on an unchanged repo the second should be zero, and if it is not, the
    baseline is being invalidated by something (a checkout that rewrites mtimes, a
    formatter, a build step) and the fast path is buying nothing.

    `unverifiable` is the third bucket, and it exists because the other two would
    otherwise have to absorb it and both would lie. Counting an unreadable claim as
    `fresh` is a report asserting a check that did not happen -- and this report is the
    only evidence anyone gets that the fast path works, so a number in it that means
    something else than it says poisons the one measurement. Counting it as `expired`
    is worse: it puts a reason in `by_reason` for a claim that did not expire, implies
    data loss that did not occur, and sends an operator looking for an edit nobody
    made. So it is its own count, `checked == fresh + expired + unverifiable` holds
    exactly, and `by_reason` continues to sum to `expired` alone.

    `files_unreadable` is the cause rather than the effect: one `chmod 000` module can
    withhold forty claims, and "1 file" is the actionable number while "40 claims" is
    the alarming one. Both are reported because an operator needs to know the blast
    radius and where to point the fix.
    """

    checked: int = 0
    fresh: int = 0
    expired: int = 0
    by_reason: Mapping[str, int] = field(default_factory=dict)
    files_statted: int = 0
    files_read: int = 0
    spans_fast_pathed: int = 0
    spans_hashed: int = 0
    # Assertions withheld because a cited file could not be read. NOT expired: their
    # stored status is untouched and they return on the next healthy pass.
    unverifiable: int = 0
    # Distinct cited files that could not be read this pass.
    files_unreadable: int = 0

    def summary(self) -> str:
        reasons = ", ".join(f"{n} {r}" for r, n in sorted(self.by_reason.items()))
        detail = f" ({reasons})" if reasons else ""
        # Shown only when there is something to show. A permanent ", 0 unverifiable"
        # is noise an operator learns to skip, and this is the line it must not be
        # possible to skip on the one day it is not zero.
        withheld = (
            f", {self.unverifiable} withheld unread "
            f"({self.files_unreadable} unreadable files)"
            if self.unverifiable
            else ""
        )
        return (
            f"{self.checked} active, {self.fresh} still fresh, "
            f"{self.expired} expired{detail}{withheld}; "
            f"{self.files_statted} stat, {self.files_read} read, "
            f"{self.spans_fast_pathed} spans fast-pathed, "
            f"{self.spans_hashed} re-hashed"
        )


def _stat_file(root: Path, path: str) -> _Stat | _Unread:
    """Stage one on one cited file: is it there, is it a file, and can we open it?

    Returns the mtime/size pair, or an `_Unread` naming why there is none. The
    disposition type is imported from `store` rather than decided again here, because
    two verifiers each choosing their own answer for the same repo state is exactly the
    disagreement this module is measured against.

    Three things are settled here rather than downstream, and each is a defect fixed:

    * **Absent vs unopenable.** The old code caught every `OSError` and returned
      `None`, which the caller expired as `file_missing`. `stat()` fails with `EACCES`
      when a parent directory is unsearchable and with `ESTALE`/`ENOTCONN` off a dead
      NFS mount, neither of which means anything was deleted -- and expiring is
      irreversible while the condition is not.
    * **Not a regular file.** `S_ISREG` on the `st_mode` we already have, so it costs
      nothing. A FIFO stats as a zero-byte file, which used to be caught downstream by
      `st.size < span.byte_end` and reported as `span_truncated` -- an accident of
      check ordering that named the wrong finding and expired the claim for it.
    * **Readability.** `stat()` succeeds on a `chmod 000` file and reports the same
      mtime and size it always did, so the fast path matched its baseline and served
      the claim as fresh, while the reference verifier -- which must actually open the
      file -- withheld it. Same index, same second, two verdicts; and
      `serve_assertions(force_hash=True)` disagreed with `serve_assertions()`. An
      accelerator that reaches conclusions the authority would not reach is not
      accelerating anything. One `faccessat` per distinct cited file restores it.

    `os.access` answers with the real uid/gid rather than the effective one and can be
    defeated by ACLs or an NFS server that disagrees with its client, so this narrows
    the window rather than closing it; `force_hash=True` is the exact answer and always
    was. Stated rather than quietly carried.
    """
    try:
        st = (root / path).stat()
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except OSError:
        return _UNREADABLE
    if not stat_module.S_ISREG(st.st_mode):
        return _UNREADABLE
    if not os.access(root / path, os.R_OK):
        return _UNREADABLE
    return _Stat(mtime_ns=st.st_mtime_ns, size=st.st_size)


def _read_file(root: Path, path: str) -> bytes | _Unread:
    """Read one cited file whole. An `_Unread` naming why, if it cannot be read.

    A module-level function rather than an inline `read_bytes()` on purpose: this is
    the single seam through which stage two touches the disk, which is what lets a
    test assert that the fast path did NOT read -- the claim "no read happened" is
    otherwise untestable, and an untestable performance claim is just a comment.

    The `is_file` test is not redundant with the `except OSError`. Catching OSError
    handles every way a read fails loudly; a FIFO fails quietly, by blocking inside
    `read_bytes` until some other process opens the write end. Nothing raises, nothing
    is logged, and the pass simply stops -- and this runs on the serve path, where the
    single-threaded MCP server has no other thread to notice. `is_file()` is False for
    a FIFO, a directory, a socket and a device node, and True for a regular file or a
    symlink to one, so one test covers the whole class. Same defence, and the same
    reasoning, as `store._read_source`; duplicated rather than imported, because a
    private cross-package helper is a worse coupling than four lines said twice.

    It does not close the window between the test and the read: a regular file
    swapped for a FIFO in between is still opened blocking. Only an fd-based open
    (`os.open` with `O_NONBLOCK`, then `fstat`) closes that, which is more machinery
    than this seam earns. The exposure narrows from "a pipe committed in the repo" to
    "a race won against this function", and that residual is stated rather than
    quietly carried.

    `_stat_file` now settles the same question one syscall earlier, so in practice this
    branch fires only on a file that turned into a pipe between the stat and the read.
    Kept anyway: this is the seam that actually opens the file, and a guard that only
    holds because some other function ran first is a guard that disappears the day the
    call order changes.

    The disposition split matters as much here as at the stat. `_ABSENT` means the file
    was deleted between the two calls and the claim genuinely expires; `_UNREADABLE`
    means this reader could not open what is there, and the claim is withheld with its
    status untouched. Returning one `None` for both is what let a permission bit
    permanently destroy a claim -- see `store._read_source`.
    """
    target = root / path
    try:
        st = target.stat()
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except OSError:
        return _UNREADABLE
    if not stat_module.S_ISREG(st.st_mode):
        return _UNREADABLE
    try:
        return target.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except OSError:
        return _UNREADABLE


def _now(conn: sqlite3.Connection) -> str:
    """The current time, from SQLite's clock -- the one that stamps the rows."""
    return str(conn.execute(_NOW_SQL).fetchone()[0])


def _chunks(items: Sequence[int], size: int = _BATCH) -> Iterator[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _load_baselines(
    conn: sqlite3.Connection, span_ids: Sequence[int]
) -> dict[int, _Baseline]:
    """Fetch the stat baselines for these spans. Missing means never hash-verified."""
    out: dict[int, _Baseline] = {}
    for batch in _chunks(span_ids):
        placeholders = ",".join("?" * len(batch))
        for row in conn.execute(
            "SELECT span_id, mtime_ns, size_bytes, verified_hash, verified_at "  # noqa: S608
            f"FROM span_verifications WHERE span_id IN ({placeholders})",
            tuple(batch),
        ):
            out[int(row["span_id"])] = _Baseline(
                mtime_ns=int(row["mtime_ns"]),
                size_bytes=int(row["size_bytes"]),
                verified_hash=str(row["verified_hash"]),
                verified_at=str(row["verified_at"]),
            )
    return out


class _Pass:
    """One verification pass: caches disk access, counts it, and buffers its writes.

    Writes are buffered rather than issued as they are found so that a pass that
    changes nothing takes no write lock at all. That is what makes "the steady state
    is one stat per file and zero reads" true of the database as well as the
    filesystem -- a read path that opens a transaction on every query is not a read
    path.
    """

    def __init__(self, root: Path, *, force_hash: bool = False) -> None:
        self.root = root
        self.force_hash = force_hash
        self.files_statted = 0
        self.files_read = 0
        self.spans_fast_pathed = 0
        self.spans_hashed = 0
        self._stats: dict[str, _Stat | _Unread] = {}
        self._sources: dict[str, bytes | _Unread] = {}
        self._baseline_writes: list[tuple[int, int, int, str, str]] = []
        # Distinct files this pass could not open. A set rather than a counter so a
        # module cited by forty claims counts once -- the operator's question is "which
        # file do I fix", and a count that scales with citations does not answer it.
        self.unreadable_files: set[str] = set()

    def stat(self, path: str) -> _Stat | _Unread:
        if path not in self._stats:
            result = _stat_file(self.root, path)
            self._stats[path] = result
            self.files_statted += 1
            if result is _UNREADABLE:
                self.unreadable_files.add(path)
        return self._stats[path]

    def source(self, path: str) -> bytes | _Unread:
        if path not in self._sources:
            result = _read_file(self.root, path)
            self._sources[path] = result
            self.files_read += 1
            if result is _UNREADABLE:
                self.unreadable_files.add(path)
        return self._sources[path]

    def check_span(self, span: EvidenceSpan, baseline: _Baseline | None, now: str) -> SpanCheck:
        """Two-stage check of one citation.

        Ordering matches `store._first_failure` exactly -- unavailable (missing or
        unreadable), then truncated, then mismatched -- so the two verifiers cannot
        report different reasons for the same repo state. The unavailability test has
        to come FIRST and not merely early: a file that is both `chmod 000` and shorter
        than the cited range would otherwise be reported as `span_truncated` here, from
        a `st.size` this process was never entitled to act on, and expired for it,
        while the reference verifier withheld it unread.
        """
        seen_at = None if baseline is None else baseline.verified_at
        st = self.stat(span.path)
        if isinstance(st, _Unread):
            # `_ABSENT` -> file_missing, terminal. `_UNREADABLE` -> withheld. The
            # reason travels from where the errno was in scope; nothing re-decides it.
            return SpanCheck(
                span, METHOD_STAT, ok=False, verified_at=seen_at, reason=st.reason,
            )

        if (
            baseline is not None
            and not self.force_hash
            and baseline.mtime_ns == st.mtime_ns
            and baseline.size_bytes == st.size
            # The baseline must vouch for the hash the span cites NOW. A re-cited span
            # keeps its id, so without this a new hash could inherit an old witness.
            and baseline.verified_hash == span.content_hash
        ):
            self.spans_fast_pathed += 1
            return SpanCheck(span, METHOD_STAT, ok=True, verified_at=baseline.verified_at)

        # Stage two. `st.size` already settles truncation without a read: a file
        # shorter than the cited end cannot contain the cited range, whatever is in it.
        if st.size < span.byte_end:
            return SpanCheck(
                span, METHOD_STAT, ok=False, verified_at=seen_at,
                reason=REASON_SPAN_TRUNCATED,
            )
        source = self.source(span.path)
        if isinstance(source, _Unread):
            # Statted fine, would not read -- something landed between the two calls.
            # A delete reports `file_missing` and expires; a permission change or a
            # swap for a FIFO reports `unreadable` and withholds. Which of those it
            # was is decided at the seam that saw the errno, not guessed here.
            return SpanCheck(
                span, METHOD_HASH, ok=False, verified_at=seen_at, reason=source.reason,
            )
        if span.byte_end > len(source):
            return SpanCheck(
                span, METHOD_HASH, ok=False, verified_at=seen_at,
                reason=REASON_SPAN_TRUNCATED,
            )
        observed = content_hash(source[span.byte_start:span.byte_end])
        self.spans_hashed += 1
        if observed != span.content_hash:
            return SpanCheck(
                span, METHOD_HASH, ok=False, verified_at=seen_at,
                reason=REASON_HASH_MISMATCH, observed_hash=observed,
            )
        if span.id is not None:
            # Re-baseline on success, which is what turns "somebody touched the file"
            # into a one-off cost instead of a permanent one.
            self._baseline_writes.append((span.id, st.mtime_ns, st.size, observed, now))
        return SpanCheck(
            span, METHOD_HASH, ok=True, verified_at=now, observed_hash=observed
        )

    def flush(self, conn: sqlite3.Connection) -> int:
        """Persist the new stat baselines. Returns how many rows were written."""
        if not self._baseline_writes:
            return 0
        with _atomic(conn):
            conn.executemany(
                "INSERT INTO span_verifications "
                " (span_id, mtime_ns, size_bytes, verified_hash, verified_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(span_id) DO UPDATE SET mtime_ns = excluded.mtime_ns, "
                " size_bytes = excluded.size_bytes, "
                " verified_hash = excluded.verified_hash, "
                " verified_at = excluded.verified_at",
                self._baseline_writes,
            )
        written = len(self._baseline_writes)
        self._baseline_writes.clear()
        return written


def _check(
    pass_: _Pass, assertion: Assertion, baselines: Mapping[int, _Baseline], now: str
) -> ServedAssertion:
    """Verify one assertion's citations. Pure: decides, does not write.

    Stops at the first failing citation, like `store._first_failure` and for the same
    reason -- one moved span already expires the claim, and WHICH one moved is the
    actionable fact. Stopping early also means a claim about a deleted file costs one
    `stat()`, not a read of every other file it happens to cite.
    """
    if not assertion.spans:
        # Vacuous truth guard, kept identical to the one in `store`. "Every cited span
        # still matches" is trivially TRUE of no spans, so an assertion that lost its
        # evidence would otherwise be promoted by the same code that verifies a
        # well-cited one. Nothing was checked, so no method is reported.
        return ServedAssertion(
            assertion=assertion, stale=True, checked_at=now, verified_at=None,
            method=None, reason=REASON_NO_EVIDENCE,
        )

    checks: list[SpanCheck] = []
    for span in assertion.spans:
        baseline = baselines.get(span.id) if span.id is not None else None
        check = pass_.check_span(span, baseline, now)
        checks.append(check)
        if not check.ok:
            # The one branch that decides whether a status is written. `unreadable`
            # says the check did not happen, so `stale` stays False and `_expire`
            # never sees this record -- the claim is withheld and comes back on its
            # own. Everything else is a finding about the repository and expires.
            could_not_check = check.reason == REASON_UNREADABLE
            return ServedAssertion(
                assertion=assertion, stale=not could_not_check, checked_at=now,
                # `method` names the stage that CONFIRMED something, and on an
                # unreadable claim no stage did -- the stat only established that it
                # could not look. Reporting `method='stat'` here would put the exact
                # string a fast-path confirmation carries onto a claim nothing
                # confirmed, which is the misreading this whole split exists to
                # prevent. `verified_at` is kept, and is the useful half: it is when
                # these bytes were last genuinely hashed, so a caller sees "last
                # confirmed on Tuesday, cannot look today" rather than a bare refusal.
                verified_at=check.verified_at, method=None if could_not_check else check.method,
                checks=tuple(checks), reason=check.reason,
                failing_span=check.span, observed_hash=check.observed_hash,
                unreadable=could_not_check,
            )

    seen = [c.verified_at for c in checks if c.verified_at is not None]
    return ServedAssertion(
        assertion=assertion,
        stale=False,
        checked_at=now,
        # The OLDEST confirmation, not the newest: a claim citing two files is only as
        # fresh as the staler of them, and reporting the newest would flatter it.
        verified_at=min(seen) if seen else None,
        # Weakest link. One span that only got a stat makes the whole claim
        # stat-confirmed.
        method=METHOD_STAT if any(c.method == METHOD_STAT for c in checks) else METHOD_HASH,
        checks=tuple(checks),
    )


def _span_ids(assertions: Iterable[Assertion]) -> list[int]:
    return [s.id for a in assertions for s in a.spans if s.id is not None]


def _expire(
    conn: sqlite3.Connection, results: Sequence[ServedAssertion]
) -> list[ServedAssertion]:
    """Write the status flips and their log rows as one batch, and restate the rows.

    Detection and demotion stay in the same operation (the caller invokes this before
    returning anything), because splitting them leaves a window in which the engine
    knows a claim is wrong and is still willing to hand it to the next caller.

    The returned records carry `status = 'stale'` on the assertion itself, not just the
    `stale` flag beside it. Two fields disagreeing about the same fact is how a caller
    reads the wrong one.

    Each expiry is logged with the SAME instant this pass reports as `checked_at`, so
    one expiry carries one timestamp -- see the note at the `detected_at` argument
    below.

    Records carrying `unreadable=True` are not in `failed` and never reach this
    function's writes. That is the entire point of keeping `unreadable` off `stale`:
    the only place a status is written is here, and a claim nobody could check must
    not pass through it.
    """
    failed = [r for r in results if r.stale]
    if not failed:
        return list(results)
    with _atomic(conn):
        for result in failed:
            mark_stale(
                conn,
                result.assertion.id,
                result.reason or REASON_HASH_MISMATCH,
                span_id=None if result.failing_span is None else result.failing_span.id,
                expected_hash=(
                    None if result.failing_span is None else result.failing_span.content_hash
                ),
                observed_hash=result.observed_hash,
                # The SAME instant this pass reports as `checked_at`. Letting the
                # column default fire its own read makes one expiry carry two
                # timestamps a millisecond apart, so a later call that faithfully
                # reports the logged `detected_at` contradicts what the detecting
                # call returned.
                detected_at=result.checked_at,
            )
    return [
        replace(r, assertion=replace(r.assertion, status=STATUS_STALE))
        if r.stale and r.assertion.status == STATUS_ACTIVE
        else r
        for r in results
    ]


def refresh_staleness(
    conn: sqlite3.Connection,
    repo_root: db.StrPath | None = None,
    *,
    force_hash: bool = False,
) -> RefreshReport:
    """Sweep every active assertion, expire the ones whose evidence moved, report counts.

    The sweep is NOT the mechanism that keeps the index honest -- `serve_assertions`
    verifies on the read path, so a claim cannot be served between an edit and the next
    sweep. This exists for the two things a read path cannot do: it finds claims that
    went stale and were never asked for again (which is most of them, and they are
    exactly the ones a "how fast does this repo invalidate its own inferences" number
    has to include), and it gives an operator one command whose output says whether the
    fast path is earning its keep.

    `force_hash=True` skips stage one entirely and re-hashes every cited span. That is
    the deliberate way to close the mtime hole after something that rewrites timestamps
    without changing content, or before trusting the index for anything expensive.

    Verification runs first and writes happen once at the end, so a sweep over an
    unchanged repo takes no write lock.

    Claims withheld because a cited file could not be READ are counted in
    `unverifiable` and in neither `fresh` nor `expired`. Folding them into `fresh`
    would make this report -- the only evidence anyone has that the fast path works --
    assert a check it did not perform; folding them into `expired` would put a reason
    in `by_reason` for a claim whose status never moved. `checked` is the sum of all
    three, exactly.
    """
    root = _repo_root(conn, repo_root)
    assertions = _load_assertions(conn, "status = ?", (STATUS_ACTIVE,))
    now = _now(conn)
    baselines = _load_baselines(conn, _span_ids(assertions))
    pass_ = _Pass(root, force_hash=force_hash)

    results = [_check(pass_, a, baselines, now) for a in assertions]
    pass_.flush(conn)
    results = _expire(conn, results)

    by_reason: dict[str, int] = {}
    for result in results:
        if result.stale:
            key = result.reason or "unknown"
            by_reason[key] = by_reason.get(key, 0) + 1
    return RefreshReport(
        checked=len(results),
        # `not stale AND not unreadable`. A claim nobody could open is not fresh, and
        # the old `not r.stale` would now count it as such.
        fresh=sum(1 for r in results if not r.stale and not r.unreadable),
        expired=sum(1 for r in results if r.stale),
        by_reason=by_reason,
        files_statted=pass_.files_statted,
        files_read=pass_.files_read,
        spans_fast_pathed=pass_.spans_fast_pathed,
        spans_hashed=pass_.spans_hashed,
        unverifiable=sum(1 for r in results if r.unreadable),
        files_unreadable=len(pass_.unreadable_files),
    )


def _latest_reasons(
    conn: sqlite3.Connection, assertion_ids: Sequence[int]
) -> dict[int, tuple[str, str]]:
    """The most recent `(reason, detected_at)` per assertion, from `staleness_log`.

    Used only to LABEL claims that were already stale before this call. They are not
    re-verified -- re-checking a stale claim would let a coincidence (a revert, a
    rename back) quietly resurrect a claim that nothing has re-adjudicated.
    """
    out: dict[int, tuple[str, str]] = {}
    for batch in _chunks(assertion_ids):
        placeholders = ",".join("?" * len(batch))
        for row in conn.execute(
            "SELECT assertion_id, reason, detected_at FROM staleness_log "  # noqa: S608
            f"WHERE assertion_id IN ({placeholders}) ORDER BY id",
            tuple(batch),
        ):
            out[int(row["assertion_id"])] = (str(row["reason"]), str(row["detected_at"]))
    return out


def serve_assertions(
    conn: sqlite3.Connection,
    repo_root: db.StrPath | None = None,
    *,
    subject_qualname: str | None = None,
    kind: str | None = None,
    include_stale: bool = False,
    include_unverifiable: bool = False,
    force_hash: bool = False,
) -> list[ServedAssertion]:
    """Serve claims, verified two-stage, each carrying the provenance of its check.

    Default: stale claims are WITHHELD. Not marked, not de-ranked -- withheld, because
    a caller that did not ask about staleness is a caller that will not read the flag,
    and a wrong claim about code is worse than no claim about code.

    `include_stale=True` returns them alongside the fresh ones, every one carrying
    `stale=True`, the reason it expired, and the citation that moved. Anything expiring
    during THIS call is expired first and then labelled, so the flag and the stored
    status never disagree.

    Claims whose citations could not be READ are withheld too, and by DEFAULT they are
    withheld from `include_stale=True` as well. That is the more important of the two
    decisions here. `include_stale` promises a caller expired claims -- code that
    handles the result reasonably reads `if r.stale: ... else: <treat as fresh>`, and
    an unreadable claim arriving in that stream with `stale=False` lands in the `else`
    and is presented as verified. It would be the cached-freshness-verdict failure
    delivered through the very flag added to prevent it. So there is a second, separate
    opt-in: `include_unverifiable=True`. A caller cannot receive one of these records
    without having typed the word, and having typed it has demonstrated it knows the
    state exists. They arrive with `unreadable=True`, `stale=False`, `method=None`, and
    a `label` reading UNVERIFIED.

    `force_hash=True` bypasses the stat fast path for this call.
    """
    root = _repo_root(conn, repo_root)
    where = ["status = ?"]
    params: list[object] = [STATUS_ACTIVE]
    if subject_qualname is not None:
        where.append("subject_qualname = ?")
        params.append(subject_qualname)
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    predicate = " AND ".join(where)

    active = _load_assertions(conn, predicate, params)
    now = _now(conn)
    baselines = _load_baselines(conn, _span_ids(active))
    pass_ = _Pass(root, force_hash=force_hash)
    results = [_check(pass_, a, baselines, now) for a in active]
    pass_.flush(conn)
    results = _expire(conn, results)

    # Three buckets, two of them opt-in. `active` was loaded ordered by id and this
    # preserves that order, so only the `include_stale` branch below needs to re-sort.
    results = [
        r
        for r in results
        if (not r.stale and not r.unreadable)
        or (r.stale and include_stale)
        or (r.unreadable and include_unverifiable)
    ]
    if not include_stale:
        return results

    # Claims that were already stale before this call. Loaded with the same predicate
    # so `subject_qualname`/`kind` mean the same thing whichever set a claim is in.
    prior = _load_assertions(conn, predicate, [STATUS_STALE, *params[1:]])
    reasons = _latest_reasons(conn, [a.id for a in prior])
    known = {r.assertion.id for r in results}
    for assertion in prior:
        if assertion.id in known:
            continue  # expired during this call; already in `results`, already labelled
        reason, detected_at = reasons.get(assertion.id, ("unknown", assertion.created_at))
        results.append(
            ServedAssertion(
                assertion=assertion,
                stale=True,
                # Not `now`: nothing was checked. The last thing established about this
                # claim is its expiry, and that is the timestamp a caller should see.
                checked_at=detected_at,
                verified_at=None,
                method=None,
                reason=reason,
            )
        )
    results.sort(key=lambda r: r.assertion.id)
    return results


def verification_state(conn: sqlite3.Connection, assertion_id: int) -> list[sqlite3.Row]:
    """The stat baselines behind one assertion's citations, for inspection.

    The engine's own state, readable. A fast path nobody can look at is a fast path
    nobody can catch lying.
    """
    return list(
        conn.execute(
            "SELECT e.id AS span_id, e.path, e.line_start, e.line_end, "
            "       e.content_hash AS cited_hash, v.mtime_ns, v.size_bytes, "
            "       v.verified_hash, v.verified_at "
            "FROM evidence_spans e LEFT JOIN span_verifications v ON v.span_id = e.id "
            "WHERE e.assertion_id = ? ORDER BY e.id",
            (assertion_id,),
        )
    )
