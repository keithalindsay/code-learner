"""The tier-2 store: admit a claim only with citations, serve it only while they hold.

This module is the gate, not the pipeline. Nothing here calls a model. What it does
is refuse to let an inferred claim into the index on terms that would make it
unaccountable later, and refuse to hand one back once the ground under it has moved.

Three rules, and every one of them exists because the failure it prevents is silent:

**1. Nothing is admitted that a later reader could not check.** `write_assertion`
raises before it opens a transaction, so a refused claim leaves no row behind -- not
a row, not an id, not a gap in the sequence that would need explaining. Six
conditions are refused there and they are all the same rule wearing different
clothes: no spans, no claim text, a span that is not a non-empty byte range, a span
with no hash to check it against, a subject this index has never parsed, and a
citation that does not match the bytes on disk right now. Each one produces something
that is *indistinguishable from a good claim at every later stage* -- it stores, it
serves, and it verifies forever -- which is why none of them can be left to the
caller. `write_assertion` is the one door, and every lock is on it.

That was not true until recently, and the way it failed is the argument for the
shape. Hash verification and subject-existence lived in `server/app.py`, so an
agent reaching the store through the MCP tool met three rules and `codelearner learn`
-- or any library caller -- met one. A claim with a perfectly valid citation and an
empty claim string was admitted, stored `active`, and served. A zero-length span was
admitted and verified forever against any content the file later held, while
`span_for` refused exactly that, with exactly the right reasoning, in a constructor
the gate did not require anyone to use. A rule enforced in the caller is a rule the
next caller does not have.

**2. Servable means re-verified, not merely stored.** `servable_assertions` re-reads
the cited bytes off disk and re-hashes them on every call. `status = 'active'` alone
is never enough. The check is on the read path rather than in a background job for
one reason: a sweep that runs hourly has an hour-wide window in which the index
serves claims about code that no longer exists, and nobody would ever see it happen.
Verification at serve time has no window.

**3. Nothing is deleted.** A refuted claim becomes `status = 'rejected'` and keeps
its spans and its verdict. An expired one becomes `'stale'` and gets a row in
`staleness_log` naming the citation that moved. Rejections are the only evidence
that the gate does anything -- a pipeline that deletes what it rejected can report
any pass rate it likes, and cannot tell a generator that improved from a judge that
got lazier.

**4. "We could not look" is not "the evidence changed".** Verification has three
outcomes, not two. A citation can hold, it can be contradicted by the bytes on disk,
or it can be *unavailable for inspection* -- and only the second one is a fact about
the repository. `_read_source` separates real absence (`FileNotFoundError`,
`NotADirectoryError`) from every other way a read fails: a permission bit, `EMFILE`,
`EIO`, an NFS blip, a repo on an unmounted volume, or a path that is no longer a
regular file. Absence expires the claim. Everything else withholds it for this call
and *leaves the stored status alone*.

That split is the whole of WP10, and the reason is that the old behaviour was
irreversible in one direction. `chmod 000 leases.py` expired every claim citing it,
logged `file_missing` for a file that was sitting right there, and `chmod 644` did
not bring any of them back, because nothing in this module ever moved an assertion
towards `active`. One transient failure -- a lock on a mounted volume, a moved
checkout, a file descriptor limit hit under load -- permanently converted "we could
not look" into "the evidence changed", which is the exact distinction the whole tier
exists to preserve. Failing closed is right. Failing closed *permanently*, on
evidence that never moved, is data loss with a reassuring log line.

`reinstate` is the way back, and it is evidence-driven rather than a flag: it
re-hashes every cited span and promotes `stale` to `active` only if all of them match
exactly. There is deliberately no override -- an operator who could assert freshness
by hand would be the cached freshness verdict this project refuses to build.

The one non-obvious guard is the empty evidence set. "Every cited span still
matches" is *trivially true of no spans*, so an assertion that somehow lost all of
its evidence would be promoted to servable by the exact same code path that verifies
a well-cited one. Rule 1 should make that unreachable; `servable_assertions` checks
for it anyway and logs it as `no_evidence`, because a vacuous truth reads as success
everywhere it is not specifically looked for -- and `reinstate` refuses an
evidence-free claim for the same reason, one door further on.
"""
from __future__ import annotations

import sqlite3
import stat as stat_module
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .. import db
from ..ingest.types import content_hash

# Assertion lifecycle. See schema.sql for what each one promises; the CHECK
# constraint there is the enforcement, these are just the names for it.
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"
STATUS_STALE = "stale"

# Adjudication outcomes. Anything that is not `supported` stops a claim being
# served: "the evidence is silent" and "the evidence says otherwise" are different
# problems, but neither is a reason to keep answering questions with the claim.
VERDICT_SUPPORTED = "supported"
VERDICT_REFUTED = "refuted"
VERDICT_UNSUPPORTED = "unsupported"

# Why a citation stopped verifying. Recorded rather than collapsed into one flag,
# because "the function was edited" and "the file is gone" call for different
# repairs and only one of them is routine.
#
# The first four are TERMINAL: they are findings about the repository, they expire the
# claim, and they are written to `staleness_log`.
REASON_HASH_MISMATCH = "hash_mismatch"
REASON_FILE_MISSING = "file_missing"
REASON_SPAN_TRUNCATED = "span_truncated"
REASON_NO_EVIDENCE = "no_evidence"

# The fifth is NOT terminal, and the difference is the point. `unreadable` is a fact
# about this process at this instant -- a permission bit, `EMFILE`, `EIO`, an
# unmounted volume, a FIFO where a module used to be -- and says nothing whatever
# about whether the cited bytes still match. It withholds the claim for this call and
# leaves `status` exactly as it found it, so the next call over a healthy filesystem
# serves the claim again with no operator action at all.
#
# It is therefore never written to `staleness_log`: that table is one row per EXPIRY,
# and its growth rate is documented as a real signal about how fast this repo
# invalidates its own inferences. Logging non-expiries there would turn a measurement
# into a count of how often the disk was busy. `unreadable` is reported in the return
# value instead -- `stale.SpanCheck.reason`, `ServedAssertion.unreadable`, and the
# `unverifiable` count in `RefreshReport` -- which is where a caller can act on it.
REASON_UNREADABLE = "unreadable"

# The sixth, and terminal, and the only one that is not a finding about the bytes.
# `decorators_excluded` says the cited bytes are exactly what they always were and the
# CITATION BOUNDARY is wrong: the span stops short of the symbol's decorators, so it
# ends where the symbol ends and begins at its `def`/`class` -- the span pre-v6 code
# wrote, before WP8 widened a decorated symbol to start at its outermost `@`.
#
# It needs its own name because reading it as `hash_mismatch` would send an operator
# looking for an edit that never happened, and because the repair is different: no
# re-index can fix it. Re-indexing rebuilds the symbol table, which is already right;
# the claim itself was derived from bytes that excluded `@app.post("/intent",
# dependencies=[Depends(require_token)])`, and only re-deriving the claim can fix
# that. Everything else in this list is repaired by the repository settling down.
#
# The exposure it closes is WP8's, surviving inside carried data. Strip a decorator
# that a narrowed citation never covered and every verifier in this package still
# reports the claim fresh, forever -- which is the fail-open case WP8 existed to
# remove, and which `--carry-assertions` walks straight past because the bytes on disk
# genuinely did not change.
REASON_DECORATORS_EXCLUDED = "decorators_excluded"

# Status transitions are timestamped by SQLite, not by Python. `created_at` in the
# schema already uses this expression, and two clocks stamping rows in one table is
# a way for "created after it went stale" to happen in the data.
#
# The trailing `AND status = ?` is not decoration. It is the only thing that makes a
# transition safe against a concurrent one: `reinstate` checks the evidence and then
# promotes, and between those two steps another connection may have rejected the
# claim. Naming the expected FROM-state in the WHERE clause means the promotion simply
# does not fire, rather than overwriting an adjudication that landed first.
_TOUCH_STATUS = (
    "UPDATE assertions "
    "SET status = ?, status_changed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
    "WHERE id = ? AND status = ?"
)

# SQLite refuses a statement carrying more than SQLITE_MAX_VARIABLE_NUMBER bound
# parameters -- 32,766 on every build this project runs against. A variable-length
# `IN (...)` binds one variable per element, so any query built that way has a size at
# which it stops working, and the size is "a repo with enough claims in it". Batching
# is the fix, and the number only has to be comfortably under the limit; 500 keeps the
# round-trip count sane without going near it.
#
# Defined HERE rather than in `stale`, which had its own copy: `stale` batches its
# `span_verifications` and `staleness_log` lookups against this hazard and then called
# `_load_assertions`, which did not batch -- so a sweep large enough to need the
# batching raised inside the one query that had none. One constant, one chunker, both
# imported by the module that reads them.
_BATCH = 500


def _chunks(items: Sequence[object], size: int = _BATCH) -> Iterator[Sequence[object]]:
    """`items` in slices of at most `size`, so an `IN (...)` can never overflow."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


class EvidenceRequired(ValueError):
    """An assertion was submitted with no evidence spans, and was not written.

    Raised by `write_assertion` before any row exists. Deliberately not a status: an
    uncited claim is not a claim that failed adjudication, it is one that was never
    admissible, and storing it would put an unciteable row in the same table as the
    citeable ones where a later bug could flip it active."""


# The five siblings below are the rules that used to live in `server/app.py`, or in
# nobody at all. Each is its own class rather than one `Inadmissible`, because the
# caller that has to tell an agent what to fix cannot do it from a single type -- and
# collapsing them would also collapse the refusal codes, which is how the negative
# controls in `eval.gate_controls` tell "refused by its own rule" from "refused".


class EmptyClaim(ValueError):
    """The claim text was empty or whitespace, and the assertion was not written.

    Verified evidence carrying no statement. Every arithmetic check the gate makes
    passes on this submission -- the spans exist, they hash correctly, the subject is
    real -- so it stores `active`, is reported servable, and is handed back next to
    the code it is allegedly about, saying nothing. A judge asked to adjudicate it has
    no proposition to adjudicate; a human following the citation finds correct bytes
    and no reason they were cited. Absent this rule an empty claim is
    indistinguishable from a good one at every stage, including the ones designed to
    catch bad ones."""


class InvalidSpan(ValueError):
    """A cited byte range was empty, negative, or inverted; nothing was written.

    The dangerous case is `byte_start == byte_end`. sha256 of nothing is a perfectly
    stable hash, so a zero-length citation VERIFIES FOREVER -- against the file as it
    is, as it becomes, and as it would be after the symbol it pointed at was deleted.
    It is not merely unfalsifiable; it is a claim that positively reports `fresh` on
    every re-read, which reads as the strongest possible evidence that the claim still
    holds. `span_for` has always refused this, with the right reasoning, but a
    constructor is only a rule for callers who use it."""


class SpanEscapesRepo(ValueError):
    """A cited path leaves the indexed repository; nothing was written.

    Found by running the adversarial corpus against `write_assertion` rather than only
    against the MCP tool, which is the whole reason WP5 exists: `path_escapes_repo`
    lived solely in `server.app._verify_span`, so the door the MCP server uses refused
    this attack 100% of the time while `codelearner learn` and every library caller
    admitted it. Reproduced before the fix -- a claim citing `../outside_secret.py`
    with that file's real hash was stored `active` and reported `servable`.

    It is the nastiest shape this store can hold, because nothing downstream looks
    wrong. The bytes exist, the hash is correct, the claim re-verifies on every serve,
    and `refresh_staleness` will report it fresh for as long as the file is untouched.
    The only thing wrong with it is that a reader of THIS repository cannot open the
    thing it cites -- and that is invisible to every check that runs after admission.
    It also quietly breaks the isolation property the README calls structurally
    impossible to violate: one repo's index ends up describing another's bytes.

    The check here is lexical -- absolute paths and any `..` component -- and is
    deliberately weaker than `_verify_span`'s, which resolves the path and compares it
    to the root. Lexical is what can be enforced at the door: `verify=False` is a
    supported call, and a rule that only holds when a caller opted into re-reading is
    not a rule the gate enforces. A symlink pointing out of the tree still gets past
    this and is caught downstream at verification time; that gap is stated rather than
    papered over."""


class EvidenceUnverifiable(ValueError):
    """A span carried nothing to check it against, or nothing to check it from.

    Two shapes, one failure. A span with no `content_hash` asserts nothing about what
    is at those bytes, so no future state of the file can contradict it -- the
    vacuous-truth failure that `servable_assertions` guards against for a whole
    assertion, one level down at the span. And a write asked to verify against a repo
    root the index is not bound to has no bytes it is entitled to read, which is the
    same hole approached from the other side: verification that cannot happen must not
    be reported as verification that passed."""


class UnknownSubject(ValueError):
    """The subject qualname names no symbol in this index; nothing was written.

    An inference no reader can reach. Every span can hash-match perfectly while the
    qualname they are attached to was invented, and the result is a row that is
    `active`, servable, and permanently unreachable, because `get_symbol` answers
    `no_such_symbol` for the only name that would find it. Indistinguishable from a
    good claim by inspection -- the evidence is genuinely correct -- and
    distinguishable from one only by the fact that nothing will ever ask for it."""


class EvidenceStale(ValueError):
    """A cited span did not match the bytes on disk at admission time.

    The claim was never true of this repository, or stopped being true between being
    drafted and being stored. Admitting it would put a row in the store whose first
    verification is guaranteed to fail, and the store's own vocabulary would then
    record that as `stale` -- as though the repository had moved under a claim that
    was once good. The distinction is worth keeping: `stale` means the world changed,
    and a claim that never matched anything is a defect in whatever produced it."""


class NotReinstatable(ValueError):
    """`reinstate` was asked to revive a claim a judge refused; nothing was changed.

    `rejected` and `stale` look adjacent -- both are "not being served" -- and they are
    opposites. `stale` means the evidence moved out from under a claim, so re-reading
    the evidence is exactly the right way to settle whether it should come back.
    `rejected` means a judge read the claim against evidence that was correct at the
    time and said it is not supported; the bytes were never the disputed part. A
    re-hash has nothing to say about that verdict, so promoting on one would let a
    `git checkout` overturn an adjudication -- and the promoted row would then be
    indistinguishable from a claim that was never refuted, because `verdicts` is a
    separate table nothing on the serve path reads.

    Raised rather than returned as `False` deliberately. `False` is this function's
    word for "the evidence still does not match", which is routine, expected, and
    unalarming. Asking to reinstate a refuted claim is none of those things, and the
    two must not arrive at the caller looking the same."""


@dataclass(frozen=True)
class EvidenceSpan:
    """A concrete `file:line` citation plus the hash of exactly those bytes.

    `content_hash` is the hash of `source[byte_start:byte_end]` as it read when the
    claim was made -- not of the whole file. That is what lets an unrelated edit
    elsewhere in the same module leave this claim alone, which is the difference
    between staleness that gets acted on and staleness that gets ignored.
    """

    path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    content_hash: str
    id: int | None = None  # None until stored

    @property
    def citation(self) -> str:
        """`path:start-end`, the form a human is expected to go look at."""
        return f"{self.path}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class Assertion:
    """A stored tier-2 claim, carrying the citations it was admitted on.

    The spans travel with the claim rather than being fetchable separately, because
    every caller that shows a claim should be able to show what it rests on without
    a second query it might forget to make.
    """

    id: int
    subject_qualname: str
    subject_symbol_id: int | None
    kind: str
    claim: str
    status: str
    generator: str | None
    confidence: float | None
    created_at: str
    spans: tuple[EvidenceSpan, ...] = ()


@contextmanager
def _atomic(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write batch atomically, joining the caller's transaction if one is open.

    `db.transaction` refuses to nest, for a good reason that does not apply here:
    these writers are called both from inside a caller's batch (a pipeline admitting
    forty assertions as one unit) and from outside it (a read path expiring a single
    claim it just found). Joining an open transaction keeps both correct without
    making every caller ask which situation it is in.
    """
    if conn.in_transaction:
        yield conn
        return
    with db.transaction(conn) as joined:
        yield joined


def _sync_assertion_document(conn: sqlite3.Connection, assertion_id: int) -> None:
    """Load the derived-index dependency only at the mutation boundary."""
    from . import search_index

    search_index.sync_assertion_document(conn, assertion_id)


def _repo_root(conn: sqlite3.Connection, repo_root: db.StrPath | None) -> Path:
    """Resolve where the cited files live, preferring the index's own binding.

    Refuses to guess. Verification means re-reading the cited bytes, so serving
    without knowing the repo root would mean serving claims nothing checked -- the
    exact outcome this module exists to prevent.
    """
    if repo_root is not None:
        return Path(str(repo_root))
    stored = db.stored_repo_root(conn)
    if stored is None:
        raise ValueError(
            "this index is not bound to a repo root, so evidence spans cannot be "
            "re-read from disk. Pass repo_root explicitly, or call "
            "db.bind_repo_root first. Serving an assertion without verifying its "
            "citations is the one thing this store will not do."
        )
    return Path(stored)


def _escapes_repo(path: str) -> bool:
    """True if `path` cannot possibly name a file inside the repo, read lexically.

    Deliberately does no filesystem work. It runs at the door, where `verify=False` is
    a supported call and no root need have been resolved yet; a containment rule that
    only fires when the caller opted into re-reading is not a rule the gate enforces.

    Three shapes, and the third is the one that motivated it. An absolute path names
    somewhere else outright. A NUL byte is not a path at all -- it truncates inside
    the C library, so the name Python checked and the name the kernel opened differ.
    And any `..` component walks out of the tree, which is how a claim about
    `../outside_secret.py` came to be stored `active` and reported `servable`.

    `..` is rejected wherever it appears, not just in the lead. `a/../../b` escapes
    while never starting with one, and normalising first to find out would mean
    resolving a path this function has deliberately not touched the disk to resolve.
    A repo-relative citation has no legitimate reason to contain one in any position:
    every span this package builds comes from `span_for_symbol`, whose paths are
    already normalised relative to the root.
    """
    if not path or "\x00" in path:
        return True
    pure = PurePosixPath(path)
    return pure.is_absolute() or ".." in pure.parts or Path(path).is_absolute()


def _verification_root(conn: sqlite3.Connection, repo_root: db.StrPath | None) -> Path:
    """Resolve where `write_assertion` re-reads the cited bytes from. Stricter than
    `_repo_root`, deliberately.

    The read path prefers a caller's root and falls back to the index's binding. This
    one refuses a caller's root that DISAGREES with the binding, and the asymmetry is
    not an oversight. Admission is the moment a claim is bound to this index, and
    every later verification of it will use the stored root -- so a claim admitted
    against some other tree would be re-checked tomorrow against bytes it was never
    compared to. That failure is silent in the worst direction: the write reports
    verified, and the first serve reports `stale`, naming an edit nobody made.

    Refuses to guess when there is no binding and no argument, for the same reason
    `_repo_root` does: verification that cannot happen must not be reported as
    verification that passed. `verify=False` is the way to say that out loud.
    """
    stored = db.stored_repo_root(conn)
    if repo_root is None:
        if stored is None:
            raise EvidenceUnverifiable(
                "this index is not bound to a repo root, so the cited spans cannot be "
                "re-read and this claim cannot be verified at the door. Pass "
                "repo_root explicitly, call db.bind_repo_root first, or say "
                "verify=False and own that nothing checked these citations."
            )
        return Path(stored)
    given = Path(str(repo_root)).resolve()
    if stored is not None and str(given) != stored:
        raise EvidenceUnverifiable(
            f"asked to verify citations against {str(given)!r}, but this index is "
            f"bound to {stored!r}. Every later verification of this claim will use "
            "the bound root, so admitting it here would mean it was checked against "
            "one tree and will be re-checked against another -- which surfaces as an "
            "expiry blaming an edit nobody made."
        )
    return given


def span_for(repo_root: db.StrPath, path: str, byte_start: int, byte_end: int) -> EvidenceSpan:
    """Build a citation for `path[byte_start:byte_end]`, hashing it off disk now.

    Line numbers are derived from the byte range rather than accepted next to it. A
    citation whose lines and bytes disagree points a human at one place and the
    verifier at another, and nothing about it would ever look wrong.

    Anything that is not a regular file is refused before a byte is read, and the
    hazard is not the obvious one. A missing file raises promptly and a directory
    raises promptly; a FIFO does neither. `read_bytes` on a FIFO blocks until some
    other process opens the write end, which in the single-threaded MCP server means
    the call never returns, no exception is ever raised, no log line is written, and
    the process simply stops answering -- a repo-relative path is enough to wedge it.
    `is_file()` is the cheap test that separates "bytes exist" from "opening this
    will block".

    This is a library function and it raises `ValueError`, as the range check below
    does, rather than the server's `ToolError`: `store` has no transport to be polite
    towards and must not import one. `server.app._verify_span` runs the same check
    first and turns it into a structured `file_missing` refusal, so the server path
    never reaches this raise; it is here for every other caller, which is all of them
    except one.
    """
    target = Path(str(repo_root)) / path
    if not target.is_file():
        raise ValueError(
            f"{path!r} is not a regular file, so there are no bytes to cite. Reading "
            "a FIFO or a device node here would block this call until another "
            "process obliged, which is not a bound on anything."
        )
    source = target.read_bytes()
    if not 0 <= byte_start < byte_end <= len(source):
        raise ValueError(
            f"span {path}[{byte_start}:{byte_end}] is not a non-empty range inside "
            f"a {len(source)}-byte file. An empty or out-of-range citation would "
            "hash to something stable and verify forever while pointing at nothing."
        )
    line_start = source.count(b"\n", 0, byte_start) + 1
    # Count to byte_end - 1, not byte_end: a span that stops exactly on a newline
    # ends on the line that newline terminates, not on the one after it.
    line_end = source.count(b"\n", 0, byte_end - 1) + 1
    return EvidenceSpan(
        path=path,
        line_start=line_start,
        line_end=line_end,
        byte_start=byte_start,
        byte_end=byte_end,
        content_hash=content_hash(source[byte_start:byte_end]),
    )


def span_for_symbol(conn: sqlite3.Connection, symbol_id: int) -> EvidenceSpan:
    """Build a citation for an indexed symbol, from what the index already stored.

    The natural evidence unit for a claim about code is a symbol, and `symbols`
    already holds its exact byte span and the sha256 of exactly those bytes. Reusing
    them costs no disk read and, more importantly, guarantees the citation and the
    index agree about where the symbol is -- deriving it a second time from source
    would introduce a way for them to differ.
    """
    row = conn.execute(
        "SELECT f.path, s.line_start, s.line_end, s.byte_start, s.byte_end, "
        "       s.content_hash "
        "FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
        (symbol_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"no symbol with id {symbol_id} in this index")
    return EvidenceSpan(
        path=row["path"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        byte_start=row["byte_start"],
        byte_end=row["byte_end"],
        content_hash=row["content_hash"],
    )


def write_assertion(
    conn: sqlite3.Connection,
    *,
    subject_qualname: str,
    kind: str,
    claim: str,
    spans: Sequence[EvidenceSpan],
    subject_symbol_id: int | None = None,
    generator: str | None = None,
    confidence: float | None = None,
    status: str = STATUS_ACTIVE,
    repo_root: db.StrPath | None = None,
    verify: bool = True,
    allow_unindexed_subject: bool = False,
) -> int:
    """Admit one claim with its citations. Returns the new assertion id.

    THE door. Every admission rule this project has is enforced here and nowhere
    else that matters, and all of them raise BEFORE the transaction opens, so a
    refused claim leaves nothing behind -- not a row, not an id, not a gap in the
    sequence that would need explaining. `server/app.py` still runs richer versions of
    two of these checks first, and should: it can name the offending field, quote the
    bytes that are actually there, and accept either honest reading of a line range,
    none of which a library function with no transport can do. But it runs them as a
    PRE-check for message quality. Nothing downstream depends on it having run.

    In order, and cheapest first:

    1. `EvidenceRequired` -- no spans.
    2. `EmptyClaim` -- nothing said about them.
    3. `InvalidSpan` -- a span that is not a non-empty, non-negative byte range.
    4. `EvidenceUnverifiable` -- a span with no hash to check it against.
    5. `UnknownSubject` -- a subject this index has never parsed.
    6. `EvidenceStale` -- a citation that does not match the bytes on disk NOW.

    `verify=True` is the default and the whole point of (6): the same `_first_failure`
    that decides tomorrow whether this claim may still be served decides today whether
    it may be admitted, so the two can never disagree about what verification means.
    It costs one read per cited file. `verify=False` exists for a caller that has
    genuinely just hashed the bytes itself and knows the tree cannot have moved --
    it is a statement that nothing checked these citations, and it should be rare
    enough to be conspicuous.

    `allow_unindexed_subject=True` is the escape from (5), for fixtures that build an
    index-shaped database without an index in it. It is a parameter rather than a
    fallback -- a rule that turns itself off when the `symbols` table happens to be
    empty is a rule that turns itself off on a fresh index, which is exactly when the
    first claims get written.

    `status` is passed straight through to the CHECK constraint in the schema rather
    than being validated here first. That is on purpose: two validators disagreeing
    about the allowed set is a bug that only shows up as data, so there is one.
    """
    spans = tuple(spans)
    if not spans:
        raise EvidenceRequired(
            f"assertion about {subject_qualname!r} cites no evidence spans and was "
            "not written. An uncited claim cannot be adjudicated, cannot expire, "
            "and cannot be checked by a reader -- it is indistinguishable from a "
            "good one at every stage after this."
        )
    if not claim.strip():
        raise EmptyClaim(
            f"assertion about {subject_qualname!r} carries no claim text and was not "
            "written. Its citations may be perfect; there is still no proposition "
            "here for a judge to adjudicate or a reader to disagree with, and every "
            "check after this one would pass."
        )
    for span in spans:
        if not 0 <= span.byte_start < span.byte_end:
            raise InvalidSpan(
                f"span {span.path}[{span.byte_start}:{span.byte_end}] is not a "
                "non-empty byte range, and the assertion was not written. An empty "
                "range hashes to a stable value and would go on verifying against "
                "whatever the file becomes -- reporting fresh evidence for a claim "
                "that points at nothing."
            )
        if span.line_start < 1 or span.line_end < span.line_start:
            raise InvalidSpan(
                f"span {span.path}[{span.byte_start}:{span.byte_end}] cites lines "
                f"{span.line_start}-{span.line_end}, which is not a line range a "
                "reader can open. The bytes are what the verifier checks and the "
                "lines are what a human follows; a citation whose two halves "
                "disagree sends them to different places and looks wrong to "
                "neither."
            )
        if _escapes_repo(span.path):
            raise SpanEscapesRepo(
                f"span {span.citation} resolves outside the indexed repository, and "
                "the assertion was not written. This claim would have verified "
                "forever -- the bytes are real and their hash is correct -- while "
                "citing a file this index does not describe and a reader of this "
                "repository cannot open."
            )
        if not span.content_hash.strip():
            raise EvidenceUnverifiable(
                f"span {span.citation} carries no content hash, and the assertion "
                "was not written. A citation that asserts nothing about what is at "
                "those bytes can never be found to be wrong -- it is the vacuous "
                "truth `servable_assertions` guards against for a whole assertion, "
                "one level down at the span."
            )
    if not allow_unindexed_subject:
        known = conn.execute(
            "SELECT 1 FROM symbols WHERE qualname = ?", (subject_qualname,)
        ).fetchone()
        if known is None:
            raise UnknownSubject(
                f"no symbol named {subject_qualname!r} in this index, so the claim "
                "was not written. Verified spans do not make a claim about a symbol "
                "that does not exist accountable to anyone: the row would be active, "
                "servable, and unreachable, because the only name that would find it "
                "is one nothing can look up."
            )
    if verify:
        # The same function the serve path uses, on purpose. A second implementation
        # of "does this citation still hold" is a second answer waiting to differ
        # from the first, and the difference would show up as a claim that was
        # admitted as verified and expired as stale on its very first read.
        failure = _first_failure(_verification_root(conn, repo_root), spans, {})
        if failure is not None:
            reason, moved, observed = failure
            if reason == REASON_UNREADABLE:
                # Not `EvidenceStale`. Nothing was found to be wrong with this
                # citation -- it could not be checked at all, and admitting a claim
                # whose verification did not happen is the same lie as reporting a
                # freshness check that did not run. The caller's repair is to fix the
                # filesystem and submit again, not to re-derive the claim.
                raise EvidenceUnverifiable(
                    f"span {moved.citation} could not be read, so nothing verified "
                    "it and the assertion was not written. The file is present but "
                    "this process cannot open it -- a permission bit, a descriptor "
                    "limit, a dead mount, or something that is not a regular file. "
                    "This is not a finding about the citation: fix the read and "
                    "submit the same claim again."
                )
            raise EvidenceStale(
                f"span {moved.citation} does not match the bytes on disk "
                f"({reason}), and the assertion was not written. Cited "
                f"{moved.content_hash}, found {observed or 'nothing readable'}. This "
                "claim's first verification would have failed, and the store would "
                "have recorded that as the repository moving under a good claim "
                "rather than as a claim that never matched anything."
            )
    with _atomic(conn):
        cur = conn.execute(
            "INSERT INTO assertions (subject_qualname, subject_symbol_id, kind, "
            " claim, status, generator, confidence) "
            "VALUES (?,?,?,?,?,?,?) RETURNING id",
            (subject_qualname, subject_symbol_id, kind, claim, status, generator,
             confidence),
        )
        assertion_id = int(cur.fetchone()[0])
        conn.executemany(
            "INSERT INTO evidence_spans (assertion_id, path, line_start, line_end, "
            " byte_start, byte_end, content_hash) VALUES (?,?,?,?,?,?,?)",
            [
                (assertion_id, s.path, s.line_start, s.line_end, s.byte_start,
                 s.byte_end, s.content_hash)
                for s in spans
            ],
        )
        _sync_assertion_document(conn, assertion_id)
    return assertion_id


def record_verdict(
    conn: sqlite3.Connection,
    assertion_id: int,
    judge: str,
    verdict: str,
    rationale: str | None = None,
) -> int:
    """Record an adjudication, and reject the claim if the judge did not support it.

    One unsupportive verdict is enough to stop a claim being served; a consensus
    rule would make the gate exactly as strong as its most permissive judge, which
    is the opposite of what more judges are for.

    The assertion row, its spans, and this verdict all stay. `rejected` is a state,
    not a deletion -- the rejected set IS the measurement of whether the gate works.
    """
    with _atomic(conn):
        cur = conn.execute(
            "INSERT INTO verdicts (assertion_id, judge, verdict, rationale) "
            "VALUES (?,?,?,?) RETURNING id",
            (assertion_id, judge, verdict, rationale),
        )
        verdict_id = int(cur.fetchone()[0])
        if verdict != VERDICT_SUPPORTED:
            # Only from 'active': a claim already stale stays stale, and its
            # staleness is the more actionable fact about it.
            conn.execute(
                _TOUCH_STATUS, (STATUS_REJECTED, assertion_id, STATUS_ACTIVE)
            )
        _sync_assertion_document(conn, assertion_id)
    return verdict_id


def mark_stale(
    conn: sqlite3.Connection,
    assertion_id: int,
    reason: str,
    *,
    span_id: int | None = None,
    expected_hash: str | None = None,
    observed_hash: str | None = None,
    detected_at: str | None = None,
) -> bool:
    """Expire an active assertion and log which citation moved. Returns True if it
    transitioned.

    Idempotent, and the log row is written only on the transition. A claim that is
    already stale (or rejected) is left exactly as it is, so `staleness_log` holds
    one row per expiry event rather than one per read -- which is what makes its
    growth rate a real signal about how fast this repo invalidates its own
    inferences, instead of a count of how often somebody asked.

    `detected_at` exists so ONE expiry carries ONE timestamp. A caller that has
    already read the clock -- `stale.serve_assertions` reports that read back as
    `checked_at` -- must be able to stamp the log with the same instant instead of
    letting the column default fire a second read. Two reads of the same event drift
    by a millisecond, and a later call that correctly reports the logged
    `detected_at` then disagrees with what the detecting call returned. The record
    of when a claim expired must not depend on which function looked at the clock.
    Omitted, the schema default applies, which is right for callers with no clock
    read of their own to reconcile.
    """
    with _atomic(conn):
        cur = conn.execute(
            _TOUCH_STATUS, (STATUS_STALE, assertion_id, STATUS_ACTIVE)
        )
        if cur.rowcount == 0:
            return False
        if detected_at is None:
            conn.execute(
                "INSERT INTO staleness_log (assertion_id, span_id, reason, "
                " expected_hash, observed_hash) VALUES (?,?,?,?,?)",
                (assertion_id, span_id, reason, expected_hash, observed_hash),
            )
        else:
            conn.execute(
                "INSERT INTO staleness_log (assertion_id, span_id, reason, "
                " expected_hash, observed_hash, detected_at) VALUES (?,?,?,?,?,?)",
                (assertion_id, span_id, reason, expected_hash, observed_hash,
                 detected_at),
            )
        _sync_assertion_document(conn, assertion_id)
    return True


@dataclass(frozen=True, slots=True)
class _Unread:
    """A cited file yielded no bytes, carrying WHY -- and therefore what may be done.

    A bare `None` was the defect. It collapsed "this file has been deleted" and "this
    process could not open this file just now" into one value, and every caller
    downstream then had to pick a single disposition for both; the one it picked was
    expiry. Carrying the reason instead means the decision is made where the errno is
    still in scope, and made once for both verifiers -- `stale` imports this type
    rather than deciding again, because two verifiers each inventing their own
    disposition for the same repo state is precisely the disagreement WP10 found.

    `reason` is `REASON_FILE_MISSING` (terminal) or `REASON_UNREADABLE` (not).
    """

    reason: str


# The two dispositions, as singletons, so a caller can compare reasons rather than
# instances and nothing has to allocate on the hot path.
_ABSENT = _Unread(REASON_FILE_MISSING)
_UNREADABLE = _Unread(REASON_UNREADABLE)


def reinstate(
    conn: sqlite3.Connection, assertion_id: int, repo_root: db.StrPath | None = None
) -> bool:
    """Re-hash a stale claim's citations and return it to `active` if they ALL match.

    The only route out of `stale` that is not hand-edited SQL, and the counterweight
    to a store that had exactly one direction of travel. Before WP10 an assertion
    could be expired by a transient read failure -- `chmod 000`, a descriptor limit, a
    volume that was not mounted at the wrong moment -- and nothing anywhere could
    bring it back, so a filesystem hiccup was permanent data loss dressed as a
    staleness event. `_read_source` now stops most of those expiries from happening;
    this exists for the ones already in the table, and for the honest case where a
    revert really did restore the cited bytes.

    **It is evidence-driven, and there is deliberately no override.** No `force`, no
    `assume_fresh`, no way to say "trust me". A flag that promotes a claim without
    re-reading its evidence is a cached freshness verdict entered by hand, which is
    the single failure this whole tier is built to refuse -- and it would be a worse
    one than a cache, because it would carry a human's authority and no timestamp.
    The re-hash is the entire mechanism: the same `_first_failure` that expires claims
    is the one that un-expires them, so the two can never disagree about what "the
    evidence matches" means.

    Four refusals, and each is a different kind of not-allowed:

    * **`rejected` raises `NotReinstatable`.** A judge refused this claim on evidence
      that was correct at the time. Re-reading the bytes cannot answer that, and a
      promotion here would let a `git revert` overturn an adjudication -- see the
      exception's own docstring.
    * **`active` returns `False`.** Nothing to do; not an error. Same shape as
      `mark_stale` returning `False` for a claim that was already stale.
    * **No spans returns `False`.** "Every cited span matches" is trivially true of no
      spans, so an evidence-free claim would otherwise be promoted by the very code
      that verifies a well-cited one -- the vacuous truth guarded against on the serve
      path, guarded against again here because this path *writes* `active`.
    * **Any citation that does not match returns `False`.** Including one that could
      not be READ: an unreadable file establishes nothing, and "we could not check" is
      not grounds for promotion any more than it is grounds for expiry. Every span
      must match; a claim is only as reinstated as its weakest citation.

    Nothing is written to `staleness_log`. That table is one row per expiry, and its
    growth rate is the measurement of how fast this repo invalidates its own
    inferences; adding un-expiry rows would make the number mean nothing. The record
    of a reinstatement is the log rows that stay exactly where they were plus a
    `status_changed_at` later than the `detected_at` above them, which is readable and
    does not corrupt the count.

    Returns True only if this call performed the transition.
    """
    root = _repo_root(conn, repo_root)
    found = _load_assertions(conn, "id = ?", (assertion_id,))
    if not found:
        raise KeyError(f"no assertion with id {assertion_id} in this index")
    assertion = found[0]
    if assertion.status == STATUS_REJECTED:
        raise NotReinstatable(
            f"assertion {assertion_id} is {STATUS_REJECTED!r}, not {STATUS_STALE!r}, "
            "and was not changed. A refuted claim is not an expired one: its evidence "
            "was correct when a judge read it and found the claim unsupported, so "
            "re-hashing that evidence cannot overturn the verdict. Record a "
            "supporting verdict if the adjudication was wrong."
        )
    if assertion.status != STATUS_STALE:
        return False
    if not assertion.spans:
        return False
    if _first_failure(root, assertion.spans, {}) is not None:
        return False
    with _atomic(conn):
        cur = conn.execute(_TOUCH_STATUS, (STATUS_ACTIVE, assertion_id, STATUS_STALE))
        if cur.rowcount:
            _sync_assertion_document(conn, assertion_id)
    # `rowcount == 0` means another connection moved this row between the load above
    # and the write -- most likely a rejection. The WHERE clause declined to clobber
    # it, and saying so honestly is the whole reason the FROM-state is in the SQL.
    return bool(cur.rowcount)


def _read_source(root: Path, path: str, cache: dict[str, bytes | _Unread]) -> bytes | _Unread:
    """Read a cited file once per verification pass, distinguishing gone from unopenable.

    Cached because a batch of claims about one module would otherwise re-read that
    module once per claim, and verification runs on every serve.

    Three outcomes, and the split between the last two is the whole of WP10:

    * **bytes** -- the file is there and was read.
    * **`_ABSENT`** -- `FileNotFoundError` or `NotADirectoryError`. Nothing is at this
      path and nothing can be: the file was deleted, or renamed, or a parent component
      of the path is a file rather than a directory. That is a finding about the
      repository, it expires the claim, and it is logged as `file_missing`.
    * **`_UNREADABLE`** -- every other `OSError`, plus anything that is not a regular
      file. `EACCES` from a permission bit, `EMFILE` under load, `EIO` off a failing
      disk, `ENOTCONN`/`ESTALE` off an NFS mount, a repo root that is not mounted right
      now. None of these say anything about whether the cited bytes still match, so
      the claim is withheld for this call and its status is left alone.

    The old code caught every `OSError` and returned `None`, which the callers expired
    as `file_missing`. `chmod 000` on a cited file therefore destroyed every claim
    citing it, permanently, and told the operator the file was gone while it sat there
    at 6 bytes; `chmod 644` brought nothing back, because no code path in this module
    moved an assertion towards `active`. Failing closed is correct. Failing closed
    with no way back, on evidence that never moved, is data loss.

    The regular-file test is not redundant with the `except OSError`. Catching OSError
    handles every way a read can fail loudly; a FIFO fails quietly, by blocking in
    `read_bytes` until another process opens the write end. This function runs on the
    serve path, so a claim citing a FIFO would hang whoever asked for it -- and since
    nothing raises, the caller sees a call that never returns rather than an assertion
    that went stale. It is `_UNREADABLE` rather than `_ABSENT`, because a FIFO is not
    an absence: something is there, this reader cannot safely open it, and putting a
    regular file back must be enough to restore the claim.

    `stat()` is called explicitly rather than via `Path.is_file()` so that the errno
    survives. `is_file()` swallows the same `OSError`s into a bare `False`, which is
    how the two cases got conflated in the first place.
    """
    if path in cache:
        return cache[path]
    target = root / path
    try:
        st = target.stat()
    except (FileNotFoundError, NotADirectoryError):
        cache[path] = _ABSENT
        return cache[path]
    except OSError:
        # EACCES on a parent directory, ELOOP, ENOTCONN off a dead mount. The path may
        # be perfectly intact; this process simply cannot see it from here.
        cache[path] = _UNREADABLE
        return cache[path]
    if not stat_module.S_ISREG(st.st_mode):
        cache[path] = _UNREADABLE
        return cache[path]
    try:
        cache[path] = target.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        # Deleted between the stat and the open. Genuinely gone, so genuinely terminal.
        cache[path] = _ABSENT
    except OSError:
        cache[path] = _UNREADABLE
    return cache[path]


def _first_failure(
    root: Path, spans: Sequence[EvidenceSpan], cache: dict[str, bytes | _Unread]
) -> tuple[str, EvidenceSpan, str | None] | None:
    """Return the first citation that no longer verifies, or None if all still do.

    First rather than all: one moved span is already enough to expire the claim, and
    the one that moved is the one worth recording. Which span failed is more useful
    than how many did.

    The returned reason may be `REASON_UNREADABLE`, which is NOT a failure of the
    citation and must not be treated as one. Every caller branches on it: the serve
    paths withhold without touching `status`, and `write_assertion` raises
    `EvidenceUnverifiable` rather than `EvidenceStale`. A caller that forgets the
    branch reintroduces the exact defect this split exists to remove, which is why
    each of the three call sites says so out loud.
    """
    for span in spans:
        source = _read_source(root, span.path, cache)
        if isinstance(source, _Unread):
            return (source.reason, span, None)
        if span.byte_end > len(source):
            # Slicing past the end of a bytes object silently returns a short
            # result, which would then hash to something that is simply "not the
            # expected hash". Naming truncation separately keeps a deleted tail
            # distinguishable from an edit.
            return (REASON_SPAN_TRUNCATED, span, None)
        observed = content_hash(source[span.byte_start:span.byte_end])
        if observed != span.content_hash:
            return (REASON_HASH_MISMATCH, span, observed)
    return None


def _load_assertions(
    conn: sqlite3.Connection, where: str, params: Sequence[object]
) -> list[Assertion]:
    """Load assertions matching `where`, each with its spans attached.

    S608: `where` is assembled from this module's own literals and never from a
    caller's string -- every value a caller supplies arrives as a bound parameter.
    The two interpolations here (this predicate and the `IN` placeholder run below)
    exist because SQLite cannot bind a column list or a variable-length `IN`.

    The span lookup is CHUNKED, and that is a correctness fix rather than a tuning
    one. One `IN (...)` over every matching id binds one variable per assertion, so
    this query -- which sits under `servable_assertions`, `assertions_with_status`,
    `serve_assertions` and `refresh_staleness`, i.e. every path that serves or sweeps
    -- raised `too many SQL variables` above 32,766 active claims and took all four
    down at once. `stale` had already batched its own `IN` lookups against exactly
    this hazard; the query they all depend on had not, so the batching stopped one
    statement short of the one that would fail first. Spans accumulate ACROSS batches
    into one dict, so an assertion's citations are never split by where a batch
    boundary happened to land.
    """
    rows = list(
        conn.execute(
            "SELECT id, subject_qualname, subject_symbol_id, kind, claim, status, "  # noqa: S608
            "       generator, confidence, created_at "
            f"FROM assertions WHERE {where} ORDER BY id",
            tuple(params),
        )
    )
    if not rows:
        return []
    spans: dict[int, list[EvidenceSpan]] = {}
    ids = [r["id"] for r in rows]
    for batch in _chunks(ids):
        placeholders = ",".join("?" * len(batch))
        for s in conn.execute(
            "SELECT id, assertion_id, path, line_start, line_end, byte_start, "  # noqa: S608
            "       byte_end, content_hash FROM evidence_spans "
            f"WHERE assertion_id IN ({placeholders}) ORDER BY id",
            tuple(batch),
        ):
            spans.setdefault(s["assertion_id"], []).append(
                EvidenceSpan(
                    path=s["path"],
                    line_start=s["line_start"],
                    line_end=s["line_end"],
                    byte_start=s["byte_start"],
                    byte_end=s["byte_end"],
                    content_hash=s["content_hash"],
                    id=s["id"],
                )
            )
    return [
        Assertion(
            id=r["id"],
            subject_qualname=r["subject_qualname"],
            subject_symbol_id=r["subject_symbol_id"],
            kind=r["kind"],
            claim=r["claim"],
            status=r["status"],
            generator=r["generator"],
            confidence=r["confidence"],
            created_at=r["created_at"],
            spans=tuple(spans.get(r["id"], ())),
        )
        for r in rows
    ]


def load_assertions_by_ids(
    conn: sqlite3.Connection, assertion_ids: Sequence[int]
) -> list[Assertion]:
    """Load only the supplied assertions, preserving first-occurrence input order.

    The public bounded reader for retrieval paths. IDs are deduplicated before any
    query and split into SQLite-safe chunks; missing IDs are omitted. This is
    deliberately separate from ``assertions_with_status`` and
    ``servable_assertions`` so a candidate window never becomes a scan of the whole
    assertion store by accident.
    """
    ordered_ids = list(dict.fromkeys(assertion_ids))
    loaded: dict[int, Assertion] = {}
    for batch in _chunks(ordered_ids):
        placeholders = ",".join("?" * len(batch))
        for assertion in _load_assertions(
            conn,
            f"id IN ({placeholders})",  # noqa: S608 - placeholders only
            tuple(batch),
        ):
            loaded[assertion.id] = assertion
    return [loaded[assertion_id] for assertion_id in ordered_ids if assertion_id in loaded]


def verify_assertions(
    conn: sqlite3.Connection,
    repo_root: db.StrPath,
    assertion_ids: Sequence[int],
) -> list[Assertion]:
    """Re-hash only the supplied assertions and return those servable right now.

    Terminal evidence failures use :func:`mark_stale`, preserving the authoritative
    lifecycle transition, staleness log, derived-document removal, and transaction
    behavior. A transient unreadability withholds the assertion for this call and
    writes nothing. IDs are deduplicated before both database and filesystem work.
    """
    root = _repo_root(conn, repo_root)
    cache: dict[str, bytes | _Unread] = {}
    verified: list[Assertion] = []
    for assertion in load_assertions_by_ids(conn, assertion_ids):
        if assertion.status != STATUS_ACTIVE:
            continue
        if not assertion.spans:
            mark_stale(conn, assertion.id, REASON_NO_EVIDENCE)
            continue
        failure = _first_failure(root, assertion.spans, cache)
        if failure is None:
            verified.append(assertion)
            continue
        reason, span, observed = failure
        if reason == REASON_UNREADABLE:
            continue
        mark_stale(
            conn,
            assertion.id,
            reason,
            span_id=span.id,
            expected_hash=span.content_hash,
            observed_hash=observed,
        )
    return verified


def servable_assertions(
    conn: sqlite3.Connection,
    repo_root: db.StrPath | None = None,
    *,
    subject_qualname: str | None = None,
    kind: str | None = None,
) -> list[Assertion]:
    """Return the assertions that may be served right now, verifying as it goes.

    Servable means `status = 'active'` AND every cited span still hashes to what was
    cited. Both halves are checked here on every call -- the stored status is a
    filter, never the answer, because it is a record of the last time somebody
    looked rather than a statement about the code as it is now.

    Anything that fails verification is expired as a side effect: marked stale, with
    the failing citation written to `staleness_log`. Detection and demotion are the
    same operation on purpose. Splitting them would leave a window in which the
    store knows a claim is wrong and is still willing to hand it to the next caller.

    A citation that could not be READ is withheld from the result and NOT expired --
    see `_read_source`. Withheld rather than served-and-flagged, because this function
    returns bare `Assertion` rows and has nowhere to put a flag: everything in the
    returned list has, by this function's own contract, just been re-verified, and an
    unchecked claim in that list is indistinguishable from a checked one. "We could
    not look" must never leave here wearing the shape of "we looked and it is fine".
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

    cache: dict[str, bytes | _Unread] = {}
    servable: list[Assertion] = []
    for assertion in _load_assertions(conn, " AND ".join(where), params):
        if not assertion.spans:
            # Unreachable while the write gate holds -- and checked anyway. "Every
            # cited span matches" is trivially true of no spans, so this is the one
            # case where the verifier would otherwise hand back a claim resting on
            # nothing, and report it as verified.
            mark_stale(conn, assertion.id, REASON_NO_EVIDENCE)
            continue
        failure = _first_failure(root, assertion.spans, cache)
        if failure is None:
            servable.append(assertion)
            continue
        reason, span, observed = failure
        if reason == REASON_UNREADABLE:
            # Withheld, not expired, and no log row: nothing was established about
            # this claim, so there is nothing to record and nothing to change. It
            # returns of its own accord on the next call that can read the file.
            continue
        mark_stale(
            conn,
            assertion.id,
            reason,
            span_id=span.id,
            expected_hash=span.content_hash,
            observed_hash=observed,
        )
    return servable


def is_servable(
    conn: sqlite3.Connection, assertion_id: int, repo_root: db.StrPath | None = None
) -> bool:
    """Whether one assertion may be served, with the same verification and the same
    expiry side effect as `servable_assertions`.

    A `False` here does not on its own mean the claim expired: an unreadable citation
    is `False` too, and leaves the status untouched. The boolean answers "may this be
    served right now", which is the question the caller asked."""
    root = _repo_root(conn, repo_root)
    found = _load_assertions(conn, "id = ?", (assertion_id,))
    if not found:
        return False
    assertion = found[0]
    if assertion.status != STATUS_ACTIVE:
        return False
    if not assertion.spans:
        mark_stale(conn, assertion.id, REASON_NO_EVIDENCE)
        return False
    failure = _first_failure(root, assertion.spans, {})
    if failure is None:
        return True
    reason, span, observed = failure
    if reason == REASON_UNREADABLE:
        # Not servable, and not expired. See `servable_assertions`.
        return False
    mark_stale(
        conn,
        assertion.id,
        reason,
        span_id=span.id,
        expected_hash=span.content_hash,
        observed_hash=observed,
    )
    return False


def assertions_with_status(conn: sqlite3.Connection, status: str) -> list[Assertion]:
    """Every assertion in `status`, spans attached, verified against nothing.

    The un-verified reader, and the reason the rejected and stale sets are worth
    keeping: reviewing what the gate threw out is not the same act as serving it,
    and must not quietly re-check or re-promote anything while doing so.
    """
    return _load_assertions(conn, "status = ?", (status,))


def verdicts_for(conn: sqlite3.Connection, assertion_id: int) -> list[sqlite3.Row]:
    """Every verdict recorded against one assertion, oldest first."""
    return list(
        conn.execute(
            "SELECT * FROM verdicts WHERE assertion_id = ? ORDER BY id",
            (assertion_id,),
        )
    )


def staleness_events(
    conn: sqlite3.Connection, assertion_id: int | None = None
) -> list[sqlite3.Row]:
    """Expiry events, newest last. All of them, or one assertion's."""
    if assertion_id is None:
        return list(conn.execute("SELECT * FROM staleness_log ORDER BY id"))
    return list(
        conn.execute(
            "SELECT * FROM staleness_log WHERE assertion_id = ? ORDER BY id",
            (assertion_id,),
        )
    )
