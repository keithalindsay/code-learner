"""The five commands: index, search, stats, learn, gpu.

Every failure here is somebody's Tuesday afternoon, so the rule this module follows
is that a user must never see a traceback for a condition the tool could have
predicted. A missing index, an index without embeddings, a model that does not match
the vectors in the file -- these are all normal states of the world, and each one
gets a sentence that says what happened and what to do about it. `CliError` is the
carrier for exactly that: raised here, printed without a stack by `main`.

The same principle extends one step further, and the drift survey below is where it
lands: **a user must not see a WRONG ANSWER for a condition the tool could have
predicted either.** An index built on Monday and queried on Friday serves tier-0
line numbers -- "deterministic, reproducible from source alone" -- for a file that
has since moved under it. There is no traceback and no error; the citation is simply
off by twenty lines and looks exactly like one that is right. That is a worse failure
than the ones this module was already built to catch, because nothing about the
output invites doubt.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from .. import db, gpu
from ..assertions import boundaries, store
from ..evidence import EvidenceBundle, EvidenceError, assemble_evidence
from ..index import Embedder, embed_chunks
from ..ingest import index_repo, iter_python_files
from ..retrieve import load_reranker, search, stored_embed_model
from .render import count_line, facts_only, format_hit, hit_json

# Kept in step with `indexer.index_repo`'s own default. One file per repo is what
# makes cross-repo contamination structurally impossible, so the CLI must not
# invent a second convention for where that file lives.
INDEX_RELPATH = Path(".codelearner") / "index.db"

EmbedderFactory = Callable[[str], Embedder]


class CliError(RuntimeError):
    """A condition with a known remedy. Printed as one line, never as a traceback."""


def resolve_index_path(repo: Path, index_path: Path | None) -> Path:
    """Where this invocation's index lives: explicit if given, else the default."""
    if index_path is not None:
        return index_path.expanduser()
    return repo / INDEX_RELPATH


# ---------------------------------------------------------------------------
# drift: the index measured against the tree it was built from
# ---------------------------------------------------------------------------
#
# `files.content_hash`, `files.mtime_ns` and `files.size_bytes` were written at index
# time and read by NO code path. Tier 2 meanwhile has a two-stage staleness engine, a
# `span_verifications` baseline and a `staleness_log` -- so the tier this project
# describes as *inferred* was the one with a drift check, and the tier it describes as
# *fact* was the one that would serve a line number that had moved and say nothing.
# That inversion is what this section closes.
#
# Three failures, deliberately counted apart, because they are not the same event and
# they do not have the same symptom:
#
#   changed   -- the file is there and its bytes are not the ones that were parsed.
#                Citations still resolve, still look right, and point at the wrong
#                lines. This is the dangerous one: it is invisible in the output.
#   missing   -- an indexed file is gone. Citations name a path with nothing behind
#                it, so the failure at least announces itself when followed.
#   unindexed -- a .py file the tree has and the index does not. Nothing is wrong
#                with what IS returned; the problem is what is absent. A query about
#                a symbol added after indexing returns "no results", exit 0, which is
#                indistinguishable from a repo that genuinely does not contain it.
#
# Folding them into one number would tell a reader to expect the wrong symptom.

# Named in the payload and in the note so nobody has to guess what was compared, and
# therefore what was not. Content hashing every file would be exhaustive; it would
# also read every byte of the repository on every `search`, which is a cost this check
# cannot justify when the cheap comparison catches every edit made by an ordinary
# editor. The residual is stated wherever the count is printed.
DRIFT_METHOD = "mtime_ns+size_bytes"


@dataclass(frozen=True)
class DriftReport:
    """How far the tree has moved since the index was built, by three measures.

    `checked` is False when the survey could not be run at all -- an index bound to
    no repo root, or bound to one that is not on this machine. It is NOT a synonym
    for "clean", which is why every count is None rather than 0 in that state: a
    reader who cannot tell "measured zero" from "did not measure" will read the
    second as the first, and the whole value of this check is that it does not
    quietly assert things it did not establish.

    `unindexed` is separately nullable, because the tree enumeration can fail (an
    unreadable directory) while the stat sweep over the indexed files succeeds.
    """

    checked: bool
    repo_root: str | None = None
    indexed: int | None = None
    changed: int | None = None
    missing: int | None = None
    unindexed: int | None = None

    def as_json(self) -> dict[str, Any]:
        """The machine-readable half of the note, for `--json` on stdout."""
        return {
            "checked": self.checked,
            "indexed": self.indexed,
            "changed": self.changed,
            "missing": self.missing,
            "unindexed": self.unindexed,
            "method": DRIFT_METHOD,
        }


def _count_unindexed(root: Path, indexed: set[str]) -> int | None:
    """How many .py files the tree has that the index does not. None if unknowable.

    Enumerated with `iter_python_files` -- the indexer's own function, not a second
    walk written to look like it. Any independent enumeration would drift from the
    one that built the index and start reporting `.venv/` or a worktree copy as
    "added", which is a false alarm on a warning whose entire value is that it is
    never a false alarm. Using the same function means "in the tree and not in the
    index" is exactly "would be indexed now, and was not".

    Two causes, and the note says both: a file written since the index was built,
    and a file that was present then and was counted in `IndexStats.skipped` because
    it would not parse or would not read. The second does not go away on re-index,
    and a message that promised it would is a message that sends its reader round a
    loop.

    The comparison runs the index's relative paths OUT to absolute strings rather
    than running the tree's absolute paths back IN via `Path.relative_to`, and it
    builds them by string join rather than with `/`. Both are measured choices and
    neither is stylistic, because this runs on every `search`: at 2,556 files,
    `relative_to().as_posix()` per file cost ~90ms and `root / rel` per file another
    ~15ms, against ~0.5ms for the join and ~32ms for the enumeration they were
    wrapping. The naive spelling was three times the cost of the work itself.
    """
    try:
        found = list(iter_python_files(root))
    except OSError:
        return None
    prefix = str(root)
    if not prefix.endswith(os.sep):
        prefix += os.sep
    if os.sep == "/":
        known = {prefix + rel for rel in indexed}
    else:  # pragma: no cover - stored paths are POSIX; this is the Windows spelling
        known = {prefix + rel.replace("/", os.sep) for rel in indexed}
    return sum(1 for path in found if str(path) not in known)


def survey_drift(conn: sqlite3.Connection) -> DriftReport:
    """`stat()` every indexed file and compare against what the index recorded.

    **Every file, never a sample.** Measured on a 2,556-file repository: 25ms for the
    stat sweep over the indexed files, 32ms for the tree enumeration, 58ms together,
    against 2ms of actual query work and a ~100ms interpreter-and-import floor that
    `codelearner search` pays before it does anything at all. So the sweep is roughly
    a third of a search invocation, and it is linear -- 25,000 files would be ~0.6s,
    at which point it is worth revisiting.

    Sampling was rejected on correctness rather than on that cost. The question this
    check answers is "is this index behind the tree", which is a property of the union
    of the files; a sample of k answers only "these k are current", and the useful
    output -- silence -- is exactly the output a sample is not entitled to produce. A
    note reading "no changes detected (sampled 200 of 2,556)" would be a statement the
    tool cannot back, printed at the moment a user is deciding whether to trust a
    citation, and this project's entire thesis is the difference between measured and
    asserted. A cheap check that is wrong is worse than no check: it converts "I do not
    know whether this index is fresh" into "I have been told it is".

    The repo root comes from the index's own `meta`, never from `--repo`. The
    question being asked is "has the tree this index was built from moved", and the
    index is the only thing that knows which tree that was; taking the answer from a
    command-line argument would let a mistyped `--repo` report drift against a
    directory the index was never about.

    **An `OSError` that is not an absence is not counted.** A `chmod 000` file, an
    `EMFILE`, an NFS blip: none of those are evidence that the file changed, and
    reporting them as drift would make a transient environmental fault look like an
    edit -- the same mistake WP10.3 records on the tier-2 side, where a swallowed
    `OSError` permanently expired claims. Withholding is the honest answer, and it
    errs towards under-reporting, which is the direction this check is already
    documented to err in.
    """
    root_text = db.stored_repo_root(conn)
    if root_text is None:
        return DriftReport(checked=False)
    root = Path(root_text)
    if not root.is_dir():
        # The index is on this machine and the tree it describes is not. Nothing can
        # be compared, and saying "0 changed" would be a claim about a directory this
        # process cannot see.
        return DriftReport(checked=False, repo_root=root_text)
    try:
        rows = conn.execute("SELECT path, size_bytes, mtime_ns FROM files").fetchall()
    except sqlite3.Error:
        return DriftReport(checked=False, repo_root=root_text)

    indexed: set[str] = set()
    changed = missing = 0
    for row in rows:
        rel = str(row["path"])
        indexed.add(rel)
        try:
            st = os.stat(root / rel)
        except (FileNotFoundError, NotADirectoryError):
            # Real absence, the same split `assertions/stale.py` makes: the file is
            # not there, as opposed to this process being unable to look at it.
            missing += 1
            continue
        except OSError:
            continue
        if st.st_size != int(row["size_bytes"]) or st.st_mtime_ns != int(row["mtime_ns"]):
            changed += 1

    return DriftReport(
        checked=True,
        repo_root=root_text,
        indexed=len(rows),
        changed=changed,
        missing=missing,
        unindexed=_count_unindexed(root, indexed),
    )


def drift_note(report: DriftReport) -> str | None:
    """One sentence-set for stderr, or None when there is nothing to say.

    **None on a clean tree is the design, not an omission.** A line printed on every
    invocation is a line a user has stopped reading by Wednesday, and a warning that
    has been trained out of its reader is worth less than no warning -- it occupies
    the place where a real one would have gone. So this speaks only when the
    condition is true, which is also why there is no flag to silence it: the mute
    switch for a warning that only fires when the index IS behind the tree is
    `codelearner index --force`, and anything else silences the sole indication that
    tier-0 answers have stopped being facts. A pipeline that wants it gone has
    `2>/dev/null` and, better, the `drift` object in the `--json` document.

    The counts are described as floors, in the message and not only in this
    docstring, because mtime and size can both survive an edit -- a writer that
    restores the timestamp, or a same-length substitution. This is a cheap check that
    is right when it speaks and incomplete when it is silent, and it says so.
    """
    if not report.checked:
        return None
    moved: list[str] = []
    if report.changed:
        moved.append(
            f"{report.changed} of {_plural(report.indexed or 0, 'indexed file')} "
            f"{'has' if report.changed == 1 else 'have'} changed on disk since this "
            "index was built"
        )
    if report.missing:
        moved.append(
            f"{_plural(report.missing, 'indexed file')} "
            f"{'is' if report.missing == 1 else 'are'} no longer on disk"
        )

    sentences: list[str] = []
    if moved:
        sentences.append(
            " and ".join(moved)
            + "; hits and claims may cite bytes that have moved or are not there."
        )
    if report.unindexed:
        sentences.append(
            f"{_plural(report.unindexed, '.py file')} in the tree "
            f"{'is' if report.unindexed == 1 else 'are'} not in this index at all "
            "(written since it was built, or skipped as unreadable at index time), so "
            "a query about them returns nothing rather than something wrong."
        )
    if not sentences:
        return None
    sentences.append(
        f"Re-run `codelearner index {report.repo_root or '<repo>'} --force "
        "--carry-assertions`."
    )
    sentences.append(
        f"Compared by {DRIFT_METHOD} only -- an edit that preserves both is not "
        "detected, so these counts are floors rather than an audit."
    )
    return " ".join(sentences)


def open_index(index_path: Path) -> tuple[sqlite3.Connection, DriftReport]:
    """Open an EXISTING index, or explain how to make one. Survey it for drift.

    `db.connect` happily creates an empty SQLite file at any path, which is how a
    typo'd path becomes "0 results" instead of "no such index". Checking for the
    file first is the difference between a wrong answer and an error message.

    **The drift survey lives here for the same reason the schema check does**: every
    read command arrives through this function, so a rule enforced here cannot be
    forgotten by the next one added. The note is printed here rather than by each
    caller, so a future command gets the human warning whether or not its author
    thought about staleness; the report is RETURNED as well, so a `--json` caller can
    put the same facts in its document without re-running the sweep. The worst a
    forgetful caller can now do is omit the machine-readable copy, never the warning.

    Printing to stderr is what makes that safe: this module's standing rule is that
    stdout under `--json` is one parseable document, so a note on stderr can be
    unconditional without any caller having to know it exists.

    `SchemaVersionError` is caught here and not only in `cmd_index`, because it is
    the READ paths that meet it. `db.connect` gained its version check precisely so
    that a stale index cannot answer a query, and every one of those queries arrives
    through this function -- so catching only `sqlite3.Error` meant the single most
    predicted failure in the design (the stamp has moved five times) came out of
    `stats`, `search`, and `learn` as a traceback, which this module's first
    sentence promises it never will. `RepoRootMismatchError` rides along: it is
    raised by `bind_repo_root` rather than by `connect` today, but it is the same
    class of condition -- a file this code refuses to read -- and a caller that
    starts binding here would otherwise reintroduce the same traceback.

    `SchemaVersionError` is caught here and not only in `cmd_index`, because it is
    the READ paths that meet it. `db.connect` gained its version check precisely so
    that a stale index cannot answer a query, and every one of those queries arrives
    through this function -- so catching only `sqlite3.Error` meant the single most
    predicted failure in the design (the stamp has moved five times) came out of
    `stats`, `search`, and `learn` as a traceback, which this module's first
    sentence promises it never will. `RepoRootMismatchError` rides along: it is
    raised by `bind_repo_root` rather than by `connect` today, but it is the same
    class of condition -- a file this code refuses to read -- and a caller that
    starts binding here would otherwise reintroduce the same traceback.
    """
    if not index_path.exists():
        raise CliError(
            f"no index at {index_path}. Build one with "
            f"`codelearner index <repo>`, or point at an existing one with "
            f"--index-path."
        )
    try:
        conn = db.connect(index_path)
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        # The exception's own remedy says "delete the index file and re-index",
        # which was the honest advice until the tier-2 store could survive a
        # rebuild. It costs the embeddings now, and nothing else.
        raise CliError(f"{exc} {REBUILD_ADVICE}") from exc
    except sqlite3.Error as exc:
        raise CliError(f"could not open the index at {index_path}: {exc}") from exc
    drift = survey_drift(conn)
    note = drift_note(drift)
    if note is not None:
        print(f"codelearner: {note}", file=sys.stderr)
    return conn, drift


def build_embedder(factory: EmbedderFactory, model_name: str) -> Embedder:
    """Construct an embedder, turning every way that can fail into one sentence.

    Loading a model reaches for torch, a GPU, and ~1.2GB of weights on disk. Any of
    the three can be absent, and none of them produce an error a user can act on
    without being told what was being attempted.
    """
    try:
        return factory(model_name)
    except ImportError as exc:
        raise CliError(
            f"embedding needs the optional dependencies: {exc}. "
            'Install them with `pip install -e ".[embed]"`.'
        ) from exc
    except Exception as exc:  # noqa: BLE001 - the CLI's job is to never traceback
        raise CliError(f"could not load the embedding model {model_name!r}: {exc}") from exc


def _delete_index(index_path: Path) -> None:
    """Remove an index file and its WAL sidecars.

    The `-wal` and `-shm` files are not incidental: deleting only the main file
    leaves a write-ahead log that SQLite will replay into the fresh database,
    resurrecting rows from the index that was supposed to be gone.
    """
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(index_path) + suffix)
        if candidate.exists():
            candidate.unlink()


# ---------------------------------------------------------------------------
# the tier-2 store across a rebuild
# ---------------------------------------------------------------------------
#
# A rebuild replaces `files`, `symbols`, `edges`, and `chunks` wholesale, and every
# one of those is re-derivable from source in seconds. The four tables below are not
# re-derivable at all: a verdict is what a judge concluded on one particular day, and
# the rejected set is the only evidence the gate does anything. Deleting the DB file
# threw all of it away and called it "discards its embeddings".
#
# The `ON DELETE SET NULL` on `assertions.subject_symbol_id` was written for exactly
# this moment and had never fired, because nothing in the package deletes a `symbols`
# row -- the file simply went. Carrying the store means the schema's reasoning
# finally executes: the qualname is durable and is re-resolved against the rebuilt
# graph, the id link is disposable and comes back NULL when it cannot be re-made.

CARRY_SUFFIX = ".carry"

# Bumped if the column lists below change, so a carry file written by older code is
# refused loudly instead of being read into the wrong columns.
CARRY_FORMAT = 1

REBUILD_ADVICE = (
    "Since --carry-assertions that no longer costs the assertions: rebuild with "
    "`codelearner index <repo> --force --carry-assertions` and the tier-2 store "
    "(assertions, verdicts, staleness log) is carried across; only the embeddings "
    "have to be re-derived."
)

# The four tables and every column of each, in insert order. Written out rather than
# read from `PRAGMA table_info`, because a carry file has to be a statement about
# which columns were preserved: a column added to `schema.sql` and forgotten here
# should surface as an obviously missing field, not as a silently narrower dump.
#
# `span_verifications` is deliberately NOT carried. It is the stat() baseline for the
# fast path, the schema calls it disposable, and dropping it costs one re-hash per
# cited span. Carrying it would be worse than useless: a rebuild happens because the
# repo moved, so a baseline saying "these bytes were fine at this mtime" is the one
# piece of state most likely to be wrong and most able to authorise skipping the read
# that would have found out.
_CARRY_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "assertions",
        ("id", "subject_qualname", "subject_symbol_id", "kind", "claim", "status",
         "generator", "confidence", "created_at", "status_changed_at"),
    ),
    (
        "evidence_spans",
        ("id", "assertion_id", "path", "line_start", "line_end", "byte_start",
         "byte_end", "content_hash"),
    ),
    ("verdicts", ("id", "assertion_id", "judge", "verdict", "rationale", "created_at")),
    (
        "staleness_log",
        ("id", "assertion_id", "span_id", "reason", "expected_hash", "observed_hash",
         "detected_at"),
    ),
)

# The carry file's own schema carries no types, no CHECK constraints, and no foreign
# keys. It is a transport, not a second store: a constraint here could refuse rows the
# real store already holds, and the first time that happened would be halfway through
# a rebuild with the index already deleted.
_CARRY_SCHEMA = "\n".join(
    [f"CREATE TABLE {table} ({', '.join(columns)});" for table, columns in _CARRY_TABLES]
    + ["CREATE TABLE carry_meta (key TEXT PRIMARY KEY, value TEXT);"]
)

Dump = dict[str, list[tuple[Any, ...]]]


@dataclass(frozen=True)
class CarryReport:
    """What survived a rebuild, and in what state."""

    assertions: int
    evidence_spans: int
    verdicts: int
    staleness_events: int
    subjects_resolved: int
    subjects_unresolved: int
    expired_by_rebuild: int
    # Counted apart from `expired_by_rebuild` because the two ask for different
    # things. That number is claims whose cited bytes moved, which the next
    # generation run re-derives from a repo that has settled; this one is claims
    # whose bytes are untouched and whose citation boundary predates the schema this
    # rebuild just installed, and no amount of re-indexing repairs it. Folding them
    # together would hand the operator one number and the wrong remedy for half of it.
    narrowed_citations: int = 0
    recovered: bool = False


def carry_path(index_path: Path) -> Path:
    """Where the store waits while the index it came from does not exist.

    Next to the index rather than in a temp directory, and on disk rather than in a
    Python list, because the failure this whole mechanism exists for is the process
    not surviving. `index_repo` raising after `_delete_index` -- a parse error, a full
    disk, a Ctrl-C, an OOM kill -- used to end with the index gone and the store gone
    with it. An in-memory dump dies with the interpreter; a sidecar outlives it, and
    the next `codelearner index` finds it and puts the store back.
    """
    return Path(str(index_path) + CARRY_SUFFIX)


def _store_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for the four tier-2 tables; 0 for any table this DB lacks.

    A pre-v4 index has none of them, and that is a legitimate thing to find rather
    than an error: there is simply no store to carry.
    """
    counts: dict[str, int] = {}
    for table, _ in _CARRY_TABLES:
        try:
            row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
        except sqlite3.Error:
            counts[table] = 0
            continue
        counts[table] = 0 if row is None else int(row[0])
    return counts


def _dump_store(index_path: Path) -> Dump:
    """Read the whole tier-2 store out of an index that is about to be deleted.

    Opened with `check_schema=False` on purpose. `--force` IS the documented remedy
    for a schema mismatch, so the one moment this function matters most is the moment
    the version check would refuse -- a v4 index being rebuilt by v5 code holds a
    perfectly readable assertion store, and refusing to look at it would make the
    upgrade path the thing that destroys the store.

    Refuses to guess when a table exists, holds rows, and cannot be read with the
    columns named above. That is a store this code cannot carry, and deleting it
    quietly on the grounds that the dump failed is the exact behaviour being removed.
    """
    conn = db.connect(index_path, check_schema=False)
    try:
        counts = _store_counts(conn)
        dump: Dump = {}
        for table, columns in _CARRY_TABLES:
            if counts[table] == 0:
                dump[table] = []
                continue
            try:
                rows = conn.execute(
                    f"SELECT {', '.join(columns)} FROM {table} ORDER BY id"  # noqa: S608
                ).fetchall()
            except sqlite3.Error as exc:
                raise CliError(
                    f"the {table} table in {index_path} holds {counts[table]} row(s) "
                    f"that this code cannot read ({exc}), so a rebuild would destroy "
                    "a tier-2 store it is unable to carry across. Remedy: rebuild "
                    "with a code-learner that understands this file, or re-run with "
                    "--force --discard-assertions to destroy it deliberately."
                ) from exc
            dump[table] = [tuple(row) for row in rows]
        return dump
    finally:
        conn.close()


def _dump_totals(dump: Dump) -> dict[str, int]:
    return {table: len(dump.get(table, [])) for table, _ in _CARRY_TABLES}


def _write_carry_file(carry: Path, dump: Dump, *, repo: Path, index_path: Path) -> None:
    """Publish the dump atomically, so a crash can never leave half a store.

    Written to `<carry>.partial` and renamed into place. `os.replace` is atomic on
    every platform this runs on, which is what makes the sidecar's presence mean
    exactly one thing: a complete store is waiting to be restored. A file that could
    be found half-written would need its own validation pass, and a validation pass
    that ran on the recovery path is a second place for the store to be lost.

    A second SQLite file rather than JSON: the rows come back with the same types,
    the same NULLs, and the same integer ids they went in with, without a bespoke
    encoder standing between the store and its only backup.
    """
    partial = Path(str(carry) + ".partial")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(partial) + suffix)
        if candidate.exists():
            candidate.unlink()
    out = sqlite3.connect(str(partial))
    try:
        out.executescript(_CARRY_SCHEMA)
        for table, columns in _CARRY_TABLES:
            placeholders = ",".join("?" * len(columns))
            out.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({placeholders})",
                dump.get(table, []),
            )
        out.executemany(
            "INSERT INTO carry_meta (key, value) VALUES (?,?)",
            [
                ("carry_format", str(CARRY_FORMAT)),
                ("repo_root", str(repo)),
                ("source_index", str(index_path)),
            ],
        )
        out.commit()
    finally:
        out.close()
    os.replace(partial, carry)


def _read_carry_file(carry: Path, *, repo: Path) -> Dump:
    """Read a waiting store back, refusing anything it cannot vouch for.

    The repo root is checked because a carry file names claims by qualname and by
    repo-relative path, and restoring one into an index of a different tree would
    attach real citations to unrelated bytes -- which every later verification would
    then report as staleness blaming an edit nobody made.

    An unreadable carry file is refused rather than ignored. Ignoring it would let a
    single corrupt sidecar turn into a silent, permanent loss of the one thing in
    this index that cannot be re-derived; refusing leaves the file where it is and
    tells the operator it is there.
    """
    try:
        conn = sqlite3.connect(f"file:{carry}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CliError(
            f"a tier-2 store from an interrupted rebuild is waiting at {carry} but "
            f"could not be opened ({exc}). Nothing has been deleted. Remedy: move "
            "that file aside once you accept losing the assertions in it."
        ) from exc
    try:
        try:
            meta = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT key, value FROM carry_meta")
            }
            dump: Dump = {}
            for table, columns in _CARRY_TABLES:
                dump[table] = [
                    tuple(row)
                    for row in conn.execute(
                        f"SELECT {', '.join(columns)} FROM {table} ORDER BY id"  # noqa: S608
                    )
                ]
        except sqlite3.Error as exc:
            raise CliError(
                f"the tier-2 store waiting at {carry} could not be read ({exc}). "
                "Nothing has been deleted. Remedy: move that file aside once you "
                "accept losing the assertions in it."
            ) from exc
    finally:
        conn.close()
    if meta.get("carry_format") != str(CARRY_FORMAT):
        raise CliError(
            f"the tier-2 store waiting at {carry} is carry format "
            f"v{meta.get('carry_format') or '(unstamped)'}, and this code writes and "
            f"reads v{CARRY_FORMAT}. Nothing has been deleted. Remedy: restore it "
            "with the code-learner that wrote it, or move it aside."
        )
    stored_root = meta.get("repo_root")
    if stored_root is not None and stored_root != str(repo):
        raise CliError(
            f"the tier-2 store waiting at {carry} was dumped from an index of "
            f"{stored_root!r}, and this run is indexing {str(repo)!r}. Its citations "
            "are paths and qualnames relative to the other tree, so restoring it "
            "here would bind real claims to unrelated bytes. Nothing has been "
            "deleted. Remedy: index this repo at its own --index-path, or move that "
            "file aside."
        )
    return dump


def _restore_store(conn: sqlite3.Connection, repo: Path, dump: Dump) -> CarryReport:
    """Put the store back into the rebuilt index, and let the staleness engine judge it.

    **Restored by direct INSERT, deliberately not through `write_assertion`.** That
    function is the admission gate and enforces six rules, one of which re-verifies
    every citation against disk. None of them apply here: these claims were already
    admitted, on evidence that hashed at the time, and this is not a second admission
    -- it is the same rows surviving a file being replaced underneath them. Sending
    them back through the door would refuse exactly the ones most worth keeping. A
    `rejected` claim would be re-admitted as `active` or refused outright; a `stale`
    one would be refused as `EvidenceStale`, which means the store's record of what
    went wrong would be destroyed by the machinery that exists to record it; and a
    claim about a symbol this rebuild no longer parses would be refused as
    `UnknownSubject` when the schema is explicitly shaped to keep it with a NULL link.
    Re-verification at admission answers "should this be let in"; these are already
    in, and the question that remains is the freshness question, which is answered
    below by the same engine that answers it on every read.

    Ids, statuses, `created_at` and `status_changed_at` are all preserved: an id is
    what the verdicts and the staleness log point at, and a `created_at` rewritten by
    a rebuild would turn "we served that claim for three months" into "we wrote it
    today".

    `subject_symbol_id` is the one field that is NOT preserved. It is re-resolved from
    `subject_qualname`, which is `NOT NULL` for precisely this reason, and left NULL
    when the rebuilt graph has no such symbol -- the case the schema's `ON DELETE SET
    NULL` was written for and which, until now, could never happen because the row was
    deleted along with the file.
    """
    assertion_rows = dump.get("assertions", [])
    resolved = unresolved = 0
    prepared: list[tuple[Any, ...]] = []
    for row in assertion_rows:
        qualname = row[1]
        found = conn.execute(
            "SELECT id FROM symbols WHERE qualname = ?", (qualname,)
        ).fetchone()
        if found is None:
            unresolved += 1
            symbol_id: int | None = None
        else:
            resolved += 1
            symbol_id = int(found[0])
        prepared.append((row[0], qualname, symbol_id, *row[3:]))

    with db.transaction(conn):
        for table, columns in _CARRY_TABLES:
            rows = prepared if table == "assertions" else dump.get(table, [])
            if not rows:
                continue
            placeholders = ",".join("?" * len(columns))
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({placeholders})",
                rows,
            )

    # Read AFTER the carried rows have landed, not before. `staleness_log` is one of
    # the carried tables, so a store arriving with prior expiries would otherwise have
    # its own history counted as this rebuild's work -- a repo that had never gone
    # stale reported 0 and one that had gone stale last week reported last week's
    # number, which is the shape of a count nobody would question until it mattered.
    # The baseline is what is in the table once the carry is complete and before the
    # verification below is allowed to add to it.
    before = _scalar(conn, "SELECT count(*) FROM staleness_log")

    # The honest outcome for a claim whose evidence moved while the index was being
    # rebuilt is `stale` with a log row naming the citation that moved -- not a
    # deletion, and not a silent promotion either. `servable_assertions` is the same
    # verification the read path runs, called here so that the answer a rebuild gives
    # and the answer the next query gives cannot differ.
    store.servable_assertions(conn, repo)
    expired = _scalar(conn, "SELECT count(*) FROM staleness_log") - before

    # Second, and only after the bytes have had their say. A carried claim can be
    # perfectly fresh and still be citing a symbol without the decorators that were
    # added to that symbol's span by the very schema bump this rebuild is applying --
    # the one shape of wrongness a re-hash cannot see, because the bytes did not move.
    # See `assertions.boundaries`. Run second so that a claim which is BOTH edited and
    # narrowed is logged once, as the edit, which is the more urgent of the two.
    narrowed = boundaries.expire_narrowed_citations(conn, repo)

    totals = _dump_totals(dump)
    return CarryReport(
        assertions=totals["assertions"],
        evidence_spans=totals["evidence_spans"],
        verdicts=totals["verdicts"],
        staleness_events=totals["staleness_log"],
        subjects_resolved=resolved,
        subjects_unresolved=unresolved,
        expired_by_rebuild=expired,
        narrowed_citations=narrowed,
    )


def _recover_carry(carry: Path, repo: Path) -> Dump:
    """Take a waiting store, and say out loud that a previous run did not finish.

    Announced on stderr rather than left to the summary, because the operator's
    mental model at this point is "my index is gone"; a run that silently produced a
    correct index would leave them believing the store went with it.
    """
    dump = _read_carry_file(carry, repo=repo)
    print(
        f"codelearner: recovering the tier-2 store left by an interrupted rebuild "
        f"({carry})",
        file=sys.stderr,
    )
    return dump


def _plural(count: int, noun: str) -> str:
    """`1 verdict`, `2 verdicts`.

    Worth four lines because of where the string is read: by somebody deciding
    whether to destroy the one part of this index that cannot be rebuilt. "1
    verdicts" reads as generated boilerplate at the exact moment the number needs to
    read as a fact about their data.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _rebuild_refusal(index_path: Path, totals: dict[str, int]) -> CliError:
    """The refusal that used to be a deletion, with the real numbers in it.

    "Discards its embeddings" was true and beside the point. Embeddings are minutes
    of GPU time; a verdict is a judgement that was made once, and the rejected set is
    the only evidence the gate does anything at all.
    """
    named = (
        f"{_plural(totals['assertions'], 'assertion')}, "
        f"{_plural(totals['verdicts'], 'verdict')}, "
        f"{_plural(totals['staleness_log'], 'staleness event')}"
    )
    return CliError(
        f"an index already exists at {index_path}, and it holds a tier-2 store: "
        f"{named}. Rebuilding from scratch discards {named} and any embeddings -- "
        "and only the embeddings are re-derivable. Re-run with --force "
        "--carry-assertions to rebuild and carry the store across (a claim whose "
        "evidence moved comes back stale, with a log row, rather than vanishing), or "
        "--force --discard-assertions to destroy it deliberately."
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def cmd_index(args: Any, factory: EmbedderFactory) -> int:
    """Build an index, and carry the one part of it that cannot be rebuilt.

    The order of operations is the whole point and is not negotiable: dump the
    tier-2 store to a sidecar, THEN delete the index, THEN rebuild, THEN restore and
    only then remove the sidecar. Every failure in the middle leaves the store on
    disk with a file whose presence means "an interrupted rebuild owes this index a
    store", which the next run of this command finds and honours -- including the
    plain, no-flags run the operator is most likely to reach for after a crash.
    """
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise CliError(f"{repo} is not a directory, so there is nothing to index.")

    index_path = resolve_index_path(repo, args.index_path)
    carry = carry_path(index_path)
    discarding = bool(getattr(args, "discard_assertions", False))
    carried: Dump | None = None
    recovered = False

    if index_path.exists():
        if not args.force:
            raise CliError(
                f"an index already exists at {index_path}. There is no incremental "
                "update yet, so re-indexing means rebuilding from scratch. Re-run "
                "with --force to delete and rebuild it -- note that this discards "
                "any embeddings, which are the expensive part -- or use "
                "--index-path to build a second index elsewhere."
            )
        if carry.exists() and not discarding:
            # An earlier rebuild died between the delete and the restore, so this
            # file is the only copy of the store. `index_repo` never writes an
            # assertion, which is what makes the sidecar the superset: whatever sits
            # in the half-built index that replaced the old one either came from this
            # file or is not a claim. Overwriting it with a dump of that index would
            # finish the loss the crash only started.
            carried = _recover_carry(carry, repo)
            recovered = True
        elif not discarding:
            dump = _dump_store(index_path)
            totals = _dump_totals(dump)
            if totals["assertions"]:
                if not getattr(args, "carry_assertions", False):
                    raise _rebuild_refusal(index_path, totals)
                _write_carry_file(carry, dump, repo=repo, index_path=index_path)
                carried = dump
        _delete_index(index_path)
    elif carry.exists() and not discarding:
        # The index is not here and a store is. That is the crash-after-delete state
        # exactly, and it is reached by the command an operator types next.
        carried = _recover_carry(carry, repo)
        recovered = True
    if discarding and carry.exists():
        # --discard-assertions means what it says, including about a store that is
        # only waiting because a previous rebuild was interrupted.
        carry.unlink()

    try:
        conn, stats = index_repo(repo, index_path=index_path)
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        raise CliError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise CliError(f"indexing failed while writing {index_path}: {exc}") from exc

    carry_report: CarryReport | None = None
    if carried is not None:
        report = _restore_store(conn, repo, carried)
        # Only now, and only after the restore committed. The sidecar is the store's
        # only copy for the whole window above; removing it any earlier would open a
        # gap in which a crash loses everything, which is the gap this exists to close.
        carry.unlink(missing_ok=True)
        carry_report = replace(report, recovered=recovered)

    embed_info: dict[str, Any] | None = None
    if args.embed:
        # Warn BEFORE the model load, not after. The embedder's own CPU-fallback
        # warning is correct and it is also too late: by the time it fires, a minute
        # has gone into loading weights onto the wrong device, and it fires from
        # inside a library into a log rather than at the person who just typed the
        # command. This says the same thing while there is still a decision to make.
        # It warns and never acts -- see `gpu.warn_if_contended` for why evicting
        # someone else's model is not this command's call.
        gpu.warn_if_contended()
        embedder = build_embedder(factory, args.model)
        try:
            estats = embed_chunks(conn, embedder)
        except RuntimeError as exc:
            # sqlite-vec missing or unloadable. The structural half of the index is
            # already written and useful, so this reports rather than unwinds.
            raise CliError(str(exc)) from exc
        embed_info = {
            "model": estats.model,
            "dim": estats.dim,
            "embedded": estats.embedded,
            "skipped_unchanged": estats.skipped_unchanged,
        }

    rstats = stats.resolve
    in_repo = rstats.total - rstats.external
    payload = {
        "repo": str(repo),
        "index": str(index_path),
        "files": stats.files,
        "symbols": stats.symbols,
        "edges": stats.edges,
        "chunks": stats.chunks,
        "skipped": stats.skipped,
        "resolution": {
            "total": rstats.total,
            "resolved": rstats.resolved,
            "external": rstats.external,
            "ambiguous": rstats.ambiguous,
            "in_repo": in_repo,
            "rate": round(rstats.rate, 6),
            "rate_of_internal": round(rstats.rate_of_internal, 6),
            "by_resolver": dict(sorted(rstats.by_resolver.items())),
        },
        "embeddings": embed_info,
        # Reported rather than left implicit: a rebuild that quietly expired half the
        # store would look exactly like one that carried all of it, and the number
        # that separates them is the one worth printing.
        "tier2": None if carry_report is None else {
            "carried": True,
            "recovered": carry_report.recovered,
            "assertions": carry_report.assertions,
            "evidence_spans": carry_report.evidence_spans,
            "verdicts": carry_report.verdicts,
            "staleness_events": carry_report.staleness_events,
            "subjects_resolved": carry_report.subjects_resolved,
            "subjects_unresolved": carry_report.subjects_unresolved,
            "expired_by_rebuild": carry_report.expired_by_rebuild,
            "narrowed_citations": carry_report.narrowed_citations,
        },
    }

    # Close before returning, and not as tidiness. A `sqlite3.Connection` forms a
    # reference cycle with its own statement cache, so dropping the last reference does
    # NOT finalize it -- only the cyclic collector does, and only then does SQLite
    # checkpoint and unlink `index.db-wal`. Leaving that to the collector makes the
    # moment a build's WAL disappears depend on unrelated allocation pressure, which is
    # how `test_an_unmeasurable_tree_is_not_reported_as_a_clean_one` came to fail in
    # roughly a quarter of full-suite runs and never when run alone: a directory listing
    # taken before the collector fired was walked after it did.
    conn.close()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"indexed {repo}")
    print(f"  index      {index_path}")
    print(count_line("files", stats.files))
    print(count_line("symbols", stats.symbols))
    print(count_line("edges", stats.edges))
    print(count_line("chunks", stats.chunks))
    if stats.skipped:
        print(count_line("skipped", stats.skipped))
    # Two denominators, because only one of them is honest. Roughly half the calls
    # in real code target stdlib or third-party code and are CORRECTLY unresolvable;
    # counting those as failures makes a working resolver look broken.
    print(
        f"  resolved   {rstats.resolved:>9,}  "
        f"{rstats.rate_of_internal:.1%} of {in_repo:,} in-repo references "
        f"({rstats.external:,} target code outside this repo)"
    )
    if embed_info is not None:
        print(
            f"  embedded   {embed_info['embedded']:>9,}  "
            f"chunks with {embed_info['model']} ({embed_info['dim']}-dim)"
        )
    if carry_report is not None:
        print(
            f"  carried    {carry_report.assertions:>9,}  "
            f"assertions ({carry_report.subjects_resolved} re-linked to a symbol, "
            f"{carry_report.subjects_unresolved} left unlinked), "
            f"{_plural(carry_report.verdicts, 'verdict')}, "
            f"{_plural(carry_report.staleness_events, 'staleness event')}"
        )
        if carry_report.expired_by_rebuild:
            print(
                f"  expired    {carry_report.expired_by_rebuild:>9,}  "
                "carried claims whose cited bytes have moved -- marked stale, with a "
                "log row naming the citation, not deleted"
            )
        if carry_report.narrowed_citations:
            # Worded to send the operator somewhere different from the line above.
            # Re-indexing is the reflex when a rebuild reports a number, and it is
            # the one thing that cannot help here: the bytes are unchanged and the
            # symbol table is already correct. The claim was derived from a span that
            # stopped short of its own decorators, and only re-deriving the claim --
            # over the whole symbol this time -- repairs that.
            print(
                f"  narrowed   {carry_report.narrowed_citations:>9,}  "
                f"carried claims citing a symbol without its decorators -- marked "
                f"stale ({store.REASON_DECORATORS_EXCLUDED}). Their bytes are "
                "unchanged, so re-indexing will not repair them; they have to be "
                "redrafted against the whole symbol."
            )
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def _print_evidence(bundle: EvidenceBundle) -> None:
    """Render opt-in current source after the ranked results."""
    print()
    print(
        "source evidence "
        f"({bundle.used_bytes}/{bundle.budget_bytes} bytes; "
        f"{bundle.sections_omitted} section(s) omitted)"
    )
    for section in bundle.sections:
        print(
            f"--- {section.qualname}  {section.path}:{section.line_start}-{section.line_end}  "
            f"sha256:{section.content_hash} ---"
        )
        print(section.source)


def cmd_search(args: Any, factory: EmbedderFactory) -> int:
    repo = args.repo.expanduser().resolve()
    index_path = resolve_index_path(repo, args.index_path)
    conn, drift = open_index(index_path)

    use_lexical = not args.no_lexical
    use_dense = not args.no_dense
    use_graph = not args.no_graph
    notes: list[str] = []
    embedder: Embedder | None = None

    if use_dense:
        stored = stored_embed_model(conn)
        if stored is None:
            # The common case, and not an error: an index without embeddings still
            # answers with lexical and graph. Degrading loudly beats failing.
            use_dense = False
            notes.append(
                "dense retrieval unavailable: this index has no embeddings. Build "
                "them with `codelearner index <repo> --embed --force`."
            )
        else:
            embedder = build_embedder(factory, stored)
            if embedder.name != stored:
                # Vectors from two models are not comparable. Querying anyway
                # returns results that look plausible and mean nothing, which is
                # strictly worse than returning none.
                notes.append(
                    f"dense retrieval disabled: this index was embedded with "
                    f"{stored!r} but the loaded model is {embedder.name!r}, and "
                    "vectors from two models are not comparable."
                )
                use_dense = False
                embedder = None

    if not use_lexical and not use_dense:
        # Graph expansion cannot run alone. It has no query representation of its
        # own -- it is seeded by the text modalities -- so with both of them off it
        # would return nothing at all, for every query, silently.
        raise CliError(
            "no text modality is available, so there is nothing to search with. "
            "Graph expansion has no query representation of its own; it is seeded "
            "by lexical and dense results and cannot run alone. Drop --no-lexical, "
            "or build embeddings with `codelearner index <repo> --embed --force`."
        )

    reranker = None
    if getattr(args, "rerank", False):
        reranker = load_reranker(conn=conn)
        if reranker is None:
            # Asked for and not available. Say so and keep going -- the fused order
            # is the result every release before Phase 3b returned, and refusing the
            # query would be a strictly worse answer than a slightly worse ranking.
            notes.append(
                "reranking unavailable: no cross-encoder could be loaded (no model "
                "weights, or not enough memory). Returning the fused order."
            )

    result = search(
        conn,
        args.query,
        k=args.k,
        embedder=embedder,
        use_lexical=use_lexical,
        use_dense=use_dense,
        use_graph=use_graph,
        reranker=reranker,
    )
    hits = facts_only(result.hits) if args.facts_only else list(result.hits)
    evidence: EvidenceBundle | None = None
    if args.include_source:
        stored_root = db.stored_repo_root(conn)
        if stored_root is None:
            raise CliError(
                f"the index at {index_path} is not bound to a repository. Re-index "
                "it before requesting source evidence."
            )
        evidence_root = Path(stored_root).expanduser().resolve()
        if getattr(args, "repo_explicit", False) and repo != evidence_root:
            raise CliError(
                f"--repo names {repo}, but the index at {index_path} belongs to the "
                f"different repository {evidence_root}. Use that repository or its "
                "index."
            )
        try:
            evidence = assemble_evidence(
                conn,
                evidence_root,
                hits,
                budget_bytes=args.evidence_budget,
            )
        except EvidenceError as exc:
            raise CliError(exc.message) from exc

    # Notes go to stderr unconditionally so that `--json` on stdout stays a single
    # parseable document and a shell pipeline does not have to strip warnings.
    for note in notes:
        print(f"codelearner: {note}", file=sys.stderr)

    if args.json:
        payload: dict[str, Any] = {
            "query": args.query,
            "index": str(index_path),
            "k": args.k,
            "facts_only": args.facts_only,
            "modalities": {
                "lexical": use_lexical,
                "dense": use_dense,
                "graph": use_graph,
            },
            "count": len(hits),
            # The machine-readable half of the stderr note. A consumer that
            # never sees stderr would otherwise be the one surface that
            # cannot tell it is being handed line numbers that have moved.
            "drift": drift.as_json(),
            "hits": [hit_json(hit, i) for i, hit in enumerate(hits, start=1)],
        }
        if evidence is not None:
            payload["evidence"] = evidence.as_json()
        print(json.dumps(payload, indent=2))
        return 0

    enabled = [
        name
        for name, on in (("lexical", use_lexical), ("dense", use_dense), ("graph", use_graph))
        if on
    ]
    if not hits:
        print(f"no results for {args.query!r}  [{'+'.join(enabled)}]")
    else:
        print(f"{len(hits)} result(s) for {args.query!r}  [{'+'.join(enabled)}, k={args.k}]")
        for rank, hit in enumerate(hits, start=1):
            print(format_hit(hit, rank))
    if evidence is not None:
        _print_evidence(evidence)
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return 0 if row is None else int(row[0])


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def _classify_unresolved(conn: sqlite3.Connection) -> tuple[int, int]:
    """Split unresolved edges into (external, ambiguous), mirroring `resolve_all`.

    Recomputed from the stored graph rather than remembered from index time, so
    `stats` tells the truth about the file in front of it even if the resolver was
    re-run since. External means "no symbol in this repo even shares the basename",
    which is the only honest way to say that a call to `json.dumps` is not a
    resolution failure.
    """
    names = {row["name"] for row in conn.execute("SELECT DISTINCT name FROM symbols")}
    external = ambiguous = 0
    for row in conn.execute("SELECT dst_name FROM edges WHERE dst_symbol_id IS NULL"):
        base = str(row["dst_name"]).rsplit(".", 1)[-1]
        if base in names:
            ambiguous += 1
        else:
            external += 1
    return external, ambiguous


def _embedding_info(conn: sqlite3.Connection) -> dict[str, Any]:
    """What vectors this index holds, if any, and from which model."""
    model = stored_embed_model(conn)
    dim = _meta(conn, "embed_dim")
    try:
        vectors = _scalar(conn, "SELECT count(*) FROM vec_chunks")
    except sqlite3.OperationalError:
        # No vec table, or sqlite-vec is not loadable on this handle. Either way
        # there is nothing to report and nothing to fail about.
        vectors = 0
    return {
        "present": bool(model and vectors),
        "model": model,
        "dim": int(dim) if dim is not None else None,
        "vectors": vectors,
    }


def _print_freshness(drift: DriftReport) -> None:
    """The one place the clean case is printed as well as the dirty one.

    `search` stays silent on a fresh index, because a line on every query is a line
    nobody reads. `stats` is the opposite situation: somebody typed it to ask what
    state this index is in, so "nothing has moved" is an answer to the question they
    asked rather than noise attached to a different one. Both halves state what was
    compared, so neither reads as a guarantee about bytes.
    """
    if not drift.checked:
        body = (
            "not checked: this index names no repository root, or names one that is "
            "not a directory on this machine."
        )
    else:
        body = drift_note(drift) or (
            f"{_plural(drift.indexed or 0, 'indexed file')} still match the tree by "
            f"{DRIFT_METHOD}, and no .py file in the tree is missing from the index. "
            "An edit that preserves mtime and size is not detected by this check, so "
            "this is not a statement about bytes."
        )
    print("freshness")
    # `break_long_words=False` is load-bearing: the note carries the remedy command
    # with an absolute repo path in it, and a wrapper that splits that path mid-token
    # produces a line the reader cannot copy, which is the only thing they wanted it
    # for. Better an over-long line than a broken command.
    print(
        textwrap.fill(
            body,
            width=88,
            initial_indent="  ",
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def cmd_stats(args: Any, factory: EmbedderFactory) -> int:
    """What is in this index -- including the half of it that was not derived from source.

    This command was blind to the assertion store for the whole of Phase 9. It read a
    `T2 INFERRED` count out of `edges.tier`, a column tier 2 never occupies, so the
    number was structurally always 0; and it annotated that structural 0 with "the
    inference layer is not built yet" while the layer sat in the same file holding
    claims, verdicts and expiries. After a full `learn` run, the one command whose
    entire job is to say what an index contains reported nothing about what had been
    learned, and said something false about why.

    The assertion payload deliberately matches the MCP `index_stats` tool's, field for
    field: `counts["assertions"]` and `assertions_by_status` with explicit zeros. Two
    surfaces over one index that disagree about the shape of the answer are worse than
    either shape, because the disagreement is what a reader ends up debugging.
    """
    del factory  # stats never loads a model; the stored name is all it reports
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    conn, drift = open_index(index_path)

    try:
        counts = {
            "files": _scalar(conn, "SELECT count(*) FROM files"),
            "symbols": _scalar(conn, "SELECT count(*) FROM symbols"),
            "edges": _scalar(conn, "SELECT count(*) FROM edges"),
            "chunks": _scalar(conn, "SELECT count(*) FROM chunks"),
            "assertions": _scalar(conn, "SELECT count(*) FROM assertions"),
        }
        tier_rows = conn.execute(
            "SELECT tier, count(*) AS n FROM edges GROUP BY tier"
        ).fetchall()
        status_rows = conn.execute(
            "SELECT status, count(*) AS n FROM assertions GROUP BY status"
        ).fetchall()
        kind_rows = conn.execute(
            "SELECT kind, count(*) AS n FROM symbols GROUP BY kind ORDER BY n DESC"
        ).fetchall()
        resolver_rows = conn.execute(
            "SELECT resolver, count(*) AS n, avg(confidence) AS conf FROM edges "
            "WHERE dst_symbol_id IS NOT NULL GROUP BY resolver ORDER BY n DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise CliError(
            f"{index_path} does not look like a code-learner index ({exc}). "
            "Remedy: point --index-path at the right file, or re-index."
        ) from exc

    by_tier = {int(r["tier"]): int(r["n"]) for r in tier_rows}
    by_status = {str(r["status"]): int(r["n"]) for r in status_rows}
    resolved = _scalar(conn, "SELECT count(*) FROM edges WHERE dst_symbol_id IS NOT NULL")
    external, ambiguous = _classify_unresolved(conn)
    in_repo = counts["edges"] - external
    rate = resolved / counts["edges"] if counts["edges"] else 0.0
    rate_of_internal = resolved / in_repo if in_repo else 0.0
    embeddings = _embedding_info(conn)

    payload = {
        "index": str(index_path),
        "repo_root": db.stored_repo_root(conn),
        "schema_version": _meta(conn, "schema_version"),
        "counts": counts,
        # The tier column lives on edges: 0 is the call site as written, 1 is that
        # site bound to a symbol. Symbols themselves are all T0 by construction --
        # they were parsed, not decided. `T2` is kept in the map at a permanent 0 so
        # the shape matches the MCP payload, and is explained rather than left to
        # imply that nothing was inferred: tier 2 lives in `assertions`, below.
        "tiers": {
            "T0": by_tier.get(0, 0),
            "T1": by_tier.get(1, 0),
            "T2": by_tier.get(2, 0),
        },
        # Explicit zeros, so the shape does not change when the store is empty and so
        # `rejected` is visible rather than merely absent. The rejected set is the
        # only evidence the gate does anything at all.
        "assertions_by_status": {
            "active": by_status.get("active", 0),
            "rejected": by_status.get("rejected", 0),
            "stale": by_status.get("stale", 0),
        },
        "drift": drift.as_json(),
        "symbol_kinds": {str(r["kind"]): int(r["n"]) for r in kind_rows},
        "resolution": {
            "total": counts["edges"],
            "resolved": resolved,
            "external": external,
            "ambiguous": ambiguous,
            "in_repo": in_repo,
            "rate": round(rate, 6),
            "rate_of_internal": round(rate_of_internal, 6),
            "by_resolver": {
                str(r["resolver"]): {
                    "count": int(r["n"]),
                    "confidence": round(float(r["conf"]), 4) if r["conf"] is not None else None,
                }
                for r in resolver_rows
            },
        },
        "embeddings": embeddings,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"index      {index_path}")
    print(f"repo       {payload['repo_root'] or '(unbound)'}")
    print(f"schema     v{payload['schema_version'] or '?'}")
    print()
    print("counts")
    for label in ("files", "symbols", "edges", "chunks"):
        print(count_line(label, counts[label]))
    print()
    _print_freshness(drift)
    print()
    print("edges by tier")
    print(count_line("T0 FACT", by_tier.get(0, 0), width=12) + "  call site as written, unbound")
    print(count_line("T1 RESOLVED", by_tier.get(1, 0), width=12) + "  bound to a symbol, with confidence")
    print(
        count_line("T2 INFERRED", by_tier.get(2, 0), width=12)
        + "  always 0 here: inference lives in assertions, not on edges"
    )
    print()
    print("assertions (tier 2)")
    if counts["assertions"]:
        for label in ("active", "rejected", "stale"):
            print(count_line(label, by_status.get(label, 0), width=12))
        print(count_line("total", counts["assertions"], width=12))
    else:
        print(
            "  none. Draft some with `codelearner learn`; only claims that cite "
            "evidence the gate can re-verify are admitted."
        )
    print()
    print("symbol kinds")
    for row in kind_rows:
        print(count_line(str(row["kind"]), int(row["n"]), width=12))
    print()
    print("resolution")
    print(
        f"  {resolved:,} of {counts['edges']:,} edges resolved "
        f"-- {rate_of_internal:.1%} of {in_repo:,} in-repo "
        f"references ({external:,} external, {ambiguous:,} ambiguous)"
    )
    for row in resolver_rows:
        conf = row["conf"]
        suffix = f"  confidence {float(conf):.2f}" if conf is not None else ""
        print(count_line(str(row["resolver"]), int(row["n"]), width=24) + suffix)
    print()
    print("embeddings")
    if embeddings["present"]:
        print(
            f"  {embeddings['vectors']:,} vectors from {embeddings['model']} "
            f"({embeddings['dim']}-dim)"
        )
    else:
        print(
            "  none. Dense retrieval is unavailable on this index; build vectors "
            "with `codelearner index <repo> --embed --force`."
        )
    return 0


def cmd_learn(args: Any, factory: EmbedderFactory) -> int:
    """Fill the tier-2 store: draft a cited claim per symbol, admit what survives the gate.

    The only command in this tool that calls a language model, and the only one whose
    output is not derived from source alone. That is why it prints a refusal breakdown
    rather than a success count: `admitted` on its own is indistinguishable between a
    generator that understood the repo and one that cited whatever was in front of it,
    and the numbers that separate those -- how often it abstained, how often it cited
    something that was not on its menu -- are the ones worth reading.

    Progress goes to stderr so that `--json` on stdout stays a clean document even
    while a local model spends twenty minutes working through a repository.
    """
    from ..generate import DEFAULT_GENERATOR_MODEL, OllamaClaimGenerator, learn
    from ..generate.llm import collides_with_judge
    from ..generate.pipeline import PHASE_DONE
    from ..generate.types import GeneratorUnavailable

    repo = args.repo.expanduser().resolve()
    index_path = resolve_index_path(repo, args.index_path)
    conn, drift = open_index(index_path)

    model = args.model or DEFAULT_GENERATOR_MODEL
    generator = OllamaClaimGenerator(model=model, host=args.host)

    # Said once, at the point of use, rather than left to a log nobody reads. The run
    # still happens -- measuring the collision deliberately is legitimate -- but a
    # faithfulness score computed afterwards is no longer cross-family, and the person
    # who has to know that is the one who just typed the command.
    if collides_with_judge(model):
        print(
            f"codelearner: warning: {model!r} is the same model family as the "
            f"faithfulness judge. Claims from it can still be stored, but scoring them "
            f"with `qwen3.5:9b` measures two relatives agreeing rather than an "
            f"independent audit.",
            file=sys.stderr,
        )

    total = [0]

    def on_progress(progress: Any) -> None:
        if progress.phase != PHASE_DONE:
            return
        total[0] += 1
        if not args.json and not args.quiet:
            print(
                f"\r  {total[0]}/{progress.total} {progress.candidate.qualname[:60]:<60}",
                end="",
                file=sys.stderr,
                flush=True,
            )

    try:
        report = learn(
            conn,
            repo,
            generator,
            limit=args.limit,
            max_offers=args.max_offers,
            include_callers=not args.no_callers,
            skip_existing=not args.redo,
            on_progress=on_progress,
        )
    except GeneratorUnavailable as exc:
        # An outage is not a result, and must not be reported as a run that finished
        # with nothing to say. `learn` already refuses to absorb it; this turns it into
        # the one-line, no-traceback failure the rest of the CLI promises.
        raise CliError(str(exc)) from exc
    finally:
        generator.release()

    if not args.json and not args.quiet:
        print("\r" + " " * 72 + "\r", end="", file=sys.stderr)

    if args.json:
        # DERIVED from `LearnReport`, never listed. The hand-written dict this
        # replaces omitted every counter waves 1-2 added -- `refused_invalid_span`,
        # `refused_unverifiable`, `refused_unknown_subject`, `refused_stale_evidence`,
        # `refused_escaping_span`, and `offers_dropped_oversize` with them -- so a run
        # the gate refused outright serialised as a run that admitted nothing and gave
        # no reason. That is precisely the reading `learn` exists to make impossible:
        # `admitted` alone cannot distinguish a generator that understood the repo
        # from one that cited whatever was in front of it, and the refusal breakdown
        # is the thing that separates them. A list of keys is a place for the next
        # counter to be forgotten; the field set cannot be.
        #
        # `results` is the one field excluded, by name rather than by sniffing its
        # type, because a type filter is exactly how the next counter would be
        # silently dropped. `tests/test_cli.py` asserts this key set against
        # `dataclasses.fields(LearnReport)`, so a field that is neither serialised
        # nor deliberately excluded fails the suite rather than a user's pipeline.
        counters = {
            field.name: getattr(report, field.name)
            for field in dataclass_fields(report)
            if field.name != "results"
        }
        print(
            json.dumps(
                {
                    "repo": str(repo),
                    "index": str(index_path),
                    **counters,
                    # Properties, not fields, so they are named here on purpose.
                    "admission_rate": report.admission_rate,
                    "refused_by_the_gate": report.refused_by_the_gate,
                    # A run that drafted claims against an index the tree has moved
                    # under is a run whose refusal counts are about the drift as much
                    # as about the model.
                    "drift": drift.as_json(),
                },
                indent=2,
            )
        )
    else:
        print(report.format_report())
    return 0


# ---------------------------------------------------------------------------
# gpu: who holds the card, and getting it back
# ---------------------------------------------------------------------------

# `--free` exit codes, and the reasoning is that two different readers are watching.
#
# A human wants the whole story printed. A script wants one number it can branch on,
# and the only branch it can act on is "is the card clear or not". So the code follows
# the OUTCOME and not the effort: ollama down and ollama holding nothing are both 0,
# because in both cases the thing the caller wanted is true. Only an attempt that was
# made and did not achieve it is 1 -- which is `main`'s existing meaning for 1, "a
# condition the tool predicted and explained", and the condition here was predicted in
# detail.
#
# Reporting 0 for "asked politely, nothing happened" is the exact bug `gpu.py` exists
# to prevent, one layer up.
#
# "Resident but in use" gets a code of its OWN rather than sharing either, and that is
# the third reading of the same question. It is not 0: the card is not clear, and a
# measurement script that saw success here would start onto a full one. It is not 1
# either, because 1 and this have opposite remedies -- 1 needs a human with sudo, this
# resolves itself the moment the other job finishes, so a script can sensibly sleep and
# retry on 3 while escalating on 1. Collapsing them would force every caller to either
# wait on a condition that will never clear or escalate one that would have.
#
# 3 and not 2: `main` reserves 2 for argparse's "the command line was wrong", and a
# busy GPU is the world being a certain way, not a typo.
GPU_EXIT_OK = 0
GPU_EXIT_NOT_FREED = 1
GPU_EXIT_IN_USE = 3


def cmd_gpu(args: Any, factory: EmbedderFactory) -> int:
    """Show what holds VRAM; with `--free`, ask for it back and verify.

    The only command in this tool that touches no index. That is deliberate -- the
    question "why is my run about to be ten times slower" is asked from a directory
    that may have no index in it, and making the answer depend on one would put it out
    of reach exactly when it is needed.
    """
    del factory  # nothing here loads a model; loading one is what this is about

    # The library defaults to NOT sampling usage, because `warn_if_contended` runs
    # ahead of every `index --embed` and a second and a half there is a second and a
    # half on a hot path. The command defaults the other way: a human is waiting, the
    # cost is below noticing, and the answer decides whether the next thing they are
    # told to do would destroy someone else's work. Opt-in for callers, opt-out here.
    gap = None if args.no_usage_check else gpu.USAGE_SAMPLE_GAP_S

    if not args.free:
        state = gpu.read_state(host=args.host, usage_gap_s=gap)
        if args.json:
            print(json.dumps(state.as_json(), indent=2))
            return GPU_EXIT_OK
        print(gpu.format_state(state))
        for line in gpu.next_step(state):
            print()
            print(line)
        return GPU_EXIT_OK

    report = gpu.release(host=args.host, wait_s=args.wait, force=args.force, usage_gap_s=gap)
    if args.json:
        print(json.dumps(report.as_json(), indent=2))
    else:
        print(gpu.format_release(report))
    if report.declined:
        return GPU_EXIT_IN_USE
    return GPU_EXIT_OK if report.ok else GPU_EXIT_NOT_FREED
