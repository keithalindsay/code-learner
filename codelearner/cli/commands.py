"""The four commands: index, search, stats, learn.

Every failure here is somebody's Tuesday afternoon, so the rule this module follows
is that a user must never see a traceback for a condition the tool could have
predicted. A missing index, an index without embeddings, a model that does not match
the vectors in the file -- these are all normal states of the world, and each one
gets a sentence that says what happened and what to do about it. `CliError` is the
carrier for exactly that: raised here, printed without a stack by `main`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import db
from ..assertions import store
from ..index import Embedder, embed_chunks
from ..ingest import index_repo
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


def open_index(index_path: Path) -> sqlite3.Connection:
    """Open an EXISTING index, or explain how to make one.

    `db.connect` happily creates an empty SQLite file at any path, which is how a
    typo'd path becomes "0 results" instead of "no such index". Checking for the
    file first is the difference between a wrong answer and an error message.

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
        return db.connect(index_path)
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        # The exception's own remedy says "delete the index file and re-index",
        # which was the honest advice until the tier-2 store could survive a
        # rebuild. It costs the embeddings now, and nothing else.
        raise CliError(f"{exc} {REBUILD_ADVICE}") from exc
    except sqlite3.Error as exc:
        raise CliError(f"could not open the index at {index_path}: {exc}") from exc


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

    before = _scalar(conn, "SELECT count(*) FROM staleness_log")
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

    # The honest outcome for a claim whose evidence moved while the index was being
    # rebuilt is `stale` with a log row naming the citation that moved -- not a
    # deletion, and not a silent promotion either. `servable_assertions` is the same
    # verification the read path runs, called here so that the answer a rebuild gives
    # and the answer the next query gives cannot differ.
    store.servable_assertions(conn, repo)
    expired = _scalar(conn, "SELECT count(*) FROM staleness_log") - before

    totals = _dump_totals(dump)
    return CarryReport(
        assertions=totals["assertions"],
        evidence_spans=totals["evidence_spans"],
        verdicts=totals["verdicts"],
        staleness_events=totals["staleness_log"],
        subjects_resolved=resolved,
        subjects_unresolved=unresolved,
        expired_by_rebuild=expired,
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
        },
    }

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
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args: Any, factory: EmbedderFactory) -> int:
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    conn = open_index(index_path)

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

    # Notes go to stderr unconditionally so that `--json` on stdout stays a single
    # parseable document and a shell pipeline does not have to strip warnings.
    for note in notes:
        print(f"codelearner: {note}", file=sys.stderr)

    if args.json:
        print(
            json.dumps(
                {
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
                    "hits": [hit_json(hit, i) for i, hit in enumerate(hits, start=1)],
                },
                indent=2,
            )
        )
        return 0

    enabled = [
        name
        for name, on in (("lexical", use_lexical), ("dense", use_dense), ("graph", use_graph))
        if on
    ]
    if not hits:
        print(f"no results for {args.query!r}  [{'+'.join(enabled)}]")
        return 0
    print(f"{len(hits)} result(s) for {args.query!r}  [{'+'.join(enabled)}, k={args.k}]")
    for rank, hit in enumerate(hits, start=1):
        print(format_hit(hit, rank))
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


def cmd_stats(args: Any, factory: EmbedderFactory) -> int:
    del factory  # stats never loads a model; the stored name is all it reports
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    conn = open_index(index_path)

    try:
        counts = {
            "files": _scalar(conn, "SELECT count(*) FROM files"),
            "symbols": _scalar(conn, "SELECT count(*) FROM symbols"),
            "edges": _scalar(conn, "SELECT count(*) FROM edges"),
            "chunks": _scalar(conn, "SELECT count(*) FROM chunks"),
        }
        tier_rows = conn.execute(
            "SELECT tier, count(*) AS n FROM edges GROUP BY tier"
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
        # they were parsed, not decided.
        "tiers": {
            "T0": by_tier.get(0, 0),
            "T1": by_tier.get(1, 0),
            "T2": by_tier.get(2, 0),
        },
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
    print("edges by tier")
    print(count_line("T0 FACT", by_tier.get(0, 0), width=12) + "  call site as written, unbound")
    print(count_line("T1 RESOLVED", by_tier.get(1, 0), width=12) + "  bound to a symbol, with confidence")
    print(count_line("T2 INFERRED", by_tier.get(2, 0), width=12) + "  the inference layer is not built yet")
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
    conn = open_index(index_path)

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
        print(
            json.dumps(
                {
                    "repo": str(repo),
                    "index": str(index_path),
                    "generator": report.generator,
                    "considered": report.considered,
                    "skipped_existing": report.skipped_existing,
                    "symbols_without_offers": report.symbols_without_offers,
                    "drafts_requested": report.drafts_requested,
                    "admitted": report.admitted,
                    "refused_empty_claim": report.refused_empty_claim,
                    "refused_no_citation": report.refused_no_citation,
                    "invalid_refs": report.invalid_refs,
                    "drafts_citing_off_menu": report.drafts_citing_off_menu,
                    "offers_dropped_unreadable": report.offers_dropped_unreadable,
                    "generator_errors": report.generator_errors,
                    "admission_rate": report.admission_rate,
                },
                indent=2,
            )
        )
    else:
        print(report.format_report())
    return 0
