"""The five tools, and the gate that stands behind one of them.

Two rules shape this module.

**Never raise into the transport.** A traceback crossing an MCP boundary tells the
agent that the tool is broken, which is the one conclusion that stops it trying
again. Every predictable condition -- no index, no such symbol, an unverifiable
citation -- comes back as a structured object with a `code`, a `message`, and
whatever the agent needs to fix it. `CliError` does the same job for the human CLI;
this is the machine-facing half of the same policy.

**One derivation, shared, rather than one per surface.** Tier labels, the
`facts_only` filter and the per-hit JSON shape come from `codelearner.tier`; the
assertion gate is `assertions.store.write_assertion` called, not reimplemented. Two
surfaces that answer "is this a fact or a guess" from two code paths will drift, and
the drift shows up as a caller who asked for facts and got a resolver's guess.

That module used to be `cli.render`, which made this server import upward into the
part a person types in order to answer a machine. Sharing the derivation was the
right instinct; putting it in one of the two surfaces was not, and the tier model --
the project's central claim -- now sits in a leaf that both import as peers.
"""
from __future__ import annotations

import asyncio
import math
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from .. import db
from ..assertions import store
from ..assertions.policy import PRODUCTION_POLICY
from ..ingest.types import TIER_RESOLVED
from ..retrieve import stored_embed_model
from ..retrieve.assertions import AssertionSearchUnavailable
from ..retrieve.mixed import search_candidates
from ..retrieve.serialize import candidate_json
from ..retrieve.types import SourceCandidate

# From the leaf packages, not from `..cli.render`. The tier model and the candidate
# shape are the rules both surfaces answer to; they are not the CLI's, and an MCP
# tool should not have to import the thing a person types in order to say what tier
# a result rests on. Tier filtering itself now happens inside retrieval, before the
# page is cut, so this module no longer post-filters a finished list.

# Deferred to first use, for the same reason `_default_embedder` defers the model
# loader: an MCP client launches this server as a subprocess and decides whether its
# tools are offered at all from how fast the handshake comes back, so every import on
# the path between `exec` and the `initialize` reply is charged against the tools ever
# being called. `..cli.commands`, `..onboard`, `..ingest.types` and `..index` are each
# reached only from inside a tool body or an error path, so none of them belongs in
# the startup cost.
#
# Measured, because the sizes are not what they look like (min of 40 cold starts,
# this machine): `import mcp` is 562 ms of a 625 ms start -- 90% of it, and not
# reducible from this repository. Everything `codelearner` itself adds is 44 ms, of
# which 25 ms is `..cli.commands`, and that one is still paid at startup on the real
# entry point because `main()` needs `resolve_index_path` before it can build
# anything. Deferring it here buys the SERVER about 3 ms today. It is done anyway
# because it is where the cost belongs, and because the 25 ms is recoverable the day
# `resolve_index_path` and `INDEX_RELPATH` move out of `cli/` into a leaf: after that
# the only thing between `exec` and `initialize` is the SDK.
if TYPE_CHECKING:
    from mcp.server.context import CallNext, ServerRequestContext

    from ..index import Embedder

SERVER_NAME = "codelearner"

# Ceilings on what one call may ask for. Not politeness: `k` feeds
# `CANDIDATE_MULTIPLIER * k` rows out of FTS5 and a graph expansion seeded from all
# of them, and an agent that passes k=100000 by accident should get a clamped answer
# rather than a server that stops responding.
MAX_K = 100
MAX_STOPS = 100

# Ceilings on one submission. Unlike `MAX_K` these do not bound a transient cost:
# nothing in `assertions.store` deletes, so an oversized assertion is re-loaded and
# its spans re-read and re-hashed on every later `get_symbol` that names its subject,
# for the life of the index. An auditor stored 5,000 spans and a 5MB claim in a
# single call, and every one of those spans is a file read on every subsequent serve.
# 32 spans is more citations than any honest claim about one symbol has -- the
# generator's own ceiling is 12 offers (`generate.pipeline.DEFAULT_MAX_OFFERS`) --
# and 4096 characters is several paragraphs about a single function.
MAX_EVIDENCE_SPANS = 32
MAX_CLAIM_CHARS = 4096

# What one refusal may read, and how much of it may come back. See `_verify_span`:
# the `hash_mismatch` refusal quotes the bytes that are actually there so the agent
# can correct its citation, and that quotation was an unbounded read of any file
# inside the repo root. 4MiB is ~67x the largest file in this repository and still
# small enough that a call cannot be used to move a disk image through an error
# message; 2048 characters is enough to show an agent the symbol it mis-cited.
MAX_CITED_FILE_BYTES = 4 * 1024 * 1024
MAX_OBSERVED_TEXT_CHARS = 2048
TRUNCATION_MARKER = "\n... (truncated)"

# What an assertion is about, when the caller does not say. `store` treats `kind` as
# free text and the schema does not constrain it, so the default lives here where the
# agent-facing contract is described rather than being invented per call site.
DEFAULT_ASSERTION_KIND = "purpose"

# Every way the store can refuse an admission, and the code an agent sees for it. One
# entry per exception class rather than one shared `store_refused`, because the whole
# value of a refusal code is that it names WHICH rule said no: `eval.gate_controls`
# scores a negative control as held only when the code matches the rule its family
# targets, so collapsing these would turn "refused by the right rule" back into
# "refused", which is the measurement failure the negative controls exist to prevent.
# `unknown_subject` deliberately reuses the code `_submit_body` raises above it -- the
# same rule reached through a different door must not look like a different rule.
_STORE_REFUSAL_CODES: dict[type[BaseException], str] = {
    store.EvidenceRequired: "evidence_required",
    store.EmptyClaim: "empty_claim",
    store.InvalidSpan: "invalid_span",
    store.EvidenceUnverifiable: "evidence_unverifiable",
    store.UnknownSubject: "unknown_subject",
    store.EvidenceStale: "evidence_stale",
    store.SpanEscapesRepo: "span_escapes_repo",
}

# Spelled out rather than `tuple(_STORE_REFUSAL_CODES)`, so that a class added to the
# mapping and forgotten here fails as an unhandled ValueError that `_guard` reports as
# `bad_request` -- visible -- rather than being caught and looked up as a KeyError.
_STORE_REFUSALS = (
    store.EvidenceRequired,
    store.EmptyClaim,
    store.InvalidSpan,
    store.EvidenceUnverifiable,
    store.UnknownSubject,
    store.EvidenceStale,
    store.SpanEscapesRepo,
)

# Every `code` this module can put in front of an agent, grouped by what the agent
# should DO about it. Kept here rather than only in the tool docstrings because the
# codes are the branchable half of the contract and the docstrings are prose: a code
# that exists in the code and in no table is one an agent meets for the first time at
# runtime, with nothing to match it against. `test_mcp` asserts this table and the
# `ToolError` codes raised in this file are the same set, so adding a refusal without
# documenting it fails rather than drifts.
#
# RETRY THE SAME CALL -- the condition is transient or one-shot:
#   index_replaced        the index was rebuilt under this server; the next call opens
#                         the new one. Re-read your hashes first: they are one build old.
#
# FIX THE CALL AND RESUBMIT -- something in the arguments is wrong:
#   no_such_symbol        get_symbol: no symbol by that qualname
#   unknown_subject       submit_assertion: the subject qualname is not in the index
#   hash_mismatch         the cited bytes are not what you said; the observed hash and
#                         text come back with it
#   evidence_unverifiable a span carries neither content_hash nor text
#   evidence_required     zero spans
#   empty_claim           no claim text
#   invalid_span          a span the store cannot store (byte_end <= byte_start, etc.)
#   evidence_stale        the store's own re-hash disagrees with the file on disk
#   bad_range             the line range does not exist in that file
#   bad_path              the path contains a NUL byte
#   path_escapes_repo     the cited path resolves outside the repository
#   span_escapes_repo     the store's copy of that rule, reached through write_assertion
#   file_missing          the path is not one this index parsed, or is not readable now
#   file_too_large        the cited file is over MAX_CITED_FILE_BYTES
#   too_many_spans        over MAX_EVIDENCE_SPANS citations in one submission
#   claim_too_long        over MAX_CLAIM_CHARS characters of claim
#   bad_confidence        confidence is not a real number in [0, 1]
#   evidence_unavailable  requested source is stale or unsafe to assemble; re-index or
#                         fix the working tree before asking for source again
#
# TELL THE HUMAN -- no argument of yours will change the answer:
#   no_index              nothing at the index path; someone must run `codelearner index`
#   index_unreadable      the file is not a code-learner index, or will not open
#   index_unbound         the index has no repo root, so citations cannot be re-read
#   schema_mismatch       the index was built by different code; it must be rebuilt
#   incompatible_index    the index predates semantic retrieval and has no assertion
#                         search structures. Refused rather than answered with silence,
#                         because "no claims match" and "this index cannot hold claims"
#                         are different answers and only one of them is about the query
#
# A BUG IN THIS FILE:
#   bad_request           `_guard` caught a ValueError this module should have named
#                         itself. A rising count here is a defect report, not a user error.
ERROR_CODES = frozenset(
    {
        "bad_confidence",
        "bad_path",
        "bad_range",
        "bad_request",
        "claim_too_long",
        "empty_claim",
        "evidence_unavailable",
        "evidence_required",
        "evidence_stale",
        "evidence_unverifiable",
        "file_missing",
        "file_too_large",
        "hash_mismatch",
        "incompatible_index",
        "index_replaced",
        "index_unbound",
        "index_unreadable",
        "invalid_span",
        "no_index",
        "no_such_symbol",
        "path_escapes_repo",
        "schema_mismatch",
        "span_escapes_repo",
        "too_many_spans",
        "unknown_subject",
    }
)

INSTRUCTIONS = """\
GraphRAG over an indexed codebase.

Retrieval is tier-labelled and the labels mean something. T0 is a parsed fact: the
text was read out of the source and nothing had to be decided to reach it. T1 is a
resolved name -- a call site bound to a symbol by a resolver that carries a
confidence below 1 and can be wrong. T2 is an inference, and this index holds one
only if it was submitted with citations that hash-matched the file on disk.

This server does not call a model. If you want an inference stored, you write it:
call submit_assertion with the claim and the exact spans that support it. The spans
are re-read and re-hashed off disk before anything is written. Zero spans is
refused; a span whose bytes have changed is refused, and the refusal tells you what
the file says now. Cite what you actually read -- retrieval hands you the
content_hash of every hit for exactly this purpose.

Every refusal carries a stable `code` worth branching on. One of them concerns the
index rather than your call: if a human rebuilds the index while you are working,
the next call refuses with `index_replaced`. That refusal is not about the call --
it is telling you that every hash, symbol_id and line number you are holding
describes the previous build. Re-run your retrieval before you cite anything.
"""


class ToolError(Exception):
    """A predicted condition, on its way to becoming a structured tool result.

    Carries a stable `code` for the agent to branch on and a `message` for it to
    read. The codes are part of the contract: an agent that retries on
    `hash_mismatch` and gives up on `no_index` is behaving correctly, and it can only
    tell those apart if the distinction survives the trip.
    """

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def payload(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message, **self.detail}}


class EvidenceSpanInput(BaseModel):
    """One citation, as the agent supplies it.

    `line_start`/`line_end` are 1-based and inclusive, and the bytes they name are
    the lines themselves WITHOUT the newline that terminates the last one -- so
    `text` is what you would get by copying those lines out of an editor.

    Exactly one thing is required beyond the location: something to check it against.
    `content_hash` is the sha256 retrieval already handed you for that symbol;
    `text` is the source you read. Either can be compared to the file on disk. A span
    with neither is a pointer, not evidence -- it names a place without asserting
    anything about what is there, so nothing about it can ever be found to be wrong.
    """

    path: str = Field(description="repo-root-relative path, e.g. 'codelearner/db.py'")
    line_start: int = Field(description="1-based first line of the cited range")
    line_end: int = Field(description="1-based last line, inclusive")
    content_hash: str | None = Field(
        default=None,
        description="sha256 of the cited bytes, as returned by search_code/get_symbol",
    )
    text: str | None = Field(
        default=None,
        description="the exact source at those lines, without a trailing newline",
    )


@dataclass
class IndexSource:
    """Where the index is, and the one connection and embedder held open over it.

    Resolution is deferred to the first tool call rather than done at construction.
    A server that refuses to start because the index has not been built yet is a
    server the agent's client marks as failed and stops launching; one that starts
    and says `no_index` on the first call tells the agent what to do about it.

    Everything held here is bound to a FILE, not to a path, and this process outlives
    the file: an index is deleted and rebuilt by one `codelearner index --force` in
    another terminal. `_identity` is what makes that visible from inside; `connect`
    is where it is checked and what it costs to miss it.
    """

    path: Path
    embedder_factory: Any | None = None
    _conn: sqlite3.Connection | None = None
    # `(st_dev, st_ino)` of the file `_conn` was opened on. The only thing that can
    # tell a rebuilt index from the one this server is already holding -- see
    # `_identity_now`, and `connect` for what happens when it moves.
    _identity: tuple[int, int] | None = None
    _embedder: Embedder | None = None
    _embed_checked: bool = False
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    async def run_sync(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run one index operation on this source's sole owning worker.

        SQLite connections and embedders stay on the thread that created them, all
        calls are serialized, and the MCP event loop remains available for protocol
        traffic while retrieval or filesystem work runs.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="codelearner-index"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, partial(fn, *args, **kwargs)
        )

    def close(self) -> None:
        """Close worker-owned state and stop the worker."""
        executor, self._executor = self._executor, None
        if executor is None:
            self._drop()
            return
        executor.submit(self._drop).result()
        executor.shutdown(wait=True)

    def _identity_now(self) -> tuple[int, int] | None:
        """`(st_dev, st_ino)` of the file at `path`, or None if nothing is there.

        The pair identifies the FILE. The path identifies only a name, and a name can
        be made to point at a different file between two tool calls without anything
        observable changing about the name.

        Deliberately not `st_mtime_ns` or `st_size`, which look like the better
        staleness signals and are the wrong ones here: the index is opened in WAL
        mode, so a checkpoint rewrites the main database file, and a server that read
        its own committed assertion as a replacement would refuse every call it made
        after its first write. `meta.indexed_at` is worse still -- reading it means
        querying THIS connection, which is the handle whose staleness is the question,
        so it would report the old file's stamp forever. Only an out-of-band stat of
        the path can see past a cached handle.

        Inode reuse cannot defeat this during the only window in which the answer
        matters. While a connection is open the kernel cannot free the old inode, so
        no file created afterwards can be given the same number: a rebuild is always a
        different pair, never a coincidentally equal one.
        """
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino)

    def _drop(self) -> None:
        """Close and forget everything cached off the file that used to be here.

        `close()`, and not merely rebinding `self._conn = None` -- which is what this
        did, and which leaves sqlite holding the descriptor, and therefore the
        unlinked inode, for the life of the process. A server watching a repository
        that is re-indexed daily leaked one handle and one deleted database file per
        rebuild, and neither is visible from anywhere a human looks.

        The embedder question is reopened as well. `_embed_checked` records an answer
        read out of the OLD index's `meta`, so an index rebuilt with `--embed` would
        otherwise be told "this index has no embeddings" by a server that decided
        that before the embeddings existed and never looked again. The loaded model
        itself is kept: what went stale is the stamp, not the weights, and `embedder`
        reuses them when the new stamp names the same model.
        """
        conn, self._conn, self._identity = self._conn, None, None
        self._embed_checked = False
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                # Already unusable, which is the case this exists to clean up after.
                # Raising here would replace a precise `index_replaced` with a bare
                # transport failure, which is this module's first rule broken.
                pass

    def connect(self) -> sqlite3.Connection:
        """The open connection, or a `ToolError` naming what changed underneath it.

        Two things are re-checked on every call even when a connection is cached, and
        they defend different failures.

        EXISTENCE, because `sqlite3.connect` will happily create an empty file at any
        path -- which is how a typo'd `--index-path` becomes "0 results" instead of
        "no such index", and how a deleted index becomes an empty one.

        IDENTITY, because existence alone is a check that `codelearner index --force`
        walks straight through. That command deletes the index file and builds a new
        one, so the path exists again the moment it finishes -- and the cached
        connection, still bound to the unlinked inode, kept being returned. An auditor
        submitted an assertion through a live server after a rebuild and was told
        `ok: true, accepted: true, servable: true` for a row written into a file
        nobody could open; zero assertions survived on disk. Every READ had the same
        shape and made less noise: hits, hashes and line numbers derived from the
        previous build, served as though they were current. This is not an exotic
        sequence. It is the second morning -- an agent session left open in the
        editor, a human re-indexing in a terminal.

        The first call after a swap fails rather than transparently reconnecting, and
        the choice is deliberate. A silent reconnect would return correct data about a
        codebase the agent is not holding: every `content_hash` it collected before
        the rebuild was published by the old index, and a rebuild happens because the
        source changed. The agent would carry on citing hashes that no longer describe
        anything, planning against symbol ids that now number different symbols, and
        the only rule that would catch any of it is the citation gate -- which reports
        `hash_mismatch`, i.e. "you cited something that changed", when the truth is
        "everything you know is one build old". Succeeding quietly is precisely the
        bug being fixed here, in a politer form. The refusal costs one retry and is
        self-clearing: the connection is dropped on the way out, so the very next call
        opens the new file and answers normally.
        """
        identity = self._identity_now()
        if identity is None:
            self._drop()
            raise ToolError(
                "no_index",
                f"no index at {self.path}. Build one with `codelearner index <repo>`, "
                "or start this server with --index-path pointing at an existing one.",
                index=str(self.path),
            )
        if self._conn is not None and identity != self._identity:
            self._drop()
            raise ToolError(
                "index_replaced",
                f"the index at {self.path} has been rebuilt since this server opened "
                "it -- `codelearner index --force` replaces the file rather than "
                "updating it. Nothing was read from the old one. Every content_hash, "
                "symbol_id and line number you are holding came from the previous "
                "build and may no longer describe anything: re-run your retrieval and "
                "cite what it returns now. Retrying this call opens the new index.",
                index=str(self.path),
            )
        if self._conn is None:
            try:
                self._conn = db.connect(self.path)
            except sqlite3.Error as exc:
                raise ToolError(
                    "index_unreadable",
                    f"could not open the index at {self.path}: {exc}",
                    index=str(self.path),
                ) from exc
            self._identity = identity
        return self._conn

    def replaced(self) -> bool:
        """Whether the file at `path` is no longer the one `_conn` is bound to.

        The after-the-fact half of `connect`'s identity check, for a caller that has
        already written something. `connect` runs once, before a tool body starts; a
        rebuild that lands while that body is running passes the check and the write
        commits into the deleted file regardless. Nothing available here closes that
        window -- sqlite cannot be asked whether the file under an open handle has
        been unlinked, and there is no lock shared with the `codelearner index`
        process to take against it.

        What this closes is the REPORT of it. A lost write returned as `ok: true` and
        a lost write returned as `index_replaced` lose the same row; only the second
        makes the agent submit it again. Nothing else in this process will ever notice
        that the row is gone.

        Drops the cached connection when the answer is yes, so one swap produces one
        refusal and the next call opens what is actually at the path.
        """
        if self._conn is None:
            return False
        if self._identity_now() == self._identity:
            return False
        self._drop()
        return True

    def repo_root(self, conn: sqlite3.Connection) -> Path:
        root = db.stored_repo_root(conn)
        if root is None:
            raise ToolError(
                "index_unbound",
                f"the index at {self.path} is not bound to a repo root, so cited "
                "spans cannot be re-read from disk. Re-index the repository.",
                index=str(self.path),
            )
        return Path(root)

    def embedder(self, conn: sqlite3.Connection) -> tuple[Embedder | None, list[str]]:
        """The embedder matching this index's vectors, or None plus why not.

        Loaded once and kept. The model is ~1.2GB of weights and tens of seconds;
        rebuilding it per tool call would make dense retrieval cost more than the
        answer is worth, and this process outlives many calls.

        The CHECK is cheaper to redo than the load, and they expire differently. A
        rebuilt index carries a new `meta` stamp -- possibly its first, if the rebuild
        was the one that added `--embed` -- so `_drop` clears `_embed_checked` and the
        stamp is read again off the new file. The weights survive that, because a
        rebuild does not change what `Qwen3-Embedding-0.6B` is, and reloading it to
        learn that would charge tens of seconds for an answer already in memory.
        """
        if self._embed_checked:
            return self._embedder, []
        self._embed_checked = True
        stored = stored_embed_model(conn)
        if stored is not None and self._embedder is not None and self._embedder.name == stored:
            # A rebuild that kept the same model. Nothing to reload.
            return self._embedder, []
        # Every path below reaches a different index from the one `_embedder` was
        # loaded for, so the old model is forgotten first. Leaving it set would let a
        # rebuild that DROPPED its embeddings keep answering dense queries out of the
        # previous index's model while the note explaining their absence went unsent.
        self._embedder = None
        if stored is None:
            # Not an error. Lexical and graph still answer; degrading loudly beats
            # failing, and the note tells the agent why dense is absent.
            return None, [
                "dense retrieval unavailable: this index has no embeddings. Build "
                "them with `codelearner index <repo> --embed --force`."
            ]
        try:
            candidate = (self.embedder_factory or _default_embedder)(stored)
        except Exception as exc:  # noqa: BLE001 - a missing GPU is not a tool failure
            return None, [f"dense retrieval unavailable: could not load {stored!r} ({exc})."]
        if candidate.name != stored:
            # Vectors from two models are not comparable. Querying anyway returns
            # results that look plausible and mean nothing.
            return None, [
                f"dense retrieval disabled: this index was embedded with {stored!r} "
                f"but the loaded model is {candidate.name!r}, and vectors from two "
                "models are not comparable."
            ]
        self._embedder = candidate
        return candidate, []


def _default_embedder(model_name: str) -> Embedder:
    """Build the real embedder. Imported late -- the import pulls in torch.

    Same loader the CLI uses. Deliberately not a second one: an MCP server and a
    shell command that disagree about which model reads an index would produce two
    different rankings from one file, and neither would look wrong.
    """
    from ..index import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name)


def _guard(source: IndexSource, fn: Any, /, **kwargs: Any) -> dict[str, Any]:
    """Run a tool body with a connection, converting every predicted failure.

    `sqlite3.Error` is caught here as well as `ToolError`: a file that opens but is
    not a code-learner index fails on the first query rather than on connect, and
    "no such table: symbols" is a condition with a remedy, not a bug to report.

    `ValueError` is caught for a weaker reason, and deliberately as a backstop rather
    than as a contract. The standard library raises it for input this module should
    have refused itself -- a path containing a NUL byte reaches `Path.resolve` and
    comes back as `ValueError: embedded null byte`, which crossed this boundary as a
    traceback and told the agent the tool was broken. The specific case is now
    rejected up front in `_verify_span` as `bad_path`; this clause exists so that the
    next such value, whatever it turns out to be, is still an answer rather than a
    crash. The generic code is honest about that: this is the module failing to name
    a condition, not a condition with a known remedy. A rising count of `bad_request`
    is a bug report about this file.

    `SchemaVersionError` and `RepoRootMismatchError` are caught for the strongest
    reason of the three. `db.connect` refuses an index whose stamp is not the current
    `SCHEMA_VERSION`, and that refusal is a `RuntimeError` -- not a `sqlite3.Error`,
    not a `ToolError` -- so it went straight through this function and out into the
    transport, which is the failure this module's first rule names. It is also the
    most predictable failure in the whole design: the stamp has moved five times, an
    agent's client keeps this server running across the re-index that follows, and the
    first tool call afterwards met a version check with nothing to catch it. The
    remedy travels with the code so the agent can tell its human what to type.

    Positional-only up front so that a tool body may itself take a parameter named
    `source` or `fn` without colliding with this frame's own arguments.
    """
    try:
        conn = source.connect()
        return fn(conn, source, **kwargs)
    except ToolError as exc:
        return exc.payload()
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        from ..cli.commands import REBUILD_ADVICE

        return ToolError(
            "schema_mismatch",
            f"{exc} {REBUILD_ADVICE}",
            index=str(source.path),
        ).payload()
    except ValueError as exc:
        return ToolError(
            "bad_request",
            f"an argument to this call could not be used ({exc}). Check the paths and "
            "numbers you passed; this tool takes repo-root-relative paths and 1-based "
            "line numbers.",
        ).payload()
    except sqlite3.Error as exc:
        return ToolError(
            "index_unreadable",
            f"{source.path} does not look like a code-learner index ({exc}). "
            "Point --index-path at the right file, or re-index.",
            index=str(source.path),
        ).payload()


# ---------------------------------------------------------------------------
# evidence verification -- the gate
# ---------------------------------------------------------------------------

def _line_bytes(source: bytes, line_start: int, line_end: int) -> tuple[int, int]:
    """Byte range of an inclusive 1-based line range, excluding the final newline.

    Excluding it is what makes `text` round-trip: an agent that copies lines 10-12
    out of a file has three lines and no trailing newline, and a citation format that
    silently required one would reject correct evidence for a reason nobody could
    see.
    """
    starts = [0]
    idx = source.find(b"\n")
    while idx != -1:
        starts.append(idx + 1)
        idx = source.find(b"\n", idx + 1)
    # A file ending in a newline has an empty trailing entry that is not a line.
    line_count = len(starts) - 1 if source.endswith(b"\n") else len(starts)
    if line_start < 1 or line_end < line_start or line_end > line_count:
        raise ToolError(
            "bad_range",
            f"lines {line_start}-{line_end} are not a valid range in a "
            f"{line_count}-line file.",
            line_count=line_count,
        )
    byte_start = starts[line_start - 1]
    byte_end = starts[line_end] - 1 if line_end < len(starts) else len(source)
    return byte_start, byte_end


def _symbol_bytes_at(
    conn: sqlite3.Connection, path: str, line_start: int, line_end: int
) -> list[tuple[int, int]]:
    """Byte ranges of any indexed symbols occupying exactly these lines.

    A symbol's stored bytes are NOT its lines' bytes, and the gap is not rare. The
    parser records the symbol node: it begins at `def`, not in the indentation before
    it, and for a decorated symbol it begins at the `@`, because the decorators are
    part of what the symbol is. A module's span runs to the last byte of the file,
    which is one line past the last line anything is written on.

    The proportion of this repository's symbols where the two readings disagree is
    NOT STATED HERE PENDING RE-MEASUREMENT (WP8). The figure this docstring used to
    quote -- "36 modules, 36 methods, 11 functions and 2 classes out of 383 symbols
    -- around 15%" -- is one measurement reported three incompatible ways across this
    file and the README (15% / 22.2% / 25.5%; 85 of 383 is 22.2%), so at most one of
    them was ever right and nothing in the tree says which. The decorator span change
    moves the true value again. Do not quote a number from here until WP8 replaces
    this paragraph with a measured one.

    That the gap exists at all is what matters here, because the hash `search_code`
    and `get_symbol` hand back is the hash of the SYMBOL's bytes. Checking a citation
    only against the lines' bytes would reject the exact hash this server just
    published, for a substantial minority of symbols, with a message accusing the
    agent of citing something that had changed. Looking the symbol up is what closes
    that gap -- and it is the same reason `store.span_for_symbol` exists.
    """
    return [
        (int(r["byte_start"]), int(r["byte_end"]))
        for r in conn.execute(
            "SELECT s.byte_start, s.byte_end FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE f.path = ? AND s.line_start = ? AND s.line_end = ? ORDER BY s.id",
            (path, line_start, line_end),
        )
    ]


def _verify_span(
    conn: sqlite3.Connection, root: Path, raw: EvidenceSpanInput
) -> store.EvidenceSpan:
    """Turn one submitted citation into a stored span, or refuse it.

    The whole inversion rests here. Everything this function checks is arithmetic --
    the file exists, the lines exist, sha256 of those bytes equals the sha256 that
    was cited -- so there is nothing in it for a confident model to argue with. What
    it CANNOT check is whether the span supports the claim; that is what the judge in
    `store.record_verdict` is for. This is the cheap gate, and it is the one that
    stops a fabricated citation before it becomes a row.

    There are two honest readings of "lines 10-14 of this file" -- the symbol that
    occupies them, and the whole lines themselves -- and they hash differently (see
    `_symbol_bytes_at`). Both are built and both are re-hashed off disk; the cited
    hash may match either. Accepting both is not a loosening of the gate. Every
    candidate is read from the file as it is right now, so a stale or invented hash
    still matches nothing. What it removes is a false rejection, which is the more
    dangerous failure here: an agent told its correct citation is wrong learns that
    the gate is noise.

    Everything between the path check and the read is about the refusal rather than
    the admission, because the refusal is the dangerous half. This function reads a
    file off disk and quotes it back inside `hash_mismatch`, so every restriction on
    what it will open is a restriction on what a deliberately wrong `content_hash`
    can be made to print. See the `files` lookup and the size ceiling below.
    """
    if "\x00" in raw.path:
        # Rejected here, by name, rather than left to `Path.resolve` -- which raises
        # `ValueError: embedded null byte`, a bare traceback across the MCP boundary
        # and a direct breach of this module's first rule. `_guard` now catches
        # ValueError as well, but a caller that gets `bad_request` learns only that
        # something was wrong somewhere; one that gets `bad_path` with the offending
        # field named can fix it.
        raise ToolError(
            "bad_path",
            "the citation path contains a NUL byte, which no file on disk can. "
            "Pass the `path` exactly as search_code or get_symbol returned it.",
            path=raw.path.replace("\x00", "\\x00"),
        )
    target = (root / raw.path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ToolError(
            "path_escapes_repo",
            f"{raw.path!r} resolves outside the indexed repository. Citations must "
            "be repo-root-relative paths.",
            path=raw.path,
        )

    # The read is restricted to files this index parsed, and the reason is not
    # tidiness. Before this check the gate refused paths that escaped the repo root
    # and read ANY file inside it, indexed or not, then returned the decoded bytes as
    # `observed_text` in the refusal below. An auditor recovered `.env` secrets and an
    # SSH private key by submitting a deliberately wrong `content_hash` and walking
    # the line ranges. The gate was never defeated; the refusal was the exfiltration
    # channel, and a channel is closed by narrowing what feeds it.
    #
    # `file_missing`, and NOT a new code, which is the more interesting half of the
    # choice. A distinct code -- `file_not_indexed` for `.env`, `file_missing` for a
    # path that is simply absent -- would tell an honest agent slightly more, and
    # would tell a probing one exactly what it came for: run the two cases and the
    # difference in the code IS the answer to "does this file exist on disk". The
    # oracle would survive the fix in a smaller form. One code, raised from one place,
    # for every path this index did not parse, means absent, present-but-ignored, and
    # `.env` are indistinguishable from outside. The remaining `file_missing` raises
    # below can afford to say more, because by then the path is one this index parsed
    # and its existence is already public through search_code.
    #
    # The message carries the remedy, which is where the honest agent's information
    # actually lives: cite what retrieval handed you, or re-index. That agent cites
    # what it retrieved and never sees this refusal at all -- which is why restricting
    # the read to indexed files costs nothing real.
    if conn.execute("SELECT 1 FROM files WHERE path = ?", (raw.path,)).fetchone() is None:
        raise ToolError(
            "file_missing",
            f"{raw.path!r} is not a file in this index, so there is nothing here to "
            "cite. Cite a path that search_code or get_symbol returned. If the file is "
            "part of this repository but was added or renamed after the last index "
            "run, re-index it first -- this tool reads only what it has parsed.",
            path=raw.path,
        )

    if not target.is_file():
        # Checked before the read, not inferred from an OSError afterwards. A missing
        # file and a directory both fail loudly; a FIFO does not fail at all --
        # `read_bytes` blocks until another process opens the write end, and this
        # server is single-threaded, so one citation of a named pipe inside the repo
        # stops it answering anything, forever, without raising or logging.
        raise ToolError(
            "file_missing",
            f"{raw.path!r} is not a readable file in the repository right now. Cite a "
            "file that exists; if it was deleted or replaced, re-index.",
            path=raw.path,
        )
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ToolError(
            "file_missing",
            f"cannot stat {raw.path!r} ({exc}). Cite a file that exists in the "
            "indexed repository.",
            path=raw.path,
        ) from exc
    if size > MAX_CITED_FILE_BYTES:
        # Refused on `st_size` before `read_bytes`, which is the only order that
        # helps: the cost being avoided is the read itself. One call against a 200MB
        # file returned a 209,715,200-character payload at ~479MB peak RSS -- the
        # whole file, decoded, inside an error message.
        raise ToolError(
            "file_too_large",
            f"{raw.path!r} is {size} bytes, over the {MAX_CITED_FILE_BYTES}-byte "
            "ceiling on a citable file. Nothing this large is source a claim can rest "
            "on; cite the symbol you actually read.",
            path=raw.path,
            size=size,
            limit=MAX_CITED_FILE_BYTES,
        )
    try:
        source = target.read_bytes()
    except OSError as exc:
        raise ToolError(
            "file_missing",
            f"cannot read {raw.path!r} ({exc}). Cite a file that exists in the "
            "indexed repository.",
            path=raw.path,
        ) from exc

    ranges = _symbol_bytes_at(conn, raw.path, raw.line_start, raw.line_end)
    line_range: tuple[int, int] | None = None
    range_error: ToolError | None = None
    try:
        line_range = _line_bytes(source, raw.line_start, raw.line_end)
    except ToolError as exc:
        # Not fatal on its own: a module's `line_end` is legitimately one past the
        # last written line, so a citation of it is a valid symbol span and an
        # invalid line span at the same time.
        range_error = exc
    if line_range is not None and line_range not in ranges:
        ranges.append(line_range)

    candidates: list[store.EvidenceSpan] = []
    for byte_start, byte_end in ranges:
        try:
            # `span_for` re-reads and re-hashes off disk. Nothing here trusts the
            # `content_hash` column, which records what the file said at index time
            # and is exactly as stale as the index.
            candidates.append(store.span_for(root, raw.path, byte_start, byte_end))
        except ValueError:
            # An empty range -- a blank line, most likely. It would hash to something
            # stable and verify forever while pointing at nothing.
            continue
    if not candidates:
        raise ToolError(
            "bad_range",
            range_error.message
            if range_error is not None
            else f"lines {raw.line_start}-{raw.line_end} of {raw.path!r} are empty, so "
            "there are no bytes to cite.",
            path=raw.path,
        )

    if raw.content_hash is not None:
        cited = raw.content_hash
    elif raw.text is not None:
        from ..ingest.types import content_hash

        cited = content_hash(raw.text.encode())
    else:
        raise ToolError(
            "evidence_unverifiable",
            f"the span {candidates[-1].citation} carries neither content_hash nor "
            "text, so there is nothing to check it against. Pass the content_hash "
            "that search_code or get_symbol returned for this symbol, or the exact "
            "source text you read at those lines.",
            path=raw.path,
            line_start=raw.line_start,
            line_end=raw.line_end,
        )

    for span in candidates:
        if cited == span.content_hash:
            return span

    # Report against the whole-lines reading when there is one: it is what a human
    # opening the file at those lines would see, and the point of this message is to
    # let the agent correct itself rather than guess again.
    shown = candidates[-1]
    # Quoting the bytes is what makes this refusal correctable rather than merely
    # correct -- and it is also the thing an attacker submits a wrong hash in order to
    # read. Bounded, with the cut marked, because a truncation an agent cannot see is
    # a lie about what the file says. The bound is on the answer; the bounds on what
    # can be opened at all are above, and they are the ones doing the security work.
    observed_text = source[shown.byte_start:shown.byte_end].decode("utf-8", "replace")
    if len(observed_text) > MAX_OBSERVED_TEXT_CHARS:
        observed_text = observed_text[:MAX_OBSERVED_TEXT_CHARS] + TRUNCATION_MARKER
    raise ToolError(
        "hash_mismatch",
        f"the bytes at {shown.citation} do not match what you cited. The file has "
        "changed since you read it, or the citation points somewhere else. Re-read "
        "those lines and resubmit.",
        path=raw.path,
        citation=shown.citation,
        cited_hash=cited,
        observed_hash=shown.content_hash,
        observed_hashes=[s.content_hash for s in candidates],
        observed_text=observed_text,
    )


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def _symbol_hashes(conn: sqlite3.Connection, symbol_ids: list[int]) -> dict[int, str]:
    """The stored sha256 of each symbol's source bytes.

    Attached to every hit so that citing what you just retrieved needs no second
    call and no local hashing. The retrieval -> citation -> gate loop only closes if
    the hash travels with the result.
    """
    if not symbol_ids:
        return {}
    placeholders = ",".join("?" * len(symbol_ids))
    rows = conn.execute(
        f"SELECT id, content_hash FROM symbols WHERE id IN ({placeholders})",  # noqa: S608
        tuple(symbol_ids),
    )
    return {int(r["id"]): str(r["content_hash"]) for r in rows}


def _span_json(span: store.EvidenceSpan) -> dict[str, Any]:
    return {
        "citation": span.citation,
        "path": span.path,
        "line_start": span.line_start,
        "line_end": span.line_end,
        "content_hash": span.content_hash,
    }


def _assertion_json(assertion: store.Assertion) -> dict[str, Any]:
    return {
        "id": assertion.id,
        "tier": "T2",
        "kind": assertion.kind,
        "claim": assertion.claim,
        "generator": assertion.generator,
        "confidence": assertion.confidence,
        "created_at": assertion.created_at,
        "evidence": [_span_json(s) for s in assertion.spans],
    }


def _servable_for(
    conn: sqlite3.Connection, source: IndexSource, qualname: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assertions about `qualname` that still verify, plus any note about why not.

    Every call re-reads and re-hashes the cited bytes -- that is `servable_assertions`
    doing its job, not a cache miss. A claim whose evidence moved is expired on the
    way past rather than served one last time.
    """
    try:
        root = source.repo_root(conn)
    except ToolError as exc:
        return [], [exc.message]
    return [_assertion_json(a) for a in store.servable_assertions(conn, root, subject_qualname=qualname)], []


# ---------------------------------------------------------------------------
# tool bodies
# ---------------------------------------------------------------------------

def _search_body(
    conn: sqlite3.Connection,
    source: IndexSource,
    query: str,
    k: int,
    facts_only: bool,
    include_source: bool,
    evidence_budget: int,
    include_assertions: bool = True,
    debug_scores: bool = False,
) -> dict[str, Any]:
    k = max(1, min(int(k), MAX_K))
    embedder, notes = source.embedder(conn)

    # Serving a claim means re-reading the bytes it cites, so semantic retrieval
    # needs the repository and not only the index. Without one the server answers
    # from source alone and says so, rather than serving unverified claims.
    repo_root: Path | None = None
    try:
        repo_root = source.repo_root(conn)
    except ToolError as exc:
        if include_source:
            raise
        if include_assertions:
            include_assertions = False
            notes = [*notes, exc.message]

    policy = PRODUCTION_POLICY
    if facts_only:
        # Applied before fusion so the vacated slots refill with source. Filtering
        # the finished page would hand a caller who asked for facts a short one.
        policy = replace(policy, max_tier=TIER_RESOLVED)
    try:
        result = search_candidates(
            conn,
            repo_root or Path("."),
            query,
            k=k,
            policy=policy,
            embedder=embedder,
            use_dense=embedder is not None,
            use_assertions=include_assertions and repo_root is not None,
            debug=debug_scores,
        )
    except AssertionSearchUnavailable as exc:
        raise ToolError("incompatible_index", str(exc)) from exc
    candidates = list(result.candidates)
    hashes = _symbol_hashes(
        conn,
        [c.symbol_id for c in candidates if isinstance(c, SourceCandidate)],
    )
    payload = {
        "ok": True,
        "query": query,
        "k": k,
        "facts_only": facts_only,
        "include_assertions": include_assertions,
        "count": len(candidates),
        "notes": notes,
        # `candidate_json` is the CLI's shape, reused verbatim so the two surfaces
        # cannot disagree about a tier or about what a claim is. `content_hash` is
        # the one addition, and only on source: an agent that wants to cite a symbol
        # needs it, and a human reading a terminal does not. A claim already carries
        # the hash of every range it cites.
        "hits": [
            dict(
                candidate_json(candidate, rank, debug=debug_scores),
                content_hash=hashes.get(candidate.symbol_id),
            )
            if isinstance(candidate, SourceCandidate)
            else candidate_json(candidate, rank, debug=debug_scores)
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }
    if include_source:
        from ..evidence import EvidenceError, assemble_candidate_evidence

        try:
            evidence = assemble_candidate_evidence(
                conn,
                source.repo_root(conn),
                candidates,
                budget_bytes=evidence_budget,
            )
        except EvidenceError as exc:
            raise ToolError(
                "evidence_unavailable",
                exc.message,
                evidence_code=exc.code,
                symbol_id=exc.symbol_id,
            ) from exc
        payload["evidence"] = evidence.as_json()
    return payload


def _edge_json(row: sqlite3.Row, other: str) -> dict[str, Any]:
    return {
        "qualname": row[f"{other}_qualname"],
        "symbol_id": row[f"{other}_id"],
        "path": row["path"],
        "line": row["line"],
        "kind": row["edge_kind"],
        "confidence": row["confidence"],
        "resolver": row["resolver"],
        "tier": "T1",
    }


_CALLEE_SQL = """
SELECT e.kind AS edge_kind, e.line, e.confidence, e.resolver,
       d.id AS dst_id, d.qualname AS dst_qualname, f.path
FROM edges e
JOIN symbols d ON d.id = e.dst_symbol_id
JOIN files f ON f.id = d.file_id
WHERE e.src_symbol_id = ? AND e.dst_symbol_id IS NOT NULL
ORDER BY e.line
"""

_CALLER_SQL = """
SELECT e.kind AS edge_kind, e.line, e.confidence, e.resolver,
       s.id AS src_id, s.qualname AS src_qualname, f.path
FROM edges e
JOIN symbols s ON s.id = e.src_symbol_id
JOIN files f ON f.id = s.file_id
WHERE e.dst_symbol_id = ?
ORDER BY s.qualname, e.line
"""


def _get_symbol_body(
    conn: sqlite3.Connection, source: IndexSource, qualname: str, facts_only: bool
) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT s.id, s.kind, s.name, s.qualname, s.line_start, s.line_end, "
        "       s.content_hash, s.docstring, s.signature, f.path, f.is_test "
        "FROM symbols s JOIN files f ON f.id = s.file_id "
        "WHERE s.qualname = ? ORDER BY s.id",
        (qualname,),
    ).fetchall()
    if not rows:
        raise ToolError(
            "no_such_symbol",
            f"no symbol named {qualname!r} in this index. Qualnames are dotted paths "
            "from the module root, e.g. 'codelearner.db.init_db' -- use search_code "
            "to find the exact one.",
            qualname=qualname,
        )
    row = rows[0]
    symbol_id = int(row["id"])
    assertions, notes = _servable_for(conn, source, qualname)
    # The one place in this server where `facts_only` is not inert. Retrieval has no
    # tier-2 modality, so the flag on `search_code` filters a list that never contains
    # anything above T1; the assertions below are the only T2 content the server
    # returns anywhere, so this is where the promise "parsed facts and resolved names,
    # nothing asserted" either holds or does not.
    #
    # The claims are still fetched and still re-verified before being dropped. That
    # costs a re-hash of the cited bytes for a caller who asked not to see them, and
    # it is deliberate: `notes` reports evidence that has moved, and a caller running
    # facts-only should not get a quieter answer about the state of the index than one
    # who did not. Suppressed rather than never-looked-for.
    withheld = 0
    if facts_only:
        withheld = len(assertions)
        assertions = []
        if withheld:
            notes = [
                *notes,
                f"facts_only: {withheld} tier-2 assertion(s) about this symbol were "
                "withheld. They exist and they verify; pass facts_only=false to read "
                "them.",
            ]
    unresolved = conn.execute(
        "SELECT kind, dst_name, line FROM edges "
        "WHERE src_symbol_id = ? AND dst_symbol_id IS NULL ORDER BY line",
        (symbol_id,),
    ).fetchall()
    return {
        "ok": True,
        "notes": notes,
        # Echoed, like `search_code` does, so a caller reading a stored response can
        # tell an empty `assertions` list that means "none exist" from one that means
        # "you asked not to see them". `assertions_withheld` says which.
        "facts_only": facts_only,
        "assertions_withheld": withheld,
        "symbol": {
            "tier": "T0",
            "symbol_id": symbol_id,
            "qualname": row["qualname"],
            "name": row["name"],
            "kind": row["kind"],
            "path": row["path"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "signature": row["signature"],
            "docstring": row["docstring"],
            "is_test": bool(row["is_test"]),
            "content_hash": row["content_hash"],
        },
        # Qualname is not unique in the schema -- two branches of a package can each
        # define `utils.helper`. Reported rather than silently collapsed, so a caller
        # reading callers/callees knows they belong to the first match only.
        "duplicate_qualnames": len(rows) - 1,
        "callees": [_edge_json(r, "dst") for r in conn.execute(_CALLEE_SQL, (symbol_id,))],
        "callers": [_edge_json(r, "src") for r in conn.execute(_CALLER_SQL, (symbol_id,))],
        # The tier-0 half of the same question: references written at a call site
        # that no resolver could bind. Mostly stdlib and third-party calls, and
        # omitting them would make a symbol that calls only `json.dumps` look inert.
        "unresolved_calls": [
            {"tier": "T0", "kind": r["kind"], "name": r["dst_name"], "line": r["line"]}
            for r in unresolved
        ],
        "assertions": assertions,
    }


def _reading_path_body(
    conn: sqlite3.Connection, source: IndexSource, topic: str, limit: int
) -> dict[str, Any]:
    from ..onboard import build_reading_path

    limit = max(1, min(int(limit), MAX_STOPS))
    embedder: Embedder | None = None
    notes: list[str] = []
    if topic:
        # Only a topic-seeded tour retrieves; the repo-wide tour is pure centrality
        # and would pay for a model it never asks a question of.
        embedder, notes = source.embedder(conn)
    path = build_reading_path(conn, topic=topic or None, limit=limit, embedder=embedder)
    return {
        "ok": True,
        "topic": topic or None,
        "notes": notes,
        "repo_root": path.repo_root,
        "graph_symbols": path.graph_symbols,
        "graph_edges": path.graph_edges,
        "tiers": path.tiers,
        "cycles": [list(c) for c in path.cycles],
        "count": len(path.stops),
        "stops": [
            {
                "order": stop.order,
                "qualname": stop.qualname,
                "kind": stop.kind,
                "path": stop.path,
                "line_start": stop.line_start,
                "line_end": stop.line_end,
                "location": stop.location,
                "signature": stop.signature,
                "summary": stop.summary,
                "depth_tier": stop.tier,
                "centrality_rank": stop.centrality_rank,
                "reason": stop.reason,
                "read_before_this": list(stop.calls_here),
                "read_after_this": list(stop.called_by_here),
                "callers_repo": stop.callers_repo,
                "cycle": list(stop.cycle),
                "recursive": stop.recursive,
            }
            for stop in path.stops
        ],
    }


def _submit_body(
    conn: sqlite3.Connection,
    source: IndexSource,
    subject_qualname: str,
    claim: str,
    evidence_spans: list[EvidenceSpanInput],
    kind: str,
    generator: str | None,
    confidence: float | None,
) -> dict[str, Any]:
    root = source.repo_root(conn)

    # Size first, before the index is touched and long before a file is opened. These
    # are not rules about whether a claim is true -- they are the only defence against
    # a submission whose cost outlives it. Nothing in `assertions.store` deletes, so a
    # 5,000-span assertion is 5,000 file reads on every later `get_symbol` naming its
    # subject, and a 5MB claim is 5MB in every response that serves it, for as long as
    # the index exists. There is no repair short of rebuilding.
    if len(evidence_spans) > MAX_EVIDENCE_SPANS:
        raise ToolError(
            "too_many_spans",
            f"{len(evidence_spans)} evidence spans, over the limit of "
            f"{MAX_EVIDENCE_SPANS}. A claim about one symbol that needs more citations "
            "than that is really several claims -- submit them separately, each "
            "standing on the spans that actually support it.",
            spans=len(evidence_spans),
            limit=MAX_EVIDENCE_SPANS,
        )
    if len(claim) > MAX_CLAIM_CHARS:
        raise ToolError(
            "claim_too_long",
            f"the claim is {len(claim)} characters, over the limit of "
            f"{MAX_CLAIM_CHARS}. A stored claim is read back next to the code it is "
            "about; anything longer than a few paragraphs is a document, and this is "
            "not where a document goes.",
            length=len(claim),
            limit=MAX_CLAIM_CHARS,
        )
    # Bounded here in Python and NOT by a CHECK constraint on `assertions.confidence`,
    # which is where it belongs. Adding the constraint is a DDL change, and this
    # project's schema policy is refuse-and-rotate: every bump makes every existing
    # index refuse to open until it is rebuilt. Bumping twice in one remediation would
    # charge that twice, so the constraint rides along with the WP8 v6 change and this
    # check holds the line until then. When it lands, this stays -- a caller deserves
    # `bad_confidence` rather than an IntegrityError, and the store deserves a rule
    # that holds for library callers who never come through this tool.
    #
    # `isfinite` is the half that matters: `1e308` is merely absurd, but `inf` and
    # `nan` compare false against every threshold, so a claim carrying one is
    # permanently neither above nor below any confidence filter later written.
    if confidence is not None and not (math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
        raise ToolError(
            "bad_confidence",
            f"confidence must be a real number between 0 and 1, not {confidence!r}. "
            "It is read as a probability that this claim is right.",
            confidence=repr(confidence),
        )

    # The cheapest rule, and the one the negative controls in `eval.gate_controls`
    # found missing: a claim whose subject is not in the index is refused before any
    # file is read. Every span can hash-match perfectly while the qualname they are
    # attached to was invented -- `core.frobnicate_nothing`, cited with real bytes out
    # of core.py. Verified spans made that submission indistinguishable from a good
    # one: stored `active`, reported `servable`, and unreachable forever, because
    # `get_symbol` answers `no_such_symbol` for the only name that would find it. An
    # inference no reader can reach is the accountability failure the zero-evidence
    # rule exists to prevent, arrived at through the other door.
    subject = conn.execute(
        "SELECT id FROM symbols WHERE qualname = ? ORDER BY id", (subject_qualname,)
    ).fetchone()
    if subject is None:
        raise ToolError(
            "unknown_subject",
            f"no symbol named {subject_qualname!r} in this index, so there is nothing "
            "for this claim to be about. Qualnames are dotted paths from the module "
            "root -- use search_code or get_symbol to find the exact one. Only what "
            "this index parsed can be the subject of a stored claim.",
            subject_qualname=subject_qualname,
        )

    # Verified before anything is written, and one failure stops the whole
    # submission. Admitting the spans that happened to verify would leave a claim
    # standing on a subset of the evidence its author thought it had.
    spans = [_verify_span(conn, root, raw) for raw in evidence_spans]

    try:
        assertion_id = store.write_assertion(
            conn,
            subject_qualname=subject_qualname,
            kind=kind or DEFAULT_ASSERTION_KIND,
            claim=claim,
            spans=spans,
            subject_symbol_id=int(subject["id"]),
            generator=generator,
            confidence=confidence,
            repo_root=root,
        )
    except _STORE_REFUSALS as exc:
        # Translation, not enforcement. Every rule above -- the subject lookup, the
        # per-span verification -- is enforced again inside `write_assertion`, and
        # THAT is the copy that decides. This tool runs its own first because it can
        # say more about what to fix (which field, which bytes are really there,
        # which of the two honest readings of a line range was tried), and a refusal
        # an agent cannot act on teaches it that the gate is noise. The order matters
        # in exactly one direction: the richer check may run first, and the store's
        # must run regardless. `codelearner learn` and every library caller reach the
        # store without passing here at all, which is the whole reason the rules
        # moved.
        raise ToolError(
            _STORE_REFUSAL_CODES[type(exc)],
            str(exc),
            subject_qualname=subject_qualname,
        ) from exc

    # Asked again, on the far side of the write. `connect` checked identity before
    # this body started; a `codelearner index --force` that lands in between passes
    # that check and the commit above goes into the deleted file, where nothing will
    # ever read it again. The window cannot be closed from here (see `replaced`), so
    # what is refused is the CLAIM OF SUCCESS: the one report an agent will not act
    # on is `ok: true`.
    if source.replaced():
        raise ToolError(
            "index_replaced",
            f"the index at {source.path} was rebuilt while this submission was being "
            "written, so the assertion was committed to the "
            "file that was deleted rather than to the one at this path, and nothing "
            "here can recover it. Re-run your retrieval against the new index, check "
            "the hashes you were holding, and submit again. A duplicate claim is "
            "recoverable; a silently lost one is not.",
            index=str(source.path),
            subject_qualname=subject_qualname,
        )

    return {
        "ok": True,
        "accepted": True,
        "assertion_id": assertion_id,
        "subject_qualname": subject_qualname,
        "subject_symbol_id": int(subject["id"]),
        "kind": kind or DEFAULT_ASSERTION_KIND,
        "tier": "T2",
        # Re-read and re-hashed a second time, by the same function that will decide
        # whether to serve this claim tomorrow. If that ever disagrees with the gate
        # that just admitted it, the caller finds out now rather than from silence.
        "servable": store.is_servable(conn, assertion_id, root),
        "evidence": [_span_json(s) for s in spans],
    }


def _stats_body(conn: sqlite3.Connection, source: IndexSource) -> dict[str, Any]:
    from ..cli.commands import _classify_unresolved, _embedding_info, _scalar

    counts = {
        "files": _scalar(conn, "SELECT count(*) FROM files"),
        "symbols": _scalar(conn, "SELECT count(*) FROM symbols"),
        "edges": _scalar(conn, "SELECT count(*) FROM edges"),
        "chunks": _scalar(conn, "SELECT count(*) FROM chunks"),
        "assertions": _scalar(conn, "SELECT count(*) FROM assertions"),
    }
    by_tier = {
        int(r["tier"]): int(r["n"])
        for r in conn.execute("SELECT tier, count(*) AS n FROM edges GROUP BY tier")
    }
    by_status = {
        str(r["status"]): int(r["n"])
        for r in conn.execute("SELECT status, count(*) AS n FROM assertions GROUP BY status")
    }
    resolved = _scalar(conn, "SELECT count(*) FROM edges WHERE dst_symbol_id IS NOT NULL")
    external, ambiguous = _classify_unresolved(conn)
    in_repo = counts["edges"] - external
    return {
        "ok": True,
        "index": str(source.path),
        "repo_root": db.stored_repo_root(conn),
        "counts": counts,
        # Edges carry the tier column: 0 is the call site as written, 1 is that site
        # bound to a symbol. Symbols are all T0 by construction -- they were parsed,
        # not decided -- and T2 lives in `assertions`, not in `edges`, which is why
        # the assertion counts sit alongside rather than inside this map.
        "edges_by_tier": {
            "T0": by_tier.get(0, 0),
            "T1": by_tier.get(1, 0),
            "T2": by_tier.get(2, 0),
        },
        "assertions_by_status": {
            # Listed with explicit zeros so the shape does not change when the store
            # is empty, and so `rejected` is visible rather than merely absent. The
            # rejected set is the only evidence the gate does anything.
            "active": by_status.get("active", 0),
            "rejected": by_status.get("rejected", 0),
            "stale": by_status.get("stale", 0),
        },
        "symbol_kinds": {
            str(r["kind"]): int(r["n"])
            for r in conn.execute(
                "SELECT kind, count(*) AS n FROM symbols GROUP BY kind ORDER BY n DESC"
            )
        },
        "resolution": {
            "total": counts["edges"],
            "resolved": resolved,
            "external": external,
            "ambiguous": ambiguous,
            "in_repo": in_repo,
            "rate_of_internal": round(resolved / in_repo, 6) if in_repo else 0.0,
        },
        "embeddings": _embedding_info(conn),
    }


# ---------------------------------------------------------------------------
# capability advertisement
# ---------------------------------------------------------------------------

def _advertise_only_what_is_served(server: MCPServer) -> Any:
    """Middleware that strikes `prompts` and `resources` off `initialize` when empty.

    The reference `mcp` SDK derives `ServerCapabilities` from which request handlers
    are registered (`server.lowlevel.Server.get_capabilities`), and `MCPServer` always
    registers `prompts/list` and `resources/list` -- so every server built on it tells
    every client it has prompts and resources whether or not one was ever added. This
    server has neither: five tools, no prompt, no resource, no resource template.

    That is a false statement about the server, and it costs. A client reads the
    capability block and asks one list request per capability declared: measured
    against Claude Code's opening sequence, an unmodified build was asked for
    `tools/list`, `prompts/list` AND `resources/list`, and answered the last two with
    empty arrays -- two round trips spent establishing something the handshake had
    already been in a position to say.

    Two things this is NOT, both worth stating because the measurement said so. It is
    not the reason this server's tools arrive late: the two extra round trips are
    ~0.5 ms each against a ~625 ms start, so the latency lives entirely in the SDK
    import (see the import note at the top of this module). And it is not a claim that
    prompts/resources are unsupported in principle -- the handlers stay registered, so
    a client that asks anyway still gets a well-formed empty list rather than
    METHOD_NOT_FOUND. What changes is only what we assert about ourselves.

    Derived per handshake rather than hardcoded, so this cannot become the second
    false statement: `server.resource()` or `server.prompt()` used anywhere puts the
    capability straight back. `Server.middleware` is the SDK's supported seam for
    exactly this ("append an `async (ctx, call_next)` callable to observe, refuse, or
    rewrite messages"); `initialize` is not otherwise overridable, and `MCPServer`
    exposes no way to pass custom `InitializationOptions` through `run()`.

    Fails open in the two places the SDK could change under it -- a non-dict result,
    or no `capabilities` member -- because a handshake that still says too much is a
    working server, and one that raises here is not. `tests/test_mcp.py` drives a real
    subprocess handshake, so a silent no-op is red rather than invisible.
    """

    async def middleware(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> Any:
        result = await call_next(ctx)
        if ctx.method != "initialize" or not isinstance(result, dict):
            return result
        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict):
            return result
        declared = dict(capabilities)
        if not await server.list_prompts():
            declared.pop("prompts", None)
        if not (await server.list_resources() or await server.list_resource_templates()):
            declared.pop("resources", None)
        return {**result, "capabilities": declared}

    return middleware


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def build_server(
    index_path: Path | str,
    *,
    embedder_factory: Any | None = None,
    version: str = "0.0.1",
) -> MCPServer:
    """Wire the five tools over one index. Does not open it.

    `embedder_factory` is the same seam the CLI uses, and for the same reason: the
    test suite must never load `Qwen3-Embedding-0.6B`, which costs tens of seconds
    and ~1.2GB of VRAM to prove wiring that three floats prove just as well.
    """
    source = IndexSource(path=Path(index_path), embedder_factory=embedder_factory)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        try:
            yield
        finally:
            source.close()

    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        version=version,
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
    )

    @server.tool()
    async def search_code(
        query: str,
        k: int = 10,
        facts_only: bool = False,
        include_source: bool = False,
        evidence_budget: int = 16_384,
        include_assertions: bool = True,
        debug_scores: bool = False,
    ) -> dict[str, Any]:
        """Find the symbols in this repository that bear on a question.

        Hybrid retrieval over a pre-built index: lexical (FTS5) + dense (embeddings,
        when the index holds them) + call-graph expansion, fused into one ranked list.
        One call covers the whole repository.

        By default returns LOCATIONS, not source. Each hit carries qualname, path, a
        line range, the tier its evidence rests on, the modalities that found it,
        `via` -- the symbol whose resolved call edge reached it, non-empty only when
        graph expansion produced the hit -- and the `content_hash` submit_assertion
        needs before it will accept a citation of it.

        Set `include_source=true` to add opt-in evidence sections: each is a complete,
        line-numbered symbol body from the current working tree. `evidence_budget` is
        clamped to 65,536 bytes. If indexed source is stale or unsafe to read, the
        call fails rather than presenting indexed source as current.

        Two things it does that reading and grepping files does not. The dense
        modality matches on meaning, so a query phrased as the question you actually
        have finds code you do not know the name of; `notes` says so when the index
        was built without embeddings and that modality is absent. And graph expansion
        returns symbols carrying none of your query terms that sit one RESOLVED call
        edge from something that does -- `via` names the edge -- which no text search
        reaches at any level of cleverness.

        Where it does not help. Compact mode returns locations from the index
        snapshot; `include_source` returns complete, current, verified symbol bodies
        and refuses stale or unsafe source. get_symbol is the cheaper stop first if
        you want the signature, docstring and call edges. For an exact string, a
        regex, or anything outside the parsed languages, searching the working tree
        directly is both better and current; index_stats says what the index covers.
        `k` is clamped to 100.

        Results are tagged. `candidate_type` is `source` for a symbol and
        `assertion` for a stored semantic claim, and the two are different kinds of
        thing: a claim is something a generator wrote and a judge read, and it is
        served ONLY when it is active, at least one judge recorded `supported`, no
        judge recorded `unsupported` or `refuted`, and every range it cites still
        hashes to the bytes it was written against -- re-checked on this call, not
        remembered. Pending, rejected and stale claims are never returned here at
        any setting; a claim carries its verdicts, its freshness and the citations
        you can go read yourself, so you never have to take it on trust.

        `facts_only` drops everything above tier 1 -- so every claim -- BEFORE the
        page is cut, and the freed slots refill with source. It now changes results
        on an index holding claims. `include_assertions=false` is the ablation: the
        same query answered from source alone, which is how you find out what the
        semantic layer is worth. `debug_scores=true` adds the per-modality rank
        contributions behind each fused score.

        `include_source=true` returns cited bytes for a claim, its subject symbol,
        and complete bodies for source hits. A claim whose citations cannot be
        re-read exactly is withheld whole rather than shown with a partial basis.

        get_symbol remains the surface for every claim attached to one symbol,
        including the ones this tool declines to serve.
        """
        return await source.run_sync(
            _guard,
            source,
            _search_body,
            query=query,
            k=k,
            facts_only=facts_only,
            include_source=include_source,
            evidence_budget=evidence_budget,
            include_assertions=include_assertions,
            debug_scores=debug_scores,
        )

    @server.tool()
    async def get_symbol(qualname: str, facts_only: bool = False) -> dict[str, Any]:
        """Everything the index knows about one symbol, including who calls it.

        Returns its kind, path, line range, signature, docstring and content_hash;
        its callers and callees; the call sites in it that no resolver could bind;
        and any tier-2 assertion stored about it.

        `qualname` is the dotted path from the module root, e.g.
        'codelearner.db.init_db'; a name this index does not hold is refused with
        'no_such_symbol' -- use search_code to find the exact one.

        What this gives you that reading the file does not is `callers`. They are
        RESOLVED edges: call sites a resolver bound to this symbol, each carrying the
        confidence it assigned. That is not the same list as every place the name
        appears, which is what searching for a common method name returns and why
        that answer is usually unusable. Assembling it by hand is a repo-wide search
        plus a disambiguation pass per hit; here it is one call. Callees are the same
        edges in the other direction, and call sites nothing could bind come back
        separately as `unresolved_calls` at tier 0 -- mostly stdlib and third-party
        calls, listed so that a symbol which only calls `json.dumps` does not look
        inert.

        Where it does not help. No implementation body comes back -- signature and
        docstring only -- so when the body is the question, open the file at the path
        and lines returned. A qualname is not unique in the schema either: if two
        symbols share one, callers and callees describe the first, and
        `duplicate_qualnames` says how many were passed over. And a resolver's
        binding can be wrong, which is exactly what the tier-1 label and the
        confidence on each edge are there to tell you.

        `assertions` is the only tier-2 content this server ever returns. Each is
        re-verified against the file on disk before it is returned, so one whose
        evidence has moved is expired rather than served.

        `facts_only` withholds them -- and this is the surface where that flag changes
        an answer, unlike search_code's, where no modality retrieves at tier 2. Use it
        when you want only what was parsed out of the source and what a resolver bound,
        with nothing another agent asserted. The count of withheld claims comes back as
        `assertions_withheld` and is repeated in `notes`, so a suppressed claim is
        never an invisible one: you are told that something exists and that you chose
        not to read it.
        """
        return await source.run_sync(
            _guard, source, _get_symbol_body, qualname=qualname, facts_only=facts_only
        )

    @server.tool()
    async def reading_path(topic: str = "", limit: int = 12) -> dict[str, Any]:
        """Where to start reading an unfamiliar codebase, in dependency order.

        With a `topic`, the tour is seeded from retrieval and answers "read these N
        things to understand auth". Without one, it is seeded from call-graph
        centrality and answers "read these N things to understand this repo". Stops
        are ordered by dependency depth, so a stop's callees come before it, and each
        carries its location, signature, docstring summary, why it was chosen, how
        many callers it has in the repo, and any cycle it belongs to.

        This answers a question the other tools cannot be asked. Both search_code and
        a file search need you to already know what to look for; this one takes the
        call graph as the map and orders symbols by how much else depends on them, so
        it is the tool for "I have never seen this repository" and for "which of
        these forty files is load-bearing".

        Where it does not help. It is an ordering over symbols, not their contents --
        the reading is still yours, from the paths and line ranges it hands you. When
        you already know which symbol you care about, get_symbol is the shorter path,
        and when you have a specific question, search_code is. `limit` is clamped
        to 100.
        """
        return await source.run_sync(
            _guard, source, _reading_path_body, topic=topic, limit=limit
        )

    @server.tool()
    async def submit_assertion(
        subject_qualname: str,
        claim: str,
        evidence_spans: list[EvidenceSpanInput],
        kind: str = DEFAULT_ASSERTION_KIND,
        generator: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Store an inference about a symbol, if and only if its citations hold.

        This is the inversion: the index does no inference of its own, you do, and
        this gate decides whether it may be stored. Every rule is arithmetic.

        What an accepted claim becomes: a tier-2 answer that get_symbol returns to
        whoever asks about that symbol next, re-hashed against the file first, so a
        claim whose evidence has since moved expires instead of being repeated. That
        is the reason to spend the call -- a conclusion you reached by reading is
        otherwise yours alone and gone at the end of this session.

        An assertion with zero evidence spans is refused ('evidence_required'), and so
        is one with no claim text ('empty_claim') -- correct citations under a blank
        statement are still nothing anyone can check. A span whose bytes no longer
        hash to what you cited is refused ('hash_mismatch'), and the refusal returns
        the observed hash and the text that is actually there, so you can correct the
        citation rather than guess again. A subject_qualname that names no indexed
        symbol is refused ('unknown_subject') -- verified spans do not make a claim
        about a symbol that does not exist accountable to anyone.

        Cite what you read: pass the content_hash that search_code or get_symbol
        returned, or the exact source text at those lines. A span carrying neither is
        'evidence_unverifiable'. Spans must name a file this index parsed
        ('file_missing'), inside the repository ('path_escapes_repo',
        'span_escapes_repo'), at lines that exist in it ('bad_range'), by a path
        containing no NUL byte ('bad_path'), in a file under 4MiB ('file_too_large').
        At most 32 spans per submission ('too_many_spans'); the claim is capped at
        4096 characters ('claim_too_long'); confidence, if given, must be a real
        number between 0 and 1 ('bad_confidence').

        One refusal is not about your call at all. If a human rebuilds the index while
        you are working, this returns 'index_replaced' -- before writing anything if
        the rebuild is already visible, after writing if it lands mid-call, in which
        case the claim went into the deleted file and is gone. Either way the hashes
        you are holding are one build old: re-run your retrieval, then submit again.
        """
        return await source.run_sync(
            _guard,
            source,
            _submit_body,
            subject_qualname=subject_qualname,
            claim=claim,
            evidence_spans=evidence_spans,
            kind=kind,
            generator=generator,
            confidence=confidence,
        )

    @server.tool()
    async def index_stats() -> dict[str, Any]:
        """What this index covers, and what is in it by tier.

        The repo root it is bound to; counts of files, symbols, edges and chunks;
        symbol counts by kind; edges split by tier (T0 call sites as written, T1
        bound to a symbol); assertions split by status, including the rejected ones;
        name resolution rates; and whether the index holds vectors.

        Worth one call before leaning on the other four, because nothing they return
        says which codebase they are describing or how much of it they saw.
        `repo_root` says whether this index is about the code you are looking at at
        all. `counts.files` against what you can see on disk says whether it covers
        the repository or a fraction of it -- a fraction is the state in which
        search_code returns thin results that look like absence of evidence.
        `embeddings.present` says whether search_code has a dense modality or is
        running on lexical and graph alone. And a low `resolution.rate_of_internal`
        is advance warning that get_symbol's callers will be sparse, since only bound
        edges appear there.
        """
        return await source.run_sync(_guard, source, _stats_body)

    # Appended after the tools are registered, so the first handshake sees the real
    # answer to "does this server hold any prompt or resource" rather than the answer
    # at construction time.
    server.middleware.append(_advertise_only_what_is_served(server))
    return server


def __getattr__(name: str) -> Any:
    """`resolve_index_path`, kept as a name here without its import cost.

    It has always been re-exported from this module and is in `__all__`, so removing
    it would break an importer for no gain; but it lives in `..cli.commands`, and
    reaching it drags the whole `codelearner.cli` package -- 25 ms of argparse
    construction -- into a server that only ever needed a path join. PEP 562 keeps the
    name and moves the cost to whoever actually asks for it.
    """
    if name == "resolve_index_path":
        from ..cli.commands import resolve_index_path

        return resolve_index_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ERROR_CODES",
    "MAX_CITED_FILE_BYTES",
    "MAX_CLAIM_CHARS",
    "MAX_EVIDENCE_SPANS",
    "MAX_K",
    "MAX_OBSERVED_TEXT_CHARS",
    "MAX_STOPS",
    "SERVER_NAME",
    "EvidenceSpanInput",
    "IndexSource",
    "ToolError",
    "build_server",
    # Served by this module's `__getattr__`, which ruff's static pass cannot see.
    # Listed here because it is the public surface; imported there because reaching it
    # costs the whole `codelearner.cli` package.
    "resolve_index_path",  # noqa: F822
]
