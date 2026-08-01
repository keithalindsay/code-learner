"""The five tools, and the gate that stands behind one of them.

Two rules shape this module.

**Never raise into the transport.** A traceback crossing an MCP boundary tells the
agent that the tool is broken, which is the one conclusion that stops it trying
again. Every predictable condition -- no index, no such symbol, an unverifiable
citation -- comes back as a structured object with a `code`, a `message`, and
whatever the agent needs to fix it. `CliError` does the same job for the human CLI;
this is the machine-facing half of the same policy.

**Reuse the CLI's derivations rather than re-deriving them.** Tier labels, the
`facts_only` filter, and the per-hit JSON shape all come from `cli.render`; the
assertion gate is `assertions.store.write_assertion` called, not reimplemented. Two
surfaces that answer "is this a fact or a guess" from two code paths will drift, and
the drift shows up as a caller who asked for facts and got a resolver's guess.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from .. import db
from ..assertions import store
from ..cli.commands import (
    _classify_unresolved,
    _embedding_info,
    _scalar,
    resolve_index_path,
)
from ..cli.render import facts_only as facts_only_filter
from ..cli.render import hit_json
from ..index import Embedder
from ..ingest.types import content_hash
from ..onboard import build_reading_path
from ..retrieve import search, stored_embed_model

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
    """

    path: Path
    embedder_factory: Any | None = None
    _conn: sqlite3.Connection | None = None
    _embedder: Embedder | None = None
    _embed_checked: bool = False

    def connect(self) -> sqlite3.Connection:
        """The open connection, or a `ToolError` explaining what is missing.

        Existence is re-checked on every call even when a connection is cached,
        because `sqlite3.connect` will happily create an empty file at any path --
        which is how a typo'd `--index-path` becomes "0 results" instead of "no such
        index", and how a deleted index becomes an empty one.
        """
        if not self.path.exists():
            self._conn = None
            raise ToolError(
                "no_index",
                f"no index at {self.path}. Build one with `codelearner index <repo>`, "
                "or start this server with --index-path pointing at an existing one.",
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
        return self._conn

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
        """
        if self._embed_checked:
            return self._embedder, []
        self._embed_checked = True
        stored = stored_embed_model(conn)
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

    Positional-only up front so that a tool body may itself take a parameter named
    `source` or `fn` without colliding with this frame's own arguments.
    """
    try:
        conn = source.connect()
        return fn(conn, source, **kwargs)
    except ToolError as exc:
        return exc.payload()
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
    it, and for a decorated symbol it begins at the `@`. A module's span runs to the
    last byte of the file, which is one line past the last line anything is written
    on. Measured on this repository at 36 modules, 36 methods, 11 functions and 2
    classes out of 383 symbols -- around 15% -- where the two disagree.

    That matters because the hash `search_code` and `get_symbol` hand back is the
    hash of the SYMBOL's bytes. Checking a citation only against the lines' bytes
    would reject the exact hash this server just published, for 15% of symbols, with
    a message accusing the agent of citing something that had changed. Looking the
    symbol up is what closes that gap -- and it is the same reason
    `store.span_for_symbol` exists.
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
    conn: sqlite3.Connection, source: IndexSource, query: str, k: int, facts_only: bool
) -> dict[str, Any]:
    k = max(1, min(int(k), MAX_K))
    embedder, notes = source.embedder(conn)
    result = search(conn, query, k=k, embedder=embedder, use_dense=embedder is not None)
    hits = facts_only_filter(result.hits) if facts_only else list(result.hits)
    hashes = _symbol_hashes(conn, [h.symbol_id for h in hits])
    return {
        "ok": True,
        "query": query,
        "k": k,
        "facts_only": facts_only,
        "count": len(hits),
        "notes": notes,
        # `hit_json` is the CLI's shape, reused verbatim so the two surfaces cannot
        # disagree about a tier. `content_hash` is the one addition: an agent that
        # wants to cite this hit needs it, and a human reading a terminal does not.
        "hits": [
            dict(hit_json(hit, rank), content_hash=hashes.get(hit.symbol_id))
            for rank, hit in enumerate(hits, start=1)
        ],
    }


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
    conn: sqlite3.Connection, source: IndexSource, qualname: str
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
    unresolved = conn.execute(
        "SELECT kind, dst_name, line FROM edges "
        "WHERE src_symbol_id = ? AND dst_symbol_id IS NULL ORDER BY line",
        (symbol_id,),
    ).fetchall()
    return {
        "ok": True,
        "notes": notes,
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
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        version=version,
        instructions=INSTRUCTIONS,
    )

    @server.tool()
    def search_code(query: str, k: int = 10, facts_only: bool = False) -> dict[str, Any]:
        """Hybrid retrieval over the index: lexical + dense + graph expansion, fused.

        Returns tier-labelled hits. Each carries qualname, path with a line range, the
        modalities that found it, and `via` -- the account of which symbol's call edge
        reached it, non-empty only when graph expansion produced the hit.

        Set facts_only=true to exclude tier 2, leaving only parsed facts (T0) and
        resolved names (T1). Each hit also carries the `content_hash` you need to cite
        it in submit_assertion.
        """
        return _guard(source, _search_body, query=query, k=k, facts_only=facts_only)

    @server.tool()
    def get_symbol(qualname: str) -> dict[str, Any]:
        """One symbol, its resolved callers and callees, and any servable assertions.

        `qualname` is the dotted path from the module root, e.g.
        'codelearner.db.init_db'. Callers and callees are tier 1 -- resolved name
        bindings, each with the confidence its resolver assigned. Unbound call sites
        are returned separately as tier 0. Assertions are re-verified against the file
        on disk before being returned; one whose evidence has moved is expired rather
        than served.
        """
        return _guard(source, _get_symbol_body, qualname=qualname)

    @server.tool()
    def reading_path(topic: str = "", limit: int = 12) -> dict[str, Any]:
        """An ordered tour of the codebase: what to read, in what order, and why.

        With a topic, the tour is seeded from retrieval and answers "read these N
        things to understand auth". Without one, it is seeded from call-graph
        centrality and answers "read these N things to understand this repo". Stops
        are ordered by dependency depth, so a stop's callees come before it.
        """
        return _guard(source, _reading_path_body, topic=topic, limit=limit)

    @server.tool()
    def submit_assertion(
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

        An assertion with zero evidence spans is refused ('evidence_required'), and so
        is one with no claim text ('empty_claim') -- correct citations under a blank
        statement are still nothing anyone can check. A span whose bytes no longer
        hash to what you cited is refused ('hash_mismatch'), and the refusal returns
        the observed hash and the text that is actually there, so you can correct the
        citation rather than guess again. A subject_qualname that names no indexed
        symbol is refused ('unknown_subject') -- verified spans do not make a claim
        about a symbol that does not exist accountable to anyone.

        Cite what you read: pass the content_hash that search_code or get_symbol
        returned, or the exact source text at those lines. Spans must name files this
        index parsed ('file_not_indexed'), at most 32 of them; the claim is capped at
        4096 characters and confidence, if given, is a probability between 0 and 1.
        """
        return _guard(
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
    def index_stats() -> dict[str, Any]:
        """What is in this index, by tier.

        Counts of files, symbols, edges and chunks; edges split by tier (T0 call sites
        as written, T1 bound to a symbol); assertions split by status, including the
        rejected ones; name resolution rates; and whether the index holds vectors.
        """
        return _guard(source, _stats_body)

    return server


__all__ = [
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
    "resolve_index_path",
]
