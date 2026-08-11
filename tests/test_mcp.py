"""The MCP surface: the gate, the tier filter, and never raising into the transport.

Every test here names a rule that would otherwise fail silently, and is written so
that deleting the rule turns it red. Three of them are the gate itself -- zero
evidence refused, a moved citation refused, a good one admitted -- because those are
the only reason an agent's inference is worth storing at all.

No test loads a real embedding model. The fixture index holds no vectors, so the
dense modality is skipped by the same code path a user hits, and `build_server`
takes an embedder factory for the cases that need one. `Qwen3-Embedding-0.6B` costs
tens of seconds and ~1.2GB of VRAM to prove wiring that three floats prove.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the optional `mcp` dependency")

from codelearner import db  # noqa: E402
from codelearner.assertions import store  # noqa: E402
from codelearner.ingest import index_repo  # noqa: E402
from codelearner.retrieve import Hit  # noqa: E402
from codelearner.retrieve.search import SearchResult  # noqa: E402
from codelearner.server import app as server_app  # noqa: E402
from codelearner.server import build_server  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOOL_NAMES = {"search_code", "get_symbol", "reading_path", "submit_assertion", "index_stats"}

# `frobnicate_widgets` is reachable by text; `_plumbing` shares no vocabulary with
# the query, so the only route to it is the resolved call edge -- which is what makes
# the tier-1 `via` assertion mean something.
CORE = (
    'def frobnicate_widgets():\n'
    '    """Frobnicate every widget on the tray."""\n'
    '    return _plumbing()\n'
    '\n'
    '\n'
    'def _plumbing():\n'
    '    """Detail."""\n'
    '    return 42\n'
)

QUERY = "frobnicate widgets"


class FakeEmbedder:
    """Deterministic, dependency-free `Embedder`. Never touches a GPU or a network."""

    MARKERS = ("frobnicate", "widget", "plumbing")

    def __init__(self, name: str = "fake/v1") -> None:
        self._name = name

    @property
    def dim(self) -> int:
        return len(self.MARKERS)

    @property
    def name(self) -> str:
        return self._name

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [float(lowered.count(m)) for m in self.MARKERS]
        norm = sum(v * v for v in raw) ** 0.5
        return [v / norm for v in raw] if norm else [0.0] * len(raw)

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


def _mkrepo(root: Path, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in (files or {"core.py": CORE}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


def call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool the way a client would, and return its structured result.

    Asserts `is_error` is false on every call. That is not incidental: the module's
    first rule is that a predicted condition comes back as data, so a test that
    accepted an error result would let a traceback through unnoticed.
    """
    result = asyncio.run(server.call_tool(name, arguments))
    assert result.is_error is False, f"{name} raised into the transport: {result.content}"
    assert result.structured_content is not None
    return dict(result.structured_content)


@pytest.fixture
def served(tmp_path):
    """A one-file repo, an index over it, and a server bound to that index."""
    repo = _mkrepo(tmp_path / "repo")
    index_path = tmp_path / "index.db"
    conn, _ = index_repo(repo, index_path=index_path)
    conn.close()
    return repo, index_path, build_server(index_path, embedder_factory=FakeEmbedder)


# Decorators, a property, and a class body -- the three shapes where a symbol's
# stored bytes are NOT its lines' bytes. `@memoize` puts the symbol's first byte a
# line above its `def`; `@property` puts it four columns in from the start of the
# line; and the module's span runs one line past the last line anything is on.
TRAY = (
    'import functools\n'
    '\n'
    '\n'
    'def memoize(fn):\n'
    '    """Cache a nullary method."""\n'
    '    return functools.cache(fn)\n'
    '\n'
    '\n'
    'class Tray:\n'
    '    """A tray of widgets."""\n'
    '\n'
    '    @property\n'
    '    def widgets(self):\n'
    '        return self._widgets\n'
    '\n'
    '    @memoize\n'
    '    def count(self):\n'
    '        return len(self._widgets)\n'
)


@pytest.fixture
def decorated(tmp_path):
    """A repo whose symbols exercise every way lines and symbol bytes disagree."""
    repo = _mkrepo(tmp_path / "repo", {"tray.py": TRAY})
    index_path = tmp_path / "index.db"
    conn, _ = index_repo(repo, index_path=index_path)
    yield repo, conn, build_server(index_path, embedder_factory=FakeEmbedder)
    conn.close()


def _hash_of(server: Any, qualname: str) -> tuple[str, int, int]:
    """The stored hash and line span of a symbol, as an agent would obtain them."""
    payload = call(server, "get_symbol", qualname=qualname)
    symbol = payload["symbol"]
    return symbol["content_hash"], symbol["line_start"], symbol["line_end"]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_all_five_tools_are_registered_with_descriptions_and_schemas(served):
    """A tool with no description is a tool an agent will not choose correctly. The
    description IS the interface here -- there is no documentation page an agent can
    go and read at call time."""
    _, _, server = served
    tools = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == TOOL_NAMES
    for tool in tools:
        assert tool.description and len(tool.description) > 40
        assert tool.input_schema["type"] == "object"
    assert set(by_name["search_code"].input_schema["properties"]) == {
        "query",
        "k",
        "facts_only",
        "include_source",
        "evidence_budget",
    }
    search_schema = by_name["search_code"].input_schema
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["k"]["default"] == 10
    assert search_schema["properties"]["facts_only"]["default"] is False
    assert search_schema["properties"]["include_source"]["default"] is False
    assert search_schema["properties"]["evidence_budget"]["default"] == 16_384
    # `facts_only` is on BOTH read tools, and on this one it is the copy that can
    # change an answer -- `get_symbol` returns the only tier-2 content the server has.
    # Pinned in the schema because an agent chooses a call from the schema, and a flag
    # the schema does not mention is a flag no agent will ever pass.
    assert set(by_name["get_symbol"].input_schema["properties"]) == {
        "qualname",
        "facts_only",
    }
    # The span shape has to survive into the schema, or the agent has no way to know
    # a citation needs a hash and guesses the field names.
    span_schema = by_name["submit_assertion"].input_schema["properties"]["evidence_spans"]
    assert span_schema["type"] == "array"


def test_search_code_description_distinguishes_compact_and_verified_source_modes(served):
    """The tool description must not call opt-in source unavailable after it was added."""
    _, _, server = served
    description = {tool.name: tool.description for tool in asyncio.run(server.list_tools())}["search_code"]
    description = " ".join(description.split()).lower()

    assert "compact mode returns locations from the index snapshot" in description
    assert "`include_source` returns complete, current, verified symbol bodies" in description
    assert "refuses stale or unsafe source" in description


# The descriptions are the product surface an agent actually reads, so they are held
# to the same rule as the rest of this file: say what is true, and say what is not.
# The three tests below pin the two halves of that rule which are easiest to lose
# under pressure to be chosen more often.

def test_search_code_still_says_that_facts_only_filters_nothing_today(served):
    """The most tempting sentence to delete while making a description more
    persuasive, and the one that must survive. `facts_only` is wired at the seam a
    tier-2 retrieval modality would arrive through and there is no such modality yet,
    so passing it changes no result -- an agent that reads a description implying
    otherwise will pass it believing it has excluded inferences it never received.

    A description is allowed to argue for its tool. It is not allowed to be quiet
    about a flag that does nothing."""
    _, _, server = served
    description = {t.name: t.description for t in asyncio.run(server.list_tools())}
    search = description["search_code"]
    assert "drops nothing" in search
    # And the pointer to where the flag is not inert, so the caller who actually
    # wanted tier 2 withheld is sent to the surface that does it.
    assert "get_symbol" in search
    assert "withholds them" in description["get_symbol"]


@pytest.mark.parametrize("name", ["search_code", "get_symbol", "reading_path"])
def test_a_tool_that_says_when_to_prefer_it_also_says_when_not_to(served, name):
    """Every one of the three retrieval tools now argues for itself -- names the thing
    it does that reading and grepping files does not. That claim is only usable if the
    boundary comes with it, and the boundary is the part a description written to win
    a comparison would drop first: naming when your tool is the wrong call costs you
    calls.

    It is also the more useful half. An agent that knows `search_code` returns
    locations and not source, that the index can be behind the working tree, and that
    an exact-string search is better served by the working tree, routes correctly.
    One that has only been told the tool is good does not."""
    _, _, server = served
    description = {t.name: t.description for t in asyncio.run(server.list_tools())}
    assert "does not" in description[name]


@pytest.mark.parametrize("name", sorted(TOOL_NAMES))
def test_no_description_demands_priority_instead_of_describing_itself(served, name):
    """The line this project draws, made mechanical.

    A competing server's description opens "PRIMARY TOOL -- call FIRST" and spends
    4,597 characters instructing the model not to read files. It works, and it is not
    a description: nothing in it is a checkable statement about what that tool
    returns, so the identical text would sit equally well on a tool that did nothing.
    Ours may say what it returns, what it costs, and the conditions under which it
    beats opening a file -- all of which are wrong if the tool changes, which is what
    makes them descriptions. It may not issue instructions about which tool the agent
    should reach for first, because that claim is not about this tool at all and no
    change to this server could ever falsify it.

    Held as a test rather than a convention because the pressure to cross it arrives
    later, from a benchmark result, and by then whoever crosses it will have a reason
    that sounds good."""
    _, _, server = served
    description = {t.name: t.description for t in asyncio.run(server.list_tools())}
    lowered = description[name].lower()
    for demand in (
        "primary tool",
        "call first",
        "call this first",
        "use this first",
        "before any other",
        "instead of reading",
        "do not read",
        "always use",
        "always call",
        "you must use",
        "you must call",
    ):
        assert demand not in lowered, f"{name} demands priority rather than earning it: {demand!r}"


# ---------------------------------------------------------------------------
# the handshake -- what the server says it is
# ---------------------------------------------------------------------------

def _stdio_handshake(cwd: Path, timeout: float = 120.0) -> dict[str, Any]:
    """Open a real session against a real subprocess and return its replies.

    Over the wire, and as a subprocess, because that is the only place the thing
    under test exists: the capability block is produced by the SDK's own runner from
    handlers this package does not own, and every in-process shortcut around it would
    be testing the assertion instead of the server.

    Sequenced, not pipelined, and stdin stays open until both replies are in. Sending
    all three messages at once and closing stdin is what a `communicate()` call wants
    and it loses the `tools/list`: EOF on stdin ends the session, and it reached the
    server while the request was still queued behind the inline `initialize`. That
    failed roughly one run in three -- which is worse than failing always, so it is
    written down here.

    Reading happens on a daemon thread against a deadline, so a server that never
    answers fails this test in `timeout` seconds rather than parking the suite on a
    pipe read.
    """
    proc = subprocess.Popen(  # noqa: S603 - argv is this interpreter and this package
        [sys.executable, "-m", "codelearner.server", str(cwd)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(cwd),
    )
    replies: dict[str, Any] = {}
    errors: list[str] = []
    arrived = threading.Event()

    def read_stdout() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if not line.startswith("{"):
                continue
            msg = json.loads(line)
            if msg.get("id") in (1, 2):
                replies[str(msg["id"])] = msg
                arrived.set()

    def read_stderr() -> None:
        errors.extend(proc.stderr)  # type: ignore[arg-type]

    for target in (read_stdout, read_stderr):
        threading.Thread(target=target, daemon=True).start()

    def send(message: dict[str, Any]) -> None:
        proc.stdin.write(json.dumps(message) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    def wait_for(key: str, what: str) -> None:
        deadline = time.monotonic() + timeout
        while key not in replies and time.monotonic() < deadline:
            arrived.wait(timeout=1.0)
            arrived.clear()
        assert key in replies, f"no {what} within {timeout}s. stderr:\n{''.join(errors)}"

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "codelearner-tests", "version": "0"}}})
        wait_for("1", "initialize reply")
        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        wait_for("2", "tools/list reply")
    finally:
        proc.kill()
        proc.wait(timeout=timeout)
    return replies


def test_the_handshake_declares_tools_and_not_the_prompts_and_resources_we_lack(tmp_path):
    """This server has five tools, no prompt and no resource -- and said otherwise.

    `MCPServer` registers `prompts/list` and `resources/list` unconditionally, and the
    SDK derives `ServerCapabilities` from which handlers exist, so every server built
    on it advertises prompts and resources whether or not one was ever added. A client
    reads that block and asks one list request per capability declared: measured
    against Claude Code's opening sequence, an unmodified build was asked for
    `tools/list`, `prompts/list` AND `resources/list` and answered the last two with
    empty arrays.

    The cost is small -- ~0.5 ms a round trip -- and the reason to fix it is not the
    cost. It is that the handshake is the one place this server describes itself, and
    two of the three things it said there were untrue.

    Runs against no index at all, deliberately: the handshake must be honest before
    anything has been built, which is the state a client meets this server in on the
    morning somebody installs it."""
    replies = _stdio_handshake(tmp_path)
    capabilities = replies["1"]["result"]["capabilities"]
    assert "tools" in capabilities
    assert "prompts" not in capabilities, capabilities
    assert "resources" not in capabilities, capabilities
    # The rewrite must not have cost anything else in the block a client reads.
    assert replies["1"]["result"]["serverInfo"]["name"] == server_app.SERVER_NAME
    assert replies["1"]["result"]["instructions"] == server_app.INSTRUCTIONS
    # And the tools still arrive, which is the only reason any of this matters.
    assert {t["name"] for t in replies["2"]["result"]["tools"]} == TOOL_NAMES


def test_a_prompt_or_a_resource_puts_its_capability_straight_back(tmp_path):
    """Derived per handshake, not hardcoded to "we have none".

    A capability block asserted as a constant is correct exactly until somebody calls
    `server.resource(...)`, at which point it becomes the same lie in the other
    direction -- a real resource that no client will ever ask for, and nothing to
    notice it. This drives the middleware directly with a stub handshake result so the
    derivation is exercised without a subprocess per case."""
    server = build_server(tmp_path / "index.db")
    middleware = server_app._advertise_only_what_is_served(server)

    class _Init:
        method = "initialize"

    async def _call_next(_ctx):
        return {"capabilities": {"prompts": {"listChanged": False},
                                 "resources": {"listChanged": False},
                                 "tools": {"listChanged": False}}}

    empty = asyncio.run(middleware(_Init(), _call_next))
    assert set(empty["capabilities"]) == {"tools"}

    @server.resource("data://note")
    def note() -> str:
        return "something a client could read"

    @server.prompt()
    def explain() -> str:
        return "something a client could run"

    stocked = asyncio.run(middleware(_Init(), _call_next))
    assert set(stocked["capabilities"]) == {"prompts", "resources", "tools"}


def test_only_the_initialize_reply_is_rewritten(tmp_path):
    """The middleware sits in front of EVERY inbound message, so a rule written for
    one method has to be a rule about that method. An earlier shape of this keyed off
    the presence of a `capabilities` member rather than off `ctx.method`, which is a
    key a tool result is free to have."""
    server = build_server(tmp_path / "index.db")
    middleware = server_app._advertise_only_what_is_served(server)

    class _Other:
        method = "tools/call"

    async def _call_next(_ctx):
        return {"capabilities": {"prompts": {}, "resources": {}}}

    assert set(asyncio.run(middleware(_Other(), _call_next))["capabilities"]) == {
        "prompts",
        "resources",
    }


def test_console_entry_point_is_registered():
    """The server is only reachable if a client can launch it by name. An entry point
    that is documented in the README and missing from pyproject fails at the one
    moment nobody is watching -- inside somebody else's MCP client."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    scripts = config["project"]["scripts"]
    assert scripts["codelearner-mcp"] == "codelearner.server:main"
    module, _, attr = scripts["codelearner-mcp"].partition(":")
    assert callable(getattr(__import__(module, fromlist=[attr]), attr))


# ---------------------------------------------------------------------------
# missing / unusable index -- structured, never a traceback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_code", {"query": QUERY}),
        ("get_symbol", {"qualname": "core.frobnicate_widgets"}),
        ("reading_path", {}),
        ("index_stats", {}),
        (
            "submit_assertion",
            {
                "subject_qualname": "core.frobnicate_widgets",
                "claim": "it frobnicates",
                "evidence_spans": [
                    {"path": "core.py", "line_start": 1, "line_end": 3, "content_hash": "x"}
                ],
            },
        ),
    ],
)
def test_every_tool_reports_a_missing_index_as_data(tmp_path, tool, arguments):
    """No index is a normal state of the world -- the server starts before the repo
    is indexed, on purpose. Every tool must say so as a result the agent can read.
    Delete the existence check in `IndexSource.connect` and sqlite creates an empty
    file instead, turning "no index" into "no results" or a bare OperationalError."""
    server = build_server(tmp_path / "nothing" / "index.db", embedder_factory=FakeEmbedder)
    payload = call(server, tool, **arguments)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "no_index"
    assert "codelearner index" in payload["error"]["message"]


def test_a_file_that_is_not_an_index_is_reported_rather_than_raised(tmp_path):
    """The path exists and opens; it just has no tables. That fails on the first
    query rather than on connect, which is why `_guard` catches sqlite3.Error too."""
    bogus = tmp_path / "notes.txt"
    bogus.write_text("this is not a database")
    server = build_server(bogus, embedder_factory=FakeEmbedder)
    payload = call(server, "index_stats")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "index_unreadable"


def test_an_index_deleted_underneath_a_live_server_is_reported(served):
    """Existence is re-checked per call, not once at connect. A cached handle to a
    deleted file would keep answering from a database nobody can see."""
    _, index_path, server = served
    assert call(server, "index_stats")["ok"] is True
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(index_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    assert call(server, "index_stats")["error"]["code"] == "no_index"


# ---------------------------------------------------------------------------
# the index replaced underneath a live server
#
# The second morning: an agent session open in the editor, a human running
# `codelearner index --force` in a terminal. That command DELETES the index and
# builds a new one, so an existence check sees nothing happen -- the path is
# occupied again before the next tool call arrives -- while the cached connection
# still points at the unlinked inode. Reads answered from the previous build; a
# write was accepted, reported `servable: true`, and left zero rows on disk.
# ---------------------------------------------------------------------------

def _rebuild(repo: Path, index_path: Path) -> None:
    """What `codelearner index --force` does to the file: delete it, build again.

    By inode replacement rather than by writing into the file already there, because
    the replacement is the whole bug. Asserts the inode actually moved: a filesystem
    that handed the rebuild the same number would leave every test below passing
    while testing nothing, and the guarantee that it cannot is that somebody still
    holds the old file open.
    """
    before = index_path.stat().st_ino if index_path.exists() else None
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(index_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    conn, _ = index_repo(repo, index_path=index_path)
    conn.close()
    if before is not None:
        assert index_path.stat().st_ino != before, (
            "the rebuild reused the inode, so nothing here is testing what it claims "
            "to -- an open connection to the old file should make that impossible"
        )


def _assertions_on_disk(index_path: Path) -> int:
    """How many assertions are in the file that is AT this path, opened fresh.

    Fresh is the entire point. The server's own report of what it stored is exactly
    the thing under test, and the failure being measured is a `write_assertion` that
    commits, returns an id, verifies as servable, and lands in a file nobody can
    open again.
    """
    conn = db.connect(index_path)
    try:
        return int(conn.execute("SELECT count(*) FROM assertions").fetchone()[0])
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("index_stats", {}),
        ("search_code", {"query": QUERY}),
        ("get_symbol", {"qualname": "core.frobnicate_widgets"}),
        ("reading_path", {}),
    ],
)
def test_a_rebuilt_index_is_refused_once_instead_of_served_from_the_old_inode(
    served, tool, arguments
):
    """Existence is not identity, and `--force` is the difference. Drop the
    `(st_dev, st_ino)` check from `IndexSource.connect` and every one of these answers
    `ok: true` out of a deleted database -- correct-looking results about a build that
    no longer exists, which is the one failure with nothing downstream to catch it.

    The refusal is one-shot by design: the cached connection is dropped on the way
    out, so the retry opens the file that is there now."""
    repo, index_path, server = served
    assert call(server, "index_stats")["ok"] is True  # opens and caches the handle

    _rebuild(repo, index_path)

    payload = call(server, tool, **arguments)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "index_replaced"

    assert call(server, tool, **arguments)["ok"] is True


def test_the_index_replaced_refusal_names_something_the_agent_can_act_on(served):
    """A refusal an agent cannot act on teaches it that the server is flaky. This one
    has to say more than "try again", because trying again with the same hashes is
    precisely the wrong move: they were published by the previous build."""
    repo, index_path, server = served
    _hash_of(server, "core.frobnicate_widgets")  # the agent now holds a stale hash

    _rebuild(repo, index_path)

    error = call(server, "index_stats")["error"]
    assert error["code"] == "index_replaced"
    assert error["index"] == str(index_path)
    assert "content_hash" in error["message"]
    assert "retrieval" in error["message"]


def test_a_submission_after_a_rebuild_is_refused_before_it_reaches_the_deleted_file(
    served,
):
    """The auditor's sequence, exactly: submit through a live server after a rebuild
    and be told `ok: true, accepted: true, servable: true` with zero assertions
    surviving on disk. Every check inside `submit_assertion` passed -- the subject
    resolved, the spans re-hashed off disk and matched -- because all of them ran
    against a database that had been unlinked."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    submission = {
        "subject_qualname": "core.frobnicate_widgets",
        "claim": "frobnicates every widget on the tray",
        "evidence_spans": [
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    }

    _rebuild(repo, index_path)

    refused = call(server, "submit_assertion", **submission)
    assert refused["ok"] is False
    assert refused["error"]["code"] == "index_replaced"
    assert _assertions_on_disk(index_path) == 0

    # And the retry the refusal asks for lands in the index that is actually there.
    accepted = call(server, "submit_assertion", **submission)
    assert accepted["ok"] is True
    assert _assertions_on_disk(index_path) == 1


def test_a_rebuild_landing_mid_submission_is_not_reported_as_success(served, monkeypatch):
    """The window `connect` cannot see. Identity is checked once, before the tool body
    runs; a rebuild that lands between that check and the commit passes it, and the
    row goes into the deleted file regardless. Nothing available to this process
    closes that window -- sqlite will not say whether the file under an open handle
    has been unlinked, and there is no lock shared with the indexing process.

    What is closable is the report. The claim is lost either way; only `index_replaced`
    makes the agent submit it again."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    real_write = store.write_assertion

    def rebuild_then_write(conn, **kwargs):
        _rebuild(repo, index_path)
        return real_write(conn, **kwargs)

    monkeypatch.setattr(server_app.store, "write_assertion", rebuild_then_write)

    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "index_replaced"
    # It has to admit the write is gone rather than imply a clean rejection.
    assert "submit again" in payload["error"]["message"]
    assert _assertions_on_disk(index_path) == 0


def test_a_vanished_index_closes_its_connection_instead_of_leaking_it(served):
    """`self._conn = None` drops the reference, not the descriptor. sqlite goes on
    holding the file open, which also pins the unlinked inode, so a server watching a
    repository that is re-indexed nightly leaks one handle and one deleted database
    per rebuild -- and neither shows up anywhere a human looks."""
    _, index_path, _ = served
    source = server_app.IndexSource(path=index_path, embedder_factory=FakeEmbedder)
    conn = source.connect()
    assert conn.execute("SELECT count(*) FROM symbols").fetchone()[0] >= 1

    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(index_path) + suffix)
        if candidate.exists():
            candidate.unlink()

    with pytest.raises(server_app.ToolError) as caught:
        source.connect()
    assert caught.value.code == "no_index"
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_a_swap_closes_the_old_connection_and_reopens_the_embedder_question(served):
    """Two things were cached off the file that is now gone, and both go. The
    connection is the loud one. The embedder CHECK is the quiet one: `_embed_checked`
    records an answer read out of the old index's `meta`, so a rebuild that finally
    ran `--embed` would be told "this index has no embeddings" by a server that
    decided that before they existed and never looked again."""
    repo, index_path, _ = served
    source = server_app.IndexSource(path=index_path, embedder_factory=FakeEmbedder)
    conn = source.connect()
    source.embedder(conn)
    assert source._embed_checked is True

    _rebuild(repo, index_path)

    with pytest.raises(server_app.ToolError) as caught:
        source.connect()
    assert caught.value.code == "index_replaced"
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    assert source._embed_checked is False


def test_every_refusal_code_the_server_can_raise_is_in_the_documented_table():
    """The codes are the branchable half of the agent-facing contract, and a code that
    exists in the code and in no table is one an agent meets for the first time at
    runtime with nothing to match it against. Several arrived that way -- `bad_path`,
    `file_too_large`, `too_many_spans`, `claim_too_long`, `bad_confidence`,
    `schema_mismatch`, `span_escapes_repo`, `index_replaced` -- and `unknown_subject`
    was undocumented before any of them. Reading the raises out of the source is what
    makes this fail on the next one instead of drifting again."""
    source = (PROJECT_ROOT / "codelearner" / "server" / "app.py").read_text()
    raised = {
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ToolError"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    # The store's refusals reach `ToolError` through a lookup rather than a literal,
    # so they are invisible to the scan above and are exactly the family most likely
    # to grow.
    raised |= set(server_app._STORE_REFUSAL_CODES.values())
    assert raised == set(server_app.ERROR_CODES)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("index_stats", {}),
        ("search_code", {"query": QUERY}),
        ("get_symbol", {"qualname": "core.frobnicate_widgets"}),
    ],
)
def test_an_index_from_another_schema_is_data_and_not_a_transport_error(
    served, tool, arguments
):
    """The most predicted failure in the design, and it went out as a traceback.

    `db.connect` refuses an index whose stamp is not the current `SCHEMA_VERSION` --
    that check is the whole reason a stale index cannot answer a query -- and it
    refuses with a `RuntimeError`, which is neither a `ToolError` nor a
    `sqlite3.Error`. So it walked straight through `_guard` and into the transport,
    where an agent reads it as "this tool is broken" and stops calling it. The stamp
    has moved five times, and the realistic sequence is an agent session left open
    across the re-index that follows the sixth.
    """
    _, index_path, server = served
    conn = db.connect(index_path, check_schema=False)
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.close()

    payload = call(server, tool, **arguments)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "schema_mismatch"
    assert "v4" in payload["error"]["message"]
    # The remedy has to travel with the code: the agent cannot re-index, and the
    # human who can needs to know it no longer costs them the assertion store.
    assert "--carry-assertions" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------

def test_search_returns_tier_labelled_hits_with_locations_and_hashes(served):
    """Tier, qualname and file:line are the answer; the hash is what makes citing it
    possible without a second call. Drop `content_hash` from the hit and the
    retrieval-to-citation loop no longer closes -- an agent would have to hash bytes
    itself, which it cannot reliably do."""
    _, _, server = served
    payload = call(server, "search_code", query=QUERY, k=5)
    assert payload["ok"] is True
    assert payload["count"] >= 1
    top = payload["hits"][0]
    assert top["qualname"] == "core.frobnicate_widgets"
    assert top["tier"] in {"T0", "T1"}
    assert top["path"] == "core.py"
    assert top["line_start"] >= 1
    assert len(top["content_hash"]) == 64
    # Always present, empty when the hit was not reached by graph expansion. A
    # consumer must not have to probe for the key to learn whether one exists.
    assert "via" in top


def test_search_without_source_options_keeps_the_compact_response_contract(served):
    """The opt-in evidence response must not quietly reshape the default payload."""
    _, _, server = served

    payload = call(server, "search_code", query=QUERY)

    assert set(payload) == {"ok", "query", "k", "facts_only", "count", "notes", "hits"}
    assert "evidence" not in payload


def test_search_with_source_returns_the_complete_top_symbol_as_evidence(served):
    """Requested source is whole-symbol, line-numbered, and identifies a returned hit."""
    _, _, server = served

    payload = call(
        server,
        "search_code",
        query="frobnicate widgets",
        k=2,
        include_source=True,
        evidence_budget=4096,
    )

    evidence = payload["evidence"]
    assert set(evidence) == {
        "budget_bytes",
        "used_bytes",
        "truncated",
        "sections_omitted",
        "omitted_symbol_ids",
        "sections",
    }
    assert evidence["sections"][0]["source"] == (
        "1 | def frobnicate_widgets():\n"
        "2 |     \"\"\"Frobnicate every widget on the tray.\"\"\"\n"
        "3 |     return _plumbing()"
    )
    section_ids = {section["symbol_id"] for section in evidence["sections"]}
    hit_ids = {hit["symbol_id"] for hit in payload["hits"]}
    assert evidence["sections"][0]["symbol_id"] == payload["hits"][0]["symbol_id"]
    assert section_ids <= hit_ids


def test_search_with_source_and_zero_budget_reports_every_hit_as_omitted(served):
    """A zero byte source allowance returns explicit omission metadata, not partial text."""
    _, _, server = served

    payload = call(server, "search_code", query=QUERY, k=2, include_source=True, evidence_budget=0)

    assert payload["evidence"]["budget_bytes"] == 0
    assert payload["evidence"]["used_bytes"] == 0
    assert payload["evidence"]["sections"] == []
    assert payload["evidence"]["omitted_symbol_ids"] == [
        hit["symbol_id"] for hit in payload["hits"]
    ]


def test_search_with_source_clamps_evidence_budget_to_the_byte_ceiling(served):
    """An oversized request cannot turn MCP retrieval into an unbounded source read."""
    _, _, server = served

    payload = call(
        server,
        "search_code",
        query=QUERY,
        include_source=True,
        evidence_budget=65_537,
    )

    assert payload["evidence"]["budget_bytes"] == 65_536


def test_search_with_source_refuses_source_changed_since_indexing(served):
    """Indexed bytes are not presented as current source after an edit on disk."""
    repo, _, server = served
    (repo / "core.py").write_text(CORE.replace("every widget on the tray", "nothing at all"))

    payload = call(server, "search_code", query=QUERY, include_source=True)

    assert payload["ok"] is False
    assert payload["error"]["code"] == "evidence_unavailable"
    assert "Source evidence could not be assembled." in payload["error"]["message"]
    serialized = json.dumps(payload, sort_keys=True)
    assert str(repo) not in serialized
    assert str(repo / "core.py") not in serialized


def test_search_without_source_does_not_read_a_deleted_indexed_file(served):
    """Compact search remains available when no source evidence was requested."""
    repo, _, server = served
    (repo / "core.py").unlink()

    payload = call(server, "search_code", query=QUERY, include_source=False)

    assert payload["ok"] is True
    assert payload["hits"]
    assert "evidence" not in payload


def test_graph_reached_hits_are_tier_1_and_explain_themselves(served):
    """`_plumbing` shares no vocabulary with the query, so the only way to it is the
    resolved call edge -- and a hit that arrived that way is T1, not T0, because
    reaching it depended on a name binding that can be wrong."""
    _, _, server = served
    hits = {h["qualname"]: h for h in call(server, "search_code", query=QUERY, k=10)["hits"]}
    plumbing = hits.get("core._plumbing")
    assert plumbing is not None, "graph expansion did not surface the callee"
    assert plumbing["tier"] == "T1"
    assert "graph" in plumbing["modality"]
    assert plumbing["via"], "a graph hit with no account of how it was reached"


def test_facts_only_excludes_tier_2(served, monkeypatch):
    """The whole promise of the flag. Nothing retrieves at T2 yet, so the T2 hit is
    injected at the retrieval seam -- which is exactly where one will arrive when the
    inference layer starts serving. Delete the `facts_only_filter` call and this
    fails, because the inferred hit comes straight through."""
    _, _, server = served
    fact = Hit(
        symbol_id=1,
        qualname="core.frobnicate_widgets",
        kind="function",
        path="core.py",
        line_start=1,
        line_end=3,
        score=1.0,
        modality="lexical",
        header="",
    )
    inferred = Hit(
        symbol_id=2,
        qualname="core._plumbing",
        kind="function",
        path="core.py",
        line_start=6,
        line_end=8,
        score=0.5,
        modality="inferred",
        header="",
    )
    monkeypatch.setattr(
        server_app,
        "search",
        lambda *a, **kw: SearchResult(hits=[fact, inferred], per_modality={}),
    )

    everything = call(server, "search_code", query=QUERY, facts_only=False)
    assert [h["tier"] for h in everything["hits"]] == ["T0", "T2"]

    facts = call(server, "search_code", query=QUERY, facts_only=True)
    assert [h["tier"] for h in facts["hits"]] == ["T0"]
    assert facts["count"] == 1


def _submit(server, qualname, claim, **kw):
    """Store one assertion about `qualname` through the shipped write surface."""
    good_hash, line_start, line_end = _hash_of(server, qualname)
    payload = call(
        server,
        "submit_assertion",
        subject_qualname=qualname,
        claim=claim,
        evidence_spans=[
            {
                "path": qualname.split(".")[0] + ".py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
        **kw,
    )
    assert payload["ok"] is True, payload
    return payload


def test_facts_only_on_get_symbol_withholds_the_only_tier_2_the_server_returns(served):
    """The flag on the surface where tier 2 actually appears.

    `search_code`'s `facts_only` cannot drop anything: no modality retrieves at tier 2,
    so the filter runs over a list that never contains one (the test above has to
    inject a T2 hit at the retrieval seam to give it something to do). Stored
    assertions are the only T2 content this server returns anywhere, and they come
    back from `get_symbol` -- so this is the call where "parsed facts and resolved
    names, nothing asserted" is either true or a slogan.

    Delete the `facts_only` branch in `_get_symbol_body` and this fails: the assertion
    comes straight back.
    """
    _, _, server = served
    _submit(server, "core.frobnicate_widgets", "frobnicates every widget on the tray")

    everything = call(server, "get_symbol", qualname="core.frobnicate_widgets")
    assert [a["tier"] for a in everything["assertions"]] == ["T2"]
    assert everything["facts_only"] is False
    assert everything["assertions_withheld"] == 0

    facts = call(server, "get_symbol", qualname="core.frobnicate_widgets", facts_only=True)
    assert facts["assertions"] == []
    assert facts["facts_only"] is True
    assert facts["assertions_withheld"] == 1


def test_facts_only_on_get_symbol_says_that_something_was_withheld(served):
    """A suppressed claim must not be an invisible one.

    An empty `assertions` list has two very different meanings -- "nothing has been
    asserted about this symbol" and "you asked not to see what has" -- and a caller
    that cannot tell them apart will read the first when the second is true. That is
    the same failure mode as a cached freshness verdict: a confident answer standing
    in for one nobody looked at.
    """
    _, _, server = served
    _submit(server, "core.frobnicate_widgets", "frobnicates every widget on the tray")

    facts = call(server, "get_symbol", qualname="core.frobnicate_widgets", facts_only=True)
    assert any("facts_only" in note and "withheld" in note for note in facts["notes"]), (
        facts["notes"]
    )

    # A symbol with nothing asserted about it does NOT get the note, so its presence
    # carries information rather than being boilerplate on every facts-only call.
    quiet = call(server, "get_symbol", qualname="core._plumbing", facts_only=True)
    assert quiet["assertions"] == []
    assert quiet["assertions_withheld"] == 0
    assert not any("withheld" in note for note in quiet["notes"])


def test_facts_only_on_get_symbol_changes_nothing_but_the_assertions(served):
    """The flag is about tier 2 and only tier 2.

    Everything else `get_symbol` returns is T0 (the symbol, unresolved call sites) or
    T1 (resolved callers and callees), and a flag that quietly dropped a resolved edge
    as well would be a second, undocumented filter riding on the first.
    """
    _, _, server = served
    _submit(server, "core.frobnicate_widgets", "frobnicates every widget on the tray")

    everything = call(server, "get_symbol", qualname="core.frobnicate_widgets")
    facts = call(server, "get_symbol", qualname="core.frobnicate_widgets", facts_only=True)

    for key in ("symbol", "callers", "callees", "unresolved_calls", "duplicate_qualnames"):
        assert facts[key] == everything[key], key


def test_facts_only_still_expires_a_claim_it_is_not_going_to_show_you(served):
    """Withheld is not unchecked.

    The claims are fetched and re-verified before being dropped, so a facts-only
    caller gets the same account of the index's state as anyone else -- and an
    assertion whose evidence has moved is expired on the way past rather than left
    active for the next caller who does want to read it. Suppressing output must not
    also suppress the verification, or `facts_only=True` becomes a way to keep a stale
    claim alive.
    """
    repo, index_path, server = served
    _submit(server, "core.frobnicate_widgets", "frobnicates every widget on the tray")

    (repo / "core.py").write_text(CORE.replace("every widget on the tray", "nothing at all"))
    facts = call(server, "get_symbol", qualname="core.frobnicate_widgets", facts_only=True)
    assert facts["assertions"] == []
    # Nothing was withheld -- it was expired, which is a different fact and the one
    # that matters. The count separates the two.
    assert facts["assertions_withheld"] == 0

    conn = db.connect(index_path)
    try:
        assert [a.claim for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [
            "frobnicates every widget on the tray"
        ]
    finally:
        conn.close()


def test_search_says_why_dense_is_missing_instead_of_silently_dropping_it(served):
    """An index with no vectors still answers, and says so. Silence here reads as
    "dense found nothing", which is a different and much worse claim."""
    _, _, server = served
    payload = call(server, "search_code", query=QUERY)
    assert any("no embeddings" in note for note in payload["notes"])


def test_k_is_clamped_rather_than_honoured(served):
    """`k` drives a candidate depth of 4k out of FTS5 plus a graph expansion seeded
    from all of it. An agent that passes a huge k by accident should get a clamped
    answer, not a server that stops responding."""
    _, _, server = served
    assert call(server, "search_code", query=QUERY, k=10_000)["k"] == server_app.MAX_K
    assert call(server, "search_code", query=QUERY, k=0)["k"] == 1


# ---------------------------------------------------------------------------
# get_symbol
# ---------------------------------------------------------------------------

def test_get_symbol_returns_resolved_callers_and_callees(served):
    """Both directions, both tier 1, each carrying the confidence its resolver
    assigned. A caller shown without its confidence looks like a fact."""
    _, _, server = served
    caller = call(server, "get_symbol", qualname="core.frobnicate_widgets")
    assert caller["ok"] is True
    assert caller["symbol"]["tier"] == "T0"
    assert [c["qualname"] for c in caller["callees"]] == ["core._plumbing"]
    assert caller["callees"][0]["tier"] == "T1"
    assert caller["callees"][0]["confidence"] is not None

    callee = call(server, "get_symbol", qualname="core._plumbing")
    assert [c["qualname"] for c in callee["callers"]] == ["core.frobnicate_widgets"]
    assert callee["callees"] == []


def test_get_symbol_reports_an_unknown_qualname_as_data(served):
    """A typo'd qualname is the most likely single failure on this tool, and the
    remedy -- use search_code -- has to travel with the refusal."""
    _, _, server = served
    payload = call(server, "get_symbol", qualname="core.no_such_thing")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "no_such_symbol"
    assert "search_code" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# reading_path
# ---------------------------------------------------------------------------

def test_reading_path_orders_dependencies_before_their_callers(served):
    """The tour's one substantive promise: what a stop calls comes before the stop.
    A tour that lists a function above the helper it depends on is just a ranking."""
    _, _, server = served
    payload = call(server, "reading_path", limit=5)
    assert payload["ok"] is True
    order = {stop["qualname"]: stop["order"] for stop in payload["stops"]}
    assert order["core._plumbing"] < order["core.frobnicate_widgets"]
    frob = next(s for s in payload["stops"] if s["qualname"] == "core.frobnicate_widgets")
    assert "core._plumbing" in frob["read_before_this"]
    assert frob["reason"]


def test_reading_path_without_a_topic_never_loads_a_model(served, monkeypatch):
    """The repo-wide tour is pure call-graph centrality -- it asks a model no
    question, so it must not pay for one. Loading 1.2GB of weights to rank a graph
    would make the default invocation the expensive one."""
    _, index_path, _ = served

    def explode(name: str):
        raise AssertionError("the untopiced reading path loaded an embedder")

    server = build_server(index_path, embedder_factory=explode)
    assert call(server, "reading_path", limit=3)["ok"] is True


# ---------------------------------------------------------------------------
# submit_assertion -- the gate
# ---------------------------------------------------------------------------

def test_submit_assertion_rejects_zero_evidence(served):
    """Rule one, and the one that matters most: an uncited claim cannot be
    adjudicated, cannot expire, and cannot be checked by a reader. It is
    indistinguishable from a good claim at every stage after this, so the only place
    to stop it is the door. Delete the `EvidenceRequired` path and an unciteable row
    lands in the same table as the citeable ones."""
    repo, index_path, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        evidence_spans=[],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "evidence_required"

    from codelearner import db

    conn = db.connect(index_path)
    assert conn.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
    conn.close()


def test_submit_assertion_rejects_a_perfectly_cited_empty_claim(served):
    """The same rule as zero evidence, reached through the opposite door: real
    citations under no statement at all. Nothing this tool checks would catch it --
    the span exists and hashes correctly -- so before the rule moved into
    `write_assertion` this stored `active`, reported servable, and came back out of
    `get_symbol` as an empty string beside the code it was allegedly about.

    Refused by the store rather than by a check here, which is why it is refused for
    `codelearner learn` and every library caller too."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="   \n ",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "empty_claim"

    from codelearner import db

    conn = db.connect(index_path)
    assert conn.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
    conn.close()


def test_submit_assertion_rejects_a_hash_mismatch(served):
    """Rule two. The agent read the file, the file changed, and the citation it is
    about to store no longer describes anything. sha256 is not arguable, which is the
    entire reason the gate can be trusted with an agent's output. Delete the hash
    comparison and a fabricated or stale citation is stored as evidence."""
    repo, index_path, server = served
    stale_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")

    # Same shape, different bytes -- so the line range stays valid and only the
    # content moves. That is the realistic failure: an edit, not a deleted file.
    (repo / "core.py").write_text(CORE.replace("every widget on the tray", "nothing at all"))

    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": stale_hash,
            }
        ],
    )
    assert payload["ok"] is False
    error = payload["error"]
    assert error["code"] == "hash_mismatch"
    # The refusal has to be correctable, not just correct: it names the citation, the
    # hash that is actually there, and the text -- so the next attempt is a fix.
    assert error["cited_hash"] == stale_hash
    assert error["observed_hash"] != stale_hash
    assert "nothing at all" in error["observed_text"]

    from codelearner import db

    conn = db.connect(index_path)
    assert conn.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
    conn.close()


def test_every_indexed_symbol_can_be_cited_by_the_hash_it_was_handed(decorated):
    """The loop the README promises -- retrieve, cite what you retrieved -- must hold
    for EVERY symbol, not just the convenient ones.

    A symbol's stored bytes are not its lines' bytes. The parser records the symbol
    node, which starts at `def` rather than in the indentation before it, at the `@`
    for a decorated symbol, and runs to the last byte of the file for a module --
    one line past the last line anything is written on. Measured on code-learner
    itself: 85 of 383 symbols, about 15%, where the two disagree.

    Checking a citation only against the whole-lines reading rejects the exact hash
    this server just published, for every one of them, with a message accusing the
    agent of citing something that had changed. Delete the `_symbol_bytes_at` lookup
    and this test fails on the decorated method, the property, and the module."""
    _, conn, server = decorated
    symbols = conn.execute(
        "SELECT s.qualname, s.kind, s.line_start, s.line_end, s.content_hash, f.path "
        "FROM symbols s JOIN files f ON f.id = s.file_id ORDER BY s.id"
    ).fetchall()
    kinds = {row["kind"] for row in symbols}
    assert {"module", "class", "method", "function"} <= kinds, "fixture lost its coverage"

    refused = []
    for row in symbols:
        payload = call(
            server,
            "submit_assertion",
            subject_qualname=row["qualname"],
            claim=f"a claim about {row['qualname']}",
            evidence_spans=[
                {
                    "path": row["path"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "content_hash": row["content_hash"],
                }
            ],
        )
        if not payload["ok"]:
            refused.append((row["qualname"], row["kind"], payload["error"]["code"]))
    assert refused == [], f"the gate refused its own published hashes: {refused}"


def test_a_stale_hash_is_still_refused_for_a_decorated_symbol(decorated):
    """The other half of the fix above. Accepting two readings of a line range is only
    safe because both are re-hashed off disk -- so an invented hash matches neither.
    Delete that and the widened check becomes a hole rather than a repair."""
    _, conn, server = decorated
    row = conn.execute(
        "SELECT s.qualname, s.line_start, s.line_end, f.path FROM symbols s "
        "JOIN files f ON f.id = s.file_id WHERE s.qualname = 'tray.Tray.count'"
    ).fetchone()
    payload = call(
        server,
        "submit_assertion",
        subject_qualname=row["qualname"],
        claim="counts the widgets",
        evidence_spans=[
            {
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "content_hash": "0" * 64,
            }
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "hash_mismatch"
    # Both readings are reported, so an agent can see which one it was aiming at.
    assert len(payload["error"]["observed_hashes"]) == 2


def test_submit_assertion_accepts_a_citation_that_still_hashes(served):
    """The gate has to admit as well as refuse, or it is just an off switch. The hash
    comes straight back out of `get_symbol`, which is the loop the whole design
    rests on: retrieve, cite what you retrieved, be checked."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")

    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray by delegating to _plumbing",
        kind="purpose",
        generator="test-agent/v1",
        confidence=0.8,
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert payload["tier"] == "T2"
    assert payload["servable"] is True
    assert payload["subject_symbol_id"] is not None
    assert payload["evidence"][0]["citation"] == f"core.py:{line_start}-{line_end}"

    # And it is servable through the read surface, verified against disk on the way.
    served_back = call(server, "get_symbol", qualname="core.frobnicate_widgets")["assertions"]
    assert [a["claim"] for a in served_back] == [
        "frobnicates every widget on the tray by delegating to _plumbing"
    ]
    assert served_back[0]["tier"] == "T2"
    assert served_back[0]["evidence"][0]["content_hash"] == good_hash


def test_a_stored_assertion_stops_being_served_when_its_evidence_moves(served):
    """Admission is not a permanent licence. `servable_assertions` re-hashes on every
    read, so an edit under a stored claim expires it at the moment somebody asks --
    with no window in which the index serves a claim it already knows is wrong."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    accepted = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert accepted["ok"] is True

    (repo / "core.py").write_text(CORE.replace("every widget on the tray", "nothing at all"))
    assert call(server, "get_symbol", qualname="core.frobnicate_widgets")["assertions"] == []

    # Expired, not deleted: the claim and what it used to rest on are still there.
    from codelearner import db

    conn = db.connect(index_path)
    stale = store.assertions_with_status(conn, store.STATUS_STALE)
    assert [a.id for a in stale] == [accepted["assertion_id"]]
    conn.close()
    assert call(server, "index_stats")["assertions_by_status"]["stale"] == 1


def test_submit_assertion_accepts_quoted_text_in_place_of_a_hash(served):
    """An agent can copy the lines it read; it cannot reliably compute a sha256 in its
    head. Accepting the text and hashing it here keeps the gate arithmetic while
    leaving citation possible for spans retrieval never handed back."""
    repo, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="returns the constant 42",
        evidence_spans=[{"path": "core.py", "line_start": 8, "line_end": 8, "text": "    return 42"}],
    )
    assert payload["ok"] is True
    assert payload["servable"] is True


def test_quoted_text_that_is_not_what_the_file_says_is_refused(served):
    """The same rule as a bad hash, reached the other way. Text is only evidence
    while it is the text that is actually there."""
    _, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="returns the constant 99",
        evidence_spans=[{"path": "core.py", "line_start": 8, "line_end": 8, "text": "    return 99"}],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "hash_mismatch"


def test_a_span_with_nothing_to_check_against_is_refused(served):
    """A location with no assertion about what is there can never be found to be
    wrong -- it would verify forever while pointing at whatever the file becomes.
    That is the vacuous-truth failure the store's own empty-evidence guard exists
    for, one level up."""
    _, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="returns a constant",
        evidence_spans=[{"path": "core.py", "line_start": 6, "line_end": 8}],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "evidence_unverifiable"


def test_one_bad_span_refuses_the_whole_submission(served):
    """Admitting the spans that happened to verify would leave a claim standing on a
    subset of the evidence its author thought it had -- and nothing would record that
    the rest was dropped."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="delegates to _plumbing, which returns 42",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            },
            {"path": "core.py", "line_start": 8, "line_end": 8, "text": "    return 99"},
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "hash_mismatch"

    from codelearner import db

    conn = db.connect(index_path)
    assert conn.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
    conn.close()


def test_a_citation_outside_the_repo_is_refused(served):
    """Spans are re-read off disk by path. Without this the tool would read any file
    the server process can reach and store its hash as evidence about this repo."""
    _, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="the host has an /etc/passwd",
        evidence_spans=[
            {"path": "../../../etc/passwd", "line_start": 1, "line_end": 1, "text": "root"}
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] in {"path_escapes_repo", "file_missing"}


def test_a_citation_past_the_end_of_the_file_is_refused(served):
    """Lines that do not exist are not evidence, and slicing past the end of a bytes
    object silently returns a short result rather than failing."""
    _, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="there is a hidden line 900",
        evidence_spans=[
            {"path": "core.py", "line_start": 900, "line_end": 901, "text": "whatever"}
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad_range"


# ---------------------------------------------------------------------------
# submit_assertion -- the refusal, which is the half that reads files
# ---------------------------------------------------------------------------

# S105: it is meant to look exactly like a leaked credential -- that is the point of
# the fixture. AWS's own documentation example key, which is not a key.
SECRET = "AKIAIOSFODNN7EXAMPLE-not-a-real-key"  # noqa: S105

# One symbol whose bytes are comfortably past the 2048-character cap on quoted text,
# so that the truncation is exercised by a legitimately large function rather than by
# lowering the limit for the test.
WIDE = "def widgets():\n" + "".join(f'    x{i} = "{"y" * 48}"\n' for i in range(64))


@pytest.fixture
def with_a_secret(tmp_path):
    """An indexed repo that also contains a file the index never parsed.

    `.env` is the auditor's actual target and is not a contrivance: every repo has
    files git tracks, the indexer skips, and nobody expects a code-search tool to
    read. The point of the fixture is that `core.py` and `.env` sit in the same
    directory under the same repo root, so nothing about the path distinguishes them.
    """
    repo = _mkrepo(tmp_path / "repo", {"core.py": CORE, "wide.py": WIDE})
    (repo / ".env").write_text(f"AWS_SECRET_ACCESS_KEY={SECRET}\n")
    index_path = tmp_path / "index.db"
    conn, _ = index_repo(repo, index_path=index_path)
    conn.close()
    return repo, index_path, build_server(index_path, embedder_factory=FakeEmbedder)


def _cite(server: Any, path: str, line_start: int, line_end: int) -> dict[str, Any]:
    """Submit a deliberately wrong hash for a range -- the read-oracle manoeuvre.

    A wrong `content_hash` is guaranteed to fail the gate, and failing the gate is
    how the caller gets the file's bytes quoted back at it. Walk the line range and
    the whole file arrives one refusal at a time.
    """
    return call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="a claim",
        evidence_spans=[
            {
                "path": path,
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": "0" * 64,
            }
        ],
    )


def test_an_unindexed_file_inside_the_repo_is_not_read_into_the_refusal(with_a_secret):
    """The gate was never defeated here -- the refusal was the exfiltration channel.

    `path_escapes_repo` correctly stops `../../etc/passwd`. Inside the root there was
    no check at all: any file the server process could open was read, decoded and
    returned as `observed_text` on a hash mismatch. An auditor pulled `.env` secrets
    and an SSH private key out of error messages by submitting a wrong hash and
    walking the line ranges. Delete the `files` lookup in `_verify_span` and the
    secret is in this payload again.

    The refusal must also not become the oracle it replaced, which is why an absent
    path is asserted to give the SAME code as a present-but-unindexed one. A refusal
    that said `file_missing` for one and something else for the other would still
    answer "does this file exist", one guess at a time."""
    _, _, server = with_a_secret

    payload = _cite(server, ".env", 1, 1)
    # The leak first, before anything about codes: this assertion is the finding, and
    # a test that checked the code first would report a renamed error rather than a
    # recovered credential.
    assert SECRET not in json.dumps(payload), "the refusal handed back the secret"
    assert "observed_text" not in payload["error"]
    assert payload["ok"] is False
    assert payload["error"]["code"] == "file_missing"

    absent = _cite(server, "no-such-file-anywhere.py", 1, 1)
    assert absent["error"]["code"] == payload["error"]["code"]
    assert absent["error"]["message"].replace("no-such-file-anywhere.py", ".env") == (
        payload["error"]["message"]
    ), "the two refusals differ, which is enough to answer 'does this file exist'"


def test_the_quoted_bytes_in_a_refusal_are_capped_and_say_they_were_cut(with_a_secret):
    """Quoting the bytes is what makes a `hash_mismatch` correctable rather than just
    correct, so the cap is on the size of the answer and not on whether there is one.
    Unbounded, one refusal returns as much text as the cited range holds -- which is
    the whole file when the range is the module.

    The marker is not decoration. Text silently cut at 2048 characters is a false
    statement about what the file says at those lines, and an agent that trusts it
    will resubmit a citation built from a half-read line."""
    _, _, server = with_a_secret
    _, line_start, line_end = _hash_of(server, "wide.widgets")
    payload = _cite(server, "wide.py", line_start, line_end)

    assert payload["ok"] is False
    assert payload["error"]["code"] == "hash_mismatch"
    text = payload["error"]["observed_text"]
    # Literals, deliberately, and checked against the module's constants only
    # afterwards. A test written as `len(text) == MAX_OBSERVED_TEXT_CHARS + ...` goes
    # red against code that has no such constant with an AttributeError, which proves
    # the constant is new and proves nothing at all about the behaviour.
    assert len(WIDE) > 2048, "fixture is too small to be cut"
    assert len(text) <= 2100, f"quoted {len(text)} characters back, uncapped"
    assert "truncated" in text[-32:], "cut without saying so"

    assert server_app.MAX_OBSERVED_TEXT_CHARS == 2048
    assert len(text) == server_app.MAX_OBSERVED_TEXT_CHARS + len(server_app.TRUNCATION_MARKER)
    assert text.endswith(server_app.TRUNCATION_MARKER)


def test_a_file_grown_past_the_ceiling_is_refused_before_it_is_read(with_a_secret):
    """Refused on `st_size`, before `read_bytes`, because the read is the cost being
    avoided. One call against a large file returned a 209,715,200-character payload at
    ~479MB peak RSS -- the entire file, decoded, inside an error message.

    The file is indexed and then grown, rather than indexed large, because that is the
    reachable path: the ceiling has to hold against what is on disk now, not against
    what the indexer saw."""
    repo, _, server = with_a_secret
    # A literal 8MiB rather than `MAX_CITED_FILE_BYTES + 1`, so that this goes red
    # against code with no ceiling by refusing to refuse -- not by failing to find a
    # constant that only the fix introduces.
    (repo / "core.py").write_text(f"# {SECRET}\n" + "x = 1\n" * ((8 * 1024 * 1024) // 6))

    payload = _cite(server, "core.py", 1, 1)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "file_too_large"
    assert SECRET not in json.dumps(payload)

    assert server_app.MAX_CITED_FILE_BYTES == 4 * 1024 * 1024
    assert payload["error"]["limit"] == server_app.MAX_CITED_FILE_BYTES


# ---------------------------------------------------------------------------
# the transport contract: no predictable condition raises into it
# ---------------------------------------------------------------------------

def _within(seconds: float, fn, *args, **kwargs):
    """Run `fn`, failing the test rather than the suite if it never comes back.

    A blocked `open()` on a FIFO is the thing under test, so these tests have to be
    able to survive the code being wrong -- and a wrong answer here is not an
    exception, it is a call that never returns. Signals were the obvious tool and do
    not work: the tool body runs on a worker thread inside the MCP server, so SIGALRM
    is delivered to a main thread that is merely waiting, and the blocked read is
    never interrupted (measured against the unfixed code -- the run hung).

    A daemon thread is abandoned instead. It stays blocked on the pipe for the rest of
    the session and is killed at interpreter exit without being joined, so a
    regression costs one failed test and one leaked thread rather than a suite that
    has to be killed from outside."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(
            f"blocked for more than {seconds}s -- a citation read is waiting on a FIFO "
            "that nothing will ever write to, which in the real server means it has "
            "stopped answering entirely"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX")


def test_a_nul_in_a_citation_path_does_not_raise_into_the_transport(served):
    """"No predictable condition raises into the transport" is the module's own first
    rule and the README's stated guarantee, and a NUL byte in a path broke it: the
    path reached `Path.resolve`, which raised `ValueError: embedded null byte`, and a
    traceback crossed the MCP boundary. That is the one result that tells an agent the
    tool itself is broken, which is the one conclusion that stops it retrying.

    `bad_path` rather than the generic `bad_request` `_guard` now falls back to: the
    refusal names the field, so the fix is visible."""
    _, _, server = served
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="a claim",
        evidence_spans=[
            {"path": "core\x00.py", "line_start": 1, "line_end": 1, "content_hash": "0" * 64}
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad_path"


def test_a_nul_in_a_search_query_does_not_make_the_server_accuse_its_own_index(served):
    """The same guarantee, broken more quietly. Quoting a term does not survive a NUL:
    SQLite stops at the NUL, never sees the closing quote this code appended, and FTS5
    raises `unterminated string`. That is an `sqlite3.Error`, so `_guard` reported
    `index_unreadable` -- the server telling its caller to re-index a database with
    nothing whatsoever wrong with it, because somebody pasted a control byte into a
    search box. A misdiagnosis that costs an hour is worse than a crash that costs a
    minute."""
    _, _, server = served
    payload = call(server, "search_code", query="frobnicate\x00widgets")
    assert payload.get("error", {}).get("code") != "index_unreadable"
    assert payload["ok"] is True
    assert [h["qualname"] for h in payload["hits"]][:1] == ["core.frobnicate_widgets"]


@needs_fifo
def test_a_cited_fifo_is_refused_rather_than_blocking_the_server(served):
    """The failure that does not look like a failure. `read_bytes` on a FIFO blocks
    until another process opens the write end; this server is single-threaded, so one
    citation of a named pipe inside the repo root stops it answering anything, with no
    exception, no log line and no timeout. Every other unreadable thing -- a missing
    file, a directory -- raises promptly, which is why `except OSError` was believed to
    cover this and does not.

    The pipe replaces an indexed file so that the citation gets past the `files`
    lookup: the two guards are independent and this one has to hold on its own."""
    repo, _, server = served
    (repo / "core.py").unlink()
    os.mkfifo(repo / "core.py")

    payload = _within(10, _cite, server, "core.py", 1, 3)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "file_missing"


@needs_fifo
def test_a_fifo_under_a_stored_claim_is_withheld_instead_of_blocking_the_read(served):
    """The other side of the same defect, and the one that reaches further: the serve
    path re-reads every cited file on every `get_symbol`, so a FIFO left where a cited
    file used to be wedges reads, not just writes. Reading it must not block, and the
    claim must not be served -- neither of those is negotiable.

    What it must NOT do is expire the claim. A FIFO is not an absent file; it is a
    file this process could not read, in the same class as `EACCES`, `EIO`, an NFS
    blip, or a repo that moved. `stale` means the cited bytes changed, and nothing in
    this codebase ever moves an assertion back to `active` -- so recording "we could
    not look" as "the evidence changed" made a transient condition into a permanent,
    irreversible loss of the claim and of the record of why. Withholding is the
    reversible disposition: the claim is not served on this call, its status is not
    touched, and the next call after the FIFO is gone serves it again.
    """
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    accepted = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert accepted["ok"] is True

    (repo / "core.py").unlink()
    os.mkfifo(repo / "core.py")

    # The no-hang guarantee. A cited FIFO used to wedge the single-threaded server
    # until some other process opened the write end: no exception, no log line, no
    # timeout -- it simply stopped answering.
    payload = _within(10, call, server, "get_symbol", qualname="core.frobnicate_widgets")
    assert payload["assertions"] == []

    conn = db.connect(index_path)
    try:
        # Withheld, not expired: no staleness event, and the claim is still active
        # and still serving-eligible the moment the file can be read again.
        assert store.staleness_events(conn, accepted["assertion_id"]) == []
        row = conn.execute(
            "SELECT status FROM assertions WHERE id = ?", (accepted["assertion_id"],)
        ).fetchone()
        assert row["status"] == store.STATUS_ACTIVE
    finally:
        conn.close()

    # And reversible is the whole argument, so it is asserted rather than described:
    # put the real bytes back and the claim is served again, with no repair step and
    # nothing to un-expire. Expiring it would have made this unreachable, because
    # nothing anywhere moves an assertion back to `active`.
    (repo / "core.py").unlink()
    (repo / "core.py").write_text(CORE)
    restored = call(server, "get_symbol", qualname="core.frobnicate_widgets")
    assert [a["claim"] for a in restored["assertions"]] == [
        "frobnicates every widget on the tray"
    ]


@needs_fifo
def test_span_for_refuses_a_fifo_with_its_own_exception_type(tmp_path):
    """`store` is a library and has no transport to be polite towards, so it raises
    `ValueError` -- what `span_for` already raises for a range that is not a citable
    one -- rather than importing the server's error type. Checked directly because the
    server guards this first, so no test that goes through the tool would notice this
    guard being deleted."""
    os.mkfifo(tmp_path / "pipe.py")
    with pytest.raises(ValueError, match="not a regular file"):
        _within(10, store.span_for, tmp_path, "pipe.py", 0, 10)


# ---------------------------------------------------------------------------
# submit_assertion -- what one call may cost the index forever
# ---------------------------------------------------------------------------

def _stored(index_path: Path) -> int:
    from codelearner import db

    conn = db.connect(index_path)
    try:
        return int(conn.execute("SELECT count(*) FROM assertions").fetchone()[0])
    finally:
        conn.close()


def test_a_submission_with_too_many_spans_is_refused(served):
    """Nothing in the store deletes, so this is not a transient cost: an assertion
    with 5,000 spans -- which the auditor stored in one call -- is 5,000 file reads on
    every later `get_symbol` that names its subject, for the life of the index, and
    the only repair is rebuilding it."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    span = {
        "path": "core.py",
        "line_start": line_start,
        "line_end": line_end,
        "content_hash": good_hash,
    }
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="a claim resting on the same span many times over",
        # 33, written out, so that unfixed code fails by accepting it rather than by
        # not having the constant this test would otherwise have counted from.
        evidence_spans=[dict(span) for _ in range(33)],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "too_many_spans"
    assert _stored(index_path) == 0

    assert server_app.MAX_EVIDENCE_SPANS == 32
    assert payload["error"]["limit"] == server_app.MAX_EVIDENCE_SPANS


def test_a_claim_longer_than_the_cap_is_refused(served):
    """A 5MB claim was accepted. It is returned in full by every `get_symbol` that
    reaches its subject, so one call permanently makes an unrelated read path expensive
    -- and the claim is not adjudicable by anything, which is the failure the evidence
    rule exists to prevent, arrived at by volume."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="x" * 4097,  # a literal, for the reason given in the spans test above
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "claim_too_long"
    assert _stored(index_path) == 0
    assert server_app.MAX_CLAIM_CHARS == 4096


@pytest.mark.parametrize("confidence", [1e308, -0.5, 2.0, float("inf"), float("nan")])
def test_a_confidence_that_is_not_a_probability_is_refused(served, confidence):
    """`confidence=1e308` was stored. It is read as the probability that a claim is
    right, so anything outside 0..1 makes every comparison against it meaningless --
    and `inf`/`nan` are worse than absurd, because they compare false against every
    threshold, so a claim carrying one sits permanently neither above nor below any
    filter later written over the column.

    Enforced in Python and not yet by a CHECK constraint: the DDL rides along with the
    next schema bump, because this project refuses to open an index whose version does
    not match and bumping twice would charge every user for the rebuild twice."""
    repo, index_path, server = served
    good_hash, line_start, line_end = _hash_of(server, "core.frobnicate_widgets")
    payload = call(
        server,
        "submit_assertion",
        subject_qualname="core.frobnicate_widgets",
        claim="frobnicates every widget on the tray",
        confidence=confidence,
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad_confidence"
    assert _stored(index_path) == 0


# ---------------------------------------------------------------------------
# index_stats
# ---------------------------------------------------------------------------

def test_index_stats_reports_content_by_tier(served):
    """The tier split is the point of the tool. Edges carry T0/T1; T2 lives in the
    assertion store, and its rejected and stale counts are reported with explicit
    zeros -- a rejected set that is merely absent cannot be distinguished from a gate
    that is not running."""
    _, _, server = served
    payload = call(server, "index_stats")
    assert payload["ok"] is True
    assert payload["counts"]["files"] == 1
    assert payload["counts"]["symbols"] >= 2
    assert payload["edges_by_tier"]["T1"] >= 1
    assert payload["edges_by_tier"]["T0"] + payload["edges_by_tier"]["T1"] == (
        payload["counts"]["edges"]
    )
    assert payload["assertions_by_status"] == {"active": 0, "rejected": 0, "stale": 0}
    assert payload["resolution"]["resolved"] >= 1
    assert payload["embeddings"]["present"] is False
    assert payload["repo_root"]


def test_index_stats_counts_an_admitted_assertion_as_active(served):
    """`active` moving is the only externally visible sign that the gate admitted
    something, which is what makes the pass rate measurable at all."""
    _, _, server = served
    good_hash, line_start, line_end = _hash_of(server, "core._plumbing")
    call(
        server,
        "submit_assertion",
        subject_qualname="core._plumbing",
        claim="returns 42",
        evidence_spans=[
            {
                "path": "core.py",
                "line_start": line_start,
                "line_end": line_end,
                "content_hash": good_hash,
            }
        ],
    )
    assert call(server, "index_stats")["assertions_by_status"]["active"] == 1
