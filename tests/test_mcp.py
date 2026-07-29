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

import asyncio
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the optional `mcp` dependency")

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
    }
    # The span shape has to survive into the schema, or the agent has no way to know
    # a citation needs a hash and guesses the field names.
    span_schema = by_name["submit_assertion"].input_schema["properties"]["evidence_spans"]
    assert span_schema["type"] == "array"


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
