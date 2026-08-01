"""Symbol-boundary chunking and lexical retrieval."""
from __future__ import annotations

import subprocess

from codelearner.chunk import chunk_for_symbol
from codelearner.ingest import index_repo
from codelearner.ingest.types import KIND_CLASS, KIND_FUNCTION, KIND_MODULE
from codelearner.retrieve import search_lexical
from codelearner.retrieve.lexical import escape_fts_query


def _mkrepo(root, files: dict[str, str]):
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


# --------------------------------------------------------------------------
# chunk shape
# --------------------------------------------------------------------------

SRC = b'def acquire(parcel_id, mode):\n    """Take a lease."""\n    return 1\n'


def test_a_function_chunk_carries_a_self_describing_header():
    """A bare body is not self-describing. `def acquire(...)` retrieved alone says
    nothing about which module it belongs to -- and the embedding loses that too."""
    text, header, truncated = chunk_for_symbol(
        source=SRC, path="pkg/leases.py", kind=KIND_FUNCTION,
        qualname="pkg.leases.acquire", byte_start=0, byte_end=len(SRC),
        signature="acquire(parcel_id, mode)", docstring="Take a lease.",
    )
    assert not truncated
    assert "pkg/leases.py" in header
    assert "pkg.leases.acquire" in header
    assert "signature: acquire(parcel_id, mode)" in header
    assert "Take a lease." in header
    assert text.startswith(header)
    assert "def acquire(parcel_id, mode):" in text


def test_a_function_chunk_is_never_split_mid_body():
    """The reason chunking happens after parsing rather than on raw text: the
    parser knows exactly where a symbol starts and ends."""
    text, _, _ = chunk_for_symbol(
        source=SRC, path="m.py", kind=KIND_FUNCTION, qualname="m.acquire",
        byte_start=0, byte_end=len(SRC),
    )
    body = text.split("\n", 1)[1] if "\n" in text else text
    assert body.count("def ") == 1
    assert text.rstrip().endswith("return 1")


def test_a_class_chunk_summarises_members_instead_of_inlining_them():
    """A class chunk holding every method would duplicate the per-method chunks and
    blur the class's own identity across everything it contains."""
    text, _, _ = chunk_for_symbol(
        source=b"class C:\n    def a(self): pass\n    def b(self): pass\n",
        path="m.py", kind=KIND_CLASS, qualname="m.C",
        byte_start=0, byte_end=50, docstring="A container.",
        members=["a", "b"],
    )
    assert "methods: a, b" in text
    assert "def a(self)" not in text


def test_a_module_chunk_lists_what_it_defines():
    text, _, _ = chunk_for_symbol(
        source=b"", path="m.py", kind=KIND_MODULE, qualname="m",
        byte_start=0, byte_end=0, docstring="Module doc.",
        members=["helper", "C"],
    )
    assert "defines: helper, C" in text
    assert "Module doc." in text


def test_an_oversized_symbol_is_truncated_visibly_not_silently():
    big = b"def f():\n" + b"    x = 1\n" * 20_000
    text, _, truncated = chunk_for_symbol(
        source=big, path="m.py", kind=KIND_FUNCTION, qualname="m.f",
        byte_start=0, byte_end=len(big),
    )
    assert truncated
    assert "truncated by code-learner" in text


# --------------------------------------------------------------------------
# building chunks over a repo
# --------------------------------------------------------------------------

def test_index_builds_one_chunk_per_non_empty_symbol(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "m.py": 'def helper():\n    """Help."""\n    return 1\n',
    })
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    assert stats.chunks > 0
    quals = {r["qualname"] for r in conn.execute(
        "SELECT s.qualname FROM chunks c JOIN symbols s ON s.id = c.symbol_id")}
    assert "m.helper" in quals


def test_a_decorated_function_chunk_shows_the_model_its_decorators(tmp_path):
    """The compounding half of the decorator defect. `generate/pipeline.py` drafts
    from the same bytes this chunk holds, so a chunk that started at `def` asked a
    model to describe a route handler it had never been shown the route of -- and
    then the retrieval index could not match `@route` either. The chunker takes the
    span straight from `symbols`, so this stays true only while extraction keeps the
    decorator inside the span, which is exactly what it is here to catch."""
    repo = _mkrepo(tmp_path / "r", {
        "app.py": '@route("/users")\n@cache(ttl=60)\ndef list_users():\n    return []\n',
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    text = conn.execute(
        "SELECT c.text FROM chunks c JOIN symbols s ON s.id = c.symbol_id "
        "WHERE s.qualname = 'app.list_users'"
    ).fetchone()["text"]
    assert '@route("/users")' in text
    assert "@cache(ttl=60)" in text


def test_a_module_with_nothing_in_it_produces_no_chunk(tmp_path):
    """A chunk that is only its own header can never be anything but a false
    positive in the retrieval set.

    Note the boundary: an empty `pkg/__init__.py` that *contains* submodules still
    earns a chunk, because "defines: m" is real retrievable content -- "what is in
    this package" is a question someone asks. Only a module with no docstring and
    no members is skipped."""
    repo = _mkrepo(tmp_path / "r", {
        "pkg/__init__.py": "",          # has a submodule -> keeps its chunk
        "pkg/m.py": "def f():\n    return 1\n",
        "empty.py": "",                 # nothing at all -> skipped
    })
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    quals = {r["qualname"] for r in conn.execute(
        "SELECT s.qualname FROM chunks c JOIN symbols s ON s.id = c.symbol_id")}
    assert "empty" not in quals
    assert "pkg" in quals
    assert stats.chunk.skipped_empty >= 1


def test_rebuilding_chunks_does_not_duplicate_them(tmp_path):
    from codelearner.chunk import build_chunks

    repo = _mkrepo(tmp_path / "r", {"m.py": "def f():\n    return 1\n"})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    before = conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    build_chunks(conn, repo)
    after = conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    assert before == after


def test_deleting_a_chunk_removes_it_from_the_fts_index(tmp_path):
    """REGRESSION guard on the FTS sync triggers. An external-content FTS5 index
    does not follow its source table on its own -- without the triggers a deleted
    chunk keeps matching queries forever."""
    repo = _mkrepo(tmp_path / "r", {"m.py": "def findme_unique_token():\n    return 1\n"})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert search_lexical(conn, "findme_unique_token")
    conn.execute("DELETE FROM chunks")
    assert search_lexical(conn, "findme_unique_token") == []


# --------------------------------------------------------------------------
# lexical retrieval
# --------------------------------------------------------------------------

def test_lexical_search_finds_a_symbol_by_its_terms(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "db.py": 'def connect():\n    """Open a WAL journal mode connection."""\n    return 1\n',
        "other.py": "def unrelated():\n    return 2\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    hits = search_lexical(conn, "WAL journal mode", k=5)
    assert hits
    assert hits[0].qualname == "db.connect"
    assert hits[0].path == "db.py"
    assert hits[0].modality == "lexical"


def test_lexical_scores_are_higher_is_better(tmp_path):
    """FTS5's bm25() is more-negative-is-better. Every modality in this package
    must agree that higher wins, or fusion silently ranks backwards."""
    repo = _mkrepo(tmp_path / "r", {
        "a.py": "def alpha():\n    # lease lease lease ttl\n    return 1\n",
        "b.py": "def beta():\n    # lease\n    return 2\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    hits = search_lexical(conn, "lease ttl", k=5)
    assert len(hits) >= 2
    assert hits[0].score >= hits[-1].score
    assert hits[0].score > 0


def test_query_punctuation_does_not_blow_up_the_fts_parser(tmp_path):
    """Raw user text hits FTS5 operator syntax: `foo(` is a syntax error and `a-b`
    silently becomes a NOT query. Both look like 'no results' rather than a bug."""
    repo = _mkrepo(tmp_path / "r", {"m.py": "def connect():\n    return 1\n"})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    for query in ["db.connect(", "connect()", 'a-b "quoted"', "^caret*", "  "]:
        search_lexical(conn, query, k=3)  # must not raise


def test_escape_fts_query_quotes_every_term():
    assert escape_fts_query("foo bar") == '"foo" OR "bar"'
    assert escape_fts_query("db.connect(") == '"db.connect"'
    assert escape_fts_query("   ") == '""'


def test_empty_query_returns_nothing_rather_than_everything(tmp_path):
    repo = _mkrepo(tmp_path / "r", {"m.py": "def f():\n    return 1\n"})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert search_lexical(conn, "", k=5) == []
    assert search_lexical(conn, "()*^", k=5) == []
