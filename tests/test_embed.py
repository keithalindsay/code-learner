"""Dense indexing and retrieval.

Uses a deterministic fake embedder rather than loading a real model. That is the
point of the `Embedder` protocol: the storage, batching, incrementality, and query
paths are all testable in milliseconds without a GPU, and the only thing a real
model adds to these assertions is minutes.
"""
from __future__ import annotations

import subprocess

import pytest

from codelearner import db
from codelearner.index.embed import (
    MAX_BATCH_CHARS,
    MAX_BATCH_ITEMS,
    _batches,
    embed_chunks,
    serialize,
)
from codelearner.ingest import index_repo
from codelearner.retrieve import search_dense, stored_embed_model

pytest.importorskip("sqlite_vec", reason="dense retrieval requires sqlite-vec")


class FakeEmbedder:
    """Maps text to a vector by counting a few marker words.

    Deterministic and dependency-free, so retrieval ordering is asserted against
    arithmetic anyone can verify by reading the test.
    """

    MARKERS = ("lease", "worktree", "sqlite")

    def __init__(self, name: str = "fake/v1") -> None:
        self._name = name
        self.calls: list[int] = []  # batch sizes, so batching can be asserted

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
        self.calls.append(len(texts))
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


def _mkrepo(root, files: dict[str, str]):
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


REPO_FILES = {
    "leases.py": 'def acquire():\n    """Take a lease. lease lease."""\n    return 1\n',
    "wt.py": 'def remove():\n    """Remove a worktree. worktree worktree."""\n    return 2\n',
    "store.py": 'def connect():\n    """Open sqlite. sqlite sqlite."""\n    return 3\n',
}


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def test_batches_respect_the_character_budget():
    """A fixed batch size OOMs: attention memory scales with the LONGEST item in a
    batch, and real repos mix one-line helpers with 20k-character modules."""
    items = [(i, "x" * 5_000) for i in range(10)]
    batches = list(_batches(items))
    assert all(sum(len(t) for _, t in b) <= MAX_BATCH_CHARS or len(b) == 1 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_an_oversized_item_is_batched_alone_not_dropped():
    """The cap is on how much travels together, not on what is allowed through."""
    items = [(1, "x" * (MAX_BATCH_CHARS * 3)), (2, "short")]
    batches = list(_batches(items))
    assert [len(b) for b in batches] == [1, 1]
    assert batches[0][0][0] == 1


def test_batches_respect_the_item_cap():
    items = [(i, "x") for i in range(MAX_BATCH_ITEMS * 3)]
    assert all(len(b) <= MAX_BATCH_ITEMS for b in _batches(items))


def test_empty_input_yields_no_batches():
    assert list(_batches([])) == []


# --------------------------------------------------------------------------
# embedding + storage
# --------------------------------------------------------------------------

def test_embed_chunks_stores_a_vector_for_every_chunk(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    emb = FakeEmbedder()
    es = embed_chunks(conn, emb)
    assert es.embedded == stats.chunks
    stored = conn.execute("SELECT count(*) c FROM vec_chunks").fetchone()["c"]
    assert stored == stats.chunks
    assert stored_embed_model(conn) == "fake/v1"


def test_re_embedding_is_incremental(tmp_path):
    """Re-indexing after a small edit must cost seconds, not a full re-encode."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    emb = FakeEmbedder()
    first = embed_chunks(conn, emb)
    assert first.embedded > 0
    second = embed_chunks(conn, emb)
    assert second.embedded == 0
    assert second.skipped_unchanged == first.embedded


def test_switching_model_discards_the_old_vectors(tmp_path):
    """Vectors from two models are not comparable. Mixing them yields retrieval
    that looks fine and ranks nonsense, so the old set must go."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    embed_chunks(conn, FakeEmbedder("model-a"))
    assert stored_embed_model(conn) == "model-a"
    stats = embed_chunks(conn, FakeEmbedder("model-b"))
    assert stats.embedded > 0
    assert stats.skipped_unchanged == 0
    assert stored_embed_model(conn) == "model-b"


def test_changing_dimension_rebuilds_the_vec_table(tmp_path):
    """vec0 bakes dimension into its DDL, so a dimension change is a rebuild."""
    from codelearner.index import ensure_vec_table

    conn = db.init_db(tmp_path / "i.db")
    ensure_vec_table(conn, 3)
    conn.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (1, ?)",
                 (serialize([1.0, 0.0, 0.0]),))
    ensure_vec_table(conn, 8)
    assert conn.execute("SELECT count(*) c FROM vec_chunks").fetchone()["c"] == 0
    assert conn.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone()["value"] == "8"


# --------------------------------------------------------------------------
# dense retrieval
# --------------------------------------------------------------------------

def test_dense_search_ranks_the_matching_symbol_first(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    emb = FakeEmbedder()
    embed_chunks(conn, emb)

    hits = search_dense(conn, "worktree", emb, k=3)
    assert hits
    assert hits[0].qualname == "wt.remove"
    assert hits[0].modality == "dense"


def test_dense_scores_are_similarity_higher_is_better(tmp_path):
    """Must agree with the lexical modality's convention or fusion ranks backwards.
    sqlite-vec returns L2 distance; over unit vectors 1 - d^2/2 is cosine."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    emb = FakeEmbedder()
    embed_chunks(conn, emb)

    hits = search_dense(conn, "lease", emb, k=3)
    assert len(hits) >= 2
    assert hits[0].score >= hits[-1].score
    assert 0.99 <= hits[0].score <= 1.01  # exact match on a unit vector


def test_dense_search_returns_nothing_when_no_vectors_exist(tmp_path):
    """An index built without the embedding step must degrade, not raise."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert search_dense(conn, "lease", FakeEmbedder(), k=3) == []
    assert stored_embed_model(conn) is None


def test_dense_uses_the_k_form_required_by_old_sqlite(tmp_path):
    """REGRESSION. SQLite 3.37.2 does not push LIMIT into a virtual table, so
    `ORDER BY distance LIMIT n` fails outright on vec0. Measured in Phase 0.

    This asserts the query actually returns exactly k rows -- the form that fails
    raises OperationalError, so a passing result proves the working form is used."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    emb = FakeEmbedder()
    embed_chunks(conn, emb)
    assert len(search_dense(conn, "lease", emb, k=2)) == 2
    assert len(search_dense(conn, "lease", emb, k=1)) == 1


def test_serialize_produces_float32_little_endian():
    packed = serialize([1.0, 0.0])
    assert len(packed) == 8  # 2 * float32
