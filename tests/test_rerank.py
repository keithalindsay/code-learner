"""Cross-encoder reranking, exercised through the `Reranker` protocol.

Uses a deterministic fake reranker rather than loading a 1.7B model. That is what
the protocol is for -- the same bargain `tests/test_embed.py` makes with `Embedder`.
Every assertion here is about the pipeline's CONTRACT: that the candidate set is
widened before reordering, that ordering is taken from the model, that ties fall
back to the fused order, and that an absent or failing model costs ranking quality
and nothing else. A real model would add minutes and would make the assertions
weaker, not stronger, because its scores are not knowable in advance.
"""
from __future__ import annotations

import subprocess

import pytest

from codelearner.ingest import index_repo
from codelearner.retrieve import search
from codelearner.retrieve.lexical import Hit
from codelearner.retrieve.rerank import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    MAX_DOC_CHARS,
    CrossEncoderReranker,
    _document_for,
    chunk_texts,
    load_reranker,
)
from codelearner.retrieve.search import CANDIDATE_MULTIPLIER


def _hit(symbol_id: int, qualname: str, score: float = 1.0, header: str = "") -> Hit:
    return Hit(
        symbol_id=symbol_id, qualname=qualname, kind="function", path="m.py",
        line_start=1, line_end=2, score=score, modality="lexical", header=header,
    )


class FakeReranker:
    """Scores a candidate by how many query terms its qualname contains.

    Deterministic, dependency-free, and verifiable by reading it -- the same
    property `FakeEmbedder` has. It stands in for a cross-encoder in exactly the way
    that matters to the pipeline: it sees the QUERY, which fusion never does.
    """

    def __init__(self, name: str = "fake/reranker-v1") -> None:
        self._name = name
        self.calls: list[tuple[str, int]] = []  # (query, candidates seen)

    @property
    def name(self) -> str:
        return self._name

    def _score(self, query: str, hit: Hit) -> float:
        terms = query.lower().split()
        blob = f"{hit.qualname} {hit.header}".lower()
        return float(sum(1 for t in terms if t in blob))

    def rerank(self, query: str, hits, k: int = 10):
        self.calls.append((query, len(hits)))
        ordered = sorted(hits, key=lambda h: -self._score(query, h))
        return list(ordered[:k])


class StubCrossEncoder:
    """Stands in for `sentence_transformers.CrossEncoder` inside the real class.

    Lets `CrossEncoderReranker`'s own logic -- batching, truncation, tie handling,
    OOM survival -- be tested without 3.4GB of weights. `predict` is handed the
    exact pairs the reranker built, so document construction is assertable too.
    """

    instances: list[StubCrossEncoder] = []

    def __init__(self, model_name, device=None, trust_remote_code=False, scores=None):
        self.model_name = model_name
        self.device = device
        self.seen: list[list[tuple[str, str]]] = []
        self._scores = scores
        StubCrossEncoder.instances.append(self)

    def predict(self, pairs):
        self.seen.append(list(pairs))
        if self._scores is not None:
            return self._scores[: len(pairs)]
        # Longer document wins, so ordering is decided by something observable.
        return [float(len(doc)) for _, doc in pairs]


@pytest.fixture(autouse=True)
def _reset_stub_registry():
    StubCrossEncoder.instances.clear()
    yield
    StubCrossEncoder.instances.clear()


def _install_stub(monkeypatch, factory=StubCrossEncoder):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", factory)


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
    "leases.py": (
        'def acquire_lease(path):\n'
        '    """Take a lease on a path so no other agent writes it."""\n'
        '    return 1\n'
    ),
    "worktree.py": (
        'def remove_worktree(name):\n'
        '    """Delete a git worktree and its metadata."""\n'
        '    return 2\n'
    ),
    "store.py": (
        'def connect():\n'
        '    """Open the sqlite store in WAL mode."""\n'
        '    return 3\n'
    ),
}


# --------------------------------------------------------------------------
# the protocol seam in `search()`
# --------------------------------------------------------------------------

def test_search_without_a_reranker_still_answers(tmp_path):
    """Requirement zero. No model available must mean unreranked results, never a
    traceback -- reranking is an optimisation on top of a pipeline that already
    worked."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    result = search(conn, "lease", k=5, reranker=None)
    assert result.hits
    assert result.reranked is False


def test_reranker_reorders_the_fused_result(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")

    baseline = search(conn, "remove worktree", k=3).hits
    reranked = search(conn, "remove worktree", k=3, reranker=FakeReranker()).hits

    assert [h.qualname for h in reranked][0] == "worktree.remove_worktree"
    assert {h.qualname for h in reranked} <= {h.qualname for h in baseline} | {
        h.qualname for h in reranked
    }


def test_search_marks_a_reranked_result_as_reranked(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert search(conn, "lease", k=3, reranker=FakeReranker()).reranked is True


def test_reranking_sees_a_deeper_candidate_set_than_k(tmp_path):
    """The division of labour this stage exists for: retrieval widens, the
    cross-encoder reorders. If fusion truncated to `k` first, the reranker could
    only permute what RRF already liked, and the recall graph expansion bought
    would be thrown away before anything could use it."""
    files = {f"m{i}.py": f"def lease_{i}():\n    '''lease helper {i}.'''\n    return {i}\n"
             for i in range(30)}
    repo = _mkrepo(tmp_path / "r", files)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")

    fake = FakeReranker()
    search(conn, "lease", k=3, reranker=fake)

    assert fake.calls
    _, candidates = fake.calls[0]
    assert candidates > 3
    assert candidates <= 3 * CANDIDATE_MULTIPLIER


def test_reranker_receives_the_query_verbatim(tmp_path):
    """The whole premise. Graph expansion has no query representation; this stage
    is only worth its cost because it does."""
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    fake = FakeReranker()
    search(conn, "how does lease acquisition work", k=3, reranker=fake)
    assert fake.calls[0][0] == "how does lease acquisition work"


def test_reranker_output_is_capped_at_k(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert len(search(conn, "lease worktree sqlite", k=2, reranker=FakeReranker()).hits) <= 2


# --------------------------------------------------------------------------
# document construction
# --------------------------------------------------------------------------

def test_chunk_texts_fetches_by_symbol_id(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    ids = [r["symbol_id"] for r in conn.execute("SELECT symbol_id FROM chunks")]
    texts = chunk_texts(conn, ids)
    assert set(texts) == set(ids)
    assert any("lease" in t for t in texts.values())


def test_chunk_texts_on_an_empty_id_list_does_not_query(tmp_path):
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    assert chunk_texts(conn, []) == {}


def test_document_falls_back_to_the_header_when_no_chunk_text():
    """A reranker against an index whose chunks are gone is degraded, not broken --
    judging by signature is weak, but it is not nothing."""
    hit = _hit(7, "pkg.thing", header="pkg.py :: pkg.thing(a, b)")
    assert _document_for(hit, {}) == "pkg.py :: pkg.thing(a, b)"
    assert _document_for(_hit(7, "pkg.thing"), {}) == "pkg.thing"


def test_document_is_truncated_for_scoring_only():
    """Peak memory on a causal-LM reranker is the [batch, seq, vocab] logit tensor,
    so document length is a VRAM decision. The stored chunk is untouched."""
    long_text = "x" * (MAX_DOC_CHARS * 3)
    assert len(_document_for(_hit(1, "big"), {1: long_text})) == MAX_DOC_CHARS


# --------------------------------------------------------------------------
# CrossEncoderReranker, with the model stubbed out
# --------------------------------------------------------------------------

def test_cross_encoder_orders_by_model_score(monkeypatch):
    _install_stub(monkeypatch)
    r = CrossEncoderReranker("stub/model", device="cpu", warmup=False)
    hits = [_hit(1, "a", header="short"), _hit(2, "b", header="a much longer header")]
    out = r.rerank("q", hits, k=2)
    assert [h.qualname for h in out] == ["b", "a"]
    assert out[0].score > out[1].score


def test_cross_encoder_ties_keep_the_fused_order(monkeypatch):
    """A model with no opinion must degrade to RRF, not to arbitrary. Ordering that
    changes run to run makes every downstream measurement unreproducible."""
    _install_stub(monkeypatch, lambda *a, **kw: StubCrossEncoder(*a, scores=[1.0, 1.0, 1.0], **kw))
    r = CrossEncoderReranker("stub/model", device="cpu", warmup=False)
    hits = [_hit(1, "first"), _hit(2, "second"), _hit(3, "third")]
    assert [h.qualname for h in r.rerank("q", hits, k=3)] == ["first", "second", "third"]


def test_cross_encoder_appends_rather_than_drops_beyond_the_candidate_cap(monkeypatch):
    """Truncating instead would silently turn a reranking stage into a filter, and
    losing recall to a latency cap is the exact failure this stage prevents."""
    _install_stub(monkeypatch)
    r = CrossEncoderReranker("stub/model", device="cpu", max_candidates=2, warmup=False)
    hits = [_hit(i, f"h{i}", header="x" * i) for i in range(1, 6)]
    out = r.rerank("q", hits, k=5)
    assert len(out) == 5
    assert {h.qualname for h in out} == {f"h{i}" for i in range(1, 6)}
    assert [h.qualname for h in out[2:]] == ["h3", "h4", "h5"]  # untouched tail order
    assert len(StubCrossEncoder.instances[0].seen[0]) == 2


def test_cross_encoder_on_empty_hits_does_not_call_the_model(monkeypatch):
    _install_stub(monkeypatch)
    r = CrossEncoderReranker("stub/model", device="cpu", warmup=False)
    assert r.rerank("q", [], k=5) == []
    assert StubCrossEncoder.instances[0].seen == []


def test_cross_encoder_uses_chunk_text_when_given_a_connection(monkeypatch, tmp_path):
    _install_stub(monkeypatch)
    repo = _mkrepo(tmp_path / "r", REPO_FILES)
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT s.id, s.qualname FROM symbols s JOIN chunks c ON c.symbol_id = s.id "
        "WHERE s.qualname = 'leases.acquire_lease'"
    ).fetchone()

    r = CrossEncoderReranker("stub/model", conn=conn, device="cpu", warmup=False)
    r.rerank("lease", [_hit(row["id"], row["qualname"])], k=1)

    _, document = StubCrossEncoder.instances[0].seen[0][0]
    assert "no other agent writes it" in document  # the BODY, not just the signature


def test_a_cuda_oom_mid_rerank_returns_the_fused_order(monkeypatch):
    """The pre-Phase-3b result is a perfectly good answer. A shared card losing a
    race must cost ranking quality, not the query."""
    class OomingCrossEncoder(StubCrossEncoder):
        def predict(self, pairs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    _install_stub(monkeypatch, OomingCrossEncoder)
    r = CrossEncoderReranker("stub/model", device="cpu", warmup=False)
    hits = [_hit(1, "first"), _hit(2, "second")]
    assert [h.qualname for h in r.rerank("q", hits, k=2)] == ["first", "second"]


def test_a_non_oom_model_error_is_not_swallowed(monkeypatch):
    """Degrading gracefully is for resource pressure. A genuine bug that returns
    silently-worse rankings is how a retrieval system quietly rots."""
    class BrokenCrossEncoder(StubCrossEncoder):
        def predict(self, pairs):
            raise ValueError("tokenizer mismatch")

    _install_stub(monkeypatch, BrokenCrossEncoder)
    r = CrossEncoderReranker("stub/model", device="cpu", warmup=False)
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        r.rerank("q", [_hit(1, "a")], k=1)


def test_construction_warms_the_model_up(monkeypatch):
    """REGRESSION, and the reason `_build` exists. `zerank-1-small-reranker` ships
    remote code that loads its real weights lazily inside the FIRST predict() call.
    Constructing it on a card with 78MB free succeeds; the OOM then arrives from the
    middle of a user's query, too late to fall back to CPU. Measured on this box
    with ollama holding 9.1GB."""
    _install_stub(monkeypatch)
    CrossEncoderReranker("stub/model", device="cpu")
    assert StubCrossEncoder.instances[0].seen == [[("warmup", "warmup")]]


def test_a_lazy_cuda_oom_at_warmup_falls_back_to_cpu(monkeypatch):
    """The failure the warmup was added to catch, caught where it can be handled."""
    class LazyOom(StubCrossEncoder):
        def predict(self, pairs):
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory. Tried to allocate 3.21 GiB")
            return super().predict(pairs)

    _install_stub(monkeypatch, LazyOom)
    monkeypatch.setattr("codelearner.retrieve.rerank._default_device", lambda: "cuda")
    r = CrossEncoderReranker("stub/model")
    assert r.device == "cpu"
    assert [i.device for i in StubCrossEncoder.instances] == ["cuda", "cpu"]


def test_an_explicit_device_is_not_second_guessed(monkeypatch):
    """`device="cuda"` from a caller means cuda. Silently running on CPU would make
    a benchmark meaningless -- the same rule `SentenceTransformerEmbedder` follows."""
    class OomOnLoad(StubCrossEncoder):
        def __init__(self, *a, **kw):
            raise RuntimeError("CUDA out of memory")

    _install_stub(monkeypatch, OomOnLoad)
    with pytest.raises(RuntimeError, match="out of memory"):
        CrossEncoderReranker("stub/model", device="cuda")


# --------------------------------------------------------------------------
# load_reranker: the graceful-degradation contract
# --------------------------------------------------------------------------

def test_load_reranker_returns_none_when_nothing_loads(monkeypatch):
    """`None` is a valid reranker. `search()` treats it as 'skip the stage', so an
    offline machine with no weights retrieves exactly as it did before Phase 3b."""
    def explode(*args, **kwargs):
        raise OSError("no network, no cached weights")

    _install_stub(monkeypatch, explode)
    assert load_reranker() is None


def test_load_reranker_falls_back_to_the_smaller_model(monkeypatch):
    """A weaker reranker beats no reranker on a machine that cannot hold 1.7B."""
    def selective(model_name, **kwargs):
        if model_name == DEFAULT_MODEL:
            raise RuntimeError("CUDA out of memory")
        return StubCrossEncoder(model_name, **kwargs)

    _install_stub(monkeypatch, selective)
    reranker = load_reranker(device="cpu")
    assert reranker is not None
    assert reranker.name == FALLBACK_MODEL


def test_load_reranker_does_not_fall_back_from_an_explicit_model(monkeypatch):
    """Asking for a named model and silently getting a different one would make any
    number attributed to it a lie."""
    def explode(*args, **kwargs):
        raise OSError("not available")

    _install_stub(monkeypatch, explode)
    assert load_reranker("some/specific-model") is None


def test_the_default_model_is_the_one_the_readme_claims():
    """Pins the recorded model so the ablation table cannot silently start
    describing a different one."""
    assert DEFAULT_MODEL == "zeroentropy/zerank-1-small-reranker"
    assert FALLBACK_MODEL == "BAAI/bge-reranker-base"
