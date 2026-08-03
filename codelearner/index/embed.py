"""Embed chunks into sqlite-vec, in the same file as everything else.

**Model choice.** `Qwen3-Embedding-0.6B`: 1024-dim, 32K sequence length, ~1.2GB
VRAM. The originally planned `bge-small-en-v1.5` is a general-text model and the
weakest link in a code pipeline -- code-specialised models score roughly 75 on
MTEB Code v1 against `bge`-class models in the 57-68 range.

`C2LLM-0.5B` scores marginally higher (75.46 vs 75.42) but its modeling code pulls
in `deepspeed` and `peft`, which is a large and brittle dependency for a
statistically indistinguishable gain. Qwen3 is a standard architecture that loads
with plain sentence-transformers. That trade is worth naming rather than burying.

**Asymmetric encoding.** Queries and documents are encoded differently -- the model
ships a `query` prompt for exactly this. A question ("how do I open a WAL
connection") and the code that answers it are not the same kind of text, and
encoding both identically throws away a real signal.

**Sequence length is capped well below the model's ceiling, and VRAM is why.** The
model advertises 32K tokens, which sounds like the chunking guarantee ("a symbol is
never split") survives intact into the vector. On a 10GB card it does not: attention
memory is set by the longest sequence in a batch, and honouring 32K at any useful
batch size OOMs. `MAX_SEQ_TOKENS` therefore caps encoding at 2048 tokens -- above
the great majority of code symbols, but not all of them. A very large symbol IS
truncated at encode time even though its chunk was stored whole -- named here rather
than left for someone to discover from a retrieval miss.

Measured on swarm-sync: median chunk 770 characters, p95 2,988, max 24,472. Seven of
1,087 chunks (0.6%) exceed the cap. Small enough to accept, large enough to state.

**`strict_device` is opt-in and stays that way.** The CPU fallback below is correct
and its reasoning is unchanged: a shared card is shared, and evicting whoever else is
on it is not this tool's decision to make. But a *measurement* run has a different
requirement from an indexing run -- an eval that silently becomes ten times slower
mid-flight, and may then be interrupted, is a corrupted measurement reported under the
name of a clean one. `strict_device=True` says "CPU is not an acceptable outcome for
this run" and turns the warning into a `gpu.CpuFallbackRefused`. Nothing changes for
any caller that does not ask for it.
"""
from __future__ import annotations

import logging
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .. import db
from ..gpu import CpuFallbackRefused, refusal_message

logger = logging.getLogger(__name__)


def _is_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory condition.

    Matched on type name and message rather than by importing torch's exception
    class, so this module stays importable without torch installed.
    """
    return (
        type(exc).__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Encoding is batched by CHARACTER BUDGET, not by a fixed count.
#
# A fixed batch size is the obvious choice and it OOMs. Chunk lengths in a real repo
# span three orders of magnitude -- a one-line helper next to a 24,000-character
# module -- and attention memory grows with the square of the longest sequence in the
# batch, not the average. A batch size tuned on short chunks dies the moment it meets
# a long one. Measured on a 10GB RTX 3080: 16 chunks/batch OOM'd on swarm-sync.
#
# Budgeting by total characters makes one long chunk travel alone and lets short ones
# pack densely, which bounds peak memory by construction rather than by luck.
MAX_BATCH_CHARS = 16_000
MAX_BATCH_ITEMS = 32

# Cap on tokens per chunk at encode time. Well above the median code symbol, far
# below the model's 32K ceiling. The ceiling is not the constraint -- VRAM is.
MAX_SEQ_TOKENS = 2048


@dataclass
class EmbedStats:
    embedded: int = 0
    skipped_unchanged: int = 0
    batches: int = 0
    dim: int = 0
    model: str = ""


def _batches(items: list[tuple[int, str]]) -> Iterable[list[tuple[int, str]]]:
    """Group `(chunk_id, text)` pairs into batches bounded by total characters.

    A single item longer than the budget is yielded alone rather than dropped --
    the cap is on how much travels together, not on what is allowed through.
    """
    batch: list[tuple[int, str]] = []
    chars = 0
    for item in items:
        size = len(item[1])
        if batch and (chars + size > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_ITEMS):
            yield batch
            batch, chars = [], 0
        batch.append(item)
        chars += size
    if batch:
        yield batch


class Embedder(Protocol):
    """The seam that keeps the model swappable.

    Everything downstream depends on this, not on sentence-transformers. Swapping
    to a different model -- or measuring that swap, which the Phase 8 eval will --
    is then a config change rather than a rewrite.
    """

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """`Embedder` backed by sentence-transformers, GPU when one is available.

    `strict_device=True` means exactly one thing: this run must not end up on CPU.
    That covers every route to CPU and not only the interesting one -- a CUDA OOM
    falling back, but also no torch installed, no CUDA visible, and an explicit
    `device="cpu"`. All four produce the same corrupted measurement, and a strictness
    that only caught the OOM would be a strictness that let the commonest case
    through: an eval launched on a machine where `torch.cuda.is_available()` is False
    never OOMs at all. It just quietly takes all night.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_seq_tokens: int = MAX_SEQ_TOKENS,
        strict_device: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        explicit_device = device is not None
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if strict_device and device == "cpu":
            # Refused before the model is built, not after: loading a 1.2GB model onto
            # CPU and then complaining costs a minute to say something knowable now.
            raise CpuFallbackRefused(
                refusal_message(
                    what=f"embedding with {model_name}",
                    cause="CPU was requested"
                    if explicit_device
                    else "no CUDA device is available to torch",
                )
            )

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            # A 10GB card is shared. ollama holding a 9GB model, another training
            # run, a browser with hardware acceleration -- any of these can leave
            # too little VRAM, and it is not this tool's place to evict them.
            # Falling back to CPU is slower but correct; failing outright is
            # neither. Observed in practice: `ollama ps` showing qwen3:14b resident
            # made model load fail with 129MB free.
            if device != "cuda" or not _is_oom(exc):
                raise
            if strict_device:
                # Raised as `CpuFallbackRefused` rather than re-raised as the OOM even
                # though an explicit `device="cuda"` would also have re-raised: a
                # caller running a strict measurement wants ONE exception type to
                # catch across the embedder and the reranker, and the OOM is preserved
                # as `__cause__` for anyone who needs the original.
                detail = str(exc).split("\n")[0][:160]
                raise CpuFallbackRefused(
                    refusal_message(
                        what=f"embedding with {model_name}",
                        cause=f"CUDA is out of memory: {detail}",
                    )
                ) from exc
            if explicit_device:
                raise
            logger.warning(
                "could not load %s on CUDA (%s); falling back to CPU. Embedding "
                "will be slower. Free VRAM and re-run for full speed.",
                model_name,
                str(exc).split("\n")[0][:160],
            )
            device = "cpu"
            self._model = SentenceTransformer(model_name, device=device)
        self._device = device
        # The model advertises 32K, which VRAM cannot honour at any useful batch
        # size. Capping here rather than hoping is the difference between a bounded
        # run and an OOM partway through a large repo.
        self._model.max_seq_length = max_seq_tokens
        self._name = model_name
        self._dim = int(self._model.get_sentence_embedding_dimension() or 0)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        return self._device

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vecs = self._model.encode(
                list(texts),
                batch_size=len(texts) or 1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            # Mid-run OOM: another process grabbed VRAM after load succeeded.
            # Retry this batch one item at a time before giving up, since the
            # batch is usually what does not fit rather than the model.
            if not _is_oom(exc):
                raise
            logger.warning("CUDA OOM on a batch of %d; retrying one at a time", len(texts))
            self._empty_cache()
            vecs = [
                self._model.encode(
                    [t], batch_size=1, normalize_embeddings=True, show_progress_bar=False
                )[0]
                for t in texts
            ]
        return [list(map(float, v)) for v in vecs]

    def _empty_cache(self) -> None:
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def encode_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [text],
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return list(map(float, vec))


def serialize(vector: Iterable[float]) -> bytes:
    """Pack a float vector the way sqlite-vec expects (little-endian float32)."""
    values = list(vector)
    return struct.pack(f"{len(values)}f", *values)


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the vec0 table for `dim`-dimensional vectors.

    Not part of schema.sql because a vec0 table cannot be created without the
    extension loaded, and the extension is optional. Dimension is baked into the
    DDL, so changing embedding model means dropping and rebuilding -- which is
    correct: vectors from two different models are not comparable, and silently
    mixing them would produce retrieval that looks fine and ranks nonsense.
    """
    stored = conn.execute("SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()
    if stored is not None and int(stored["value"]) != dim:
        conn.execute("DROP TABLE IF EXISTS vec_chunks")
        conn.execute("DELETE FROM meta WHERE key IN ('embed_dim', 'embed_model')")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f"  chunk_id INTEGER PRIMARY KEY,"
        f"  embedding float[{dim}]"
        f")"  # noqa: S608 - dim is an int from the model, never caller text
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_dim', ?)", (str(dim),)
    )


# Test chunks are embedded like everything else, and that is a MEASURED decision
# rather than an oversight. `files.is_test` is right there and this module ignores it
# on purpose.
#
# On 170 hand-written leak-controlled gold queries across swarm-sync, kalshi-bot and
# TradingAgents, test code was 57.2% of dense retrieval's top 10. That is semantically
# correct behaviour -- a test named `test_two_agents_disjoint_functions_same_file`
# really is the closest match for "what stops two agents editing the same file at
# once" -- and it is the wrong answer to the question asked.
# `retrieve.fuse.prefer_implementation` repairs it at query time and still leaves
# 20.1%, which is the argument for repairing it HERE instead.
#
# Not embedding test chunks does repair it, and on the dense modality alone it beats
# the query-time heuristic outright: nDCG@10 0.179 -> 0.293 against the heuristic's
# 0.275 (+0.019, paired bootstrap 95% CI [+0.008, +0.031], 2000 resamples, seed
# 20250801), dense test share 57.2% -> 0.0%, and 64-74% of the embedding work
# disappears with it. Replicated on held-out kalshi-bot, which was not tuned against.
#
# It is not the default because of what the gold set cannot see. None of the 317
# labelled relevant symbols is a test, so no gold query can reward finding one. On a
# separate 40-query "show me the test that covers X" probe, the shipping default
# (lexical + dense + prefer_implementation) finds the intended test 100% of the time
# and dropping tests from the vectors takes that to 0%. The mechanism is specific: the
# demotion in `fuse` is a multiplicative factor on an RRF score, and it works today
# only because a test wins votes from BOTH modalities. Remove its dense vote and the
# same factor stops being a nudge and becomes a filter -- the demoted test falls below
# every dense-only candidate. Sweeping that factor does not buy both: at 1.0 the probe
# holds at 90% and the nDCG gain goes non-significant (+0.016, [-0.009, +0.040]); at
# anything <= 0.9 the gain is real (+0.027) and the probe is <= 35%.
#
# So the ordering is: fix the demotion cliff in `retrieve.fuse` first, then exclude
# here. Two ingest-side alternatives were measured and both lost. Dropping tests from
# the LEXICAL corpus as well is significantly worse in fusion (-0.017, [-0.034,
# -0.001]) and makes tests unreachable outright. Summarising test chunks to their
# header instead of excluding them -- the "down-weight, do not exclude" arm -- barely
# moves the dense test share at all (79.6% -> 77.1% on swarm-sync), because the signal
# that makes a test win is its NAME and docstring, which a summary keeps, not its body.
def embed_chunks(
    conn: sqlite3.Connection,
    embedder: Embedder,
    rebuild: bool = False,
) -> EmbedStats:
    """Embed every chunk that needs it and store the vectors in the index.

    Incremental by default: a chunk whose `text_hash` already has a vector is
    skipped. Re-indexing a repo after a small edit then costs seconds rather than
    re-encoding everything.
    """
    if not db.load_vec_extension(conn):
        raise RuntimeError(
            "sqlite-vec is not available on this connection, so dense retrieval "
            "cannot be built. Install `sqlite-vec` and ensure Python's sqlite3 "
            "supports loadable extensions."
        )

    ensure_vec_table(conn, embedder.dim)
    stats = EmbedStats(dim=embedder.dim, model=embedder.name)

    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    if rebuild or (row is not None and row["value"] != embedder.name):
        # A different model's vectors are not comparable with this one's.
        conn.execute("DELETE FROM vec_chunks")

    already = {
        r["chunk_id"]
        for r in conn.execute("SELECT chunk_id FROM vec_chunks")
    }
    pending = [
        (r["id"], r["text"])
        for r in conn.execute("SELECT id, text FROM chunks ORDER BY id")
        if r["id"] not in already
    ]
    stats.skipped_unchanged = len(already)

    # Sort by length so each batch is roughly homogeneous. Padding is to the longest
    # item in the batch, so mixing a 200-char helper with a 20,000-char module wastes
    # most of the batch on padding -- and sizes peak memory off the outlier.
    pending.sort(key=lambda item: len(item[1]))

    for batch in _batches(pending):
        vectors = embedder.encode_documents([text for _, text in batch])
        with db.transaction(conn):
            for (chunk_id, _), vector in zip(batch, vectors, strict=True):
                conn.execute(
                    "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize(vector)),
                )
        stats.embedded += len(batch)
        stats.batches += 1

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)",
        (embedder.name,),
    )
    return stats
