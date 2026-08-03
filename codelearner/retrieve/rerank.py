"""Cross-encoder reranking: the stage that gives graph expansion a query back.

**Why this exists, in one measurement.** The Phase 8 ablation showed graph expansion
raising recall@5 from 0.604 to 0.646 and recall@10 from 0.781 to 0.802, while
*lowering* MRR from 0.516 to 0.463. That is not a tuning accident, it is structural:
graph expansion has no query representation. It reaches symbols the text modalities
missed -- which is exactly why recall climbs -- but every vote it casts is evidence
about the CODE rather than about the QUESTION, so those votes dilute the top of the
ranking. Weighting graph down to 0.3 bounds the damage; it does not remove it.

A cross-encoder is the honest fix rather than another weight to tune. It reads the
query and one candidate *together* in a single forward pass, which is precisely the
thing fusion cannot do: RRF only ever sees positions, and the bi-encoder only ever
sees the two texts separately. So the division of labour is

    lexical + dense + graph  ->  widen the candidate set   (recall)
    cross-encoder            ->  reorder it                (MRR)

and it is deliberate that reranking runs on a DEEPER candidate list than `k`
(`search()` fuses to `k * CANDIDATE_MULTIPLIER` when a reranker is present). Nothing
downstream can recover a symbol that never entered the candidate set, so the widening
has to happen first and the reordering second.

**Model actually run: `zeroentropy/zerank-1-small-reranker`** (1.7B, Qwen3-based,
~3.4GB of bf16 weights), on a 10GB RTX 3080. `BAAI/bge-reranker-base` is kept as a
fallback because it is 30x smaller and loads anywhere -- if a machine cannot hold the
1.7B model, a weaker reranker is still better than none -- but NOTHING here has been
measured with it. Which model a run used is recorded on `.name`, never inferred,
because a measurement attributed to the wrong model is worse than no measurement.

**What it actually bought,** on the same 16-query swarm-sync gold set, against the
`hybrid + prefer_impl` default (recall@5 0.646 / recall@10 0.802 / hit@5 0.750 /
MRR 0.453):

    hybrid + rerank             0.750   0.781   0.875   0.679

MRR +0.226 and hit@5 +0.125: the largest single lever measured in this project, and
it lands exactly where the diagnosis said the damage was. It also beats the best
pre-rerank MRR of any configuration (0.516, lexical+dense+prefer_impl) by 0.163.

**And what it cost, which is the part not to skip: recall@10 went DOWN, 0.802 to
0.781.** Reranking cannot manufacture recall -- it reorders a fixed candidate set --
and here it pushed a gold symbol that RRF had parked at rank 9 or 10 out of the top
ten while pulling better answers up. The project's own rule is that one or two points
on 16 queries is noise, so this is not a finding so much as a refusal to round it to
zero: the headline is "much better ranking, no better recall", not "better".

**Two more results that were not the hypothesis.** With reranking on, turning graph
expansion OFF scores exactly the same -- 0.750 / 0.781 / 0.875 / 0.679 on all four
metrics, to three decimals. That was suspicious enough to check per-query rather than
report, and the check says it is a real tie and not a broken flag: graph expansion
contributed 434 symbols the text modalities never returned, and 7 of the 16 reranked
top-tens genuinely differ between the two configurations. It changes WHAT comes back;
it just never changes whether a gold-labelled symbol is in the top ten. So the premise
this stage was built on -- "graph widens the candidate set, the cross-encoder reorders
it" -- is only half confirmed. The reordering is worth its cost. The widening, on this
gold set, is not yet demonstrably worth anything.

And `prefer_implementation` still earns recall@5 (0.750 with it, 0.688 without) but no
longer earns MRR (0.679 vs 0.677). The cross-encoder has taken over the job the test
demotion was doing at the top of the ranking, and kept only the part of it that
reaches further down.

Scoring conventions differ between the two -- zerank returns a calibrated probability
in (0, 1), bge returns an uncalibrated logit -- and this module deliberately does not
normalise them. Only the ORDER is used, and only the order is comparable.

**Reranking is optional and never fatal.** `load_reranker()` returns `None` when the
model cannot be had -- no torch, no weights, no VRAM, no network -- and `search()`
treats `None` as "skip the stage". An index that cannot rerank still retrieves. This
mirrors how `index/embed.py` treats a CUDA OOM on model load: degrade to something
slower or weaker, never raise at the user.

**Except when a caller says it must not be.** `strict_device=True` is the opt-in for a
measurement run, and it refuses BOTH silent substitutions this file otherwise makes:
the device (cuda -> cpu) and the model (`zerank` -> `bge`). The second is not scope
creep, it is the same rule as the first and this module's own docstring already states
it -- "a measurement attributed to the wrong model is worse than no measurement", and
every number quoted above was measured with zerank and none with bge. Under strict, a
`gpu.CpuFallbackRefused` comes back instead of a quietly weaker reranker.

The limit of that is worth naming rather than discovering: strict is about SUBSTITUTION,
not about presence. A `load_reranker(strict_device=True)` that cannot reach the weights
at all still returns `None`, because "no reranker" is a visible condition the caller
chose to allow when it accepted an optional stage.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from ..gpu import CpuFallbackRefused, refusal_message
from .lexical import Hit

logger = logging.getLogger(__name__)

# The model the reported numbers were measured with. See the module docstring.
DEFAULT_MODEL = "zeroentropy/zerank-1-small-reranker"

# Loaded only if the default will not. Roughly 1/30th the parameters and a much
# weaker reranker, but it runs on CPU in seconds and needs no remote code.
FALLBACK_MODEL = "BAAI/bge-reranker-base"

# Characters of chunk text shown to the cross-encoder per candidate.
#
# Sized by the same reasoning as `embed.MAX_SEQ_TOKENS`, and by the same constraint:
# a shared 10GB card. A causal-LM reranker materialises a `[batch, seq, vocab]` logit
# tensor, and with Qwen3's 151,936-token vocabulary that tensor -- not the weights --
# is what dominates peak memory. Measured on swarm-sync the median chunk is 770
# characters and p95 is 2,988, so 4,000 keeps the great majority of symbols whole
# while capping the worst case. A longer symbol is truncated for SCORING only; the
# stored chunk and everything the caller is shown remain intact.
MAX_DOC_CHARS = 4_000

# Queries are short by construction ("how does X work"), but a pasted stack trace is
# not, and one runaway query would blow the batch budget for every candidate in it.
MAX_QUERY_CHARS = 1_000

# Cap on how many candidates are scored. Cross-encoding is O(candidates) forward
# passes -- there is no index to amortise it against, which is the whole reason it is
# a second stage and not the first. `search()` hands over k * 4 = 40 by default; this
# is a ceiling for callers that fuse deeper, not a tuned value.
MAX_CANDIDATES = 100


class Reranker(Protocol):
    """The seam that keeps the reranking model swappable -- and omittable.

    `search()` depends on this and not on sentence-transformers, so the pipeline is
    testable with a deterministic fake (see `tests/test_rerank.py`) and a different
    model is a constructor argument rather than a rewrite. It is the same bargain
    the `Embedder` protocol makes for the dense modality.
    """

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]: ...


def chunk_texts(conn: sqlite3.Connection, symbol_ids: Sequence[int]) -> dict[int, str]:
    """Fetch chunk text for `symbol_ids` in one query, keyed by symbol id.

    One query rather than one per hit: reranking already costs a model forward pass
    per candidate, and adding 40 round trips to sqlite on top of that is latency
    bought for nothing.
    """
    if not symbol_ids:
        return {}
    placeholders = ",".join("?" * len(symbol_ids))
    rows = conn.execute(
        f"SELECT symbol_id, text FROM chunks WHERE symbol_id IN ({placeholders})",  # noqa: S608 - placeholders only
        tuple(symbol_ids),
    ).fetchall()
    return {r["symbol_id"]: r["text"] for r in rows}


def _document_for(hit: Hit, texts: dict[int, str]) -> str:
    """The text the cross-encoder judges for one candidate.

    Chunk text when the index can supply it. The chunk already opens with the
    generated header -- file, qualname, enclosing scope, signature, docstring -- so
    this is body-with-provenance rather than a bare code fragment, which matters: a
    method body read alone often does not say what it belongs to.

    Falling back to the header alone (rather than to an empty string) keeps a
    reranker useful against a connection whose chunks are gone, at reduced quality.
    """
    text = texts.get(hit.symbol_id) or hit.header or hit.qualname
    return text[:MAX_DOC_CHARS]


class CrossEncoderReranker:
    """`Reranker` backed by a sentence-transformers CrossEncoder, GPU when free.

    Handles both supported models through one code path because both expose
    `predict(pairs) -> scores`, higher is better. Their score SCALES differ and are
    not reconciled -- see the module docstring -- because only ordering is consumed.

    `conn` is optional and supplying it is strongly recommended: without it the model
    only ever sees a candidate's header, and judging code by its signature is most of
    the way back to the bi-encoder this stage exists to improve on.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        conn: sqlite3.Connection | None = None,
        device: str | None = None,
        max_candidates: int = MAX_CANDIDATES,
        warmup: bool = True,
        strict_device: bool = False,
    ) -> None:
        explicit_device = device is not None
        if device is None:
            device = _default_device()

        if strict_device and device == "cpu":
            raise CpuFallbackRefused(
                refusal_message(
                    what=f"reranking with {model_name}",
                    cause="CPU was requested"
                    if explicit_device
                    else "no CUDA device is available to torch",
                )
            )

        try:
            self._model = self._build(model_name, device, warmup)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            # Same bargain `embed.SentenceTransformerEmbedder` strikes: the 10GB card
            # is shared, and it is not this tool's place to evict whoever else is on
            # it. A reranker on CPU is slow; a traceback is useless. Observed exactly
            # as embed.py predicted: `ollama` resident with 9.1GB left 78MB free and
            # the 3.2GB weight allocation failed.
            if device != "cuda" or not _is_oom(exc):
                raise
            if strict_device:
                detail = str(exc).split("\n")[0][:160]
                raise CpuFallbackRefused(
                    refusal_message(
                        what=f"reranking with {model_name}",
                        cause=f"CUDA is out of memory: {detail}",
                    )
                ) from exc
            if explicit_device:
                raise
            logger.warning(
                "could not load reranker %s on CUDA (%s); falling back to CPU. "
                "Reranking will be slow. Free VRAM and re-run for full speed.",
                model_name,
                str(exc).split("\n")[0][:160],
            )
            device = "cpu"
            self._model = self._build(model_name, device, warmup)

        self._name = model_name
        self._device = device
        self._conn = conn
        self._max_candidates = max_candidates

    def _build(self, model_name: str, device: str, warmup: bool):
        """Construct the CrossEncoder and force its weights to actually land.

        The warmup pair is not paranoia. `zerank-1-small-reranker` ships remote code
        that loads its real model LAZILY, inside the first `predict()` call -- the
        constructor only builds a shell, and it succeeds on a card with 78MB free.
        Without a warmup the CUDA OOM surfaces from the middle of a user's query,
        by which point the only survivable response is to skip reranking entirely.
        Scoring one throwaway pair here moves the failure to construction time,
        where falling back to CPU is still on the table.
        """
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        if warmup:
            self._run(model, [("warmup", "warmup")])
        return model

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        return self._device

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]:
        """Reorder `hits` by cross-encoder relevance and return the top `k`.

        Candidates beyond `max_candidates` keep their fused order and are appended
        after the scored ones. Truncating the list instead would silently convert a
        reranking stage into a filter, and losing recall to a latency cap is exactly
        the failure this stage was built to prevent.
        """
        if not hits:
            return []

        scored, unscored = list(hits[: self._max_candidates]), list(hits[self._max_candidates :])
        texts = chunk_texts(self._conn, [h.symbol_id for h in scored]) if self._conn else {}
        pairs = [(query[:MAX_QUERY_CHARS], _document_for(h, texts)) for h in scored]

        try:
            scores = self._predict(pairs)
        except Exception as exc:  # noqa: BLE001 - reranking is an optimisation
            # A reranker that fails must cost ranking quality, not the answer. The
            # fused order is a perfectly good result; it is what every run before
            # Phase 3b returned.
            if not _is_oom(exc):
                raise
            logger.warning(
                "reranking failed (%s); returning the fused order unchanged",
                str(exc).split("\n")[0][:160],
            )
            return list(hits[:k])

        # Stable on ties: `sorted` preserves the fused order among equal scores, so a
        # model with no opinion degrades to RRF rather than to arbitrary.
        order = sorted(range(len(scored)), key=lambda i: -scores[i])
        reranked = [replace(scored[i], score=float(scores[i])) for i in order]
        return (reranked + unscored)[:k]

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._run(self._model, pairs)

    @staticmethod
    def _run(model, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs under `no_grad`.

        The `no_grad` is load-bearing rather than tidy. A causal-LM reranker emits a
        `[batch, seq, vocab]` logit tensor, and zerank's remote code calls the model
        without disabling autograd, so every one of those tensors is retained for a
        backward pass that never comes. On a 151,936-token vocabulary that is the
        difference between a batch fitting and the card OOMing.
        """
        try:
            import torch

            with torch.no_grad():
                return [float(s) for s in model.predict(pairs)]
        except ImportError:
            return [float(s) for s in model.predict(pairs)]


def load_reranker(
    model_name: str | None = None,
    conn: sqlite3.Connection | None = None,
    device: str | None = None,
    strict_device: bool = False,
) -> Reranker | None:
    """Build a reranker, or return `None` if none can be had.

    Tries `model_name` (default `zerank-1-small-reranker`), then
    `bge-reranker-base`, then gives up quietly. Every failure mode here -- no
    sentence-transformers, no network, no disk, no VRAM even on CPU fallback -- is a
    reason to retrieve WITHOUT reranking, not a reason to fail the query. Callers
    pass the result straight to `search(reranker=...)`, which treats `None` as
    "skip the stage".

    Returning `None` rather than raising is a deliberate asymmetry with
    `embed_chunks`, which DOES raise when sqlite-vec is missing. The difference is
    that dense retrieval without vectors produces nothing, while retrieval without a
    reranker produces the previous release's results.

    `strict_device=True` narrows that asymmetry rather than removing it. The fallback
    MODEL is dropped from the candidate list -- a run that asked for the reranker it
    measured with does not want a different one substituted for it -- and a
    `CpuFallbackRefused` propagates instead of being swallowed. Any other failure
    still returns `None`; see the module docstring for why that line is where it is.
    """
    if model_name:
        candidates = [model_name]
    elif strict_device:
        candidates = [DEFAULT_MODEL]
    else:
        candidates = [DEFAULT_MODEL, FALLBACK_MODEL]
    for name in candidates:
        try:
            return CrossEncoderReranker(
                name, conn=conn, device=device, strict_device=strict_device
            )
        except CpuFallbackRefused:
            # The one failure a strict caller asked to hear about. Logging it and
            # returning `None` would be the silent quality loss `strict_device` exists
            # to make impossible.
            raise
        except Exception as exc:  # noqa: BLE001 - every failure here is survivable
            logger.warning(
                "could not load reranker %s (%s: %s); %s",
                name,
                type(exc).__name__,
                str(exc).split("\n")[0][:160],
                "trying the fallback" if name != candidates[-1] else "reranking disabled",
            )
    return None


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _is_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory condition.

    Matched on type name and message rather than by importing torch's exception
    class, so this module stays importable without torch installed. Same test as
    `index/embed._is_oom`, duplicated rather than shared because retrieval importing
    from the indexer to borrow a four-line predicate is the worse coupling.
    """
    return (
        type(exc).__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )
