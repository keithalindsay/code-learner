"""Retrieval modalities and the fusion over them."""

from .dense import search_dense, stored_embed_model
from .fuse import reciprocal_rank_fusion
from .graph import expand, neighbours
from .lexical import Hit, search_lexical
from .mixed import SEMANTIC_WEIGHTS, mixed_rank_fusion, search_candidates
from .rerank import CrossEncoderReranker, Reranker, load_reranker
from .search import SearchResult, search
from .types import (
    AssertionCandidate,
    Candidate,
    CandidateSearchResult,
    Freshness,
    ScoreContribution,
    SourceCandidate,
    VerdictSummary,
)

__all__ = [
    "CrossEncoderReranker",
    "AssertionCandidate",
    "Candidate",
    "CandidateSearchResult",
    "Freshness",
    "Hit",
    "Reranker",
    "SEMANTIC_WEIGHTS",
    "ScoreContribution",
    "SearchResult",
    "SourceCandidate",
    "VerdictSummary",
    "expand",
    "load_reranker",
    "mixed_rank_fusion",
    # Exported because `generate` traverses the call graph and must do it the way
    # retrieval does. See `graph.neighbours` for why that is a contract and not a
    # convenience.
    "neighbours",
    "reciprocal_rank_fusion",
    "search",
    "search_candidates",
    "search_dense",
    "search_lexical",
    "stored_embed_model",
]
