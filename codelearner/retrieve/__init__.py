"""Retrieval modalities and the fusion over them."""

from .dense import search_dense, stored_embed_model
from .fuse import reciprocal_rank_fusion
from .graph import expand, neighbours
from .lexical import Hit, search_lexical
from .rerank import CrossEncoderReranker, Reranker, load_reranker
from .search import SearchResult, search

__all__ = [
    "CrossEncoderReranker",
    "Hit",
    "Reranker",
    "SearchResult",
    "expand",
    "load_reranker",
    # Exported because `generate` traverses the call graph and must do it the way
    # retrieval does. See `graph.neighbours` for why that is a contract and not a
    # convenience.
    "neighbours",
    "reciprocal_rank_fusion",
    "search",
    "search_dense",
    "search_lexical",
    "stored_embed_model",
]
