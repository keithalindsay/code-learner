"""Dense vector indexing: embed chunks, store them beside everything else."""

from .embed import (
    DEFAULT_MODEL,
    Embedder,
    EmbedStats,
    SentenceTransformerEmbedder,
    embed_chunks,
    ensure_vec_table,
)

__all__ = [
    "DEFAULT_MODEL",
    "EmbedStats",
    "Embedder",
    "SentenceTransformerEmbedder",
    "embed_chunks",
    "ensure_vec_table",
]
