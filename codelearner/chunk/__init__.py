"""Symbol-boundary chunking: turn symbols into retrieval units."""

from .chunker import ChunkStats, build_chunks, chunk_for_symbol

__all__ = ["ChunkStats", "build_chunks", "chunk_for_symbol"]
