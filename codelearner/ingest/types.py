"""Value types the extractors produce and the indexer persists.

These are deliberately plain and language-agnostic: a second language extractor
should only have to emit `Symbol`/`Edge`, never touch SQL. That seam is the one
thing that decides whether "add another language" is a day or a rewrite.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Symbol kinds. 'module' is a real symbol, not a container fiction: module-level
# code makes calls and holds imports, and those edges need a source.
KIND_MODULE = "module"
KIND_CLASS = "class"
KIND_FUNCTION = "function"
KIND_METHOD = "method"

# Edge kinds.
EDGE_CALLS = "calls"
EDGE_IMPORTS = "imports"
EDGE_INHERITS = "inherits"

# Tier constants (see schema.sql).
TIER_FACT = 0
TIER_RESOLVED = 1
TIER_INFERRED = 2


def content_hash(source: bytes) -> str:
    """sha256 of exactly these bytes, hex.

    THE binding primitive for the inference layer: a tier-2 assertion cites spans,
    stores their hashes, and expires when any of them changes. Keeping the hash
    function in one place means "what does this assertion depend on" has exactly
    one answer.
    """
    return hashlib.sha256(source).hexdigest()


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    qualname: str
    line_start: int  # 1-based, inclusive
    line_end: int    # 1-based, inclusive
    byte_start: int
    byte_end: int
    content_hash: str
    parent_qualname: str | None = None
    docstring: str | None = None
    signature: str | None = None


@dataclass(frozen=True)
class Edge:
    """A reference site.

    `dst_name` is the name exactly as written at the site -- a tier-0 fact that does
    not depend on resolution succeeding. Binding it to a specific symbol is a
    separate, fallible, tier-1 step; an edge that never resolves is still true and
    is kept rather than dropped.

    `local_name` matters only for imports, and it is the whole point of them: for
    `from a.b import events as events_mod`, `dst_name` is `a.b.events` but the code
    below writes `events_mod.tail()`. Discarding the alias -- as the first version
    of this did -- throws away the only key that can match the call site.
    """
    src_qualname: str
    kind: str
    dst_name: str
    line: int
    local_name: str | None = None


@dataclass
class FileExtract:
    path: str          # repo-root-relative, POSIX separators
    lang: str
    content_hash: str
    size_bytes: int
    mtime_ns: int
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
