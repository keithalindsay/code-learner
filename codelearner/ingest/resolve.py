"""Tier-1 name resolution: bind a written reference to the symbol it means.

Runs as a separate pass over an already-populated index, not inline with parsing.
Two reasons, both learned the hard way in the Phase-0 spike:

  - Resolution needs the WHOLE symbol table plus every import edge. Doing it during
    extraction means resolving against a half-built graph.
  - Resolvers are the part most likely to be wrong and to improve. Keeping the pass
    separate, and stamping each edge with the `resolver` that bound it, means a bad
    strategy can be found and re-run without re-parsing 430 files.

**Why this module exists at all.** The spike's naive strategy -- "bind a name if it
is unique repo-wide" -- resolved 98 of 34,013 call edges (0.3%). The diagnosis was
not a bug: 49.4% of calls target stdlib or third-party code and are *correctly*
unresolvable, while 50.3% are ambiguous by basename (`execute` has 19 definitions in
swarm-sync, `get` has 29). Uniqueness is simply not a signal that exists in real
code. Import context is.

Every strategy below carries an explicit confidence, and confidence never reaches
1.0 unless the binding is a tautology (the written name IS the qualname). A
resolver that guesses confidently is worse than one that abstains, because every
downstream retrieval trusts the graph.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .types import EDGE_IMPORTS, EDGE_INHERITS, TIER_FACT, TIER_RESOLVED

# Resolver identities, stamped onto each bound edge.
R_EXACT = "exact_qualname/v1"
R_SELF = "self_attr/v1"
R_SELF_INHERITED = "self_attr_inherited/v1"
R_MODULE_LOCAL = "module_local/v1"
R_IMPORT_ALIAS = "import_alias/v1"
R_CLASS_ATTR = "class_attr/v1"
R_UNIQUE = "unique_basename/v1"

# Confidence per strategy. These are judgements, not measurements -- the eval exists
# to check them. Ordered high to low; the resolver takes the first strategy that hits.
CONF = {
    R_EXACT: 1.0,           # the written name IS the qualname; tautological
    R_SELF: 0.95,           # `self.x` inside class C, and C defines x
    R_MODULE_LOCAL: 0.90,   # bare name defined at module scope in the same file
    R_SELF_INHERITED: 0.85, # `self.x` found on a base class rather than C itself
    R_IMPORT_ALIAS: 0.85,   # name traced through this module's own imports
    R_CLASS_ATTR: 0.85,     # `C.x` where C is a class in this module
    R_UNIQUE: 0.75,         # unique basename repo-wide; a property of this snapshot
}

# Attribute prefixes that denote "the enclosing class instance".
_SELF_PREFIXES = ("self.", "cls.")


@dataclass
class ResolveStats:
    total: int = 0
    resolved: int = 0
    by_resolver: dict[str, int] = field(default_factory=dict)
    external: int = 0
    ambiguous: int = 0

    @property
    def rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    @property
    def rate_of_internal(self) -> float:
        """Resolution rate excluding edges whose target is not in this repo at all.

        The honest denominator. Counting stdlib calls as resolution failures makes
        the resolver look broken when it is behaving correctly.
        """
        internal = self.total - self.external
        return self.resolved / internal if internal else 0.0


class _Index:
    """In-memory views of the symbol table, built once per resolve pass."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.by_qualname: dict[str, int] = {}
        self.kind_of: dict[int, str] = {}
        self.qual_of: dict[int, str] = {}
        self.by_name: dict[str, list[int]] = {}
        for row in conn.execute("SELECT id, qualname, name, kind FROM symbols"):
            sid, qual = row["id"], row["qualname"]
            self.by_qualname[qual] = sid
            self.kind_of[sid] = row["kind"]
            self.qual_of[sid] = qual
            self.by_name.setdefault(row["name"], []).append(sid)

        # module qualname -> {local binding name: target dotted path}.
        #
        # Keyed on `local_name`, NOT on the target's last segment. `import events
        # as events_mod` binds `events_mod`, and that is the name the call site
        # writes; keying on `events` silently misses every one of those calls.
        self.aliases: dict[str, dict[str, str]] = {}
        for row in conn.execute(
            "SELECT s.qualname AS src, e.dst_name AS dst, e.local_name AS local "
            "FROM edges e JOIN symbols s ON s.id = e.src_symbol_id WHERE e.kind = ?",
            (EDGE_IMPORTS,),
        ):
            module = _module_of(row["src"], self)
            target = row["dst"]
            key = row["local"] or target.rsplit(".", 1)[-1]
            self.aliases.setdefault(module, {})[key] = target

        # class qualname -> [base class names as written]
        self.bases: dict[str, list[str]] = {}
        for row in conn.execute(
            "SELECT s.qualname AS src, e.dst_name AS dst FROM edges e "
            "JOIN symbols s ON s.id = e.src_symbol_id WHERE e.kind = ?",
            (EDGE_INHERITS,),
        ):
            self.bases.setdefault(row["src"], []).append(row["dst"])


def _module_of(qualname: str, idx: _Index) -> str:
    """Walk up a dotted qualname to the enclosing module symbol."""
    parts = qualname.split(".")
    while parts:
        candidate = ".".join(parts)
        sid = idx.by_qualname.get(candidate)
        if sid is not None and idx.kind_of[sid] == "module":
            return candidate
        parts.pop()
    return ""


def _enclosing_class(qualname: str, idx: _Index) -> str | None:
    """The nearest enclosing class of a symbol, or None."""
    parts = qualname.split(".")[:-1]
    while parts:
        candidate = ".".join(parts)
        sid = idx.by_qualname.get(candidate)
        if sid is not None and idx.kind_of[sid] == "class":
            return candidate
        parts.pop()
    return None


def _lookup_on_class(cls_qual: str, attr: str, idx: _Index, depth: int = 0) -> tuple[int, str] | None:
    """Find `attr` on `cls_qual` or, failing that, on its base classes.

    Returns `(symbol_id, resolver_id)` so the caller can record whether the hit was
    direct or inherited -- they deserve different confidences, because an inherited
    match depends on the base-class binding also being right.
    """
    direct = idx.by_qualname.get(f"{cls_qual}.{attr}")
    if direct is not None:
        return direct, R_SELF if depth == 0 else R_SELF_INHERITED
    if depth >= 3:  # cycle + deep-hierarchy guard
        return None
    module = _module_of(cls_qual, idx)
    for base in idx.bases.get(cls_qual, []):
        base_qual = _resolve_class_name(base, module, idx)
        if base_qual is None:
            continue
        found = _lookup_on_class(base_qual, attr, idx, depth + 1)
        if found is not None:
            return found[0], R_SELF_INHERITED
    return None


def _resolve_class_name(written: str, module: str, idx: _Index) -> str | None:
    """Best-effort map a base-class name as written to a class qualname."""
    if written in idx.by_qualname:
        return written
    local = f"{module}.{written}" if module else written
    if local in idx.by_qualname:
        return local
    alias = idx.aliases.get(module, {}).get(written.split(".")[0])
    if alias is not None:
        tail = written.split(".", 1)[1] if "." in written else ""
        candidate = f"{alias}.{tail}" if tail else alias
        if candidate in idx.by_qualname:
            return candidate
    candidates = idx.by_name.get(written.rsplit(".", 1)[-1], [])
    classes = [c for c in candidates if idx.kind_of[c] == "class"]
    if len(classes) == 1:
        return idx.qual_of[classes[0]]
    return None


def _resolve_one(dst_name: str, src_qual: str, idx: _Index) -> tuple[int, str] | None:
    """Resolve one reference. Returns `(symbol_id, resolver_id)` or None.

    Strategies are tried in descending confidence. Returning None is a legitimate,
    common, and *correct* outcome -- roughly half of all calls in a real repo target
    code that simply is not in the repo.
    """
    module = _module_of(src_qual, idx)

    # 1. The written name is already a qualname.
    sid = idx.by_qualname.get(dst_name)
    if sid is not None:
        return sid, R_EXACT

    # 2. `self.foo` / `cls.foo` inside a class -- including inherited attributes.
    for prefix in _SELF_PREFIXES:
        if dst_name.startswith(prefix):
            attr = dst_name[len(prefix):]
            if "." in attr:  # `self.a.b` -- would need type inference; abstain.
                break
            cls = _enclosing_class(src_qual, idx)
            if cls is not None:
                hit = _lookup_on_class(cls, attr, idx)
                if hit is not None:
                    return hit
            break

    head, _, tail = dst_name.partition(".")

    if not tail:
        # 3. Bare name defined at module scope in this same file.
        local = f"{module}.{dst_name}" if module else dst_name
        sid = idx.by_qualname.get(local)
        if sid is not None:
            return sid, R_MODULE_LOCAL
        # 4. Bare name this module imported.
        alias = idx.aliases.get(module, {}).get(dst_name)
        if alias is not None:
            sid = idx.by_qualname.get(alias)
            if sid is not None:
                return sid, R_IMPORT_ALIAS
    else:
        # 5. `mod.func` where `mod` is something this module imported.
        alias = idx.aliases.get(module, {}).get(head)
        if alias is not None:
            sid = idx.by_qualname.get(f"{alias}.{tail}")
            if sid is not None:
                return sid, R_IMPORT_ALIAS
        # 6. `Class.attr` where `Class` lives in this module.
        cls_local = f"{module}.{head}" if module else head
        if cls_local in idx.by_qualname and idx.kind_of[idx.by_qualname[cls_local]] == "class":
            sid = idx.by_qualname.get(f"{cls_local}.{tail}")
            if sid is not None:
                return sid, R_CLASS_ATTR

    # 7. Last resort: a BARE name that is unique in the whole repo.
    #
    # Restricted to bare names deliberately. Applying it to dotted attribute access
    # was measured on swarm-sync and was actively harmful: 472 of 519 such bindings
    # were attribute calls, and the largest single group bound 38 `r.json()` calls
    # on an httpx response to a nested `_R.json` helper inside one test file. The
    # receiver's type is unknown, so `x.foo()` carries no evidence about which
    # `foo` is meant -- uniqueness of the name is a fact about the repo, not about
    # the call. Abstaining leaves a tier-0 edge, which is true; guessing produced a
    # tier-1 edge that was false and outranked everything else in the call graph.
    if not tail:
        candidates = idx.by_name.get(dst_name, [])
        if len(candidates) == 1:
            return candidates[0], R_UNIQUE

    return None


def resolve_all(conn: sqlite3.Connection) -> ResolveStats:
    """Resolve every edge in the index, updating `dst_symbol_id`/`tier`/`confidence`.

    Idempotent: re-running re-derives every binding from scratch, so an improved
    resolver can simply be run again over an existing index.
    """
    idx = _Index(conn)
    stats = ResolveStats(by_resolver={})

    rows = list(
        conn.execute(
            "SELECT e.id, e.kind, e.dst_name, s.qualname AS src_qual "
            "FROM edges e JOIN symbols s ON s.id = e.src_symbol_id"
        )
    )

    updates: list[tuple[int, int, float, str, int]] = []
    for row in rows:
        stats.total += 1
        hit = _resolve_one(row["dst_name"], row["src_qual"], idx)
        if hit is None:
            base = row["dst_name"].rsplit(".", 1)[-1]
            n = len(idx.by_name.get(base, []))
            if n == 0:
                stats.external += 1
            else:
                stats.ambiguous += 1
            continue
        sid, resolver = hit
        stats.resolved += 1
        stats.by_resolver[resolver] = stats.by_resolver.get(resolver, 0) + 1
        updates.append((sid, TIER_RESOLVED, CONF[resolver], resolver, row["id"]))

    from .. import db

    with db.transaction(conn):
        conn.execute(
            "UPDATE edges SET dst_symbol_id = NULL, tier = ?, confidence = NULL, "
            "resolver = NULL",
            (TIER_FACT,),
        )
        conn.executemany(
            "UPDATE edges SET dst_symbol_id = ?, tier = ?, confidence = ?, "
            "resolver = ? WHERE id = ?",
            updates,
        )

    return stats
