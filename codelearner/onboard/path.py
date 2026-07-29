"""Reading paths: an ORDERED tour of a codebase, not a ranked list.

Retrieval answers *"what matches this query"*. Onboarding answers a different
question -- *"what do I read first, and what after that"* -- and a ranked list is
the wrong shape for it. Rank 1 in the hybrid pipeline is usually the most
*relevant* symbol, which in practice is the most *abstract* one: the orchestrator
that calls six helpers the reader has not met yet. Handing that to a new engineer
first is exactly backwards, and no amount of reranking fixes it, because relevance
and reading order are different properties.

**No LLM decides the order.** An ordering that changes between runs is not a
curriculum, and the three signals that matter are already sitting in the graph:

1. **Dependency depth.** Leaves first. A symbol that calls nothing else on the
   path is readable standalone; its callers only mean something once its callees
   are known. Depth is the *longest* path to a leaf, not the shortest -- with the
   shortest, a symbol two hops above a leaf could tie with the leaf's direct caller
   and be scheduled before something it depends on.
2. **Centrality (PageRank over the resolved call graph).** Within one depth tier
   the order is not arbitrary: the load-bearing symbol goes first. Rank flows
   *along* call edges (caller -> callee), so a symbol called by important code is
   itself important. Direction semantics are inherited from `retrieve.graph`
   (outbound = calls, inbound = called-by) rather than re-derived here.
3. **Module clustering.** A path that alternates between four files is technically
   well-ordered and practically unreadable -- every jump costs the reader the
   context they just built. Within a tier, stops are grouped by file, and files are
   ordered by their strongest member.

Only **tier-1 resolved** `calls` edges are traversed, for the same reason
`retrieve.graph` gives: an unresolved edge names something that could not be bound
to a symbol, so there is nothing on the far end to schedule.

**Cycles are handled, not assumed away.** Real code has them -- mutual recursion,
a module's parse/eval pair, a retry helper that calls back into its caller. Two
naive implementations both fail here: a DFS that follows call edges without a
visited set never terminates, and a Kahn-style topological sort silently *drops*
every node that is in a cycle, which is worse because the tour looks fine and is
missing exactly the code that was hardest to understand. This module condenses
strongly-connected components (Tarjan, iterative -- a recursive Tarjan blows the
stack on a real repo's import graph) and orders the resulting DAG. Every node
survives, a cycle occupies one depth tier as a unit, and the rendered tour *says*
it is a cycle, because "read these three together, they call each other" is real
information about the code and hiding it helps nobody.

**Scope of the ordering guarantee.** Depth and cycles are computed on the
*induced* subgraph of the stops actually selected, not on the whole repo. That is
the only claim the tour can honestly make: "nothing here is introduced before
something else here that it calls". A stop may still call symbols off the path
(stdlib, third-party, or repo code that did not make the cut) -- roughly half of
all calls in real code are out-of-repo and correctly unresolvable, so a tour that
promised total closure would be lying.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from ..index.embed import Embedder
from ..retrieve.graph import expand
from ..retrieve.lexical import Hit
from ..retrieve.search import search

# A tour longer than this stops being a tour. Twelve stops is roughly a
# 30-45 minute read of real code, which is about as much as anyone absorbs in one
# sitting before the early stops start leaking out of working memory.
DEFAULT_STOPS = 12

# How many symbols seed a repo-wide (no topic) tour before graph expansion pulls
# in what they depend on. Kept small deliberately: seeding with the top 12 by
# centrality and stopping there produces 12 hubs with almost no edges *between*
# them, which flattens the whole tour into one depth tier and throws away the
# ordering signal. Seeding with 4 and expanding downwards produces a connected
# subgraph, which is the thing that can actually be ordered.
SEED_COUNT = 4

# Standard PageRank damping. Not tuned -- there is no gold set for "correct
# centrality", so an invented constant would be false precision. 0.85 is the value
# every published comparison uses, which makes the number comparable to something.
DAMPING = 0.85

# Power iteration bounds. Converges in ~30 iterations on the swarm-sync graph
# (1,095 symbols); the cap exists so a pathological graph cannot hang a CLI.
PAGERANK_ITERATIONS = 100
PAGERANK_TOLERANCE = 1e-10

# Symbol kinds worth putting on a reading path. Modules are excluded: "read
# swarmsync/blackboard/db.py" is not a stop, it is the whole file, and a module
# symbol has no signature or body of its own to show.
READABLE_KINDS = ("function", "method", "class")


@dataclass(frozen=True)
class Stop:
    """One stop on a reading path, with everything needed to justify its position."""

    order: int  # 1-based position in the tour
    symbol_id: int
    qualname: str
    name: str
    kind: str
    path: str
    line_start: int
    line_end: int
    signature: str
    summary: str  # first line of the docstring; '' when there is none
    tier: int  # dependency depth; 0 == calls nothing else on this path
    centrality: float
    centrality_rank: int  # 1-based, within this path
    calls_here: tuple[str, ...]  # in-path callees -- read BEFORE this stop
    called_by_here: tuple[str, ...]  # in-path callers -- read AFTER this stop
    callers_repo: int  # resolved callers anywhere in the repo
    cycle: tuple[str, ...]  # the OTHER members of its call cycle; empty if none
    recursive: bool  # calls itself
    reason: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line_start}"


@dataclass(frozen=True)
class ReadingPath:
    """An ordered tour plus the graph facts that produced it."""

    stops: tuple[Stop, ...]
    topic: str | None
    repo_root: str
    graph_symbols: int  # nodes in the full resolved call graph the tour was cut from
    graph_edges: int
    cycles: tuple[tuple[str, ...], ...]  # multi-member SCCs found on the path
    tiers: int  # number of distinct depth tiers

    def __len__(self) -> int:
        return len(self.stops)


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------

Adjacency = dict[int, set[int]]


def load_call_graph(
    conn: sqlite3.Connection, include_tests: bool = False
) -> tuple[set[int], Adjacency, Adjacency]:
    """Load the resolved call graph as `(nodes, outbound, inbound)`.

    Self-edges are dropped from the adjacency. Direct recursion is real and is
    reported per-stop, but it is not a *dependency on something else*, and leaving
    it in would make every recursive function its own one-member cycle -- true,
    useless, and noisy in the output.

    Tests are excluded by default. A test calls the implementation, so including
    tests puts every test one tier ABOVE the code it exercises and floods the tour
    with assertions. The same measurement that motivated `files.is_test` applies:
    tests and implementations are different kinds of answer.
    """
    test_filter = "" if include_tests else " AND sf.is_test = 0 AND df.is_test = 0"
    kinds = ",".join(f"'{k}'" for k in READABLE_KINDS)

    nodes = {
        row["id"]
        for row in conn.execute(
            f"""
            SELECT s.id FROM symbols s JOIN files f ON f.id = s.file_id
            WHERE s.kind IN ({kinds})
            {'' if include_tests else 'AND f.is_test = 0'}
            """  # noqa: S608 - kinds is a module constant, never user data
        )
    }

    out: Adjacency = {}
    inn: Adjacency = {}
    for row in conn.execute(
        f"""
        SELECT DISTINCT e.src_symbol_id AS src, e.dst_symbol_id AS dst
        FROM edges e
        JOIN symbols s  ON s.id  = e.src_symbol_id
        JOIN files   sf ON sf.id = s.file_id
        JOIN symbols d  ON d.id  = e.dst_symbol_id
        JOIN files   df ON df.id = d.file_id
        WHERE e.kind = 'calls' AND e.dst_symbol_id IS NOT NULL
          AND s.kind IN ({kinds}) AND d.kind IN ({kinds})
          {test_filter}
        """  # noqa: S608 - both interpolations are module constants
    ):
        src, dst = row["src"], row["dst"]
        if src == dst or src not in nodes or dst not in nodes:
            continue
        out.setdefault(src, set()).add(dst)
        inn.setdefault(dst, set()).add(src)
    return nodes, out, inn


def pagerank(
    nodes: Iterable[int],
    out: Mapping[int, set[int]],
    damping: float = DAMPING,
    iterations: int = PAGERANK_ITERATIONS,
    tolerance: float = PAGERANK_TOLERANCE,
) -> dict[int, float]:
    """PageRank over the call graph, deterministic to the last bit.

    Every accumulation walks nodes in sorted id order. Floating-point addition is
    not associative, so iterating a `set` -- whose order depends on hash seeding
    and insertion history -- makes the scores differ in the last few digits between
    runs, which is enough to swap two near-tied stops and make the tour
    irreproducible. A tour that reorders itself on re-run is not a curriculum.

    Rank flows caller -> callee (along `out`): being called by important code makes
    a symbol important. Dangling nodes -- symbols that call nothing resolved, i.e.
    the leaves a reader should meet first -- would otherwise leak their mass out of
    the system entirely, so it is redistributed uniformly the standard way.
    """
    order = sorted(nodes)
    n = len(order)
    if n == 0:
        return {}
    rank = {v: 1.0 / n for v in order}
    outdeg = {v: len({w for w in out.get(v, ()) if w in rank}) for v in order}
    dangling = [v for v in order if outdeg[v] == 0]
    base = (1.0 - damping) / n

    for _ in range(iterations):
        leak = damping * sum(rank[v] for v in dangling) / n
        nxt = {v: base + leak for v in order}
        for v in order:
            degree = outdeg[v]
            if not degree:
                continue
            share = damping * rank[v] / degree
            for w in sorted(out[v]):
                if w in nxt:
                    nxt[w] += share
        delta = sum(abs(nxt[v] - rank[v]) for v in order)
        rank = nxt
        if delta < tolerance:
            break
    return rank


def strongly_connected_components(
    nodes: Iterable[int], out: Mapping[int, set[int]]
) -> list[tuple[int, ...]]:
    """Tarjan's SCC algorithm, iteratively.

    Iterative rather than recursive because the natural recursion depth is the
    length of the longest call chain, and Python's default limit (1000) is inside
    the range a real repo hits -- a chain of 1,200 resolved calls is not exotic in
    generated or deeply-layered code, and the failure mode is a RecursionError from
    inside a tour generator, which reads like a bug in the tour rather than a limit.

    Returns components with members sorted, in reverse topological order (a
    component is emitted before any component that calls into it).
    """
    node_set = set(nodes)
    index_of: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    components: list[tuple[int, ...]] = []
    counter = 0

    for root in sorted(node_set):
        if root in index_of:
            continue
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work: list[tuple[int, object]] = [
            (root, iter(sorted(w for w in out.get(root, ()) if w in node_set)))
        ]
        while work:
            v, children = work[-1]
            descended = False
            for w in children:  # type: ignore[attr-defined]
                if w not in index_of:
                    index_of[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append(
                        (w, iter(sorted(x for x in out.get(w, ()) if x in node_set)))
                    )
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index_of[w])
            if descended:
                continue
            work.pop()
            if low[v] == index_of[v]:
                component: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                components.append(tuple(sorted(component)))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return components


def dependency_depths(
    components: list[tuple[int, ...]], out: Mapping[int, set[int]]
) -> dict[int, int]:
    """Longest-path depth of each component in the condensation DAG.

    Depth 0 is a component that calls nothing else in scope -- a leaf, read first.
    Depth is `1 + max(depth of callees)`, i.e. the LONGEST path to a leaf, so a
    symbol is never scheduled in the same tier as something it transitively calls.
    (With shortest-path depth, `a -> b -> c` and `a -> c` puts `b` and `a` in
    tiers 1 and 1, and the reader meets `a` before `b`, which `a` calls.)

    Terminates because the condensation of any digraph is acyclic -- that is the
    entire reason cycles are condensed before this runs.
    """
    comp_of = {node: i for i, comp in enumerate(components) for node in comp}
    dag: dict[int, set[int]] = {i: set() for i in range(len(components))}
    for i, comp in enumerate(components):
        for node in comp:
            for callee in out.get(node, ()):
                j = comp_of.get(callee)
                if j is not None and j != i:
                    dag[i].add(j)

    depth: dict[int, int] = {}
    for start in range(len(components)):
        if start in depth:
            continue
        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            i, expanded = stack.pop()
            if expanded:
                depth[i] = 1 + max((depth[j] for j in dag[i]), default=-1)
                continue
            if i in depth:
                continue
            stack.append((i, True))
            for j in sorted(dag[i]):
                if j not in depth:
                    stack.append((j, False))
    return depth


# ---------------------------------------------------------------------------
# building the path
# ---------------------------------------------------------------------------


def build_reading_path(
    conn: sqlite3.Connection,
    topic: str | None = None,
    limit: int = DEFAULT_STOPS,
    include_tests: bool = False,
    embedder: Embedder | None = None,
) -> ReadingPath:
    """Build an ordered reading path over the indexed repo.

    With a `topic`, the tour is seeded from retrieval hits, so it answers "read
    these N things to understand auth". Without one, it is seeded from repo-wide
    centrality, so it answers "read these N things to understand this repo".

    Either way the seeds are then expanded *downwards* along outbound call edges
    (`retrieve.graph.expand(..., include_callers=False)`): a seed's dependencies
    are what a reader needs first, whereas its callers are what they can derive
    once they understand it. Reusing `expand` rather than re-walking the edges is
    deliberate -- its decay and fan-out caps are already measured, and a second
    hand-rolled traversal with slightly different direction semantics is precisely
    the kind of silent divergence that makes a graph layer untrustworthy.
    """
    nodes, out, inn = load_call_graph(conn, include_tests=include_tests)
    centrality = pagerank(nodes, out)
    edge_count = sum(len(v) for v in out.values())

    seeds = _seeds(
        conn,
        topic=topic,
        nodes=nodes,
        centrality=centrality,
        limit=limit,
        include_tests=include_tests,
        embedder=embedder,
    )
    selected = _select(
        conn,
        seeds=seeds,
        nodes=nodes,
        centrality=centrality,
        limit=limit,
        include_tests=include_tests,
    )
    if not selected:
        return ReadingPath(
            stops=(),
            topic=topic,
            repo_root=_repo_root(conn),
            graph_symbols=len(nodes),
            graph_edges=edge_count,
            cycles=(),
            tiers=0,
        )

    stops, cycles = _order(conn, selected, out, inn, centrality)
    return ReadingPath(
        stops=stops,
        topic=topic,
        repo_root=_repo_root(conn),
        graph_symbols=len(nodes),
        graph_edges=edge_count,
        cycles=cycles,
        tiers=len({s.tier for s in stops}),
    )


def _repo_root(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'repo_root'").fetchone()
    return "" if row is None else str(row["value"])


def _seeds(
    conn: sqlite3.Connection,
    topic: str | None,
    nodes: set[int],
    centrality: Mapping[int, float],
    limit: int,
    include_tests: bool,
    embedder: Embedder | None,
) -> list[Hit]:
    """Starting points for the tour: retrieval hits for a topic, hubs otherwise."""
    if topic:
        result = search(conn, topic, k=limit, embedder=embedder)
        hits = [h for h in result.hits if h.symbol_id in nodes]
        if not include_tests:
            hits = [h for h in hits if not h.is_test]
        if hits:
            return hits[:limit]
        # Fall through to centrality rather than returning nothing: a topic with no
        # lexical or dense match should still produce a usable repo tour, clearly
        # labelled with the topic that missed, instead of an empty page.

    ranked = sorted(nodes, key=lambda v: (-centrality.get(v, 0.0), v))[:SEED_COUNT]
    return _hydrate(conn, ranked, modality="centrality")


def _select(
    conn: sqlite3.Connection,
    seeds: list[Hit],
    nodes: set[int],
    centrality: Mapping[int, float],
    limit: int,
    include_tests: bool,
) -> list[int]:
    """Choose the symbols the tour covers: the seeds' call-graph neighbourhood.

    Seeds always win a slot -- they are what was asked for. Remaining slots go to
    the highest-activation neighbours, so the tour is a *connected* subgraph, which
    is the thing that can be ordered at all.

    **Selection walks both directions; ordering only walks one.** These got
    conflated in the first cut and the result was measurably bad: seeding a
    repo-wide tour from the top symbols by centrality and expanding only downwards
    produced 4 stops out of 12, because the most central symbols in a real repo are
    *leaves* -- on swarm-sync the top 14 by PageRank have a combined out-degree of
    5 among themselves, so there is nothing below them to expand into. Their
    callers are what a reader needs to see the hub being used. So callers are
    pulled into the *set* (at `retrieve.graph`'s measured `IN_WEIGHT`, which
    already discounts them) and then ordered strictly after what they call.
    """
    chosen: list[int] = []
    seen: set[int] = set()
    for hit in seeds[:limit]:
        if hit.symbol_id not in seen:
            seen.add(hit.symbol_id)
            chosen.append(hit.symbol_id)

    if len(chosen) < limit and seeds:
        extra = expand(conn, seeds, k=limit * 4, include_callers=True)
        candidates = [
            h
            for h in extra
            if h.symbol_id in nodes
            and h.symbol_id not in seen
            and (include_tests or not h.is_test)
        ]
        candidates.sort(
            key=lambda h: (-h.score, -centrality.get(h.symbol_id, 0.0), h.qualname)
        )
        for hit in candidates:
            if len(chosen) >= limit:
                break
            seen.add(hit.symbol_id)
            chosen.append(hit.symbol_id)

    return chosen


@dataclass(frozen=True)
class _Group:
    """One placement unit: a single symbol, or a whole call cycle kept together."""

    tier: int
    members: tuple[int, ...]
    centrality: float
    module: str
    key: str


def _order(
    conn: sqlite3.Connection,
    selected: list[int],
    out: Adjacency,
    inn: Adjacency,
    centrality: Mapping[int, float],
) -> tuple[tuple[Stop, ...], tuple[tuple[str, ...], ...]]:
    """Turn a set of symbols into the ordered tour. This is the whole ordering rule."""
    chosen = set(selected)
    induced: Adjacency = {
        v: {w for w in out.get(v, ()) if w in chosen and w != v} for v in chosen
    }
    induced_in: Adjacency = {
        v: {w for w in inn.get(v, ()) if w in chosen and w != v} for v in chosen
    }

    components = strongly_connected_components(chosen, induced)
    depth = dependency_depths(components, induced)
    meta = _symbol_meta(conn, selected)
    repo_callers = _repo_caller_counts(conn, selected)

    def qual(sid: int) -> str:
        row = meta.get(sid)
        return str(row["qualname"]) if row else str(sid)

    # A component (usually a single symbol; a cycle when not) is the unit that
    # gets placed, so a cycle's members can never be split apart by an unrelated
    # symbol wedged between them. Its placement is decided by its strongest member,
    # so a cycle containing one load-bearing symbol is not buried behind trivia.
    groups: list[_Group] = []
    for i, comp in enumerate(components):
        members = sorted(comp, key=lambda v: (-centrality.get(v, 0.0), qual(v)))
        best = members[0]
        groups.append(
            _Group(
                tier=depth[i],
                members=tuple(members),
                centrality=centrality.get(best, 0.0),
                module=str(meta[best]["path"]) if best in meta else "",
                key=qual(best),
            )
        )

    # Module clustering: within a tier, a file's position is set by its strongest
    # group, and all of that file's groups then run consecutively. Sorting by
    # centrality alone produces a correctly-ordered tour that ping-pongs between
    # files, and every jump costs the reader the context they just built.
    module_best: dict[tuple[int, str], float] = {}
    for g in groups:
        key = (g.tier, g.module)
        module_best[key] = max(module_best.get(key, 0.0), g.centrality)

    groups.sort(
        key=lambda g: (
            g.tier,  # 1. leaves before their callers
            -module_best[(g.tier, g.module)],  # 2. strongest file in the tier first
            g.module,  # 3. ...and keep that file's stops together
            -g.centrality,  # 4. load-bearing first within the file
            g.key,  # 5. deterministic tiebreak
        )
    )

    ordered: list[int] = [m for g in groups for m in g.members]
    rank_by_centrality = {
        sid: i + 1
        for i, sid in enumerate(
            sorted(ordered, key=lambda v: (-centrality.get(v, 0.0), qual(v)))
        )
    }
    tier_of = {m: g.tier for g in groups for m in g.members}
    cycle_of = {
        m: tuple(qual(x) for x in g.members if x != m)
        for g in groups
        if len(g.members) > 1
        for m in g.members
    }

    stops: list[Stop] = []
    for position, sid in enumerate(ordered, start=1):
        row = meta[sid]
        calls_here = tuple(sorted(qual(w) for w in induced.get(sid, ())))
        called_by_here = tuple(sorted(qual(w) for w in induced_in.get(sid, ())))
        cycle = cycle_of.get(sid, ())
        recursive = sid in out.get(sid, set()) or _self_calls(conn, sid)
        stop = Stop(
            order=position,
            symbol_id=sid,
            qualname=str(row["qualname"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            path=str(row["path"]),
            line_start=int(row["line_start"]),
            line_end=int(row["line_end"]),
            signature=_one_line(row["signature"]) or f"{row['name']}(...)",
            summary=_first_line(row["docstring"]),
            tier=tier_of[sid],
            centrality=centrality.get(sid, 0.0),
            centrality_rank=rank_by_centrality[sid],
            calls_here=calls_here,
            called_by_here=called_by_here,
            callers_repo=repo_callers.get(sid, 0),
            cycle=cycle,
            recursive=recursive,
            reason="",
        )
        stops.append(replace(stop, reason=_reason(stop, total=len(ordered))))

    cycles = tuple(
        tuple(qual(m) for m in g.members) for g in groups if len(g.members) > 1
    )
    return tuple(stops), cycles


def _reason(stop: Stop, total: int) -> str:
    """Say why this stop is at this position, in the graph's own terms.

    Every clause is a countable fact about the graph -- "called by 7 others", not
    "important". A justification a reader cannot check against the code is not a
    justification, and it is the first thing that goes stale.
    """
    parts: list[str] = []

    if stop.cycle:
        others = ", ".join(f"`{q}`" for q in stop.cycle)
        parts.append(
            f"Part of a {len(stop.cycle) + 1}-symbol call cycle with {others} -- "
            "a cycle has no leaf-first order, so the group is placed together "
            "and read as a unit."
        )
    elif stop.tier == 0 and stop.called_by_here:
        n = len(stop.called_by_here)
        parts.append(
            f"Leaf: it calls nothing else on this path, and {_plural(n, 'later stop')} "
            f"here {'call' if n != 1 else 'calls'} it -- read before its callers."
        )
    elif stop.tier == 0:
        parts.append(
            "Leaf: it neither calls nor is called by anything else on this path, "
            "so it can be read cold."
        )
    else:
        deps = ", ".join(f"`{q}`" for q in stop.calls_here)
        clause = (
            f"Depth {stop.tier}: builds on "
            f"{_plural(len(stop.calls_here), 'earlier stop')} ({deps})"
        )
        if stop.called_by_here:
            n = len(stop.called_by_here)
            clause += (
                f", and {_plural(n, 'later stop')} "
                f"{'call' if n != 1 else 'calls'} it"
            )
        else:
            clause += ", and nothing here calls it -- a top-level entry point"
        parts.append(clause + ".")

    parts.append(
        f"PageRank {stop.centrality:.4f} (#{stop.centrality_rank} of {total} on this "
        f"path); {_plural(stop.callers_repo, 'resolved caller')} repo-wide."
    )
    if stop.recursive:
        parts.append("Calls itself -- expect the base case to carry the logic.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _symbol_meta(conn: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    return {
        row["id"]: row
        for row in conn.execute(
            f"""
            SELECT s.id, s.qualname, s.name, s.kind, s.line_start, s.line_end,
                   s.signature, s.docstring, f.path
            FROM symbols s JOIN files f ON f.id = s.file_id
            WHERE s.id IN ({placeholders})
            """,  # noqa: S608 - placeholders are generated '?' marks, never data
            ids,
        )
    }


def _repo_caller_counts(conn: sqlite3.Connection, ids: list[int]) -> dict[int, int]:
    """Distinct resolved callers of each symbol across the WHOLE repo, tests included.

    Deliberately a different denominator from `Stop.called_by_here`: how many
    things on the tour call this is a statement about the tour, while how many
    things in the repo call it is a statement about the code. A reader deciding how
    much attention to spend wants the second one.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    return {
        row["dst"]: row["n"]
        for row in conn.execute(
            f"""
            SELECT dst_symbol_id AS dst, COUNT(DISTINCT src_symbol_id) AS n
            FROM edges
            WHERE kind = 'calls' AND dst_symbol_id IN ({placeholders})
              AND src_symbol_id != dst_symbol_id
            GROUP BY dst_symbol_id
            """,  # noqa: S608 - placeholders are generated '?' marks, never data
            ids,
        )
    }


def _self_calls(conn: sqlite3.Connection, symbol_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM edges WHERE kind = 'calls' AND src_symbol_id = ? "
        "AND dst_symbol_id = ? LIMIT 1",
        (symbol_id, symbol_id),
    ).fetchone()
    return row is not None


def _hydrate(conn: sqlite3.Connection, ids: list[int], modality: str) -> list[Hit]:
    """Turn symbol ids into `Hit`s so they can seed `retrieve.graph.expand`."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = {
        r["symbol_id"]: r
        for r in conn.execute(
            f"""
            SELECT s.id AS symbol_id, s.qualname, s.kind, s.line_start, s.line_end,
                   f.path, f.is_test, COALESCE(c.header, '') AS header
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            LEFT JOIN chunks c ON c.symbol_id = s.id
            WHERE s.id IN ({placeholders})
            """,  # noqa: S608 - placeholders are generated '?' marks, never data
            ids,
        )
    }
    hits = []
    for rank, sid in enumerate(ids):
        row = rows.get(sid)
        if row is None:
            continue
        hits.append(
            Hit(
                symbol_id=sid,
                qualname=row["qualname"],
                kind=row["kind"],
                path=row["path"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                score=1.0 / (1.0 + rank),
                modality=modality,
                header=row["header"],
                is_test=bool(row["is_test"]),
            )
        )
    return hits


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _one_line(value: object) -> str:
    """Collapse a possibly multi-line signature onto one line."""
    if not value:
        return ""
    return " ".join(str(value).split())


def _first_line(docstring: object) -> str:
    """First non-empty line of a docstring -- the summary, by PEP 257 convention."""
    if not docstring:
        return ""
    for line in str(docstring).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_markdown(path: ReadingPath, title: str | None = None) -> str:
    """Render a reading path as Markdown a human can follow top to bottom.

    Every stop carries its own justification. A tour that only says "read these in
    this order" is asking for trust; one that says "read this before its 7 callers"
    is showing its reasoning, and a reader who disagrees can check it against the
    call graph in a minute.
    """
    heading = title or (
        f"Reading path: {path.topic}" if path.topic else "Reading path"
    )
    lines: list[str] = [f"# {heading}", ""]

    if not path.stops:
        lines.append(
            "_No symbols matched. The index may be empty, or the topic may not "
            "appear in this repo._"
        )
        return "\n".join(lines) + "\n"

    scope = f"topic **{path.topic}**" if path.topic else "the repository as a whole"
    lines += [
        f"{len(path.stops)} stops, ordered for {scope}. Cut from a resolved call "
        f"graph of {path.graph_symbols} symbols and {path.graph_edges} call edges "
        f"(tests excluded).",
        "",
        "**How the order was chosen** -- deterministically, no model involved:",
        "",
        "1. **Dependency depth first.** A stop never appears before something on "
        "this path that it calls.",
        "2. **Centrality (PageRank) within a depth tier.** The load-bearing symbol "
        "goes first.",
        "3. **Module clustering.** Stops from the same file are kept together so "
        "the tour does not ping-pong across the repo.",
        "",
    ]
    if path.cycles:
        rendered = "; ".join(
            " <-> ".join(f"`{q}`" for q in cycle) for cycle in path.cycles
        )
        lines += [
            f"**{len(path.cycles)} call cycle"
            f"{'s' if len(path.cycles) != 1 else ''} on this path:** {rendered}. "
            "A cycle has no leaf-first order, so its members share one tier and are "
            "listed together -- read them as a unit.",
            "",
        ]
    lines += [
        "Depth and cycles are computed on this path's own subgraph. A stop may "
        "still call code that is not on the tour (stdlib, third-party, or repo "
        "code that did not make the cut).",
        "",
        "## At a glance",
        "",
        "| # | depth | symbol | where |",
        "|---|---|---|---|",
    ]
    for stop in path.stops:
        lines.append(
            f"| {stop.order} | {stop.tier} | `{stop.qualname}` | "
            f"`{stop.location}` |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    current_tier: int | None = None
    for stop in path.stops:
        if stop.tier != current_tier:
            current_tier = stop.tier
            label = (
                "Depth 0 -- leaves: read these cold"
                if stop.tier == 0
                else f"Depth {stop.tier} -- each of these calls something above"
            )
            lines += [f"## {label}", ""]
        lines += [
            f"### {stop.order}. `{stop.qualname}`",
            "",
            f"`{stop.path}:{stop.line_start}-{stop.line_end}` &middot; {stop.kind}",
            "",
            "```python",
            stop.signature,
            "```",
            "",
        ]
        if stop.summary:
            lines += [f"> {stop.summary}", ""]
        lines += [f"**Why here:** {stop.reason}", ""]
        if stop.calls_here:
            lines.append(
                "**Prerequisites already covered:** "
                + ", ".join(f"`{q}`" for q in stop.calls_here)
            )
            lines.append("")
        if stop.called_by_here:
            lines.append(
                "**Coming up, uses this:** "
                + ", ".join(f"`{q}`" for q in stop.called_by_here)
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
