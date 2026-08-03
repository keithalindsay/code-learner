"""Graph expansion: the modality that answers what lexical and dense cannot.

**The problem this exists to solve, measured rather than assumed.** On swarm-sync,
the query *"how does a lease expire and get reclaimed"* returns junk from BM25
(`_deny_response`, `_keepalive`) and reaper *tests* from vector search. Neither
modality surfaces `blackboard.leases` itself. Both rank tests above the code they
test, and for a good reason: a test states the behaviour in prose, names it in the
function title, and repeats the vocabulary. The implementation just does it.

The structural fix is that **a test is an edge to its subject**. If three tests
about lease expiry all call `leases.acquire`, then `acquire` accumulates activation
from all three even though no single test outranks it. Text similarity cannot see
that; the call graph can.

**Spreading activation, not neighbour-listing.** Each seed injects activation
proportional to its rank, and activation flows along edges with per-hop decay.
Something reached from several seeds accumulates -- convergence is the signal, and
it is exactly what distinguishes "the thing all these tests are about" from "a
function that happens to be one hop away".

**Direction is not symmetric and is not a detail.**
  - *Outbound* (`calls`): from a test to its subject, from a caller to its helpers.
    This is the direction that fixes the measured problem.
  - *Inbound* (reverse `calls`): from an implementation to its callers and tests.
    Useful for "who uses this", actively harmful for "how does this work".
Both are supported and weighted separately, because collapsing them into "the
neighbours" is how a graph modality becomes noise.

Only **tier-1 resolved** edges are traversed. An unresolved edge names something
that could not be bound to a symbol, so there is nothing on the far end to walk to.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .lexical import Hit

# Per-hop decay. Activation at distance d is scaled by DECAY**d, so a two-hop
# neighbour needs several independent paths to outrank a one-hop one. Chosen so a
# symbol reached from three seeds at one hop beats one reached from a single seed.
DECAY = 0.45

# Outbound edges (a calls b) carry more weight than inbound (b called by a).
# A test calling `acquire` is strong evidence the query about acquiring is about
# `acquire`. `acquire` having many callers says only that it is popular, which is
# a fact about the codebase rather than about the query.
OUT_WEIGHT = 1.0
IN_WEIGHT = 0.35

MAX_HOPS = 2

# Guard against activation flooding out of a hub. `db.init_db` has hundreds of
# callers; expanding all of them turns any query that grazes it into a scan of the
# whole repo. Hubs are informative precisely because they are connected to
# everything, which is the same reason they carry almost no query-specific signal.
MAX_FANOUT = 24


@dataclass
class _Node:
    activation: float = 0.0
    via: str = ""
    hops: int = 0
    sources: set[int] = field(default_factory=set)


def expand(
    conn: sqlite3.Connection,
    seeds: list[Hit],
    k: int = 10,
    max_hops: int = MAX_HOPS,
    include_callers: bool = True,
) -> list[Hit]:
    """Expand `seeds` along the call graph and return the top `k` NEW symbols.

    Seeds themselves are excluded from the result: this modality's job is to
    contribute what text retrieval missed, and the fusion step already has the
    seeds. Returning them would let one modality vote twice.
    """
    if not seeds:
        return []

    seed_ids = {h.symbol_id for h in seeds}
    nodes: dict[int, _Node] = {}

    # Seed activation falls off by rank: the top hit is a stronger starting point
    # than the tenth, and treating them equally lets a weak match steer expansion.
    frontier: dict[int, tuple[float, str]] = {}
    for rank, hit in enumerate(seeds):
        frontier[hit.symbol_id] = (1.0 / (1.0 + rank), hit.qualname)

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        next_frontier: dict[int, tuple[float, str]] = {}
        for symbol_id, (activation, origin) in frontier.items():
            for neighbour, direction in neighbours(conn, symbol_id, include_callers):
                weight = OUT_WEIGHT if direction == "calls" else IN_WEIGHT
                delivered = activation * weight * (DECAY**hop)
                if delivered <= 0.0:
                    continue
                node = nodes.setdefault(neighbour, _Node())
                node.activation += delivered
                node.sources.add(symbol_id)
                if not node.via or delivered > 0:
                    verb = "calls" if direction == "calls" else "called by"
                    node.via = f"{verb} {origin}"
                    node.hops = hop
                prior = next_frontier.get(neighbour, (0.0, ""))
                if delivered > prior[0]:
                    next_frontier[neighbour] = (delivered, origin)
        # Do not re-expand seeds or already-settled nodes through the next hop.
        frontier = {
            sid: val for sid, val in next_frontier.items() if sid not in seed_ids
        }

    ranked = sorted(
        ((sid, n) for sid, n in nodes.items() if sid not in seed_ids),
        key=lambda item: -item[1].activation,
    )[:k]
    if not ranked:
        return []
    return _hydrate(conn, ranked)


def neighbours(
    conn: sqlite3.Connection, symbol_id: int, include_callers: bool
) -> list[tuple[int, str]]:
    """Resolved graph neighbours of `symbol_id`, capped at `MAX_FANOUT` per direction.

    Public, and exported from `retrieve/__init__`, because `generate/pipeline.py`
    depends on it across a package boundary with a documented reason: the evidence
    a claim is offered over must be reached by the SAME traversal that retrieval
    uses, or the pipeline is quietly answering a different question about the graph
    than `search` is. That makes the fanout cap, the `calls`-only restriction and
    the confidence ordering part of a contract between two packages.

    It was `_neighbours`. A leading underscore on a name another package must import
    says "nobody depends on this, change it freely", which was false, and the cost of
    a wrong marker here is a refactor that silently changes what a claim is grounded
    in. `_neighbours` remains as an alias so nothing breaks.
    """
    out = conn.execute(
        "SELECT dst_symbol_id AS id FROM edges "
        "WHERE src_symbol_id = ? AND dst_symbol_id IS NOT NULL AND kind = 'calls' "
        "ORDER BY confidence DESC LIMIT ?",
        (symbol_id, MAX_FANOUT),
    ).fetchall()
    result = [(r["id"], "calls") for r in out]
    if include_callers:
        inbound = conn.execute(
            "SELECT src_symbol_id AS id FROM edges "
            "WHERE dst_symbol_id = ? AND kind = 'calls' "
            "ORDER BY confidence DESC LIMIT ?",
            (symbol_id, MAX_FANOUT),
        ).fetchall()
        result += [(r["id"], "called_by") for r in inbound]
    return result


def _hydrate(conn: sqlite3.Connection, ranked: list[tuple[int, _Node]]) -> list[Hit]:
    """Turn scored symbol ids into full `Hit`s in one query."""
    ids = [sid for sid, _ in ranked]
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
    for symbol_id, node in ranked:
        row = rows.get(symbol_id)
        if row is None:
            continue
        hits.append(
            Hit(
                symbol_id=symbol_id,
                qualname=row["qualname"],
                kind=row["kind"],
                path=row["path"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                score=node.activation,
                modality="graph",
                header=row["header"],
                is_test=bool(row["is_test"]),
                via=node.via,
            )
        )
    return hits


# The old private name, kept so `generate/pipeline.py` and any out-of-tree caller keep
# working across the rename. An alias, not a wrapper: the same function object, so
# patching one patches the other and a test cannot pass against a stale copy.
_neighbours = neighbours
