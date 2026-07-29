"""Reading paths: dependency ordering, centrality, module clustering, cycles.

Every rule in `codelearner.onboard.path` has a test here that FAILS if the rule is
deleted, not merely one that passes while the rule happens to be present. That
distinction is the whole standard: an ordering test built on a graph where the
correct answer also falls out of alphabetical order, or out of centrality alone,
proves nothing about the rule it claims to cover.

The fixtures are therefore adversarial on purpose:
  - the depth fixture gives the CALLER higher centrality than its callee, so a
    centrality-only implementation gets it wrong;
  - the centrality fixture names symbols so that alphabetical order is the exact
    reverse of centrality order;
  - the clustering fixture interleaves two modules by centrality, so an
    implementation without clustering produces an interleaved path.
"""
from __future__ import annotations

import subprocess

from codelearner.ingest import index_repo
from codelearner.onboard import (
    build_reading_path,
    dependency_depths,
    load_call_graph,
    pagerank,
    render_markdown,
    strongly_connected_components,
)


def _mkrepo(root, files: dict[str, str]):
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


def _index(tmp_path, name: str, files: dict[str, str]):
    repo = _mkrepo(tmp_path / name, files)
    conn, _ = index_repo(repo, index_path=tmp_path / f"{name}.db")
    return conn


def _names(path) -> list[str]:
    return [s.name for s in path.stops]


def _position(path, name: str) -> int:
    for stop in path.stops:
        if stop.name == name:
            return stop.order
    raise AssertionError(f"{name!r} is not on the path: {_names(path)}")


# ---------------------------------------------------------------------------
# rule 1 -- dependency depth: leaves before their callers
# ---------------------------------------------------------------------------

# `hub` is called by five callers and spreads its own rank across four helpers, so
# PageRank ranks `hub` strictly ABOVE the helper it depends on. A tour ordered by
# centrality alone therefore puts `hub` first, which is exactly the mistake the
# depth rule exists to prevent: the reader would meet a call to `tiny` before ever
# seeing `tiny`. (The first version of this fixture had `hub` calling only `tiny`,
# which made `tiny` inherit all of `hub`'s rank and come out MORE central -- the
# test passed, but it was not testing depth against centrality at all.)
_DEPTH_REPO = {
    "core.py": '''
def tiny():
    """The helper hub depends on."""
    return 1


def spare_one():
    return 1


def spare_two():
    return 2


def spare_three():
    return 3


def hub():
    """Called by everything, but it calls tiny first."""
    return tiny() + spare_one() + spare_two() + spare_three()


def a():
    return hub()


def b():
    return hub()


def c():
    return hub()


def d():
    return hub()


def e():
    return hub()
''',
}


def test_a_leaf_comes_before_its_caller(tmp_path):
    conn = _index(tmp_path, "depth", _DEPTH_REPO)
    path = build_reading_path(conn, limit=12)
    assert _position(path, "tiny") < _position(path, "hub")


def test_depth_beats_centrality_when_they_disagree(tmp_path):
    """The caller is MORE central than its callee, and still comes second.

    Deleting the depth sort key leaves centrality in charge, and centrality puts
    `hub` first -- so this fails if the rule is removed.
    """
    conn = _index(tmp_path, "depth2", _DEPTH_REPO)
    path = build_reading_path(conn, limit=12)
    hub = next(s for s in path.stops if s.name == "hub")
    tiny = next(s for s in path.stops if s.name == "tiny")
    assert hub.centrality > tiny.centrality, "fixture no longer adversarial"
    assert tiny.order < hub.order
    assert tiny.tier < hub.tier


def test_every_in_path_call_edge_points_backwards(tmp_path):
    """The invariant the whole tour rests on, checked over a layered repo.

    For every edge on the path, the callee's position must be strictly earlier
    than the caller's. This is the machine-checkable form of "a reader never meets
    a concept before its definition".
    """
    conn = _index(
        tmp_path,
        "layers",
        {
            "app.py": (
                "def leaf_one():\n    return 1\n\n"
                "def leaf_two():\n    return 2\n\n"
                "def mid_one():\n    return leaf_one() + leaf_two()\n\n"
                "def mid_two():\n    return leaf_two()\n\n"
                "def top():\n    return mid_one() + mid_two() + leaf_one()\n\n"
                "def entry():\n    return top()\n"
            ),
        },
    )
    path = build_reading_path(conn, limit=12)
    order = {s.qualname: s.order for s in path.stops}
    assert len(order) >= 6
    for stop in path.stops:
        for callee in stop.calls_here:
            assert order[callee] < stop.order, (
                f"{stop.qualname} (#{stop.order}) calls {callee} "
                f"(#{order[callee]}), which is scheduled later"
            )


def test_depth_is_longest_path_not_shortest():
    """`a -> b -> c` plus a shortcut `a -> c` must still put `b` strictly between.

    With shortest-path depth, `a` would sit at depth 1 (one hop from leaf `c`) and
    tie with `b`, which `a` calls. Longest-path depth is the reason that cannot
    happen.
    """
    out = {1: {2, 3}, 2: {3}, 3: set()}
    components = strongly_connected_components({1, 2, 3}, out)
    depth = dependency_depths(components, out)
    by_node = {comp[0]: depth[i] for i, comp in enumerate(components)}
    assert by_node[3] == 0
    assert by_node[2] == 1
    assert by_node[1] == 2


# ---------------------------------------------------------------------------
# rule 2 -- centrality within a depth tier
# ---------------------------------------------------------------------------

# All four leaves live in ONE module and call nothing, so they land in one depth
# tier and module clustering cannot separate them -- centrality is the only signal
# left. Their names are chosen so alphabetical order is the exact REVERSE of
# centrality order: `zebra` has four callers, `aardvark` has one. If the centrality
# key is deleted, the tiebreak falls through to qualname and the assertion flips.
_CENTRALITY_REPO = {
    "leaves.py": (
        "def zebra():\n    return 1\n\n"
        "def yak():\n    return 2\n\n"
        "def badger():\n    return 3\n\n"
        "def aardvark():\n    return 4\n"
    ),
    "users.py": (
        "from leaves import zebra, yak, badger, aardvark\n\n"
        "def u1():\n    return zebra() + yak() + badger() + aardvark()\n\n"
        "def u2():\n    return zebra() + yak() + badger()\n\n"
        "def u3():\n    return zebra() + yak()\n\n"
        "def u4():\n    return zebra()\n"
    ),
}


def test_centrality_orders_within_a_depth_tier(tmp_path):
    conn = _index(tmp_path, "central", _CENTRALITY_REPO)
    path = build_reading_path(conn, limit=12)
    tier0 = [s for s in path.stops if s.tier == 0 and s.path == "leaves.py"]
    names = [s.name for s in tier0]
    assert names == ["zebra", "yak", "badger", "aardvark"], names
    # And the fixture really is adversarial to an alphabetical fallback.
    assert names != sorted(names)


def test_pagerank_ranks_the_widely_called_symbol_highest(tmp_path):
    conn = _index(tmp_path, "pr", _CENTRALITY_REPO)
    nodes, out, _ = load_call_graph(conn)
    ranks = pagerank(nodes, out)
    names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM symbols")}
    by_name = {names[sid]: score for sid, score in ranks.items()}
    assert by_name["zebra"] > by_name["yak"] > by_name["badger"] > by_name["aardvark"]


def test_pagerank_is_deterministic_across_runs(tmp_path):
    """Same graph, same floats -- to the last bit.

    Fails if the accumulation stops iterating in sorted order, because float
    addition is not associative and set iteration order is not stable.
    """
    conn = _index(tmp_path, "det", _CENTRALITY_REPO)
    nodes, out, _ = load_call_graph(conn)
    assert pagerank(nodes, out) == pagerank(nodes, out)


def test_pagerank_of_an_empty_graph_is_empty():
    assert pagerank([], {}) == {}


# ---------------------------------------------------------------------------
# rule 3 -- module clustering
# ---------------------------------------------------------------------------

# Two modules whose leaves interleave STRICTLY by centrality:
#   alpha_hi > beta_hi > alpha_lo > beta_lo
# Ordering by centrality alone therefore yields alpha, beta, alpha, beta -- a tour
# that changes file on every single stop. Clustering must keep each file's stops
# contiguous. (An earlier version of this fixture left `beta_hi` and `alpha_lo`
# tied on centrality, so the qualname tiebreak grouped the files by accident and
# the test passed even with clustering deleted. Verified adversarial by mutation.)
_CLUSTER_REPO = {
    "alpha.py": (
        "def alpha_hi():\n    return 1\n\n"
        "def alpha_lo():\n    return 2\n"
    ),
    "beta.py": (
        "def beta_hi():\n    return 3\n\n"
        "def beta_lo():\n    return 4\n"
    ),
    "users.py": (
        "from alpha import alpha_hi, alpha_lo\n"
        "from beta import beta_hi, beta_lo\n\n"
        "def c1():\n    return alpha_hi() + beta_hi() + alpha_lo() + beta_lo()\n\n"
        "def c2():\n    return alpha_hi() + beta_hi() + alpha_lo()\n\n"
        "def c3():\n    return alpha_hi() + beta_hi()\n\n"
        "def c4():\n    return alpha_hi()\n"
    ),
}


def test_stops_from_one_file_are_contiguous(tmp_path):
    conn = _index(tmp_path, "cluster", _CLUSTER_REPO)
    path = build_reading_path(conn, limit=12)
    tier0 = [s for s in path.stops if s.tier == 0]
    # Confirm the fixture still interleaves by centrality, or the test is vacuous.
    by_centrality = sorted(tier0, key=lambda s: -s.centrality)
    assert [s.path for s in by_centrality] == [
        "alpha.py", "beta.py", "alpha.py", "beta.py",
    ]
    files = [s.path for s in tier0]
    # Each file must appear as exactly one unbroken run.
    runs = [f for i, f in enumerate(files) if i == 0 or files[i - 1] != f]
    assert len(runs) == len(set(runs)), f"path ping-pongs between files: {files}"
    assert {"alpha.py", "beta.py"} <= set(files)


def test_clustering_does_not_break_the_centrality_order_inside_a_file(tmp_path):
    conn = _index(tmp_path, "cluster2", _CLUSTER_REPO)
    path = build_reading_path(conn, limit=12)
    alpha = [s.name for s in path.stops if s.path == "alpha.py"]
    beta = [s.name for s in path.stops if s.path == "beta.py"]
    assert alpha == ["alpha_hi", "alpha_lo"]
    assert beta == ["beta_hi", "beta_lo"]
    # The file with the strongest member leads.
    assert _position(path, "alpha_hi") < _position(path, "beta_hi")


# ---------------------------------------------------------------------------
# cycles
# ---------------------------------------------------------------------------

_CYCLE_REPO = {
    "cyc.py": (
        "def helper():\n    return 0\n\n"
        "def alpha():\n    return gamma() + helper()\n\n"
        "def beta():\n    return alpha()\n\n"
        "def gamma():\n    return beta()\n\n"
        "def outside():\n    return alpha()\n"
    ),
}


def test_a_cyclic_call_graph_terminates_and_keeps_every_node(tmp_path):
    """The failure mode this guards is silent, not loud.

    A naive DFS over call edges never returns here; a Kahn topological sort returns
    promptly and DROPS alpha, beta and gamma -- the three functions in the cycle --
    leaving a tour that looks complete and is missing the hardest part of the code.
    Both are caught by asserting the node set, not just that the call finished.
    """
    conn = _index(tmp_path, "cycle", _CYCLE_REPO)
    path = build_reading_path(conn, limit=12)
    names = set(_names(path))
    assert {"alpha", "beta", "gamma", "helper", "outside"} <= names


def test_cycle_members_share_a_tier_and_are_listed_together(tmp_path):
    conn = _index(tmp_path, "cycle2", _CYCLE_REPO)
    path = build_reading_path(conn, limit=12)
    members = [s for s in path.stops if s.name in {"alpha", "beta", "gamma"}]
    assert len({s.tier for s in members}) == 1, "a cycle must occupy one tier"
    orders = sorted(s.order for s in members)
    assert orders == list(range(orders[0], orders[0] + 3)), (
        f"cycle members are not consecutive: {[(s.order, s.name) for s in path.stops]}"
    )


def test_a_cycle_is_reported_rather_than_hidden(tmp_path):
    conn = _index(tmp_path, "cycle3", _CYCLE_REPO)
    path = build_reading_path(conn, limit=12)
    assert len(path.cycles) == 1
    assert set(path.cycles[0]) == {"cyc.alpha", "cyc.beta", "cyc.gamma"}
    for stop in path.stops:
        if stop.name in {"alpha", "beta", "gamma"}:
            assert len(stop.cycle) == 2
            assert "cycle" in stop.reason.lower()


def test_the_leaf_a_cycle_depends_on_still_comes_first(tmp_path):
    """Condensing a cycle must not lose the ordering constraints around it."""
    conn = _index(tmp_path, "cycle4", _CYCLE_REPO)
    path = build_reading_path(conn, limit=12)
    assert _position(path, "helper") < _position(path, "alpha")
    assert _position(path, "gamma") < _position(path, "outside")


def test_tarjan_handles_a_long_chain_without_recursion_error():
    """5,000 deep -- five times Python's default recursion limit.

    A recursive Tarjan raises RecursionError here, and the traceback points at the
    tour generator rather than at the chain length that caused it.
    """
    n = 5000
    out = {i: {i + 1} for i in range(n - 1)}
    out[n - 1] = set()
    components = strongly_connected_components(range(n), out)
    assert len(components) == n
    depth = dependency_depths(components, out)
    assert max(depth.values()) == n - 1


def test_tarjan_finds_a_self_loop_as_a_single_component():
    """Direct recursion is one node, not a two-member cycle."""
    out = {1: {1, 2}, 2: set()}
    components = strongly_connected_components({1, 2}, out)
    assert sorted(components) == [(1,), (2,)]


def test_two_disjoint_cycles_are_separate_components():
    out = {1: {2}, 2: {1}, 3: {4}, 4: {3}}
    components = strongly_connected_components({1, 2, 3, 4}, out)
    assert sorted(components) == [(1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# selection, determinism, and the rendered output
# ---------------------------------------------------------------------------


def test_topic_seeds_the_path_from_retrieval(tmp_path):
    conn = _index(
        tmp_path,
        "topic",
        {
            "auth.py": (
                "def hash_password(pw):\n"
                '    """Hash a password for storage."""\n'
                "    return pw\n\n"
                "def verify_password(pw, stored):\n"
                '    """Verify a password against its stored hash."""\n'
                "    return hash_password(pw) == stored\n"
            ),
            "billing.py": (
                "def compute_invoice(items):\n"
                '    """Total an invoice."""\n'
                "    return sum(items)\n\n"
                "def render_invoice(items):\n"
                '    """Render an invoice."""\n'
                "    return str(compute_invoice(items))\n"
            ),
        },
    )
    path = build_reading_path(conn, topic="password verification", limit=4)
    assert path.topic == "password verification"
    names = _names(path)
    assert "verify_password" in names
    assert "hash_password" in names
    # ...and the dependency still precedes its caller.
    assert _position(path, "hash_password") < _position(path, "verify_password")


def test_a_topic_that_matches_nothing_falls_back_to_a_repo_tour(tmp_path):
    conn = _index(tmp_path, "miss", _CENTRALITY_REPO)
    path = build_reading_path(conn, topic="quantum chromodynamics", limit=5)
    assert len(path.stops) > 0
    assert path.topic == "quantum chromodynamics"


def test_tests_are_excluded_by_default_and_includable_on_request(tmp_path):
    conn = _index(
        tmp_path,
        "withtests",
        {
            "impl.py": "def widget():\n    return 1\n",
            "tests/test_impl.py": (
                "from impl import widget\n\n"
                "def test_widget():\n    assert widget() == 1\n"
            ),
        },
    )
    default = build_reading_path(conn, limit=10)
    assert all(not s.path.startswith("tests/") for s in default.stops)
    with_tests = build_reading_path(conn, limit=10, include_tests=True)
    assert any(s.path.startswith("tests/") for s in with_tests.stops)


def test_the_path_is_reproducible(tmp_path):
    """Two builds off one index must be identical, stop for stop.

    A tour that reorders itself between runs is not a curriculum, and it makes the
    generated Markdown churn in version control for no reason.
    """
    conn = _index(tmp_path, "repro", _CLUSTER_REPO)
    first = build_reading_path(conn, limit=8)
    second = build_reading_path(conn, limit=8)
    assert [s.qualname for s in first.stops] == [s.qualname for s in second.stops]
    assert render_markdown(first) == render_markdown(second)


def test_limit_is_respected(tmp_path):
    conn = _index(tmp_path, "limit", _CENTRALITY_REPO)
    assert len(build_reading_path(conn, limit=3).stops) == 3


def test_an_empty_index_yields_an_empty_path_not_a_crash(tmp_path):
    conn = _index(tmp_path, "empty", {"README": "not python\n"})
    path = build_reading_path(conn, limit=5)
    assert path.stops == ()
    assert "No symbols matched" in render_markdown(path)


def test_every_stop_carries_a_location_a_signature_and_a_reason(tmp_path):
    conn = _index(tmp_path, "meta", _DEPTH_REPO)
    path = build_reading_path(conn, limit=8)
    for stop in path.stops:
        assert stop.path and stop.line_start > 0
        assert stop.signature.strip()
        assert "\n" not in stop.signature, "signatures must be collapsed to one line"
        assert stop.reason.strip()
        assert "PageRank" in stop.reason


def test_the_reason_counts_the_callers_it_claims(tmp_path):
    """"called by 5 others" has to be checkable against the graph."""
    conn = _index(tmp_path, "reason", _DEPTH_REPO)
    path = build_reading_path(conn, limit=8)
    hub = next(s for s in path.stops if s.name == "hub")
    assert hub.callers_repo == 5
    assert "5 resolved callers repo-wide" in hub.reason
    tiny = next(s for s in path.stops if s.name == "tiny")
    assert "read before its callers" in tiny.reason


def test_markdown_shows_symbol_location_summary_and_why(tmp_path):
    conn = _index(tmp_path, "md", _DEPTH_REPO)
    path = build_reading_path(conn, limit=8)
    md = render_markdown(path)
    assert md.startswith("# Reading path")
    assert "core.py:2" in md  # file:line for `tiny`
    assert "The helper hub depends on." in md  # docstring first line
    assert "**Why here:**" in md
    assert "1. **Dependency depth first.**" in md
    # Stops appear in tour order in the body.
    positions = [md.index(f"### {s.order}. `{s.qualname}`") for s in path.stops]
    assert positions == sorted(positions)


def test_markdown_names_the_topic_when_there_is_one(tmp_path):
    conn = _index(tmp_path, "mdtopic", _CENTRALITY_REPO)
    md = render_markdown(build_reading_path(conn, topic="zebra", limit=4))
    assert "Reading path: zebra" in md
