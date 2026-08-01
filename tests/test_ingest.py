"""Tier-0 extraction and tier-1 resolution.

Several tests here are regressions for bugs the Phase-0 spike found on a real
repository. Each one is named for the behaviour it pins, and each fails if the fix
is removed -- a test that passes with the fix deleted is not a test.
"""
from __future__ import annotations

import subprocess

import pytest

from codelearner import db
from codelearner.ingest import extract, index_repo, module_qualname
from codelearner.ingest.resolve import R_IMPORT_ALIAS, R_MODULE_LOCAL, R_SELF, R_UNIQUE
from codelearner.ingest.types import (
    EDGE_CALLS,
    EDGE_IMPORTS,
    EDGE_INHERITS,
    KIND_CLASS,
    KIND_METHOD,
    KIND_MODULE,
    TIER_FACT,
    TIER_RESOLVED,
)

# --------------------------------------------------------------------------
# module_qualname
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/sub/mod.py", "pkg.sub.mod"),
        ("pkg/sub/__init__.py", "pkg.sub"),
        ("mod.py", "mod"),
        ("__init__.py", ""),
    ],
)
def test_module_qualname_maps_paths_to_dotted_names(path, expected):
    assert module_qualname(path) == expected


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

SAMPLE = b'''"""Module doc."""
import os
import json as j
from a.b import events as events_mod
from a.b import thing


class Base:
    def shared(self):
        return 1


class Child(Base):
    """Child doc."""

    def go(self, n: int) -> str:
        return self.shared() + helper(n) + events_mod.tail()


def helper(n):
    return str(n)
'''


def test_extract_finds_every_definition_with_spans_and_hashes():
    fx = extract(SAMPLE, "pkg/mod.py")
    kinds = {s.qualname: s.kind for s in fx.symbols}
    assert kinds["pkg.mod"] == KIND_MODULE
    assert kinds["pkg.mod.Base"] == KIND_CLASS
    assert kinds["pkg.mod.Child"] == KIND_CLASS
    assert kinds["pkg.mod.Child.go"] == KIND_METHOD
    assert kinds["pkg.mod.helper"] == "function"

    go = next(s for s in fx.symbols if s.qualname == "pkg.mod.Child.go")
    assert go.line_start < go.line_end
    assert SAMPLE[go.byte_start : go.byte_end].startswith(b"def go")
    assert go.signature == "go(self, n: int) -> str"
    assert len(go.content_hash) == 64  # sha256 hex


def test_extract_records_docstrings_only_from_the_first_statement():
    fx = extract(SAMPLE, "pkg/mod.py")
    docs = {s.qualname: s.docstring for s in fx.symbols}
    assert docs["pkg.mod"] == "Module doc."
    assert docs["pkg.mod.Child"] == "Child doc."
    # `go` has a return, not a docstring -- must not pick up a random string.
    assert docs["pkg.mod.Child.go"] is None


def test_extract_attributes_calls_to_their_enclosing_symbol():
    fx = extract(SAMPLE, "pkg/mod.py")
    from_go = {e.dst_name for e in fx.edges
               if e.kind == EDGE_CALLS and e.src_qualname == "pkg.mod.Child.go"}
    assert {"self.shared", "helper", "events_mod.tail"} <= from_go


def test_extract_records_base_classes_as_inherits_edges():
    fx = extract(SAMPLE, "pkg/mod.py")
    inherits = [e for e in fx.edges if e.kind == EDGE_INHERITS]
    assert [(e.src_qualname, e.dst_name) for e in inherits] == [("pkg.mod.Child", "Base")]


def test_extract_keeps_the_import_alias_not_just_the_target():
    """REGRESSION. `import events as events_mod` binds `events_mod`, and that is the
    name the call site writes. The first version stored only the target's last
    segment (`events`), so every `events_mod.tail()` call silently failed to
    resolve. Deleting `local_name` makes this fail."""
    fx = extract(SAMPLE, "pkg/mod.py")
    imports = {e.local_name: e.dst_name for e in fx.edges if e.kind == EDGE_IMPORTS}
    assert imports["events_mod"] == "a.b.events"
    assert imports["j"] == "json"
    assert imports["thing"] == "a.b.thing"
    assert imports["os"] == "os"


def test_extract_survives_a_syntax_error_without_raising():
    """tree-sitter is error-tolerant; one broken file must not abort an index."""
    fx = extract(b"def broken(:\n    pass\n", "bad.py")
    assert fx.path == "bad.py"  # produced a result rather than raising


# --------------------------------------------------------------------------
# repo indexing + isolation
# --------------------------------------------------------------------------

def _mkrepo(root, files: dict[str, str], git: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if git:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


def test_index_repo_builds_symbols_and_resolves_local_calls(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "pkg/__init__.py": "",
        "pkg/core.py": "def helper(n):\n    return n\n\ndef go():\n    return helper(1)\n",
    })
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    assert stats.files == 2
    quals = {r["qualname"] for r in conn.execute("SELECT qualname FROM symbols")}
    assert {"pkg", "pkg.core", "pkg.core.helper", "pkg.core.go"} <= quals

    row = conn.execute(
        "SELECT e.resolver, e.tier, s.qualname FROM edges e "
        "JOIN symbols s ON s.id = e.dst_symbol_id "
        "WHERE e.kind = ? AND e.dst_name = 'helper'", (EDGE_CALLS,)
    ).fetchone()
    assert row["qualname"] == "pkg.core.helper"
    assert row["resolver"] == R_MODULE_LOCAL
    assert row["tier"] == TIER_RESOLVED


def test_index_resolves_calls_through_an_import_alias(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "pkg/__init__.py": "",
        "pkg/events.py": "def tail():\n    return []\n",
        "pkg/user.py": "from pkg import events as events_mod\n\ndef go():\n    return events_mod.tail()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT e.resolver, s.qualname FROM edges e JOIN symbols s ON s.id = e.dst_symbol_id "
        "WHERE e.dst_name = 'events_mod.tail'"
    ).fetchone()
    assert row is not None, "alias-qualified call did not resolve"
    assert row["qualname"] == "pkg.events.tail"
    assert row["resolver"] == R_IMPORT_ALIAS


def test_index_resolves_self_calls_to_the_enclosing_class(tmp_path):
    repo = _mkrepo(tmp_path / "r", {
        "m.py": "class C:\n    def a(self):\n        return 1\n    def b(self):\n        return self.a()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT e.resolver, s.qualname FROM edges e JOIN symbols s ON s.id = e.dst_symbol_id "
        "WHERE e.dst_name = 'self.a'"
    ).fetchone()
    assert row["qualname"] == "m.C.a"
    assert row["resolver"] == R_SELF


def test_unique_basename_never_binds_a_dotted_attribute_call(tmp_path):
    """REGRESSION. `r.json()` on an HTTP response must NOT bind to the only symbol
    in the repo that happens to be named `json`. Measured on swarm-sync, this
    strategy produced 472 attribute bindings, the largest group pointing 38
    `r.json()` calls at a nested test helper -- and those wrong edges then
    dominated the call graph.

    Deleting the `if not tail:` guard in resolve._resolve_one makes this fail."""
    repo = _mkrepo(tmp_path / "r", {
        "helpers.py": "class Fake:\n    def json(self):\n        return {}\n",
        "client.py": "import httpx\n\ndef fetch(r):\n    return r.json()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT dst_symbol_id, tier, resolver FROM edges WHERE dst_name = 'r.json'"
    ).fetchone()
    assert row is not None, "the call site itself must still be recorded"
    assert row["dst_symbol_id"] is None, "must abstain, not guess"
    assert row["tier"] == TIER_FACT
    assert row["resolver"] is None


def test_unique_basename_still_binds_a_bare_name(tmp_path):
    """The complement of the test above: restricting the strategy must not disable
    it. A bare `only_one()` call has to be in scope somehow."""
    repo = _mkrepo(tmp_path / "r", {
        "a.py": "def only_one():\n    return 1\n",
        "b.py": "from a import *\n\ndef go():\n    return only_one()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT e.resolver, s.qualname FROM edges e JOIN symbols s ON s.id = e.dst_symbol_id "
        "WHERE e.dst_name = 'only_one'"
    ).fetchone()
    assert row["qualname"] == "a.only_one"
    assert row["resolver"] == R_UNIQUE


def test_index_skips_agent_worktree_copies(tmp_path):
    """REGRESSION. `.claude/worktrees/` holds near-complete copies of the repo.
    Indexing them duplicated every symbol and produced cross-copy edges -- a call
    site in one copy binding to a definition in another. Found on swarm-sync, where
    it inflated a 68-file repo to 430 files."""
    repo = _mkrepo(tmp_path / "r", {
        "real.py": "def f():\n    return 1\n",
        ".claude/worktrees/agent-x/real.py": "def f():\n    return 1\n",
    }, git=False)
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    paths = {r["path"] for r in conn.execute("SELECT path FROM files")}
    assert paths == {"real.py"}
    assert stats.files == 1


def test_git_tracked_files_exclude_untracked_noise(tmp_path):
    """A git repo's own ignore rules are a better filter than any hand-kept list."""
    repo = _mkrepo(tmp_path / "r", {
        "kept.py": "def a():\n    return 1\n",
        ".gitignore": "generated.py\n",
    })
    (repo / "generated.py").write_text("def b():\n    return 2\n")
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    paths = {r["path"] for r in conn.execute("SELECT path FROM files")}
    assert "kept.py" in paths
    assert "generated.py" not in paths


def test_git_repo_with_nothing_committed_still_indexes(tmp_path):
    """REGRESSION. `git init` with no commit makes `git ls-files` return nothing.
    Taking that at face value indexed zero files and reported success -- found by
    pointing code-learner at its own repo before its first commit."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text("def f():\n    return 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)  # noqa: S603, S607
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    assert stats.files == 1
    assert {r["path"] for r in conn.execute("SELECT path FROM files")} == {"m.py"}


def _make_hostile_repo(tmp_path, name: str) -> tuple:
    """A normal repo whose `.git/config` carries an exec payload, and its sentinel.

    The payload is appended AFTER `git init`/`git add`, so the setup itself never
    runs it -- if the sentinel exists at the end of a test, the code under test is
    what ran it. Everything the payload can touch is inside `tmp_path`.
    """
    sentinel = tmp_path / f"{name}-EXECUTED"
    repo = _mkrepo(tmp_path / name, {"m.py": "def f():\n    return 1\n"})
    cfg = repo / ".git" / "config"
    hooks = tmp_path / f"{name}-hooks"
    hooks.mkdir()
    cfg.write_text(
        cfg.read_text()
        + f'[core]\n\tfsmonitor = "touch {sentinel}"\n\thooksPath = "{hooks}"\n'
    )
    return repo, sentinel


def test_indexing_a_repo_does_not_execute_its_git_config(tmp_path):
    """SECURITY REGRESSION. Git honours `core.fsmonitor` by EXECUTING it, and it
    reads that key from the config of the repo it has been pointed at -- so indexing
    any directory a second party can write to ran their command as us, silently,
    while the index reported success. Reproduced against the unhardened call: the
    sentinel below appeared during `git ls-files`.

    The overrides have to be on the command line. A config-file mitigation is
    defeated by the very thing being defended against, because the repo's config is
    what git is reading; only `-c` outranks it. `safe.directory` is no help either --
    it guards repos owned by a different uid, which is precisely the case this
    misses."""
    repo, sentinel = _make_hostile_repo(tmp_path, "hostile")
    conn, stats = index_repo(repo, index_path=tmp_path / "i.db")
    assert not sentinel.exists(), "indexing executed the repo's core.fsmonitor"
    # And the hardening did not cost the git listing it protects.
    assert stats.files == 1
    assert {r["path"] for r in conn.execute("SELECT path FROM files")} == {"m.py"}


def test_gold_mining_git_helper_does_not_execute_a_repo_git_config(tmp_path):
    """The same defect at the second call site. `eval/gold_from_history` funnels
    every git invocation it makes -- `rev-list`, the `log -L` line log, `show` --
    through one helper, and that helper needs the same argv because it is pointed at
    the same second-party repos.

    Exercised through `ls-files` rather than through `log`: on git 2.34 only the
    subcommands that refresh the index consult `core.fsmonitor`, so the history
    subcommands the miner actually runs cannot demonstrate the vector today. That is
    a property of one git version, not a guarantee, and the helper is generic over
    its arguments -- hardening the chokepoint is the only version-independent
    answer. Lives here, beside its twin, so the pair cannot drift apart unnoticed."""
    from codelearner.eval.gold_from_history import _git

    repo, sentinel = _make_hostile_repo(tmp_path, "hostile-eval")
    out = _git(repo, "ls-files")
    assert not sentinel.exists(), "the gold miner executed the repo's core.fsmonitor"
    assert out is not None and "m.py" in out


def test_two_repos_indexed_in_one_session_share_nothing(tmp_path):
    """The isolation guarantee: separate files, no shared rows, no cross-links."""
    a = _mkrepo(tmp_path / "a", {"m.py": "def alpha():\n    return 1\n"})
    b = _mkrepo(tmp_path / "b", {"m.py": "def beta():\n    return 2\n"})
    conn_a, _ = index_repo(a, index_path=tmp_path / "a.db")
    conn_b, _ = index_repo(b, index_path=tmp_path / "b.db")

    names_a = {r["name"] for r in conn_a.execute("SELECT name FROM symbols")}
    names_b = {r["name"] for r in conn_b.execute("SELECT name FROM symbols")}
    assert "alpha" in names_a and "alpha" not in names_b
    assert "beta" in names_b and "beta" not in names_a
    assert (tmp_path / "a.db").exists() and (tmp_path / "b.db").exists()


def test_reusing_one_index_across_two_repos_is_refused(tmp_path):
    """Isolation is structural (a file per repo) AND enforced -- pointing an
    existing index at a different root raises before anything is written."""
    a = _mkrepo(tmp_path / "a", {"m.py": "def alpha():\n    return 1\n"})
    b = _mkrepo(tmp_path / "b", {"m.py": "def beta():\n    return 2\n"})
    shared = tmp_path / "shared.db"
    index_repo(a, index_path=shared)
    with pytest.raises(db.RepoRootMismatchError):
        index_repo(b, index_path=shared)


def test_unresolved_edges_are_kept_not_dropped(tmp_path):
    """An unresolved call is still a true statement about the code. Roughly half of
    all calls in a real repo target stdlib or third-party code."""
    repo = _mkrepo(tmp_path / "r", {"m.py": "import os\n\ndef go():\n    return os.getcwd()\n"})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute("SELECT tier, dst_symbol_id FROM edges WHERE dst_name = 'os.getcwd'").fetchone()
    assert row is not None
    assert row["dst_symbol_id"] is None
    assert row["tier"] == TIER_FACT


def test_resolve_is_idempotent(tmp_path):
    """An improved resolver must be re-runnable over an existing index."""
    from codelearner.ingest.resolve import resolve_all

    repo = _mkrepo(tmp_path / "r", {
        "m.py": "def helper():\n    return 1\n\ndef go():\n    return helper()\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    first = conn.execute("SELECT count(*) c FROM edges WHERE dst_symbol_id IS NOT NULL").fetchone()["c"]
    stats = resolve_all(conn)
    second = conn.execute("SELECT count(*) c FROM edges WHERE dst_symbol_id IS NOT NULL").fetchone()["c"]
    assert first == second == stats.resolved
