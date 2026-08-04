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
# decorators are part of the symbol
# --------------------------------------------------------------------------
#
# The failure being defended against is the only FAIL-OPEN defect the audits found.
# tree-sitter's `function_definition` starts at `def`, not at `@`, so a symbol's
# stored span used to exclude every decorator above it. A claim "serves GET /users,
# cached 60s" then cited bytes containing neither `@route` nor `@cache`: rewrite the
# decorators and both verifiers report `fresh`, `force_hash=True` does not help
# because the cited bytes genuinely did not move, nothing lands in `staleness_log`,
# the faithfulness judge is shown the same truncated span and correctly rules the
# claim supported, and a human following the citation finds those exact bytes
# unchanged. There is no signal anywhere -- which is why these tests are here and
# why one of them goes all the way to expiry rather than stopping at the span.
#
# `test_an_undecorated_symbol_span_is_unchanged` is the control. Without it a change
# that widened EVERY span -- to the whole file, say -- would pass this whole section.

def _sym(source: bytes, qualname: str):
    return next(s for s in extract(source, "app.py").symbols if s.qualname == qualname)


def test_a_decorated_function_span_begins_at_the_at_sign():
    src = (
        b'@cache(ttl=60)\n'
        b'def list_users():\n'
        b'    """List users."""\n'
        b'    return []\n'
    )
    fn = _sym(src, "app.list_users")
    cited = src[fn.byte_start : fn.byte_end]
    assert cited.startswith(b"@cache(ttl=60)")
    assert b"def list_users" in cited
    assert fn.line_start == 1  # the `@` line, not the `def` line
    assert src.splitlines()[fn.line_start - 1].startswith(b"@")


def test_a_stacked_decorator_span_reaches_the_outermost_at_sign():
    """The span must start at the FIRST decorator. Taking `node.parent` gets this
    right for free; walking up one sibling at a time would stop at `@b`."""
    src = (
        b'@route("/users")\n'
        b'@auth_required\n'
        b'@cache(ttl=60)\n'
        b'def list_users():\n'
        b'    return []\n'
    )
    fn = _sym(src, "app.list_users")
    cited = src[fn.byte_start : fn.byte_end]
    assert cited.startswith(b'@route("/users")')
    for dec in (b'@route("/users")', b"@auth_required", b"@cache(ttl=60)"):
        assert dec in cited
    assert fn.line_start == 1


def test_a_decorated_class_span_begins_at_the_at_sign():
    src = b'@register\nclass Widget:\n    """A widget."""\n'
    cls = _sym(src, "app.Widget")
    assert src[cls.byte_start : cls.byte_end].startswith(b"@register\nclass Widget")
    assert cls.line_start == 1


def test_a_decorated_async_function_span_begins_at_the_at_sign():
    """`async def` is not a separate node type in this grammar -- the `async`
    keyword is a child of `function_definition` -- so it needs no special case, but
    it needs a test, because a grammar that DID split it would break silently."""
    src = b'@retry(times=3)\nasync def fetch():\n    return 1\n'
    fn = _sym(src, "app.fetch")
    assert src[fn.byte_start : fn.byte_end].startswith(b"@retry(times=3)\nasync def fetch")
    assert fn.line_start == 1


def test_a_decorator_whose_arguments_span_lines_is_inside_the_span():
    """The prevalence figure quoted in the audit was a lower bound precisely because
    the single-line-lookback heuristic that produced it misses this shape."""
    src = (
        b'@route(\n'
        b'    "/users",\n'
        b'    methods=["GET"],\n'
        b')\n'
        b'def list_users():\n'
        b'    return []\n'
    )
    fn = _sym(src, "app.list_users")
    cited = src[fn.byte_start : fn.byte_end]
    assert cited.startswith(b"@route(")
    assert b'methods=["GET"]' in cited
    assert fn.line_start == 1


def test_a_decorated_method_inside_a_decorated_class_keeps_both_decorators():
    """Nesting is where an implementation that looked at the enclosing node rather
    than the immediate parent goes wrong: the method's own `@property` and the
    class's `@register` are two different spans, and each symbol gets its own."""
    src = (
        b'@register\n'
        b'class Tray:\n'
        b'    @property\n'
        b'    def count(self):\n'
        b'        return 0\n'
    )
    cls = _sym(src, "app.Tray")
    meth = _sym(src, "app.Tray.count")
    assert src[cls.byte_start : cls.byte_end].startswith(b"@register\nclass Tray")
    assert src[meth.byte_start : meth.byte_end].startswith(b"@property\n    def count")
    assert meth.line_start == 3
    # The method's span is strictly inside its class's.
    assert cls.byte_start < meth.byte_start and meth.byte_end <= cls.byte_end


def test_a_comment_between_decorators_is_inside_the_span():
    """Not cosmetic: tree-sitter keeps the comment as a child of the
    `decorated_definition`, so a span taken from the parent covers it, and a reader
    following the citation sees the same bytes the hash was taken over."""
    src = b'@a\n# why this is here\n@b\ndef f():\n    return 1\n'
    fn = _sym(src, "app.f")
    assert src[fn.byte_start : fn.byte_end].startswith(b"@a\n# why this is here\n@b\ndef f")


def test_an_undecorated_symbol_span_is_unchanged():
    """THE CONTROL. Widening every span would satisfy every test above; this one
    fails if the span moved for a symbol that has no decorator."""
    src = b'def plain():\n    return 1\n\n\nclass Bare:\n    pass\n'
    fn = _sym(src, "app.plain")
    cls = _sym(src, "app.Bare")
    assert fn.byte_start == 0
    assert src[fn.byte_start : fn.byte_end] == b"def plain():\n    return 1"
    assert fn.line_start == 1 and fn.line_end == 2
    assert src[cls.byte_start : cls.byte_end] == b"class Bare:\n    pass"
    assert cls.line_start == 5 and cls.line_end == 6


def test_a_widened_span_takes_its_name_signature_and_docstring_from_the_def():
    """Only the START moves. The name, the signature and the docstring still come
    from the inner definition -- a symbol called `cache` because its decorator was
    read as the definition would be worse than the bug being fixed."""
    src = (
        b'@cache(ttl=60)\n'
        b'def list_users(active: bool = True) -> list:\n'
        b'    """List users."""\n'
        b'    return []\n'
    )
    fn = _sym(src, "app.list_users")
    assert fn.name == "list_users"
    assert fn.signature == "list_users(active: bool = True) -> list"
    assert fn.docstring == "List users."
    assert fn.line_end == 4


def test_the_stored_hash_covers_the_decorator_bytes():
    """The hash is what every verifier compares against, so it has to be the hash of
    the widened span. A span that moved without its hash moving would expire every
    decorated claim on the next sweep instead of none of them."""
    from codelearner.ingest.types import content_hash as _hash

    src = b'@cache(ttl=60)\ndef list_users():\n    return []\n'
    fn = _sym(src, "app.list_users")
    assert fn.content_hash == _hash(src[fn.byte_start : fn.byte_end])
    assert fn.content_hash != _hash(src[src.index(b"def") : fn.byte_end])


DECORATED_APP = (
    '@cache(ttl=60)\n'
    'def list_users():\n'
    '    """List every user."""\n'
    '    return []\n'
)
# Same LENGTH as the original, so every byte offset in the file is unmoved and the
# function\'s own bytes are identical. Under the old span this edit was invisible to
# both verifiers; it is the exact shape of the fail-open failure.
DECORATOR_REWRITTEN = DECORATED_APP.replace("@cache(ttl=60)", "@cache(ttl=99)")
assert len(DECORATOR_REWRITTEN) == len(DECORATED_APP)


def test_rewriting_only_a_decorator_expires_the_claim_that_cites_it(tmp_path):
    """THE TEST THAT WOULD HAVE CAUGHT THE BUG, end to end.

    Admit a claim about a decorated symbol, citing the span the index itself
    published; change nothing but the decorator's arguments, keeping the byte length
    identical so the function body neither moves nor changes; then ask the store to
    serve it. Before the fix the claim came back `fresh` with nothing in
    `staleness_log`, because the cited bytes genuinely had not changed -- the claim
    "responses are cached for 60 seconds" outliving the 60.
    """
    from codelearner.assertions import stale, store

    repo = _mkrepo(tmp_path / "r", {"app.py": DECORATED_APP})
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    sid = conn.execute(
        "SELECT id FROM symbols WHERE qualname = 'app.list_users'"
    ).fetchone()["id"]

    span = store.span_for_symbol(conn, sid)
    assert (repo / "app.py").read_bytes()[span.byte_start : span.byte_end].startswith(
        b"@cache(ttl=60)"
    ), "the index published a span that does not contain the decorator"

    aid = store.write_assertion(
        conn,
        subject_qualname="app.list_users",
        subject_symbol_id=sid,
        kind="purpose",
        claim="lists every user; responses are cached for 60 seconds",
        spans=[span],
        generator="test-model/v1",
        confidence=0.9,
    )
    assert [r.assertion.id for r in stale.serve_assertions(conn)] == [aid]

    (repo / "app.py").write_text(DECORATOR_REWRITTEN)

    assert stale.serve_assertions(conn) == [], "a rewritten decorator left the claim servable"
    assert [a.id for a in store.assertions_with_status(conn, store.STATUS_STALE)] == [aid]
    assert [row["reason"] for row in store.staleness_events(conn, aid)] == [
        stale.REASON_HASH_MISMATCH
    ]


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


def test_unique_basename_never_binds_a_bare_name_to_a_method(tmp_path):
    """REGRESSION. `range(...)` must NOT bind to the only symbol in the repo that
    happens to be a method named `range`. A bare name cannot reach a method -- that
    needs a receiver, and the statically knowable ones (`self`/`cls`) are handled by
    an earlier strategy. Measured on kalshi-bot, 366 `range(...)` calls bound to
    `KalshiCandle.range` at confidence 0.75 purely because Python's builtin is not
    in the symbol table; on swarm-sync 15 `json()` calls bound to a nested test
    helper's method. All 381 were wrong.

    Deleting the `kind_of[sid] != "method"` guard in resolve._resolve_one makes this
    fail."""
    repo = _mkrepo(tmp_path / "r", {
        "candles.py": "class Candle:\n    def range(self):\n        return 1\n",
        "loop.py": "def go(n):\n    return [i for i in range(n)]\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT dst_symbol_id, tier, resolver FROM edges WHERE dst_name = 'range'"
    ).fetchone()
    assert row is not None, "the call site itself must still be recorded"
    assert row["dst_symbol_id"] is None, "must abstain, not guess"
    assert row["tier"] == TIER_FACT
    assert row["resolver"] is None


def test_unique_basename_still_binds_a_method_called_inside_its_own_class(tmp_path):
    """The complement: a bare name IS a method reference inside the class body,
    where the method is in scope during execution. Restricting the strategy must not
    cost that binding."""
    repo = _mkrepo(tmp_path / "r", {
        "c.py": "class C:\n    def only_meth():\n        return 1\n\n    x = only_meth()\n",
        "d.py": "def unrelated():\n    return 2\n",
    })
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    row = conn.execute(
        "SELECT e.resolver, s.qualname FROM edges e "
        "JOIN symbols s ON s.id = e.dst_symbol_id WHERE e.dst_name = 'only_meth'"
    ).fetchone()
    assert row is not None, "a bare call inside the class body must still bind"
    assert row["qualname"] == "c.C.only_meth"
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


def test_a_from_import_does_not_emit_a_phantom_edge_naming_its_own_module(tmp_path):
    """REGRESSION. `child_by_field_name` returns a fresh wrapper on every call, so the
    identity test that was meant to drop the module node from its own import statement
    never matched. `from swarmsync import leases` emitted the real edge AND a phantom
    one named `swarmsync.swarmsync`.

    It mattered because those phantoms land in the resolution denominator whenever the
    basename matches a real module, so the rate was scored against references that do
    not appear anywhere in the source. Measured at 211, 942 and 754 phantom in-repo
    references on the three benchmark repos -- worth 3.5, 2.3 and 7.7 points of
    apparent coverage that was never real either way."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "leases.py").write_text("def acquire():\n    return True\n")
    (root / "use.py").write_text("from pkg import leases\nfrom pkg.leases import acquire\n")

    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    names = [r["dst_name"] for r in conn.execute("SELECT dst_name FROM edges")]

    assert not any(n.split(".")[-1] == n.split(".")[0] and "." in n for n in names), (
        f"a module was named as its own import target: {names}"
    )
    assert "pkg.pkg" not in names
    assert "pkg.leases.pkg" not in names
