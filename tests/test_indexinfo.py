"""The leaf extraction that closes `server/app.py`'s upward import into `cli`.

WP17.2. Three call sites in `server/app.py` reached `from ..cli.commands import ...`
for `REBUILD_ADVICE`, `_classify_unresolved`/`_embedding_info`/`_scalar`, and
`resolve_index_path` -- a server importing the module a person types commands into.
`codelearner/indexinfo.py` is the leaf both surfaces now depend on instead:
`cli/commands.py` re-exports the same seven names for existing importers, and
`server/app.py` points at the leaf.

This is read from source with `ast`, not from `sys.modules`, for the same reason
`tests/test_generate_purpose.py::test_the_package_import_graph_is_a_dag` reads
source: two of the three offending imports were function-local, invisible to
anything that only inspects what actually got imported at collection time.
"""
from __future__ import annotations

import ast
from pathlib import Path

import codelearner
import codelearner.indexinfo as indexinfo
from codelearner.cli import commands

_PACKAGE_ROOT = Path(codelearner.__file__).resolve().parent


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an `import` or `from import` anywhere in `path`.

    Walks the whole tree rather than just `tree.body`, so a function-local import
    counts exactly as much as a module-level one -- that is the gap that let the
    three `cli.commands` imports in `server/app.py` hide from a shallower check.
    Relative imports are resolved against the file's own package, honouring `level`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package_parts = path.resolve().relative_to(_PACKAGE_ROOT.parent).with_suffix("")
    parts = list(package_parts.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    # The package this file lives in, for resolving `level > 0` relative imports.
    is_package_init = path.name == "__init__.py"
    base = parts if is_package_init else parts[:-1]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.add(node.module)
                continue
            up = node.level - 1
            resolved = base[: len(base) - up] if up <= len(base) else []
            prefix = ".".join(resolved)
            if node.module:
                found.add(f"{prefix}.{node.module}" if prefix else node.module)
            else:
                for alias in node.names:
                    found.add(f"{prefix}.{alias.name}" if prefix else alias.name)
    return found


def _imports_from(path: Path, banned_prefixes: tuple[str, ...]) -> set[str]:
    return {
        m
        for m in _imported_modules(path)
        if any(m == p or m.startswith(p + ".") for p in banned_prefixes)
    }


def test_server_app_imports_nothing_from_cli():
    """The layering inversion this WP exists to close, checked at the source.

    `server/app.py` used to reach `from ..cli.commands import ...` three times, two
    of them inside function bodies (deferred for import cost, per the module's own
    docstring) so a check that only looked at `sys.modules` after a normal import
    would miss them entirely.
    """
    path = _PACKAGE_ROOT / "server" / "app.py"
    offenders = _imports_from(path, ("codelearner.cli",))
    assert offenders == set(), (
        f"server/app.py must not import from codelearner.cli, found: {offenders}"
    )


def test_indexinfo_is_a_leaf():
    """The new module imports nothing from `cli`, `server`, or `eval`.

    Those are exactly the three packages that must never depend on this one for the
    move to have actually closed a layering cycle rather than merely renamed it.
    """
    path = _PACKAGE_ROOT / "indexinfo.py"
    offenders = _imports_from(
        path, ("codelearner.cli", "codelearner.server", "codelearner.eval")
    )
    assert offenders == set(), f"indexinfo.py must be a leaf, found: {offenders}"


def test_indexinfo_exposes_the_seven_moved_names():
    for name in (
        "INDEX_RELPATH",
        "resolve_index_path",
        "REBUILD_ADVICE",
        "_scalar",
        "_meta",
        "_classify_unresolved",
        "_embedding_info",
    ):
        assert hasattr(indexinfo, name), f"codelearner.indexinfo has no {name}"


def test_cli_commands_still_reexports_the_same_objects():
    """Existing importers of these names from `cli.commands` must keep working.

    Identity, not just presence: `commands.X` has to be the very object
    `indexinfo.X` is, not a second definition that could drift from it.
    """
    for name in (
        "INDEX_RELPATH",
        "resolve_index_path",
        "REBUILD_ADVICE",
        "_scalar",
        "_meta",
        "_classify_unresolved",
        "_embedding_info",
    ):
        assert getattr(commands, name) is getattr(indexinfo, name), (
            f"cli.commands.{name} is no longer the same object as indexinfo.{name}"
        )
