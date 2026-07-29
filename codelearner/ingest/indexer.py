"""Walk a repository, extract every supported file, and persist the tier-0 graph.

Two passes, deliberately separated:

  1. **Extract + insert symbols** for every file. Symbols must all exist before any
     edge can be bound, because an edge's target routinely lives in a file that has
     not been read yet.
  2. **Insert edges**, resolving `dst_name` to a symbol id where possible. This is
     the tier-0 / tier-1 boundary: every edge is written, and `dst_symbol_id` is set
     only when a resolver is confident. Unresolved edges are kept, not discarded --
     an unresolved call is still a true statement about the code.
"""
from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .. import db
from .python_extract import extract_file
from .resolve import ResolveStats, resolve_all
from .types import TIER_FACT, FileExtract

# Directories never worth indexing. Only used by the non-git fallback walk; a git
# repo gets a far better answer from `git ls-files` (see `iter_python_files`).
SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
        "build", "dist", ".eggs", "site-packages", ".codelearner",
        # Agent scratch worktrees. Found the hard way in the Phase-0 spike: these
        # hold near-complete COPIES of the repo, so indexing them duplicated every
        # symbol several times over and produced cross-copy edges -- a call site in
        # one worktree binding to a definition in the main tree. Silent, and it
        # would have corrupted every retrieval measurement built on top.
        ".claude", ".worktrees",
    }
)



@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    edges: int = 0
    edges_resolved: int = 0
    skipped: int = 0
    resolve: ResolveStats = field(default_factory=ResolveStats)

    @property
    def resolution_rate(self) -> float:
        return self.edges_resolved / self.edges if self.edges else 0.0


def iter_python_files(repo_root: Path) -> Iterator[Path]:
    """Yield every indexable .py file under `repo_root`.

    Prefers `git ls-files` when the root is a git repo. That is not a convenience
    -- it is the correctness path. A hand-maintained skip list can only exclude
    directories somebody thought of, whereas the repo's own `.gitignore` already
    encodes exactly which files are real source and which are generated, vendored,
    or scratch. The Phase-0 spike indexed swarm-sync's `.claude/worktrees/` agent
    copies and produced a graph with five duplicate codebases cross-linked to each
    other; git would have excluded them without anyone needing to notice.

    Falls back to a filesystem walk with `SKIP_DIRS` for non-git directories.
    """
    tracked = _git_tracked_python_files(repo_root)
    if tracked is not None:
        yield from tracked
        return
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        if path.is_file():
            yield path


def _git_tracked_python_files(repo_root: Path) -> list[Path] | None:
    """Tracked `.py` files per git, or None if this is not a usable git repo."""
    if not (repo_root / ".git").exists():
        return None
    try:
        # S603/S607: `git` is resolved from PATH on purpose -- hard-coding a path
        # breaks every non-standard install. The argument vector is a fixed list
        # with no shell, and the only interpolated value is a filesystem path the
        # caller already chose to index.
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*.py"],  # noqa: S607
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    names = [n for n in proc.stdout.decode("utf-8", "replace").split("\0") if n]
    if not names:
        # A git repo that tracks no Python at all is almost always a repo with
        # nothing committed YET (`git init` run, first commit pending) rather than
        # a repo with no source. Returning an empty list there would silently index
        # zero files and report success -- found by pointing code-learner at itself
        # before its own first commit. Fall back to the filesystem walk.
        return None
    paths = [repo_root / n for n in names]
    return sorted(p for p in paths if p.is_file())


def index_repo(
    repo_root: Path,
    index_path: Path | None = None,
    files: Iterable[Path] | None = None,
) -> tuple[sqlite3.Connection, IndexStats]:
    """Index `repo_root` into a per-repo SQLite index and return (conn, stats).

    `index_path` defaults to `<repo_root>/.codelearner/index.db` -- one file per
    repo, which is what makes cross-repo contamination structurally impossible
    rather than merely discouraged.
    """
    repo_root = repo_root.resolve()
    if index_path is None:
        index_path = repo_root / ".codelearner" / "index.db"

    conn = db.init_db(index_path)
    db.bind_repo_root(conn, repo_root)

    stats = IndexStats()
    extracts: list[FileExtract] = []
    targets = list(files) if files is not None else list(iter_python_files(repo_root))

    for path in targets:
        try:
            extracts.append(extract_file(path, repo_root))
        except (OSError, UnicodeDecodeError, ValueError):
            # One unreadable file must not abort the index. Counted, not silent.
            stats.skipped += 1
            continue

    # Pass 1 -- files and symbols.
    with db.transaction(conn):
        for fx in extracts:
            cur = conn.execute(
                "INSERT INTO files (path, lang, content_hash, size_bytes, mtime_ns) "
                "VALUES (?,?,?,?,?) RETURNING id",
                (fx.path, fx.lang, fx.content_hash, fx.size_bytes, fx.mtime_ns),
            )
            file_id = cur.fetchone()[0]
            stats.files += 1
            for sym in fx.symbols:
                conn.execute(
                    "INSERT OR IGNORE INTO symbols "
                    "(file_id, kind, name, qualname, line_start, line_end, "
                    " byte_start, byte_end, content_hash, docstring, signature) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        file_id, sym.kind, sym.name, sym.qualname,
                        sym.line_start, sym.line_end, sym.byte_start, sym.byte_end,
                        sym.content_hash, sym.docstring, sym.signature,
                    ),
                )
                stats.symbols += 1

        # Parent links, now that every symbol row exists.
        conn.execute(
            "UPDATE symbols SET parent_id = ("
            "  SELECT p.id FROM symbols p"
            "  WHERE p.qualname = substr(symbols.qualname, 1,"
            "        length(symbols.qualname) - length(symbols.name) - 1)"
            ") WHERE instr(qualname, '.') > 0"
        )

    # Pass 2 -- edges, written unresolved (tier 0). Binding them to targets is a
    # separate pass so an improved resolver can re-run without re-parsing.
    qualnames = {
        row["qualname"]: row["id"]
        for row in conn.execute("SELECT id, qualname FROM symbols")
    }
    with db.transaction(conn):
        for fx in extracts:
            for edge in fx.edges:
                src_id = qualnames.get(edge.src_qualname)
                if src_id is None:
                    stats.skipped += 1
                    continue
                conn.execute(
                    "INSERT INTO edges "
                    "(src_symbol_id, kind, dst_name, local_name, tier, line) "
                    "VALUES (?,?,?,?,?,?)",
                    (src_id, edge.kind, edge.dst_name, edge.local_name,
                     TIER_FACT, edge.line),
                )
                stats.edges += 1

    # Pass 3 -- tier-1 resolution.
    rstats = resolve_all(conn)
    stats.edges_resolved = rstats.resolved
    stats.resolve = rstats

    return conn, stats
