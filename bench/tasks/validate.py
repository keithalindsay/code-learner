#!/usr/bin/env python3
"""Validate `bench/tasks/*.json` against the repos they claim to be about.

A benchmark task set is a set of assertions about somebody else's code, and every
one of them rots. A task naming a symbol that does not exist scores every arm zero
and reads as a hard task; an EXPLAIN task whose answer is sitting in a docstring
scores every arm high and reads as an easy one. Both are indistinguishable from a
real result unless something checks. This is that something.

Checks, in the order they run:

  1. SCHEMA        -- required fields present, ids unique, repo resolvable.
  2. SYMBOLS       -- every ground-truth qualname, every decoy, and every EXPLAIN
                      anchor resolves in that repo's `.codelearner/index.db`.
                      Decoys are checked too: a decoy that does not exist is not a
                      decoy, and the task is easier than its metadata claims.
  3. PROVENANCE    -- every EXPLAIN commit sha exists in the repo, and the quoted
                      committer prose really is a substring of that commit's
                      message (whitespace-normalised). This is what makes the
                      ground truth checkable by a reader rather than by trust.
  4. LEAK          -- `gold_from_history.find_leaks` over the anchor file's whole
                      text with the committer prose as the secret. A 32-character
                      shared clause means the answer was copied into the code.
  5. GREPPABLE     -- the task's own `reason_terms` must not all co-occur in any
                      single tracked file. This is the check that decides whether
                      an EXPLAIN task measures understanding or measures grep, and
                      it is deliberately run over the WHOLE tracked tree (markdown
                      and config prose included), not just the anchor file.
  6. QUESTION LEAK -- the question must not hand over its own answer's vocabulary.
                      For LOCATE, question tokens against the target's own symbol
                      name; for EXPLAIN, question tokens against the rubric.
                      Reported as a warning with the overlap named, because some
                      overlap is unavoidable domain vocabulary and a hard failure
                      on it would be noise.

Exit status is 0 only when there are no errors. Warnings do not fail the run but
are printed with the task id so they can be argued with.

Usage:  .venv/bin/python bench/tasks/validate.py [--tasks-dir DIR] [-v]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from codelearner.eval.gold_from_history import _tokens, find_leaks  # noqa: E402

PROJECTS = Path("/home/keith/projects")

#: Stale full copies of the docs live here in swarm-sync; a naive grep hits them
#: nine times over and says nothing new. Excluded from the greppability check for
#: the same reason `git ls-files` is used at all: we check what the repo IS.
SKIP_PREFIXES = (".claude/worktrees/",)

REQUIRED = ("id", "repo", "kind", "question", "ground_truth")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []
        self.checks = 0

    def error(self, tid: str, msg: str) -> None:
        self.errors.append(f"{tid}: {msg}")

    def warn(self, tid: str, msg: str) -> None:
        self.warnings.append(f"{tid}: {msg}")

    def note(self, tid: str, msg: str) -> None:
        self.notes.append(f"{tid}: {msg}")

    def ok(self) -> None:
        self.checks += 1


def repo_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else PROJECTS / name


def load_symbols(repo: Path) -> set[str]:
    db = repo / ".codelearner" / "index.db"
    if not db.exists():
        return set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("select qualname from symbols")}
    finally:
        con.close()


def symbol_paths(repo: Path) -> dict[str, str]:
    db = repo / ".codelearner" / "index.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {
            q: p
            for q, p in con.execute(
                "select s.qualname, f.path from symbols s join files f on f.id = s.file_id"
            )
        }
    finally:
        con.close()


def tracked_files(repo: Path) -> list[str]:
    # S603/S607: see `commit_message` -- fixed argv, no shell, read-only subcommand.
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "ls-files"],  # noqa: S607
        capture_output=True,
        check=False,
    ).stdout.decode("utf-8", "replace")
    keep = []
    for n in out.split("\n"):
        n = n.strip()
        if not n or any(n.startswith(d) for d in SKIP_PREFIXES):
            continue
        p = repo / n
        try:
            if p.is_file() and p.stat().st_size < 4_000_000:
                keep.append(n)
        except OSError:
            pass
    return keep


def commit_message(repo: Path, sha: str) -> str | None:
    # S603/S607: `git` from PATH, fixed argument vector, no shell, read-only
    # subcommand -- the same posture `gold_from_history._git` documents.
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "log", "-1", "--format=%B", sha],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def read_text(repo: Path, cache: dict[str, str], name: str) -> str:
    """`repo/name`'s text, memoised in `cache`.

    A free function taking its cache rather than a closure over the per-repo loop:
    a closure that captures the loop variable is the live foot-gun ruff's B023 is
    right about, and the validator walks one cache per repo inside that loop.
    """
    if name not in cache:
        try:
            cache[name] = (repo / name).read_text("utf-8", "replace")
        except OSError:
            cache[name] = ""
    return cache[name]


def flat(text: str) -> str:
    return " ".join(text.split())


def load_tasks(tasks_dir: Path) -> list[tuple[dict, Path]]:
    out: list[tuple[dict, Path]] = []
    for path in sorted(tasks_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        items = payload if isinstance(payload, list) else payload.get("tasks", [payload])
        for item in items:
            out.append((item, path))
    return out


def validate(tasks_dir: Path, verbose: bool = False) -> Report:
    rep = Report()
    tasks = load_tasks(tasks_dir)
    if not tasks:
        rep.errors.append(f"{tasks_dir}: no tasks found")
        return rep

    seen: dict[str, Path] = {}
    by_repo: dict[str, list[dict]] = {}
    for task, src in tasks:
        tid = task.get("id", f"<{src.name}:unnamed>")
        for field in REQUIRED:
            if not task.get(field):
                rep.error(tid, f"missing required field {field!r}")
        if tid in seen:
            rep.error(tid, f"duplicate id (also in {seen[tid].name})")
        seen[tid] = src
        if task.get("kind") not in ("locate", "explain"):
            rep.error(tid, f"kind must be 'locate' or 'explain', got {task.get('kind')!r}")
        by_repo.setdefault(task.get("repo", ""), []).append(task)

    for repo_name, group in by_repo.items():
        repo = repo_path(repo_name)
        if not repo.is_dir():
            for t in group:
                rep.error(t.get("id", "?"), f"repo not found: {repo}")
            continue
        symbols = load_symbols(repo)
        paths = symbol_paths(repo)
        if not symbols:
            for t in group:
                rep.error(t.get("id", "?"), f"no index at {repo}/.codelearner/index.db")
            continue
        files = tracked_files(repo)
        file_text: dict[str, str] = {}

        for task in group:
            tid = task["id"]
            gt = task.get("ground_truth") or {}

            # --- 2. SYMBOLS ------------------------------------------------------
            quals = list(gt.get("qualnames") or [])
            anchors = list((task.get("anchor") or {}).get("qualnames") or [])
            decoys = list(gt.get("decoys") or [])
            for q in quals + anchors:
                rep.ok()
                if q not in symbols:
                    rep.error(tid, f"ground-truth qualname does not resolve: {q}")
            for d in decoys:
                rep.ok()
                if d not in symbols:
                    rep.error(tid, f"decoy qualname does not resolve: {d}")
            if task["kind"] == "locate" and not quals:
                rep.error(tid, "locate task has no ground_truth.qualnames")

            if task["kind"] != "explain":
                # --- 6. QUESTION LEAK (locate) ----------------------------------
                #
                # Only a token that DISAMBIGUATES is a leak. "kelly" appears in the
                # question and in the target's name, but also in every decoy's name,
                # so handing it over selects nothing and the task is exactly as hard
                # as it was. Subtracting the decoys' vocabulary is what makes this
                # check quiet enough to be worth reading.
                qtok = set(_tokens(task["question"]))
                decoy_tok: set[str] = set()
                for d in decoys:
                    decoy_tok |= set(_tokens(d.rsplit(".", 1)[-1]))
                for q in quals:
                    leaf = q.rsplit(".", 1)[-1]
                    shared = (qtok & set(_tokens(leaf))) - decoy_tok
                    if shared:
                        rep.warn(
                            tid,
                            f"question shares {sorted(shared)} with target {leaf!r} "
                            f"and with no decoy -- it disambiguates",
                        )
                continue

            # --- 3. PROVENANCE ---------------------------------------------------
            prov = task.get("provenance") or {}
            sha = prov.get("commit", "")
            prose = (gt.get("committer_prose") or "").strip()
            rubric = gt.get("rubric") or []
            if not prose:
                rep.error(tid, "explain task has no ground_truth.committer_prose")
            if len(rubric) < 2:
                rep.error(tid, f"explain task needs >= 2 rubric points, has {len(rubric)}")
            rep.ok()
            msg = commit_message(repo, sha) if sha else None
            if msg is None:
                rep.error(tid, f"provenance commit not found in {repo.name}: {sha!r}")
            elif prose and flat(prose) not in flat(msg):
                rep.error(tid, f"committer_prose is not a substring of commit {sha[:8]}")

            # --- 4. LEAK ---------------------------------------------------------
            for q in anchors:
                path = paths.get(q)
                if path is None:
                    continue
                rep.ok()
                hits = find_leaks(read_text(repo, file_text, path), [prose]) if prose else []
                if hits:
                    rep.error(tid, f"committer prose is copied into {path}: {hits[0]}")

            # --- 5. GREPPABLE ----------------------------------------------------
            terms = [t.lower() for t in (task.get("flags") or {}).get("reason_terms", [])]
            if not terms:
                rep.error(tid, "explain task has no flags.reason_terms (greppability unchecked)")
            else:
                rep.ok()
                hits = [
                    n
                    for n in files
                    if all(t in read_text(repo, file_text, n).lower() for t in terms)
                ]
                if hits:
                    rep.error(
                        tid,
                        f"reason_terms {terms} all co-occur in {len(hits)} file(s): "
                        + ", ".join(hits[:3]),
                    )
                # Evidence, not a complaint. A reason term that appears in NO tracked
                # file is the strongest form of a pass: the committer's word for the
                # reason was never written into the repo. It is recorded per task so
                # the pass is auditable rather than asserted -- but a term must be
                # reason-bearing English, not an identifier the anchor contains by
                # construction, or the check proves nothing. `present` counts the ones
                # that do occur somewhere.
                present = sum(
                    1
                    for t in terms
                    if any(t in read_text(repo, file_text, n).lower() for n in files)
                )
                rep.note(
                    tid,
                    f"reason_terms {present}/{len(terms)} occur anywhere in tree, "
                    f"0 files contain all",
                )

            # --- 6. QUESTION LEAK (explain) --------------------------------------
            qtok = {t for t in _tokens(task["question"]) if len(t) >= 5}
            rtok = {t for t in _tokens(" ".join(rubric)) if len(t) >= 5}
            shared = qtok & rtok
            if len(shared) > 4:
                rep.warn(tid, f"question shares {len(shared)} content tokens with rubric: "
                              f"{sorted(shared)[:8]}")

    if verbose:
        for task, _ in tasks:
            print(f"  {task.get('id'):<12} {task.get('kind'):<8} {task.get('repo')}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    rep = validate(args.tasks_dir, args.verbose)
    tasks = load_tasks(args.tasks_dir)
    kinds: dict[tuple[str, str], int] = {}
    for t, _ in tasks:
        kinds[(t.get("repo", "?"), t.get("kind", "?"))] = (
            kinds.get((t.get("repo", "?"), t.get("kind", "?")), 0) + 1
        )
    print(f"{len(tasks)} tasks, {rep.checks} assertions checked")
    for (repo, kind), n in sorted(kinds.items()):
        print(f"  {repo:<12} {kind:<8} {n}")
    for n in rep.notes:
        print(f"NOTE  {n}")
    for w in rep.warnings:
        print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")
    print("FAIL" if rep.errors else "OK")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
