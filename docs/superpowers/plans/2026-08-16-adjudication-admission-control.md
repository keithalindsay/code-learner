# Adjudication as Admission Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the existing adjudication machinery a shipped `codelearner judge` command so submitted claims can earn the verdict serving already requires — closing the `search → submit → judge → serve` loop.

**Architecture:** Extract the judge-facing symbols from `eval/faithfulness.py` into a leaf module `codelearner/adjudicate.py` (re-exported for compatibility) so the CLI can drive judging without importing `eval` upward. Add a `judge` CLI subcommand that enumerates active-but-unjudged claims, judges each against its own citations with an independent model, and records verdicts through `record_verdict`. Serving policy is untouched: `require_verdict=True` is already the default.

**Tech Stack:** Python 3.11+, SQLite/FTS5, frozen dataclasses, argparse subcommands, pytest, Ruff, mypy. Judge backend is `OllamaJudge` (local Ollama over stdlib `urllib`); tests inject a deterministic fake `Judge`.

**Spec:** `docs/superpowers/specs/2026-08-16-adjudication-admission-control-design.md`

## Global Constraints

- Python 3.11+; frozen dataclasses for value types; full type annotations.
- **No test may call a real model.** Every judging test injects a fake `Judge` (the `Judge` protocol exists for exactly this).
- **Every verdict write goes through `store.record_verdict`**, never around it. What a non-supportive verdict does to a claim's status is that function's single decision.
- Serving policy is not modified. `PRODUCTION_POLICY` / `require_verdict=True` stay as-is.
- Per-task gate before every commit: the named tests pass, `ruff check .` clean, `mypy codelearner` clean, `git diff --check` clean.
- Judging is CLI-only. Do **not** add an MCP judge tool.
- Follow existing module and docstring conventions; re-export moved symbols so no existing import breaks.

---

## File Structure

- **Create** `codelearner/adjudicate.py` — leaf home for the judge machinery.
- **Modify** `codelearner/eval/faithfulness.py` — remove moved defs, re-export from `..adjudicate`; keep the report, Wilson, `score`, and `JudgeMisbehaving`.
- **Modify** `codelearner/eval/__init__.py` — imports still resolve (via faithfulness re-export or directly from `..adjudicate`).
- **Modify** `codelearner/assertions/store.py` — add `unjudged_assertions`.
- **Modify** `codelearner/cli/commands.py` — add `cmd_judge`.
- **Modify** `codelearner/cli/main.py` — add the `judge` subparser.
- **Create** `tests/test_adjudicate.py`, `tests/test_judge_command.py`.
- **Modify** `tests/test_mcp.py` — WP16 + WP17.7 pins.
- **Modify** `README.md`, `docs/REMEDIATION.md`, `docs/IMPLEMENTATION-GUIDE.md` — reconcile status.

---

### Task 1: Extract the judge machinery to a leaf module

This is a **pure move plus re-export**. Do not change any moved code's behavior; the existing suite is the safety net that it still works.

**Files:**
- Create: `codelearner/adjudicate.py`
- Modify: `codelearner/eval/faithfulness.py`, `codelearner/eval/__init__.py`
- Test: `tests/test_adjudicate.py`

**Interfaces:**
- Produces (importable from `codelearner.adjudicate`): `Judge` (Protocol with `name: str` and `judge(*, claim, evidence, subject) -> Judgement`), `Judgement`, `Adjudication`, `OllamaJudge`, `JudgeUnavailable`, `adjudicate_assertion(conn, judge, assertion, root, *, record=True) -> Adjudication`, `render_evidence(root, assertion) -> str`, `parse_judgement(text, judge) -> Judgement`, the `LABEL_*` constants and their `store.VERDICT_*` mapping, the `CAUSE_*` constants with `CAUSES`/`INSTRUMENT_CAUSES`, `DEFAULT_OLLAMA_HOST`, `_NO_EVIDENCE`.
- Stays in `eval/faithfulness.py`: `FaithfulnessReport`, `wilson_interval`, `score`/`score_decided`, the store-run loop, and `JudgeMisbehaving` (it carries a `FaithfulnessReport` and is raised only by the measurement run).

**Partition rule:** a symbol moves to the leaf iff it is needed to *judge one claim* (the protocol, the backend, parsing, evidence rendering, the per-claim result, the shared cause/label vocab). A symbol stays iff it is *measurement over many claims* (the report, intervals, aggregate scores, and the misbehavior abort that references the report). `JudgeUnavailable` is raised inside `OllamaJudge`, so it moves; `JudgeMisbehaving` references `FaithfulnessReport`, so it stays.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adjudicate.py
from __future__ import annotations

import ast
from pathlib import Path

import codelearner.adjudicate as adj


def test_leaf_exposes_the_judging_symbols():
    for name in (
        "Judge", "Judgement", "Adjudication", "OllamaJudge", "JudgeUnavailable",
        "adjudicate_assertion", "render_evidence", "parse_judgement",
        "DEFAULT_OLLAMA_HOST",
    ):
        assert hasattr(adj, name), name


def test_faithfulness_still_reexports_for_compatibility():
    from codelearner.eval import faithfulness
    assert faithfulness.adjudicate_assertion is adj.adjudicate_assertion
    assert faithfulness.OllamaJudge is adj.OllamaJudge
    # Measurement stays put.
    assert hasattr(faithfulness, "FaithfulnessReport")
    assert hasattr(faithfulness, "JudgeMisbehaving")


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module)
        elif isinstance(node, ast.Import):
            seen.update(alias.name for alias in node.names)
    return seen


def test_adjudicate_leaf_does_not_import_eval_or_cli():
    src = Path(adj.__file__)
    for module in _module_imports(src):
        assert not module.startswith("codelearner.eval"), module
        assert not module.startswith("codelearner.cli"), module
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adjudicate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'codelearner.adjudicate'`.

- [ ] **Step 3: Create the leaf and move the symbols**

Create `codelearner/adjudicate.py` and move the symbols named in the Interfaces block out of `codelearner/eval/faithfulness.py` **verbatim** (same code, same docstrings). The leaf's imports are only: `sqlite3`, `re`, `pathlib.Path`, stdlib `urllib`, `..assertions import store`, `..generate.llm import model_family` (used later by the command, safe to import), and `..ingest`/span types as the moved code already uses. In `faithfulness.py`, replace the removed defs with:

```python
from ..adjudicate import (  # re-exported: this module's home is the leaf now
    CAUSE_FORMAT_FAILURE,
    CAUSE_JUDGED,
    CAUSE_NO_EVIDENCE,
    CAUSE_PARSE_FAILURE,
    CAUSES,
    DEFAULT_OLLAMA_HOST,
    INSTRUMENT_CAUSES,
    LABEL_NOT_SUPPORTED,
    LABEL_SUPPORTED,
    LABEL_UNCERTAIN,
    Adjudication,
    Judge,
    Judgement,
    JudgeUnavailable,
    OllamaJudge,
    adjudicate_assertion,
    parse_judgement,
    render_evidence,
)
```

Keep `JudgeMisbehaving`, `FaithfulnessReport`, `wilson_interval`, `score`, and the run loop in `faithfulness.py`. If `eval/__init__.py` imported any moved symbol from `faithfulness`, it keeps working through the re-export — leave it unless mypy/ruff flags it, in which case point it at `..adjudicate`.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_adjudicate.py tests/test_faithfulness.py -q`
Expected: PASS. Then the safety net for a pure move:
Run: `.venv/bin/python -m pytest tests/ -q` → all pass. `.venv/bin/ruff check .` and `.venv/bin/mypy codelearner` → clean.

- [ ] **Step 5: Commit**

```bash
git add codelearner/adjudicate.py codelearner/eval/faithfulness.py codelearner/eval/__init__.py tests/test_adjudicate.py
git commit -m "refactor: extract judge machinery to codelearner.adjudicate leaf"
```

---

### Task 2: `unjudged_assertions` store query

**Files:**
- Modify: `codelearner/assertions/store.py`
- Test: `tests/test_judge_command.py`

**Interfaces:**
- Produces: `unjudged_assertions(conn, *, limit: int | None = None, subject: str | None = None) -> list[Assertion]` — active assertions that have no accepted (`supported`) verdict yet, i.e. exactly the claims serving withholds for lack of judgement. Read-only. Ordered oldest-first (`created_at`, then `id`) so a `--limit` run is deterministic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge_command.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codelearner.assertions import store
from codelearner.ingest import index_repo

SRC = '''def clamp(value):
    """Clamp."""
    if value < 0:
        value = 0
    return value
'''


@pytest.fixture()
def indexed(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text(SRC)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "i.db")
    return root, conn


def _submit(root, conn, claim="clamp forces value to be non-negative", generator="gen/v1"):
    row = conn.execute(
        "SELECT s.id, s.byte_start, s.byte_end, f.path FROM symbols s "
        "JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
        ("m.clamp",),
    ).fetchone()
    return store.write_assertion(
        conn,
        subject_qualname="m.clamp",
        subject_symbol_id=int(row["id"]),
        kind="invariant",
        claim=claim,
        spans=[store.span_for(root, row["path"], row["byte_start"], row["byte_end"])],
        generator=generator,
        repo_root=root,
    )


def test_unjudged_returns_active_claims_without_a_supporting_verdict(indexed):
    root, conn = indexed
    pending = _submit(root, conn)
    judged = _submit(root, conn, claim="clamp returns an int")
    store.record_verdict(conn, judged, "judge/v1", store.VERDICT_SUPPORTED, "ok")

    ids = [a.id for a in store.unjudged_assertions(conn)]
    assert pending in ids
    assert judged not in ids


def test_unjudged_does_not_mutate(indexed):
    root, conn = indexed
    _submit(root, conn)
    before = conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"]
    store.unjudged_assertions(conn, limit=1)
    after = conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"]
    assert before == after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judge_command.py -q`
Expected: FAIL — `AttributeError: module 'codelearner.assertions.store' has no attribute 'unjudged_assertions'`.

- [ ] **Step 3: Implement `unjudged_assertions`**

Add to `store.py`, reusing the existing assertion-loading helper (`_load_assertions`) so span hydration is not re-implemented. Select active assertion ids whose id has no row in `verdicts` with `verdict = VERDICT_SUPPORTED`, optionally filtered by `subject_qualname`, ordered `created_at, id`, limited by `limit`. Batch the id load through the existing chunked path — do not build an unbounded `IN`.

```python
def unjudged_assertions(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    subject: str | None = None,
) -> list[Assertion]:
    """Active claims with no accepted verdict yet -- what serving withholds for lack
    of judgement. Read-only; oldest first so a limited run is deterministic."""
    sql = (
        "SELECT a.id FROM assertions a WHERE a.status = ? "
        "AND NOT EXISTS (SELECT 1 FROM verdicts v WHERE v.assertion_id = a.id "
        "AND v.verdict = ?)"
    )
    params: list[object] = [STATUS_ACTIVE, VERDICT_SUPPORTED]
    if subject is not None:
        sql += " AND a.subject_qualname = ?"
        params.append(subject)
    sql += " ORDER BY a.created_at, a.id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    ids = [int(row["id"]) for row in conn.execute(sql, params)]
    return _load_assertions(conn, ids)
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_judge_command.py -q` → PASS. `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add codelearner/assertions/store.py tests/test_judge_command.py
git commit -m "feat: enumerate active unjudged assertions"
```

---

### Task 3: `cmd_judge` core — judge unjudged claims and close the serving loop

**Files:**
- Modify: `codelearner/cli/commands.py`, `codelearner/cli/main.py`
- Test: `tests/test_judge_command.py`

**Interfaces:**
- Consumes: `store.unjudged_assertions`, `adjudicate.adjudicate_assertion`, `adjudicate.Judge`.
- Produces: `cmd_judge(args, factory) -> int`; a `judge` subparser with `repo`, `--index`, `--limit`, `--model`, `--subject`, `--allow-same-family`, `--dry-run`, `--json`. A seam for tests to inject a fake judge: `cmd_judge` builds its judge via a module-level `_build_judge(args)` that tests monkeypatch, mirroring how the embedder factory is injected elsewhere.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_judge_command.py
from codelearner.adjudicate import Judgement
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import AssertionCandidate


class _FakeJudge:
    name = "fake/judge-b"

    def __init__(self, label):
        self._label = label

    def judge(self, *, claim, evidence, subject):
        from codelearner.adjudicate import CAUSE_JUDGED
        return Judgement(label=self._label, reasoning="fake", judge=self.name, cause=CAUSE_JUDGED)


def test_judge_makes_a_submitted_claim_servable(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    # Before judging: the claim is withheld.
    before = search_candidates(conn, root, "non-negative clamp", k=5)
    assert not [c for c in before.candidates if isinstance(c, AssertionCandidate)]

    args = _judge_args(root, tmp_path)
    assert commands.cmd_judge(args, factory=None) == 0

    after = search_candidates(conn, root, "non-negative clamp", k=5)
    assert [c for c in after.candidates if isinstance(c, AssertionCandidate)]
```

Add a `_judge_args` helper building a `types.SimpleNamespace` with the parsed fields (`repo=root`, `index=<the fixture db path>`, `limit=None`, `model=None`, `subject=None`, `allow_same_family=False`, `dry_run=False`, json=False), pointing `index` at the fixture's `i.db`. Reuse the existing index-path resolution the other commands use so the command opens the same store the fixture wrote to.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judge_command.py::test_judge_makes_a_submitted_claim_servable -q`
Expected: FAIL — `AttributeError: module 'codelearner.cli.commands' has no attribute 'cmd_judge'`.

- [ ] **Step 3: Implement `cmd_judge` and wire the subparser**

In `commands.py`: add `_build_judge(args)` returning `OllamaJudge(model=args.model)` (default model when `None`), and `cmd_judge(args, factory)` that resolves the index/root, opens the store, calls `store.unjudged_assertions(conn, limit=args.limit, subject=args.subject)`, builds the judge, and for each candidate calls `adjudicate_assertion(conn, judge, assertion, root, record=True)`, tallying `judgement.verdict`. Print a summary (`supported/refuted/uncertain/no_evidence`). Return `0`. (Independence, dry-run, and JSON arrive in Tasks 4–5; leave clean seams.)

In `main.py`, next to `p_learn`:

```python
p_judge = sub.add_parser("judge", help="adjudicate unjudged claims so serving can use them")
p_judge.add_argument("repo", type=Path, nargs="?", default=Path("."))
p_judge.add_argument("--index", type=Path, default=None)
p_judge.add_argument("--limit", type=int, default=None)
p_judge.add_argument("--model", default=None, help="ollama judge model tag")
p_judge.add_argument("--subject", default=None, help="only judge claims about this qualname")
p_judge.add_argument("--allow-same-family", action="store_true")
p_judge.add_argument("--dry-run", action="store_true")
p_judge.add_argument("--json", action="store_true", dest="json")
p_judge.set_defaults(func=cmd_judge)
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the whole test file plus a CLI smoke that `codelearner judge --help` parses. `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add codelearner/cli/commands.py codelearner/cli/main.py tests/test_judge_command.py
git commit -m "feat: add codelearner judge command"
```

---

### Task 4: Independence check + `--allow-same-family`

**Files:**
- Modify: `codelearner/cli/commands.py`
- Test: `tests/test_judge_command.py`

**Interfaces:**
- Consumes: `generate.llm.model_family(tag: str) -> str`, each `Assertion.generator`, `judge.name`.
- Produces: a `skipped_same_family` count in the summary; default behavior skips (records no verdict) when families match; `--allow-same-family` proceeds.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_judge_command.py
def test_same_family_judge_is_skipped_by_default(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    # generator family "qwen"; judge name "qwen3.5:9b" -> family "qwen" -> match.
    _submit(root, conn, generator="qwen3-coder:7b")
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))
    monkeypatch.setattr(_FakeJudge, "name", "qwen3.5:9b")

    args = _judge_args(root, tmp_path)
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 0


def test_allow_same_family_records_the_verdict(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn, generator="qwen3-coder:7b")
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))
    monkeypatch.setattr(_FakeJudge, "name", "qwen3.5:9b")

    args = _judge_args(root, tmp_path, allow_same_family=True)
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 1
```

Extend `_judge_args` to accept `allow_same_family=False`.

- [ ] **Step 2: Run test to verify it fails**

Expected: `test_same_family_judge_is_skipped_by_default` FAILS (verdict count is 1, not 0) because independence is not enforced yet.

- [ ] **Step 3: Implement the independence gate**

In `cmd_judge`, before adjudicating a candidate:

```python
from ..generate.llm import model_family
...
if not args.allow_same_family and model_family(judge.name) == model_family(assertion.generator):
    skipped_same_family += 1
    continue
```

Count and report `skipped_same_family`. A skipped claim records no verdict and stays unjudged.

- [ ] **Step 4: Run the tests and verify GREEN**

Both new tests pass; earlier tests still pass. `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add codelearner/cli/commands.py tests/test_judge_command.py
git commit -m "feat: refuse same-family judging unless explicitly allowed"
```

---

### Task 5: `--dry-run` and `--json` reporting

**Files:**
- Modify: `codelearner/cli/commands.py`
- Test: `tests/test_judge_command.py`

**Interfaces:**
- Produces: with `--dry-run`, `adjudicate_assertion(..., record=False)` so nothing is written; with `--json`, a machine-readable object of per-claim `{assertion_id, subject, verdict}` plus the summary counts.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_judge_command.py
import json


def test_dry_run_records_nothing(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    args = _judge_args(root, tmp_path, dry_run=True)
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 0


def test_json_emits_per_claim_verdicts(indexed, monkeypatch, tmp_path, capsys):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    args = _judge_args(root, tmp_path, as_json=True)
    commands.cmd_judge(args, factory=None)
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["supported"] == 1
    assert payload["results"][0]["verdict"] == "supported"
```

Extend `_judge_args` to accept `dry_run=False` and `as_json=False` (mapping to the `json` attribute).

- [ ] **Step 2: Run test to verify it fails**

Expected: `test_json_emits_per_claim_verdicts` FAILS (output is the plain summary, not JSON); `test_dry_run_records_nothing` FAILS (a verdict is written).

- [ ] **Step 3: Implement dry-run and JSON**

Thread `record=not args.dry_run` into `adjudicate_assertion`. Collect per-claim `{assertion_id, subject, verdict}` and summary counts; when `args.json`, `print(json.dumps({"summary": counts, "results": results}, indent=2))` instead of the text summary.

- [ ] **Step 4: Run the tests and verify GREEN**

All `tests/test_judge_command.py` pass. `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add codelearner/cli/commands.py tests/test_judge_command.py
git commit -m "feat: add judge --dry-run and --json"
```

---

### Task 6: WP16 pin — MCP tool refuses foreign vectors

This is a **characterization/pin**. The guard already exists in `IndexSource.embedder`; this locks it at the MCP-tool surface. It may pass on first run — if it does, say so in the commit body rather than manufacturing a failure.

**Files:**
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: the MCP `search_code` tool and the `IndexSource.embedder` mismatch note.

- [ ] **Step 1: Write the test**

Build an index embedded with one fake model, then drive the MCP search tool with an `embedder_factory` that returns a *different*-named model, and assert the tool response carries the "not comparable" / dense-disabled note and still returns lexical results (degrades, not crashes). Mirror the existing `test_cli.py:1021` assertion at the MCP layer, using the MCP harness already in `test_mcp.py`.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_mcp.py -k embedder -q`. Record whether it was RED first or GREEN on arrival.

- [ ] **Step 3: If RED, make the MCP surface forward the note; if GREEN, no code change**

- [ ] **Step 4: Verify GREEN;** `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mcp.py
git commit -m "test: pin MCP refusal of mismatched embedder vectors (WP16)"
```

---

### Task 7: WP17.7 pin — single-worker serialization guard

Also a **pin**. The shared-connection concurrency bug is closed by `run_sync`'s single owning worker; lock that so a future change back to a pool fails loudly.

**Files:**
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write the test**

Assert that `IndexSource`'s executor is single-threaded — after a `run_sync` call, `source._executor._max_workers == 1` — and (behavioral) that two overlapping `run_sync` calls never run concurrently (extend the existing `run_sync` concurrency test at `test_mcp.py:872`: have both tasks record entry/exit and assert no overlap).

- [ ] **Step 2: Run it.** Record RED-vs-GREEN honestly.

- [ ] **Step 3: If GREEN, no code change.**

- [ ] **Step 4: Verify GREEN;** `ruff`/`mypy` clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mcp.py
git commit -m "test: pin single-worker serialization of index ops (WP17.7)"
```

---

### Task 8: Reconcile the status docs

**Files:**
- Modify: `README.md`, `docs/REMEDIATION.md`, `docs/IMPLEMENTATION-GUIDE.md`

- [ ] **Step 1: Update the README roadmap and WP notes**

Mark Phase 10 (adjudication as admission control) **done** — a `codelearner judge` command records independent verdicts through `record_verdict`; serving already required them. Note that judging is CLI-only and out-of-band by design (independence), and that the lift question (Phase 11b) is still unmeasured and untouched.

- [ ] **Step 2: Update REMEDIATION.md**

WP17.4 → **done** (this branch). WP17.1/WP17.2 → note `adjudicate.py` joins `tier.py`/`sourceview.py` as an extracted leaf and the `cli`-does-not-import-`eval` acyclicity is now tested. WP16 (embedder guard) and WP17.7 (single-worker) → **closed, now pinned**, citing the new tests. Do not claim WP17.2 fully closed if `indexinfo.py` is still unextracted — state precisely what remains.

- [ ] **Step 3: Update IMPLEMENTATION-GUIDE.md**

In the Phase 3 / adjudication section, replace any "unreachable from any shipped surface" language with the shipped command, and keep the Phase 2.5 / 11b lift boundary marked not-measured.

- [ ] **Step 4: Verify docs match reality**

Grep the docs for stale claims (`facts_only is inert`, `adjudication is unreachable`, `no codelearner judge`) and fix any that remain. No test to run; read the diff.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/REMEDIATION.md docs/IMPLEMENTATION-GUIDE.md
git commit -m "docs: reconcile adjudication and closed audit items"
```

---

## Final acceptance

- [ ] `.venv/bin/python -m pytest tests/ -q` — all pass, count stamped.
- [ ] `.venv/bin/ruff check .` — clean.
- [ ] `.venv/bin/mypy codelearner` — clean.
- [ ] `git diff --check` — clean.
- [ ] End-to-end proof present: a submitted claim is withheld by search, and becomes servable only after `codelearner judge` records a supporting, independent verdict.
- [ ] No MCP judge tool was added; serving policy is unchanged.
