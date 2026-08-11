# Source-Rich Evidence Responses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one Code Learner search return bounded, whole-symbol, line-numbered source sections that an agent can evaluate and cite without a second filesystem call.

**Architecture:** Add a leaf `codelearner.evidence` package that owns source loading, whole-symbol rendering, deterministic budgeting, and omission reporting. Retrieval continues returning `Hit` objects; the evidence assembler hydrates their indexed byte coordinates and hashes from SQLite, verifies current bytes before rendering, and returns typed sections. MCP and CLI serialize the same `EvidenceBundle` without changing the default compact response.

**Tech Stack:** Python 3.12, SQLite, dataclasses, pathlib, pytest, existing CLI and MCP adapters.

## Global Constraints

- Scope is source-rich responses only; do not add assertion retrieval, serving policy, answer generation, new graph algorithms, or schema changes.
- Follow test-driven development: every production behavior is preceded by a focused failing test whose expected failure is observed.
- Source paths and byte coordinates come only from indexed rows; caller-supplied paths or ranges never reach the assembler.
- Resolve every source path beneath the repository root and refuse symlinks, non-regular files, missing files, and files above `MAX_SOURCE_FILE_BYTES = 2_000_000`.
- Verify the current whole-symbol bytes against `symbols.content_hash` before returning source.
- Include whole symbols or omit them; never slice a symbol to fit a response budget.
- Count UTF-8 encoded response content bytes against the budget. The server ceiling is `MAX_EVIDENCE_BYTES = 65_536`.
- Given the same database, repository bytes, hit order, and budget, output is deterministic.
- Preserve existing CLI and MCP response shapes unless `include_source=true` or `--include-source` is explicitly requested.
- Do not duplicate tier, ranking, or search logic inside `codelearner.evidence`.
- Keep public presentation logic shared below CLI and MCP.

---

### Task 1: Evidence value types and line-number rendering

**Files:**
- Create: `codelearner/evidence/__init__.py`
- Create: `codelearner/evidence/types.py`
- Create: `codelearner/evidence/render.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: no project-level API beyond standard-library dataclasses.
- Produces: `EvidenceSection`, `EvidenceBundle`, `number_source(source: str, line_start: int) -> str`, and `content_bytes(text: str) -> int`.

- [ ] **Step 1: Write failing value-type and rendering tests**

Add these tests to `tests/test_evidence.py`:

```python
from codelearner.evidence import EvidenceBundle, EvidenceSection, content_bytes, number_source


def test_number_source_uses_original_one_based_lines_and_preserves_content():
    assert number_source("def f():\n    return 1\n", 7) == (
        "7 | def f():\n8 |     return 1\n"
    )


def test_content_bytes_counts_encoded_bytes_not_characters():
    assert content_bytes("λ\n") == 3


def test_bundle_json_has_stable_explicit_omission_metadata():
    section = EvidenceSection(
        symbol_id=3,
        qualname="pkg.f",
        path="pkg.py",
        line_start=7,
        line_end=8,
        content_hash="abc",
        source="7 | def f():\n8 |     return 1\n",
        content_bytes=36,
    )
    bundle = EvidenceBundle(
        sections=(section,), budget_bytes=100, used_bytes=36,
        sections_omitted=2, omitted_symbol_ids=(8, 13),
    )
    assert bundle.as_json() == {
        "budget_bytes": 100,
        "used_bytes": 36,
        "truncated": True,
        "sections_omitted": 2,
        "omitted_symbol_ids": [8, 13],
        "sections": [{
            "symbol_id": 3,
            "qualname": "pkg.f",
            "path": "pkg.py",
            "line_start": 7,
            "line_end": 8,
            "content_hash": "abc",
            "content_bytes": 36,
            "source": "7 | def f():\n8 |     return 1\n",
        }],
    }
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: collection fails because `codelearner.evidence` does not exist.

- [ ] **Step 3: Implement the minimal immutable types and renderers**

In `types.py`, add frozen dataclasses with the fields shown by the test. Implement
`EvidenceBundle.truncated` as `sections_omitted > 0` and `as_json()` with the exact stable
shape in the test. In `render.py`, implement:

```python
def number_source(source: str, line_start: int) -> str:
    if line_start < 1:
        raise ValueError("line_start must be >= 1")
    return "".join(
        f"{line_number} | {line}"
        for line_number, line in enumerate(source.splitlines(keepends=True), line_start)
    )


def content_bytes(text: str) -> int:
    return len(text.encode("utf-8"))
```

Re-export all four public names from `codelearner/evidence/__init__.py`.

- [ ] **Step 4: Add edge tests and verify GREEN**

Add tests proving `number_source("", 1) == ""`, a final line without a newline remains
without one, and `line_start=0` raises `ValueError`. Run the focused test file and then:

```bash
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/evidence tests/test_evidence.py
/home/keith/projects/code-learner/.venv/bin/mypy codelearner/evidence --ignore-missing-imports
```

Expected: all commands pass with no warnings.

- [ ] **Step 5: Commit**

```bash
git add codelearner/evidence tests/test_evidence.py
git commit -m "feat: define source evidence response types"
```

---

### Task 2: Safe whole-symbol evidence assembler

**Files:**
- Create: `codelearner/evidence/assemble.py`
- Modify: `codelearner/evidence/__init__.py`
- Modify: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `retrieve.lexical.Hit`, a SQLite connection, repository root, and Task 1 types.
- Produces: `assemble_evidence(conn, repo_root, hits, *, budget_bytes) -> EvidenceBundle`, `EvidenceError`, `MAX_EVIDENCE_BYTES`, and `MAX_SOURCE_FILE_BYTES`.

- [ ] **Step 1: Add a real indexed-repository fixture and failing happy-path test**

Create a temporary git repository containing:

```python
def alpha():
    """A unicode λ docstring."""
    return 1


def beta():
    return alpha()
```

Index it through `index_repo`, search for `alpha`, call `assemble_evidence`, and assert:

- exactly one section is returned;
- the complete `alpha` body is present with original line numbers;
- `beta` is absent;
- the section hash equals the indexed symbol hash;
- `used_bytes` equals the UTF-8 byte length of the rendered section source.

- [ ] **Step 2: Run the happy-path test and verify RED**

Run the exact new test with `pytest -q`. Expected: import failure for `assemble_evidence`.

- [ ] **Step 3: Implement hydration and guarded loading**

Implement one parameterized hydration query over hit IDs selecting:

```text
s.id, s.qualname, s.line_start, s.line_end, s.byte_start, s.byte_end,
s.content_hash, f.path
```

Restore hit order after the query. For each row:

1. Resolve `(repo_root / path)` and require `candidate.is_relative_to(repo_root.resolve())`.
2. Require `candidate.is_file()` and reject symlinks with `candidate.is_symlink()`.
3. Require `stat().st_size <= MAX_SOURCE_FILE_BYTES`.
4. Read bytes once.
5. Require `byte_start >= 0`, `byte_end > byte_start`, and `byte_end <= len(source)`.
6. Require `content_hash(source[byte_start:byte_end]) == symbols.content_hash`.
7. Decode with UTF-8 using `errors="replace"`, then line-number with Task 1's renderer.

Raise `EvidenceError` with one of these stable codes: `path_escapes_repo`, `file_missing`,
`file_not_regular`, `file_too_large`, `invalid_span`, or `source_changed`. The exception
must expose `.code`, `.symbol_id`, and a non-sensitive `.message`.

- [ ] **Step 4: Add failing budget and safety tests**

Add focused tests proving:

- A first section larger than the budget is omitted rather than sliced.
- Later small sections may fit after an oversized earlier section is omitted.
- `budget_bytes=0` returns no sections and records all hit IDs as omitted.
- Negative budgets raise `ValueError`.
- Budgets above `MAX_EVIDENCE_BYTES` are clamped to the maximum.
- Duplicate symbol hits produce one section at the first occurrence.
- An edited symbol raises `EvidenceError(code="source_changed")`.
- A symlink replacing an indexed source raises `EvidenceError(code="file_not_regular")`.
- A row with an invalid byte end raises `EvidenceError(code="invalid_span")` without slicing.

Run the new tests and verify each fails for the missing behavior before implementing it.

- [ ] **Step 5: Implement deterministic whole-section budgeting**

Walk deduplicated hits in input order. A section fits only when its rendered byte count is less
than or equal to the remaining budget. Omitted IDs retain input order. Do not stop after an
oversized section because a later section may fit. Return immutable tuples in the bundle.

- [ ] **Step 6: Verify focused and package tests**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_evidence.py tests/test_read_guards.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/evidence tests/test_evidence.py
/home/keith/projects/code-learner/.venv/bin/mypy codelearner/evidence --ignore-missing-imports
```

Expected: all pass with no warnings.

- [ ] **Step 7: Commit**

```bash
git add codelearner/evidence tests/test_evidence.py
git commit -m "feat: assemble bounded whole-symbol evidence"
```

---

### Task 3: Source-rich MCP search

**Files:**
- Modify: `codelearner/server/app.py`
- Modify: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `assemble_evidence` from Task 2.
- Produces: backward-compatible `search_code(query, k=10, facts_only=False, include_source=False, evidence_budget=16_384)`.

- [ ] **Step 1: Write a failing compact-compatibility test**

Call the existing MCP `search_code` without the new arguments and assert that its top-level
keys remain exactly the pre-task set and that no `evidence` key appears. This pins the default
as a compatible response, not merely a response whose old fields happen to survive.

- [ ] **Step 2: Write a failing source-rich response test**

Call:

```python
payload = call(
    server,
    "search_code",
    query="frobnicate widgets",
    k=2,
    include_source=True,
    evidence_budget=4096,
)
```

Assert `payload["evidence"]` uses `EvidenceBundle.as_json()`, contains the complete
line-numbered body of the highest-ranked symbol, and its section's symbol ID occurs in
`payload["hits"]`.

- [ ] **Step 3: Run both tests and verify RED**

Expected: the compatibility test fails because the callable rejects no argument yet only after
the new test invokes it; the source-rich test fails with an unexpected keyword/tool argument.

- [ ] **Step 4: Implement MCP integration**

Extend `_search_body` and `search_code` with the exact parameters above. Only call
`source.repo_root(conn)` and `assemble_evidence` when `include_source` is true. Add the
serialized bundle under top-level `evidence` only in that mode.

Translate `EvidenceError` through the existing guarded tool-error vocabulary without exposing
absolute paths. Use a single stable MCP error code `evidence_unavailable` and include the
assembler's safe message in the error message.

Update the tool description to say:

- source is opt-in;
- sections are whole-symbol and line-numbered;
- the budget is clamped to 65,536 bytes;
- stale or unsafe source makes the call fail rather than returning indexed source as current.

- [ ] **Step 5: Add boundary and failure tests**

Test budget `0`, budget above the ceiling, source edited after indexing, and
`include_source=False` on a repository whose source file was deleted. The last case must still
return the compact indexed hits because source was not requested.

- [ ] **Step 6: Verify MCP tests and static checks**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_mcp.py tests/test_evidence.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/server/app.py tests/test_mcp.py
/home/keith/projects/code-learner/.venv/bin/mypy codelearner/server/app.py --ignore-missing-imports
```

Expected: all pass with no warnings.

- [ ] **Step 7: Commit**

```bash
git add codelearner/server/app.py tests/test_mcp.py
git commit -m "feat: return source evidence from MCP search"
```

---

### Task 4: Source-rich CLI search and documentation

**Files:**
- Modify: `codelearner/cli/main.py`
- Modify: `codelearner/cli/commands.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 evidence assembler and existing CLI search hits.
- Produces: `codelearner search ... --include-source --evidence-budget BYTES` in human and JSON modes.

- [ ] **Step 1: Write failing parser tests**

Assert `build_parser().parse_args(["search", "lease", "--include-source"])` sets
`include_source is True` and `evidence_budget == 16_384`. Assert an explicit
`--evidence-budget 2048` is parsed as integer `2048`.

- [ ] **Step 2: Write failing JSON response test**

Run search with `--json --include-source --evidence-budget 4096`. Assert the JSON response adds
the same `evidence` shape MCP returns. Run without `--include-source` and assert the `evidence`
key is absent.

- [ ] **Step 3: Run the tests and verify RED**

Expected: parser rejects `--include-source` and `--evidence-budget`.

- [ ] **Step 4: Implement parser and JSON integration**

Add both flags only to the search subparser. In `cmd_search`, after facts filtering, call the
shared assembler only when requested. Add `evidence` to the JSON dictionary only in that mode.
Convert `EvidenceError` into `CliError` without an absolute path or traceback.

- [ ] **Step 5: Implement deterministic human rendering**

After the existing ranked hit list, print:

```text

source evidence (USED/BUDGET bytes; N section(s) omitted)
--- QUALNAME  PATH:START-END  sha256:HASH ---
LINE | source
```

Print the omission count even when zero. Do not print source to stderr. Existing drift notes
remain on stderr.

- [ ] **Step 6: Update README and add output assertions**

Document the two flags and record the security/currentness boundary. Add a CLI test asserting
the section header, original line numbers, used/budget counts, and omission count. Add a test
that editing the source produces a clean `CliError` and no source body.

- [ ] **Step 7: Verify the phase**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_evidence.py tests/test_mcp.py tests/test_cli.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check .
/home/keith/projects/code-learner/.venv/bin/mypy codelearner --ignore-missing-imports
```

Then run the full suite:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/ -q
```

Expected: all pass with no warnings other than documented optional-dependency skips.

- [ ] **Step 8: Commit**

```bash
git add codelearner/cli/main.py codelearner/cli/commands.py tests/test_cli.py README.md
git commit -m "feat: expose source-rich CLI search"
```

