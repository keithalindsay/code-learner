# Task 2 report: schema v7 assertion search documents

## Status

Implemented schema v7 derived assertion documents and external-content FTS5 indexing.
Active assertions are indexed regardless of whether they have a verdict, preserving
the explicit pending-assertion research policy. Rejected and stale authoritative rows
and history remain stored while their derived search documents are removed.

## Files

- Created `codelearner/assertions/search_index.py`
- Created `tests/test_assertion_search_index.py`
- Modified `codelearner/schema.sql`
- Modified `codelearner/db.py`
- Modified `codelearner/assertions/store.py`
- Modified `codelearner/cli/commands.py`
- Modified `tests/test_db.py`
- Modified `tests/test_assertions.py`
- Modified `tests/test_cli.py`

## RED evidence

### Schema and canonical document

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_db.py -q
```

Observed: exit 1 with four expected failures. `assertion_documents` and
`assertions_fts` were absent, `codelearner.assertions.search_index` did not exist,
and `db.SCHEMA_VERSION` was 6 rather than 7.

### Search-index helpers

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py -q
```

Observed: exit 1 with five expected failures. The new behavior tests could not import
`sync_assertion_document`, `remove_assertion_document`,
`rebuild_assertion_documents`, or `assertion_search_structures_present`.

### Atomic authoritative synchronization

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_assertions.py -q
```

Observed: exit 1 with six expected mutation failures: admission created no document;
refuted and unsupported verdicts retained documents; staleness retained a document;
reinstatement did not restore one; and a monkeypatched sync failure was never called,
so it could not exercise rollback.

### v6 carry and derived rebuild

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_cli.py::test_v6_carry_preserves_history_and_rebuilds_only_active_search_documents -q
```

Observed: exit 1. All exact authoritative assertion, evidence-span, verdict, and
staleness rows survived, but `assertion_documents` was empty instead of containing
the active supported and active pending IDs.

### Full-suite architectural regression

The first full-suite run found one failure:
`test_the_module_import_graph_is_a_dag` reported
`store -> search_index -> store`. Root-cause reproduction with the single DAG test
confirmed a new module-level back-edge. The store dependency was moved to a local
mutation-boundary import; the DAG and transaction-rollback regression tests then
passed together.

## GREEN evidence

- Schema/canonical/helper focused run: exit 0, 24 tests passed.
- Search-index and assertion mutation focused run: exit 0, 77 tests passed.
- v6 carry/rebuild test: exit 0.
- Required four-module focused run:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_assertions.py tests/test_db.py tests/test_cli.py -q
```

  Exit 0.

- Ruff:

```text
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/assertions/search_index.py codelearner/assertions/store.py codelearner/db.py codelearner/cli/commands.py tests/test_assertion_search_index.py
```

  `All checks passed!`

- mypy:

```text
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/assertions/search_index.py codelearner/assertions/store.py
```

  `Success: no issues found in 2 source files`

- `git diff --check`: exit 0.
- Final full suite:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest -q
```

  Exit 0 at 100% with the existing skip and no failures.

## Self-review

- Transaction rollback: all four authoritative mutation paths call derived sync
  before leaving the existing `_atomic` block. The regression test makes sync delete
  a document and then raise; verdict insertion, status rejection, and document
  deletion all roll back together.
- Helper transaction ownership: `search_index.py` contains no `BEGIN`, `commit`, or
  transaction context. Callers own atomicity.
- Trigger/FTS consistency: insert, delete, and update triggers mirror the existing
  chunk external-content FTS pattern. Tests query FTS after insert and after removal;
  carry testing proves rejected/stale IDs are absent from FTS.
- Status policy: `sync_assertion_document` checks `status == active`; it does not
  require a supported verdict, so pending active assertions remain searchable.
- History preservation: `_CARRY_TABLES` is unchanged and contains exactly
  `assertions`, `evidence_spans`, `verdicts`, and `staleness_log`. The v6 fixture
  compares exact IDs, statuses, timestamps, spans, verdicts, rationales, and staleness
  history before and after rebuild.
- Restore ordering: direct authoritative restore commits first; byte verification and
  decorator-boundary expiry then establish final statuses; only after both passes does
  a caller-owned transaction rebuild derived documents.
- Canonical determinism: evidence spans sort by `(path, byte_start, byte_end,
  id or -1)` and the document hash uses the project's shared SHA-256 helper.
- Import architecture: no module-level cycle remains; the dedicated DAG regression
  passes.

## Concerns

None blocking. Dense assertion embeddings and assertion retrieval are intentionally
outside Task 2; this commit supplies only the v7 derived lexical storage and lifecycle
synchronization required for the following retrieval task.
