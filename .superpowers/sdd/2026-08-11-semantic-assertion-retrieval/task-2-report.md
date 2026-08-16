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

## Review fix round 1

### Findings verified

The FTS divergence report reproduced against the committed implementation. With one
active authoritative assertion and its `assertion_documents` row intact,

```sql
INSERT INTO assertions_fts(assertions_fts) VALUES ('delete-all');
```

left zero searchable tokens. Calling `rebuild_assertion_documents()` then raised
`sqlite3.DatabaseError: database disk image is malformed` because the external-content
delete trigger attempted to delete a token entry that no longer existed.

The structure probe also returned `True` based only on object names when
`assertions_fts` was replaced by a view or an ordinary table, and when any one of the
three required FTS synchronization triggers was missing.

### RED evidence

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py -q
```

Observed: exit 1 with six expected failures. The divergence regression raised
`DatabaseError: database disk image is malformed`; the FTS view and ordinary-table
impostors returned `True`; and deletion of each of `assertions_fts_insert`,
`assertions_fts_delete`, and `assertions_fts_update` still returned `True`.

An additional strict-TDD detection pass reverted the production probe before adding
the explicit `assertion_documents`-must-be-a-table regression. The same focused
command exited 1 with six expected detection failures: FTS view,
`assertion_documents` view, ordinary non-FTS table, and each missing trigger.
The `assertion_documents` view fixture was then isolated by recreating all three
required names as real `INSTEAD OF` triggers on the view; with only the table-type
check removed, its single test exited 1 because the probe returned `True`.

### Repair

`rebuild_assertion_documents()` now issues the FTS5 `rebuild` command before deleting
documents. That reconstructs a consistent token index from the retained external
content so delete triggers are safe. It then regenerates canonical documents only
from authoritative assertion/evidence rows and issues a final FTS5 `rebuild` from
those fresh documents. These are ordinary connection statements: the helper still
does not begin or commit a transaction.

`assertion_search_structures_present()` now reads `name`, `type`, and `sql` from
`sqlite_master`. It requires `assertion_documents` to be a table,
`assertions_fts` to be a table whose DDL is `CREATE VIRTUAL TABLE ... USING fts5`,
and all three named synchronization objects to have type `trigger`.

### GREEN and fix verification

Focused search-index command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py -q
```

Exit 0: 15 tests passed.

Requested regression command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_assertions.py::test_sync_failure_rolls_back_verdict_status_and_document_together tests/test_cli.py::test_v6_carry_preserves_history_and_rebuilds_only_active_search_documents tests/test_db.py::test_transaction_rolls_back_on_error tests/test_generate_purpose.py::test_the_module_import_graph_is_a_dag -q
```

Exit 0: 19 tests passed. This covers FTS recovery and structure detection together
with authoritative/derived rollback, v6 carry and post-verification rebuild, DB
transaction rollback, and the module import DAG.

Static verification:

```text
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/assertions/search_index.py tests/test_assertion_search_index.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/assertions/search_index.py
git diff --check
```

Observed respectively: `All checks passed!`, `Success: no issues found in 1 source
file`, and exit 0. A source scan found no `BEGIN`, `commit`, or transaction helper in
`codelearner/assertions/search_index.py`.

### Fix-round concerns

None blocking. Recovery uses SQLite FTS5's supported external-content `rebuild`
operation; it does not attempt to recover arbitrary corruption in authoritative
SQLite tables, which is outside this derived-index repair contract.
