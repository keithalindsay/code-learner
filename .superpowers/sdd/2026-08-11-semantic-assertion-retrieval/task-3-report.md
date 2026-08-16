# Task 3 report: bounded lexical assertion retrieval and refill

## Status

Implemented deterministic, bounded lexical assertion retrieval with centralized
serving-policy filtering, ID-scoped batch readers, live citation verification, and
refill after ineligible or expired candidates. Retrieval never calls
`servable_assertions`; only supplied candidate IDs reach the authoritative verifier.

## Files

- Created `codelearner/retrieve/assertions.py`
- Created `tests/test_assertion_retrieve.py`
- Modified `codelearner/assertions/store.py`
- Inspected `codelearner/retrieve/lexical.py`; no edit was necessary because
  `escape_fts_query` was already public and source lexical behavior remained intact.

## RED evidence

### Initial retrieval contract

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py -q
```

Observed: collection failed exactly because the production module did not exist:

```text
ModuleNotFoundError: No module named 'codelearner.retrieve.assertions'
```

The test file covered supported retrieval, verdict/freshness metadata, production and
research policy, deterministic refill, hostile and empty input, duplicate IDs,
metadata-before-filesystem ordering, transient unreadability, terminal evidence
failures, equal-score ordering, invalid bounds, the hard candidate cap, missing search
structures, ordered ID readers, and ID-scoped verification.

### Metadata-before-I/O seam

After the first minimal implementation, the focused command failed one test:
`test_metadata_policy_runs_before_filesystem_verification`. An empty metadata-eligible
set still invoked the filesystem-verification seam with `[]`. The implementation was
tightened so an empty eligible page performs no verification call.

### Mutation-safe refill cursor

Self-review identified a paging hazard: a terminal verification failure removes its
derived FTS document, shrinking an OFFSET-ordered result set. A new regression was
written first.

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py::test_expired_top_row_does_not_shift_refill_past_next_candidate -q
```

Observed: exit 1. Search returned `[]` rather than the eligible second assertion because
the removed top document shifted that row behind the existing offset. Cursor accounting
now subtracts only current-page documents actually removed by the authoritative stale
transition; unreadable documents remain present and do not alter the cursor.

## GREEN evidence

Task 3 focused command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py -q
```

Exit 0: 23 tests passed.

Required compatibility command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py tests/test_assertions.py tests/test_retrieve.py -q
```

Exit 0 at 100% with no failures.

Ruff:

```text
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/assertions.py codelearner/retrieve/lexical.py codelearner/assertions/store.py tests/test_assertion_retrieve.py
```

`All checks passed!`

mypy:

```text
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/assertions.py codelearner/assertions/store.py
```

`Success: no issues found in 2 source files`

`git diff --check`: exit 0.

Single final full-suite run:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest -q
```

Exit 0 at 100% with the existing skip and no failures.

## Self-review

- Bounds: `k`, `page_size`, and `max_candidates` reject zero and negative values.
  Every FTS LIMIT is the smaller of the page size and remaining hard-cap allowance;
  paging stops on `k`, exhaustion, or cap.
- Ranking: `_search_page` binds MATCH, LIMIT, and OFFSET values and orders by
  `bm25(assertions_fts), assertion_id`. Assertion scores are negated to retain the
  package-wide higher-is-better convention.
- Deduplication: page IDs are deduplicated globally before metadata loading and again
  by the store reader before verification/filesystem work. A page-boundary duplicate
  regression proves each assertion is verified at most once.
- Policy ordering: assertions and verdicts are batch-loaded for the bounded page;
  `evaluate_metadata` runs before `_verify_loaded_assertions`. An entirely ineligible
  page does no filesystem verification.
- Store scope: `load_assertions_by_ids` preserves first-occurrence input order, omits
  missing IDs, and splits all variable-length `IN` statements using the existing
  SQLite-safe chunk size. Span loading remains batched under `_load_assertions`.
- Freshness: `verify_assertions` loads only supplied IDs, shares one file cache per
  pass, calls the existing `_first_failure`, and routes terminal findings through
  `mark_stale`. Therefore status, staleness logging, derived-document removal, caller
  transaction joining, and rollback behavior remain authoritative.
- Transient failure: `REASON_UNREADABLE` withholds without status, log, or document
  mutation. The test uses a forced unreadable result, so it is deterministic even
  under privileged test users.
- Refill under mutation: terminal verification document removal is measured through a
  bounded ID query and compensated in the next OFFSET. This prevents skipped lower
  candidates while preserving the hard count of candidates already examined.
- Error behavior: the strengthened Task 2 structure probe runs before empty-query
  handling, so a missing or incompatible assertion search index raises
  `AssertionSearchUnavailable` rather than masquerading as an empty result.
- Import DAG: store owns authoritative loading/verification and imports no retrieval
  types. Retrieval converts store rows to candidate and verdict types. The full-suite
  DAG regression remains green.
- Compatibility: source lexical escaping was already exposed as
  `escape_fts_query`; Task 3 reuses it without changing `Hit`, `search_lexical`, or
  existing source ranking.

## Concerns

None blocking. This task deliberately implements lexical assertion retrieval only;
mixed source/assertion fusion, promotion, conflict labeling, and package-level
orchestration remain Task 4 scope.

## Review fix round 1: immutable bounded ranking snapshot

### Finding verified

The mutable OFFSET refill was not sound when verification expired a candidate. FTS5
BM25 is corpus-relative: removing a stale assertion document can change both scores and
the order of every surviving row. Subtracting the number of removed rows repaired only
a positional shrink; it could not repair a reorder across the cursor.

The reviewer's fourteen-filler standalone construction initially passed when copied
into this test fixture because the stored canonical documents also contain kind,
subject, and evidence fields. A diagnostic sweep verified the same mechanism and found
that nineteen fillers is the smallest locally deterministic value that produces the
load-bearing order change with those complete canonical documents: initial IDs
`3,2,1`, followed by survivor order `1,2` after ID 3 expires. The public API, query,
policy states, page size, candidate cap, and expected result are otherwise unchanged.

### RED evidence

Command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py::test_bm25_reordering_after_expiry_does_not_skip_snapshot_candidate -q
```

Observed: exit 1. `search_assertions(..., "a b", k=1, page_size=2,
max_candidates=10)` returned `[]` instead of supported assertion ID 1. The first page
contained stale supported ID 3 and pending ID 2. Verification expired ID 3 and removed
its FTS document; the changed BM25 corpus reordered survivors to `1,2`, so the
compensated OFFSET skipped ID 1.

### Fix

`search_assertions` now performs one deterministic FTS query ordered by
`bm25(assertions_fts), assertion_id` with `LIMIT max_candidates` and `OFFSET 0` before
any metadata or filesystem work can mutate the index. This captures only bounded IDs
and scores. Retrieval then processes that immutable sequence in `page_size` slices,
batch-loading page metadata, applying serving policy before filesystem verification,
deduplicating IDs before I/O, and stopping once `k` verified candidates are found or
the snapshot is exhausted. The mutable OFFSET loop, document-presence query, and
positional compensation were removed.

The hard cap is unchanged: the snapshot query cannot return more than
`max_candidates`. Metadata and citation verification remain page-bounded rather than
loading or verifying the entire snapshot at once.

### GREEN evidence

Regression command: exit 0.

Task 3 focused command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py -q
```

Exit 0: 24 tests passed.

Compatibility and import-DAG command:

```text
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py tests/test_assertions.py tests/test_retrieve.py tests/test_generate_purpose.py::test_the_module_import_graph_is_a_dag -q
```

Exit 0 at 100%: 130 tests passed.

Static verification:

```text
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/assertions.py codelearner/retrieve/lexical.py codelearner/assertions/store.py tests/test_assertion_retrieve.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/assertions.py codelearner/assertions/store.py
git diff --check
```

Observed respectively: `All checks passed!`,
`Success: no issues found in 2 source files`, and exit 0.

### Fix-round concerns

None blocking. The snapshot deliberately holds only bounded assertion IDs and raw BM25
scores; authoritative assertion rows, verdicts, spans, and filesystem reads remain
page-scoped. This preserves refill determinism without converting the hard cap into an
unbounded or eager verification scan.
