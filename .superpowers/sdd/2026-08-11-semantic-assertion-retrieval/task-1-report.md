# Task 1 Report: Typed Retrieval Candidates and Serving Policy

## Implementation

- Added `codelearner.retrieve.types` with frozen source/assertion candidate types,
  typed keys, immutable modality result tuples, verdict/freshness metadata, and
  score-contribution records.
- Added `codelearner.assertions.policy` with immutable production/research policy
  values and pure, status-first metadata evaluation.
- Re-exported the public contracts from `codelearner.retrieve` and
  `codelearner.assertions` without changing the existing `Hit` or `SearchResult`
  APIs.

## RED evidence

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_retrieval_types.py -q
E   ModuleNotFoundError: No module named 'codelearner.retrieve.types'
```

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_serving_policy.py -q
E   ModuleNotFoundError: No module named 'codelearner.assertions.policy'
```

## GREEN evidence

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_retrieval_types.py tests/test_serving_policy.py tests/test_retrieve.py -q
..................................................                       [100%]
```

```text
$ /home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/types.py codelearner/assertions/policy.py tests/test_retrieval_types.py tests/test_serving_policy.py
All checks passed!

$ /home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/types.py codelearner/assertions/policy.py
Success: no issues found in 2 source files

$ git diff --check
```

## Files changed

- `codelearner/retrieve/types.py`
- `codelearner/retrieve/__init__.py`
- `codelearner/assertions/policy.py`
- `codelearner/assertions/__init__.py`
- `tests/test_retrieval_types.py`
- `tests/test_serving_policy.py`

## Self-review

- Source candidates copy all existing `Hit` fields and calculate tier using the
  existing `tier_of` helper.
- Assertion and source keys are type-qualified, so colliding numeric IDs stay
  distinct in future mixed fusion.
- Policy performs no filesystem or SQL work and orders checks as required:
  status, veto, support, then pending.
- Runtime imports avoid a `retrieve.types` / `assertions.policy` cycle; the
  evidence-span import is type-checking-only.
- Existing source-only `Hit`, `SearchResult`, and `search()` definitions are
  unchanged.

## Concern

The requested full `pytest -q` pre-commit run was started, but this execution
environment detached it after partial output through 39% and did not return a terminal
exit status. A second retained run also exited without a recoverable result. I have not
claimed a full-suite pass or committed the task; focused tests, Ruff, mypy, and diff
checking are green. A normal full-suite run with a recoverable exit status remains the
only outstanding gate.

## Completion follow-up (August 12, 2026)

### RED / root cause

- `codelearner.assertions.policy` originally depended on
  `codelearner.retrieve.types.VerdictSummary` only for verdict metadata typing,
  creating the `assertions -> retrieve -> assertions` cycle once retrieval also
  imported assertion-facing contracts.
- `codelearner.tier` originally imported `codelearner.retrieve.lexical.Hit`
  only for type annotations, creating the `retrieve -> tier -> retrieve` cycle
  because `retrieve.types` uses `tier_of(...)`.

### Fix

- `codelearner.assertions.policy` now defines a local structural `_Verdict`
  protocol and accepts real `VerdictSummary` values structurally, with no
  runtime or type-checking import from `codelearner.retrieve`.
- `codelearner.tier` is now a leaf: it imports only tier constants and types
  its public helpers against local `_TieredHit` / `_RenderableHit` protocols,
  which keeps existing `Hit` callers compatible without importing retrieval.

### GREEN evidence

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_generate_purpose.py::test_the_package_import_graph_is_a_dag -q
.                                                                        [100%]
```

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_retrieval_types.py tests/test_serving_policy.py tests/test_retrieve.py -q
..................................................                       [100%]
```

```text
$ /home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/types.py codelearner/assertions/policy.py codelearner/tier.py tests/test_retrieval_types.py tests/test_serving_policy.py
All checks passed!

$ /home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/types.py codelearner/assertions/policy.py codelearner/tier.py
Success: no issues found in 3 source files

$ git diff --check
```

```text
$ /home/keith/projects/code-learner/.venv/bin/python -m pytest -q | tee .superpowers/sdd/2026-08-11-semantic-assertion-retrieval/task-1-full-suite.log
........................................................................ [  6%]
........................................................................ [ 13%]
.............................................................s.......... [ 19%]
........................................................................ [ 26%]
........................................................................ [ 33%]
........................................................................ [ 39%]
....................................................................... [ 46%]
........................................................................ [ 52%]
........................................................... [ 59%]
........................................................................ [ 66%]
........................................................................ [ 72%]
........................................................................ [ 79%]
........................................................................ [ 86%]
........................................................................ [ 92%]
................................................................ [ 99%]
........                                                                 [100%]
```

### Self-review follow-up

- The DAG fix is surgical: only the dependency sources changed, not the policy
  decisions or source-hit behavior.
- `tier_of`, `facts_only`, and `hit_json` still accept the existing lexical
  `Hit` shape through structural typing, so current callers stay source-compatible.
- The recoverable full-suite output is copied into this report, so the evidence
  stays with the task artifact without adding a transient log file to the commit.
