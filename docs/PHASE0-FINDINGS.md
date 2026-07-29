# Phase 0 findings

Spike goal: confirm the substrate works before building retrieval on top of it.
Everything below was measured, not estimated. Target repo: `swarm-sync` (68 files,
1,095 symbols).

---

## 1. sqlite-vec works on SQLite 3.37.2, with one constraint

The host ships SQLite 3.37.2 (2021). This was the one flagged unknown going in,
because the vector store is load-bearing for Phase 2.

**It loads and does real KNN.** But the standard query form fails:

```sql
-- FAILS on 3.37.2: "A LIMIT or 'k = ?' constraint is required on vec0 knn queries"
SELECT rowid, distance FROM v WHERE embedding MATCH ? ORDER BY distance LIMIT 3;

-- WORKS
SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k = ? ORDER BY distance;
```

Old SQLite does not push `LIMIT` down into a virtual table's query planner. Phase 2
must use the `k = ?` form. No change to the storage choice is needed.

---

## 2. Uniqueness is not a resolution signal in real code

The first resolver bound a name only when it was unique repo-wide. Result:
**98 of 34,013 call edges (0.3%)**.

The diagnosis mattered more than the number. Splitting every call by whether its
target name exists in the repo at all:

| | count | share |
|---|---|---|
| target not in repo (stdlib / third-party) | 16,794 | 49.4% |
| name exists but is **ambiguous** | 17,121 | 50.3% |
| name is unique repo-wide | 98 | 0.3% |

`execute` has 19 definitions in swarm-sync. `get` has 29. `post` has 29. Roughly
half of all calls are *correctly* unresolvable because they target code that isn't
in the repo — counting those as failures makes a working resolver look broken.

**Consequence for the project:** import- and scope-aware resolution isn't a
refinement, it's the entire mechanism. It also means the honest denominator is
edges whose target is actually in the repo, which is now reported separately as
`rate_of_internal`.

---

## 3. The index was reading five copies of the repo

The spike reported 430 files for a 68-file repository.

`swarm-sync/.claude/worktrees/` holds agent scratch worktrees — near-complete
copies of the codebase. They were being indexed as first-class source, and worse,
producing **cross-copy edges**: a call site in one worktree binding to a definition
in the main tree.

Fix is better than a blocklist: **ask git**. `git ls-files` already encodes exactly
which files are real source, because `.gitignore` is maintained by the people who
know. A hand-kept skip list can only exclude directories somebody thought of.
Filesystem walk with `SKIP_DIRS` remains as the non-git fallback.

Follow-on found by self-indexing: a repo with `git init` and no commit yet returns
an empty `ls-files`, which silently indexed zero files and reported success. Empty
result now falls back to the walk.

---

## 4. The metric got worse and the system got better

After adding import-alias, module-local, `self.`-attribute, and class-attribute
resolution, in-repo resolution reached **76.2%**.

Inspecting the resulting call graph showed the top-ranked symbol was
`tests.test_agent.…_Recorder.post._R.json` with 137 inbound calls — a nested test
helper outranking every piece of production code.

Cause: the unique-basename fallback was being applied to dotted attribute access.
`r.json()` on an httpx response bound to the only symbol in the repo named `json`,
which happened to live inside a test file. **472 of 519** such bindings were
attribute calls.

The receiver's type is unknown, so `x.foo()` carries no evidence about which `foo`
is meant. Uniqueness of the name is a fact about the repo, not about the call.
Restricting the strategy to bare names dropped in-repo resolution from **76.2% to
63.5%** — and that is the improvement, because the 12.7 points removed were
confident fabrications that then dominated the graph.

A metric moving the wrong way while the system improves is the whole thesis of this
project, found in its own tier-1 resolver on day one. Worth stating plainly in the
README rather than quietly shipping the bigger number.

Second bug found in the same pass: `import events as events_mod` was storing only
the target's last segment (`events`), discarding the alias. Every
`events_mod.tail()` call therefore failed to resolve — the alias *is* the key the
call site uses. Both fixes are pinned by mutation-verified regression tests.

---

## Current measured state

| repo | files | symbols | edges | resolved (all) | resolved (in-repo) | time |
|---|---|---|---|---|---|---|
| swarm-sync | 68 | 1,095 | 6,531 | 31.4% | **63.5%** | 0.38s |
| code-learner (self) | 9 | 79 | 454 | 25.8% | **75.0%** | 0.03s |

Resolution strategies on swarm-sync, by volume:

| strategy | bindings | confidence |
|---|---|---|
| `import_alias/v1` | 1,069 | 0.85 |
| `module_local/v1` | 700 | 0.90 |
| `exact_qualname/v1` | 229 | 1.00 |
| `unique_basename/v1` | 47 | 0.75 |
| `self_attr/v1` | 3 | 0.95 |

`self_attr` firing only 3 times is a property of the target repo, not a gap:
swarm-sync has 5,344 plain functions against 445 methods, and only 324 `self.` call
sites in tracked source. It is function-style code.

Self-indexing correctly identifies `index_repo`, `db.init_db`, and
`python_extract.extract` as its own most-called symbols.

---

## Verification

- 33 tests, ruff clean, mypy clean across 7 modules, 3 consecutive runs no flakes.
- The three regression fixes were each mutation-verified: delete the fix, confirm
  the test fails, restore, confirm green.

## Carried into Phase 1+

- Use the `k = ?` form for every vec0 KNN query.
- Report `rate_of_internal` alongside raw resolution; the raw number is misleading
  on its own.
- Unresolved edges stay in the graph as tier-0. Roughly half of all calls in real
  code are external, and an unresolved call is still a true statement.
- Any future resolver strategy needs a measured precision check before it ships,
  not just a coverage number. Coverage without precision was the exact trap here.
