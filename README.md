# code-learner

GraphRAG over a codebase — for agents that need to trace it, and for engineers
onboarding into it.

Point it at a repository. It parses the code into a knowledge graph of modules,
classes and functions, resolves the references between them, and (from Phase 4)
layers on *purpose* — what each piece is for — where every inferred claim cites the
source spans it came from and expires when those spans change.

**Status: early.** Ingest, storage, tier-1 name resolution, symbol-boundary
chunking, and lexical (BM25) retrieval work and are tested. Embeddings, hybrid
fusion, graph expansion, reranking, the inference layer, and the eval harness are
not built yet.

---

## Why this exists

Indexing a codebase for agents is a crowded category —
[CodeGraph](https://github.com/colbymchenry/codegraph) (63.2k stars) and
[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) both do the
structural half well, and this project does not claim to beat them at it.

What none of them do is the *purpose* layer. CodeGraph's docs are explicit: **"no
inference — structural facts only."** That is a defensible refusal, because
LLM-generated purpose is unverifiable and goes stale silently, and a
confidently-wrong rationale injected into an agent's plan is worse than no
rationale at all.

The bet here is that inference isn't unusable, it's *unaccountable* — and that four
properties make it shippable:

1. **Evidence-bound** — every claim cites concrete `file:line` spans. No citation,
   no entry.
2. **Hash-bound** — the cited spans' hashes are stored. Any of them changing marks
   the claim stale rather than serving it.
3. **Adjudicated** — independent judges try to *refute* each claim using only its
   cited evidence. Refused claims are logged, not deleted.
4. **Tier-labeled** — callers can demand facts only, and trust nothing inferred.

## The tier model

| Tier | Meaning | Guarantee |
|---|---|---|
| **T0 FACT** | parsed from source | deterministic, reproducible from source alone |
| **T1 RESOLVED** | a name bound to a symbol | may be ambiguous; carries confidence + resolver identity |
| **T2 INFERRED** | LLM-asserted | must satisfy all four properties above |

The split is in the schema, not bolted on. Seeing the call site `foo()` is a T0
fact; deciding *which* `foo` it means is a fallible T1 step. Both live in one row —
`dst_name` always populated, `dst_symbol_id` only once something resolved it — so an
unresolved reference is represented honestly rather than dropped or guessed.

## Measured, on real repositories

| repo | files | symbols | edges | resolved (in-repo) | time |
|---|---|---|---|---|---|
| [swarm-sync](https://github.com/keithalindsay/swarm-sync) | 68 | 1,095 | 6,531 | 63.5% | 0.38s |
| code-learner (itself) | 9 | 79 | 454 | 75.0% | 0.03s |

"In-repo" is the honest denominator: **roughly half of all calls in real code target
stdlib or third-party code** and are correctly unresolvable. Counting those as
failures makes a working resolver look broken.

### One number that went down on purpose

In-repo resolution briefly reached **76.2%**, then was deliberately reduced to
**63.5%**.

The 76.2% included a strategy that bound dotted attribute calls by unique name.
That pointed 38 `r.json()` calls on an HTTP response at a nested helper inside a
test file, and made that helper the highest-ranked symbol in the entire call graph.
472 of 519 such bindings were attribute calls. The receiver's type is unknown, so
`x.foo()` carries no evidence about which `foo` is meant.

Removing the strategy cost 12.7 points of coverage and removed a set of confident
fabrications. That trade is the point of the project, and it showed up in its own
resolver on day one. See [docs/PHASE0-FINDINGS.md](docs/PHASE0-FINDINGS.md).

## Repository isolation

One SQLite file per repo (`.codelearner/index.db`). Cross-contamination is
**structurally impossible** — there is no shared store — and additionally enforced:
an index is pinned to one repo root and refuses a second.

Source files come from `git ls-files` where available. That is the correctness path,
not a convenience: the first spike indexed `swarm-sync/.claude/worktrees/`, five
near-complete copies of the repo, and produced cross-copy edges binding a call in
one copy to a definition in another. A repo's own `.gitignore` already knows what is
real source.

## How this was built

Written with Claude Code, by an engineer who specifies, audits, and sets the bar the
work has to clear. The standing rule: *a fix without a test that fails when you
delete the fix is not a fix.* The three regression fixes in Phase 0 were each
mutation-verified — delete the fix, confirm the test fails, restore, confirm green.

## Quickstart

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

```python
from pathlib import Path
from codelearner.ingest import index_repo

conn, stats = index_repo(Path("/path/to/repo"))
print(stats.files, stats.symbols, stats.edges)
print(f"{stats.resolve.rate_of_internal:.1%} of in-repo references resolved")
```

## Roadmap

| Phase | | Status |
|---|---|---|
| 0 | Spike on a real repo | done |
| 1 | Ingest + store + tier-1 resolution | done |
| 2 | Symbol-boundary chunking + FTS5 lexical index | done |
| 2b | Embeddings (`C2LLM-0.5B`) into sqlite-vec | next |
| 3 | Hybrid retrieval: RRF fusion, graph expansion, rerank | |
| 4 | Assertion pipeline + adversarial gate | |
| 5 | Staleness engine | |
| 6 | Onboarding tours | |
| 7 | MCP server + CLI | |
| 8 | Eval: per-modality ablation, faithfulness, gate controls | |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 47 tests
.venv/bin/ruff check .
.venv/bin/mypy codelearner --ignore-missing-imports
```
