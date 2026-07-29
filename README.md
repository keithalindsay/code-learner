# code-learner

GraphRAG over a codebase — for agents that need to trace it, and for engineers
onboarding into it.

Point it at a repository. It parses the code into a knowledge graph of modules,
classes and functions, resolves the references between them, and (from Phase 4)
layers on *purpose* — what each piece is for — where every inferred claim cites the
source spans it came from and expires when those spans change.

**Status: early.** Ingest, storage, tier-1 name resolution, symbol-boundary
chunking, and the full hybrid retrieval pipeline -- lexical, dense, graph expansion,
RRF fusion -- work, are tested, and are measured against a gold set. Cross-encoder
reranking, the inference layer, and the onboarding surface are not built yet.

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

## Retrieval: what the ablation actually showed

Three modalities — lexical (BM25 over FTS5), dense (vector), and graph expansion —
fused with Reciprocal Rank Fusion. Measured against a 16-query hand-labelled gold
set on swarm-sync:

| configuration | recall@5 | recall@10 | hit@5 | MRR |
|---|---|---|---|---|
| lexical only | 0.427 | 0.448 | 0.500 | 0.304 |
| dense only | 0.542 | 0.635 | 0.562 | 0.407 |
| graph only (dense-seeded) | 0.188 | 0.188 | 0.250 | 0.172 |
| lexical + dense | 0.573 | 0.635 | 0.625 | 0.331 |
| lexical + dense + prefer-impl | 0.604 | 0.781 | 0.688 | **0.516** |
| **hybrid, all three (default)** | **0.646** | **0.802** | **0.750** | 0.463 |

Three findings worth more than the final number.

**The graph modality made things worse before it made them better.** At its first
guessed weight of 0.6, the full hybrid scored **0.385** recall@5 — well below
lexical+dense alone at 0.573. The weight sweep is monotonic: `0.3 → 0.646`,
`0.6 → 0.615`, `1.0 → 0.552`, `1.5 → 0.354`. Graph expansion has no query
representation, so every vote it casts is evidence about the *code* rather than
about the *question*; past a low weight those votes displace better-matched answers.
The default is 0.3 because that is what measured best, and a test pins the constant
so it cannot drift without re-running the ablation.

**Recall and ranking pull in opposite directions.** Adding graph raises recall@5
(0.604 → 0.646) and hit@5 (0.688 → 0.750) while *lowering* MRR (0.516 → 0.463). It
finds code the text modalities missed, then dilutes the top of the ranking. That
trade is real and is documented rather than tuned away — reranking (not yet built)
is the honest fix.

**The biggest single lever was not a modality at all.** Demoting test code moves
recall@10 from 0.635 to 0.781 — more than adding an entire retrieval modality. Both
text modalities systematically rank tests above the implementations they exercise,
because a test states the behaviour in prose, names it in the function title, and
repeats the vocabulary; the implementation just does it.

*Caveat that limits all of the above:* the gold set is 16 queries, hand-labelled by
the author, and every one is of the form "how does X work" — the exact shape the
test demotion helps. Differences of one or two points are noise. This is enough to
tell a modality that works from one that does not, and not enough to justify
fine-grained tuning.

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

That rule caught a bad test of its own in Phase 3. A test asserting that graph
activation *accumulates* across seeds passed even when `+=` was replaced with
`max()` — it was measuring seed rank, not accumulation. It was rewritten to control
seed order so that only summing can produce the asserted outcome, then re-checked
against the same mutation. A test that survives deleting the behaviour it names is
not a test, whoever wrote it.

## Quickstart

Requires **Python 3.12+**.

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q

# Dense retrieval (optional; downloads ~1.2GB of model weights on first run)
.venv/bin/pip install -e ".[embed]"
```

> **Check `python3 -V` reports a final release, not a release candidate.** This was
> developed on a box whose `/usr/bin/python3.11` is `3.11.0rc1`, which lacks
> `sys.get_int_max_str_digits` — added in 3.11.0 final. Modern torch imports it
> unconditionally, so the whole ML stack fails with a `ModuleNotFoundError` about
> `PreTrainedModel` that says nothing about the real cause. An hour lost to that is
> why the floor is 3.12.

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
| 2b | Dense embeddings (`Qwen3-Embedding-0.6B`) into sqlite-vec | done |
| 3 | Hybrid retrieval: RRF fusion + graph expansion | done |
| 3b | Cross-encoder reranking | next |
| 4 | Assertion pipeline + adversarial gate | |
| 5 | Staleness engine | |
| 6 | Onboarding tours | |
| 7 | MCP server + CLI | |
| 8 | Eval: per-modality ablation (done), faithfulness, gate controls | partial |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 84 tests
.venv/bin/ruff check .
.venv/bin/mypy codelearner --ignore-missing-imports
```

### Reproducing the ablation

```python
from pathlib import Path
from codelearner import db
from codelearner.index import SentenceTransformerEmbedder
from codelearner.eval import run_ablation, format_table

conn = db.connect(Path("/path/to/.codelearner/index.db"))
print(format_table(run_ablation(conn, SentenceTransformerEmbedder())))
```
