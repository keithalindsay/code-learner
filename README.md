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

### The T2 assertion store

The storage layer for tier 2 is built (`codelearner/assertions/`, schema v4). It is
the gate, not the pipeline — nothing in it calls a model. What it does is refuse the
two ways an inferred claim becomes unaccountable:

- **No citation, no entry.** `write_assertion` raises before it opens a transaction,
  so a claim with zero `evidence_spans` leaves no row behind. An uncited claim can't
  be adjudicated, can't expire, and can't be checked by a reader — it is
  indistinguishable from a good one at every stage after the door.
- **Servable means re-verified, not merely stored.** `servable_assertions` re-reads
  the cited bytes off disk and re-hashes them on *every* call; `status = 'active'`
  alone is never enough. Verification at serve time rather than in a background
  sweep is the point: an hourly sweep has an hour-wide window in which the index
  answers questions using code that no longer exists.
- **Nothing is deleted.** A refuted claim becomes `rejected` and keeps its spans and
  its verdict; an expired one becomes `stale` with a `staleness_log` row naming the
  citation that moved. The rejected set is the only evidence the gate does anything
  — a pipeline that deletes what it rejected can report any pass rate it likes.

Spans are hashed, not files. An edit elsewhere in a 2,000-line module leaves a claim
about one function in it alone; staleness that fires on everything is staleness
nobody reads. Two schema decisions carry the rest: `subject_symbol_id` is `ON DELETE
SET NULL` beside a `NOT NULL` qualname (a `CASCADE` would mean a routine re-index
silently empties the store), and `evidence_spans.path` is plain text rather than a
reference to `files(id)` — because an assertion that loses its last span doesn't
become unsupported, it becomes *vacuously* supported. "Every cited span still
matches" is trivially true of no spans, and reads as success everywhere it isn't
specifically looked for. The reader checks for an empty evidence set anyway.

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

## Onboarding tours

Retrieval ranks. Onboarding **orders**. `codelearner.onboard` cuts the same call
graph into a reading path — *"read these ten things, in this order, to understand
worktree handling"* — and every position is decided by deterministic graph work,
with no model involved. Re-running it against an unchanged repo produces
byte-identical output, which is what makes a tour a curriculum rather than a
suggestion.

Three signals, applied in this order:

1. **Dependency depth.** Leaves first, so a reader never meets a call before its
   definition. Depth is the *longest* path to a leaf, not the shortest — with the
   shortest, `a → b → c` plus a shortcut `a → c` ties `a` with the `b` it calls.
2. **Centrality.** PageRank over the resolved call graph orders symbols *within* a
   depth tier, so the load-bearing one leads.
3. **Module clustering.** A file's stops run consecutively. A correctly-ordered
   tour that changes file on every stop is still unreadable, because each jump
   costs the reader the context they just built.

Cycles are condensed (Tarjan, iterative), not assumed away. That matters because
the obvious alternative fails *silently*: a Kahn-style topological sort returns
promptly and **drops every node that is in a cycle**, leaving a tour that looks
complete while missing exactly the code that was hardest to understand. Here a
cycle occupies one tier, its members are listed consecutively, and the output says
it is a cycle — "these three call each other, read them as a unit" is real
information about the code.

```python
from codelearner import db
from codelearner.onboard import build_reading_path, render_markdown

conn = db.connect("/path/to/.codelearner/index.db")
print(render_markdown(build_reading_path(conn, topic="worktree creation and cleanup")))
```

Each stop shows the symbol, its `file:line`, its signature and docstring summary,
and why it sits where it does — *"Leaf: it calls nothing else on this path, and 3
later stops here call it — read before its callers. PageRank 0.0291 (#1 of 10 on
this path); 8 resolved callers repo-wide."* Every clause is a countable fact about
the graph, so a reader who disagrees can check it.

Two real generated tours of swarm-sync, repo-wide and topic-scoped, are checked in
verbatim at [docs/EXAMPLE-TOUR.md](docs/EXAMPLE-TOUR.md) — including a note on
where the ordering runs out of information and the tie-break stops being defensible.

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

## CLI

`pip install -e .` puts `codelearner` on the path. From a checkout that has not been
installed, `python -m codelearner.cli` is the same entry point.

Every block below is real output — from indexing
[swarm-sync](https://github.com/keithalindsay/swarm-sync), except the `--json`
example, which is code-learner's index of itself. Only the absolute paths are
generalised.

### `codelearner index`

```console
$ codelearner index /path/to/swarm-sync
indexed /path/to/swarm-sync
  index      /path/to/swarm-sync/.codelearner/index.db
  files             68
  symbols        1,095
  edges          6,531
  chunks         1,087
  resolved       2,048  63.5% of 3,226 in-repo references (3,305 target code outside this repo)
```

The index goes to `<repo>/.codelearner/index.db` unless `--index-path` says
otherwise — one file per repo, which is what makes cross-repo contamination
structurally impossible rather than merely discouraged.

Two counts, because only one of them is honest: 2,048 of 6,531 edges resolved is
31%, but 3,305 of those edges target stdlib or third-party code and are *correctly*
unresolvable. 63.5% is the rate against references that could have resolved.

Re-indexing an existing index **refuses** rather than rebuilding, because there is
no incremental update yet and a rebuild throws away embeddings that cost minutes:

```console
$ codelearner index /path/to/swarm-sync
codelearner: an index already exists at /path/to/swarm-sync/.codelearner/index.db. There is
no incremental update yet, so re-indexing means rebuilding from scratch. Re-run with --force
to delete and rebuild it -- note that this discards any embeddings, which are the expensive
part -- or use --index-path to build a second index elsewhere.
$ echo $?
1
```

Dense vectors are opt-in (`--embed`, with `--model` to choose one), because building
them needs the `[embed]` extra, ~1.2GB of weights, and a GPU to be quick.

### `codelearner search`

```console
$ codelearner search "how does a lease expire and get reclaimed" --repo /path/to/swarm-sync -k 5
codelearner: dense retrieval unavailable: this index has no embeddings. Build them with
`codelearner index <repo> --embed --force`.
5 result(s) for 'how does a lease expire and get reclaimed'  [lexical+graph, k=5]
  1  T0  swarmsync.hooks.adapter.cmd_precheck
        function  swarmsync/hooks/adapter.py:396-454  score 0.0198  [graph+lexical]
        via called by swarmsync.hooks.adapter._keepalive
  2  T0  swarmsync.hooks.adapter._deny_response
        function  swarmsync/hooks/adapter.py:341-377  score 0.0164  [lexical]
  3  T0  swarmsync.hooks.adapter._keepalive
        function  swarmsync/hooks/adapter.py:388-393  score 0.0161  [lexical]
  4  T0  swarmsync.config.max_leases_per_agent
        function  swarmsync/config.py:228-243  score 0.0156  [lexical]
  5  T0  swarmsync.blackboard.models.HealthOut
        class  swarmsync/blackboard/models.py:125-136  score 0.0154  [lexical]
```

Three things that line-up is trying to make impossible to miss:

- **The tier is on every row.** `T0` is a parsed fact — the text matched, nothing had
  to be decided. `T1` means graph expansion got there, and expansion only walks
  *resolved* edges, so reaching it depended on a name binding that can be wrong.
  A hit found by both is `T0`: the text modality reached it without the binding.
- **`via` is the graph's account of itself.** Retrieval that cannot say why it
  returned something is hard to debug and harder to trust.
- **The missing modality is announced, not hidden.** This index has no embeddings, so
  dense is unavailable and says so — on stderr, so `--json` on stdout stays a single
  parseable document.

| flag | effect |
|---|---|
| `-k N` | results to return (default 10) |
| `--facts-only` | T0/T1 only — parsed facts and resolved names, nothing inferred |
| `--no-lexical`, `--no-dense`, `--no-graph` | turn a modality off; the switches the ablation needs |
| `--json` | machine-readable output on stdout, notes on stderr |
| `--repo`, `--index-path` | which index to use (default: `$PWD/.codelearner/index.db`) |

Against code-learner's own index (`codelearner index .` first), with the JSON
reflowed to fit:

```console
$ codelearner search "spreading activation" --repo . --json --k 2
{
  "query": "spreading activation",
  "index": "/path/to/code-learner/.codelearner/index.db",
  "k": 2,
  "facts_only": false,
  "modalities": { "lexical": true, "dense": false, "graph": true },
  "count": 2,
  "hits": [
    {
      "rank": 1, "tier": "T0", "tier_n": 0, "symbol_id": 103,
      "qualname": "codelearner.retrieve.graph._hydrate", "kind": "function",
      "path": "codelearner/retrieve/graph.py", "line_start": 149, "line_end": 187,
      "score": 0.020635, "modality": "graph+lexical", "is_test": false,
      "via": "calls codelearner.retrieve.graph.expand"
    },
    {
      "rank": 2, "tier": "T0", "tier_n": 0, "symbol_id": 99,
      "qualname": "codelearner.retrieve.graph", "kind": "module",
      "path": "codelearner/retrieve/graph.py", "line_start": 1, "line_end": 188,
      "score": 0.016393, "modality": "lexical", "is_test": false,
      "via": ""
    }
  ]
}
```

The first hit is the mixed case worth reading twice: lexical *and* graph both
reached it, so the tier is `T0` — the text match did not depend on the resolution —
while `via` still records the structural route that agreed.

`via` is always present and empty when the hit did not come from the graph. A
consumer should not have to probe for a key to find out whether an explanation
exists.

**`--no-lexical` with an index that has no embeddings is an error, not an empty
result.** Graph expansion has no query representation of its own; it is *seeded* by
the text modalities, so with both of them gone it would return nothing for every
query, silently, forever:

```console
$ codelearner search "anything" --no-lexical
codelearner: no text modality is available, so there is nothing to search with. Graph
expansion has no query representation of its own; it is seeded by lexical and dense
results and cannot run alone. Drop --no-lexical, or build embeddings with
`codelearner index <repo> --embed --force`.
```

### `codelearner stats`

```console
$ codelearner stats --repo /path/to/swarm-sync
index      /path/to/swarm-sync/.codelearner/index.db
repo       /path/to/swarm-sync
schema     v4

counts
  files             68
  symbols        1,095
  edges          6,531
  chunks         1,087

edges by tier
  T0 FACT          4,483  call site as written, unbound
  T1 RESOLVED      2,048  bound to a symbol, with confidence
  T2 INFERRED          0  the inference layer is not built yet

symbol kinds
  function           893
  method              79
  module              68
  class               55

resolution
  2,048 of 6,531 edges resolved -- 63.5% of 3,226 in-repo references (3,305 external, 1,178 ambiguous)
  import_alias/v1              1,069  confidence 0.85
  module_local/v1                700  confidence 0.90
  exact_qualname/v1              229  confidence 1.00
  unique_basename/v1              47  confidence 0.75
  self_attr/v1                     3  confidence 0.95

embeddings
  none. Dense retrieval is unavailable on this index; build vectors with
  `codelearner index <repo> --embed --force`.
```

The per-resolver breakdown is the reason this command exists. "63.5% resolved" is one
number; *which strategy* produced each binding, and at what confidence, is what tells
you whether a resolution rate is trustworthy — the 12.7 points removed in Phase 0 were
one resolver's entire output, and a summary rate would never have shown it.

Which model produced the vectors is reported alongside whether any exist, because
vectors from two different models are not comparable: querying with a mismatched
embedder returns results that look plausible and mean nothing. `search` checks the
same thing and disables dense rather than answering from incomparable vectors.

### Exit codes

| code | meaning |
|---|---|
| 0 | success — including "no results", which is an answer, not a failure |
| 1 | a condition the tool predicted: no index, no modality, an index already there |
| 2 | usage error (argparse's own convention) |

No predictable failure produces a traceback. A missing index, an index without
embeddings, an embedder that does not match the vectors on disk — these are normal
states of the world, and each one gets a sentence saying what happened and what to
do about it.

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
| 6 | Onboarding tours | done |
| 7 | MCP server + CLI | |
| 8 | Eval: per-modality ablation (done), faithfulness, gate controls | partial |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 167 tests
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
