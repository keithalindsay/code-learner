# Interfaces

[← code-learner](../README.md) · the case study

Two front doors over one index (`<repo>/.codelearner/index.db`, one file per repo): the CLI for a human, the MCP server for the agent already in your editor. Every console block below is real captured output.

---

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

> **`codelearner` is not on the path in this repo's own venv, and every example below
> is written as if it were.** `pyproject.toml` declares both `codelearner` and
> `codelearner-mcp` under `[project.scripts]`, and a test asserts that declaration —
> but the package is not installed into `.venv` at all, so neither script exists and
> `import codelearner` works only because the checkout is the working directory. The
> invocation that works today is:
>
> ```bash
> .venv/bin/python -m codelearner.cli <subcommand>
> ```
>
> Run `pip install -e .` to get the short form. This is WP18.2 and the honest reading
> is that the test pins the declaration and not the installation, which is the exact
> shape of a green test over a broken behaviour that this project is otherwise written
> against.

Every console block below is real output, captured at code-learner@`3212972`. Only
absolute paths are generalised.

### `codelearner index`

```console
$ codelearner index /path/to/code-learner
indexed /path/to/code-learner
  index      /path/to/code-learner/.codelearner/index.db
  files             69
  symbols        1,714
  edges          9,475
  chunks         1,706
  resolved       3,431  75.8% of 4,529 in-repo references (4,946 target code outside this repo)
```

The index goes to `<repo>/.codelearner/index.db` unless `--index-path` says
otherwise — one file per repo, which is what makes cross-repo contamination
structurally impossible rather than merely discouraged.

Two counts, because only one of them is honest: 3,431 of 9,475 edges resolved is
36%, but 4,946 of those edges target stdlib or third-party code and are *correctly*
unresolvable. 75.8% is the rate against references that could have resolved.

Re-indexing an existing index **refuses** rather than rebuilding, because there is
no incremental update yet and a rebuild throws away embeddings that cost minutes:

```console
$ codelearner index /path/to/repo
codelearner: an index already exists at /path/to/repo/.codelearner/index.db. There is no
incremental update yet, so re-indexing means rebuilding from scratch. Re-run with --force to
delete and rebuild it -- note that this discards any embeddings, which are the expensive part
-- or use --index-path to build a second index elsewhere.
$ echo $?
1
```

And if that index holds a tier-2 store, `--force` alone is not enough either:

```console
$ codelearner index /path/to/repo --force
codelearner: an index already exists at /path/to/repo/.codelearner/index.db, and it holds a
tier-2 store: 150 assertions, 147 verdicts, 10 staleness events. Rebuilding from scratch
discards 150 assertions, 147 verdicts, 10 staleness events and any embeddings -- and only the
embeddings are re-derivable. Re-run with --force --carry-assertions to rebuild and carry the
store across (a claim whose evidence moved comes back stale, with a log row, rather than
vanishing), or --force --discard-assertions to destroy it deliberately.
```

| flag | effect |
|---|---|
| `--index-path` | where to write the index |
| `--force` | delete and rebuild. Always discards embeddings; refuses if a tier-2 store is present without one of the two flags below |
| `--carry-assertions` | with `--force`: dump the store to a sidecar and restore it after the rebuild. Subjects are re-resolved by qualname; a claim whose subject is gone keeps a NULL link; a claim whose cited bytes moved comes back `stale` with a log row |
| `--discard-assertions` | with `--force`: destroy the store deliberately. Irreversible |
| `--embed`, `--model` | build dense vectors (needs the `[embed]` extra, ~1.2GB of weights, and a GPU to be quick) |
| `--json` | machine-readable output |

Restore deliberately **bypasses `write_assertion`**. These rows were already admitted;
sending them back through the door would refuse exactly the ones worth keeping — a
rejected claim re-admitted as active, a stale one refused as `EvidenceStale` so the
machinery that records what went wrong destroys the record, a claim whose subject the
rebuild no longer parses refused when the schema is shaped to keep it with a NULL link.
`span_verifications` is deliberately not carried, for the reason given under
[the staleness engine](ARCHITECTURE.md#the-staleness-engine).

### `codelearner search`

```console
$ codelearner search "how does a lease expire and get reclaimed" --repo /path/to/swarm-sync -k 5
5 result(s) for 'how does a lease expire and get reclaimed'  [lexical+dense+graph, k=5]
  1  T0  swarmsync.coordinator.reaper.reap_once
        function  swarmsync/coordinator/reaper.py:103-157  score 0.0178  [dense+graph]
        via calls tests.test_reaper.test_reap_once_ignores_already_released_lease
  2  T0  swarmsync.hooks.adapter.cmd_precheck
        function  swarmsync/hooks/adapter.py:420-479  score 0.0177  [graph+lexical]
        via called by swarmsync.hooks.adapter._keepalive
  3  T0  swarmsync.hooks.adapter._deny_response
        function  swarmsync/hooks/adapter.py:365-401  score 0.0164  [lexical]
  4  T0  swarmsync.hooks.adapter._keepalive
        function  swarmsync/hooks/adapter.py:412-417  score 0.0161  [lexical]
  5  T0  swarmsync.worktree.git_ops.prune_orphan_worktrees
        function  swarmsync/worktree/git_ops.py:422-501  score 0.0159  [lexical]
```

Two things that line-up is trying to make impossible to miss:

- **The tier is on every row.** `T0` is a parsed fact — the text matched, nothing had
  to be decided. `T1` means graph expansion got there, and expansion only walks
  *resolved* edges, so reaching it depended on a name binding that can be wrong.
  A hit found by both is `T0`: the text modality reached it without the binding.
- **`via` is the graph's account of itself.** Retrieval that cannot say why it
  returned something is hard to debug and harder to trust.

A third, on an index with no embeddings — **the missing modality is announced, not
hidden**, on stderr, so `--json` on stdout stays a single parseable document:

```console
$ codelearner search "spreading activation" --repo /path/to/code-learner --json -k 2
codelearner: dense retrieval unavailable: this index has no embeddings. Build them with
`codelearner index <repo> --embed --force`.
{
  "query": "spreading activation",
  "index": "/path/to/code-learner/.codelearner/index.db",
  "k": 2,
  "facts_only": false,
  "modalities": { "lexical": true, "dense": false, "graph": true },
  "count": 2,
  "drift": {
    "checked": true, "indexed": 69, "changed": 0, "missing": 0,
    "unindexed": 0, "method": "mtime_ns+size_bytes"
  },
  "hits": [
    {
      "rank": 1, "tier": "T0", "tier_n": 0, "symbol_id": 689,
      "qualname": "codelearner.retrieve.graph.expand", "kind": "function",
      "path": "codelearner/retrieve/graph.py", "line_start": 68, "line_end": 124,
      "score": 0.020418, "modality": "graph+lexical", "is_test": false,
      "via": "called by codelearner.generate.pipeline._neighbour_ids"
    },
    {
      "rank": 2, "tier": "T0", "tier_n": 0, "symbol_id": 687,
      "qualname": "codelearner.retrieve.graph", "kind": "module",
      "path": "codelearner/retrieve/graph.py", "line_start": 1, "line_end": 188,
      "score": 0.016393, "modality": "lexical", "is_test": false,
      "via": ""
    }
  ]
}
```

The first hit is the mixed case worth reading twice: lexical *and* graph both reached
it, so the tier is `T0` — the text match did not depend on the resolution — while
`via` still records the structural route that agreed. `via` is always present and
empty when the hit did not come from the graph; a consumer should not have to probe
for a key to find out whether an explanation exists.

`drift` is the tier-0/1 staleness check described under [the tier
model](ARCHITECTURE.md#the-tier-model). It reports what it can see and names its own method, so the
counts read as floors.

By default `search` returns locations from the index snapshot, not source. Pass
`--include-source` to append complete, line-numbered current symbol bodies after the
ranked results (or an `evidence` object in `--json` mode). The assembler re-reads only
indexed paths beneath `--repo`, refuses symlinks and unsafe files, and verifies each
whole symbol against its indexed hash before printing it. If those bytes have changed
since indexing, search fails rather than presenting stale source as current; compact
search remains available when source was not requested.

`--evidence-budget BYTES` sets the source-response allowance (default 16,384; capped
at 65,536). It includes whole symbols in retrieval order or records them as omitted —
it never truncates a symbol to fit. The human header and JSON `evidence` metadata
always report the used bytes, budget, and omitted sections.

| flag | effect |
|---|---|
| `-k N` | results to return (default 10) |
| `--facts-only` | T0/T1 only — drops semantic claims before the cut and refills the page from source |
| `--no-lexical`, `--no-dense`, `--no-graph` | turn a modality off; the switches the ablation needs |
| `--no-assertions` | turn tier-2 retrieval off; the source-only control the semantic layer is measured against |
| `--debug-scores` | show the per-modality rank contributions behind each fused score |
| `--rerank` | reorder with a cross-encoder that reads the query (opt-in; downloads ~3.4GB on first use, and says so and answers anyway if it cannot load) |
| `--include-source` | add complete, current, hash-verified source evidence for returned symbols |
| `--evidence-budget BYTES` | byte budget for `--include-source` (default 16,384; maximum 65,536) |
| `--json` | machine-readable output on stdout, notes on stderr |
| `--repo`, `--index-path` | which index to use (default: `$PWD/.codelearner/index.db`) |

**`--no-lexical` with an index that has no embeddings is an error, not an empty
result** — with both text modalities gone the pipeline would return nothing for every
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
schema     v6

counts
  files             75
  symbols        1,345
  edges          8,232
  chunks         1,336

freshness
  75 indexed files still match the tree by mtime_ns+size_bytes, and no .py file in the
  tree is missing from the index. An edit that preserves mtime and size is not detected
  by this check, so this is not a statement about bytes.

edges by tier
  T0 FACT          5,682  call site as written, unbound
  T1 RESOLVED      2,550  bound to a symbol, with confidence
  T2 INFERRED          0  always 0 here: inference lives in assertions, not on edges

assertions (tier 2)
  active              73
  rejected            67
  stale               10
  total              150

symbol kinds
  function         1,080
  method             124
  module              75
  class               66

resolution
  2,550 of 8,232 edges resolved -- 63.8% of 3,999 in-repo references (4,233 external, 1,449 ambiguous)
  import_alias/v1              1,254  confidence 0.85
  module_local/v1                957  confidence 0.90
  exact_qualname/v1              258  confidence 1.00
  unique_basename/v1              71  confidence 0.75
  self_attr/v1                     9  confidence 0.95
  class_attr/v1                    1  confidence 0.85

embeddings
  1,336 vectors from Qwen/Qwen3-Embedding-0.6B (1024-dim)
```

The per-resolver breakdown is the reason this command exists. "63.8% resolved" is one
number; *which strategy* produced each binding, and at what confidence, is what tells
you whether a resolution rate is trustworthy — the 12.7 points removed in Phase 0 were
one resolver's entire output, and a summary rate would never have shown it.

The `T2 INFERRED 0` row used to be captioned "the inference layer is not built yet",
which stopped being true when Phase 9 shipped and stayed on screen anyway. It is
structurally zero: inference lives in the `assertions` table, which is the block above
it, and `stats` was blind to that block entirely until recently while MCP `index_stats`
reported it — two surfaces over one index disagreeing.

Which model produced the vectors is reported alongside whether any exist, because
vectors from two different models are not comparable: querying with a mismatched
embedder returns results that look plausible and mean nothing. `search` checks the
same thing and disables dense rather than answering from incomparable vectors.

### `codelearner learn`

The only command that calls a language model. Drafts one claim per candidate symbol,
cites by menu reference, and admits what survives `write_assertion`. See
[Generation](RESULTS.md#generation-the-claims-and-the-numbers-that-judge-them) for what a run
produces and what it is worth.

```bash
codelearner learn --repo /path/to/repo
codelearner learn --repo /path/to/repo --limit 20 --json
codelearner learn --repo /path/to/repo --no-callers    # harder on purpose
```

| flag | |
|---|---|
| `--repo`, `--index-path` | which index to use |
| `--model` | ollama model to draft with. Default `llama3.1:8b` — **not** a Qwen model, because the faithfulness judge is one |
| `--host` | ollama host (default `localhost`) |
| `--limit` | stop after N symbols |
| `--max-offers` | menu size including the subject (default 12) |
| `--no-callers` | offer only the subject and its callees. A symbol's purpose is usually visible from its callers, so this makes the task harder |
| `--redo` | re-draft symbols that already hold an active claim from this generator |
| `--quiet` | no per-symbol progress |
| `--json` | machine-readable, carrying every counter |

Re-running **resumes** rather than duplicates: a symbol that already holds an active
claim from the same generator is skipped. The store never deletes, so a second run
without this would double it permanently and re-weight every rate computed over it
afterwards. A claim that went *stale* is no longer active, so its symbol becomes a
candidate again — the repo invalidated that claim and the pipeline re-derives it.

Passing a Qwen model prints a warning and runs anyway. Measuring the family collision on
purpose is a legitimate experiment; doing it by accident and then reading the
faithfulness score as an independent audit is not.

### Exit codes

| code | meaning |
|---|---|
| 0 | success — including "no results", which is an answer, not a failure |
| 1 | a condition the tool predicted: no index, no modality, an index already there, a schema mismatch |
| 2 | usage error (argparse's own convention) |

No predictable failure produces a traceback. A missing index, an index without
embeddings, an embedder that does not match the vectors on disk, an index built by a
different schema version — these are normal states of the world, and each one gets a
sentence saying what happened and what to do about it. `SchemaVersionError` used to
traceback out of `search`, `stats` and `learn` and raise into the MCP transport, which
violated both of those guarantees on the most-predicted failure in the whole design:
`SCHEMA_VERSION` has moved six times.

## MCP server

The CLI is for a human. The MCP server is for the agent already sitting in your
editor, and it exists because of one architectural decision: **this tool does not
call an LLM.**

A retrieval tool that wanted to answer "what does this function guarantee" would
have to call a model — which means an API key, a bill, a rate limit, and a second
place where an unciteable sentence can be born. Inverting it costs nothing and buys
everything. The agent is already running and already paid for, so it calls *in*. The
tool does the deterministic half — parse, retrieve, hash, gate, store — and the agent
supplies the judgement through `submit_assertion`, where a gate decides whether that
judgement may be kept.

The gate is worth something only because it cannot be argued with. It checks that the
subject is a symbol this index actually parsed, that every cited span exists at the
lines given, and that its bytes still hash to what was cited. All three are
arithmetic. A model cannot talk its way past a sha256, and when it fails it is told
which citation moved and what the file says now — so the next attempt is a correction
rather than another guess. What that is worth, measured, is [The
gate](RESULTS.md#the-gate).

### Install and configure

```bash
pip install -e ".[mcp]"            # `mcp` pulls in pydantic/starlette/uvicorn
codelearner index /path/to/repo    # the server serves an index; build one first
```

Paste this into your MCP client's config (`claude_desktop_config.json`, `.mcp.json`,
or your client's equivalent):

```json
{
  "mcpServers": {
    "codelearner": {
      "command": "codelearner-mcp",
      "args": ["/path/to/repo"]
    }
  }
}
```

`--index-path /elsewhere/index.db` overrides the default location, and
`--transport streamable-http` serves over a port instead of stdio for a remote or
shared index. From a bare checkout, before anything is installed — which, per the
note under [CLI](#cli), is the state this repo's own venv is in — point `command` at
the venv's own interpreter instead:

```json
{
  "mcpServers": {
    "codelearner": {
      "command": "/path/to/code-learner/.venv/bin/python",
      "args": ["-m", "codelearner.server", "/path/to/repo"]
    }
  }
}
```

The server **starts even when the index does not exist**. That is deliberate: an MCP
client launches its servers at session start and marks one that exits non-zero as
failed, so refusing to start over a missing index would take the whole integration
down over a condition that one command fixes. It starts, and the first tool call
returns `{"ok": false, "error": {"code": "no_index", ...}}` with the command to run.

**A rebuilt index is refused once, not reconnected to silently.** A live server used
to keep serving a rebuilt index from the deleted inode and report `ok: true`,
`accepted: true`, `servable: true` for writes that vanished — which is the realistic
second-day sequence: agent session open in the editor, human re-indexes in a terminal.
`IndexSource` now carries `(st_dev, st_ino)` and returns `index_replaced` on the first
call after a swap. A quiet reconnect would be worse than an error: every
`content_hash` the agent holds was published by the previous build, and a rebuild
happens *because the source changed*, so a silent reconnect returns correct data about
a codebase the agent is not holding. The refusal is self-clearing. The window between
`connect()` and commit cannot be closed from here — sqlite will not say whether the
file under an open handle was unlinked — but what is closed is the *report*: the write
is no longer returned as success.

### The tools

| tool | what it returns |
|---|---|
| `search_code(query, k, facts_only, include_assertions, debug_scores, …)` | hybrid retrieval — lexical + dense + graph + tier-2 assertions, RRF-fused. Every result carries `candidate_type` (`source` or `assertion`) and `candidate_key`. A source result has qualname, `path`, line range, the modalities that found it, `via`, and the `content_hash` needed to cite it; an assertion result has the claim, its supporting verdicts, its freshness, and the citations behind it. `facts_only` drops claims before the cut and refills with source; `include_assertions=false` is the source-only ablation. |
| `get_symbol(qualname)` | one symbol, its resolved callers and callees (T1, each with its resolver's confidence), its unbound call sites (T0), and any servable assertions about it. |
| `reading_path(topic, limit)` | the onboarding tour, ordered by dependency depth so a stop's callees come before it. With a topic it is seeded from retrieval; without one, from call-graph centrality. |
| `submit_assertion(subject_qualname, claim, evidence_spans, …)` | the inversion. Stores a tier-2 claim if and only if its citations hold. |
| `index_stats()` | what is in the index, by tier: counts, edges split T0/T1, assertions split active/rejected/stale, resolution rates, and whether vectors are present. |

Every tool returns `{"ok": true, …}` or `{"ok": false, "error": {"code", "message",
…}}`. No predictable condition raises into the transport — a traceback crossing an
MCP boundary tells the agent the tool is broken, which is the one conclusion that
stops it trying again. This is the machine-facing half of the same policy the CLI's
exit codes implement for a human.

### The gate

An evidence span is `{path, line_start, line_end}` plus **something to check it
against**: either the `content_hash` retrieval already handed you, or the exact
`text` you read at those lines. Lines are 1-based and inclusive.

`submit_assertion` refuses, and says why. The authoritative list is `app.ERROR_CODES`,
and a test asserts that set and the codes actually raised in that file are the same,
so a refusal cannot be added without documenting it:

| code | when |
|---|---|
| `evidence_required` | zero evidence spans. An uncited claim cannot be adjudicated, cannot expire, and cannot be checked by a reader — it is indistinguishable from a good one at every stage after this, so the only place to stop it is the door. |
| `empty_claim` | correct citations under no statement. A judge has no proposition to adjudicate and a reader finds correct bytes with no reason they were cited. |
| `hash_mismatch` | the cited bytes no longer hash to what was cited. Returns `observed_hash` and `observed_text`, so the citation can be corrected rather than re-guessed. |
| `evidence_unverifiable` | a span with neither hash nor text. A location that asserts nothing about what is there can never be found to be wrong. |
| `invalid_span` | a byte range that is empty, negative or inverted. sha256 of nothing is stable, so a zero-length citation does not merely fail to expire — it reports `fresh` forever, against whatever the file becomes. |
| `unknown_subject` | a subject qualname this index never parsed. Verified spans do not make a claim about a symbol that does not exist reachable by anyone. |
| `evidence_stale` | the store's own re-read disagreed with the citation at the moment of writing. A claim whose first verification is guaranteed to fail was never true of this repository. |
| `path_escapes_repo` / `span_escapes_repo` | the cited path resolves outside the repository. Two codes because the rule is enforced at both doors and a refusal must name the rule that fired; the second is the store's copy, reached through `write_assertion`. |
| `bad_range` / `file_missing` | the citation does not point at readable bytes this index parsed. |
| `bad_path` | the path contains a NUL byte. |
| `file_too_large` / `too_many_spans` / `claim_too_long` / `bad_confidence` | the submission is over a cap. One call once stored 5,000 spans and a 5MB claim, permanently, amplified on every later read. |

`file_missing` deliberately covers both "no such file" and "a file this index never
parsed", and that is not sloppiness. A distinct `file_not_indexed` code would preserve
an oracle in reduced form: submit `.env` and `.envv`, and the difference between two
codes answers "does this file exist on disk". Absent, present-but-unindexed, and
secret are now indistinguishable from outside, and it was verified after the fix that
`.env` and `.envv` return byte-identical refusals.

That oracle was real and it was the refusal, not the gate. `observed_text` used to
return the full cited byte range, decoded and uncapped, inside `hash_mismatch`. The
gate refused paths escaping the repo root correctly, but *inside* the root it read any
file, indexed or not, and echoed the bytes back — so an agent submitted a deliberately
wrong hash and read the file out of the error, walking the line ranges to get whole
files. Reads are now restricted to files the index actually parsed, quoted text is
capped, and an oversized file is refused by `stat()` before it is read.

Every rule in that table is enforced in `assertions.store.write_assertion`, before it
opens a transaction, so a refused claim leaves no row behind and `codelearner learn`
and every library caller meet the same gate the MCP tool does. `submit_assertion` runs
richer versions of two of them first — it can name the offending field and quote the
bytes that are really there — but as a pre-check for message quality, never as the
enforcement. Enforcement reuses `_first_failure`, the same function the serve path
uses, so admission and serving cannot disagree about what verified means.

One bad span refuses the whole submission. Admitting the ones that happened to verify
would leave a claim standing on a subset of the evidence its author thought it had.

A line range has two honest readings — the symbol that occupies those lines, and the
whole lines themselves — and they hash differently. A symbol's stored bytes begin at
`def` rather than in the indentation before it, at the `@` for a decorated symbol, and
run to the last byte of the file for a module. Measured on code-learner@`3212972`,
that is [438 of 1,714 symbols](RESULTS.md#the-decorator-span-and-the-citations-it-could-not-reach).
Both readings are built and both are re-hashed off disk, and the cited hash may match
either. That is not a loosening: every candidate is read from the file as it is right
now, so a stale or invented hash still matches nothing. What it removes is a false
rejection — the more dangerous failure here, because an agent told that its correct
citation is wrong learns that the gate is noise.

Admission is not a permanent licence either. `get_symbol` re-reads and re-hashes
every cited span on every call, so a claim whose evidence has been edited is expired
on the way past rather than served one last time — and marked `stale`, never deleted.
The rejected and stale sets are the only evidence that the gate does anything.

The loop the whole design rests on is three calls:

```
search_code("how are leases acquired")   -> hits, each carrying a content_hash
  (the agent reads the code and forms a claim)
submit_assertion(qualname, claim, [{path, line_start, line_end, content_hash}])
  -> accepted, or refused with a reason it can act on
get_symbol(qualname)                     -> the claim, re-verified against disk
```
