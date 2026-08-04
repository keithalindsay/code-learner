# bench — does a code index actually help an agent?

Three arms, identical in everything except which MCP servers `claude -p` is given:

| arm | MCP config | what it tests |
|---|---|---|
| `bare` | `mcp/bare.json` (empty) | the built-in tools alone |
| `codegraph` | `mcp/codegraph.json` | [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph) v1.5.0 — structure only, stores **no** LLM claims |
| `code-learner` | `mcp/code-learner-fast.json` | this project — structure **plus** accountable claims about what code is *for* |

All three keep the built-in tools. The question is whether an index **helps**, not
whether it replaces `grep`; removing the built-ins from `bare` would manufacture the
win. `harness.py verify` proves the three built-in sets are byte-identical before any
budget is spent.

Nothing here is imported by `codelearner/`, nothing here is collected by pytest
(`testpaths = ["tests"]`), and no file outside `bench/` is modified.

---

## The claim under test

codegraph's README reports **89% fewer tool calls, 60% cheaper, 69% fewer tokens** from
4 runs per arm, median only — no intervals, no variance, no significance, and no check
that the answer was right. This harness asks the same question with the parts that were
missing:

- **the answer text is stored verbatim**, because an efficiency win on a wrong answer
  is not a win. Correctness is judged separately against ground truth in `tasks/*.json`;
  judging never re-runs an agent.
- **variance is measured before `n` is chosen**, because four runs cannot see it.
- **pairing and bootstrap clustering are both on task**, because runs of one task are
  not independent evidence.

---

## Quick start

```bash
# 0. static comparison, no API cost
.venv/bin/python bench/coverage.py ~/projects/swarm-sync ~/projects/kalshi-bot

# 1. prove the arms are comparable, before spending anything (~$0.30)
.venv/bin/python bench/harness.py verify --repo ~/projects/swarm-sync

# 2. measure variance BEFORE choosing n
.venv/bin/python bench/harness.py run --arms bare --reps 12 --only <one-task-id>
.venv/bin/python bench/analyze.py --variance

# 3. size the run from the cv you just measured
.venv/bin/python bench/analyze.py --sizing-cv 0.40 --budget-runs 180 --n-tasks 12

# 4. the matrix, then the paired task-clustered analysis
.venv/bin/python bench/harness.py run --reps 5
.venv/bin/python bench/analyze.py
```

Run them in that order. `verify` is worth repeating every time — an MCP server that
worked yesterday is not evidence it is on `PATH` today, and it is the only thing
standing between a broken arm and a published number.

If `shim/snapshots/codelearner.json` is missing or stale, re-capture it:

```bash
.venv/bin/python bench/shim/fastmcp.py --snapshot bench/shim/snapshots/codelearner.json \
  --cwd ~/projects/swarm-sync \
  --capture .venv/bin/python -m codelearner.server
```

---

## Task format — `bench/tasks/*.json`

Authored by a different agent; this harness only consumes them. One object or a list of
objects per file:

```json
{
  "id": "ss-lease-cas",
  "repo": "/home/keith/projects/swarm-sync",
  "prompt": "Which function performs the compare-and-swap ...?",
  "ground_truth": "anything the judge needs — carried through untouched"
}
```

`id`/`task_id`/`name`, `prompt`/`question`/`query`, `repo`/`repo_path` and
`ground_truth`/`answer`/`expected` are all accepted; a bare repo name resolves under
`/home/keith/projects`. Unknown keys survive in `extra`. `ground_truth` is copied into
every result and **never read during a run** — judging is a separate pass and must not
be able to leak into the agent's context.

---

## Result format — `bench/results/<run_id>.json`

One self-contained JSON per run, plus a `<run_id>.stream.jsonl` transcript. `run_id` is
`{task_id}__{arm}__r{rep}__{8 hex}`, so a re-run never overwrites its predecessor.

| field | why it is recorded |
|---|---|
| `answer` | **the point.** Verbatim, untruncated, judged later. |
| `ground_truth` | carried from the task so a result file is judgeable alone. |
| `tool_calls`, `tool_call_counts`, `n_tool_calls` | the "89% fewer tool calls" claim. |
| `index_tool_calls`, `toolsearch_calls`, `index_tool_errors` | did it *use* the index; what it spent finding it; whether the call failed. |
| `tools_offered`, `mcp_servers`, `mcp_status`, `index_offered`, `index_first_class` | was the index *offered*. See the traps below. |
| `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` | cache read is split out because a warm cache is a cost difference that has nothing to do with the index. |
| `total_cost_usd`, `wall_s`, `duration_api_ms`, `num_turns` | |
| `arm_position` | position within its counterbalanced block, so a cache effect shows as a *position* effect rather than an *arm* effect. |
| `ok`, `error` | `ok=False` is **void, not cheap** — excluded by `analyze.py` and counted in the diagnostics. |

---

## Traps, and what was done about each

### 1. A dead MCP server looks exactly like an unused one

…and it looks *cheap*, so it biases the arm in its own favour. Two independent checks:
`probe_server()` speaks MCP directly over stdio and lists tools without involving
`claude` at all, and every run reads the `init` event's `mcp_servers` status. A run
whose server is absent or `failed` is marked `ok=False` rather than averaged in.

### 2. MCP tools are **denied** by default under `--permission-mode dontAsk`

Found by measurement, not by reading docs. The first probe of the `code-learner` arm
came back:

> Calling `index_stats` was denied by the current permission mode ("don't ask mode")

The agent had the tool, tried it, and was refused — indistinguishable in the usage
numbers from an index that did not help. Every arm now passes `--allowedTools` naming
both MCP servers. A benchmark that skipped this would have measured its own permission
config and published it as an index result.

### 3. A slow handshake makes a server's tools **invisible**, and it nearly voided this arm

This was the single largest threat to the benchmark's validity, and it took four
experiments to pin down.

`init` fires before slow MCP servers finish connecting, and tools that arrive after it
land in the deferred pool behind `ToolSearch`. Out of the box:

| arm | status at `init` | tools |
|---|---|---|
| `codegraph` (compiled binary) | `connected` | first-class |
| `code-learner` (Python) | `pending` | **deferred, effectively invisible** |

**The consequence, measured:** across 6 pilot runs the `code-learner` arm had its
server running and its five tools loadable, and the agent called one **zero times**.
The arm scored *worse* than `bare` (1.37x the tool calls) — a false negative about the
index produced entirely by startup behaviour. Appending a note that deferred tools
exist did not help either: in 4 more runs the agent never even spent the `ToolSearch`.

**What it actually was.** Latency alone did not explain it. A relay answering
`initialize` in 39 ms was *still* `pending`, and trimming the tool list to a single
tool changed nothing. Capturing the JSON-RPC traffic settled it — Claude Code's opening
sequence is:

```
initialize -> notifications/initialized -> tools/list -> prompts/list -> resources/list
```

codegraph declares `{"tools": {}}` and is asked for `tools/list` alone: two round trips
against a 116 ms server. The reference `mcp` Python SDK declares `prompts` and
`resources` whether or not a server implements either, so `codelearner.server` is asked
for **four**, serialised behind its ~600 ms import. No `MCP_TIMEOUT` setting helps —
that caps how long a connection may take, it does not make `init` wait.

**The fix — `shim/fastmcp.py`, on by default.** A stdlib-only relay (~31 ms to first
byte) answers all four handshake methods from a captured snapshot and forwards every
`tools/call` to the real server untouched. The server's behaviour is entirely intact;
only the SDK import leaves the critical path. Result: `status: connected`, all five
tools first-class, `deferral_symmetric: true`.

The snapshot is auditable, not magic: `--capture` writes it, `--check` re-derives and
diffs it per repo, `verify` runs that check, and drift is a hard failure — a stale
snapshot would serve tools the server no longer has.

`--no-fast-handshake` reproduces the out-of-the-box condition. That number answers
"what happens if I install this today"; the default answers "does this index help".
Both belong in the report. `--defer-parity` and the per-run `toolsearch_calls` /
`n_tool_calls_adj` remain available as cross-checks.

**And the finding that survived the fix:** with the tools first-class, permitted, and
visible, the agent *still* called a code-learner tool in only 1 of 4 runs, while
codegraph's tool was called in 5 of 5 pilot runs. That is now a real behavioural result
rather than an artifact — and a plausible driver is that `codegraph_explore`'s
description opens "**PRIMARY TOOL — call FIRST for almost any question**" and ships
4,597 characters of server instructions telling the agent to use it instead of reading
files. Part of the control's advantage may be tool-description engineering rather than
index content, and `index_tool_calls` is what makes that separable.

### 3b. MCP companion tools are injected unevenly

Once our server connects, Claude Code adds `ListMcpResourcesTool`,
`ReadMcpResourceTool` and `ReadMcpResourceDirTool` — because the SDK advertised a
`resources` capability. codegraph gets none of them. Our resource list is **empty**, so
the three tools can do nothing while still occupying that arm's system prompt. They are
classified as `MCP_COMPANION_TOOLS` (not built-ins, so `same_builtins` stays meaningful
at 21 across all arms) and `verify` prints the asymmetry rather than hiding it.


### 4. Cache contamination

`cache_read_input_tokens` is recorded separately, and arm order is randomised per block
by `arm_orders()` so no arm sits second by construction. `arm_position` makes the
ordering effect measurable rather than assumed away. Every block still contains one run
of every arm, so the design stays paired and an interrupted matrix leaves whole blocks.

### 5. Inherited `CLAUDE_*` environment

A benchmark launched from inside a Claude Code session is not the environment a user's
agent runs in. With the parent's variables inherited, the child came up with a
**different built-in tool set**. `SCRUB_ENV` strips them.

### 6. This Claude Code build has no `Grep` or `Glob` tool

2.1.220 exposes neither; search happens through `Bash` (ripgrep is on `PATH`). All
three arms are equally affected so the internal comparison is fair, but a published
"N% fewer tool calls than grep+read" was measured against a baseline that *had* a
first-class `Grep`, and this is not that baseline. `verify` prints
`has_grep_tool: false` so the point cannot be forgotten.

---

## The control arm, characterised

codegraph v1.5.0, installed from npm (`@colbymchenry/codegraph`, MIT). It ships a
bundled Node runtime and a Rust kernel (`codegraph-kernel.node`), ~283 MB on disk.
Indexing writes a `.codegraph/` directory **inside the repo** and starts a background
daemon. Telemetry is on by default and is disabled here through `DO_NOT_TRACK=1`,
`CODEGRAPH_TELEMETRY=0` and `CODEGRAPH_NO_UPDATE_CHECK=1` in `mcp/codegraph.json`.

**It exposes exactly one MCP tool.** The CLI has `node`, `callers`, `callees`,
`impact`, `files` and `query`, and the server instructions mention `codegraph_node`,
but `tools/list` returns a single entry:

```
codegraph_explore
  required: ["query"]
  properties:
    query        string   symbol names, file names, short code terms, or a question
    maxFiles     number   max files to include source from (default 12)
    projectPath  string   absolute path; uses the nearest .codegraph/ at or above it
```

A response is a single markdown document — dynamic-dispatch boundaries, a symbol
count, a "blast radius" list of dependents with test files named, then verbatim
line-numbered source grouped by file with `... (gap) ...` elisions. The one sampled
for "how does lease acquisition work" on swarm-sync was 417 lines / ~19 KB covering 32
symbols across 3 files. Everything in it is source text or graph structure; there is no
generated prose about what any of it is *for*, which is exactly why it is the right
control.

For comparison, `code-learner` exposes five tools — `search_code`, `get_symbol`,
`reading_path`, `submit_assertion`, `index_stats` — and `submit_assertion` has no
counterpart at all, since codegraph stores no claims to submit.

### Index build, same repos, 3 runs each (median)

| repo | system | build | index on disk |
|---|---|---|---|
| swarm-sync (75 files) | codegraph | 0.77 s | 6.8 MiB |
| swarm-sync | code-learner | 0.89 s | 5.4 MiB (13.4 MiB with embeddings + claims) |
| kalshi-bot (~341 files) | codegraph | 2.16 s | 31.8 MiB |
| kalshi-bot | code-learner | 3.17 s | 22.3 MiB (54.5 MiB with embeddings + claims) |
| facefusion (~222 files) | codegraph | 0.91 s | 9.8 MiB |
| facefusion | code-learner | — (pre-existing index) | 10.4 MiB |

Both are fast enough that build time is not a differentiator at this scale. codegraph's
`.codegraph/` is smaller than our full index and larger than our structural-only one;
the gap is embeddings and the tier-2 claim store, which is the thing codegraph
deliberately does not have.

---

## Coverage — `coverage.py`

The two projects publish a "resolution rate" under the same word for different
quantities:

- **codegraph:** file-level, *inbound* — the share of symbol-bearing source files with
  at least one resolved cross-file dependent.
- **code-learner:** reference-level, *outbound* — the share of in-repo references that
  bound to a symbol.

A file with one incoming import and four hundred unresolved calls is fully covered
under the first and 0.2% covered under the second. `coverage.py` computes **both
definitions against both indexes**, so the report has an apples-to-apples row. It also
applies code-learner's own `external` rule (basename matches nothing in the repo ⇒ not
a resolution failure) to codegraph's `unresolved_refs`, which carry no such
distinction, and excludes codegraph's structural `contains` edges.

The file-level metric is reported twice, with and without test files. On swarm-sync it
moves from **48.4% to 96.3%** on that choice alone — a pytest module is imported by
nothing by construction, so the metric largely reports each repo's test ratio. That
48-point swing is larger than any difference this benchmark could plausibly measure
between the two systems.

Measured, both definitions, both systems, same repos:

| repo | system | ref-level (outbound) | file-level, all | file-level, non-test |
|---|---|---|---|---|
| swarm-sync | code-learner | 2,550 / 3,999 — **63.8%** | 27 / 62 — 43.5% | 25 / 27 — 92.6% |
| swarm-sync | codegraph | 4,356 / 5,355 — **81.3%** | 30 / 62 — 48.4% | 26 / 27 — 96.3% |
| kalshi-bot | code-learner | 10,467 / 20,977 — **49.9%** | 142 / 307 — 46.3% | 139 / 203 — 68.5% |
| kalshi-bot | codegraph | 17,981 / 25,997 — **69.2%** | 159 / 309 — 51.5% | 153 / 205 — 74.6% |
| facefusion | code-learner | 6,914 / 8,627 — **80.1%** | 105 / 161 — 65.2% | 104 / 120 — 86.7% |
| facefusion | codegraph | 8,253 / 9,897 — **83.4%** | 107 / 161 — 66.5% | 106 / 120 — 88.3% |

**codegraph resolves more, on both denominators, on all three repos.** That is the
honest result: 17.5 points of reference-level resolution on swarm-sync, 19.3 on
kalshi-bot, 3.3 on facefusion. Whatever case this project has to make, it is not a
structural-coverage case, and any claim it makes has to be about the claims layer
rather than about the graph underneath it.

Note also how far the same metric moves across repos for the *same* system — 49.9% to
80.1% for code-learner — which is the reason the agent benchmark pairs and clusters on
task rather than pooling. A repo is not a neutral backdrop.

Read across a row, never down a column into the other metric.

---

## Statistics — `analyze.py`

1. `--variance` — spread of tokens, cost, tool calls and wall-clock over repeats of one
   task in one arm, with a **CI on the sd itself**. From 4 runs that interval spans
   roughly 0.6×–2.9× the true sd.
2. `--sizing-cv <cv> --budget-runs <n>` — what `n` a budget buys and the smallest
   *ratio* it can resolve, via `codelearner.eval.ablation`'s `required_n`,
   `ci_half_width`, `design_effect` and `CALIBRATION_FLOOR = 128`. Sizing is done on
   `log(metric)`, where a k-fold difference is a difference of `log k`, so the bounded-
   score machinery applies to an unbounded ratio without being reimplemented.
3. default — paired per-task ratios with a **task-clustered** bootstrap CI, plus
   diagnostics that print *above* the table: too few tasks, void runs, runs where the
   index was available and never called, MCP errors, deferral asymmetry, and
   cache-read-by-position.

Wall-clock is flagged `*noisy` everywhere and weighed last: codegraph's own README
concedes a floor effect on small repos.

### Measured variance — 12 runs, one task, `bare` arm

| metric | mean | sd | cv | min | median | max | max/min | 95% CI on the sd |
|---|---|---|---|---|---|---|---|---|
| `n_tool_calls` | 3.08 | 1.24 | **40.2%** | 2 | 3 | 6 | **3.00×** | [0.88, 2.11] |
| `total_tokens` | 149,300 | 42,380 | 28.4% | 110,257 | 145,700 | 246,754 | 2.24× | [30.0k, 72.2k] |
| `total_cost_usd` | 0.126 | 0.022 | 17.4% | 0.105 | 0.119 | 0.175 | 1.66× | [0.016, 0.037] |
| `wall_s` | 20.2 | 3.10 | 15.4% | 16.9 | 19.2 | 26.0 | 1.54× | [2.20, 5.28] |

Same repo, same prompt, same arm, same model. **Tool calls — the metric the headline
claim is about — is the noisiest of the four**, ranging 2 to 6 with a coefficient of
variation of 40%. Wall-clock was *not* the noisiest in relative terms here, contrary to
the usual expectation; it is still weighed last because of the floor effect, but the
data did not support the assumption and it is recorded rather than repeated.

Note also that `cache_read_input_tokens` ran 88,847 to 231,500 across these 12
identical runs, which is why arm order is counterbalanced and cache reads are stored
separately.

### What that variance buys, and what it does not

Sizing at cv = 0.402 on `log(metric)`, effective observations (not runs):

| ratio to detect | eff. n @50% power | eff. n @80% power |
|---|---|---|
| 1.10× | 69 | 140 |
| 1.25× | 13 | 26 |
| 1.50× | 4 | 8 |
| 2.00× | 2 | 3 |

At **n = 4 — the comparison benchmark's design — the 95% interval spans 0.67× to
1.48×.** So four runs per arm genuinely can resolve a claimed 0.11× (89% fewer calls);
the direction of a large effect is not the problem. The problems are that no interval
was published so a reader cannot tell which claims are resolvable, that a median over
four runs hides a 3× spread, that the sd itself is unmeasurable from four runs (its own
95% interval spans ~0.6× to 2.9× the truth), and above all that **no correctness
measure was reported at all**.

The effective-n conversion is where a budget really goes. Runs cluster inside tasks, so
at an assumed ICC of 0.05, 180 runs (60 per arm) buy:

| tasks | reps | runs/arm | effective n | resolves @50% | @80% |
|---|---|---|---|---|---|
| 4 | 15 | 60 | 35 | 1.14× | 1.21× |
| 12 | 5 | 60 | 50 | 1.12× | 1.17× |
| 30 | 2 | 60 | 57 | 1.11× | 1.16× |

**Stated up front: at 60 runs per arm this design resolves roughly a 1.15× difference
and no better, and it sits below the measured `CALIBRATION_FLOOR` of 128, where this
repo's own percentile bootstrap over-rejects (11.9% actual against a nominal 5%).**
Every interval it prints at that size should be read as descriptive, not as a test.
Effective n cannot exceed `n_tasks / ICC` however many times each task is re-run — add
tasks, not repeats.

### What the real task set buys

The authored set is **34 tasks** across swarm-sync (11), kalshi-bot (13) and facefusion
(10). At the measured cv of 0.402, an assumed ICC of 0.05, and $0.17 per run (the pilot
mean; median $0.14, max $0.38):

| reps | runs/arm | total runs | effective n | resolves @80% | est. cost |
|---|---|---|---|---|---|
| 2 | 68 | 204 | 65 | 1.15× | ~$33 |
| 3 | 102 | 306 | 93 | 1.12× | ~$49 |
| 4 | 136 | 408 | 118 | 1.11× | ~$65 |
| **5** | **170** | **510** | **142** | **1.10×** | **~$82** |
| 8 | 272 | 816 | 201 | 1.08× | ~$131 |

**`--reps 5` is the recommendation**: 510 runs is the first point where effective n
(142) clears the `CALIBRATION_FLOOR` of 128, so it is the cheapest design whose
intervals mean what they say. Everything below that row is descriptive only, and every
row above it buys very little — the effective-n curve is already flattening because
34 tasks cap it at 680 no matter how many repeats are bought.

The ICC of 0.05 is an assumption until the matrix runs; `analyze.py` prints the
**measured** design effect per metric afterwards, and the sizing should be re-read
against it. If it comes back nearer the 5.4 this repo measured across repos for
retrieval scores, every row above needs revisiting upward.

---

---

## What this design cannot do, stated up front

- **It cannot resolve less than about a 1.15× difference** at 60 runs per arm, and that
  figure sits below the `CALIBRATION_FLOOR` of 128 where this repo's own bootstrap
  over-rejects. Anything smaller than ~1.2× should be reported as "not resolved by this
  design", never as "no difference".
- **Tool-call count, the headline metric, is the noisiest** (cv 40%, 3× spread over 12
  identical runs). It needs the most n and deserves the least confidence.
- **Two tasks are not a benchmark.** The cluster bootstrap has as many effective
  observations as there are tasks; below ~5 the pooled interval is a placeholder and the
  per-task rows are the honest reporting. In the pilot the sign of codegraph's effect
  *flipped* between the two tasks — cheaper on one, more expensive on the other — which
  is exactly why pairing is on task and why a two-task pooled number would be
  meaningless.
- **Tool-call counts are not comparable to codegraph's published figures**, which were
  measured against a baseline that had a first-class `Grep`. This build does not.
- **The `code-learner` arm needs `shim/fastmcp.py` to be measurable at all** on this
  Claude Code build. That is disclosed rather than quietly compiled in, and
  `--no-fast-handshake` reproduces the unfixed condition.
- **Correctness is not measured here.** It is judged separately from the stored answers
  against ground truth a different agent authored. Until that pass runs, every number
  in this harness is an efficiency number, and an efficiency win on a wrong answer is
  not a win.

---

## Side effects of running this

- `codegraph init` writes a **`.codegraph/` directory inside the measured repo** and
  leaves a **background daemon running** per project. Both `swarm-sync` and
  `kalshi-bot` now carry an untracked `.codegraph/`. `codegraph uninit <path>` removes
  it; `codegraph daemon` lists and stops the daemons.
- Results are about **43 KB per run** (JSON + stream transcript), so a 180-run matrix
  is ~8 MB. The transcripts are kept, not truncated — they are the audit trail behind
  every number.
- The measured repos' `.codelearner/index.db` files are **read, never rebuilt**. The
  build timings above were taken into a scratch `--index-path`, so no existing tier-2
  claim store was touched.

---

## Files

```
bench/
  README.md
  coverage.py            both coverage denominators, both systems
  harness.py             arms, run_one/run_matrix, verify, probe_server, resolve_config
  analyze.py             variance, sizing, task-clustered paired bootstrap
  mcp/
    bare.json            arm 1: no MCP servers
    codegraph.json       arm 2
    codegraph-delayed.json   arm 2 behind shim/delay.py, for --defer-parity
    code-learner.json    arm 3, exactly as shipped (--no-fast-handshake)
    code-learner-fast.json   arm 3 behind shim/fastmcp.py -- the DEFAULT
  shim/
    fastmcp.py           answers the handshake from a snapshot; forwards tool calls
    delay.py             starts a server late, to equalise connection timing
    snapshots/           captured handshake replies, checked for drift by `verify`
  tasks/                 task + ground truth (owned by another agent)
  results/               one JSON + one .jsonl transcript per run
    _exploratory/        pilots and config probes; skipped by analyze.py by default
```

Directories under `results/` whose names start with `_` are skipped by `analyze.py`, so
pilots and harness self-checks are kept as evidence without ever being swept into a
headline average. Point `--results-dir` at one explicitly to analyse it.
