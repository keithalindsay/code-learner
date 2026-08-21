# Results

[← code-learner](../README.md) · the case study

Every measurement in the project, stated once, each beside the caveat that bounds it. Conventions (stamping, intervals, the calibration floor) are in [Methodology](METHODOLOGY.md); what these numbers cannot settle is [there too](METHODOLOGY.md#what-these-numbers-cannot-resolve).

---

Where an interface section later needs a figure it links here rather than restating it.

## Ingest and resolution

Four repositories, indexed by `codelearner index`, at the shas in
[How to read the numbers](METHODOLOGY.md#how-to-read-the-numbers-in-this-document):

| repo | files | symbols | edges | chunks | resolved (in-repo) |
|---|---|---|---|---|---|
| swarm-sync@`3119a97` | 75 | 1,345 | 8,232 | 1,336 | 63.8% |
| kalshi-bot@`8a2e9b5` | 340 | 7,531 | 43,994 | 7,502 | — |
| code-learner@`3212972` (itself) | 69 | 1,714 | 9,475 | 1,706 | 75.8% |
| TradingAgents@`f362a16` | 58 | 255 | 1,664 | 245 | — |

"In-repo" is the honest denominator: **roughly half of all calls in real code target
stdlib or third-party code** and are correctly unresolvable. On swarm-sync that is
2,550 of 8,232 edges resolved — 31% flat, but 63.8% of the 3,999 references that
*could* have resolved, with 4,233 targeting code outside the repo. Counting those as
failures makes a working resolver look broken.

The self-index includes `repo/`, a two-file fixture, and the `tests/` tree; it is not
a measurement of the library alone.

### One number that went down on purpose

In-repo resolution on swarm-sync briefly reached **76.2%**, then was deliberately
reduced. It reads 63.8% today at swarm-sync@`3119a97`; it was 63.5% on the smaller
tree the decision was taken against, and the 12.7-point drop is the number that
matters, not either endpoint.

The 76.2% included a strategy that bound dotted attribute calls by unique name.
That pointed 38 `r.json()` calls on an HTTP response at a nested helper inside a
test file, and made that helper the highest-ranked symbol in the entire call graph.
472 of 519 such bindings were attribute calls. The receiver's type is unknown, so
`x.foo()` carries no evidence about which `foo` is meant.

Removing the strategy cost 12.7 points of coverage and removed a set of confident
fabrications. That trade is the point of the project, and it showed up in its own
resolver on day one. See [docs/PHASE0-FINDINGS.md](PHASE0-FINDINGS.md).

### The decorator span, and the citations it could not reach

A symbol's span used to start at `def`, not at `@`, because that is where
tree-sitter-python's `function_definition` node begins. This was the only **fail-open**
defect found across six independent audits. Every other one loses a claim or raises;
this one *served a false claim with no signal anywhere*. A claim reading "serves GET
/users; responses cached 60s" cited a span containing neither `@route` nor `@cache`.
Rewrite the decorators and both verifiers reported `fresh`, `method='hash'`,
`verified_at=now`; `force_hash=True` could not help, because the cited bytes genuinely
had not changed; `staleness_log` stayed empty; the faithfulness judge, shown the same
truncated span, correctly ruled the claim supported; and a human following the citation
found those exact bytes unchanged.

The span now starts at the outermost `@`. One `decorated_definition` node wraps a whole
decorator stack, so `node.parent` reaches it in one step; `async def` has no separate
node type; nested definitions get their own wrappers. This changes every stored hash,
hence `SCHEMA_VERSION = 6` — a no-DDL bump that still has to refuse a v5 index, because
a v5 index's hashes are well-formed and would go on verifying while excluding the
decorators the claims are about.

**The re-measurement corrected a claim this codebase had repeated in four places.**
The symbols whose stored bytes differ from their *lines'* bytes, code-learner@`3212972`:

| kind | differ | total |
|---|---|---|
| method | 325 | 325 |
| module | 69 | 69 |
| function | 36 | 1,186 |
| class | 8 | 134 |
| **all** | **438** | **1,714** (25.6%) |

Every method and every module, plus 44 functions and classes. Cross-tabulating those
44 against decoration and nesting gives the finding, and it is exact in both
directions:

| decorated | nested | disagrees | count |
|---|---|---|---|
| no | no | no | 1,138 |
| yes | no | no | 138 |
| no | yes | **yes** | 38 |
| yes | yes | **yes** | 6 |

Among functions and classes, disagreeing is **exactly** being nested — 44 of 44 in
both directions — and being decorated explains none of it: 138 decorated top-level
functions and classes agree perfectly, and not one decorated-but-unnested symbol
disagrees. A top-level `@` sits in column 0 exactly where its `def` did, so this is a
property of *indentation* and always was. The older attribution to "the decorated
functions and classes" was wrong, and the older figure ("around 15%") was inconsistent
with its own counts (85 of 383 is 22.2%, not 15%). Widening decorated spans does not
move this population by a single symbol.

Fixing the span left a gap that neither the fix nor the store-carry owned. A claim
carried across the v5→v6 rebuild keeps its **pre-v6** citation; those bytes are
unchanged on disk; so the claim stays `active` and servable — correctly, by the rules
as written, and with the exact exposure the fix existed to close. On swarm-sync's
upgraded index, 11 of 143 active evidence spans were pre-v6 narrowed citations, and one
of them cited an endpoint whose missed prefix is

```python
@app.post("/intent", dependencies=[Depends(require_token)])
```

— a live servable claim about an endpoint, citing bytes that exclude its own
authentication dependency. Strip `Depends(require_token)` and the claim still verifies
fresh.

Detection is exact rather than heuristic. `decorated_body_start` asks tree-sitter for
the offset a pre-v6 span *would* have used for exactly these bytes — the same parser
answering both halves — and the span must match the wrapper node on both ends.
`lstrip().startswith("@")` gets a multi-line decorator argument list, a comment between
decorators, an `@` inside a leading string, and `async def` wrong, and both of its error
directions leave a narrowed citation active. The distinction that keeps it from
over-firing: a legitimate sub-range citation of a function body is *narrower and
therefore stronger* evidence, and expiring those would punish the good case. On the real
index the rule took 11 of the 15 suffix spans and correctly left 4 — all claims about a
class's last method, whose span ends where the class does.

Such a claim is marked `stale` with its own reason, `decorators_excluded`, and **never
rewritten**: widening a stored span to match the new symbol would fabricate a citation
the generator never made. `hash_mismatch` would have been the wrong reason too — the
bytes did not change, the citation *boundary* did, and the two ask to be read
differently. swarm-sync's store carries 10 such rows today.

## Retrieval

Three modalities — lexical (BM25 over FTS5), dense (`Qwen3-Embedding-0.6B` vectors),
and graph expansion — fused with Reciprocal Rank Fusion, then optionally reordered by
a cross-encoder.

### The gold set

**638 rows across 6 repos, in three sources that are never pooled silently.** It was
16 queries on one repo, and that set could not resolve the comparisons it was being
asked to make: `hit@5` moved in steps of 6.25 points — one query — and 11 of the 16
carried exactly one relevant symbol, which makes `recall@k` and `hit@k` the same
number computed twice.

| source | rows | repos | how |
|---|---|---|---|
| `hand` | 170 | 3 | hand-written; 108 multi-relevant, 21 hard negatives |
| `mined_verbatim` | 227 | 6 | one sentence of a commit message, unmodified |
| `mined_name_blind` | 225 | 6 | the same sentence with every target's identifier removed |
| `unspecified` | 16 | 1 | the original hand set, kept as written |
| **total** | **638** | **6** | |

The two mined rows are 227 against 225 rather than a matched pair, and the gap is a
correctness fix rather than a rounding artefact: two candidates can blind to the *same
text* with *different* gold answers — facefusion's "Fix blank screen in
`replace_audio()`" and "Fix blank screen in `restore_audio()`" both become `fix blank
screen in`. One query string with two contradictory gold rows scores every retriever
wrong on at least one of them regardless of what it returns, so the blind row is dropped
there and the verbatim row kept unpaired. A consumer must not assume a `pair_id` appears
exactly twice.

The three biases differ, and pooling them hides that. Hand-written questions are
written by someone who knows the codebase, so they borrow the vocabulary the code
already uses — which flatters lexical retrieval. Mined prose inherits the committer's
vocabulary, which describes a *change* rather than the code. So `GoldSet` carries
`source` and `repo` on every row, scoring emits a row per stratum, and the pooled row
is labelled `POOLED` so nobody reads it as a measurement of one thing.

Three decisions in the mining are load-bearing:

- **Commit-first, not symbol-first.** Asking per symbol "which sentences name me?"
  yields exactly one relevant symbol per query *by construction* — which is the defect
  that made the old hand set's two metrics duplicates. Here the unit is a sentence and
  its relevant set is every symbol that sentence names. 28% of mined queries are
  multi-relevant (128 of 452), and the strata separate. Lexical-only over the 350 mined
  rows whose repos are indexed: `hit@5` 0.598 against `recall@5` 0.323 on the
  multi-relevant stratum, an identical 0.435/0.435 on the single one — which is the
  definition of `hit@k`, and the reason a single-relevant set cannot report both.
- **Attribution by file touch.** A sentence may only name symbols in files that commit
  changed. Without it, a swarm-sync commit about pruning worktrees that mentions
  `acquire_lease` in passing becomes a worktree query whose gold answer is a lease
  function.
- **Emitted twice, verbatim and name-blind, sharing a `pair_id`.** The mention rule
  guarantees a mined query contains its targets' identifiers, so blinding everything
  measures a mode users do not use and blinding nothing is a lexical benchmark. The
  *gap between the two rows is the finding*, and it is reported below. The two rows are
  not independent — same sentence, same symbols — so any interval over them must
  cluster on `pair_id` or on `commit`.

The hand set is the counterweight, because mining reaches only symbols a commit
happened to name. Each mined file carries a `bias` block measuring exactly what that
selects for; on swarm-sync@`3119a97` the mention rule picked 90 symbols out of 318 and
they are **80% documented against a 70.4% population and 20% private against 39.6%**,
with classes over-represented 1.48× and methods and functions near-neutral (0.94 and
0.92). A retrieval set drawn preferentially from documented, public symbols will
overstate any retriever that reads docstrings — and dense retrieval reads docstrings.
The bias is stated so a consumer can weight it, stratify it, or discount a result that
lands inside it; the trap is the version that is implicit and baited with a number that
looks like it came from the whole codebase.

Hand authoring reaches what mining structurally cannot: nested closures, module symbols
as targets, duplication-as-the-answer (kalshi-bot has two fee formulas that genuinely
disagree), and **absences** — "we do NOT retry on 4xx" is unmineable, because commits
describe changes and not deliberate non-behaviour. **166 of the 170 hand queries have
zero token overlap with any target's name**, reached by measuring, finding leaks,
rewording, and re-measuring; the 4 residuals leak one unavoidable domain noun each and
carry `name_bearing: true` derived from the measurement rather than asserted by hand.
21 are hard negatives, written so the lexically obvious answer is the wrong one.

Token overlap is **reported, never filtered**, and the argument is empirical rather
than principled: any threshold worth setting rejects the benchmark it extends. What it
turns out to measure is documentation density rather than query discipline. Median
`source_overlap` of the hand queries against their own targets' source, per repo:
swarm-sync **0.429**, kalshi-bot **0.143**, TradingAgents **0.125** — the same
authoring rule, the same author, a 3.4× spread, because swarm-sync is roughly 40%
English docstring and TradingAgents is the terse-docstring extreme. That is a confound
for any pooled row, and it is why the per-repo rows are published beside the pooled
one.

### The table

520 rows scored, being the three repos that are both indexed and embedded
(swarm-sync@`3119a97`, kalshi-bot@`8a2e9b5`, TradingAgents@`f362a16`), run at
code-learner@`3212972`. Each repo is scored against its own index. `nDCG@10` is
primary; the other columns are kept because they are what the earlier tables reported.

```
n = 520 queries. single-relevant 310/520 (60%); mean relevant per query 1.59.
CI: bootstrap over queries, 2000 resamples, seed 20250801 — marginal, not for row differences.
```

| configuration | recall@5 | recall@10 | hit@5 | MRR | **nDCG@10** | MAP@10 |
|---|---|---|---|---|---|---|
| lexical only | 0.347 | 0.423 | 0.446 | 0.303 | 0.304 | 0.246 |
| dense only | 0.269 | 0.346 | 0.321 | 0.221 | 0.237 | 0.190 |
| graph only (dense-seeded) | 0.141 | 0.199 | 0.187 | 0.125 | 0.128 | 0.094 |
| lexical + dense | 0.326 | 0.425 | 0.404 | 0.281 | 0.293 | 0.232 |
| lexical + dense + prefer-impl | 0.444 | 0.533 | 0.540 | 0.411 | 0.409 | 0.346 |
| hybrid, no prefer-impl | 0.327 | 0.440 | 0.406 | 0.273 | 0.291 | 0.225 |
| **hybrid + prefer-impl (default)** | **0.469** | **0.571** | **0.562** | **0.415** | **0.424** | **0.354** |

Graph weight sweep, demotion on throughout so the comparison is against the best
non-graph configuration rather than a strawman:

| graph weight | MRR | nDCG@10 |
|---|---|---|
| 0.3 (default) | 0.406 | 0.417 |
| 1.0 | 0.364 | 0.393 |
| 1.5 | 0.262 | 0.303 |

Monotonic decline above 0.3, which is why 0.3 is the default and why a test pins the
constant so it cannot drift without re-running this.

### What the table says

Every delta below is **paired** — the same queries scored under both configurations —
with a query bootstrap and, where clustering matters, a repo bootstrap beside it.
nDCG@10, 520 queries.

| comparison | what it isolates | Δ nDCG@10 | over queries | over repos |
|---|---|---|---|---|
| hybrid + prefer-impl − lexical only | the shipped default vs the simplest thing | **+0.120** | [+0.096, +0.146] | [+0.047, +0.159] |
| lexical + dense − lexical only | adding one modality | −0.012 | [−0.034, +0.010] | [−0.068, +0.056] |
| **hybrid, no prefer-impl − lexical only** | **adding both modalities, demotion held off** | **−0.013** | **[−0.036, +0.009]** | [−0.083, +0.047] |
| dense only − lexical only | dense on its own | −0.068 | [−0.099, −0.037] | [−0.156, +0.181] |
| lex+dense+prefer-impl − lexical+dense | the demotion, no graph | +0.116 | [+0.101, +0.133] | [+0.000, +0.155] |
| hybrid + prefer-impl − hybrid, no prefer-impl | the demotion, with graph | +0.133 | [+0.117, +0.150] | [+0.000, +0.180] |
| hybrid + prefer-impl − lex+dense+prefer-impl | graph, given the demotion | +0.015 | [+0.003, +0.029] | [−0.009, +0.025] |

**The shipped default beats lexical alone by +0.120, and essentially none of that is
the extra modalities.** The first row is the honest headline for the product: the
default configuration is +0.120 over the simplest thing that could work, and the
interval excludes zero even when the resampling unit is the repo — three repos being
few enough that that interval is itself unstable, which is why the per-repo rows are
given too.

But rows two and three are what the ablation is *for*, and they do not say what a
hybrid-retrieval project would like them to say. **Adding dense retrieval to lexical
buys −0.012 [−0.034, +0.010]. Adding both dense and graph, with the demotion held off,
buys −0.013 [−0.036, +0.009]** — and on MRR that same comparison is a *significant
loss*, −0.030 [−0.057, −0.002]. These are not wide intervals around a hopeful zero;
they are tight enough to say the gain is absent rather than merely unproven, and the
point estimates are on the wrong side.

So the modality claim does not survive in isolation. What the +0.120 is made of is rows
five and six: the test demotion, worth +0.116 and +0.133 depending on whether graph is
present. **A one-line score multiplier is doing the work three retrieval modalities and
a fusion algorithm were built to do.** That is the finding, it is not the one this
architecture was designed around, and it is stated here rather than presented as a
hybrid-retrieval win with the demotion quietly switched on inside it.

**Dense retrieval loses to lexical on this corpus, and where it does not, it is
reading names.** Pooled, dense-only is −0.068 [−0.099, −0.037] against lexical-only.
Split by source, paired, nDCG@10:

| stratum | dense | lexical | Δ | paired 95% |
|---|---|---|---|---|
| `hand` (n=170) | 0.179 | 0.219 | −0.040 | [−0.102, +0.022] |
| `mined_verbatim` (n=175) | 0.413 | 0.406 | +0.007 | [−0.037, +0.050] |
| `mined_name_blind` (n=175) | 0.117 | 0.286 | **−0.169** | [−0.216, −0.124] |

On hand-written questions the two are not separated by this corpus — the interval
crosses zero, and a previous version of this file called that a dense loss without
one. On verbatim commit prose they tie. On the *same sentences with the identifiers
removed*, dense collapses. Name-blinding costs lexical 0.406 → 0.286, about 30%,
which is the expected shape and is roughly what "how much of BM25's score is name
matching" should look like. It costs dense 0.413 → 0.117, about **72%**. The
bi-encoder is leaning on identifier tokens considerably harder than BM25 is, which is
not what a dense modality is bought for.

**Why the demotion is worth so much is directly visible.** Share of the top 10 that is
test code, same 520 queries:

| ranking | test-code share of top 10 |
|---|---|
| dense, raw | 62.1% |
| lexical, raw | 43.5% |
| lexical + dense fused, no demotion | 53.3% |
| lexical + dense fused, demotion on | **19.0%** |

Both text modalities systematically rank tests above the implementations they
exercise, because a test states the behaviour in prose, names it in the function
title, and repeats the vocabulary; the implementation just does it. Dense is the worse
offender of the two. The demotion is a halving of score, not a filter — tests still
rank, and are often genuinely the best answer.

**`prefer_implementation` per repo, and the row that is not evidence:**

| repo | test files | Δ nDCG@10 | paired 95% |
|---|---|---|---|
| swarm-sync (n=232) | 38 / 75 | +0.180 | [+0.157, +0.204] |
| kalshi-bot (n=234) | 107 / 340 | +0.117 | [+0.093, +0.142] |
| TradingAgents (n=54) | **0 / 58** | +0.000 | [+0.000, +0.000] |
| POOLED (n=520) | | +0.133 | [+0.117, +0.150], repo-clustered [+0.000, +0.180] |

TradingAgents' Δ is not a null result, it is a **no-op**: that repo has no file
`is_test_path` recognises, so the demotion has nothing to demote and the interval is
exactly `[0, 0]` rather than merely containing zero. An interval of zero width is the
signature of a mechanism that did not fire. Two things follow. The pooled row is an
average over one repo where the effect *cannot* be present, so it understates the
effect where the mechanism exists — and the repo-clustered interval touching +0.000 is
that structural zero, not evidence of fragility. And `is_test_path` is a convention,
not a guarantee: TradingAgents' one test file is `test.py` at the repo root, which
matches none of `tests/`, `test_*.py`, `*_test.py`, or `conftest.py`. Whether that is
a gap in the convention or a correct refusal to guess is a judgement call; either way
it is visible here rather than absorbed into a pooled mean.

**Graph expansion adds little, and the little it adds is inside the noise.** Turning
graph expansion on over the best non-graph configuration is +0.015 [+0.003, +0.029]
resampling queries and [−0.009, +0.025] resampling repos. It stays in at weight 0.3
because that measured best among the weights tried and because it is the only modality
that can reach code the text modalities never scored, not because this measurement
defends it. Graph expansion has no query representation, so every vote it casts is
evidence about the *code* rather than about the *question*; past a low weight those
votes displace better-matched answers, which is what the sweep above shows.

### Reranking, and what did not survive re-measurement

> **These rows were corrected once already, and the correction stands.** The
> reranking numbers first published here (0.750 / 0.781 / 0.875 / 0.679) did not
> reproduce. Re-running the ablation four times — two independently built indexes,
> two repeats each — gave identical results every time, and they were not those. The
> likely cause is that the original measurement predates a later change to what text
> the reranker is shown. Two of the three original claims did not survive:
> **recall@10 did not fall** and **hit@5 did not improve**. A number that cannot be
> reproduced against the code shipped beside it is not a measurement, and correcting
> it in place is cheaper than being wrong in public.
>
> **And the corrected rows are now withdrawn too, for a different reason.** They were
> measured on the 16-query set, which is below the calibration floor stated at the top
> of this file: the intervals that set produces are not 95% intervals, they are
> narrower ones, in the direction that invents findings. So the earlier conclusion
> that reranking was "the largest lever in the project" was computed with an interval
> that does not mean what it said. Reranking has **not** been re-measured against the
> 638-row gold set. No reranking figure is published here until it has been.

The reranker is **`zeroentropy/zerank-1-small-reranker`** — 1.7B parameters,
Qwen3-based, ~3.4GB of weights, run on a 10GB RTX 3080. `BAAI/bge-reranker-base` is
wired as a fallback for machines that cannot hold the larger one and has **not** been
benchmarked here. Reranking is opt-in (`--rerank`) and its absence is not an error:
retrieval widens the candidate set to `k × 4`, the reranker reorders it, and without
one the pipeline returns the fused rows above. The architectural argument for it is
unchanged and is independent of the withdrawn numbers — a cross-encoder reads the
query and one candidate *together*, which is the one thing neither RRF (which sees
only positions) nor the bi-encoder (which sees the two texts separately) can do.

Two findings from the 16-query era that are recorded as **unresolved rather than
established**, because they were inside that set's real noise band: reranking bought
ranking and not recall, and reranking did not vindicate graph expansion (with the
reranker on, turning graph off scored identically). Both are plausible, both have a
mechanism, and neither has been tested at a size that could distinguish them from
zero.

## The gate

None of the design argument for tier 2 is worth believing on the strength of a
paragraph describing it. `codelearner.eval.gate_controls` generates an adversarial
corpus from what an index actually holds and reports a rate — and the rate is the
least interesting thing it reports.

**Two doors, because there are two.** The corpus used to import `server.app` and
nothing else, so every rate it published described the MCP path alone, while
`codelearner learn` and every library caller reached the store directly. It now runs
at both. The columns are not expected to agree, and where they differ the difference
is the finding: the same attack is often refused by a *different rule* at each door,
so acceptable codes and mutations are declared per gate rather than per family.
Collapsing them into a union would let a refusal for the wrong reason score as
attribution, which is the one thing the codes exist to prevent.

Running it at that second door is how `SpanEscapesRepo` was found. A claim citing
`../outside_the_repo.py` with that file's real hash was admitted by `write_assertion`,
stored `active`, and reported servable, on every generated instance. `path_escapes_repo`
lived solely in the server's `_verify_span`, so the MCP caller met it and `codelearner
learn` did not.

**Sized, not just measured.** The pooled figure this apparatus used to lead with was
never wrong and was reported at a resolution the corpus cannot support. It is fifteen
attack *shapes* instantiated once per symbol, and for most of them the gate does
literally the same thing on every instance: the same lines run, the same bytes (usually
none) are read, the same `raise` fires. Four significant figures over a numerator that
can only be 0 or n does not describe twelve thousand independent adversarial probes; it
describes a handful of probes photocopied.

So replication is now *determined by instrumenting the gate* rather than by reading
family names. A `settrace` hook digests the executed `(file, line)` sequence inside the
package and the bytes the gate actually read and hashed, with boundary calls identified
by code-object identity so a rename cannot turn it into a line counter. It overturned
the brief it was written from: `empty_claim` is varying at the direct door, because
`_submit_body` verifies the spans before `write_assertion` ever refuses the claim; and
at the store door eight families are replicated, not four, with `paths == 1` for every
one of them — meaning the decision path never varies at all and only the hashed bytes
do.

**Per family, code-learner@`3212972`, 1,714 symbols, 15 attack shapes:**

| family | expect | n | probes | rate | ub95 (probes) | shape (direct / store) |
|---|---|---|---|---|---|---|
| `zero_evidence` | refused | 1,714 | 1 / 1 | 100.0% | 95.00% | replicated / replicated |
| `empty_claim` | refused | 1,714 | 1,712 / 1 | 100.0% | 0.17% / 95.00% | varying / replicated |
| `unverifiable_span` | refused | 1,714 | 1,712 / 1 | 100.0% | 0.17% / 95.00% | varying / replicated |
| `absent_file` | refused | 1,714 | 1 / 1 | 100.0% | 95.00% | replicated / replicated |
| `escaping_path` | refused | 1,714 | 1 / 1 | 100.0% | 95.00% | replicated / replicated |
| `past_eof` | refused | 69 | 69 / 69 | 100.0% | 4.25% | varying / varying |
| `blank_range` | refused | 138 | 69 / 1 | 100.0% | 4.25% / 95.00% | varying / replicated |
| `zero_length_span` | refused | 0 / 1,714 | — / 1 | 100.0% | — / 95.00% | not expressible / replicated |
| `decoy_content_hash` | refused | 1,714 | 1,712 | 100.0% | 0.17% | varying |
| `stale_but_once_valid` | refused | 1,714 | 1,714 | 100.0% | 0.17% | varying |
| `foreign_symbol_hash` | refused | 1,701 | 1,699 | 100.0% | 0.18% | varying |
| `unknown_subject` | refused | 1,714 | 1 / 1 | 100.0% | 95.00% | replicated / replicated |
| `published_hash` | **accepted** | 1,714 | 1,712 | 100.0% | 0.17% | varying |
| `quoted_lines` | **accepted** | 1,645 | 1,643 | 100.0% | 0.18% | varying |
| `multi_span` | **accepted** | 56 | 56 | 100.0% | 5.21% | varying |

```
direct  refused 100.0%  attributed 100.0%  positive 100.0%   15,620 instances / 8,691 distinct gate executions
store   refused 100.0%  attributed 100.0%  positive 100.0%   17,334 instances / 5,202 distinct gate executions
        positive controls: 3,415 legitimate submissions / 3,411 distinct executions
```

**That reframing is the whole point, and the interval carries it.** A replicated
family's honest one-sided 95% upper bound on its failure rate is **95.00% at one
probe** — not 0.17% at 1,714 instances. The bound is Clopper-Pearson exact, chosen over
the rule of three because `3/n` stops being a probability exactly where this corpus
needs one (`3/2 = 1.5`), and it is computed on probes with the naive per-instance figure
printed beside it so the difference is visible rather than argued. Even the probe-based
bound is optimistic: the probes are one template with one field varied, not a sample
from the space of attacks.

The pooled rate survives, demoted to what it is — **an existence claim about this run**:
no instance of any enumerated attack was admitted at either door. It is printed at one
decimal. `1.0000` no longer appears anywhere.

**Positive controls, which are not decoration.** A gate that refuses everything scores a
perfect rejection rate. So the corpus also submits every symbol cited by the hash this
index published for it, and by its exact lines quoted off disk, and those must be
admitted — including the 438 of 1,714 symbols (25.6%) whose stored bytes are *not*
their lines' bytes, which a narrowed gate would falsely reject while still refusing
every attack.

**Mutation, measured at both doors and both polarities.** Every family names the gate
rule it targets as the textual edit that removes it; `run_mutation` copies the package
to a temp tree, applies that edit to the *copy*, and re-runs the family in a subprocess.
`12/12 mutation-verified` was one door and one polarity. Measured across both:

```
direct   11/11 negative rules -> mutant hold rate 0.000 on every instance
         3/3 positive rules detected, 2 partially (published_hash 4/8, quoted_lines 2/6)
         zero_length_span has no mutation here: the server is given line numbers and
         derives the byte range itself, so no caller can ask it for an empty range
store    12/12 negative rules -> mutant hold rate 0.000 on every instance
         3/3 positive rules detected, 1 partially (published_hash 6/8)
---
         23/23 negative rules fully detected
         6/6 positive rules detected, 3 of them partially
```

A partial detection on a positive control is not a weak result but a different one:
deleting *one reading* of a legitimate citation flips the instances that needed that
reading and leaves the rest admitted. `MutationCensus` counts rules and never says
"detected", pinned by a test, because conflating the two is how the old number drifted.

**Adding a guard in front of a controlled rule breaks that rule's mutation, and it has
now happened three times in three distinct shapes** — each of which would have reported
a control as detecting a rule it does not name:

| shape | family | how it hid |
|---|---|---|
| duplicated | `unknown_subject` | the store's copy refused the same attack with the same code |
| bypassed | `unverifiable_span` | the server substitutes the observed hash, so the store's rule is unreachable by construction |
| stacked | `escaping_path` | deleting containment drops the attack onto the index-membership guard, and no submission can separate the two — every path escaping the repo is by construction a path the index never parsed |

**What this cannot show, and it is the most important paragraph here.** Every control is
generated from `FAMILIES`, so the corpus can only ever expose a rule some family already
names. **It has found no attack nobody had enumerated.** The holes it did find were
found some other way: two auditors found two by probing outside the family list, and
adding a second door found the third. Adding doors is a second axis with the same
property. A rate of 100.0% is a statement about the enumerated attack shapes and about
nothing else.

```bash
.venv/bin/python -m codelearner.eval.gate_controls --repo . --compare      # both doors
.venv/bin/python -m codelearner.eval.gate_controls --repo . --mutations    # direct door
.venv/bin/python -m codelearner.eval.gate_controls --repo . --surface store --mutations
```

## Faithfulness: does a claim follow from what it cites?

The assertion store checks that every cited span exists and still hashes to what was
cited. That is arithmetic, and a model cannot argue with it. What it cannot check is
whether the span *supports* the claim — a citation can be perfectly present, perfectly
unedited, and completely silent on what the claim asserts.
`codelearner.eval.faithfulness` measures that, and it is the only measurement in this
repo that needs a model to produce it.

**The judge is different weights and a different tokenizer from the generator: a proxy
for independence, not a demonstration of it.** Claude writes the assertions through
`submit_assertion`; `qwen3.5:9b` judges them. A generator grading its own output shares
its blind spots, its tokenizer, its training distribution and its particular way of
being confidently wrong, so agreement between the two measures consistency rather than
truth. But the check behind the word "cross-family" is `generate.llm.model_family`, a
string-prefix test on an ollama tag — it lowercases the tag, drops the namespace and
the suffix, and takes the leading run of letters. A Qwen-distilled model published as
`deepseek-r1` passes it; this repo's own reranker is Qwen3-based and would pass it. A
published tag is a marketing string, not a lineage. So read "cross-family" as the
narrower thing that is actually true: **this model did not write the claim.**
Independence is measurable — re-adjudicate with a second judge and publish the
disagreement rate — and it is not measured here.

**The judge is prompted to refute, and every unclear path lands on "not supported".**
An LLM asked "does this follow?" will say yes to almost anything plausible, and a
permissive judge is worse than no judge: it converts an unmeasured risk into a number
that reads like a guarantee, so everything downstream then trusts the claim *more*
than before anyone looked. So the burden of proof is on the claim, silence in the
evidence is a failure rather than a neutral outcome, and an unparseable answer, an
empty answer, or a claim with no spans left to read all fail closed. The one failure
that does *not* fail closed into a verdict is an unreachable judge: recording
"uncertain" because ollama was not running would reject every assertion in the store
and log a reason that blames the claims.

**One denominator used to hold three different events.** `uncertain` arrives by three
routes that are not the same fact: the judge saying "uncertain", which is the only
route the prompt sanctions and the only one that is evidence about the claim; a
transport/parse failure; and a judge-format failure. The last two say nothing about the
claim and are charged to the generator. They are counted apart now, and past a 20%
instrument-failure rate the run aborts with `JudgeMisbehaving` — deliberately *not* a
`JudgeUnavailable` subclass, because retrying does not fix a model emitting an
unreadable shape. The harm is not the missing verdict: with `record=True` each parse
failure becomes `VERDICT_UNSUPPORTED → STATUS_REJECTED`, blaming the claim. At a 30%
rate a run rejects a third of a live store for a formatting reason.

**`supported / n` stays the headline** and `score_decided` — `supported / (supported +
not_supported)` — is reported beside it rather than in place of it. This is the
strongest argument in the module and it is worth stating plainly: a score that divides
the instrument's own failures out of its denominator is at its most flattering exactly
when the instrument is broken. A run with 40% parse failures would report a healthy
0.55 with the badness parked in a counter nobody reads. `supported / n` also needs only
the assumption that an undecided claim is at worst unfaithful, where the alternative
assumes the undecided set is random — which this module has already measured it is
not, since the salvage bug fires on claims about code containing quote characters.
Read the pair: a small gap means the claims, a large gap means the instrument.

### Measured

> **Not re-run for this document** — it is the one measurement here that needs a model
> call. The figures below are derived from the verdicts the run left in
> swarm-sync@`3119a97`'s live store, recomputed at code-learner@`3212972`; the
> adjudication pass itself was `qwen3.5:9b` at temperature 0, thinking off, output
> constrained to a JSON schema, on a 10GB RTX 3080.

```
n = 147   supported 80   not_supported 65   uncertain 2
```

| | |
|---|---|
| **faithfulness** (`supported / n`) | **0.54  [0.46, 0.62]** (Wilson) |
| `score_decided` (`supported / (supported + not_supported)`) | 0.55 |

The gap between the two is one point, which is the small-gap case: this is a statement
about the claims and not about the instrument.

**Three decimals were a lie.** `0.544` on n=147 implies a resolution two orders of
magnitude finer than the measurement has, so it is reported as `0.54 [0.46, 0.62]`.
That interval still assumes 147 independent draws, and the claims are clustered —
several per symbol, many per file, some about toy fixtures. Positive intra-cluster
correlation makes the true interval wider; Wilson understates the width by roughly a
fifth to a third at plausible ICC. `clustered_interval` estimates the design effect
from the adjudications themselves, and `cluster_correction` returns `None` — never 1.0
— when it cannot estimate, because returning 1.0 would assert a measured independence
that was not measured.

**The judge is uncalibrated on the data it measures.** Its entire calibration is
**15/16 on a *different* 16-claim set, pre-labelled by the same model family that
authored those claims** — Wilson `[0.72, 0.99]`, which is consistent with a judge that
is right 72% of the time. "Self-consistency" measured as three runs of an identical
prompt at temperature 0 measures decoding determinism, not judge stability. Label flips
under whitespace-only prompt changes have been observed with no rate attached, and that
flip rate is the largest single uncertainty in 0.54. `measure_prompt_stability` is the
harness that would put a number on it and `export_for_review` / `score_review` are the
scaffold for human calibration on the set the number is actually computed over. **Both
are built and neither has been run.** They are here so the missing measurements are
cheap enough to actually happen, and they are not results.

The single useful disagreement from that calibration set is worth keeping. Claude
claimed `record_verdict` "moves the assertion to 'rejected' only when it is currently
'active'", which is true, and the judge refused it:

> the evidence shows `record_verdict` [...] explicitly checks for `STATUS_ACTIVE`, but
> it does not provide any information about what happens when the assertion is in a
> different state [...] so we cannot confirm the claim [...] without assuming
> behavior outside the provided code.

The judge is right and the *citation* is wrong. The SQL that enforces this
(`_TOUCH_STATUS`, with its `WHERE id = ? AND status = ?`) is a module-level constant
outside the cited symbol's byte range, so the claim genuinely rests on evidence it did
not cite. That is exactly the failure the metric is for, and it is invisible to the
hash gate — every hash matched.

**A bug this found by being run for real.** On one pass a claim scored `uncertain`
because the judge, quoting `embed.serialize` back at itself, emitted `{"verdict":
"supported", "reasoning": "... the format string `f"{len(values)}f"` ..."}` — an
unescaped double quote inside its own JSON, because the code it was quoting contains
one. Invalid JSON, verdict lost, failing closed as designed and giving the wrong
diagnosis: the claim was blamed for a judge-side transcription bug. The bias is
systematic, firing on claims about code containing quote characters, which in Python
means anything involving strings. The parser now recovers the verdict *field*
specifically — a short token that survives the escaping bug beside it — and the
recovered token still has to normalise to a recognised label, so nothing can become
`supported` unless the judge wrote `supported`. `tests/test_faithfulness.py` pins that
exact string as a regression.

### What it does not measure

- **Whether the claim is true.** Only whether the cited evidence establishes it. A
  correct claim citing an irrelevant span scores as unfaithful, and that is intended:
  the citation is the only thing a later reader can check.
- **The judge is not an oracle.** It is one 9B model with its own error rate.
  `report.unfaithful` exists so a low score is read by looking at the claims that
  failed, not by trusting the number.
- **Anything about coverage.** A store with three easy claims in it can score 1.000.
  Faithfulness is a property of the claims that exist, not evidence that the
  interesting ones were made.
- **Anything about what is served.** `record_verdict` is called here, from
  `eval/faithfulness.py`, *and* from `codelearner judge` (WP17.4, done) — but this run
  did not go through the CLI, so it recorded no verdicts and changed nothing servable.
  0.54 describes claims scored by this offline pass; whether a given claim is actually
  serving is a separate question, answered by `require_verdict=True` (the production
  default) and whatever verdicts `codelearner judge` has recorded for it. There is
  still no MCP judge tool, by design — judge independence means an agent must not
  judge its own claims, so judging stays a CLI-only, out-of-band step.

```python
from codelearner import db
from codelearner.eval import OllamaJudge, adjudicate

conn = db.connect("/path/to/.codelearner/index.db")
judge = OllamaJudge()                       # qwen3.5:9b via ollama
report = adjudicate(conn, judge, record=False)   # record=True writes the verdicts
print(report.format_report())
judge.release()                             # frees ~6.6GB of a shared 10GB card
```

Only *servable* assertions are scored — the candidate set comes from
`servable_assertions`, so every span handed to the judge has just been re-hashed off
disk. A claim whose evidence has been edited is stale, which is a different failure
with a different repair; counting it here would make faithfulness fall whenever the
repo changed and the number would be measuring two things at once.

## Purpose accuracy: gold labels mined from git history

Faithfulness asks whether a claim follows from its citations. It cannot ask whether the
claim is *right* about what the code is for — that needs ground truth, and hand-writing
a purpose statement per symbol costs a paragraph of careful prose a thousand times over.

`codelearner/eval/gold_from_history.py` tests a way around it: **the commit that
introduced a symbol usually says in prose why it was written.** If so, every repo with
real history is already a labelled purpose corpus — free, unlimited, and written years
before anyone thought about an eval, so leak-free by construction *provided the
generator never sees it*.

### The leak boundary, and why it is structural

An eval whose ground truth is reachable from its input measures nothing, and it fails
*silently* — the scores go up, which is not the direction that prompts anyone to check.
So the boundary is enforced by construction rather than by review:

- The generator is handed a `SourceView`: a frozen record with fields for source,
  signature, docstring, path and span, and **no field for a commit, message or
  subject**. A test asserts the exact field set, so adding one turns it red.
- `source_view()` never invokes git. The test that says so makes `subprocess.run` raise,
  builds a view anyway, *and then* calls the miner to prove the sabotage was really in
  force — without that second half a monkeypatch on the wrong module would leave the
  test green while proving nothing.
- `assert_view_is_source_only()` is the primary gate and it is structural: every string
  in a view must occur in the file on disk. A harness bug that routed a commit message
  into the view produces text the file does not contain. A text search cannot make that
  distinction, because an author quoting their own commit message in a docstring is
  legitimate source.
- `find_leaks()` is the second gate, a recursive walk of the view's whole object graph
  for any clause of the held-out prose. `audit_leak_boundary()` runs both across the
  full label × view cross product.
- `suspect_tokens()` checks the other direction — a rare word present in the answer and
  in the generator's output but nowhere in its input.

**The cross-product audit was reachable from no scored code path, and that mattered.**
`score_purposes` checked each view against its *own* label only, so cross-symbol
leakage was invisible to every gate that ran during a scored run, and
`audit_leak_boundary` — the only thing that checks the cross product — was called by
nothing. It now runs inside `run_purpose_eval` and **fails the run** on any finding,
rather than warning: a leak does not degrade the number, it voids it, and the failure
mode of a warning is a run that prints better scores with a line above them nobody
reads.

It had been finding a real one. A **32-character clause** from `_AffectedFiles`'
held-out label sat word for word in `_reverse_dep_files`' docstring. The obvious fix —
scope the filter to the same commit — is wrong, and the correction is worth recording:
`_AffectedFiles` was introduced by `982386a` and `_reverse_dep_files` by `d6e029a`, so
a same-commit filter misses the only real leak in the corpus. The filter now rejects a
label whose clause appears in *any* labelled sibling's source regardless of commit, and
the cross-commit property is pinned in a fixture so nobody re-scopes it back.

**And it breaks a concession this project used to make.** The README used to concede
that "correlated phrasing cannot be filtered". That is true and it does not describe
what was found. A clause copied verbatim across a symbol boundary is not correlated
phrasing, it is a copy, and a copy is exactly the thing that *can* be filtered —
`find_leaks` could already see it, and what was missing was a caller. The concession
was covering a filterable failure with the language of an unfilterable one. The
remaining honest concession is narrower: shared vocabulary between a label and its
symbol's source, below a copied clause, is not filterable and is not filtered. The
`docstring_blind` condition is what bounds it.

### Measured, on swarm-sync@`3119a97` (95 commits, 318 non-test symbols)

```
MINING FUNNEL
  symbols considered              318
  rejected: no_provenance           0
  rejected: empty_prose             0
  rejected: boilerplate             0
  rejected: no_mention            273
  rejected: too_short               0
  rejected: copied_into_source      2
  rejected: copied_into_sibling     1
  USABLE                           42   (13.2% of considered)
  distinct introducing commits     17
  most symbols from one commit      9
  leak audit: 0 finding(s) over 1764 view x label pairs (42 views x 42 labels)
```

**`1,764 pairs, 0 findings` is true for the first time**, and for a different reason
than this file used to claim. It was previously published as a property of the corpus;
it is now a property of a filter that runs, plus a caller that invokes it, plus a run
that aborts if it fires.

**The technique works and it is expensive: a 13.2% yield.** One rejection reason
accounts for essentially all of the loss — **273 of 318** symbols were introduced by a
commit whose prose never names them, which is 86% of the corpus lost to a single rule.

The prior going in was that commit messages would be too *low-quality* to use ("fix",
"wip", "address review"). On this corpus that prior was simply wrong, and the funnel
says so in one line: **the boilerplate filter rejected zero symbols.** So do the
`empty_prose` and `too_short` filters. The problem is **attribution**, not quality —
these commits are large, and their prose describes a *change*. The 42 surviving labels
come from commits touching a median of 9 files (range 3–18), and the loss is
concentrated in the same way the yield is: one commit supplies 9 of the 42 usable
labels, and the distribution runs 9, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2 and then six singles.
Excellent prose about a work package is not a purpose statement about a symbol.

Two independent checks that the surviving labels are about their symbols.

**Purpose agreement**, name-blind token-F1. Null = 500 cross-commit derangements at
seed 20250729, `p = (1 + hits) / (1 + draws)` with a floor of 0.002. CI = clustered
bootstrap over the 17 introducing commits, 2000 resamples, seed 20250801.

| condition | n | gold | null | null sd | lift | lift 95% CI | p |
|---|---|---|---|---|---|---|---|
| docstring first sentence | 42 | 0.125 | 0.017 | 0.006 | 0.109 | [0.063, 0.171] | 0.002 |
| name + signature only | 42 | 0.019 | 0.004 | 0.003 | 0.014 | [0.002, 0.032] | 0.002 |
| body identifiers | 42 | **0.193** | 0.041 | 0.005 | **0.152** | [0.107, 0.205] | 0.002 |
| body identifiers, doc-blind | 42 | 0.119 | 0.031 | 0.005 | 0.089 | [0.052, 0.143] | 0.002 |

**Four statistical corrections, and the levels fell.** The table above is the corrected
one, re-run at `3212972`; the four figures in this paragraph describe the *superseded*
code and cannot be re-derived from the current tree, so they are reported as the
one-off diagnosis that motivated the change rather than as reproducible measurements.

`lift` used to be a single draw from the null — one permutation — so it inherited that
draw's full sampling error, never reported. The shipped draw sat +2.28 sd (body
identifiers) and −1.24 sd (name + signature) from the 500-draw mean: **opposite
directions on the same table**, which is what makes a single draw a bias and not merely
imprecision. Name-blinding covered only the *leaf* qualname component, so 34 of 43
labels still shared tokens with parts that are also guaranteed present in the path and
the class statement — correcting it cost the docstring lift 20.8% and the body lift
7.1%, **asymmetric, so it biased the comparison and not merely the levels.** And nothing
carried an interval.

Gold levels fell across the board — docstring 0.155 → **0.125**, body 0.205 → **0.193**,
both re-run — and the lifts held or rose, because the null fell further. The findings
survive; the absolute agreement figures had been overstated by up to a fifth.

**The intervals are what changed the reading.** The four lifts span 0.014–0.152 and
their intervals are ±0.04 to ±0.05 wide, so `docstring` and `body identifiers,
doc-blind` (0.109 against 0.089) are **not separated by this corpus**, while `body
identifiers` against either of them is. Any claim resting on a gap smaller than about
0.05 needs a paired test, not two overlapping intervals.

**Label validity**, needing no generator at all: use each label as a search query and
see whether it retrieves the symbol it was mined from. Lexical, name-blind: **MRR
0.254, hit@5 0.405, hit@10 0.500.** So half the mined labels do not put their own
symbol in the top ten of a lexical search — those are labels whose vocabulary is about
a work package.

> **Withdrawn.** The comparison that used to sit here — "mined prose is, if anything, a
> slightly better retrieval query than a hand-written question", against a comparator of
> `MRR 0.221 / hit@10 0.435` — **reverses on re-run**: the hand set beats the mined set
> on all three measures, and the correction to name-blinding moves the mined number
> further down. On 16 and 42 items respectively neither set can support the comparison
> in either direction, so it is withdrawn rather than reversed.

### What this gold set does not establish

- **It is not a labelling of a repo, it is a 13% sample** — and a biased one. The
  symbols that get labels are the ones a commit message happened to name, which skews
  toward things that were fixed, argued about, or added late. The direction is measured
  rather than assumed — see the `bias` block quoted under [the gold
  set](#the-gold-set), which is the same mention rule applied to the same repo. Nothing
  here licenses a claim about the other 87%.
- **42 labels are not 42 independent measurements.** They come from 17 distinct commits;
  one commit supplies 9 of them. That is an input to the arithmetic rather than a caveat
  to hold in mind, and it enters at both places it can: the null is drawn cross-commit,
  so no view is ever paired with a sentence of its own commit message, and the interval
  resamples commits.
- **A mention is not a purpose statement.** For a symbol introduced by a bug-fix commit,
  the prose that names it often describes the bug rather than the symbol's standing job.
  Those labels are kept, because filtering them would take exactly the judgement the
  eval is supposed to be measuring — and they are the main reason absolute similarity
  stays low even for a good generator.
- **The label is not fully independent of the source.** One author wrote the docstring
  and the commit message in one sitting. A label found verbatim in any labelled symbol's
  source is rejected outright (3 of 45 here), and the `docstring_blind` condition costs
  the body-identifier generator 41% of its lift (0.152 → 0.089). All 42 usable-labelled
  symbols have a docstring, so this is the whole corpus, not a corner of it.
- **Token-F1 rewards vocabulary, not meaning.** It cannot tell "opens the connection"
  from "closes the connection" — there is a test pinning that. Read the *gap* to the
  control, never the score.
- **It needs fine-grained history.** Run against code-learner@`3212972` itself — 17
  commits, 715 non-test symbols — the yield is **24 labels, 3.4%**, from 8 commits, and
  the `name + signature` floor condition stops clearing zero (lift 0.017, CI [−0.003,
  0.053]). swarm-sync's 95 commits over 318 symbols buy four times the yield. This
  technique does not pay for itself on a young repo, and it is not a substitute for hand
  labelling so much as a way to get a free second opinion on a repo that has been worked
  in for a while.

```python
from pathlib import Path
from codelearner.eval import run_purpose_eval, format_report

report, cards = run_purpose_eval(Path("/path/to/some/repo"))   # read-only; git + tree
print(format_report(report, cards))
```

`to_gold_json(report, head)` dumps the mined labels in the same shape as the checked-in
gold files, including a `labelling_rule` stated in the same terms. Mined *purpose* gold
is deliberately **not** checked in: it is a function of a repo's history and goes stale
on the next commit, so it is regenerated rather than trusted. (Mined *retrieval* gold is
checked in, with the sha it was mined at in its metadata, because a retrieval gold set
has to be stable across a run to be a benchmark at all.)

## Generation: the claims, and the numbers that judge them

Everything above was built against a store filled by hand. `codelearner learn` is the
generator, and it arrives last on purpose — the gate, the staleness engine, the
negative controls and the judge all had to be measured before the thing they exist to
restrain was allowed to produce anything.

**It cites by reference number, and there is no other channel.** The model is handed a
numbered menu of spans the *index* built and answers with integers. This is the whole
design and it is worth being precise about why, because the obvious alternative looks
fine: let the model name a path and a byte range. A model asked for
`swarmsync/repolock.py[4120:4380]` produces something in that shape whether or not it
read those bytes, and an invented offset does not fail loudly — it lands inside a real
file, hashes to something stable, and verifies forever while pointing at nothing. With
integers, a citation the model invents is not a bad citation, it is **not a citation**.

That is not hypothetical. On the run below the model cited off its own menu **23 times
across 10 drafts**, and the pattern is specific: it happens on short menus. A symbol
offered one span gets cited as `[2, 3, 4, 5, 6]` — the model assumes a longer list
exists and invents entries in it. Under a byte-offset design every one of those would
have become a plausible, permanently-verifying citation to code the claim was not
about.

### Measured, on swarm-sync@`3119a97`, code-learner@`9c2e6b2`

Re-run in full at schema v6, on the corrected instruments. Every input to the numbers
previously published here had changed since they were taken — decorated spans widened,
the prompt rewritten after the signature-restatement failure, faithfulness given
intervals and separated instrument failures, the purpose eval given full-qualname
blinding, 500 cross-commit derangements and clustered intervals — so the old figures
described a system that no longer existed.

Two generators, the **same prompt byte for byte**, the same gold, and the same judge.

```
llama3.1:8b                              claude-opus-5 (via Claude Code)
admitted 148/151 (0.980)                 admitted 136/151 (0.901)
  empty_claim        0                     empty_claim       15
  no_valid_citation  3                     no_valid_citation  0
  off-menu refs     17 across 10            off-menu refs      0
227s, ~0.67 sym/s                        ~1400s, ~0.11 sym/s
```

**The failure profiles are opposite, and that is the most legible difference between
the two models.** `llama3.1:8b` never declines and invents menu references seventeen
times. `claude-opus-5` declines fifteen times and never once cites off the menu. The
lower admission rate is the prompt working: *"if the only thing you can write is the
signature, write nothing."*

The three refusals on the llama side are also worth naming, because two of them are the
same failure seen a third time: the model reciting `_next_ticket`, the **invented**
symbol from the prompt's own worked example. It still fails closed — that symbol does
not exist, so there is nothing to cite — which is why the examples were made synthetic
in the first place.

| | faithfulness | n |
|---|---|---|
| `llama3.1:8b` | 0.55 [0.47, 0.63] | 148 |
| `claude-opus-5` | **0.70 [0.62, 0.77]** | 136 |

Judged by `qwen3.5:9b` in both cases — deliberately not Claude. A generator grading its
own output shares its blind spots, and if Claude both authored and judged, faithfulness
would stop being an audit and become self-assessment.

**Read the gap with the abstention in mind.** Claude declined fifteen symbols that llama
attempted, and declining the hard ones raises the average of what remains. Some unknown
part of 0.70 − 0.55 is selection rather than quality, and this eval cannot separate them:
the honest phrasing is *0.70 among the claims it chose to make*. The instrument-failure
counters were 2/136 and 3/148 — small, and visible rather than folded into the score,
which is the whole reason they were split out.

### Purpose accuracy: a frontier model halves the gap and still loses

Scored against labels mined from commit prose both generators are structurally
prevented from seeing. **Both LLM conditions were run together, over the same 42
labels, in one pass** — that is what makes the paired test below a genuine pairing
rather than two runs subtracted.

| condition | n | gold | shuffled | lift | lift 95% CI | suspect |
|---|---|---|---|---|---|---|
| body identifiers | 42 | **0.193** | 0.041 | **0.152** | [0.107, 0.205] | 0 |
| LLM `claude-opus-5` | 42 | 0.142 | 0.027 | 0.115 | [0.074, 0.168] | 10 |
| docstring first sentence | 42 | 0.125 | 0.017 | 0.109 | [0.063, 0.171] | 0 |
| body identifiers, doc-blind | 42 | 0.119 | 0.031 | 0.089 | [0.052, 0.143] | 0 |
| LLM `claude-opus-5`, doc-blind | 42 | 0.114 | 0.023 | 0.091 | [0.053, 0.147] | 24 |
| LLM `llama3.1:8b` | 42 | 0.091 | 0.017 | 0.074 | [0.046, 0.116] | 2 |
| LLM `llama3.1:8b`, doc-blind | 42 | 0.061 | 0.014 | 0.047 | [0.019, 0.092] | 9 |
| name + signature only | 42 | 0.019 | 0.004 | 0.014 | [0.002, 0.032] | 0 |

Null: 500 cross-commit derangements, seed 20250729. Intervals: clustered bootstrap over
the 17 introducing commits, 2000 resamples, seed 20250801.

Paired differences, **resampling commits rather than labels** — nine of these labels
come from one commit, so label-level resampling would report an interval narrower than
the data supports:

```
claude  -  llama3.1:8b        +0.0515  [+0.0270,+0.0807]  *
claude  -  body identifiers   -0.0508  [-0.0774,-0.0126]  *
claude  -  docstring          +0.0172  [-0.0038,+0.0360]
llama   -  body identifiers   -0.1024  [-0.1374,-0.0580]  *
claude  -  llama (doc-blind)  +0.0527  [+0.0179,+0.0940]  *
```

Three things follow.

**A frontier model is significantly better than an 8B local one, and it survives
blinding.** `+0.052` sighted, `+0.053` doc-blind. The advantage is not docstring
copying — it holds when the docstring is stripped.

**It still loses significantly to a bag of body identifiers.** It halves the gap
(`−0.102` → `−0.051`) and cannot close it. It lands statistically level with a
docstring copier.

**That is a finding about the metric, not the model.** This file has warned from the
start that token-F1 rewards vocabulary rather than meaning; the warning was reasoning,
and this is the measurement behind it. When a much stronger model, given an identical
prompt, cannot beat *mechanically extracting identifiers from the function body*, the
metric is rewarding lexical overlap — a commit message about a symbol tends to contain
that symbol's identifiers, and the identifier bag emits exactly those. The number that
should be read from this table is the ordering and the paired intervals, not the levels.

#### The `suspect` column is not a leak detector here — it is a quality signal

`suspect` counts a rare token shared by the model's output and the held-out label but
absent from the view it was shown. Claude scores 10 sighted and **24** doc-blind against
llama's 2 and 9, and a jump like that in a leak counter is the kind of thing this
project does not wave through. All 18 flagged rows of one doc-blind run were read
individually.

**None is a leak.** Every flagged token is the ordinary English word for something the
model could see:

| token | what the view literally contains |
|---|---|
| `ascending` ×2 | `ORDER BY seq ASC` |
| `environment` ×3 | `os.environ`, in three `config.*` readers |
| `subclass` | `class BlackboardUnreachable(httpx.HTTPError)` |
| `different` | `if stored != root: raise ManagedRootMismatchError` |
| `positive` | `gate_timeout`'s positive-number validation |
| `events`, `response`, `validation` | `EventOut`, a pydantic response model |

The detector compares exact token strings, so it cannot see that *ascending* is the
English for `ASC`, or *subclass* for `class X(Y)`. The model and the committer reached
for the same word because it is the right word for a visible fact.

So the counter is measuring **vocabulary convergence with the committer, and a better
writer converges more often** — which is precisely why the stronger model scores higher
on it. It remains worth reporting, and worth reading as a prompt to go and look rather
than as evidence of a boundary failure. `assert_no_leak` is the hard gate and it is
still clean: 1,764 view × label pairs, 0 findings.

Sandbox leakage is separately ruled out for the Claude condition. The subprocess runs
with no tools at all, verified against a planted canary: the deny-list form
(`--disallowedTools`) **leaked** it in 7 turns by searching the tool namespace for an
unblocked shell, and the shipped allow-list form (`--tools ""`) blocked it in 1.

### What this run does not establish

- **151 symbols of one repo, and swarm-sync includes a `sample_repo/` fixture.**
  Several admitted claims are about `calc.add` and `calc.sub` — toy functions whose
  purpose is their name. They inflate the count and tell you nothing.
- **Two generators, one judge, one prompt, one repo.** Temperature is 0 for the local
  model, which removes the variance that is free to remove and does not make a run
  reproducible across ollama or driver versions. The Claude condition is not
  temperature-controlled at all — it goes through an agent harness — and its paired
  numbers moved by about 0.005 between two runs, which is the size of the
  nondeterminism to expect from it.
- **The faithfulness gap is confounded with abstention.** Claude declined 15 symbols
  llama attempted, and this eval cannot say how many of those llama got wrong. A
  generator that refuses the hard cases scores better on what remains, so `0.70`
  against `0.55` is an upper bound on the quality difference, not a measurement of it.
  Scoring abstentions as failures would be the other extreme and equally wrong: a
  refusal to guess is the behaviour the prompt asks for.
- **One run of the Claude condition reached its numbers on the second attempt.** The
  first aborted at symbol 98 when the harness answered with a different model. No
  claim from the substituted model was ever stored — the guard fires before the draft
  returns — but the run had to be restarted on a clean index rather than resumed,
  because 33 claims already carried verdicts and `adjudicate` only scores servable
  ones, so resuming would have quietly dropped the refuted claims and inflated the
  score.
- **`oversize=96` is a budget decision, not a defect.** Those neighbours were longer
  than `max_offer_bytes` and were dropped rather than truncated — a model that reads
  the first 4KB and cites a span covering 40KB would produce a citation that verifies.
  It sat in the same counter as `unreadable` until a run reported "96 dropped as
  unreadable" against a repo whose symbols every one hashed clean; they are now
  separate, because one is routine and the other means stop and re-index.
- **The judge and the generator must stay in different families.** `collides_with_judge`
  makes that checkable rather than remembered: pointing `--model` at a Qwen model runs
  fine, breaks no test, and silently converts faithfulness from an audit into two
  relatives agreeing. See the caveat on what "family" is actually checking, above.
