# code-learner

GraphRAG over a codebase — for agents that need to trace it, and for engineers
onboarding into it.

Point it at a repository. It parses the code into a knowledge graph of modules,
classes and functions, resolves the references between them, and (from Phase 4)
layers on *purpose* — what each piece is for — where every inferred claim cites the
source spans it came from and expires when those spans change.

**Status: early.** Ingest, storage, tier-1 name resolution, symbol-boundary
chunking, and the full hybrid retrieval pipeline -- lexical, dense, graph expansion,
RRF fusion, and optional cross-encoder reranking -- work, are tested, and are
measured against a gold set. The inference layer is not built yet.

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

The storage layer for tier 2 is built (`codelearner/assertions/`, schema v5). It is
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

### The staleness engine

`servable_assertions` re-hashes every cited byte range on every call, which is
`O(cited bytes)` per query. `codelearner/assertions/stale.py` is the two-stage version
of the same check:

1. **`stat()`.** If a cited file's `st_mtime_ns` *and* `st_size` are exactly what they
   were when that span was last actually hashed (`span_verifications`, schema v5),
   nothing is read.
2. **Full re-hash.** Runs when the stat differs, when a span has never been hashed at
   all, and whenever a caller passes `force_hash=True`.

The obvious way to make this fast is to cache the freshness verdict, and a cached
freshness verdict is the exact failure tier 2 exists to prevent — it is
indistinguishable from a real check right up until it is wrong. So every served claim
carries its own provenance instead: `checked_at` (when we looked), `verified_at` (when
the cited bytes were last genuinely hashed — *older* than `checked_at` on a fast-path
hit, and that gap is the point), `method` (`'stat'` or `'hash'`, weakest citation
wins), and `bound_hashes`. A caller seeing `method='stat', verified_at=<three days
ago>` knows precisely what it is holding.

**The fast path's limits are stated, not hidden.** It promises mtime and size are
unchanged; it does not promise the bytes are. An edit that restores the timestamp and
preserves the length gets through. There is a test that asserts that hole exists, and
`force_hash=True` closes it on demand. Relatedly, **a touch is not an edit**: `touch`
moves mtime, so stage one misses, stage two runs, the hash still matches, and nothing
is marked stale — the stat is an accelerator over the hash, never an authority beside
it. Only a hash can expire a claim.

`refresh_staleness(conn, repo_root)` sweeps every active assertion and reports counts
(`383 active, 383 still fresh, 0 expired; 36 stat, 0 read, 383 spans fast-pathed, 0
re-hashed`). The sweep is not what keeps the index honest — serve-time verification
is, and it has no window — but it reaches the claims nothing ever queries again, which
is most of them, and its `spans_hashed` vs `spans_fast_pathed` split is the only
evidence anyone gets that the fast path is working at all.

The four failure modes stay apart (`hash_mismatch`, `file_missing`, `span_truncated`,
`no_evidence`) because they call for different repairs. Serving withholds stale claims
by default and returns them only under `include_stale=True`, always labelled.

#### The number that did not go the way it was supposed to

Serving every claim in an index; median of interleaved A/B blocks; warm page cache.

| index | spans | cited bytes | always-rehash | two-stage | ratio |
|---|---|---|---|---|---|
| code-learner | 383 | 0.28 MiB | 4.97 ms | 6.91 ms | **0.72×** |
| swarm-sync | 1,100 | 0.95 MiB | 13.99 ms | 20.09 ms | **0.70×** |
| synthetic, 8 × 128 KiB | 168 | 1.01 MiB | 3.21 ms | 3.53 ms | 0.91× |
| synthetic, 8 × 256 KiB | 168 | 2.01 MiB | 4.35 ms | 3.41 ms | 1.28× |
| synthetic, 8 × 512 KiB | 168 | 4.01 MiB | 5.03 ms | 2.62 ms | 1.92× |
| synthetic, 8 × 1 MiB | 168 | 8.01 MiB | 8.65 ms | 2.60 ms | 3.33× |

On both *real* repositories the fast path is about **1.4× slower**. The premise was
wrong: sha256 over a page-cached Python file is far cheaper than assumed — re-hashing
all of swarm-sync's cited bytes costs under a millisecond — while the extra
`span_verifications` lookup and the per-span record-keeping cost a few microseconds per
span whether or not anything moved. The crossover is around 1.5 MiB of cited bytes per
query; below that, the unconditional re-hash wins.

What the table shows is the shape rather than the constant. The two-stage column is
*flat* in file size (2.6–3.5 ms across an 8× range) because it is `O(spans)`; the
re-hash column grows because it is `O(bytes)`. So the two-stage check is the one that
holds up as cited volume grows, and the only one whose cost does not depend on how the
filesystem feels that day — a page-cache miss, an NFS mount or an encrypted volume
moves the re-hash column and leaves this one alone. Both verifiers ship, a test asserts
they reach identical verdicts across every failure mode, and on a small warm repo the
unconditional one is genuinely the better choice.

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
fused with Reciprocal Rank Fusion, then optionally reordered by a cross-encoder.
Measured against a 16-query hand-labelled gold set on swarm-sync:

| configuration | recall@5 | recall@10 | hit@5 | MRR |
|---|---|---|---|---|
| lexical only | 0.427 | 0.448 | 0.500 | 0.304 |
| dense only | 0.542 | 0.635 | 0.562 | 0.407 |
| graph only (dense-seeded) | 0.188 | 0.188 | 0.250 | 0.172 |
| lexical + dense | 0.573 | 0.635 | 0.625 | 0.331 |
| lexical + dense + prefer-impl | 0.604 | 0.781 | 0.688 | 0.516 |
| hybrid, all three (default) | 0.646 | **0.802** | 0.750 | 0.453 |
| **hybrid + rerank** | **0.708** | **0.802** | 0.750 | **0.614** |
| lexical + dense + prefer-impl + rerank | 0.708 | 0.802 | 0.750 | 0.614 |
| hybrid + rerank, no prefer-impl | 0.708 | 0.802 | 0.750 | 0.613 |

The reranker is **`zeroentropy/zerank-1-small-reranker`** — 1.7B parameters,
Qwen3-based, ~3.4GB of weights, run on a 10GB RTX 3080. That is the model these rows
were produced with. `BAAI/bge-reranker-base` is wired as a fallback for machines that
cannot hold the larger one and has **not** been benchmarked here; nothing in this
table is attributable to it. Reranking is opt-in (`--rerank`) and its absence is not
an error — without it the pipeline returns the fused rows above.

Six findings worth more than the final number.

**The graph modality made things worse before it made them better.** At its first
guessed weight of 0.6, the full hybrid scored **0.385** recall@5 — well below
lexical+dense alone at 0.573. The weight sweep is monotonic: `0.3 → 0.646`,
`0.6 → 0.615`, `1.0 → 0.552`, `1.5 → 0.354`. Graph expansion has no query
representation, so every vote it casts is evidence about the *code* rather than
about the *question*; past a low weight those votes displace better-matched answers.
The default is 0.3 because that is what measured best, and a test pins the constant
so it cannot drift without re-running the ablation.

**Recall and ranking pull in opposite directions.** Adding graph raises recall@5
(0.604 → 0.646) and hit@5 (0.688 → 0.750) while *lowering* MRR (0.516 → 0.453). It
finds code the text modalities missed, then dilutes the top of the ranking. That
trade is documented rather than tuned away, and it is what Phase 3b set out to fix.

**Reranking is the largest lever measured so far, and it lands where the diagnosis
said it would — but smaller than first reported.** A cross-encoder reads the query
and one candidate *together*, the one thing neither RRF (which sees only positions)
nor the bi-encoder (which sees the two texts separately) can do. It moves MRR
0.454 → **0.614**, clearing the 0.516 that lexical+dense+prefer-impl reached before
the graph modality started diluting it. Retrieval widens the candidate set to
`k × 4`; the reranker reorders it.

> **These rows were corrected.** The reranking numbers first published here
> (0.750 / 0.781 / 0.875 / 0.679) did not reproduce. Re-running the ablation four
> times — two independently built indexes, two repeats each — gave identical results
> every time, and they are the rows above. The likely cause is that the original
> measurement predates a later change to what text the reranker is shown. Two of the
> three original claims do not survive the correction: **recall@10 does not fall**
> (0.802 → 0.802) and **hit@5 does not improve** (0.750 → 0.750). What survives is
> the MRR gain, which is still the largest in the project. A number that cannot be
> reproduced against the code shipped beside it is not a measurement, and correcting
> it in place is cheaper than being wrong in public.

**Reranking bought ranking, not recall.** recall@10 is unchanged at 0.802 and hit@5
unchanged at 0.750; only the ordering within those hits improved. That is the
expected shape — reranking reorders a fixed candidate set and cannot manufacture
recall — but it is worth stating plainly, because "biggest lever in the project" and
"found no additional code" are both true of the same change.

**The result that was not the hypothesis: reranking did not vindicate graph
expansion.** With the reranker on, turning graph expansion *off* scores identically —
0.708 / 0.802 / 0.750, and 0.614 vs 0.613 MRR. That tie was suspicious enough to
check per query rather than report, and the check says it is real and not a stuck
flag: graph expansion contributed **436 symbols** the text modalities never returned,
and **5 of the 16** reranked top-tens genuinely differ between the two
configurations, yet **zero** queries change whether a gold-labelled symbol is in the
top ten. It changes *what* comes back and never whether the right answer is present.
So "graph widens, the cross-encoder reorders" is half confirmed. The reordering earns
its cost. The widening, on this gold set, does not demonstrably earn anything — and
the modality stays in at weight 0.3 pending a gold set large enough to say otherwise,
not because this measurement defended it.

**The biggest single lever was not a modality at all** — until the reranker took
half its job. Demoting test code moves recall@10 from 0.635 to 0.781, more than
adding an entire retrieval modality. Both text modalities systematically rank tests
above the implementations they exercise, because a test states the behaviour in
prose, names it in the function title, and repeats the vocabulary; the
implementation just does it. With reranking on the demotion becomes redundant —
0.708 / 0.802 / 0.750 and 0.614 vs 0.613 MRR with it or without — because the
cross-encoder makes the same judgement from the query itself rather than from a path
convention. It stays on because it costs nothing and still carries the unreranked
path, which is the default when no reranker is installed.

*Caveat that limits all of the above:* the gold set is 16 queries, hand-labelled by
the author, and every one is of the form "how does X work" — the exact shape the
test demotion helps. Differences of one or two points are noise. This is enough to
tell a modality that works from one that does not, and not enough to justify
fine-grained tuning. It also bounds the reranking result specifically: a +0.226 MRR
swing is far outside that noise band and is safe to believe, while the recall@10
decline and the graph-with-rerank tie are both inside it, and either could reverse
on a larger set.

## Faithfulness: does a claim follow from what it cites?

The assertion store checks that every cited span exists and still hashes to what was
cited. That is arithmetic, and a model cannot argue with it. What it cannot check is
whether the span *supports* the claim — a citation can be perfectly present,
perfectly unedited, and completely silent on what the claim asserts.
`codelearner.eval.faithfulness` measures that, and it is the only measurement in this
repo that needs a model to produce it.

**The judge is a different model family from the generator, on purpose.** Claude
writes the assertions through `submit_assertion`; `qwen3.5:9b` judges them. That is a
methodological requirement, not a cost saving — a generator grading its own output
shares its blind spots, its tokenizer, its training distribution and its particular
way of being confidently wrong, so agreement between the two measures consistency
rather than truth. The number is only worth reading because the thing producing it is
not the thing being measured.

**The judge is prompted to refute, and every unclear path lands on "not supported".**
An LLM asked "does this follow?" will say yes to almost anything plausible, and a
permissive judge is worse than no judge: it converts an unmeasured risk into a number
that reads like a guarantee, so everything downstream then trusts the claim *more*
than before anyone looked. So the task is inverted — the burden of proof is on the
claim, silence in the evidence is a failure rather than a neutral outcome, and an
unparseable answer, an empty answer, or a claim with no spans left to read all fail
closed. The one failure that does *not* fail closed into a verdict is an unreachable
judge: recording "uncertain" because ollama was not running would reject every
assertion in the store and log a reason that blames the claims.

Verdicts go through the existing `store.record_verdict`, which already owns the policy
(one unsupportive verdict rejects; rejection is a state, never a delete). The judge's
three answers map onto the store's existing vocabulary rather than extending it:
`supported` → `supported`, `not_supported` → `refuted`, `uncertain` → `unsupported`.
"The evidence says otherwise" and "the judge could not tell" are different facts and
both stop a claim being served.

The score is **RAGAS-style faithfulness** — the fraction of claims a judge can support
from their own retrieved context. RAGAS decomposes an answer into atomic statements
first because an answer is prose; here that already happened upstream, so one
assertion *is* one atomic claim and its `evidence_spans` *are* its context. Over an
empty set the score is `None`, not 1.0: "every claim was supported" is trivially true
of no claims, and this repo has already been bitten once by a vacuous truth reading as
success.

### Measured

16 assertions about code-learner's own committed source, authored by Claude from the
code, each citing exactly one span (the enclosing symbol). Judged by `qwen3.5:9b` at
temperature 0, thinking off, output constrained to a JSON schema. 48s for the whole
pass on a shared 10GB RTX 3080 — about 3s per claim, which is cheap enough to audit a
whole store rather than a sample.

```
faithfulness 0.625  judge=ollama/qwen3.5:9b  n=16 supported=10 not_supported=6 uncertain=0
```

Three consecutive runs returned the same score and the same per-claim labels. Claude
had pre-labelled its own 16 claims before the judge ran — 11 expected to be supported,
five written to be refusable — so the honest headline is not 0.625 but **15/16
agreement with a label set the judge never saw**. A judge that approves everything
scores 1.000 and agrees on 11/16; one that refuses everything scores 0.000 and agrees
on 5/16. Reporting only the score cannot tell any of those apart.

The single disagreement is the useful one. Claude claimed `record_verdict` "moves the
assertion to 'rejected' only when it is currently 'active'", which is true, and the
judge refused it:

> the evidence shows `record_verdict` [...] explicitly checks for `STATUS_ACTIVE`, but
> it does not provide any information about what happens when the assertion is in a
> different state [...] so we cannot confirm the claim [...] without assuming
> behavior outside the provided code.

The judge is right and the *citation* is wrong. The SQL that enforces this
(`_TOUCH_STATUS`, with its `WHERE id = ? AND status = ?`) is a module-level constant
outside the cited symbol's byte range, so the claim genuinely rests on evidence it did
not cite. That is exactly the failure the metric is for, and it is invisible to the
hash gate — every hash matched.

**A bug this found by being run for real.** On the first pass one claim scored
`uncertain` because the judge, quoting `embed.serialize` back at itself, emitted
`{"verdict": "supported", "reasoning": "... the format string `f"{len(values)}f"` ..."}`
— an unescaped double quote inside its own JSON, because the code it was quoting
contains one. Invalid JSON, verdict lost, failing closed as designed and giving the
wrong diagnosis: the claim was blamed for a judge-side transcription bug. The bias is
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
- **The judge is not an oracle.** It is one 9B model, with its own error rate, and it
  will credit a docstring as evidence for a claim about behaviour if the prompt lets
  it. `report.unfaithful` exists so a low score is read by looking at the claims that
  failed, not by trusting the number. Temperature 0 removes the variance that is free
  to remove; it is not determinism, and the same claim has been observed flipping
  label between prompts that differed only in whitespace around the span.
- **Anything about coverage.** A store with three easy claims in it can score 1.000.
  Faithfulness is a property of the claims that exist, not evidence that the
  interesting ones were made.

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
That cost is the reason the purpose layer has no accuracy number.

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
  full label × view cross product: **1,764 pairs on swarm-sync, 0 findings.**
- `suspect_tokens()` checks the other direction — a rare word present in the answer and
  in the generator's output but nowhere in its input. Zero across every condition.

Both gates are wired into `score_purposes`, and both wires are tested by forging a view
at the seam, because a guard that is never reached reads exactly like a working one.

### Measured, on swarm-sync (93 commits, 316 non-test symbols)

**The technique works, and it is expensive: 13.3% yield.** 42 usable labels; 272 symbols
rejected because the commit that introduced them never names them, 2 because the label
was copied verbatim into the docstring and so is not held out.

The prior going in was that commit messages would be too *low-quality* — "fix", "wip",
"address review". On this corpus that prior was simply wrong: the boilerplate filter
rejected **zero** symbols, and the median commit body is 1,021 characters of careful
prose. The problem is **attribution**, not quality. 163 symbols (52%) trace to one
initial commit whose entire message is a 602-character project summary, and the rest
come from work-package commits touching 2–57 files (median 6) whose prose describes a
change, usually several. Excellent prose about a work package is not a purpose statement
about a symbol.

Two independent checks that the surviving labels are about their symbols:

| condition | n | gold | shuffled control | lift |
|---|---|---|---|---|
| docstring first sentence | 42 | 0.159 | 0.023 | 0.136 |
| name + signature only | 42 | 0.020 | 0.002 | 0.019 |
| body identifiers | 42 | 0.208 | 0.047 | 0.161 |
| body identifiers, docstring-blind | 42 | 0.124 | 0.035 | 0.089 |

Name-blind token-F1 against a deranged control. Every condition clears its control by
4–7×, so the labels carry symbol-specific signal. Swapping token-F1 for
`Qwen3-Embedding-0.6B` cosine reproduces the ordering exactly (0.631/0.427,
0.451/0.416, 0.675/0.480, 0.557/0.420) — two unrelated similarity measures ranking four
conditions the same way is evidence about the labels rather than about either metric.

Second check, needing no generator at all: use each label as a search query and see
whether it retrieves the symbol it was mined from. Lexical, name-blind: **MRR 0.288,
hit@5 0.452, hit@10 0.500.** For scale, the hand-labelled retrieval gold set scored the
same way on the same modality reaches MRR 0.221 / hit@10 0.435 — on the *easier*
criterion of several acceptable symbols per query.

### What this gold set does not establish

- **It is not a labelling of a repo, it is a 13% sample** — and a biased one. The
  symbols that get labels are the ones a commit message happened to name, which skews
  toward things that were fixed, argued about, or added late. Nothing here licenses a
  claim about the other 87%.
- **42 labels are not 42 independent measurements.** They come from 17 distinct commits;
  one commit supplies 9 of them. Treat a few points as noise, as the ablation says of
  its 16 queries.
- **A mention is not a purpose statement.** For a symbol introduced by a bug-fix commit,
  the prose that names it often describes the bug rather than the symbol's standing job.
  Those labels are kept, because filtering them would take exactly the judgement the
  eval is supposed to be measuring — and they are the main reason absolute similarity
  stays low even for a good generator.
- **The label is not independent of the source.** One author wrote the docstring and the
  commit message in one sitting, so a docstring-reading generator scores partly on
  shared authorship. Verbatim copies are rejected outright and a `docstring_blind`
  condition is reported (it costs the body-identifier generator a third of its lift,
  0.161 → 0.089), but correlated phrasing cannot be filtered away. All 42
  usable-labelled symbols in swarm-sync have a docstring, so this is the whole corpus,
  not a corner of it.
- **Token-F1 rewards vocabulary, not meaning.** It cannot tell "opens the connection"
  from "closes the connection" — there is a test pinning that. Read the *gap* to the
  control, never the score.
- **No generator is measured here yet.** The three shipped generators are baselines that
  read source deterministically: a docstring copier (the upper reference — it relays
  documentation rather than inferring anything), a name echo (the floor), and a bag of
  body identifiers. What the harness establishes is that the *measurement* has
  resolution, which is a precondition for evaluating a real generator, not a result
  about one.
- **It needs fine-grained history.** Run against code-learner itself — 7 commits, 231
  symbols — the yield is **3 labels, 1.3%**. This technique does not pay for itself on a
  young repo, and it is not a substitute for hand labelling so much as a way to get a
  free second opinion on a repo that has been worked in for a while.

```python
from pathlib import Path
from codelearner.eval import run_purpose_eval, format_report

report, cards = run_purpose_eval(Path("/path/to/some/repo"))   # read-only; git + tree
print(format_report(report, cards))
```

`to_gold_json(report, head)` dumps the mined labels in the same shape as
`gold/swarm_sync.json`, including a `labelling_rule` stated in the same terms, so the
two gold sets can be read side by side. It is deliberately **not** checked in: a mined
set is a function of a repo's history and goes stale on the next commit, so it is
regenerated rather than trusted. The two sets barely overlap in any case — only 5 of the
hand set's 23 symbols got a mined label, which makes them complementary rather than
alternative.

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
| `--rerank` | reorder with a cross-encoder that reads the query (opt-in; downloads ~3.4GB on first use, and says so and answers anyway if it cannot load) |
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
rather than another guess.

None of which is worth believing on the strength of a paragraph describing it.
`codelearner.eval.gate_controls` generates an adversarial corpus from what the index
actually holds — nine distinct attacks, seven of them instantiated per symbol and two
per file, from zero citations through a hash that was correct until the file changed
under it — and reports a rate. Measured on this repository, 850 symbols:

| | |
|---|---|
| attacks submitted | **6,091** |
| refused | **100.00%**, every one by the rule it targets, none leaving a row behind |
| correct citations submitted | **1,688** |
| admitted | **100.00%** |

The second pair of rows is not decoration. A gate that refuses everything scores a
perfect rejection rate, so the corpus also submits every symbol cited by the hash this
index published for it and by its exact lines quoted off disk, and those must be
admitted — including the 217 of 850 symbols (25.5%: every method, every module, and
the decorated functions and classes) whose stored bytes are *not* their lines' bytes,
which a narrowed gate would falsely reject while still refusing every attack.

Each of the twelve control families is verified by deleting the rule it names from a
copy of the package and confirming the attack then succeeds — 12 of 12 detected. A
control that cannot see its own rule removed is decoration, and this is the project
that has already shipped three tests which passed while asserting nothing.

The controls found a real hole on their first run, which is the argument for writing
them: a claim whose spans hash-matched perfectly but whose `subject_qualname` named no
indexed symbol was **admitted** — stored `active`, reported `servable`, and then
unreachable forever, because `get_symbol` answers `no_such_symbol` for the only name
that would find it. Verified evidence had made an unciteable claim indistinguishable
from a good one, which is the exact failure the zero-evidence rule exists to prevent,
reached through the other door. It is refused now as `unknown_subject`.

```bash
.venv/bin/python -m codelearner.eval.gate_controls --repo .      # the corpus, at scale
.venv/bin/python -m codelearner.eval.gate_controls --mutations   # the controls on it
```

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
shared index. From a bare checkout, before anything is installed, point `command` at
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

### The tools

| tool | what it returns |
|---|---|
| `search_code(query, k, facts_only)` | hybrid retrieval — lexical + dense + graph, RRF-fused. Tier-labelled hits with qualname, `path` and line range, the modalities that found it, `via` (the account of how graph expansion reached it), and the `content_hash` needed to cite it. `facts_only` drops tier 2. |
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

`submit_assertion` refuses, and says why:

| code | when |
|---|---|
| `evidence_required` | zero evidence spans. An uncited claim cannot be adjudicated, cannot expire, and cannot be checked by a reader — it is indistinguishable from a good one at every stage after this, so the only place to stop it is the door. |
| `hash_mismatch` | the cited bytes no longer hash to what was cited. Returns `observed_hash` and `observed_text`, so the citation can be corrected rather than re-guessed. |
| `evidence_unverifiable` | a span with neither hash nor text. A location that asserts nothing about what is there can never be found to be wrong. |
| `bad_range` / `file_missing` / `path_escapes_repo` | the citation does not point at readable bytes inside this repo. |

One bad span refuses the whole submission. Admitting the ones that happened to verify
would leave a claim standing on a subset of the evidence its author thought it had.

A line range has two honest readings — the symbol that occupies those lines, and the
whole lines themselves — and they hash differently. A symbol's stored bytes begin at
`def` rather than in the indentation before it, at the `@` for a decorated symbol,
and run to the last byte of the file for a module. Measured on code-learner itself,
that is 85 of 383 symbols. Both readings are built and both are re-hashed off disk,
and the cited hash may match either. That is not a loosening: every candidate is read
from the file as it is right now, so a stale or invented hash still matches nothing.
What it removes is a false rejection — the more dangerous failure here, because an
agent told that its correct citation is wrong learns that the gate is noise.

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

## Roadmap

| Phase | | Status |
|---|---|---|
| 0 | Spike on a real repo | done |
| 1 | Ingest + store + tier-1 resolution | done |
| 2 | Symbol-boundary chunking + FTS5 lexical index | done |
| 2b | Dense embeddings (`Qwen3-Embedding-0.6B`) into sqlite-vec | done |
| 3 | Hybrid retrieval: RRF fusion + graph expansion | done |
| 3b | Cross-encoder reranking (`zerank-1-small`) | done |
| 4 | Assertion pipeline + adversarial gate | |
| 5 | Staleness engine | |
| 6 | Onboarding tours | done |
| 7 | MCP server + CLI | done |
| 8 | Eval: per-modality ablation (done), faithfulness scoring with a cross-family judge (done), git-history purpose gold (done, 13% yield), gate negative controls (done — **6,091 attacks refused, 100.00%**; 1,688 correct citations admitted, 100.00%; 12/12 controls mutation-verified) | partial |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 369 tests
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

The reranked rows are opt-in for the same reason the CLI flag is — they cost a model
forward pass per candidate, roughly 640 per row, and the modality rows must stay
runnable on a machine with no GPU:

```python
from codelearner.retrieve import load_reranker

reranker = load_reranker(conn=conn)   # None if no model can be loaded
print(format_table(run_ablation(conn, SentenceTransformerEmbedder(), reranker=reranker)))
```

`load_reranker` returns `None` rather than raising when there is no model, no
network, or no memory, and `run_ablation` and `search` both read `None` as *skip the
stage*. On the 10GB RTX 3080 the full table takes 167s; the reranker needs ~3.5GB of
it and will fall back to CPU if the card is busy, which is correct but far too slow
to benchmark with.
