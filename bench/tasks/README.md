# Benchmark task set

Three arms (`bare`, `codegraph`, `code-learner`) answer the same questions about the
same repositories. The task set exists to separate two things that are usually
measured as one:

- **LOCATE** — *where is X, what calls Y, what breaks if I change Z.* Answerable from
  structure alone. This is the half `codegraph` is built for and it is the fair
  baseline: both indexed arms should do well, and if `code-learner` does *worse* here
  that is a real cost of its extra machinery, not noise.
- **EXPLAIN** — *why does this code take this form rather than the obvious one, and
  what failure is it defending against.* Answerable only by understanding what the
  code is **for**. These are the tasks that test the thesis. If tier-2 claims are
  worth their cost this is where it shows; if they are not, this is where that
  becomes visible.

Counts:

| repo | locate | explain |
|---|---|---|
| swarm-sync | 11 | 0 |
| kalshi-bot | 12 | 1 |
| facefusion | 0 | 10 |
| **total** | **23** | **11** |

LOCATE and EXPLAIN are sourced from **different repositories**, for a reason given
below. Nothing may average a LOCATE score and an EXPLAIN score into one row: they are
not measurements of the same corpus.

## Files

- `locate.json` — 23 structural tasks over swarm-sync and kalshi-bot.
- `explain.json` — 11 rationale tasks, 10 over facefusion and 1 over kalshi-bot.
- `validate.py` — checks every claim this directory makes. Run it before any run.

The harness reads `id`, `question`, `repo` and `ground_truth`; every other field
lands in `Task.extra` and is carried into each result verbatim.

## Ground truth

**LOCATE** ground truth is a set of qualnames. Every one was validated against the
repo's `symbols` table, and every *caller* and *impact* set was additionally
**grep-verified in the working tree** rather than taken from the index. That
distinction matters: the index under test is `code-learner`'s own, and using it as
ground truth would score one arm against its own opinion. Where a set was derived,
the `verified_by` field records the grep and the line numbers it returned.

**EXPLAIN** ground truth is **commit prose** — a sentence the committer wrote in a
place neither system indexes. `committer_prose` is quoted verbatim and `provenance.commit`
records the sha, so any reader can run `git log -1 <sha>` and check it. This is not
circular: both systems index *code*; the ground truth is prose written elsewhere. A
system that can answer from code alone what the committer explained in a message is
demonstrating exactly the understanding under test.

Mining reused `codelearner/eval/gold_from_history.py` rather than reinventing it:
`strip_trailers`, `is_boilerplate`, `split_units` and `mentions_symbol` for the
funnel, and `find_leaks` for the copy check.

### The discipline that makes an EXPLAIN task valid

1. **The answer must not be greppable.** If the reason is written in a docstring or a
   comment in the file, the task tests retrieval and both arms will find it. Checked
   two ways: `find_leaks` over the anchor file (verbatim clause), and a `reason_terms`
   conjunction over **every tracked file** including markdown and config prose.
2. **The question must not leak its own answer's vocabulary.** Checked, and reported
   with the overlap named rather than hard-failed, because some overlap is
   unavoidable domain vocabulary. For LOCATE the check subtracts the decoys'
   vocabulary first: a token shared by the target *and every decoy* selects nothing
   and is not a leak.
3. **The commit sha is recorded per task**, so the ground truth is auditable.

### Rubric format

Each EXPLAIN task carries two or three numbered points an answer must contain.
"Does this answer seem good" is not a measurement. The judge is `qwen3.5:9b`,
deliberately not a model under test.

Points are deliberately **graded**, and each task labels which is which:

- `inferable_points` — reachable by a good reader of the code alone. These are what a
  structure-only arm can plausibly get.
- `historical_points` — recoverable only from what the committer knew. These are what
  the thesis claims stored purpose should help with.

That split is the whole point. If `code-learner` beats `codegraph` only on
`inferable_points`, the lift is better navigation, not stored understanding — and the
thesis is not supported by this evidence.

## Decoys

Nine LOCATE tasks include `decoys`: symbols whose name is the obvious keyword match
but which are the wrong answer. Examples: `RepoLock.release` against two other
`release` methods; `constants.get_baseline_error_f` against two stale same-named
copies carrying different numbers; `financial_settler._parse_expiry_date_from_ticker`
against five other ticker parsers. Every decoy is validated to exist — a decoy that
does not resolve is not a decoy, and the task would be easier than its metadata
claims. `validate.py` caught exactly that mistake once (`integrator.run_impact_tests`
is a module-level alias, not an indexed symbol; it is now described as such rather
than listed as a symbol).

## Why EXPLAIN moved to facefusion

This is the substantive finding of building the set, and it should be published, not
summarised away.

The original plan sourced EXPLAIN from swarm-sync and kalshi-bot. It failed. Twenty
hand-picked rationale candidates were checked against the working tree, and **16 were
rejected because the reason is already written down somewhere** — a docstring, an
inline comment, a design document, a test docstring, or a `note` field in a config
JSON file.

| repo | candidates checked | rejected as greppable | surviving |
|---|---|---|---|
| swarm-sync | 8 + 15 = 23 | 21 | 2 (both narrow slices) |
| kalshi-bot | 12 + 12 = 24 | 22 | 2 (both partial) |
| facefusion | 24 screened | 14 (anchor gone at HEAD, or no anchor in indexed code) | 10 |

The cause is measurable, not anecdotal:

| repo | symbols | with docstring > 80 chars | mean docstring length | comment lines / py lines |
|---|---|---|---|---|
| swarm-sync | 1,270 | 46% | 534 chars | 2,628 / 30,600 |
| kalshi-bot | 7,191 | 16% | 155 chars | 11,321 / 152,421 |
| facefusion | 1,196 | **0%** | — | **2 / 20,393** |

swarm-sync and kalshi-bot are written in the same house style as code-learner
itself: long argued docstrings that state the failure being defended against.
swarm-sync goes further and keeps a `MEASURED_*` block expressly so empirical numbers
survive in the working tree. They are close to the *worst possible* sample for
testing whether a system can infer why from code, because the why is always already
written down.

facefusion is the opposite and the reason it was chosen: **221 tracked Python files,
20,393 lines, zero docstrings and two comment lines in the entire package.** The why
cannot be in a docstring there. Its commit history carries the rationale in a
compressed "do X to avoid Y" form — *"Avoid RGB to YUV colorshift using libx264rgb"*,
*"prevent directml using incompatible corridor_key model"*, *"Use semaphore to
prevent frame colorizer memory issues"* — which is precisely the class this benchmark
needs.

**facefusion must be indexed for the codegraph arm too.** It is already indexed for
code-learner at `/home/keith/projects/facefusion/.codelearner/index.db` (1,416
symbols, 12,842 edges, 0 docstrings).

### What this implies for the result, before any run

If mature, well-documented repositories usually record their reasoning, then a system
that *infers* why from code is competing with documentation that already exists. Its
value would concentrate in undocumented or poorly documented code. That is a scoping
result about where tier-2 claims can pay, and it belongs in the write-up as such.
It also bounds what this task set can conclude: the EXPLAIN half measures the
undocumented case only, because that is the only case in which the measurement is
meaningful.

## Running the validator

```
.venv/bin/python bench/tasks/validate.py
```

Exit status 0 only when there are no errors. It checks schema and id uniqueness;
that every ground-truth qualname, decoy and anchor resolves; that every EXPLAIN
commit exists and really contains the quoted prose; that no prose is copied into its
anchor file; that no single tracked file contains all of a task's `reason_terms`; and
that questions do not hand over their answers' vocabulary.

`NOTE` lines are evidence, not complaints: `reason_terms 0/2 occur anywhere in tree`
is the strongest possible pass — the committer's words for the reason were never
written into the repo at all. A `reason_term` must be reason-bearing English, not an
identifier the anchor contains by construction, or the check proves nothing.
