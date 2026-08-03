# Remediation plan — audit of 2026-08-01

Six independent read-only audits at `93a4084` (architecture, correctness, eval methodology,
completeness, test quality, security). This is the implementation plan derived from them.

---

## Status

| WP | State | Landed |
|---|---|---|
| WP1 git-config RCE | **done** | `31e6c97` |
| WP2 file-read oracle | **done** | `31e6c97` |
| WP3 transport + unbounded writes | **done** (schema CHECK deferred to WP8) | `31e6c97` |
| WP6 pin `Outcome.held` | **done** | `31e6c97` |
| WP4 consolidate the gate | **done** | `72e5a0c` |
| WP19 remaining unguarded reads | **done** | `72e5a0c` |
| WP7 re-index carries the store | **done** | `243095d` |
| WP10.1 `SchemaVersionError` handling | **done** | `243095d` |
| WP10.3 non-destructive expiry + `reinstate` | **done** | `243095d` |
| WP5 controls at both doors | **done** | `243095d` |
| WP20 repo containment at the chokepoint | **done** (found by WP5) | `243095d` |
| WP8 decorator span + schema v6 | **done** | `2f299c8` |
| WP9 MCP index invalidation | **done** | `2f299c8` |
| WP10.2 tier-0/1 drift detection | **done** | `2f299c8` |
| WP21 pre-v6 narrowed citations on carry | **done** (found while verifying WP8) | `7df16d6` |
| WP12 leak boundary wired into the scored run | **done** | `1021a14` |
| WP13 statistical corrections to the purpose eval | **done** | `1021a14` |
| WP14 gate numbers at their real resolution | **done** | `1021a14` |
| WP15 faithfulness denominator, interval, instrument | **done** (apparatus built, not run — see below) | `1021a14` |
| WP22 gold set + power analysis (**added post-Wave-4**) | **done** | `3212972` |
| WP11 withdraw or correct the numbers | **done** — see the row-by-row disposition below | this commit |
| WP18 documentation and packaging drift | **partial** — docs done, packaging and the Python-only guard are not | this commit |
| WP16 close the surviving mutations | **partial** — see below | |
| WP17 architecture: one rule, one place | **not started** — all seven items open | |

Test count: 484 at audit time → 631 at `243095d` → 663 at `2f299c8` → 673 at `7df16d6`
→ 753 at `1021a14` → **844** at `3212972`. Re-run against a clean tree at `3212972`
while writing WP18: 844 passed, `ruff check .` clean, `mypy codelearner
--ignore-missing-imports` clean over 45 source files.

*(The same command run later in the same session reported **852**, because other WPs were
in flight in the working tree. 844 is the number at the sha; 852 is a number about an
uncommitted tree and belongs to whichever commit lands it. Stating which is which is the
same discipline as stamping a table, applied to the one figure this document quotes most
often.)*

### Amendments from Wave 0

Three things the plan got wrong or missed. Recorded here rather than silently corrected,
because a remediation plan that quietly rewrites itself is the same failure as a measurement
that does.

1. **WP2's refusal code — the plan was wrong.** It specified a distinct `file_not_indexed`
   code. The implementing agent reused `file_missing` instead, on the grounds that a distinct
   code *preserves the oracle in reduced form*: submit `.env` and `.envv`, and the difference
   between the two codes answers "does this file exist on disk". Verified after the fix —
   both return byte-identical refusals. Take the narrower reading of "close the oracle" as
   the standing rule for the rest of this plan: **a refusal must not distinguish two states
   the caller is not entitled to tell apart.**

2. **WP6 understated the defect by half.** The audit found 3 of `held`'s 4 conjuncts
   deletable with the suite green. The implementing agent found **6 of 7**. The three the
   audit missed are masked: an accepted outcome carries `code=None`, which is in no family's
   code set, so the negative branch's `verdict == REFUSED` conjunct cannot be observed
   failing by any corpus run. These were not dead code — they were **unreachable by the
   corpus that scores them.** Generalise the lesson: the negative-control apparatus can only
   ever test itself against situations it already knows how to generate, so its coverage of
   its own scorer must be established by construction, never by running it.

3. **Defence in depth breaks single-edit mutations.** Adding three guards ahead of the
   `absent_file` read meant the family's mutation — which deleted one guard — no longer
   flipped the control, so it silently stopped detecting its own rule. Fixed by extending the
   mutation to remove all three. **Any WP that adds a guard in front of an existing
   controlled rule must re-check that rule's mutation still flips it.** This applies directly
   to WP4.

   *This has now bitten three times, in three distinct shapes, and each would have reported
   a control as detecting a rule it does not name:*

   | shape | family | how it hid |
   |---|---|---|
   | **duplicated** | `unknown_subject` (WP4) | the store's copy of the rule refused with the same code |
   | **bypassed** | `unverifiable_span` (WP5) | the server substitutes the observed hash, so the store's rule is unreachable by construction |
   | **stacked** | `escaping_path` (WP20) | deleting containment drops the attack onto the index-membership guard, and the two cannot be separated by *any* submission — every path that escapes the repo is by construction a path the index never parsed |

   Treat the rule as standing procedure, not a Wave-0 note.

### Amendments from Wave 2

4. **A declared gap is a debt the suite makes you settle.** WP5 recorded the missing
   containment rule as an `Unenforced` gate entry — controls still generated, still
   submitted, still scored as the failures they were — with `STORE_GAPS` asserted as an
   exact set. Closing the hole then *failed a test*, which is the behaviour that entry was
   built for. Keep the empty set asserted: an `Unenforced` entry added without editing that
   line now fails.

5. **Adding a measurement surface finds what reading the code does not.** Six auditors read
   `write_assertion` and none found the containment hole. Running the *existing* corpus at a
   *second door* surfaced it immediately. But note the honest limit, which WP5 stated
   itself: the corpus still found no attack nobody had enumerated. Two auditors found two by
   probing outside the family list; this apparatus has found none. Adding doors is a second
   axis with the same property — it can only expose rules some family already names.

6. **Counters must be wired, not merely declared.** During WP20 the synthesiser added a
   `refused_escaping_span` field and a term in the documented partition identity, but did not
   add it to `_OUTCOME_COUNTERS`. It would have read zero forever while the draft vanished
   out of `drafts_requested` — breaking the identity on the first such refusal and on no run
   before it. A counter wired to nothing is worse than a missing one.

### Amendments from Wave 3

7. **The plan's own prevalence figure was wrong, and wrong in an instructive direction.**
   WP8 asked for the symbol-bytes-vs-line-bytes population to be re-measured and predicted
   the decorator fix would move it. It does not move it by a single symbol. Re-measured at
   `3212972` on 1,714 symbols: 438 disagree (25.6%) — every method, every module, and 44
   functions and classes. Cross-tabulating those 44 against decoration and nesting gives an
   exact answer in both directions: **disagreeing is precisely being nested** (44 of 44 both
   ways), and 138 decorated top-level functions and classes disagree not at all. A top-level
   `@` sits in column 0 exactly where its `def` did, so this is a property of *indentation*
   and always was. The old attribution to "the decorated functions and classes" was a
   plausible story about a number nobody had cross-tabulated. **Generalise: a population
   figure quoted beside a causal attribution needs the cross-tab, not the marginal.**

   *(The Wave-3 commit reported 39 nested functions/classes out of 325 of 1,314 and said
   "not one is decorated". At `3212972` it is 44 of 438 of 1,714 — the repo grew — and six
   of the 44 **are** decorated. They are also all nested, and nesting is what explains them,
   so the finding is unchanged and the phrasing "not one is decorated" was too strong. The
   defensible statement is the cross-tab.)*

8. **Fixing a rule at the publishing seam does not reach what has already been published.**
   WP8 widened the span; WP7 carries the store across the rebuild WP8 forces; and neither
   owned the gap between them, which is that a carried claim keeps its pre-v6 citation over
   bytes that did not change. WP21 was found only by going looking for it *after* both had
   landed and been reviewed. **Any WP that changes what an artefact means, rather than what
   it contains, must be checked against artefacts already in flight** — and the residue that
   is still out of reach (citations an agent has already been handed) has to be stated, not
   closed.

9. **A staleness reason is part of the vocabulary and is chosen, not defaulted.** `hash_mismatch`
   was the available reason for WP21's expiries and would have been wrong: the bytes did not
   change, the citation *boundary* did, and a reader debugging the two needs to tell them
   apart. `decorators_excluded` was added instead. The generalisation is that a reason code
   reused because it was nearest to hand is the same defect as a refusal code reused because
   it was nearest to hand — the amendment-1 rule, applied to expiry rather than admission.

### Amendments from Wave 4

10. **A concession can be a way of not fixing something.** The README conceded that
    "correlated phrasing cannot be filtered", and that concession was true and did not
    describe the actual finding. What `audit_leak_boundary` had been finding was a
    32-character clause **copied verbatim across a symbol boundary** — not correlated
    phrasing, a copy, and a copy is exactly the thing `find_leaks` could already see. The
    concession was covering a filterable failure with the language of an unfilterable one.
    **Treat a standing caveat as a claim that expires: any caveat that has been in the
    document longer than the code it describes should be re-derived, not re-copied.**

11. **The plan was wrong about the leak filter's scope, and the agent was right to say so.**
    WP12 specified rejecting a label whose clause appears in a sibling's source *within the
    same commit*. `_AffectedFiles` was introduced by `982386a` and `_reverse_dep_files` by
    `d6e029a`, so a same-commit filter misses the only real leak in the corpus. Filtered
    across all labelled siblings regardless of commit, with the cross-commit property pinned
    in a fixture so nobody re-scopes it back.

12. **`1,764 pairs, 0 findings` is now true, and it is true for a different reason than it
    was published for.** It was published as a property of the corpus. It is a property of a
    filter that runs, plus a caller that invokes it, plus a run that aborts if it fires — and
    before WP12 the caller did not exist. **A number that was right by luck and a number that
    is right by construction read identically. Publishing the mechanism beside the number is
    the only thing that separates them.**

13. **Removing a stale figure beats correcting it, when the medium cannot be re-measured.**
    `ablation.py`'s "MEASURED" comment block held the four retracted reranking rows plus three
    conclusions drawn from them, in the most authoritative place a reader would look, for as
    long as nobody re-read the comment. It was deleted rather than corrected: a source comment
    cannot be re-measured when the reranker, the index or the repo changes, so it will always
    drift back toward that failure. A test now greps for all five retracted figures.
    **Corollary for this document and for the README: a figure that cannot be regenerated by
    a command belongs in neither.**

14. **Sizing is a finding, not a presentation choice — and instrumenting beat reasoning about
    it.** WP14 asked for per-family n. What landed determines replication by *instrumenting
    the gate* — a `settrace` digest of the executed `(file, line)` sequence and the bytes
    actually read, with boundary calls identified by code-object identity — and it overturned
    the brief it was written from. `empty_claim` is varying at the direct door, because
    `_submit_body` verifies spans before `write_assertion` refuses the claim; and at the store
    door eight families are replicated, not four, with `paths == 1` for every one, meaning the
    decision path never varies and only the hashed bytes do. **The brief's guess about which
    families were replicated was wrong in both directions, and no amount of reading the family
    table would have shown it.**

15. **Mutation counts are per (family, door, polarity), and the old number was one of each.**
    `12/12 mutation-verified` was the direct door and negative rules only. Measured at
    `3212972` across both doors: 11/11 negative at the direct door and 12/12 at the store
    (`zero_length_span` is not expressible at the direct door and has no mutation there),
    **23/23 negative rules fully detected**; 3/3 positive at each door, **6/6 detected with 3
    partial** (`published_hash` 4/8 direct and 6/8 store, `quoted_lines` 2/6 direct).
    `MutationCensus` counts *rules* and never says "detected", pinned by a test, because
    conflating the two is how the old number drifted.

16. **A headline that divides the instrument's failures out of its own denominator is at its
    most flattering exactly when the instrument is broken.** This is the strongest single
    argument produced in the remediation and it decided WP15: `supported / n` stays the
    headline and `score_decided` is reported beside it. A run with 40% parse failures would
    otherwise report a healthy 0.55 with the badness parked in a counter nobody reads.
    Generalise it: **whenever a denominator can be reduced by a failure of the measuring
    apparatus, the un-reduced denominator is the honest headline.**

17. **Building the apparatus is not the measurement, and the plan must not let the two blur.**
    WP15 shipped `measure_prompt_stability`, `export_for_review` and `score_review`. **None
    has been run.** The judge's calibration is still 15/16 on a different claim set
    pre-labelled by the same model family — Wilson `[0.72, 0.99]` — and the label-flip rate
    under whitespace-only prompt perturbation, which is the largest single uncertainty in
    0.54, still has no number attached. Recorded here as open rather than as a WP15
    deliverable, because "the harness exists" is exactly the kind of thing that reads as
    closure.

### Amendments from the gold-set work (WP22)

18. **A gold set can be too small to detect that it is too small.** The 16-query set was used
    for a year of decisions. Measured against a null built from its own per-query noise, the
    percentile paired bootstrap on it rejects at **11.9%**, not 5% — it errs *narrow*, toward
    inventing findings. So the earlier conclusion that only `prefer_implementation` survived
    the ablation was itself computed with a miscalibrated interval. `CALIBRATION_FLOOR = 128`
    now names the size below which this repo does not publish an interval. **The calibration
    of an interval is a measurement, and it had never been made.**

19. **Repos buy power; queries within a repo buy progressively less.** The repo design effect
    measures 4.6 for the nDCG@10 difference (3.9 for MRR) at `3212972`, and effective n is
    `m·q / (1 + (q−1)·ICC)`, which **saturates at `m / ICC`** — three repos cannot exceed about
    115 effective queries however much gold anyone writes against them. Any future plan to
    "write more gold" has to say which repo it is for.

20. **`run_ablation` returned a finished-looking measurement containing none.** Handed a gold
    set that did not match the index it produced a full table of `0.000` with `[0.000, 0.000]`
    intervals. Validated before scoring now, and in the multi-repo case every repo is validated
    before any is scored. This is the same shape as amendment 12 and belongs beside it: **an
    output shaped like a result is trusted like a result.**

21. **A pooled row can average an effect with a structural absence and report the mean.** WP22
    reported `prefer_implementation` as sign-flipping across repos. Re-measured at `3212972` it
    does not sign-flip: swarm-sync +0.180 [+0.157, +0.204], kalshi-bot +0.117 [+0.093, +0.142],
    TradingAgents **+0.000 [+0.000, +0.000]**. That third row is not a null result, it is a
    **no-op**: TradingAgents has zero files matching `is_test_path`, so the demotion has nothing
    to demote, and an interval of exactly zero width is the signature of a mechanism that did not
    fire. Two consequences, and the second is the reusable one. The pooled row understates the
    effect wherever the mechanism exists, and the repo-clustered interval touching `+0.000` is
    that structural zero rather than evidence of fragility. **A per-stratum row of exactly zero
    with a zero-width interval must be checked for "the mechanism was absent" before it is read
    as "the effect was absent".

22. **The modality claim does not survive, and WP18's brief said it did.** The brief for this
    package stated that "the fusion beating lexical alone, +0.052 [+0.015, +0.088], is the
    modality claim that survives". Re-measured at `3212972` over 520 queries, paired, nDCG@10:
    adding dense to lexical is **−0.012 [−0.034, +0.010]**, and adding *both* dense and graph
    with the test demotion held off is **−0.013 [−0.036, +0.009]** — which on MRR is a
    significant loss at −0.030 [−0.057, −0.002]. These are not wide intervals around a hopeful
    zero. The shipped default's +0.120 over lexical-only is real and is almost entirely the
    **test demotion** (+0.116 without graph, +0.133 with it): a one-line score multiplier doing
    the work three retrieval modalities and a fusion algorithm were built to do.

    Two things to carry forward. First, the mechanical one: **a comparison between the shipped
    default and a baseline is not a modality ablation**, because the default has other things
    switched on inside it, and the only way to see that is to run the pairing that holds them
    off. Second, and this is why it is recorded as an amendment rather than a result: the wrong
    figure arrived in a *brief*, from a previous wave's own summary, and the only reason it did
    not go into the README is that WP18 was instructed to re-run every number rather than to
    copy any. **The habit is the control. Nothing else caught this.****

### WP19 — remaining unguarded reads (added post-Wave-0, not audit-derived)

The WP2 agent found four more unguarded `read_bytes` on repo-relative paths that no auditor
caught: `assertions/stale.py:261` (`_read_file`), `generate/pipeline.py:467` (`_read_source`),
`chunk/chunker.py:149`, `eval/faithfulness.py:665`. The first is a **serve-path** read with
the same shape as the `store._read_source` bug fixed in Wave 0, reachable by the same
one-FIFO-wedges-the-server route, so Wave 0 closed the demonstrated instance and left an
equivalent one open. Guard each with `is_file()`, matching each caller's existing disposition
for an unreadable file, and state the residual check-then-open race rather than implying it
is closed.

**Provenance markers.** Every finding below is tagged:

- `[V]` — reproduced directly by the synthesising engineer, command and output in hand.
- `[A]` — reported by an auditor with evidence quoted in their report, not independently re-run.
- `[H]` — hypothesis with a stated mechanism and no demonstration. Treat as a task to *test*,
  not a task to fix.

Do not upgrade an `[A]` to a fact in a commit message, and do not fix an `[H]` before
confirming it. The project's whole thesis is the difference between measured and asserted;
a remediation plan that blurs it is self-defeating.

---

## The one-paragraph summary

The binding primitive is sound. Hashing, byte/line derivation, crash safety, concurrency
across processes, and the citation-by-integer design all survived direct attack — the
security auditor could not defeat the gate, and 63 of 76 mutations died. What does not hold
is the **boundary drawn around that primitive**: the gate is three rules living in three
places and only one of them is at the chokepoint; the cited span excludes decorators, which
is the system's only fail-open defect; the tool's only re-index destroys the store it exists
to protect; and the eval layer that produces every published number is the least-tested code
in the repo. The pattern is consistent and worth naming, because it should drive review
priorities from here: **rules were enforced at whichever surface first needed them, and the
chokepoint was never widened afterwards.**

---

## Sequencing, and the one hard dependency

Work packages are grouped into five waves. Within a wave, packages are independent and can
be done in parallel by separate agents. Between waves, the order matters.

```
plan                            landed as
WAVE 0  WP1 WP2 WP3             31e6c97  security containment — do first
WAVE 1  WP4 WP5 WP6             31e6c97 (WP6) + 72e5a0c (WP4, WP19)
                                243095d (WP5, WP7, WP10.1, WP10.3, WP20)
WAVE 2  WP7 ───────────────►    243095d  re-index preserves the store  ── MUST PRECEDE WP8
WAVE 3  WP8 WP9 WP10            2f299c8, then 7df16d6 (WP21, found by verifying WP8)
WAVE 4  WP11..WP18              1021a14 (WP12-15) + 3212972 (WP22) + this commit (WP11, WP18)
                                WP16 partial, WP17 not started
```

The waves as executed do not line up one-to-one with the waves as planned — WP5 and WP20
slid from Wave 1 into the Wave 2 commit, WP21 and WP22 did not exist when this was written,
and WP11 landed last rather than first in its wave. **The status table above is the
authority; this diagram is the plan.** They are kept apart rather than reconciled, because
rewriting the plan to match what happened is how a plan stops being able to tell you it was
wrong.

**The hard dependency is WP7 → WP8.** Fixing the decorator span (WP8) changes every stored
`content_hash`, which forces `SCHEMA_VERSION = 6`, which under the existing refuse-and-rotate
policy means every existing index must be rebuilt. Today rebuilding is `--force`, and
`--force` deletes the entire tier-2 store `[V]`. **Shipping WP8 before WP7 destroys every
assertion, verdict, and rejection any user has accumulated, as a direct consequence of a bug
fix.** WP7 exists to make that upgrade survivable. Do not reorder these.

---

# WAVE 0 — security containment

## WP1 — Indexing a repo executes arbitrary commands from its `.git/config` `[V]`

**Severity: CRITICAL.** Reproduced end to end.

`ingest/indexer.py:113-118` and `eval/gold_from_history.py:366-371` shell out to `git -C
<repo> ls-files`. Git reads the *target repo's* config and honours `core.fsmonitor`, which it
executes. A repo containing:

```
[core]
	fsmonitor = "touch /tmp/PWNED"
```

runs that command during `codelearner index`, silently, while reporting a successful index.
`core.hooksPath`, `core.pager` and `alias.*` are the same class. `safe.directory` does not
help — it only guards repos owned by a different uid, which is exactly the case this misses.

Scope honestly: `git clone` does not transfer remote config, so this is not "clone a repo and
get owned". It fires on repos delivered as archives, vendored submodules, agent-written
directories, or any directory a second party can write to.

**Change.** Put the overrides on the command line, where they outrank repo config, at both
call sites:

```python
["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
 "-C", str(repo_root), "ls-files", "-z", "--", "*.py"]
```

plus `env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}`.

**Verified fix.** The synthesiser confirmed the control executes and the hardened invocation
does not. Do not substitute a different mitigation without re-running that check.

**Acceptance.** A test building a repo with a hostile `.git/config`, indexing it, and
asserting the side effect did not occur. This test must fail against current `master`.

---

## WP2 — `submit_assertion`'s refusal is an arbitrary-file-read oracle `[A]`

**Severity: HIGH.** Demonstrated by the auditor against `.env` and an SSH private key.

`server/app.py:412` returns `observed_text` — the full cited byte range, decoded, uncapped —
inside the `hash_mismatch` refusal. The gate correctly refuses paths that escape the repo
root, but *inside* the root it reads any file, indexed or not, and echoes the bytes back. An
agent submits a deliberately wrong `content_hash` and reads the file out of the error. Walk
the line ranges and you get whole files.

The gate is not defeated here. The **refusal is the exfiltration channel**.

**Change.**
1. Restrict the read to files present in the `files` table — one `SELECT 1 FROM files WHERE
   path = ?` before `read_bytes`. A citation of an unindexed file is not a legitimate
   submission, so this costs nothing real.
2. Cap `observed_text` at ~2KB with an explicit `"…truncated"` marker.
3. Refuse files above a size ceiling by `stat().st_size` before reading (this also fixes the
   200MB-response finding: one call currently returns a 209,715,200-character payload at
   ~479MB peak RSS `[A]`).

**Acceptance.** Tests asserting: an unindexed file in the repo root is refused with a code
that does *not* include its contents; `observed_text` never exceeds the cap; a large file
is refused rather than read.

---

## WP3 — MCP transport-contract violations and unbounded writes `[A]`

**Severity: MEDIUM-HIGH.** Four distinct defects, one package because they share a file and a
test fixture.

| # | Defect | Anchor |
|---|---|---|
| a | NUL in a citation path raises `ValueError` into the transport — `_guard` catches only `ToolError`/`sqlite3.Error` | `server/app.py:326` |
| b | NUL in a search query makes FTS5 report `unterminated string`, which `_guard` maps to `index_unreadable` — the server accuses its own healthy index of corruption | `retrieve/lexical.py:40,50` |
| c | A FIFO in the repo hangs `read_bytes()` forever, wedging the single-threaded server | `server/app.py:335`, `assertions/store.py:177,367` |
| d | No cap on evidence-span count, claim length, or confidence. One call stored 5,000 spans and a 5MB claim; `confidence=1e308` was accepted. Damage is permanent (the store never deletes) and amplified on every subsequent `get_symbol` | `server/app.py:681`, `schema.sql` |

Both (a) and (b) violate the explicit README guarantee that no predictable condition raises
into the transport.

**Change.** (a) reject `"\x00" in path` up front as `bad_path`, and add `ValueError` to
`_guard`. (b) add `\x00` (or all C0 controls) to `_FTS_SPECIAL`. (c) `if not
target.is_file(): raise ToolError("file_missing", …)` before every citation read. (d) cap
spans at 32 and claim at 4096 chars in `_submit_body`; add `CHECK (confidence IS NULL OR
confidence BETWEEN 0 AND 1)` to the schema.

**Note.** (d)'s schema CHECK is a DDL change. Fold it into the WP8 v6 bump rather than
bumping twice.

**Acceptance.** One test per row, each asserting the `{"ok": false, "error": {...}}` shape
rather than an exception.

---

# WAVE 1 — make the gate one gate

This wave exists because of the single most consequential architectural finding: **the gate
is presented as one function and is actually three rules in three places.**

`store.write_assertion` — documented as "exactly one place in the project that can decide an
uncited claim is inadmissible" — enforces exactly one rule: `spans` is non-empty `[V]`. Hash
verification and subject-existence live only in `server/app.py`. And `eval/gate_controls.py`
imports `..server.app`, so the 100%-refused table measures **only the MCP path** `[A]`.
`codelearner learn` and every library caller enter through a door with one lock.

## WP4 — Move every admission rule into `write_assertion`

**Severity: CRITICAL.**

**Change.** `store.write_assertion` gains, before the transaction opens:

1. **Span validity** — assert `0 <= byte_start < byte_end` for every span. Today a
   zero-length span is admitted and verifies forever against any file content `[A]`;
   `span_for` refuses this with exactly the right reasoning, but that guard is in a
   constructor the gate does not require.
2. **Non-empty claim** — an empty claim with a valid citation is currently admitted, stored
   `active`, and returned as servable `[V]`. This is the `unknown_subject` failure class
   reached through a third door: verified evidence carrying an unactionable claim.
   `pipeline.py` already refuses it (`OUTCOME_EMPTY_CLAIM`), which is the proof the rule
   belongs in the gate rather than the caller.
3. **Subject exists** — `SELECT 1 FROM symbols WHERE qualname = ?`, with an explicit
   `allow_unindexed_subject=True` escape for eval fixtures that need it.
4. **Evidence verifiable** — reject a span carrying neither `content_hash` nor `text` `[A]`.
5. **Hash re-verification** — `verify: bool = True` re-reads each span off disk and compares.
   `server/app._verify_span` keeps its richer error payloads as a *pre*-check for message
   quality; enforcement moves to the chokepoint.

New exceptions as siblings of `EvidenceRequired`: `EvidenceStale`, `UnknownSubject`,
`EmptyClaim`, `InvalidSpan`, `EvidenceUnverifiable`.

**Acceptance.** Each rule gets a test asserting *no row is left behind*, matching the
standard set by `test_a_refused_assertion_leaves_no_row_behind`.

## WP5 — Point the negative controls at the chokepoint, and add the missing families

**Severity: HIGH.** Depends on WP4.

**Change.**
1. `gate_controls.gate_module()` gains a thin adapter so the same corpus runs against
   **both** `server.app._submit_body` and `store.write_assertion` directly. Report two
   columns.
2. Add two families the corpus lacks: `unverifiable_span` (`evidence_unverifiable`) and
   `empty_claim` — the latter targeting a rule that does not exist until WP4 lands.
3. Add a `decorator_edit` family (see WP8) and a `zero_length_span` family.

**Why this matters beyond coverage.** The corpus is generated from `FAMILIES`, so it can only
find holes someone already enumerated. It has found no new hole since `unknown_subject` — and
an auditor found two by probing outside the list. That is a property of the method, not
evidence that none remain. Adding families is necessary; treating the corpus as exhaustive is
the error to avoid.

## WP6 — Fix `Outcome.held`, the predicate the whole gate report is computed through

**Severity: CRITICAL (test integrity).** `eval/gate_controls.py:1040-1058`.

Three of its four conjuncts can be deleted with all 484 tests green `[A]`:

| mutation | result |
|---|---|
| drop `and self.rows_added == 0` | **SURVIVED** — a refusal that leaks a row scores as clean |
| drop `and self.code in spec.codes` | **SURVIVED** — a refusal for the wrong reason scores as right |
| `and self.servable is True` → `and True` | **SURVIVED** |
| `and self.evidence == self.expected_evidence` → `and True` | caught |

`held` feeds `hold_rate` → `rejection_rate` / `positive_pass_rate` → and
`MutationResult.detected`. **Every number the gate apparatus reports passes through a
predicate that is 3/4 unpinned.** This is the exact failure `assertions/store.py:23-28` warns
about — a pipeline that scores its own rejections — applied one level up.

**Change.** A table-driven unit test building synthetic `Outcome`s (refused/right-code/1-row;
refused/wrong-code/0-rows; accepted/servable=False; accepted/evidence<expected) asserting
`held is False` for each. No index needed. ~25 lines.

---

# WAVE 2 — the second day

Two auditors independently reached the same conclusion from different directions:
**everything is built to survive the repo changing except the thing that has to happen when
the repo changes.**

## WP7 — Re-index must preserve the tier-2 store

**Severity: CRITICAL. Blocks WP8.**

`cmd_index --force` → `_delete_index` unlinks the DB file `[V]`:

```
before: assertions=2 verdicts=1
$ codelearner index <repo> --force
after:  assertions=0 verdicts=0 staleness_log=0
```

The help text says it "discards its embeddings". Embeddings are re-derivable in minutes;
verdicts and the rejected set are not re-derivable at all — and the README's own argument is
that "the rejected set is the only evidence the gate does anything".

The `ON DELETE SET NULL` on `subject_symbol_id` that the schema and README spend paragraphs
justifying as protection against exactly this **has never executed**, because nothing in the
package deletes a `symbols` row `[A]`. The protection is real in the DDL, correct in its
reasoning, and unreachable in the shipped product.

**Change.**
1. Before `_delete_index`, dump `assertions`, `evidence_spans`, `verdicts`, `staleness_log`
   to a sidecar; after `index_repo` returns, re-insert and re-resolve `subject_symbol_id` by
   `subject_qualname` (which is `NOT NULL` precisely so this works).
2. Spans whose hash no longer matches the rebuilt index become `stale` with a
   `staleness_log` row — not deleted. That is the honest outcome and it is what the
   staleness engine is for.
3. The refusal message names the counts: *"discards N assertions, N verdicts, N staleness
   events and any embeddings."*
4. Refuse outright when the assertion count is non-zero without a second explicit flag.

**Acceptance.** Test: admit assertions, `--force`, assert they survive with correctly
re-resolved `subject_symbol_id`, and that ones whose evidence moved are `stale` rather than
absent.

## WP8 — Decorators are outside the cited span

**Severity: CRITICAL. The system's only fail-open defect. Depends on WP7.**

`ingest/python_extract.py:158-162` takes the span from the `function_definition` node, which
in tree-sitter-python begins at `def`, not at `@`. Reproduced `[V]`:

```
symbol: app.list_users lines 6-8
--- CITED SPAN ---
def list_users():
    """List users."""
    return []
--- decorators inside cited span? --- False / False
```

**Why this outranks everything else in the audit.** Every other defect found across six audits
fails *closed* — a claim is lost, or a call raises. This one fails **open**. A claim
"serves GET /users; responses cached 60s; read-only endpoint" cites a span that does not
contain `@route("/users")` or `@cache(ttl=60)`. Rewrite the decorators and:

- both verifiers report `fresh`, `method='hash'`, `verified_at=now`;
- `force_hash=True` does not help — the cited bytes genuinely did not change;
- nothing appears in `staleness_log`;
- the faithfulness judge is shown the same truncated span and correctly rules the claim
  supported;
- a human following the citation finds those exact bytes unchanged.

There is no signal anywhere. It applies to every routing, auth, caching, transaction,
`@property`, and `@staticmethod` decorator.

Prevalence: ~15% of code-learner's own symbols `[A]`; 7.8% of swarm-sync's 1,270 by a cruder
single-line-lookback heuristic `[V]` (a lower bound — multi-line decorators are missed).

Compounding: `pipeline.py:536` shows the model the same span, so it drafts about a function it
was never shown the decorators of.

**Also fix the docstring that asserts the opposite.** `server/app.py:283` states *"for a
decorated symbol it begins at the `@`"* `[V]`. It does not.

**Change.** In `python_extract.py:137`, when `node.parent` is a `decorated_definition`, take
`byte_start`/`line_start` from the parent while keeping `name`/`signature`/`docstring` from
the inner node. Bump `SCHEMA_VERSION = 6`. Re-measure and correct the symbol-bytes-vs-
line-bytes figures that `server/app.py:285` and the README quote three different ways
(15% / 22.2% / 25.5% for one measurement `[A]`).

**Acceptance.** A test asserting a decorated symbol's span starts at `@`; a `decorator_edit`
negative-control family (WP5) asserting a decorator rewrite expires the claim.

## WP21 — WP8's fix does not reach the citations WP7 carries (added post-Wave-3, not audit-derived)

**Severity: HIGH. WP8's fail-open defect, surviving inside carried data. Depends on WP7 and WP8.**

WP8 widened the span and bumped the schema; `codelearner index --force --carry-assertions`
carries the tier-2 store across that rebuild, and a carried claim keeps its pre-v6 citation.
Those bytes are unchanged on disk, so the claim stays `active` and servable — correctly, by
the rules as written, and therefore with the exact exposure WP8 existed to close. Measured on
swarm-sync's upgraded v6 index, 150 carried assertions `[V]`:

```
active evidence spans                                   143
strict suffixes of a symbol (pre-v6 narrow citation)     15
of those, decorator-narrowed                             11   (10 distinct assertions)
```

One cites a symbol whose missed prefix is `@app.post("/intent", dependencies=[Depends(require_token)])`
— a live, servable claim about an endpoint, citing bytes that exclude its authentication
dependency. Strip `Depends(require_token)` and the claim still verifies fresh.

**Change.** On the carry path only, after the ordinary verification, mark such a claim `stale`
with its own reason (`decorators_excluded`) so it is redrafted rather than silently retained.
Detection is exact, not heuristic: a span whose end matches a symbol's end and whose start is
the start of the definition *inside* that symbol's `decorated_definition` node, asked of
tree-sitter (`python_extract.decorated_body_start`) rather than inferred from the prefix text.

**Precision over coverage, deliberately.** Not every strict suffix of a symbol — the gate
admits a legitimate sub-range citation, and an agent quoting three lines of a function body is
making a *narrower and therefore stronger* citation. 4 of the 15 above are exactly that (a
claim about a class's last method, whose span ends where the class does) and are left alone.

**Never rewritten.** Widening the stored span to match the symbol would fabricate a citation
the generator never made. See `assertions/boundaries.py`.

**Acceptance.** A claim citing a decorated symbol's old narrow span comes back `stale` with the
new reason; a sub-range citation of a function body does not; an undecorated symbol's claim is
untouched; a `rejected` claim keeps its status; the reason reaches the carry summary with
wording that says a re-index will not repair it.

**What this does NOT fix.** Citations an agent has already been handed. This changes what the
index publishes from now on; anything already quoted downstream is out of reach.

## WP9 — A live MCP server serves the deleted index and reports success `[A]`

**Severity: CRITICAL.** `server/app.py:137-162`.

`IndexSource.connect()` re-checks `self.path.exists()` so a deleted index does not become an
empty one — but `--force` *replaces* the file. The path exists again, the check passes, and
the cached `self._conn`, still bound to the unlinked inode, is returned. Writes after the
re-index are accepted and lost:

```
submit after reindex -> True id 1 servable True
assertions surviving on disk: 0
```

This is the realistic second-day sequence: agent session open in the editor, human re-indexes
in a terminal.

**Change.** Stat `st_dev`+`st_ino` (or `meta.indexed_at`) alongside the existence check; drop
the cached connection when it changes; return an `index_replaced` error on the first call
after a swap so the agent re-reads hashes.

## WP10 — Upgrade path, drift detection, and the errors around them

**Severity: HIGH.** Three findings that all surface on day two.

1. **`SchemaVersionError` tracebacks out of `search`, `stats`, `learn`, and the MCP
   transport** `[A]`. `open_index` catches only `sqlite3.Error`; `_guard` likewise. This
   violates two explicit README guarantees, and it is the most-predicted failure in the
   design — `SCHEMA_VERSION` has moved five times, and is about to move again in WP8. Add
   `SchemaVersionError`/`RepoRootMismatchError` to both, as `code: "schema_mismatch"`.
2. **No staleness detection for tier 0/1 at all** `[A]`. `files.content_hash` and
   `files.mtime_ns` are written at index time and read by no code path. So T0 — *"deterministic,
   reproducible from source alone"* — serves line numbers that are silently wrong after any
   edit, while T2 has a whole two-stage engine. Add a `stat()` sweep at open time and one
   stderr note / MCP `notes[]` entry: *"N of M indexed files have changed since this index
   was built."*
3. **Transient read failures permanently expire claims** `[A]`. `_read_source` swallows every
   `OSError` → `file_missing` → `mark_stale`, and nothing anywhere moves an assertion back to
   `active`. A `chmod 000`, an `EMFILE`, an NFS blip, or a moved repo destroys the store
   irreversibly. Split `FileNotFoundError`/`NotADirectoryError` (real absence → expire) from
   other `OSError` (unreadable → withhold this call, do not mutate status). Add
   `reinstate(conn, assertion_id)` that re-hashes and flips back only on exact match.

---

# WAVE 3 — measurement integrity

The eval layer produces every published number and is the least-tested code in the repo.
These packages are ordered so the corrections land before the re-publication.

## WP11 — Withdraw or correct the numbers that no longer reproduce

**Severity: HIGH. Done at this commit.** All findings originally `[A]`; the disposition
column is `[V]` — every row was re-run at `3212972` before it was written, with the command
recorded in the README's reproduction section.

| Claim | Reality as audited | Disposition |
|---|---|---|
| `1,764 pairs, 0 findings` (leak audit) | **1,849 pairs, 2 findings** at audit time | **Corrected, then re-earned.** WP12 wired the audit into the scored run and widened the copy filter across commits. Re-run at `3212972` against swarm-sync@`3119a97`: 42 views × 42 labels = **1,764 pairs, 0 findings**. Published with the mechanism beside it (amendment 12) |
| "Mined prose is, if anything, a slightly better retrieval query than a hand-written question" + comparator `MRR 0.221 / hit@10 0.435` | Reverses on re-run | **Withdrawn, not reversed.** On 16 and 42 items neither set can support the comparison in either direction. The README says so |
| `+0.226 MRR swing … safe to believe` | Quotes the **retracted** delta | **Withdrawn with the whole reranking result.** The corrected rows were measured on 16 queries, below `CALIBRATION_FLOOR`, so the interval they were read against does not mean what it said. No reranking figure is published until it is re-measured on the 638-row set |
| `ablation.py:246-261`, headed "MEASURED" | Four retracted rows plus three conclusions | **Deleted** at `1021a14`, with a test that greps for all five retracted figures (amendment 13) |
| `42 / 316 / 13.3% / 17 commits / lift 0.161` vs the `43/318/13.5%/18` table ~170 lines later | Two versions of one measurement presented as co-equal | **One canonical table.** Re-run at `3212972`: 42 of 318 (13.2%), 17 commits, 9 from one, at swarm-sync@`3119a97`. The 43-label table survives only inside the generation section, captioned as predating the WP13 corrections and readable for its ordering alone |
| `0.6 → 0.615` sweep row | Not producible by the shipped sweep `(0.3, 1.0, 1.5)` | **Removed.** The published sweep is the three weights the code actually runs |
| `6,091 / 1,688 / 850 symbols` | `7,717 / 2,145 / 1,079` — scales with repo size, uncaptioned | **Re-run and captioned.** At `3212972`, 1,714 symbols: 15,620 instances / 8,691 distinct executions at the direct door and 17,334 / 5,202 at the store, with 3,415 positive controls. The pooled figure is demoted to a caption under the per-family table (WP14) |
| `85 of 383 symbols` (symbol bytes ≠ line bytes) | Three incompatible figures for one measurement (15% / 22.2% / 25.5%) | **Re-measured and cross-tabulated:** 438 of 1,714 (25.6%) at `3212972`, with the causal attribution corrected (amendment 7) |
| `0.544` faithfulness | Three decimals on n=147 | **`0.54 [0.46, 0.62]`** (Wilson), recomputed at `3212972` from the 147 verdicts the run left in swarm-sync's store — 80 supported, 65 not_supported, 2 uncertain — with `score_decided` 0.55 beside it |

**Change (done).** Corrected each in place. **Every published table is stamped `repo@sha`**,
and the four shas are named once at the top of the README rather than per table. The
conditions are deterministic and reproduce exactly at a given sha, so this is bookkeeping —
which is precisely why leaving it uncaptioned was worse than it looked: it let two versions
of one number sit in one document without either looking wrong.

## WP12 — Fix the leak boundary, and wire it into the scored run

**Severity: CRITICAL (methodology).** `eval/gold_from_history.py`.

`score_purposes:1104` calls `assert_no_leak(view, [lab.prose])` — the view against **its own**
label only. Cross-symbol leakage is invisible to every gate that runs during a scored run,
and the mining-time copy filter is per-symbol so it cannot reject the class by construction.
`audit_leak_boundary`, the only thing that checks the cross product, **is called by no
reported code path**.

**Change.**
1. Call `audit_leak_boundary` from `run_purpose_eval`; fail the run on non-zero findings.
2. Extend the mining copy filter to reject any label whose clause appears in a *sibling
   labelled symbol's* source within the same commit.
3. Re-publish the pair count with a sha.

**Note the concession this breaks.** The README concedes that "correlated phrasing cannot be
filtered". This leak is not correlated phrasing — it is a copied clause crossing a symbol
boundary, which the concession does not describe. Update the wording.

## WP13 — Statistical corrections the harness can compute for free

**Severity: HIGH.** All `[A]`, all requiring **zero model calls** (baselines are deterministic
and LLM output is already cached at `purpose.py:497`).

1. **`lift` is a single draw from the null.** `_derangement` produces one permutation, so
   `shuffled` is one sample and `lift` inherits its full sampling error, never reported.
   Averaged over 500 derangements the published controls are off by up to ±0.015 — the
   `body identifiers` control is +2.3sd above the null mean, `name + signature` is −1.3sd
   below. Average over ≥200 derangements; report null sd and a permutation p-value. (Every
   condition clears its null at p < 0.002, including the name floor — this *strengthens*
   under the fix.)
2. **Name-blinding blinds only the leaf qualname component.** 34 of 43 labels (79%) still
   share tokens with the non-blinded parts. Correcting it drops the docstring lift 19% and
   the body lift 6% — **asymmetric, so it biases the comparison, not just the levels.** Blind
   every dotted component plus the path stem.
3. **No confidence intervals anywhere.** Clustered bootstrap over the 18 introducing commits
   puts between-condition resolution at **±0.04**. This is directly relevant to the headline
   I published last night: "the LLM is beaten by a bag of body identifiers" is solid on
   `gold` (0.205 vs 0.102) but **marginal on `lift`** (0.140 vs 0.081) — and lift is the
   number the README correctly says to read. Restate with the interval.
4. **The ablation's noise band is below its own quantum.** 11 of 16 gold queries carry exactly
   one relevant symbol, so `hit@5`'s quantum is 6.25 points and binomial noise on n=16 is
   ~11.5 points sd. "Treat one or two points as noise" understates it by an order of
   magnitude. Add a paired bootstrap over queries.

## WP14 — Report the gate's numbers at their real resolution

**Severity: MEDIUM.** `[A]`

`6,091 attacks / 100.00%` is **nine attack shapes replicated per symbol**; 56% of the volume
is four probes (`zero_evidence`, `absent_file`, `escaping_path`, `unknown_subject`) repeated
once per symbol, whose decision path does not vary with the symbol at all. The rate reproduces
at 1.0000 — the number is not wrong, it is **sized** wrong, and the sizing is what persuades.
Per-family bounds are much weaker: `past_eof` at n=59 supports a 95% upper bound on failure of
~5%.

**Change.** Make the per-family n and hold rate the primary table (the code already computes
it), demote the pooled count to "instances", caption as *"9 attack shapes, instantiated per
symbol against N symbols at `<sha>`"*, and report `100.0%` at one decimal.

Similarly, restate `12/12 mutation-verified` as what the auditor re-measured: **9/9 negative
rules produce admitted attacks when deleted; 3/3 positive rules partially detected (n=2–8 on
the shapes fixture)**. The nine-family result is the best-verified thing in the repo and
should be stated in its strong form, not pooled with the partial ones.

## WP15 — Faithfulness: the denominator, the interval, and the instrument

**Severity: MEDIUM.** `[A]`

1. `score = supported / len(adjudications)` puts three different events in one denominator:
   a genuine "uncertain" verdict, a harness/transport parse failure, and a judge-format
   failure. Only the first is evidence about the claim. Report both `supported/n` and
   `supported/(supported + not_supported)`, and add `parse_failures` / `judge_uncertain`
   counters so a bad run is visible as a bad run.
2. `0.544` → **`0.54 [0.46, 0.62]`** (Wilson, ignoring clustering).
3. The judge's entire calibration is 15/16 on a *different* 16-claim set, pre-labelled by the
   same model family that authored those claims. "Self-consistency" is three runs of an
   identical prompt at temperature 0 — that measures decoding determinism, not judge
   stability. The README admits observing label flips under whitespace-only prompt changes,
   with no rate attached; that flip rate is the largest uncertainty in 0.544 and is unmeasured.
4. **"Cross-family" is a string-prefix test on a model tag.** A Qwen-distilled model published
   as `deepseek-r1` passes; the repo's own reranker is Qwen3-based and would too. Demote the
   wording from "different model family, which is the point" to "different weights and
   tokenizer, a proxy for independence rather than a demonstration of it".

**The measurement most worth adding** (deferred, needs human time, ~30 labels + 3 GPU-hours):
hand-label a stratified 30 of the 147, publish judge-vs-human precision/recall on the set the
number is actually computed over, and re-run all 147 under trivial `render_evidence`
perturbations to publish a label-flip rate. **Still not done.** `measure_prompt_stability`,
`export_for_review` and `score_review` shipped at `1021a14` and none has been run
(amendment 17).

## WP22 — The gold set could not resolve the comparisons it was being used for (added post-Wave-4)

**Severity: HIGH (methodology). Done at `3212972`.** Not audit-derived: it was found by
taking WP13.4's note about the ablation's noise band seriously and measuring it.

The 16-query set was one file, one repo, one labeller. `hit@5` moved in steps of 6.25 points
— one query — and 11 of 16 queries carried exactly one relevant symbol, which makes
`recall@k` and `hit@k` the same number computed twice. `dense-only` against `lexical-only`
came out ΔMRR `[-0.128, +0.323]`: the apparent dense win was noise.

**What landed.** 638 rows over 6 repos in three sources that are never pooled silently —
170 hand-written (3 repos, 108 multi-relevant, 21 hard negatives), 452 mined from commit
prose (6 repos, emitted twice as verbatim and name-blind sharing a `pair_id`), and the
original 16 kept as written. `GoldSet` carries `source` and `repo` per row; scoring emits a
row per stratum; the pooled row is labelled `POOLED`.

Three mining decisions are load-bearing and each closes a defect the hand set had:
commit-first rather than symbol-first (symbol-first yields one relevant per query *by
construction*, which is the defect being fixed); attribution by file touch (without it a
swarm-sync commit about pruning worktrees that mentions `acquire_lease` in passing mints a
worktree query whose gold answer is a lease function); and both name variants emitted, because
the mention rule guarantees a mined query contains its targets' identifiers, so blinding
everything measures a mode users do not use and blinding nothing is a lexical benchmark.

Token overlap is **reported, never filtered**, and the argument is empirical rather than
principled: the existing hand set scores median 0.71 with two queries at exactly 1.00, so any
threshold worth setting rejects the benchmark it extends. `source_overlap` also turns out to
measure documentation density rather than query discipline (swarm-sync 0.43, kalshi-bot 0.14,
TradingAgents 0.12), which is a confound for any pooled row.

**The power analysis is the part that changes how this gets used**, and it is amendments 18
and 19. Re-measured at `3212972` on the 520 rows whose repos are indexed and embedded:
paired sd 0.2865 (nDCG@10) and 0.3522 (MRR) on the lexical-vs-hybrid pair, giving half-widths
of `0.56/√n` and `0.69/√n` against the corpus-median `0.60/√n`; repo design effect 4.6
(nDCG) and 3.9 (MRR); and Δ=0.05 needing **~1,186 clustered queries** at 80% power. Δ=0.15
and Δ=0.10 are affordable at 132 and 297. **Δ=0.05 must not be promised.**

`nDCG@10` becomes primary on measured grounds: 30% fewer queries for the same delta, it
separates 23 of 55 real configuration pairs at n=16 against MRR's 11, and it agrees with MRR's
ordering on 54 of 55 — the same question with less noise, not a different question.

**Acceptance (met).** Every published retrieval figure in the README is a stratified row or a
paired delta with an interval, stamped with the three repo shas; and the pooled row is never
the only row shown for a comparison whose strata disagree.

---

# WAVE 4 — tests, architecture, docs

## WP16 — Close the surviving mutations

**Severity: HIGH. PARTIAL.** All originally `[A]`, each with the exact edit that survived.

**State at `3212972`.** Items 1 and 2 landed, and the "Also" list is closed. Items 3, 4, 7
and 8 have tests that name the right behaviour, found by reading `tests/` at this commit —
but WP18 **did not re-run their mutations**, so they are marked `[A]`-on-my-own-reading and
not `[V]`. Do not promote them without the delete-the-fix check; that is the whole standing
rule of this repo and it is exactly what an item ticked off by grep skips.

| # | Item | State at `3212972` |
|---|---|---|
| 1 | `eval/ablation.py` has zero tests | **done** (`1021a14`). `tests/test_ablation.py` exists with 71 tests; both previously-surviving mutations (`recall_at` ignoring `k`, `mrr` returning raw rank) now fail |
| 2 | `Outcome.held` | **done** (`31e6c97`, WP6), independently mutation-verified |
| 3 | Multi-span staleness provenance | test present (`test_stale.py:300`, one claim citing two files) — **mutation not re-run** |
| 4 | `stale.py` size check deletable | test present (`test_stale.py:466-477`, truncate + `os.utime` back → `REASON_SPAN_TRUNCATED`) — **mutation not re-run** |
| 5 | `server/app.py` embedder-mismatch guard → `if False:` SURVIVED | **OPEN.** No test in `test_mcp.py` names the embedder/vector mismatch. The agent surface can still silently serve vectors from a different model while the human surface cannot |
| 6 | `pipeline.py` subject-drop accounting | **OPEN.** No counter or test found by name |
| 7 | `store.record_verdict` unguarded status UPDATE | `_TOUCH_STATUS` now carries `WHERE id = ? AND status = ?` and is called with `STATUS_ACTIVE` — **mutation not re-run** |
| 8 | `rerank.py` `[:k]` cap asserted against the fake | **done.** `test_rerank.py` now checks against what the reranker was handed; the docstring records the old `A ⊆ B ∪ A` tautology |
| 9 | `store.assertions_with_status` non-mutating | **OPEN.** Three call sites in `test_assertions.py`, none pinning that the un-verified reader does not mutate |
| Also | `test_chunk.py` assertionless loop, `test_rerank.py` tautology, `test_gate_controls.py` `or family == "multi_span"` short-circuit, unfloored loops | **done** — each carries a comment naming the defect it replaced |

Original detail retained below.

1. **`eval/ablation.py` has zero tests** — 12 of 12 functions never entered. `recall_at`
   ignoring `k` and `mrr` returning raw rank both SURVIVED. This module produced the ablation
   table the design rests on. Add `tests/test_ablation.py` with hand-computed fixtures (~30
   lines, no index).
2. **`Outcome.held`** — see WP6.
3. **Multi-span staleness provenance** — every stale test uses a single-span assertion, so
   `any→all` on the weakest-link method and `min→max` on `verified_at` both SURVIVED. These
   are the two fields the module's honesty claim rests on. One test citing two files, hashing
   both, touching one.
4. **`stale.py:349` size check deletable** — the existing test preserves both mtime and size
   so it exercises neither guard individually. Live consequence: a truncated file with
   restored mtime is served fresh. Test: truncate, `os.utime` back, assert
   `REASON_SPAN_TRUNCATED`.
5. **`server/app.py:197` embedder-mismatch guard → `if False:` SURVIVED**, while its CLI twin
   is pinned. The agent surface can silently serve vectors from a different model; the human
   surface cannot.
6. `pipeline.py:615` subject-drop accounting (count *and* category) unpinned.
7. `store.record_verdict` unguarded status UPDATE — stale claims demotable to rejected.
8. `rerank.py:269` `[:k]` cap — the test that claims to check it asserts the *fake's* cap.
9. `store.assertions_with_status` — no test pins that the "un-verified reader" does not mutate.

**Also:** `tests/test_chunk.py:182` has a loop body with no assertion; `test_rerank.py:155` is
the tautology `A ⊆ B∪A`; `test_gate_controls.py:334` short-circuits with `or family ==
"multi_span"`; ~10 unfloored loops pass vacuously on an empty collection. The repo already
establishes the floor idiom in two files — apply it.

**Load-bearing estimate to record honestly:** ~300–330 of 484 (~65%) demonstrably pin
something; ~15 are vacuous.

## WP17 — Architecture: one rule, one place

**Severity: MEDIUM. NOT STARTED — all seven items re-verified open at `3212972` `[V]`.**

| # | Item | Check run at `3212972` | State |
|---|---|---|---|
| 1 | `generate` imports `eval` | `codelearner/generate/purpose.py:102` still reads `from ..eval.gold_from_history import (…)`; `codelearner/sourceview.py` does not exist | **open** |
| 2 | `tier_of` lives in `cli/render.py`, so `server` imports upward | `server/app.py:38-39` still import `facts_only` and `hit_json` from `..cli.render`, and `app.py:31` imports from `..cli.commands`; neither `codelearner/tier.py` nor `codelearner/indexinfo.py` exists | **open** |
| 3 | `facts_only` is a provable no-op | `app.py:1297` now *documents* that it drops nothing, which is an improvement over advertising it as working. Nothing emits a `TIER_INFERRED` modality, so the flag is still inert, and it is still absent from `get_symbol` — the only tool that returns T2. Now stated in the README too | **open, but no longer silent** |
| 4 | Adjudication unreachable from any shipped surface | `store.record_verdict` has exactly one non-test caller, `eval/faithfulness.py:1185`. No `codelearner judge` subcommand (the CLI is `index/search/stats/learn`), no MCP tool, and no `require_verdict` anywhere in the package. So 0.54 describes claims that are served regardless of verdict. The README now says this in the faithfulness section rather than leaving it inferable | **open, now stated** |
| 5 | Two-stage staleness engine has zero production callers | `span_verifications` is written and read by `refresh_staleness` and tests, and no shipped surface takes a `--verifier` flag or surfaces `method`/`verified_at`. Stated in the README as "a measured alternative that is not wired" | **open, now stated** |
| 6 | `store._load_assertions` builds an unbatched `IN` clause | Still `placeholders = ",".join("?" * len(ids))` with no chunking, while `stale.py` batches its own queries with a comment about exactly this risk. Above 32,766 active assertions every serving and sweeping path raises `too many SQL variables` | **open** |
| 7 | `_atomic`'s transaction sniff is connection-global | No `threading.local` in `server/app.py`; `_atomic` is unchanged | **open** |

Items 3, 4 and 5 are the three where the code and the documentation had been disagreeing.
WP18 closed the *disagreement* by making the documentation match the code; it did not close
the items, and the difference is worth keeping visible. **A defect that is accurately
documented is still a defect** — the reason to write it down is that an undocumented one is
two defects.

Original detail retained below.

1. **`generate` imports `eval`** (`purpose.py:102`), breaking the rule `llm.py:159-170` states
   twice in bold and duplicates a constant to protect. It also closes a real four-package
   cycle `eval → server → cli → generate → eval`, survivable only because two edges are
   function-local and neither is marked as load-bearing for acyclicity. **Fix:** extract
   `SourceView`, `Generator`, `LeakDetected`, `assert_view_is_source_only` into a leaf
   `codelearner/sourceview.py`; re-export for compatibility. Add an AST test asserting
   `generate` imports nothing from `eval`, and that the module-level import graph is a DAG.
2. **`tier_of` — the central design claim — lives in `cli/render.py`**, so `server` imports
   *upward* into the CLI and takes three privates with it. Extract `codelearner/tier.py` and
   `codelearner/indexinfo.py` (the latter for `_scalar`, `_classify_unresolved`,
   `_embedding_info`, used identically by both surfaces).
3. **`facts_only` is provably a no-op** `[V]` — nothing emits a `TIER_INFERRED` modality — and
   it is absent from `get_symbol`, the only tool that returns T2. The MCP docstring and README
   both advertise it as working. Either wire T2 retrieval or say what is true; do not ship
   the two disagreeing again.
4. **Adjudication is unreachable from any shipped surface.** `record_verdict` has one caller,
   in `eval/`. There is no `codelearner judge` and no MCP tool, so claims are served
   unjudged and 0.544 describes claims that are served regardless of verdict. Either add the
   command and `servable_assertions(require_verdict=True)`, or state that adjudication is an
   offline measurement and not admission control.
5. **The two-stage staleness engine has zero production callers** — `span_verifications`, the
   entire reason for the v4→v5 bump, is written and read only by tests. Wire it behind a
   `--verifier` flag and surface `method`/`verified_at`, or say it is a measured alternative
   that is not wired.
6. `store._load_assertions` builds an unbatched `IN` clause, so **above 32,766 active
   assertions every serving and sweeping path raises** `too many SQL variables` — and
   `stale.py` batches its own queries with a comment about exactly this risk. Apply
   `stale._chunks`. Test at `_BATCH + 1` and 33k.
7. `_atomic`'s "join the caller's transaction" is a *connection*-global sniff, and the MCP
   server shares one connection across MCP's thread pool. Demonstrated: the agent receives
   `{'assertion_id': 1, 'servable': True}` for a row that no longer exists. Give
   `IndexSource` a `threading.local()` connection, and replace the sniff with an explicit
   `join:` / savepoint.

## WP18 — Documentation and packaging drift

**Severity: MEDIUM, but cheap and high-embarrassment. Docs done at this commit; the code
items are NOT.** Everything below was re-checked at `3212972` before being marked.

| # | Item | State |
|---|---|---|
| 1 | "the inference layer is not built yet" in `stats` and at README:14 | **done.** `stats` now prints "always 0 here: inference lives in assertions, not on edges" (landed `2f299c8`); the README intro is rewritten |
| 2 | Console scripts not installed | **OPEN `[V]`.** `.venv/bin/codelearner` and `.venv/bin/codelearner-mcp` still do not exist, and there is no `code_learner*` dist-info in `site-packages` at all — the package is not installed, and `import codelearner` works only from the checkout. Documented in the README as the truth rather than papered over; the fix (`pip install -e .`) is still not applied |
| 3 | Python-only never stated; a non-Python repo indexes to 0 files with exit 0 | **half done.** "Scope: Python only" is now the second line of the README. The guard is **OPEN `[V]`** — reproduced at `3212972` against a one-file Go repo: `files 0 … resolved 0`, exit 0 |
| 4 | `stats` blind to the assertion store while MCP `index_stats` reports it | **done** (`2f299c8`). `stats` prints an `assertions (tier 2)` block |
| 5 | Qualname collisions silently drop symbols while `stats.symbols` counts them | **OPEN, and not re-verified by WP18.** No `collisions` counter exists. Needs the 2,556-file tree the auditor used |
| 6a | schema `v4`→`v6` in sample output | **done** — all sample output re-captured at `3212972` |
| 6b | code-learner self-stats row off by ~12× | **done.** It was `9 files / 79 symbols / 454 edges`; the truth at `3212972` is `69 / 1,714 / 9,475` — off by 7.7× on files and **21.7× on symbols**, so the audit's "~12×" was itself an average over two very different errors |
| 6c | `learn` flag table missing `--host`/`--quiet` | **done** — and it was also missing `--repo`/`--index-path` |
| 6d | `submit_assertion` refusal table missing `unknown_subject` | **already closed before WP18, and the brief for WP18 was wrong about it.** `unknown_subject`, `evidence_stale`, `invalid_span`, `empty_claim` and `evidence_unverifiable` were all present. What was *actually* missing, checked against `app.ERROR_CODES`: `span_escapes_repo`, `bad_path`, `file_too_large`, `too_many_spans`, `claim_too_long`, `bad_confidence`. Now listed |
| 6e | library quickstart raising a bare `IntegrityError` on second run | **not re-verified by WP18** — carried forward |
| 6f | `.codelearner/` not mentioned for the user's `.gitignore` | **done** |
| 6g | `gate_controls` invocation emitting a `RuntimeWarning` | **OPEN `[V]`.** Still emitted at `3212972`: `RuntimeWarning: 'codelearner.eval.gate_controls' found in sys.modules after import of package 'codelearner.eval'`. Cosmetic, but it is the first line a reader of the gate numbers sees |

### Found during WP18, not in the original audit

| Item | Evidence |
|---|---|
| **`_symbol_bytes_at`'s docstring still says the figure is "NOT STATED HERE PENDING RE-MEASUREMENT (WP8)"** | `server/app.py:595-602`. WP8 landed at `2f299c8` and re-measured it; the paragraph the WP8 acceptance criteria said to replace was never replaced. This is amendment 13's failure mode in a docstring rather than a comment: a figure that lives in source and cannot be regenerated by a command. `[V]` |
| **`fuse.reciprocal_rank_fusion`'s docstring quotes the superseded 16-query numbers as current** | "moves recall@10 from 0.635 to 0.781 and MRR from 0.331 to 0.516", and "All 16 gold queries are of the form how does X work". At `3212972` on 520 queries the demotion is +0.116 [+0.101, +0.133] nDCG@10, and the gold set is 638 rows over 6 repos in 3 sources. Same failure as `ablation.py`'s deleted MEASURED block, in a module Wave 4 did not touch. `[V]` |
| **`test_console_script_is_registered` pins the *declaration* and not the *installation*** | `tests/test_cli.py:1160` reads `pyproject.toml` and asserts the entry point is declared. It passes while `.venv/bin/codelearner` does not exist. A green test over a broken behaviour, in the repo whose thesis is that these are different things. The smoke test WP18.2 asked for — that the entry point *exists* — is the one that is missing. `[V]` |
| **`is_test_path` misses `test.py`** | TradingAgents ships exactly one test file, `test.py` at the repo root, which matches none of `tests/`, `test_*.py`, `*_test.py`, `conftest.py`. Consequence measured, not hypothesised: `prefer_implementation` is a no-op on that repo, its per-repo delta is exactly `+0.000 [+0.000, +0.000]`, and it dilutes the pooled row (amendment 21). Whether to widen the convention or keep refusing to guess is a judgement call; it is currently made by accident. `[V]` |
| **`docs/EXAMPLE-TOUR.md` carries stale header counts** | "68 files, 1,095 symbols, 6,531 reference edges"; swarm-sync@`3119a97` is 75 / 1,345 / 8,232. The tours themselves are still valid output of the algorithm, just of an earlier tree. Flagged in the README rather than regenerated, because regenerating it is a separate change to a file WP18 does not own. `[V]` |
| **`docs/PHASE0-FINDINGS.md` quotes 63.5% as a current rate** | It is a historical measurement of the 76.2% → 63.5% decision and is fine as history; the README now stamps it as such and gives 63.8% as the rate at `3119a97`. `[V]` |
| **The staleness A/B performance table cannot be reproduced from a clean checkout** | Its synthetic rows need fixtures that are not checked in, and its two real rows were taken on smaller versions of both repos. Kept in the README under an explicit "previously measured, not re-run" box, because the *shape* argument (`O(spans)` vs `O(bytes)`) is what it was ever for. Re-deriving it is open work. `[V]` |

---

# Deferred, with reasons

- **Incremental re-index.** `files.content_hash`, `mtime_ns`, `size_bytes` and
  `chunks.text_hash` are written and never read; `embed_chunks`' documented "incremental by
  default" keys on `chunk_id`, not `text_hash`, and cannot fire because `--force` deletes the
  file `[A]`. WP7 (store carry-over) gets ~90% of the value at a fraction of the cost. Fix the
  false docstring now; mark the four columns reserved-and-unread; build the real thing later.
- **Prompt injection from indexed source** `[H]`. Repository source is interpolated into both
  generator and judge prompts with no fencing and no instruction to treat it as data. The
  blast radius is genuinely bounded — citations are integers into a server-built menu, so no
  model-emitted string becomes a path, offset, SQL value or shell argument — so the realistic
  impact is an inflated faithfulness number, not code execution. **Not demonstrated** (the
  auditors were barred from model calls). Confirm with a stub generator before fixing.
- **Second-language support.** The type seam (`Symbol`/`Edge`/`FileExtract`) is clean and
  SQL-free — the expensive half is done right. Missing is dispatch: `*.py` is hardcoded in two
  places plus a second consumer in `eval/`. Add the registry when a second language is
  actually wanted.
- **Judge calibration against human labels.** The highest-value measurement in the audit, but
  it needs ~30 human labels. Cannot be delegated to an agent.
- **`trust_remote_code=True` on an unpinned HuggingFace reranker** — pin `revision=<sha>`.
  Trivial, but opt-in and offline-safe, so not urgent.

---

# What is genuinely good, and should not be "improved"

Recorded because a plan this long distorts the picture, and because these are the parts a
refactor is most likely to damage:

- **Citation by integer.** A model *cannot express* a fabricated citation. This is a
  structural guarantee, not a check, and it earned its keep on the first real run — 23
  off-menu references across 10 drafts that would each have been a permanently-verifying
  pointer at unrelated code under a byte-offset design.
- **The mutation result, now at both doors.** Deleting each rule causes 100% of that family's
  attacks to be admitted: **23/23 negative rules** across the direct and store doors at
  `3212972`, mutant hold rate 0.000 on every instance. Reproduced independently while writing
  WP18. Still the best-verified thing in the repo — and note that it grew from "nine families"
  to 23 (family, door) pairs without a single one weakening, which is the outcome a second
  measurement surface is supposed to have and usually does not.
- **No `eval`/`exec`/`pickle`/`shell=True` anywhere**; the analysed repo is never imported or
  executed; SQL is fully parameterised across every `noqa: S608`; path traversal via citations
  is correctly refused; git argv is injection-safe; graph expansion is provably bounded.
- **Crash safety and multi-process concurrency.** SIGKILL mid-transaction and two processes ×
  150 writes both came through clean.
- **`tests/test_assertions.py` and `tests/test_stale.py`** are the standard the rest of the
  suite should be measured against.
- **The retraction box.** The instinct that produced it is the reason this audit found what it
  found — and WP11 is that instinct applied one level deeper. It now sits in the README's
  reranking section carrying two retractions rather than one: the original figures that did
  not reproduce, and the corrected figures that reproduced perfectly and were measured below
  the calibration floor. The second is the harder one to write, because nothing about it
  *looked* wrong.

---

# Open at the close of WP18

Everything not marked done above, in the order a next wave should take it. This list is the
answer to "what does this project still owe", and it is deliberately at the end of the
document rather than folded into the status table, because a status table with 25 rows hides
a short list of real debts.

**Correctness and behaviour, ranked.**

1. **WP17.6 — the unbatched `IN` clause.** Above 32,766 active assertions *every* serving and
   sweeping path raises. It is a one-function fix (`stale._chunks` already exists), it is the
   only item on this list that turns into a hard outage, and the largest store in existence
   today is 150 rows — so it is cheap now and will not be later.
2. **WP16.5 — the MCP embedder-mismatch guard is unpinned** and its CLI twin is not. The agent
   surface can silently serve vectors from a different model, which is the failure `stats` and
   `search` both have prose about.
3. **WP17.7 — `_atomic`'s connection-global transaction sniff** against MCP's thread pool.
   Demonstrated by an auditor: the agent receives `{'assertion_id': 1, 'servable': True}` for
   a row that no longer exists.
4. **WP18.3 — the Python-only guard.** A Go repo indexes to zero files and exits 0. Reproduced
   at `3212972`. The README now says "Python only"; the tool still does not.
5. **WP18.2 — install the console scripts**, and add the smoke test that the entry point
   *exists* rather than the one that checks it is declared.
6. **WP17.4 — adjudication as admission control.** Either ship `codelearner judge` plus
   `servable_assertions(require_verdict=True)`, or keep it offline; what is not tenable long-
   term is a faithfulness number about claims that are served regardless of it.
7. **WP16.6, WP16.9** — pipeline subject-drop accounting, and pinning that the un-verified
   reader does not mutate.
8. **WP16.3, WP16.4, WP16.7 — re-run the mutations.** Tests exist that name the right
   behaviour. Nobody has deleted the fix and watched them fail, which by this repo's own
   standing rule means they are not yet known to be fixes.

**Measurement debts.**

9. **Reranking has no current number.** The 16-query result is withdrawn (below
   `CALIBRATION_FLOOR`) and no replacement exists. Re-run `run_ablation_multi(..., reranker=)`
   on the 638-row set and publish stamped. Attempted during WP18 and abandoned: the reranker
   OOM-thrashed against a 10GB card that had the embedder resident, so it needs the GPU to
   itself.
10. **The judge's flip rate under prompt perturbation** — the largest single uncertainty in
    0.54, and the harness for it has shipped unused since `1021a14` (amendment 17).
11. **Judge calibration against ~30 human labels** on the set the number is actually computed
    over. Still the highest-value measurement in the whole audit, still not delegable.
12. **The generation run's purpose table predates the WP13 corrections.** Its six rows carry
    the overstated levels and an uncorrected null; only their ordering is readable. Re-run
    `score_purposes` for those two LLM conditions under the corrected blinding and the
    500-derangement null.
13. **The staleness A/B performance table** cannot be reproduced from a clean checkout. Either
    check in the synthetic fixtures and re-derive it, or cut it and keep only the `O(spans)`
    vs `O(bytes)` argument.

**Documentation drift that WP18 found and could not fix, because it lives in `.py` files.**

14. `server/app.py:595-602` still says the symbol-bytes figure is "PENDING RE-MEASUREMENT
    (WP8)". WP8 landed four commits ago and measured it: 438 of 1,714 at `3212972`.
15. `retrieve/fuse.py`'s `reciprocal_rank_fusion` docstring quotes the superseded 16-query
    demotion numbers as current, and asserts "all 16 gold queries" of a 638-row gold set.
16. `docs/EXAMPLE-TOUR.md`'s header counts are from an earlier swarm-sync.

Items 14 and 15 are the same failure as the `ablation.py` MEASURED block that Wave 4 deleted,
in two modules Wave 4 did not touch — which suggests the right fix is not another correction
but the rule amendment 13 states: **a figure that cannot be regenerated by a command does not
belong in a docstring.** A grep-based test over the retracted figures already exists for
`ablation.py`; widening it to the package is cheaper than re-auditing prose every wave.
