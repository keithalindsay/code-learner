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
| WP4 consolidate the gate | in flight | |
| WP19 remaining unguarded reads | in flight | |
| WP5, WP7–WP18 | not started | |

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
WAVE 0  WP1 WP2 WP3            security containment — no dependencies, do first
WAVE 1  WP4 WP5 WP6            make the gate one gate
WAVE 2  WP7 ────────────────►  re-index preserves the store   ── MUST PRECEDE WP8
WAVE 3  WP8 WP9 WP10           decorator fix + schema v6
WAVE 4  WP11..WP18             measurement, tests, architecture, docs
```

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

**Severity: HIGH.** All `[A]`, each with a re-run in the auditor's report.

| Claim | Reality |
|---|---|
| `1,764 pairs, 0 findings` (leak audit) | **1,849 pairs, 2 findings** — a 32-char clause from one symbol's held-out label appears verbatim in another symbol's generator input |
| "Mined prose is, if anything, a slightly better retrieval query than a hand-written question" + comparator `MRR 0.221 / hit@10 0.435` | **Reverses on re-run.** Mined 0.280/0.419/0.512 vs hand 0.309/0.562/0.562 — worse on all three |
| `+0.226 MRR swing … safe to believe` (README:274) | Quotes the **retracted** delta. Live figure is **+0.161** |
| `ablation.py:246-261`, headed "MEASURED" | Contains all four retracted reranking rows and three conclusions drawn from them, two of which the README says did not survive `[V]` |
| `42 / 316 / 13.3% / 17 commits / lift 0.161` table | Superseded by the 43/318/13.5%/18 table ~170 lines later, presented as co-equal |
| `0.6 → 0.615` sweep row | Not producible by the shipped sweep (`(0.3, 1.0, 1.5)`) |
| `6,091 / 1,688 / 850 symbols` | Now **7,717 / 2,145 / 1,079** — scales with repo size, uncaptioned |

**Change.** Correct each in place. **Stamp every published table with `repo@sha`** — the
deterministic conditions reproduced exactly at the current sha, so this is bookkeeping, not
instability, which is precisely why leaving it uncaptioned is worse than it looks.

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
perturbations to publish a label-flip rate.

---

# WAVE 4 — tests, architecture, docs

## WP16 — Close the surviving mutations

**Severity: HIGH.** All `[A]`, each with the exact edit that survived.

Ranked; each acceptance criterion is "this mutation now fails the suite".

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

**Severity: MEDIUM.** All `[A]` except where marked.

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

**Severity: MEDIUM, but cheap and high-embarrassment.**

1. **`cli/commands.py:438` prints "the inference layer is not built yet"** `[V]`, next to a
   count that is structurally always zero. README:14 and README:941 say the same. Phase 9
   shipped. *(This one is mine — my Phase 9 edit updated the roadmap and missed the intro.)*
2. **The console scripts are not installed** `[V]` — `[project.scripts]` declares
   `codelearner`, the venv has neither it nor `codelearner-mcp`, and the dist-info predates
   the declaration. **Every README example currently fails in the author's own environment**;
   the working invocation is `.venv/bin/python -m codelearner.cli`. Re-run `pip install -e .`
   and add a smoke test that the entry point exists.
3. **Python-only is never stated**, and a Go/TS/Rust repo indexes to zero files with exit 0.
   Guard: exit 1 with an explanatory message when `stats.files == 0`; add "Scope: Python only"
   under the opening paragraph.
4. `codelearner stats` is blind to the assertion store while MCP `index_stats` reports it —
   two surfaces over one index disagreeing, which `server/__init__.py` argues must not happen.
5. **Qualname collisions silently drop symbols** while `stats.symbols` counts them (49,463
   reported vs 49,292 actual on a 2,556-file tree). Count `cursor.rowcount`; emit a
   `collisions` counter. Note `app.py:582` claims qualname is *not* unique, contradicting the
   schema's UNIQUE index.
6. Remaining drift, one line each: schema `v4`→`v5` in sample output; the code-learner
   self-stats row off by ~12×; `learn` flag table missing `--host`/`--quiet`; the
   `submit_assertion` refusal table missing `unknown_subject`; the documented library
   quickstart raising a bare `IntegrityError` on second run; `.codelearner/` not mentioned for
   the user's `.gitignore`; the `gate_controls` invocation emitting a `RuntimeWarning`.

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
- **The nine-family mutation result.** Deleting each rule causes 100% of that family's attacks
  to be admitted. Independently reproduced. The best-verified thing in the repo.
- **No `eval`/`exec`/`pickle`/`shell=True` anywhere**; the analysed repo is never imported or
  executed; SQL is fully parameterised across every `noqa: S608`; path traversal via citations
  is correctly refused; git argv is injection-safe; graph expansion is provably bounded.
- **Crash safety and multi-process concurrency.** SIGKILL mid-transaction and two processes ×
  150 writes both came through clean.
- **`tests/test_assertions.py` and `tests/test_stale.py`** are the standard the rest of the
  suite should be measured against.
- **The retraction box at README:229-238.** The instinct that produced it is the reason this
  audit found what it found — and WP11 is that instinct applied one level deeper.
