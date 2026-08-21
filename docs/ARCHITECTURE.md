# Architecture

[← code-learner](../README.md) · the case study

How the system is built: the three tiers and the schema that keeps them honest, the tier-2 assertion store that is a gate rather than a pipeline, the two-stage staleness engine, and the deterministic onboarding tours.

---

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

T0's guarantee has one hole and the tool now says so out loud. "Reproducible from
source alone" is true of the parse and false of the *index*, which is a snapshot: an
edit after indexing moves the line numbers a T0 row is still serving. Tier 2 had a
two-stage staleness engine and tiers 0 and 1 had nothing at all, so the fact tier was
the one with no drift check. Every `search` and `stats` call now stats the whole file
set — no sampling; timed at 58ms over a 2,556-file tree when it was built, a figure not
re-run for this document — and reports three counts kept apart, because
their symptoms differ: `changed` (citations moved, results still look right, the
dangerous one), `missing`, and `unindexed` (results simply absent). The check's own
limit is stated in the message it prints, so the counts read as floors rather than as
an audit, and there is no flag to suppress it: the note fires only when the index is
genuinely behind, so silencing it would silence the only signal that T0 answers have
stopped being facts.

### The T2 assertion store

The storage layer for tier 2 is `codelearner/assertions/`, schema v6. It is the gate,
not the pipeline — nothing in it calls a model. What it does is refuse the ways an
inferred claim becomes unaccountable:

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

**The gate is one gate, and it did not start that way.** `write_assertion` is
documented as "exactly one place in the project that can decide an uncited claim is
inadmissible" and for a long time it enforced exactly one rule: `spans` is non-empty.
Hash verification and subject-existence lived only in `server/app.py`, so
`codelearner learn` and every library caller entered through a door with one lock.
Two consequences were reproduced during the audit that found this: an empty claim
with a perfectly valid citation was admitted, stored active and served; and a
zero-length span was admitted and then verified forever against any file content,
because sha256 of nothing is stable. Six rules now raise before the transaction
opens: `EvidenceRequired`, `EmptyClaim`, `InvalidSpan`, `EvidenceUnverifiable`,
`UnknownSubject`, `EvidenceStale` — plus `SpanEscapesRepo`, which was added when
running the negative-control corpus at this second door found repo containment
missing from it entirely.

Spans are hashed, not files. An edit elsewhere in a 2,000-line module leaves a claim
about one function in it alone; staleness that fires on everything is staleness
nobody reads. Two schema decisions carry the rest: `subject_symbol_id` is `ON DELETE
SET NULL` beside a `NOT NULL` qualname (a `CASCADE` would mean a routine re-index
silently empties the store), and `evidence_spans.path` is plain text rather than a
reference to `files(id)` — because an assertion that loses its last span doesn't
become unsupported, it becomes *vacuously* supported. "Every cited span still
matches" is trivially true of no spans, and reads as success everywhere it isn't
specifically looked for. The reader checks for an empty evidence set anyway.

That `ON DELETE SET NULL` clause is worth one more sentence, because for most of this
project's life it was correct, load-bearing in the argument, and **had never
executed** — nothing in the package deleted a `symbols` row. The only re-index the
tool offered was `--force`, which unlinked the whole database and warned about
embeddings. Embeddings are re-derivable in minutes; verdicts and the rejected set are
not re-derivable at all. `--force` now refuses outright when a tier-2 store is
present and names the counts, and `--carry-assertions` dumps the store to a sidecar,
`os.replace`s it into position, rebuilds, and restores. The sidecar is a second
SQLite file rather than memory on one criterion: it has to survive the process, and
the process dying *is* the failure mode. Recovery is automatic — a crash after the
delete leaves the sidecar, and the next plain `codelearner index` finds it, restores
it, and says so.

### The staleness engine

`servable_assertions` re-hashes every cited byte range on every call, which is
`O(cited bytes)` per query. `codelearner/assertions/stale.py` is the two-stage version
of the same check:

1. **`stat()`.** If a cited file's `st_mtime_ns` *and* `st_size` are exactly what they
   were when that span was last actually hashed (`span_verifications`), nothing is
   read.
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

`refresh_staleness(conn, repo_root)` sweeps every active assertion and reports counts.
Three consecutive sweeps of swarm-sync@`3119a97`'s live store, code-learner@`3212972`:

```
1st  checked=73 fresh=73 expired=0  files_statted=37 files_read=37  spans_fast_pathed=0   spans_hashed=126
2nd  checked=73 fresh=73 expired=0  files_statted=37 files_read=0   spans_fast_pathed=126 spans_hashed=0
3rd  checked=73 fresh=73 expired=0  files_statted=37 files_read=0   spans_fast_pathed=126 spans_hashed=0
     force_hash=True                files_statted=37 files_read=37  spans_fast_pathed=0   spans_hashed=126
```

The first sweep fast-paths nothing, and that is correct rather than a warm-up
artefact: `span_verifications` is deliberately **not** carried across a re-index, so a
store that has just been restored has no record of when anything was last hashed. It
is the one piece of state most likely to be wrong after a repo moved, and the one able
to authorise skipping the re-read that would find out. The sweep is not what keeps the
index honest — serve-time verification is, and it has no window — but it reaches the
claims nothing ever queries again, which is most of them, and its `spans_hashed` vs
`spans_fast_pathed` split is the only evidence anyone gets that the fast path is
working at all.

The four terminal failure modes stay apart (`hash_mismatch`, `file_missing`,
`span_truncated`, `no_evidence`) because they call for different repairs. A fifth,
`decorators_excluded`, was added for a narrower case described under
[the decorator span](RESULTS.md#the-decorator-span-and-the-citations-it-could-not-reach).
Serving withholds stale claims by default and returns them only under
`include_stale=True`, always labelled.

A sixth reason, `unreadable`, is deliberately **not** terminal. A cited file that is
present but cannot be opened — a permission bit, `EMFILE`, `EIO`, an NFS blip, a FIFO —
proves nothing about the bytes, so the claim is withheld for that call and its status is
left alone; it returns by itself on the next healthy read. Only real absence
(`FileNotFoundError`, `NotADirectoryError`) expires a claim. Before this split a
`chmod 000` expired every claim citing the file, logged `file_missing` for a file that
was sitting there, and `chmod 644` brought nothing back.

Withholding also applies under `include_stale=True`, behind a second opt-in, and that
is the decision worth defending: the natural way to consume a flag is `if r.stale:
... else: <treat as fresh>`, so a record arriving `stale=False` with nothing verified
lands in the `else` and is presented as checked — the cached-freshness failure
delivered through the very flag added to prevent it.

`reinstate(conn, id)` is the way back for claims already expired: it re-hashes every
citation and promotes `stale` to `active` only on an exact match of all of them. It
refuses a `rejected` claim — a refuted claim is not an expired one, and a re-hash has
nothing to say about a verdict — and it has no override flag, because a promotion that
skipped the re-read would be a cached freshness verdict entered by hand.

#### The number that did not go the way it was supposed to

> **Previously measured, not re-run for this document, and — the part that matters —
> the sha it was taken at was never recorded.** It predates the stamping convention at
> the top of this file, which is the convention's own best argument: the two real rows
> are 383 and 1,100 spans, and the two repos hold 1,714 and 1,345 symbols today, so it
> is certainly a measurement of smaller trees, and there is no way from here to say
> which. The method is known — serving every claim in an index, median of interleaved
> A/B blocks, warm page cache — and the synthetic rows need fixtures that are not
> checked in, so the table cannot be reproduced from a clean checkout as it stands.
> Kept for the *shape* it shows rather than for its constants, which is also all that
> was ever claimed for it. Re-deriving it, stamped, is open work.

| index | spans | cited bytes | always-rehash | two-stage | ratio |
|---|---|---|---|---|---|
| code-learner | 383 | 0.28 MiB | 4.97 ms | 6.91 ms | **0.72×** |
| swarm-sync | 1,100 | 0.95 MiB | 13.99 ms | 20.09 ms | **0.70×** |
| synthetic, 8 × 128 KiB | 168 | 1.01 MiB | 3.21 ms | 3.53 ms | 0.91× |
| synthetic, 8 × 256 KiB | 168 | 2.01 MiB | 4.35 ms | 3.41 ms | 1.28× |
| synthetic, 8 × 512 KiB | 168 | 4.01 MiB | 5.03 ms | 2.62 ms | 1.92× |
| synthetic, 8 × 1 MiB | 168 | 8.01 MiB | 8.65 ms | 2.60 ms | 3.33× |

On both *real* repositories the fast path was about **1.4× slower**. The premise was
wrong: sha256 over a page-cached Python file is far cheaper than assumed — re-hashing
all of swarm-sync's cited bytes cost under a millisecond — while the extra
`span_verifications` lookup and the per-span record-keeping cost a few microseconds per
span whether or not anything moved. The crossover is around 1.5 MiB of cited bytes per
query; below that, the unconditional re-hash wins.

What the table shows is the shape rather than the constant. The two-stage column is
*flat* in file size (2.6–3.5 ms across an 8× range) because it is `O(spans)`; the
re-hash column grows because it is `O(bytes)`. So the two-stage check is the one that
holds up as cited volume grows, and the only one whose cost does not depend on how the
filesystem feels that day — a page-cache miss, an NFS mount or an encrypted volume
moves the re-hash column and leaves this one alone. Both verifiers ship, a test asserts
they reach identical verdicts across an enumerated list of repo states — untouched,
edited, deleted, truncated, touched, unreadable, and not-a-regular-file — and on a small
warm repo the unconditional one is genuinely the better choice. "Every failure mode" is
what that test used to claim; the list was five states, and `unreadable` — the sixth —
was the one on which the two verifiers were in fact disagreeing, the fast path serving a
`chmod 000` file as `fresh, method='stat'` off an unchanged mtime while the reference
verifier could not open it at all. Stage one now tests readability so it cannot reach a
conclusion stage two would refuse.

An accelerator that reaches conclusions the authority would refuse is not accelerating
anything, and the two-stage engine still has **no production caller**: it is a measured
alternative that is not wired, and `span_verifications` — the entire reason for the
v4→v5 schema bump — is written and read by `refresh_staleness` and by tests. That is
tracked as WP17.5 and is not fixed.

---

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
verbatim at [docs/EXAMPLE-TOUR.md](EXAMPLE-TOUR.md) — including a note on
where the ordering runs out of information and the tie-break stops being defensible.
That file was generated against an earlier swarm-sync and its header counts are from
that tree, not from `3119a97`.
