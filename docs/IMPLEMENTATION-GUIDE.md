# Code Learner completion guide

Implementation plan for turning the current Python code index and accountable assertion
store into a complete, production-capable Graph RAG system.

This document is written as an execution guide. Each work package has a bounded owner,
explicit inputs and outputs, likely files, acceptance tests, and a handoff contract. An
expert agent should be able to take one package without having to reinterpret the product
architecture.

The current system already has three strong substrates:

1. A Python symbol graph with explicit unresolved edges and confidence-bearing resolution.
2. Lexical, dense, and graph-expanded retrieval with optional reranking.
3. An evidence-bound assertion store with verdict history and source-span staleness.

The central missing capability is integration. Search retrieves source symbols, while
semantic assertions are only found after a caller already knows a symbol's qualname. The
target system must retrieve source, structure, and supported assertions together and return
a bounded evidence package suitable for an agent or an answer generator.

---

## 1. Definition of complete

A v1 Graph RAG system is complete when it can perform this lifecycle:

```text
repository
    -> deterministic structural index
    -> source chunks and embeddings
    -> generated or submitted semantic claims
    -> evidence validation and independent adjudication
    -> searchable semantic index
    -> query-intent-aware structural and semantic retrieval
    -> bounded, line-numbered evidence response
    -> freshness enforcement after source changes
    -> incremental repair of affected facts, vectors, and claims
```

For a user query, the returned evidence must answer five questions without requiring the
caller to infer hidden state:

- What source is relevant?
- What symbols and relationships connect it?
- What does the system infer the code is for?
- What evidence supports each inference?
- How fresh and trustworthy is each component?

The following are required v1 properties:

- Python repositories either index successfully or fail with an actionable reason.
- Structural results identify their tier and resolver provenance.
- Semantic claims participate directly in retrieval.
- Rejected, stale, unreadable, and policy-ineligible claims are not presented as fresh.
- Claims can be generated, judged, searched, expired, regenerated, and audited through
  shipped interfaces.
- Search can return usable source, not merely locations.
- Every response has a hard size budget and deterministic truncation behavior.
- Reindexing does not silently destroy human or model adjudication history.
- A completed end-to-end benchmark demonstrates correctness and usefulness against a bare
  agent and a structural-only code graph.

Not required for v1:

- A second programming language.
- Cross-repository linking.
- A continuously running file watcher.
- Autonomous answer generation inside the server.
- Receiver-blind dynamic-dispatch guesses.
- A distributed graph database.

---

## 2. Target architecture

Keep SQLite as the system of record. The current graph fits the workload, has useful
transaction semantics, and makes provenance inspectable. Do not introduce a graph database
until measured traversal or concurrency limits require it.

```text
                       +-------------------------+
source files --------> | extraction + resolution | ------+
                       +-------------------------+       |
                                                            v
                       +-------------------------+    symbols / edges
source spans --------> | chunks + embeddings    | ------+
                       +-------------------------+       |
                                                            |
                       +-------------------------+       |
claims + citations --> | assertion admission     |       |
                       +-------------------------+       |
                                  |                         |
                                  v                         |
                       +-------------------------+          |
                       | adjudication policy     |          |
                       +-------------------------+          |
                                  |                         |
                                  v                         v
                       assertion lexical/vector indexes
                                  |                         |
query ----------------------------+-------------------------+
                                  v
                       +-------------------------+
                       | query planner           |
                       | source + T2 + graph     |
                       +-------------------------+
                                  |
                                  v
                       +-------------------------+
                       | evidence assembler      |
                       | budget + citations      |
                       +-------------------------+
                                  |
                                  v
                     CLI / MCP / library response
```

### Architectural rules

1. **One policy, one home.** CLI and MCP may adapt errors and presentation, but admission,
   servability, tiering, freshness, and adjudication policy live below both interfaces.
2. **Retrieval does not confer truth.** Ranking a claim highly never makes it servable.
   Eligibility is decided before or during candidate creation.
3. **Assertions and source are different record types.** Do not force both through the
   existing `Hit` shape if that erases verdict and evidence metadata. Introduce a common
   retrieval candidate protocol or a tagged union.
4. **Every expensive derived artifact has an identity.** Embeddings record model, dimension,
   normalization, prompt/version, and source hash.
5. **Every bounded response is deterministic.** Given an index, query, configuration, and
   budget, selection and truncation must be reproducible.
6. **The source tree is adversarial input.** Repository text may influence ranking and model
   output, but must never become instructions, paths, SQL, shell arguments, or unbounded
   allocations without validation.

---

## 3. Delivery overview

| Phase | Outcome | Exit gate |
|---|---|---|
| 0 | Baseline and decisions frozen | Current behavior reproducible; schemas and policies agreed |
| 1 | Source-rich evidence responses | One search call returns bounded whole-symbol source |
| 2 | Semantic assertions are searchable | T2 claims measurably affect search and `facts_only` |
| 2.5 | Semantic retrieval is *measured* | Dense assertion retrieval and a five-repository gold set decide whether it helps |
| 3 | Adjudication is production policy | Unjudged claims can be withheld by default |
| 4 | Query-aware graph retrieval | Impact and relationship intent produce appropriate traversals |
| 5 | Incremental freshness lifecycle | Edits repair facts and expire/regenerate claims safely |
| 6 | Packaging, concurrency, observability | Installed product survives load and operational failures |
| 7 | End-to-end evaluation | Correctness and semantic lift measured against controls |
| 8 | v1 release decision | Release criteria met and product strategy selected |

Phases 1 and 2 are the shortest path to proving the product thesis. Phases 3–6 make that
proof safe to ship. Phase 7 determines whether it is worth productizing.

---

## 4. Phase 0 — freeze the baseline and contracts

### Objective

Prevent implementation agents from optimizing against stale documentation or irreproducible
numbers.

### WP0.1 — Current-state reconciliation

**Owner:** architecture/documentation agent

The historical remediation document intentionally preserves old findings, but several items
listed as open have since landed. Produce a concise `docs/CURRENT-STATE.md` generated from the
current commit rather than rewriting history.

Required content:

- Current commit and schema version.
- Supported languages and interfaces.
- Shipped versus experimental features.
- Current open correctness, scale, UX, and evaluation gaps.
- Which old remediation entries are superseded and by which commit.
- Exact commands for deterministic tests and benchmarks.

**Acceptance:** every assertion in the status document links to code, a test, or a reproducer.

### WP0.2 — Retrieval contracts

**Owner:** retrieval architecture agent

Define these types before adding semantic retrieval:

```python
Candidate = SourceCandidate | AssertionCandidate

SourceCandidate:
    symbol_id, qualname, path, span, source_hash
    score, modalities, tier, resolver provenance, is_test

AssertionCandidate:
    assertion_id, subject_symbol_id, subject_qualname
    claim, kind, generator, confidence
    verdict summary, evidence spans, freshness
    score, modalities, tier=T2

EvidenceSection:
    candidate identity, title, line-numbered content
    provenance, citations, score explanation, truncation state
```

Decide whether these are dataclasses, protocols, or typed dictionaries. Prefer dataclasses in
the retrieval core and stable dictionaries only at the interface boundary.

Specify:

- Stable identity and deduplication keys.
- Score ownership before and after fusion.
- Tier calculation for mixed candidates.
- How an assertion and its subject source reinforce one another.
- Whether multiple assertions about one subject occupy independent result slots.
- Serialization versioning.

**Acceptance:** type-level tests show CLI and MCP serialize the same candidate semantics.

### WP0.3 — Serving policy decision

**Owner:** semantic safety agent

Write a policy object rather than spreading booleans through call stacks:

```python
ServingPolicy(
    max_tier=2,
    require_verdict=True,
    accepted_verdicts={"supported"},
    minimum_supporting_judges=1,
    allowed_judges=None,
    allow_stale=False,
    allow_unreadable=False,
    freshness_mode="hash",
)
```

Keep `facts_only` as a compatibility shorthand for `max_tier=1`.

**Acceptance:** one policy evaluator is used by search, `get_symbol`, CLI, MCP, refresh,
and future answer generation.

---

## 5. Phase 1 — return complete evidence

### Objective

Close the largest measured usability gap with CodeGraph: a search result should be directly
readable and citeable without a second filesystem operation.

### WP1.1 — Evidence assembler

**Owner:** source presentation agent

Add a package such as `codelearner/evidence/` containing:

- Source slicing by indexed byte coordinates.
- Line-number rendering.
- Whole-symbol section construction.
- Response budgeting.
- Deterministic section selection.
- Explicit omissions.

Suggested API:

```python
assemble_evidence(
    conn,
    repo_root,
    candidates,
    *,
    budget: EvidenceBudget,
    include_source: bool = True,
) -> EvidenceBundle
```

`EvidenceBudget` should support at least compact, standard, and deep presets plus an explicit
byte or token ceiling. Enforce a server-side maximum regardless of caller input.

Rules:

- Never cut through a source symbol.
- Never trust a caller-supplied path or byte range.
- Re-read the indexed path through existing containment and file-type guards.
- Compare current bytes with indexed hashes before presenting source as current.
- If a complete section does not fit, omit it and record the omission.
- Prefer higher-ranked candidates, then evidence needed by included assertions.
- Avoid returning the same source span twice.
- Include `truncated`, `sections_omitted`, and `budget_used` metadata.

### WP1.2 — MCP and CLI integration

**Owner:** interface agent

Extend `search_code` with compatible optional parameters:

- `include_source`
- `budget`
- `view` or `intent`

Do not silently change the existing compact response. Make source-rich output the documented
recommended mode and consider making it the default only in a versioned interface.

CLI should support human-readable and JSON evidence bundles. Both surfaces must call the same
assembler.

### WP1.3 — Evidence security and scale tests

**Owner:** adversarial testing agent

Tests must cover:

- Symlink and path escape attempts.
- FIFO, device, oversized file, unreadable file, and replacement races.
- Invalid UTF-8.
- Deeply nested and very large symbols.
- A top result larger than the entire budget.
- Duplicate spans across source and assertions.
- Stale index coordinates.
- Stable output across repeated runs.

**Phase exit:** one MCP call for a representative question returns useful line-numbered source,
graph provenance, hashes, and a truthful omission report under a hard ceiling.

---

## 6. Phase 2 — make semantic claims searchable

**Status (2026-08-12): implemented on `codex/production-phase-2`, except WP2.4, which
moved to Phase 2.5.** Lexical retrieval over assertion documents, a centralised serving
policy, mixed rank fusion with bidirectional promotion, all-or-nothing assertion
evidence, and identical CLI and MCP candidate semantics are in. The production default
serves a claim only when it is `active`, carries at least one `supported` verdict, has
no `unsupported` or `refuted` verdict, and re-verifies every cited byte range on the
serving call. Pending claims are reachable only through `RESEARCH_PENDING_POLICY`, named
at the library boundary and exposed by neither surface; rejected and stale claims are
servable under no policy at all.

**What is NOT done, and must not be reported as done:** nothing here measures whether
semantic retrieval improves anything. There is no assertion embedding, no semantic gold
set, and no comparison against a source-only control on real repositories. The ablation
carries a tier-2 row that reports a COUNT of claims returned, not a lift, and its
metrics are not comparable with the source rows beside it. Productization criteria are
not satisfied by this phase.

### Objective

Turn the assertion store into an actual RAG modality.

### WP2.1 — Assertion search schema

**Owner:** storage/search agent

Add an assertion retrieval document for every admitted assertion. Recommended textual form:

```text
kind: purpose
subject: package.module.Class.method
claim: Coordinates lease renewal so workers do not lose ownership during long operations.
evidence: package/module.py:120-168
```

Add:

- An FTS5 assertion index.
- Optional sqlite-vec assertion embeddings.
- Embedding metadata separate from source-chunk embeddings if text construction differs.
- Triggers or explicit update functions for insertion, verdict/status changes, staleness, and
  re-linking after reindex.

Do not index rejected or stale claims as eligible search documents. Either remove them from
the live retrieval index while keeping their authoritative rows, or filter them before they
can become candidates. Defense in depth is appropriate.

Schema migration must preserve all assertion, evidence, verdict, and staleness history.

### WP2.2 — Assertion retrieval implementation

**Owner:** semantic retrieval agent

Create `retrieve/assertions.py` with lexical and dense retrieval. It must accept a
`ServingPolicy` and return only eligible claims with evidence and freshness metadata.

Important decision: freshness verification can be expensive. Use a two-step candidate path:

1. Retrieve a wider assertion candidate set from stored indexes.
2. Verify eligibility and freshness for that bounded set.
3. Remove failures and retrieve further candidates if the requested depth is no longer full.

Never rank first and verify only the final `k`; that can return too few results even when
eligible lower-ranked claims exist.

### WP2.3 — Mixed candidate fusion

**Owner:** ranking agent

Extend fusion without pretending source and assertion scores are directly calibrated.
Start with rank fusion and explicit modality weights. Avoid a learned ranker until the gold set
can support it.

Required behaviors:

- A semantic claim may promote its subject source.
- A source hit may promote eligible assertions attached to that symbol.
- Duplicate claims must not fill the result page.
- Conflicting supported assertions remain distinct and are labelled as conflict candidates;
  retrieval must not synthesize a resolution.
- `max_tier=1` removes T2 before final selection and refills vacated slots.
- Tier filtering happens before response truncation.

Add score explanations suitable for debugging but do not expose every internal float by
default.

### WP2.4 — Semantic retrieval gold set — DEFERRED TO PHASE 2.5

**Owner:** evaluation agent

Deferred deliberately rather than dropped. Building the answer key in the same phase as
the retrieval it grades invites the key to be shaped by what the retriever happens to
return; and until the serving policy and the candidate substrate stopped moving, any
number produced here would have measured a moving target. Phase 2 therefore exits on
CORRECTNESS -- the right claims are servable and the wrong ones are not -- and Phase 2.5
exits on EVIDENCE that the modality is worth its cost.

Create queries whose answer cannot be recovered reliably from names alone:

- Why does a guard exist?
- Which invariant prevents a race?
- Why is an apparently redundant branch deliberate?
- What risk motivated a timeout or retry policy?
- What responsibility spans multiple helpers?

Each label must identify:

- Relevant assertions.
- Relevant subject symbols.
- Minimum supporting evidence.
- Hard-negative claims or symbols.
- Whether source-only retrieval could reasonably answer it.

Use at least five repositories and cluster uncertainty by repository. Keep product-authored
questions separate from mined questions.

**Phase 2 exit (met):** semantic queries retrieve supported claims directly; `facts_only`
changes the candidate set and refills it from source; stale, rejected and pending claims
never appear under the default policy. Proved end to end by `tests/test_semantic_search.py`
against a repository whose symbol names reveal nothing about the invariant asked for.

**Phase 2.5 exit (not met, not started):** assertion-specific embeddings and identity
metadata; dense assertion retrieval; mixed lexical/dense assertion ablations; a frozen
semantic gold set over at least five repositories with hard negatives, product-authored
and mined questions kept separate, and uncertainty clustered by repository. Only that
measurement can say whether the semantic layer beats the source-only control.

---

## 7. Phase 3 — adjudication as admission control

### Objective

Make “adjudicated” a shipped guarantee rather than an offline measurement.

### WP3.1 — Judge service abstraction

**Owner:** model integration agent

Extract judging behind a protocol independent of the evaluation package:

```python
class Judge(Protocol):
    identity: str
    prompt_version: str
    def judge(self, claim: ClaimView) -> VerdictDraft: ...
```

The judge receives only the claim and verified cited evidence. It does not receive the whole
repository, commit history, generator trace, or previous judge rationale unless evaluating a
separately specified arbitration policy.

Preserve the existing generator/judge-family collision guard.

### WP3.2 — `codelearner judge`

**Owner:** CLI workflow agent

Add commands for:

- Judge all active unjudged claims.
- Judge one assertion or subject.
- Retry instrument failures.
- Dry run.
- JSON progress output.
- Model, host, timeout, and concurrency selection.
- Resume after interruption.

Differentiate:

- Supported
- Refuted
- Unsupported by cited evidence
- Judge unavailable
- Invalid response
- Evidence changed during judgment

Only the first three are verdicts. Transport and parsing failures must not become rejections.

### WP3.3 — Admission policy and history

**Owner:** semantic policy agent

Implement policy queries efficiently. A claim may have multiple verdicts. Define exact rules
for support, contradiction, superseded judges, and prompt versions.

Recommended v1 default:

- Generated claims require one supported verdict from an allowed judge identity.
- Any later refuted or unsupported verdict moves the claim to a review state or rejects it,
  according to an explicit conservative policy.
- Manually submitted claims are still evidence-bound but carry origin=`manual`; whether they
  require adjudication is configurable.
- A stale claim is never restored merely because it has an old supported verdict.
- Regenerated claims receive new assertion identities rather than overwriting history.

The existing three-state status may need expansion or a derived servability state. Prefer not
to overload `active` with “stored,” “judged,” and “servable.” Consider distinct lifecycle and
serving decisions:

```text
lifecycle: active | stale | rejected | superseded
adjudication: unjudged | supported | contested | unsupported
servable: derived from lifecycle + adjudication + policy + freshness
```

### WP3.4 — Judge calibration

**Owner:** human evaluation agent

Before making `require_verdict` the default:

- Label at least 100 claims by two humans.
- Reconcile disagreements.
- Measure false accepts and false rejects by judge and claim kind.
- Measure prompt perturbation stability.
- Include deliberately unsupported and subtly overbroad claims.
- Publish confusion matrices, not only accuracy.

Set release thresholds before seeing the final result. A reasonable initial target is a false
accept rate below 5% on dangerous `invariant` and `risk` claims, with a separately reported
abstention/instrument-failure rate.

**Phase exit:** production surfaces can require supported verdicts, and the chosen judge policy
has human-calibrated error bounds.

---

## 8. Phase 4 — query-aware graph capabilities

### Objective

Make the graph answer structural intents directly instead of applying one fixed two-hop walk
to every question.

### WP4.1 — First-class impact analysis

**Owner:** graph algorithms agent

Implement an `impact` operation using recursive CTEs or bounded traversal:

- Direct and transitive callers.
- Dependent files.
- Affected test files.
- Minimum path and confidence along each path.
- Cycles.
- Truncation at fanout/depth limits.

Attach a compact summary to `get_symbol`; expose detailed output through a dedicated library
operation and, if tool-surface research supports it, an MCP operation.

Do not present all reachable nodes as equally affected. Preserve distance, edge type, and
weakest resolver confidence.

### WP4.2 — Query intent planner

**Owner:** retrieval planning agent

Start with a transparent deterministic planner, not an LLM planner. Classify a small set of
intents from query terms and explicit caller options:

| Intent | Structural behavior |
|---|---|
| locate/implement | lexical+dense, then outbound helpers |
| callers/usage | direct or transitive inbound calls |
| impact/change | inbound calls, dependent modules, tests |
| inheritance | bases and subclasses |
| tests/verification | test symbols and edges without implementation demotion |
| purpose/why/risk | assertion retrieval weighted higher |

Every inferred intent must be returned in diagnostics and overrideable by the caller.

### WP4.3 — Dynamic dispatch boundaries

**Owner:** resolution agent

Return unresolved or ambiguous receiver calls as explicit boundaries:

- Receiver expression.
- Attribute name.
- Source location.
- Why resolution abstained.
- Candidate count if low-confidence candidates exist.

Add only derivable receiver-type strategies:

- Constructor assignment in local scope.
- Parameter annotations.
- Return annotations for directly visible calls.
- `self` and class attributes.
- Narrow, declared framework patterns with precision fixtures.

Any fallback guess must use a distinct resolver name and low confidence. Measure precision by
strategy and repository before enabling it in the default graph.

### WP4.4 — Graph gating ablation

**Owner:** evaluation agent

The existing graph modality did not independently justify itself on the current retrieval
gold. Test graph connectivity as a gate or reranking feature:

- Text candidates only.
- Text candidates filtered by graph reachability.
- Text candidates reranked by intent-appropriate graph features.
- Existing spreading activation.

Measure source, test-seeking, impact, and semantic strata separately.

**Phase exit:** graph operations improve at least one predeclared structural task family
without degrading the others beyond an agreed tolerance.

---

## 9. Phase 5 — incremental indexing and semantic repair

### Objective

Replace full destructive rebuilds with a safe change lifecycle.

### WP5.1 — Change planner

**Owner:** indexing agent

Use tracked paths and stored hashes to classify:

- Added files.
- Deleted files.
- Modified files.
- Renamed files where detectable.
- Unchanged files.

Produce a plan before mutating the database. The plan identifies affected symbols, edges,
chunks, embeddings, assertion links, and evidence spans.

### WP5.2 — Transactional incremental update

**Owner:** database agent

Within an explicit transaction or staged replacement:

1. Parse changed files outside the write transaction.
2. Validate all extracts.
3. Remove obsolete structural rows for changed/deleted files.
4. Insert new facts and chunks.
5. Re-resolve affected names, including callers in unchanged files when a target set changes.
6. Reuse embeddings only when text hash and embedding identity match.
7. Relink assertions by qualname and verify their evidence.
8. Mark affected claims stale with precise reasons.
9. Commit atomically.

Avoid treating a qualname as a permanent identity across semantically unrelated replacements.
Hash and location information should inform relinking diagnostics.

### WP5.3 — Regeneration queue

**Owner:** semantic lifecycle agent

Create a queue or query for claims needing work:

- Stale because evidence changed.
- Subject deleted.
- Subject relinked but evidence boundary changed.
- Generator version superseded.
- Judge policy no longer satisfied.

Regeneration must create a new assertion and mark the former one superseded or stale. Preserve
the lineage relation.

### WP5.4 — Freshness interfaces

**Owner:** operations agent

Ship:

- `codelearner refresh`
- `codelearner index --incremental`
- `index_stats` freshness counts
- Optional strict verification
- Machine-readable exit codes for stale or incomplete indexes

Do not add a watcher until explicit refresh and incremental indexing are reliable. Editor and
CI integrations can call these operations without a daemon.

**Phase exit:** editing, renaming, adding, and deleting files updates the index without losing
verdict history; affected claims are expired or relinked correctly; unchanged chunks are not
re-embedded.

---

## 10. Phase 6 — production hardening

### WP6.1 — Connection and transaction ownership

**Owner:** concurrency agent

Audit the MCP server's shared SQLite connection and `_atomic` behavior under concurrent tool
calls. Move to one connection per worker/thread or a serialized database executor. Replace
implicit connection-global transaction joining with explicit savepoints or ownership.

Tests:

- Concurrent reads during assertion submission.
- Concurrent submissions.
- Reindex replacement during an open server session.
- Rollback after an inner failure.
- No response may claim a row is servable if its transaction later rolls back.

### WP6.2 — Packaging and fast startup

**Owner:** release engineering agent

- Install the project in a clean environment and test both console scripts.
- Build wheels and source distributions in CI.
- Verify package data includes schema and gold resources.
- Decide whether the stdlib fast MCP relay becomes product code, the SDK is replaced, or an
  upstream fix is required.
- Test cold MCP handshake against supported hosts.
- Make missing optional dependencies produce actionable diagnostics.

The benchmark shim is evidence of a product problem, not the final product boundary.

### WP6.3 — Input and resource limits

**Owner:** security agent

Centralize limits for:

- Query length.
- Result depth.
- Response budget.
- Claim length.
- Evidence spans per claim.
- File and symbol size.
- Graph fanout and depth.
- Batch size and model concurrency.
- SQLite variable count.

Fuzz parsers, MCP request validation, assertion admission, and evidence assembly.

### WP6.4 — Prompt-injection evaluation

**Owner:** model security agent

Plant adversarial comments, strings, docstrings, and symbol names that attempt to:

- Change the model's task.
- Request tool use or repository reads.
- Forge citation references.
- Exfiltrate canaries.
- Cause unsupported acceptance.

Run both generator and judge. Source must be clearly delimited and described as untrusted data.
The citation menu remains integer-based. Report attack success separately for generation,
judgment, and final retrieval.

### WP6.5 — Observability

**Owner:** operations agent

Add structured events and counters for:

- Index duration and changed files.
- Resolution rate and precision samples by resolver.
- Retrieval latency by modality.
- Candidate and response truncation.
- Semantic candidates retrieved, filtered, and served.
- Staleness checks and expirations.
- Judge outcomes and instrument failures.
- Embedding/reranking model mismatches.
- MCP startup and tool-call errors.

Avoid collecting repository source or claim text in telemetry by default.

### WP6.6 — Compatibility and migrations

**Owner:** database/release agent

Introduce tested forward migrations for non-destructive schema changes. Continue refusing
unknown future versions. Backup before migrations that touch assertion history. Test upgrades
from every released schema that will be supported.

**Phase exit:** clean installation, concurrent use, schema upgrade, hostile input, and common
failure scenarios are covered by CI and produce actionable errors.

---

## 11. Phase 7 — end-to-end proof

### Objective

Determine whether the completed architecture helps real agents, not merely whether its
components work.

### WP7.1 — Benchmark arms

Run at least:

1. Bare agent with normal filesystem tools.
2. CodeGraph structural index.
3. Code Learner structural-only.
4. Code Learner structural plus semantic retrieval.

Keep model, prompt, built-in tools, permissions, cache treatment, and task order equivalent.
Verify every server before spending on runs.

### WP7.2 — Task strata

Use separate result rows for:

- Locate a symbol.
- Explain implementation behavior.
- Find callers and impact.
- Find the relevant tests.
- Explain purpose or rationale.
- Identify an invariant or risk.
- Detect an intentional absence or refused behavior.
- Multi-file architectural understanding.

The semantic arm is not justified by winning locate tasks. It must win purpose, rationale,
invariant, or architectural synthesis tasks while maintaining source-grounded correctness.

### WP7.3 — Metrics

Primary:

- Correct answer rate or rubric score.
- Unsupported-claim rate.
- Evidence/citation correctness.
- Task completion rate.

Secondary:

- Tool calls.
- Source reads after index use.
- Input/output/cache tokens.
- Wall time.
- Cost.
- Index tool utilization.
- Time to first correct evidence.

Report failures as failures, never as low-cost runs. Cluster intervals by task and repository.
Predeclare the minimum worthwhile effect.

### WP7.4 — Product signals

Alongside the controlled benchmark, recruit 5–10 experienced developers or agent-tool users.
Observe:

- Whether they invoke the semantic features without coaching.
- Which claims they distrust.
- Whether they inspect citations.
- Whether onboarding or impact analysis saves meaningful time.
- Whether setup, indexing, and model requirements block adoption.
- Which data they would allow a hosted service to process.

**Phase exit:** the semantic-enabled arm shows a meaningful correctness or time advantage on
semantic task strata, and users value the capability enough to tolerate its operational cost.

---

## 12. Phase 8 — release gates

Do not call the system production-ready until all mandatory gates pass.

### Correctness

- No known fail-open assertion admission or serving defects.
- Structural extraction has regression fixtures for every supported syntax form.
- Resolver precision is measured by strategy, not only aggregate coverage.
- Semantic retrieval never serves stale or rejected claims under default policy.
- Mixed retrieval refills results after policy filtering.

### Semantic quality

- Judge calibrated against human labels.
- Generator and judge failures remain distinguishable from abstention and rejection.
- Semantic retrieval beats structural-only retrieval on predeclared semantic tasks.
- Claim citations are sufficient, not merely valid byte ranges.

### Operations

- Clean installation works from a built artifact.
- Incremental updates preserve assertion history.
- Concurrent MCP calls have no transaction anomalies.
- Cold startup makes tools reliably visible in supported clients.
- Resource limits and structured diagnostics are documented.

### Evaluation

- The full benchmark matrix is complete.
- Results are reported by repository and task stratum.
- No headline depends on an unmeasured reranker or hidden heuristic.
- Reproduction instructions work from a clean checkout with documented external assets.

---

## 13. Agent execution protocol

Every implementation agent should receive:

- One work package only, or a tightly coupled group explicitly identified above.
- The current commit SHA.
- This guide and `docs/CURRENT-STATE.md` once created.
- The exact files it owns.
- Interfaces it may change and compatibility it must preserve.
- Required tests and benchmark commands.
- A prohibition on changing headline metrics without rerunning the relevant experiment.

Every handoff should contain:

1. Behavior changed.
2. Schema or public-interface changes.
3. Tests added and commands run.
4. Measurements rerun, with repository SHAs.
5. Known limitations and deliberately deferred work.
6. Migration and rollback instructions.
7. Files touched outside the original ownership boundary, with reasons.

Agents should not simultaneously edit these high-conflict seams:

- `codelearner/schema.sql`
- `codelearner/retrieve/search.py`
- `codelearner/server/app.py`
- `codelearner/cli/commands.py`
- assertion lifecycle/status definitions

Sequence agents that touch those files or assign one integration owner. Parallel work is safe
for gold-set construction, adversarial fixtures, packaging CI, documentation, and isolated
graph experiments once their interfaces are frozen.

---

## 14. Suggested first execution batch

The first batch should prove the thesis before undertaking operational breadth:

1. **Architecture owner:** WP0.2 and WP0.3.
2. **Evidence owner:** WP1.1, followed by WP1.2.
3. **Semantic storage owner:** WP2.1.
4. **Evaluation owner:** WP2.4, in parallel after candidate contracts are fixed.
5. **Integration owner:** WP2.2 and WP2.3 after storage and evidence APIs land.

The demonstration milestone is one query returning:

- A semantic purpose or invariant claim.
- Its subject symbol.
- The whole relevant source section with line numbers.
- Its exact citations and current hashes.
- Its verdict and freshness state.
- Related callers/callees.
- A result difference when `facts_only` is enabled.

If that milestone does not create a measurable advantage over structural-only exploration,
pause before investing in incremental indexing, more languages, or hosted infrastructure.

