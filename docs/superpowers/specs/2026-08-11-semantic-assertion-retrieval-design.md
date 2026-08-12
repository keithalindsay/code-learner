# Semantic Assertion Retrieval Design

**Date:** 2026-08-11
**Status:** Approved
**Roadmap scope:** Phase 2 — production semantic retrieval core

## Purpose

Code Learner already extracts source structure and stores evidence-bound semantic
assertions, but search cannot retrieve those assertions directly. Phase 2 turns the
assertion store into a production retrieval modality. A query about purpose, rationale,
risk, or an invariant must be able to return a supported semantic claim together with
the current source that supports it.

This phase establishes correctness and serving behavior. Assertion-specific dense
embeddings and the five-repository comparative evaluation are Phase 2.5 so model quality
and product lift can be measured after the policy and retrieval substrate are stable.

## Goals

- Retrieve semantic assertions directly through lexical search.
- Serve only fresh assertions that have independent supporting adjudication by default.
- Preserve source symbols and assertions as distinct candidate types.
- Fuse source and assertion candidates without comparing incompatible raw scores.
- Return complete, bounded evidence for assertion results through library, CLI, and MCP.
- Make `facts_only` remove semantic candidates before truncation and refill with source.
- Preserve all assertion, citation, verdict, and staleness history through migration.
- Keep ordering, filtering, truncation, and serialization deterministic.

## Non-goals

- Assertion-specific dense embeddings or vector retrieval.
- The five-repository semantic gold set and comparative product evaluation.
- Learned ranking or score calibration.
- Full judge orchestration, calibration, or a judge CLI; those remain Phase 3.
- Incremental source indexing and semantic regeneration; those remain Phase 5.
- Replacing SQLite with a graph or vector database.

## Serving policy

Serving policy has one implementation below CLI and MCP. The production default is:

- Maximum tier is 2 unless `facts_only` selects tier 1.
- An assertion must have status `active`.
- Every cited span must verify against current repository bytes at read time.
- At least one recorded verdict must be `supported`.
- Any `unsupported` or `refuted` verdict makes the assertion ineligible.
- An assertion with no verdict is pending and ineligible.

The store continues to retain pending, rejected, and stale assertions. An explicit
research policy may include pending assertions, but it must be named at the library
boundary and must never be the CLI or MCP default. Rejected or stale assertions are not
servable under any retrieval policy; historical inspection remains available through
the assertion store APIs.

This narrow admission contract is implemented in Phase 2 because searchable unjudged
claims would turn generation into publication. Phase 3 will add judge orchestration,
policy history, calibration, and operational controls without weakening this default.

## Candidate model

Retrieval uses a tagged union rather than extending the existing source `Hit` until its
fields become ambiguous.

### Source candidate

A source candidate retains the existing symbol identity and ranking information:

- stable candidate key `source:<symbol_id>`;
- symbol ID, qualname, kind, path, indexed coordinates, and content hash;
- tier 0 or 1;
- contributing modalities and graph `via` information;
- fused score and optional debug explanation.

### Assertion candidate

An assertion candidate contains:

- stable candidate key `assertion:<assertion_id>`;
- assertion ID, kind, claim text, generator, and subject identity;
- tier 2;
- status, accepted supporting verdict metadata, and freshness outcome;
- exact evidence spans with paths, byte ranges, hashes, and live coordinates;
- contributing modalities and fused score;
- conflict metadata when another eligible assertion about the same subject and kind is
  materially distinct;
- optional debug explanation.

Source and assertion candidates occupy independent result slots. An assertion can
promote its subject source and a source hit can promote eligible assertions attached to
that symbol, but neither is silently converted into the other. Duplicate assertion IDs
can never occupy more than one slot.

## Storage and migration

The next schema version adds an assertion retrieval document for each authoritative
assertion. Its canonical text is deterministic:

```text
kind: <kind>
subject: <subject qualname>
claim: <claim text>
evidence: <path:start-end>[, ...]
```

The schema includes an FTS5 assertion index and enough relational metadata to rebuild
it deterministically. Search documents are derived data, not the authoritative claim
record.

Synchronization uses explicit storage functions at the assertion mutation boundary
rather than spreading SQL across retrieval callers. Insertion, verdict changes,
staleness, reinstatement, and subject relinking update or remove the live search
document in the same transaction as the authoritative mutation. Eligibility is also
checked again at retrieval time as defense in depth.

The existing rebuild-and-carry migration mechanism must preserve assertions, evidence
spans, verdicts, and staleness logs. Derived assertion search documents are rebuilt from
the preserved authoritative rows. A migration must never convert pending assertions to
supported, reactivate rejected or stale assertions, or discard history.

## Retrieval flow

For a query requesting `k` results:

1. Source lexical, optional source dense, and graph retrieval produce their existing
   ranked source lists.
2. Assertion FTS retrieves a deterministic window wider than `k`.
3. Assertion IDs are deduplicated before filesystem work.
4. The bounded window is evaluated under `ServingPolicy` and its citations are verified
   against current bytes.
5. Ineligible candidates are removed. If fewer eligible assertion candidates remain
   than the requested assertion depth, the next bounded FTS page is retrieved and
   verified. Paging ends at sufficient depth, index exhaustion, or a fixed safety cap.
6. Eligible assertion candidates and source candidates enter mixed rank fusion.
7. Bidirectional promotion contributes additional ranked lists: assertion-to-subject
   and source-to-attached-assertion. Promotion never bypasses serving policy.
8. Tier policy is applied before the final cut. With `facts_only`, all tier-2 candidates
   are removed and source candidates refill the vacated slots.
9. Deterministic tie breakers use candidate type and stable numeric identity after fused
   score and rank contributions.
10. The top candidates are assembled into bounded evidence responses.

Freshness verification may write a stale transition through the existing authoritative
store behavior. A transient inability to inspect evidence withholds the assertion for
that request without permanently marking it stale.

## Fusion and conflicts

Mixed retrieval starts with reciprocal rank fusion because BM25, source vector scores,
graph activation, and promotion ranks are not calibrated. Modality weights are explicit
constants and appear in debug explanations. No learned ranker is introduced in Phase 2.

An assertion and its subject remain separate candidates even when they promote one
another. Promotion is a ranking vote, not automatic inclusion. It uses only eligible
assertions and existing indexed subjects.

Eligible assertions with the same normalized claim identity are deduplicated by stable
assertion ID rules. Distinct eligible assertions about the same subject and kind remain
separate. When more than one exists, each is labelled a conflict candidate; retrieval
does not select a winner or synthesize a compromise.

## Evidence responses

A source result continues to use the Phase 1 whole-symbol evidence representation.

An assertion result contains:

- candidate type, tier, rank, and modalities;
- claim text and assertion kind;
- subject symbol identity;
- supporting verdicts used by policy;
- freshness status;
- exact citation records;
- bounded, line-numbered source sections for the citations and subject;
- related callers and callees when available;
- omissions and byte-budget accounting.

All repository paths remain normalized, relative paths. Current bytes and live line
coordinates come from the Phase 1 safe evidence reader. A stale, rejected, pending, or
unverifiable assertion is withheld rather than rendered partially as a semantic result.

CLI JSON and MCP return the same stable candidate semantics. Human CLI rendering may be
more compact but must label claims as semantic assertions and show their supporting
verdict and freshness state. Debug score explanations are opt-in and do not expose every
internal float by default.

## Errors and limits

- Invalid query, `k`, budget, policy, or pagination values fail at the library boundary.
- Search text and repository content are data, never SQL fragments or instructions.
- Assertion verification examines only bounded candidate pages and bounded source files.
- FTS pagination has a fixed maximum candidate count to prevent adversarial queries from
  causing an unbounded freshness scan.
- Missing optional source embeddings do not disable assertion lexical retrieval.
- Missing assertion search structures produce a clear incompatible-index error rather
  than silently returning no semantic results.
- Unsupported platform source hydration remains fail-closed.
- CLI and MCP adapt domain errors consistently and never return host absolute paths.

## Testing strategy

Each implementation task follows strict red-green TDD and receives an independent
specification and quality review. Required coverage includes:

- production and research serving policies;
- pending, supported, unsupported, refuted, stale, missing, changed, and transiently
  unavailable evidence;
- schema creation, synchronization, migration, history preservation, and rollback;
- deterministic assertion document construction and FTS ranking;
- bounded verify-and-refill behavior when high-ranked candidates fail policy;
- source/assertion identity, deduplication, promotion, conflicts, weights, and tie breaks;
- `facts_only` filtering before truncation with source refill;
- assertion evidence budgets, citations, live coordinates, and path safety;
- library, CLI JSON, human CLI, and MCP semantic parity;
- event-loop responsiveness and serialized SQLite ownership in MCP;
- backward behavior for source-only indexes where explicitly supported;
- full-suite regression checks, Ruff, and mypy.

## Delivery slices

1. Candidate and serving-policy contracts.
2. Schema migration and synchronized assertion FTS documents.
3. Lexical assertion retrieval with bounded verification and refill.
4. Mixed fusion, promotion, tier filtering, deduplication, and conflicts.
5. Assertion evidence assembly and library response integration.
6. CLI and MCP serving integration plus documentation.
7. Phase-level semantic fixtures, ablation checks, and final review.

Each slice must be independently testable and committed. Phase 2 exits only when a
semantic query can directly return a fresh, supported assertion with its subject and
current evidence, while `facts_only` returns a refilled source-only result set and stale,
rejected, and pending claims never appear under the production policy.

## Phase 2.5 boundary

Phase 2.5 adds assertion-specific embedding documents and identity metadata, vector
retrieval, mixed lexical/dense assertion ablations, and a frozen semantic gold set across
at least five repositories. Product-authored and mined questions remain separate, hard
negatives are explicit, and uncertainty is clustered by repository. Productization is
revisited only after this evaluation measures semantic-enabled retrieval against the
source-only control.
