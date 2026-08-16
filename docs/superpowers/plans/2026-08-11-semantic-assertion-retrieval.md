# Semantic Assertion Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fresh, independently supported semantic assertions first-class lexical search results with bounded evidence and identical library, CLI, and MCP semantics.

**Architecture:** Keep the existing source-only `Hit` pipeline stable and add a tagged candidate layer above it. Store deterministic assertion FTS documents as derived data, retrieve them through bounded pages, apply centralized verdict and live-freshness policy, then fuse assertion and source candidates by rank before evidence assembly. Dense assertion retrieval and five-repository evaluation remain Phase 2.5.

**Tech Stack:** Python 3.11+, SQLite/FTS5, frozen dataclasses, pytest, Ruff, mypy, FastMCP.

## Global Constraints

- SQLite remains the system of record; no graph or vector database is introduced.
- Production semantic search serves only `active`, currently verifiable assertions with at least one `supported` verdict and no `unsupported` or `refuted` verdict.
- Pending assertions are available only through the explicitly named research policy; rejected and stale assertions are never retrievable.
- Source and assertion candidates have distinct tagged types and stable keys `source:<symbol_id>` and `assertion:<assertion_id>`.
- Tier filtering happens before final truncation; `facts_only` selects maximum tier 1 and refills from source candidates.
- Assertion search verifies only bounded candidate pages and stops at a fixed maximum candidate count.
- Every semantic result is all-or-nothing: claim, verdict, citations, subject, and current evidence are either served together or withheld.
- Repository content and queries are data, never SQL, shell arguments, paths outside the bound repository, or model instructions.
- CLI and MCP expose the same stable candidate semantics and never expose absolute host paths.
- Derived search documents may be rebuilt; authoritative assertion, evidence, verdict, and staleness history must be preserved.
- Existing source-only `search()`, `Hit`, reranker, and Phase 1 evidence behavior remain backward compatible.
- Assertion embeddings, vector search, learned ranking, and the five-repository gold set are Phase 2.5 and must not be added here.

---

### Task 1: Typed retrieval candidates and serving policy

**Files:**
- Create: `codelearner/retrieve/types.py`
- Create: `codelearner/assertions/policy.py`
- Modify: `codelearner/retrieve/__init__.py`
- Modify: `codelearner/assertions/__init__.py`
- Test: `tests/test_retrieval_types.py`
- Test: `tests/test_serving_policy.py`

**Interfaces:**
- Consumes: `codelearner.assertions.store.Assertion`, verdict constants, and tier constants.
- Produces: frozen `SourceCandidate`, `AssertionCandidate`, `Candidate`, `VerdictSummary`, `Freshness`, `ScoreContribution`, `CandidateSearchResult`, `ServingPolicy`, `PolicyDecision`, `PRODUCTION_POLICY`, `RESEARCH_PENDING_POLICY`, and `evaluate_metadata(...)`.

- [ ] **Step 1: Write failing candidate identity and immutability tests**

```python
def test_candidate_keys_are_type_qualified():
    source = SourceCandidate.from_hit(_hit(symbol_id=7))
    assertion = _assertion_candidate(assertion_id=7)
    assert source.key == "source:7"
    assert assertion.key == "assertion:7"
    assert source.tier in (0, 1)
    assert assertion.tier == 2

def test_candidate_result_keeps_per_modality_explanations():
    candidate = SourceCandidate.from_hit(_hit(symbol_id=1))
    result = CandidateSearchResult(
        candidates=(candidate,), per_modality={"source_lexical": (candidate,)}
    )
    assert result.per_modality["source_lexical"] == (candidate,)
```

- [ ] **Step 2: Run the candidate tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_retrieval_types.py -q`

Expected: collection fails because `codelearner.retrieve.types` does not exist.

- [ ] **Step 3: Implement frozen tagged candidate contracts**

Use non-optional fields for type-specific data. `SourceCandidate.from_hit(hit)` copies the existing `Hit` fields and derives the source tier from the existing tier helper without mutating `Hit`. `AssertionCandidate` contains `assertion_id`, `subject_symbol_id`, `subject_qualname`, `kind`, `claim`, `generator`, `status`, `verdicts`, `freshness`, `spans`, `score`, `modality`, `conflict`, and `contributions`. Both types expose a `.key` property. `CandidateSearchResult` uses immutable tuples.

```python
Candidate = SourceCandidate | AssertionCandidate

@dataclass(frozen=True)
class VerdictSummary:
    judge: str
    verdict: str
    rationale: str | None

@dataclass(frozen=True)
class Freshness:
    verified: bool
    method: str

@dataclass(frozen=True)
class ScoreContribution:
    modality: str
    rank: int
    weight: float
    value: float
```

- [ ] **Step 4: Write failing policy matrix tests**

```python
@pytest.mark.parametrize(
    ("status", "verdicts", "policy", "eligible", "reason"),
    [
        ("active", ("supported",), PRODUCTION_POLICY, True, "eligible"),
        ("active", (), PRODUCTION_POLICY, False, "verdict_required"),
        ("active", (), RESEARCH_PENDING_POLICY, True, "eligible_pending"),
        ("active", ("supported", "refuted"), PRODUCTION_POLICY, False, "vetoed"),
        ("active", ("supported", "unsupported"), PRODUCTION_POLICY, False, "vetoed"),
        ("rejected", ("supported",), RESEARCH_PENDING_POLICY, False, "status"),
        ("stale", ("supported",), RESEARCH_PENDING_POLICY, False, "status"),
    ],
)
def test_serving_policy_matrix(status, verdicts, policy, eligible, reason):
    decision = evaluate_metadata(_assertion(status=status), _verdicts(verdicts), policy)
    assert (decision.eligible, decision.reason) == (eligible, reason)
```

- [ ] **Step 5: Run the policy tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_serving_policy.py -q`

Expected: collection fails because `codelearner.assertions.policy` does not exist.

- [ ] **Step 6: Implement centralized policy evaluation**

```python
@dataclass(frozen=True)
class ServingPolicy:
    max_tier: int = 2
    require_verdict: bool = True
    accepted_verdicts: frozenset[str] = frozenset({store.VERDICT_SUPPORTED})
    allow_pending: bool = False

PRODUCTION_POLICY = ServingPolicy()
RESEARCH_PENDING_POLICY = ServingPolicy(require_verdict=False, allow_pending=True)

@dataclass(frozen=True)
class PolicyDecision:
    eligible: bool
    reason: str
    accepted: tuple[VerdictSummary, ...] = ()
```

Validate tier and policy invariants in `ServingPolicy.__post_init__`. `evaluate_metadata` must reject non-active status first, then veto verdicts, then accept supported verdicts, then apply pending behavior. It performs no filesystem or SQL work.

- [ ] **Step 7: Verify Task 1 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_retrieval_types.py tests/test_serving_policy.py tests/test_retrieve.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/types.py codelearner/assertions/policy.py tests/test_retrieval_types.py tests/test_serving_policy.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/types.py codelearner/assertions/policy.py
git diff --check
git add codelearner/retrieve/types.py codelearner/retrieve/__init__.py codelearner/assertions/policy.py codelearner/assertions/__init__.py tests/test_retrieval_types.py tests/test_serving_policy.py
git commit -m "feat: define semantic retrieval policy and candidates"
```

### Task 2: Schema v7 assertion search documents

**Files:**
- Create: `codelearner/assertions/search_index.py`
- Modify: `codelearner/schema.sql`
- Modify: `codelearner/db.py`
- Modify: `codelearner/assertions/store.py`
- Modify: `codelearner/cli/commands.py`
- Test: `tests/test_assertion_search_index.py`
- Test: `tests/test_db.py`
- Test: `tests/test_assertions.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: authoritative `Assertion` rows and the existing `_atomic` transaction boundary.
- Produces: `canonical_document(assertion)`, `sync_assertion_document(conn, assertion_id)`, `remove_assertion_document(conn, assertion_id)`, `rebuild_assertion_documents(conn)`, and `assertion_search_structures_present(conn)`.

- [ ] **Step 1: Write failing schema and canonical-document tests**

```python
def test_v7_schema_contains_assertion_documents_and_fts(index):
    names = {r[0] for r in index.execute("SELECT name FROM sqlite_master")}
    assert {"assertion_documents", "assertions_fts"} <= names
    assert db.SCHEMA_VERSION == 7

def test_canonical_document_is_deterministic(admitted_assertion):
    assert canonical_document(admitted_assertion) == (
        "kind: purpose\n"
        "subject: leases.acquire\n"
        "claim: Coordinates lease renewal.\n"
        "evidence: leases.py:10-18"
    )
```

- [ ] **Step 2: Run focused schema tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_db.py -q`

Expected: missing search-index module/tables and schema version mismatch.

- [ ] **Step 3: Add derived document and FTS schema**

Add `assertion_documents(assertion_id PRIMARY KEY, text NOT NULL, text_hash NOT NULL)` with cascade to `assertions`. Add external-content FTS5 storage and insert/update/delete triggers that mirror the existing chunk FTS pattern. Bump `SCHEMA_VERSION` from 6 to 7 and add the v7 history entry. Do not add derived tables to `_CARRY_TABLES`.

- [ ] **Step 4: Implement deterministic rebuild-only helpers**

`canonical_document` sorts spans by `(path, byte_start, byte_end, id or -1)`. `sync_assertion_document` loads one assertion and stores a document only while status is `active`; it indexes pending assertions so the named research policy remains possible. Rejected/stale rows remove the derived document. Helpers neither commit nor open transactions.

- [ ] **Step 5: Write failing atomic synchronization tests**

Cover admission creating a document, non-supporting verdict removing it, supported verdict retaining it, staleness removing it, reinstatement restoring it, and a monkeypatched sync exception rolling back the authoritative mutation and derived document together.

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_assertions.py -q`

Expected: authoritative mutations do not yet synchronize documents.

- [ ] **Step 6: Wire synchronization into authoritative transactions**

Call sync/remove before leaving the same `_atomic` blocks used by `write_assertion`, `record_verdict`, `mark_stale`, and `reinstate`. Preserve the rule that verdict history is never deleted. A refuted or unsupported verdict both records its row and changes status before document removal.

- [ ] **Step 7: Write and satisfy v6 carry/rebuild tests**

Create a v6 fixture containing active supported, active pending, rejected, and stale assertions with spans, verdicts, and staleness rows. Force-rebuild with carry. Assert exact authoritative IDs/status/history survive, only active documents are rebuilt, and the FTS table contains no rejected/stale rows. Rebuild documents only after restore verification and boundary-expiry passes complete.

- [ ] **Step 8: Verify Task 2 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_search_index.py tests/test_assertions.py tests/test_db.py tests/test_cli.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/assertions/search_index.py codelearner/assertions/store.py codelearner/db.py codelearner/cli/commands.py tests/test_assertion_search_index.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/assertions/search_index.py codelearner/assertions/store.py
git diff --check
git add codelearner/schema.sql codelearner/db.py codelearner/assertions/search_index.py codelearner/assertions/store.py codelearner/cli/commands.py tests/test_assertion_search_index.py tests/test_db.py tests/test_assertions.py tests/test_cli.py
git commit -m "feat: index semantic assertion documents"
```

### Task 3: Bounded lexical assertion retrieval and refill

**Files:**
- Create: `codelearner/retrieve/assertions.py`
- Modify: `codelearner/retrieve/lexical.py`
- Modify: `codelearner/assertions/store.py`
- Test: `tests/test_assertion_retrieve.py`

**Interfaces:**
- Consumes: `ServingPolicy`, assertion FTS documents, stored verdicts, and live citation verification.
- Produces: `AssertionSearchUnavailable`, `search_assertions(conn, repo_root, query, *, policy=PRODUCTION_POLICY, k=10, page_size=40, max_candidates=400)`, `load_assertions_by_ids`, `verdict_summaries`, and ID-scoped `verify_assertions`.

- [ ] **Step 1: Write failing retrieval, policy, and refill tests**

```python
def test_supported_claim_is_retrieved_with_verdict_and_freshness(assertion_index):
    hits = search_assertions(assertion_index.conn, assertion_index.root, "lease ownership", k=1)
    assert [(h.claim, h.verdicts[0].verdict, h.freshness.verified) for h in hits] == [
        ("Renews ownership during long work.", "supported", True)
    ]

def test_ineligible_top_rows_are_refilled(assertion_index):
    assertion_index.add_ranked_claims(["pending", "stale", "refuted", "supported"])
    hits = search_assertions(assertion_index.conn, assertion_index.root, "lease", k=1, page_size=2)
    assert [h.claim for h in hits] == ["supported"]
```

Also pin empty/hostile FTS queries, duplicate IDs across pages, transient unreadability withholding without status change, stale/missing expiry, deterministic equal-score order, non-positive bounds, max-candidate cap, and a clear error when assertion search structures are absent.

- [ ] **Step 2: Run retrieval tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py -q`

Expected: collection fails because `codelearner.retrieve.assertions` does not exist.

- [ ] **Step 3: Expose bounded batch store readers**

Add public ID-scoped readers that preserve input order, issue bounded `IN` queries in SQLite-safe chunks, batch-load spans and verdicts, and verify only supplied IDs. Do not call `servable_assertions`, because it scans every active assertion. Live failures must flow through the existing stale transition; unreadable evidence is withheld without mutation.

- [ ] **Step 4: Implement FTS paging and verify-refill**

Move or expose the existing FTS query escaping as `escape_fts_query`. `_search_page` uses bound parameters and deterministic `ORDER BY bm25(assertions_fts), assertion_id`. Each page is metadata-filtered before filesystem verification, IDs are deduplicated before I/O, and paging stops after `k`, exhaustion, or `max_candidates`.

```python
def search_assertions(
    conn: sqlite3.Connection,
    repo_root: Path,
    query: str,
    *,
    policy: ServingPolicy = PRODUCTION_POLICY,
    k: int = 10,
    page_size: int = 40,
    max_candidates: int = 400,
) -> list[AssertionCandidate]: ...
```

- [ ] **Step 5: Verify Task 3 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_retrieve.py tests/test_assertions.py tests/test_retrieve.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/assertions.py codelearner/retrieve/lexical.py codelearner/assertions/store.py tests/test_assertion_retrieve.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/assertions.py codelearner/assertions/store.py
git diff --check
git add codelearner/retrieve/assertions.py codelearner/retrieve/lexical.py codelearner/assertions/store.py tests/test_assertion_retrieve.py
git commit -m "feat: retrieve verified semantic assertions"
```

### Task 4: Mixed candidate fusion and semantic promotion

**Files:**
- Create: `codelearner/retrieve/mixed.py`
- Modify: `codelearner/retrieve/__init__.py`
- Test: `tests/test_mixed_retrieve.py`

**Interfaces:**
- Consumes: source-only `search()`, `SourceCandidate.from_hit`, `search_assertions`, and serving policy.
- Produces: `SEMANTIC_WEIGHTS`, `mixed_rank_fusion(ranked_lists, *, k, max_tier=2, weights=SEMANTIC_WEIGHTS, debug=False)`, and `search_candidates(conn, repo_root, query, *, k=10, policy=PRODUCTION_POLICY, embedder=None, reranker=None, use_lexical=True, use_dense=True, use_graph=True, use_assertions=True, debug=False) -> CandidateSearchResult`.

- [ ] **Step 1: Write failing mixed-fusion unit tests**

Pin independent source/assertion slots with colliding numeric IDs, weighted RRF contributions, deterministic ties `(score descending, source before assertion, numeric ID)`, duplicate keys appearing once, conflict labels for distinct normalized claims on the same subject/kind, no conflict for normalized duplicate text, and max-tier filtering before `k`. Claim normalization is Unicode NFKC followed by `casefold()` and collapse of every whitespace run to one ASCII space; it is used only for duplicate/conflict labelling and never rewrites stored or displayed claim text.

```python
def test_facts_only_filters_before_cut_and_refills():
    ranked = {"source_lexical": (_source(1), _source(2)), "assertion_lexical": (_claim(3),)}
    result = mixed_rank_fusion(ranked, k=2, max_tier=1)
    assert [c.key for c in result] == ["source:1", "source:2"]
```

- [ ] **Step 2: Run mixed-fusion tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_mixed_retrieve.py -q`

Expected: collection fails because `codelearner.retrieve.mixed` does not exist.

- [ ] **Step 3: Implement tagged RRF without changing source RRF**

Key score, candidate, and contributor maps by `.key`. Record `ScoreContribution` only when debug is enabled. Use explicit weights for `source_lexical`, `source_dense`, `source_graph`, `assertion_lexical`, `assertion_subject`, and `source_assertions`. Apply `candidate.tier <= max_tier` before ordering and slicing. Preserve existing `retrieve.search.search` and `reciprocal_rank_fusion` unchanged.

- [ ] **Step 4: Write failing end-to-end promotion tests**

Cover assertion-to-subject promotion, source-to-attached-assertion promotion, policy preventing pending/rejected/stale promotion, source-only compatibility when assertions are disabled, missing source embedder not disabling assertion FTS, and semantic-plus-reranker behavior. When a reranker is supplied, rerank the source candidate pool before mixed fusion; never pass an `AssertionCandidate` into the `Hit`-only reranker.

- [ ] **Step 5: Implement `search_candidates` orchestration**

Retrieve at wider depth, convert source hits, construct promotion lists only from eligible candidates, fuse once, and return immutable per-modality lists. `max_tier=1` must skip assertion retrieval entirely when possible and always return a full source page when enough source candidates exist.

- [ ] **Step 6: Verify Task 4 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_mixed_retrieve.py tests/test_retrieve.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/mixed.py tests/test_mixed_retrieve.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/mixed.py
git diff --check
git add codelearner/retrieve/mixed.py codelearner/retrieve/__init__.py tests/test_mixed_retrieve.py
git commit -m "feat: fuse source and semantic candidates"
```

### Task 5: Assertion evidence bundles

**Files:**
- Modify: `codelearner/evidence/types.py`
- Modify: `codelearner/evidence/assemble.py`
- Modify: `codelearner/evidence/render.py`
- Modify: `codelearner/evidence/__init__.py`
- Test: `tests/test_assertion_evidence.py`

**Interfaces:**
- Consumes: mixed `Candidate` values and the Phase 1 descriptor-safe reader.
- Produces: `AssertionEvidence`, `CandidateEvidence`, `CandidateEvidenceBundle`, `assemble_candidate_evidence(...)`, and common JSON/human rendering primitives.

- [ ] **Step 1: Write failing evidence contract tests**

Pin tagged source/assertion evidence, exact citations and supporting verdicts, live lines after equal-byte prefix edits, subject source, multiple citation ordering, caller/callee context when present, overlap deduplication, UTF-8 byte coordinates, hard byte budgets, deterministic omissions, and absence of absolute paths.

- [ ] **Step 2: Run assertion evidence tests and confirm RED**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_evidence.py -q`

Expected: imports fail because assertion evidence types and assembler do not exist.

- [ ] **Step 3: Add tagged bundle types without widening source sections**

`AssertionEvidence` carries the assertion metadata, accepted verdicts, freshness, citations, subject section, citation sections, and related symbol summaries. `CandidateEvidence` is a tagged source/assertion union. `CandidateEvidenceBundle` carries ordered results, omitted entries, used bytes, and budget bytes.

- [ ] **Step 4: Refactor and reuse the safe reader**

Move descriptor-rooted reading to a private shared helper without weakening its component-by-component `O_NOFOLLOW`, same-descriptor `fstat`, and bounded-read properties. Assertion citation reads use the same repository root and size ceiling. Hash and live-line verification occur before rendering.

- [ ] **Step 5: Implement all-or-nothing assertion assembly**

If any required citation is unsafe, changed, missing, oversized, or unreadable, omit the entire semantic candidate with a stable reason. Never return claim text with incomplete citations. Select only whole sections under the byte budget, deduplicate identical/overlapping sections deterministically, and preserve candidate order.

- [ ] **Step 6: Verify Task 5 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_assertion_evidence.py tests/test_evidence.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/evidence tests/test_assertion_evidence.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/evidence
git diff --check
git add codelearner/evidence tests/test_assertion_evidence.py tests/test_evidence.py
git commit -m "feat: assemble supported assertion evidence"
```

### Task 6: CLI and MCP semantic search parity

**Files:**
- Create: `codelearner/retrieve/serialize.py`
- Modify: `codelearner/cli/main.py`
- Modify: `codelearner/cli/commands.py`
- Modify: `codelearner/server/app.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `search_candidates`, `assemble_candidate_evidence`, and production policy.
- Produces: `candidate_json(candidate, rank, *, debug=False)`, CLI `--no-assertions`/`--debug-scores`, and MCP `include_assertions`/`debug_scores` arguments.

- [ ] **Step 1: Write failing stable serialization tests**

Assert source and assertion candidate JSON have stable tagged keys, tier labels, no absolute paths, rounded public scores, verdict/freshness/citation metadata for assertions, and contributions only when debug is true.

- [ ] **Step 2: Implement the shared serializer**

Keep existing source keys compatible while adding `candidate_type` and `candidate_key`. Assertion JSON must include `assertion_id`, `claim`, `assertion_kind`, subject identity, verdicts, freshness, conflict, and citations. CLI and MCP must call this shared serializer rather than duplicate dictionaries.

- [ ] **Step 3: Write failing CLI integration tests**

Cover production default returning only supported semantic claims, pending/rejected/stale absence, human `T2 semantic` labels, source-rich assertion evidence, `--facts-only` returning a full refilled source page, `--no-assertions` ablation, debug contributions, compact output, explicit cross-repo refusal, and incompatible v6 index errors.

- [ ] **Step 4: Implement CLI candidate search**

Map `--facts-only` to `max_tier=1` before fusion. Assertions default on; `--no-assertions` is the explicit ablation. Use the stored index root for all evidence. Compact searches do not require evidence hydration, but semantic eligibility still verifies cited bytes.

- [ ] **Step 5: Write failing MCP parity and concurrency tests**

Assert identical candidate objects for equivalent CLI/MCP calls, schema defaults `include_assertions=true` and `debug_scores=false`, production policy cannot be overridden from MCP, `facts_only` refills, semantic source output is bounded, no absolute index path leaks, early stdio handshake remains healthy, and the existing serialized worker keeps the event loop responsive.

- [ ] **Step 6: Implement MCP candidate search on the existing worker**

Keep all SQLite, filesystem verification, and model work inside `IndexSource.run_sync`. Async adapters only await that worker. Do not add a second executor or reopen the index on the event-loop thread.

- [ ] **Step 7: Update user documentation**

Document supported-only semantic defaults, pending research behavior at the library boundary, `facts_only`, ablation/debug switches, upgrade requirements, assertion result shape, and the Phase 2.5 boundary. Remove statements saying search emits no tier-2 modality or `facts_only` is inert.

- [ ] **Step 8: Verify Task 6 and commit**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_cli.py tests/test_mcp.py tests/test_assertion_evidence.py tests/test_mixed_retrieve.py -q
/home/keith/projects/code-learner/.venv/bin/ruff check codelearner/retrieve/serialize.py codelearner/cli codelearner/server/app.py tests/test_cli.py tests/test_mcp.py
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner/retrieve/serialize.py codelearner/cli codelearner/server/app.py
git diff --check
git add codelearner/retrieve/serialize.py codelearner/cli codelearner/server/app.py README.md tests/test_cli.py tests/test_mcp.py
git commit -m "feat: serve semantic search through CLI and MCP"
```

### Task 7: Phase-level semantic acceptance and regression

**Files:**
- Create: `tests/test_semantic_search.py`
- Modify: `codelearner/eval/ablation.py`
- Modify: `docs/IMPLEMENTATION-GUIDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete Phase 2 library, CLI, and MCP behavior.
- Produces: deterministic phase-exit fixtures and a source-only versus semantic lexical ablation hook; no five-repository claims.

- [ ] **Step 1: Write the phase-exit scenario before changing evaluation code**

Build a temporary repository whose opaque symbol names do not reveal a semantic invariant. Admit one cited claim, record a supporting verdict, and assert a why/invariant query returns the assertion, subject, current source, citations, graph context, verdict, and freshness. Assert source-only mode cannot return the claim, `facts_only` refills with source, and pending/rejected/stale controls are absent.

- [ ] **Step 2: Run the phase-exit test and repair only integration gaps**

Run: `/home/keith/projects/code-learner/.venv/bin/python -m pytest tests/test_semantic_search.py -q`

Expected before final integration repair: at least one end-to-end assertion fails while task-level unit tests remain green. Any repair must get a focused regression test in the owning test file before production code changes.

- [ ] **Step 3: Add type-compatible ablation reporting**

Add an assertion modality switch/count to the existing evaluation result only where it accepts candidate keys. Do not coerce assertion candidates into symbol IDs and do not claim quality lift from the deterministic fixture. Label this as plumbing for Phase 2.5 measurement.

- [ ] **Step 4: Reconcile roadmap documentation**

Mark Phase 2 implementation behavior accurately, name the supported-only serving default, and state that dense assertion retrieval plus five-repository lift evaluation remain Phase 2.5. Do not mark productization criteria satisfied.

- [ ] **Step 5: Run full verification**

Run:

```bash
/home/keith/projects/code-learner/.venv/bin/python -m pytest -q
/home/keith/projects/code-learner/.venv/bin/ruff check .
/home/keith/projects/code-learner/.venv/bin/mypy --ignore-missing-imports codelearner
git diff --check
```

Expected: pytest exits 0 with only documented skips; Ruff and mypy exit 0; diff check emits no output.

- [ ] **Step 6: Commit the phase acceptance slice**

```bash
git add tests/test_semantic_search.py codelearner/eval/ablation.py docs/IMPLEMENTATION-GUIDE.md README.md
git commit -m "test: prove supported semantic search end to end"
```

## Phase completion gate

After all task reviews are clean, commission one whole-branch review against the design specification. Permit one coherent fix wave and one scoped re-review. Then independently rerun the full suite, Ruff, mypy, and worktree cleanliness checks before presenting branch integration options.
