# Roadmap & status

[← code-learner](../README.md) · the case study

What ships and is measured, what does not, and where the open defects live.

---

## Status

**Status.** Ingest, storage, tier-1 name resolution, symbol-boundary chunking, the
full hybrid retrieval pipeline, the tier-2 assertion store and its staleness engine,
onboarding tours, the CLI, the MCP server, and claim generation (`codelearner learn`)
all ship and are measured. The parts that do *not* ship are named in
[Roadmap](#roadmap) and in [docs/REMEDIATION.md](REMEDIATION.md), which is the
open-defect list rather than a wish list.

## Roadmap

| Phase | | Status |
|---|---|---|
| 0 | Spike on a real repo | done |
| 1 | Ingest + store + tier-1 resolution | done |
| 2 | Symbol-boundary chunking + FTS5 lexical index | done |
| 2b | Dense embeddings (`Qwen3-Embedding-0.6B`) into sqlite-vec | done |
| 3 | Hybrid retrieval: RRF fusion + graph expansion | done |
| 3b | Cross-encoder reranking (`zerank-1-small`) | shipped, **not measured** — the 16-query result is withdrawn and no replacement exists |
| 4 | Assertion pipeline + adversarial gate | done |
| 5 | Staleness engine | done — the two-stage verifier ships with no production caller (WP17.5) |
| 6 | Onboarding tours | done |
| 7 | MCP server + CLI | done |
| 8 | Eval: per-modality ablation, faithfulness with a same-weights-different-model judge, git-history purpose gold, gate negative controls | done; see [Measured](RESULTS.md) |
| 9 | Claim generation: cited drafts, `codelearner learn`, a measured generator | done |
| 10 | Adjudication as admission control — a `judge` command and `require_verdict` | **done** (WP17.4). `codelearner judge` drives the independent-judge machinery and records verdicts through `store.record_verdict`; serving already refused an unjudged claim (`require_verdict=True`), so this closes the `search → submit → judge → serve` loop. Judging is CLI-only and out-of-band by design — an agent must not judge its own claims, so there is no MCP judge tool |
| 11 | Semantic retrieval: claims as first-class search results | done — lexical assertion retrieval, mixed fusion, all-or-nothing evidence, one candidate shape across CLI and MCP. `facts_only` and `--no-assertions` are real controls now |
| 11b | Semantic retrieval **lift** — assertion embeddings and a five-repository semantic gold set | **not started**. Nothing yet measures whether serving claims helps; the ablation's tier-2 row is a count, not a result |
| 12 | Second language | deferred; the type seam is clean, the dispatch is not built |

Deliberately not on this list, because a roadmap is not the place to hide a defect: the
open items are in [docs/REMEDIATION.md](REMEDIATION.md), which names WP16 (surviving
mutations — partial; the embedder-mismatch guard is now closed and pinned) and WP17 (one
rule in one place — `indexinfo.py` still unextracted; the rest of the list has moved,
see REMEDIATION.md for the current item-by-item state) and WP18 (this document,
packaging, and the Python-only guard) as outstanding. WP17.4 (adjudication) is now
done — see Phase 10 above.
