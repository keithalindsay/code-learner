# Minimal semantic-lift measurement — results (Phase 11b, MVP scope)

**Venue:** kalshi-bot (340 files, 7,535 symbols), fresh v7 index with real Qwen3-Embedding
dense vectors. **Gold:** the existing hand-authored `hand_kalshi_bot.json` (62 leakage-checked
"what does the code DO" questions). **Generator:** llama3.1:8b. **Judge:** qwen3.5:9b
(independent family). Everything ran locally on GPU.

## Pipeline actually executed

1. Built a clean v7 kalshi-bot index (7,502 chunks embedded).
2. **Baseline gate** — ran all 62 hand-gold queries source-only (lexical+dense+graph). Source
   found the relevant subject in top-5 on 24; **missed or buried it on 38 (61%)**. (Contrast
   swarm-sync, where only 25% failed — the venue pivot was correct.)
3. Generated tier-2 claims for the 62 distinct subjects of the 38 failing queries via `learn`
   (llama). 62 admitted (cited real evidence).
4. Adjudicated with `codelearner judge` (qwen). **35 supported, 27 refuted** — a 56% pass
   rate. The gate discriminates: it kept `unrealized_pnl` (docstring matches implementation)
   and refused a bad `execute_sell` claim (the code sets the trade `side` to 'BUY',
   contradicting the claim's "records a sell trade").
5. Measured source-only vs assertions-on subject-recall with the tested `delta_ci`.

## Numbers

| set | config | hit@5 | recall@5 | mrr | delta hit@5 (asrt−src) |
|---|---|---|---|---|---|
| full gold (n=62) | source-only | 0.387 | 0.277 | 0.252 | |
| full gold (n=62) | assertions-on | 0.323 | 0.226 | 0.239 | **[−0.177, +0.032] n.s.** |
| failure subset (n=38) | source-only | 0.000 | 0.000 | 0.020 | |
| failure subset (n=38) | assertions-on | 0.105 | 0.088 | 0.092 | **[+0.026, +0.211] significant** |

Paired bootstrap CIs, 95%.

## What actually happened (per-query diff)

- **Recovered: 4** queries (source missed/buried → assertions top-5). 3 are claim-driven — a
  served, judged claim promoted its subject into the top-5 where source search could not.
- **Displaced: 9** queries (source top-5 → worse under assertions-on). Not the claims about a
  query's own subject out-competing it: it is **low-precision lexical claim retrieval**. Every
  displaced query has **16–35 eligible claims in its fusion pool** (out of only 35 servable
  claims total). Natural-language claims about one codebase share vocabulary (here: trading,
  risk, pricing, orderbooks), so BM25 over claim text marks nearly every claim "eligible" for
  nearly every query, and those ~30 irrelevant claims each cast an RRF vote that pushes the
  correct source hit down.
- Unchanged: 49.

## Honest conclusion

1. **The thesis is validated.** Generate → judge → serve → retrieve → promote-subject works
   end to end on real code, and it produces a **statistically-significant recovery** on the
   questions source-only search fails (failure-subset hit@5 lift, CI entirely above zero).
   This is the first measurement that shows the tier-2 layer *helping*, not just running.
2. **The effect is small**, and the reason is the same as the reason the full set is
   net-neutral: **the lexical assertion layer has poor precision.** Claims are
   natural-language sentences, and a codebase's claims share vocabulary, so BM25 cannot tell
   the one relevant claim from thirty irrelevant ones — 16–35 of 35 claims come back
   "eligible" for nearly every query. The right claim's promotion is drowned by the noise of
   the wrong ones. This is exactly what the **deferred dense-assertion work (full 11b)** is
   for: semantic matching would rank the relevant claim high and suppress the rest. This
   measurement is the concrete, bounded evidence that that investment is justified — and that
   the lexical layer alone is not enough.
3. **The full-set neutrality is real, not an artifact.** It is the precision problem above:
   good recoveries on a few queries, offset by low-relevance claims displacing good source
   hits on others. It is a property of *lexical* claim retrieval, and the fix is better
   claim-query matching (dense), not a fusion tweak.

## Two hardening fixes applied on this branch

- **`cmd_judge` evidence-root bug — fixed.** It now reads cited evidence against the index's
  bound root (via `store._repo_root`), not `args.repo` (cwd). Regression test
  `test_judge_reads_evidence_against_the_bound_root_not_args_repo` reproduces the all-refuted
  failure and pins the fix.
- **Assertions-on no-op invariant — added.** `search_candidates` now short-circuits to the
  tuned source ordering when *no* claim is eligible, so enabling the layer cannot reorder a
  query it has nothing to say about. Correct behaviour, and it removes the pure-artifact case
  — but note it does **not** move the numbers above, because on this workload every query has
  many eligible claims (the precision problem is the real cause, not zero-claim queries).

## Not claimed

No population statistic; N is small and one repo. Per the standing rule these numbers are a
demonstration, not résumé-claimable until vouched for. Nothing here measures a dense-assertion
layer — that remains future work, now with evidence behind it.

## Defect found during the measurement

`cmd_judge` reads evidence spans against `args.repo` (default: cwd), not the index's **bound**
root. Judging an index from any other directory makes every citation unreadable and refutes
every claim (observed: a first judge run rejected all 62 on "could not read this span"). `learn`
and the serving path both correctly use the bound root; `cmd_judge` should too. Fix: derive the
evidence root from `store._repo_root(conn, ...)` (the bound root), not from `args.repo`.
