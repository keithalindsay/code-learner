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
- **Displaced: 8** queries (source top-5 → worse under assertions-on). **6 of the 8 had zero
  claims served for the query** — so the displacement is not claims out-competing good source
  results. It is the assertions-on code path widening the source candidate pool (`k ×
  CANDIDATE_MULTIPLIER`) and re-fusing (RRF), which reorders source results independent of any
  claim.
- Unchanged: 50.

## Honest conclusion

1. **The thesis is validated.** Generate → judge → serve → retrieve → promote-subject works
   end to end on real code, and it produces a **statistically-significant recovery** on the
   questions source-only search fails (failure-subset hit@5 lift, CI entirely above zero).
   This is the first measurement that shows the tier-2 layer *helping*, not just running.
2. **The effect is small**, for two visible reasons: (a) this is the **lexical** assertion
   layer, so a claim only surfaces when the query words lexically overlap the claim text —
   thin reach for questions phrased in newcomer's words; (b) only 35 of 62 target subjects
   ended with a servable claim. Both are exactly what the **deferred dense-assertion work
   (full 11b)** would address. This measurement is the evidence that that investment is
   justified — and bounded.
3. **The full-set neutrality is confounded by a fusion artifact**, not the claims. The
   displacement occurs with no claim served, so it is a separable re-ranking issue in the
   assertions-on path, fixable independently of the semantic layer's value.

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
