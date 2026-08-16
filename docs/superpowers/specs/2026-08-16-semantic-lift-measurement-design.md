# Minimal semantic-lift measurement (Phase 11b, MVP scope)

## Question

Does the tier-2 semantic-assertion layer improve retrieval on the questions source-only
search (lexical + dense + graph) fails — i.e. when a judged claim exists about the relevant
symbol, does the layer surface that symbol where the code's own text does not?

This is the measurement the project has deferred since Phase 2. It is what turns "the
plumbing works" into "the layer earns its keep." The outcome is **not predetermined**: if
the claims do not recover the missed subjects, that is the honest result and it is reported
as such.

## Scope (deliberately minimal)

- **One repo: kalshi-bot.** Chosen after a feasibility probe on swarm-sync showed swarm-sync
  is too heavily documented to demonstrate lift — its narrative docstrings let dense
  retrieval already answer 6 of 8 semantic questions, leaving nothing for the layer to add.
  kalshi-bot has average documentation (66% docstring coverage) and, per its gold notes,
  hard-negative duplicated-concept traps (two Kelly sizers, two fee formulas, etc.) — the
  conditions the layer is built for.
- **No new statistics.** The ablation harness (`eval/ablation.py`) already provides
  recall@k, hit@k, mrr, ndcg, map, and cluster-robust bootstrap confidence intervals with a
  paired `delta_ci`. This measurement is a gold subset + wiring, not new methodology.
- **Not** the full 11b: no assertion-specific embeddings, no five-repository gold set, no
  learned ranking. Those remain future work.

## Method

1. **Index prep.** Build a v7 kalshi-bot index with real dense embeddings into a scratch
   path (Keith's own `.codelearner/index.db` is left untouched). Verify: v7, embeddings
   present, zero pre-existing assertions.
2. **Baseline-failure subset.** Reuse the existing `hand_kalshi_bot.json` gold (62
   leakage-checked, coverage-coded queries, 10 hard-negatives). Run each source-only and
   keep the subset (~12–15) where the relevant subject is **missed or buried (rank > 5)**.
   Reusing the existing gold — rather than authoring questions to fit — is the stronger,
   less circular design.
3. **Claims.** For the relevant subjects in that subset, generate candidate claims with the
   real `learn` pipeline (`llama3.1:8b`) and adjudicate them with the real `codelearner
   judge` (`qwen3.5:9b` — a different model family, so independence holds and verdicts
   mean something). Only supported, live-verified claims become servable. Freeze the judged
   store so the scored comparison is reproducible.
4. **Measure.** Run the ablation on the subset in two configurations — source-only vs
   assertions-on — and report recall@5, recall@10, hit@5, mrr, and the paired `delta_ci`.
5. **Report.** Write the result honestly: N stated, one repo, labelled a demonstration, with
   the envelope (works on hard-negative / code-opaque questions; dense already wins on
   well-documented code).

## Integrity guards (what keeps this honest)

- **Anti-leakage:** questions come from the existing gold, whose authoring rule already
  forbids reusing the target's identifier tokens; the subset is filtered on measured
  source-only failure, not hand-picked.
- **Baseline verified first:** the source-only baseline is measured *before* claims exist,
  so it cannot be strawmanned. Dense is real (embeddings present), not a lexical-only stand-in.
- **Claims are model-written, not authored to fit:** the generator reads the code and writes
  the claim; the author of this measurement does not write claims to match the questions.
- **Independent judge:** generator `llama3.1:8b`, judge `qwen3.5:9b` — different families, so
  a claim is served only because a model that did not write it agreed with its evidence.
- **Reproducible:** the gold subset is committed; the judged store is frozen; the scored
  comparison re-runs deterministically.
- **Honest framing:** the result is a demonstration at its true N on one repo — not a
  population statistic, and not résumé-claimable until Keith vouches for it.

## Success criteria

Not "lift must be positive." Success is a **credible, honest number with its CI**, whatever
its sign, plus the envelope statement. A null or small result is a valid finding: it would
say the lexical assertion layer needs the deferred dense-assertion work (full 11b) to help,
which is itself worth knowing before investing in that.
