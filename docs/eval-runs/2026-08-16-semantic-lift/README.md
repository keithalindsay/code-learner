# Minimal semantic-lift run — 2026-08-16

Reproducibility artifacts for the Phase 11b MVP measurement. See
`../../superpowers/specs/2026-08-16-semantic-lift-results.md` for the write-up and
`...-design.md` for the protocol and integrity guards.

Scripts (run against a fresh v7 kalshi-bot index with dense embeddings, paths inlined):
1. `baseline_gate.py`   — source-only pass/fail over hand_kalshi_bot.json; writes baseline_subset.json
2. `generate_claims.py` — targeted `learn` (llama3.1:8b) over the failure subset's subjects
3. judge: `codelearner judge <repo> --index <idx> --model qwen3.5:9b`  (35 supported / 27 refuted)
4. `measure_lift.py`    — source-only vs assertions-on, paired delta_ci
5. `diag_ranks.py`      — per-query recovered/displaced diff

Note: the judge MUST be given the repo positionally so evidence resolves against the real
tree; see the "Defect found" section of the results doc.
