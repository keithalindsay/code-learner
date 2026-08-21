# Methodology & reproduction

[← code-learner](../README.md) · the case study

The conventions every number in this case study obeys, the power analysis that says which questions the gold sets can and cannot settle, and how to reproduce every deterministic table from a clean checkout.

---

## How to read the numbers in this document

This project's entire pitch is that it measures itself honestly, so the conventions
below apply to every table and are stated once instead of re-argued under each one.
They exist because an audit of this file found three co-existing versions of one
measurement presented as co-equal, about 170 lines apart, and a headline quoting a
figure that had been publicly retracted four sections earlier.

**Every table is stamped `repo@sha`.** These measurements are deterministic and
reproduce exactly at a given sha. They are *not* stable across shas, because the
repositories under measurement keep being worked in — a mined gold set is a function
of a repo's history and a symbol count is a function of its size. An uncaptioned
table is what turns ordinary bookkeeping into an apparent instability, and it is what
lets two versions of one number sit in the same document without either looking
wrong.

The shas everything below is measured at:

| repo | sha | role |
|---|---|---|
| code-learner (itself) | `3212972` | the code, and the repo the gate corpus runs against |
| [swarm-sync](https://github.com/keithalindsay/swarm-sync) | `3119a97` | retrieval gold, purpose gold, the tier-2 store |
| kalshi-bot | `8a2e9b5` | retrieval gold (private) |
| TradingAgents | `f362a16` | retrieval gold |

**A number never travels without its interval.** Where an interval is a bootstrap it
says what was resampled — queries, or repos. Where a comparison is between two rows
scored on the same queries, the interval is *paired*; overlapping marginal intervals
say nothing about a paired difference and are not used to argue one.

**Below 128 queries there is no 95% interval here.** The percentile bootstrap this
repo uses was calibrated against a null built from its own per-query noise: it
rejects at 11.9% at n=16, 7.9% at n=32, and reaches its nominal 5% around n=128
(`ablation.CALIBRATION_FLOOR`). It errs *narrow* — in the direction that invents
findings. Any table under that floor is readable for its shape and is not an
interval, and this is stated where it applies rather than assumed known.

**Where a number could not be re-run for this document, it says so and carries the
date and sha of the run it comes from.** One measurement in this repo needs a
language model (faithfulness) and one needs a full generation pass (`codelearner
learn`); neither was re-executed while writing this, so both are reported from the
stored artefacts they wrote, and the derivation from those artefacts is shown.

---

## What these numbers cannot resolve

This section exists because the honest answer to "should we tune X?" is usually "this
gold set cannot tell you", and that answer is worth more than a table that pretends
otherwise.

**The measured noise curve.** Measured directly on this run's lexical-vs-hybrid pair at
`3212972`, the per-query paired sd is **0.2865 for nDCG@10** and **0.3522 for MRR**,
giving bootstrap half-widths of `0.56/√n` and `0.69/√n`. The corpus-wide study behind
`ci_half_width` — 55 real paired comparisons over 11 configurations, subsampled to n=8
and resampled to n=512 — put the median at `≈ 0.60/√n` for ΔMRR, stable within 5% across
that range. Both are true and the pair is the point: **`0.60/√n` is a median, not a
constant**, and a specific comparison can be a sixth noisier than it. That study is
recorded in `ablation.ci_half_width` and was not re-executed for this document; the two
sd figures were.

The load-bearing property is that the sd belongs to the *corpus* and not to the effect
(`corr(|mean diff|, sd) = 0.15`, from the same study). A small-looking effect cannot be
assumed to be a quiet one, so "the difference is tiny, it will be easy to resolve" does
not follow.

**The 16-query "95% CI" was not one.** Under a true null built from the real per-query
noise, the percentile paired bootstrap rejects at **11.9%**, not 5% — 7.9% at n=32, 6.0%
at n=128. It errs narrow, toward inventing findings. Every conclusion this project drew
from the 16-query set was computed with a miscalibrated interval, including the earlier
finding that only `prefer_implementation` survived. Hence `CALIBRATION_FLOOR = 128`, and
hence the withdrawal of the reranking rows above. (Those three calibration figures are
asserted by `ablation.CALIBRATION_FLOOR` and its diagnostics, from the same WP22 study;
they were read from the code, not re-derived here.)

**Repos buy power where queries do not.** The repo design effect measured at `3212972`
on the per-query nDCG@10 difference between `lexical only` and `hybrid + prefer_impl` is
**4.60** (3.92 for MRR) — so 520 clustered queries carry the evidence of about **113**
independent ones. The intra-class correlation behind that is small, about **0.021**, and
the design effect is large anyway because the clusters are: `1 + (173 − 1) × 0.021`.

That arithmetic has a consequence worth more than the number. Effective n is
`m·q / (1 + (q−1)·ICC)` for `m` repos of `q` queries, which **saturates at `m / ICC`** —
three repos cannot exceed roughly **140** effective queries however much gold anyone
writes against them. The next marginal repo is worth far more than the next marginal
query, and a plan to "write more gold" has to say which repo it is for.

**What 638 rows buy.** Queries needed at 80% power, nDCG@10, at this run's measured sd:

| effect to resolve | effective n | clustered n (deff 4.6) |
|---|---|---|
| Δ = 0.15 | 29 | 132 |
| Δ = 0.10 | 65 | 297 |
| Δ = 0.05 | 258 | **1,186** |

Δ = 0.15 and Δ = 0.10 are achievable and are the sizes of the effects this file
reports. **Δ = 0.05 needs about 1,200 clustered queries and should not be promised.**
Several comparisons this project would like to iterate on — reranking configurations,
graph weights between 0.2 and 0.4, prompt variants — live below that line, and this
gold set will not settle them.

**Why nDCG@10 is primary.** On measured grounds, not taste. The mechanism is
reproducible here and is the reason: MRR reads *one* position, so a configuration that
puts the last remaining relevant symbol at rank 2 and one that loses it entirely score
identically, and on the multi-relevant subset its variance nearly doubles. The sizing
consequence is visible in this run's own sd — 0.2865 for nDCG@10 against 0.3522 for MRR
on the same pair, which is 33% fewer queries for the same delta (258 against 390 at
Δ=0.05). The WP22 study adds two facts that were not re-run for this document: nDCG@10
separates 23 of 55 real configuration pairs at n=16 against MRR's 11, and agrees with
MRR's ordering on 54 of 55 — the same question asked with less noise, not a different
question.

**`run_ablation` used to fail silently at exactly this seam.** Handed a gold set that
did not match the index, it returned a full table of `0.000` with `[0.000, 0.000]`
intervals — output shaped exactly like a finished measurement and containing none. No
retriever can rank a symbol that is not in the index, so that was never a result about
retrieval. Gold is validated before scoring now, and in the multi-repo case *every*
repo is validated before *any* is scored, so a stale third repo does not cost a GPU pass
on the first.

---

## Verification

```bash
.venv/bin/python -m pytest tests/ -q                      # 844 passed at 3212972
.venv/bin/ruff check .                                    # All checks passed!
.venv/bin/mypy codelearner --ignore-missing-imports       # no issues in 45 source files
```

844 is the count at `3212972` on a clean tree. It is stamped for the same reason the
tables are: the suite grows, and an unstamped test count is a number that looks stable
and is not.

### Reproducing what is in this document

Every deterministic table above is reproducible from a checkout plus the indexes. The
gate corpus builds its own index of a copy of the tree, so it needs nothing but the
repo:

```bash
# The gate: both doors, then the mutation census at each
.venv/bin/python -m codelearner.eval.gate_controls --repo . --compare
.venv/bin/python -m codelearner.eval.gate_controls --repo . --mutations
.venv/bin/python -m codelearner.eval.gate_controls --repo . --surface store --mutations
```

Purpose gold needs only a repo with git history, and calls no model:

```python
from pathlib import Path
from codelearner.eval import run_purpose_eval, format_report

print(format_report(*run_purpose_eval(Path("/path/to/repo"))))
```

Retrieval needs one embedded index per repo the gold set names. `run_ablation_multi`
scores each repo against *its own* index; `run_ablation` takes one connection, which is
all a single-repo gold set needs and is silently wrong for a multi-repo one:

```python
from pathlib import Path
from codelearner import db
from codelearner.index import SentenceTransformerEmbedder
from codelearner.eval import format_table, run_ablation_multi, stratified_cards

conns = {
    "swarm-sync":    db.connect(Path("/path/to/swarm-sync/.codelearner/index.db")),
    "kalshi-bot":    db.connect(Path("/path/to/kalshi-bot/.codelearner/index.db")),
    "TradingAgents": db.connect(Path("/path/to/TradingAgents/.codelearner/index.db")),
}
cards = run_ablation_multi(
    conns,
    SentenceTransformerEmbedder(),
    gold_name=["hand_swarm_sync", "hand_kalshi_bot", "hand_tradingagents",
               "mined_swarm_sync", "mined_kalshi_bot", "mined_tradingagents"],
)
print(format_table(stratified_cards(cards)))   # pooled row FIRST, then per-source, per-repo
```

Read `stratified_cards`, not the pooled row alone. Use `Scorecard.delta_ci(other,
metric="ndcg", k=10)` for a comparison and `cluster=True` to resample repos instead of
queries; `paired_sd`, `design_effect` and `required_n` are what produced [What these
numbers cannot resolve](#what-these-numbers-cannot-resolve). The whole table takes
about four minutes on a 10GB RTX 3080.

The reranked rows are opt-in for the same reason the CLI flag is — they cost a model
forward pass per candidate, and the modality rows must stay runnable on a machine with
no GPU:

```python
from codelearner.retrieve import load_reranker

reranker = load_reranker(conn=conns["swarm-sync"])   # None if no model can be loaded
cards = run_ablation_multi(conns, embedder, gold_name=[...], reranker=reranker)
```

`load_reranker` returns `None` rather than raising when there is no model, no network,
or no memory, and `run_ablation` and `search` both read `None` as *skip the stage*.

Faithfulness is the one thing here that needs a model. It is not run in CI, it is not
run by the test suite — no test in this repo may call a model — and it is not run by
anything that reports a number without saying which pass it came from.
