# Adjudication as admission control (WP17.4)

## The gap

The store enforces admission control on the *serving* side already: `ServingPolicy`
defaults to `require_verdict=True`, and `evaluate_metadata` refuses to serve any claim
that lacks an accepted `supported` verdict (`assertions/policy.py`). Search, `get_symbol`,
CLI and MCP all inherit that refusal.

What is missing is the other half: nothing shipped ever *produces* those verdicts. The
adjudication machinery exists and is well-built — `adjudicate_assertion`, the `Judge`
protocol, `OllamaJudge`, `Judgement`, the label mapping, and `record_verdict` — but every
piece lives in `eval/faithfulness.py`, and its only non-test caller is the offline
faithfulness measurement. So a claim submitted through `submit_assertion` sits `active`
and unjudged forever, and the serving gate silently withholds it. The loop
`search → submit → judge → serve` is open at `judge`.

WP17.4 closes it by giving the existing machinery a shipped surface, without importing
`eval` upward into the CLI.

## Non-goals

- No new judging logic, prompt, or parser. This wires what exists; it does not reopen how
  a verdict is reached.
- No change to serving policy. `require_verdict=True` is already the default and is not
  touched.
- No MCP judge tool. Judging is an out-of-band operator action; see "Surface" below.
- No calibration harness or judge-comparison CLI. Those remain measurement, in `eval/`.

## Surface: a CLI command, not an MCP tool

`codelearner judge` is an operator command, sibling to `learn`. It is deliberately **not**
an MCP tool.

The judge's verdict means something only because the judge did not write the claim
(`eval/faithfulness.py` states this at length: different weights, different tokenizer,
independence is the whole basis of the number). An MCP `judge` tool would invite the same
agent that generated a claim to adjudicate it inline — a generator grading its own output,
which the module already names as the failure that makes the score meaningless. Keeping
judging on the CLI, run by a human against a chosen independent model, keeps the gate
honest by construction rather than by hoping the agent picks a different model.

## Architecture: extract the machinery to a leaf

The CLI cannot import from `eval` without extending the four-package cycle
`eval → server → cli → generate → eval` that WP17 exists to close (adding `cli → eval`
would make `eval → server → cli → eval`). So the judge-facing symbols move down to a leaf.

**New module `codelearner/adjudicate.py`** holds the symbols a judging *caller* needs:
`Judge` (protocol), `Judgement`, `OllamaJudge`, the label constants and mapping,
`parse_judgement`, `render_evidence`, the `_NO_EVIDENCE` sentinel, `adjudicate_assertion`,
and the `Adjudication` result. These depend only on `assertions.store`, `generate.llm`
(for `model_family`), and the standard library — all leafward, no cycle.

`eval/faithfulness.py` keeps everything that is genuinely *measurement* (the
`FaithfulnessReport`, the Wilson intervals, the cause accounting, `score`/`score_decided`)
and **re-exports** the moved symbols from `codelearner.adjudicate` so existing imports and
tests keep working unchanged. An AST test asserts the module-level import graph stays a DAG
and that `cli` does not import `eval`.

This is the same extraction pattern Phase 2 used for `tier.py` and `sourceview.py`: move the
shared thing to a leaf, re-export for compatibility, pin acyclicity with a test.

## The command

```
codelearner judge <repo> [--index PATH] [--limit N] [--model TAG]
                         [--subject QUALNAME] [--allow-same-family] [--dry-run] [--json]
```

Flow:

1. Resolve the index (same resolution `learn`/`search` use).
2. Enumerate **candidates**: `active` assertions that have no accepted verdict yet — the
   claims the serving gate is currently withholding for lack of judgement. A new store
   query `unjudged_assertions(conn, *, limit, subject)` returns them; it must not mutate.
3. Build the judge (`OllamaJudge`, default model per the module, overridable with `--model`).
4. **Independence check.** For each candidate, compare the judge's `model_family` against
   the claim's `generator` family. If they match, refuse to record and count the claim as
   `skipped_same_family` — unless `--allow-same-family` is passed, in which case proceed and
   say so in the summary. Default is refusal: a same-family verdict is not admission
   evidence, and silently recording one is how the gate comes to certify nothing.
5. For each remaining candidate, call `adjudicate_assertion(conn, judge, assertion, root,
   record=not dry_run)`. `--dry-run` judges and reports without writing verdicts, for trying
   a model against the store before letting it change anything.
6. Report counts: supported / refuted / uncertain / skipped_same_family / no_evidence, and
   under `--json` the per-claim verdicts.

Every write goes through `record_verdict`, never around it, so what a non-supportive verdict
does to a claim's status stays that function's single decision (already tested there).

## Testing

Tests inject a deterministic fake `Judge` — the repo forbids any test from calling a real
model. Coverage:

- A submitted, unjudged claim is withheld by search; after `judge` records a `supported`
  verdict, the same query serves it. This is the end-to-end proof the loop is closed.
- A `refuted`/`uncertain` verdict leaves the claim unserved and (for refuted) marks it
  rejected, via `record_verdict`'s existing rules.
- `--dry-run` judges but records nothing; the store is byte-identical afterward.
- Independence: a judge whose family matches the claim's generator is skipped by default and
  recorded only under `--allow-same-family`.
- `unjudged_assertions` returns exactly the active-and-unjudged set and does not mutate.
- AST acyclicity: `cli` imports nothing from `eval`; the import graph is a DAG.

## Adjacent closures pinned in the same branch

Two audit items are already closed by intervening work; this branch adds the tests that
lock them so a future change cannot silently reopen them:

- **WP16** — the MCP tool surface refuses dense retrieval on an embedder-model mismatch. The
  guard is real and pinned at the unit and CLI layers; add the missing MCP-tool-level pin.
- **WP17.7** — the MCP server serializes all index work on a single owning worker, which
  removes the shared-connection concurrency the old `_atomic` bug needed. Add a regression
  guard asserting the worker stays single-threaded.
