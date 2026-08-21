# Critical decisions

[← code-learner](../README.md) · the case study

The design calls this project turns on, and the engineering discipline that caught its own mistakes. Each decision links to the section that measures or demonstrates it.

---

## How this was built

Written with Claude Code, by an engineer who specifies, audits, and sets the bar the
work has to clear. The standing rule: *a fix without a test that fails when you
delete the fix is not a fix.* The three regression fixes in Phase 0 were each
mutation-verified — delete the fix, confirm the test fails, restore, confirm green.

That rule caught a bad test of its own in Phase 3. A test asserting that graph
activation *accumulates* across seeds passed even when `+=` was replaced with
`max()` — it was measuring seed rank, not accumulation. It was rewritten to control
seed order so that only summing can produce the asserted outcome, then re-checked
against the same mutation. A test that survives deleting the behaviour it names is
not a test, whoever wrote it.

The same rule caught something larger at the level of the *apparatus*. `Outcome.held`
is the predicate every gate number is computed through, and three of its conjuncts
could be deleted with the whole suite green. The reason is worth generalising: an
accepted outcome carries `code=None`, which is in no family's code set, so the
negative branch's `verdict == REFUSED` check cannot be observed failing by any corpus
run. Those conjuncts were not dead code — they were **unreachable by the corpus that
scores them.** The negative-control apparatus can only ever test itself against
situations it already knows how to generate, so its coverage of its own scorer has to
be established by construction and never by running it. It is pinned now by a 12-row
table over synthetic outcomes, including both polarities on both branches so an
always-`False` mutation fails too.

The full open-defect list, with provenance markers separating what was reproduced from
what was reported from what is hypothesis, is [docs/REMEDIATION.md](REMEDIATION.md).

## The decisions that shaped the system

**Inference is shippable when it is accountable, not when it is certain.** The competitors that index code for agents refuse inference outright — *"structural facts only"* — because an LLM-generated rationale goes stale silently and a confidently-wrong one is worse than none. The bet here is that the problem is unaccountability, not inference, and that four properties (evidence-bound, hash-bound, adjudicated, tier-labeled) make it admissible. See [Why this exists](../README.md#why-this-exists).

**The tool never calls an LLM; the agent calls *in*.** A retrieval tool that answered "what does this guarantee" would need an API key, a bill, and a second place an unciteable sentence can be born. Inverting it costs nothing: the agent is already running, so it does the judging and submits through a gate that decides whether the judgement may be kept. See [MCP server](INTERFACES.md#mcp-server).

**Citations are menu integers, not byte offsets.** A model asked for a path and a byte range produces something in that shape whether or not it read those bytes, and an invented offset verifies forever against nothing. Handed a numbered menu the index built, an invented citation is not a bad citation — it is *not* a citation. On one run the model cited off its own menu 23 times; under a byte-offset design every one would have become a permanently-verifying lie. See [Generation](RESULTS.md#generation-the-claims-and-the-numbers-that-judge-them).

**A resolution number was cut 12.7 points on purpose.** In-repo resolution briefly reached 76.2% on a strategy that bound dotted attribute calls by unique name — pointing 38 `r.json()` calls at a helper in a test file. Removing it cost coverage and removed a class of confident fabrications; that trade is the whole project, and it showed up in the resolver on day one. See [Ingest and resolution](RESULTS.md#ingest-and-resolution).

**The headline retrieval finding is the one the architecture did not want.** The shipped default beats lexical-only by +0.120 nDCG@10 — but the ablation shows almost none of that is the extra modalities: a one-line test-demotion does the work three retrieval modalities and a fusion algorithm were built to do, and dense retrieval loses to lexical on this corpus. It is reported as the finding, not buried. See [Retrieval](RESULTS.md#retrieval).

**A number that stopped reproducing was withdrawn, not kept.** The reranking rows first published did not reproduce against the shipped code, and the corrected rows were then withdrawn too for resting on a 16-query set below the calibration floor. No reranking figure is published until it is re-measured. See [Retrieval](RESULTS.md#retrieval) and [Methodology](METHODOLOGY.md#what-these-numbers-cannot-resolve).

**Judge independence is enforced structurally.** Faithfulness is judged by a different model from the generator, and judging stays a CLI-only, out-of-band step — there is no MCP judge tool, because an agent must not judge its own claims. See [Faithfulness](RESULTS.md#faithfulness-does-a-claim-follow-from-what-it-cites).
