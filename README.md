# code-learner

GraphRAG over a codebase — for agents that need to trace it, and for engineers
onboarding into it.

Point it at a repository. It parses the code into a knowledge graph of modules,
classes and functions, resolves the references between them, and layers on *purpose*
— what each piece is for. Every inferred claim cites the source spans it was drawn
from, and expires the moment those spans change.

A case study in designing an AI system that infers, and stays accountable for what it infers. Written for a technical reader tracing the reasoning: the problem, the design calls, the architecture, and — at length — how each claim was measured and where the measurement runs out. It is built with Claude Code by an engineer who specifies, audits, and sets the bar the work has to clear.

**Scope: Python only.** Extraction is tree-sitter-python and `*.py` is hardcoded.
Pointed at a Go or TypeScript repo it indexes zero files and, today, still exits 0 —
a known defect, tracked as WP18.3 in [docs/REMEDIATION.md](docs/REMEDIATION.md).

## Why this exists

Indexing a codebase for agents is a crowded category —
[CodeGraph](https://github.com/colbymchenry/codegraph) (63.2k stars) and
[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) both do the
structural half well, and this project does not claim to beat them at it.

What none of them do is the *purpose* layer. CodeGraph's docs are explicit: **"no
inference — structural facts only."** That is a defensible refusal, because
LLM-generated purpose is unverifiable and goes stale silently, and a
confidently-wrong rationale injected into an agent's plan is worse than no
rationale at all.

The bet here is that inference isn't unusable, it's *unaccountable* — and that four
properties make it shippable:

1. **Evidence-bound** — every claim cites concrete `file:line` spans. No citation,
   no entry.
2. **Hash-bound** — the cited spans' hashes are stored. Any of them changing marks
   the claim stale rather than serving it.
3. **Adjudicated** — independent judges try to *refute* each claim using only its
   cited evidence. Refused claims are logged, not deleted.
4. **Tier-labeled** — callers can demand facts only, and trust nothing inferred.

Property 4 used to be the one to read sceptically: `facts_only` was wired through the
CLI and the MCP `search_code` tool and filtered nothing, because no retrieval path
emitted a tier-2 modality (WP17.3). Phase 2 closed that. Lexical retrieval over the
assertion store now returns claims as first-class results, `facts_only` removes them
BEFORE the page is cut and refills the freed slots with source, and the flag changes
what you get on any index that holds claims.

What is served is deliberately narrow. A claim is returned only when it is `active`,
at least one judge recorded `supported`, no judge recorded `unsupported` or
`refuted`, and every range it cites still hashes to the bytes it was written against
— re-checked on the serving call, not remembered. Pending claims are retrievable only
through a research policy named at the library boundary (`RESEARCH_PENDING_POLICY`);
they are never reachable from the CLI or MCP. Rejected and stale claims are not
servable under any policy. Searchable unjudged claims would turn generation into
publication, which is the one thing this project exists not to do.

Dense assertion embeddings and the five-repository comparative evaluation are Phase
2.5 and are NOT done: nothing here measures whether the semantic layer improves
retrieval quality. The plumbing is honest; the lift is unmeasured.

## Results at a glance

Every figure here carries the caveat that bounds it; the full table is one click away, never quoted without its interval.

- **The gate refuses every enumerated attack, at both doors — 100.0% refused, attributed, and positive.** That is a statement about the *enumerated* attack shapes and nothing else; the corpus has found no attack nobody had already named. → [Results: the gate](docs/RESULTS.md#the-gate)
- **The shipped retrieval default beats lexical-only by +0.120 nDCG@10 — and almost none of that is the extra modalities.** A one-line test-demotion does the work three modalities were built for; dense loses to lexical on this corpus. The finding is reported, not buried. → [Results: retrieval](docs/RESULTS.md#retrieval)
- **Faithfulness is 0.54 [0.46, 0.62] for local-model claims and 0.70 [0.62, 0.77] for a frontier generator** — read as *among the claims it chose to make*, because the stronger model also declined the hard ones. → [Results: faithfulness](docs/RESULTS.md#faithfulness-does-a-claim-follow-from-what-it-cites)
- **On purpose accuracy a frontier model halves the gap to a bag of body identifiers and still loses** — a finding about the metric (token-F1 rewards vocabulary, not meaning), not the model. → [Results: purpose accuracy](docs/RESULTS.md#purpose-accuracy-gold-labels-mined-from-git-history)
- **In-repo resolution was cut 12.7 points on purpose**, to remove a class of confident fabrications a unique-name binder produced. → [Results: ingest and resolution](docs/RESULTS.md#ingest-and-resolution)

## The case study, in sections

- **[Architecture](docs/ARCHITECTURE.md)** — the tier model, the tier-2 assertion store, the staleness engine, onboarding tours
- **[Critical decisions](docs/DECISIONS.md)** — the design calls and the test-that-fails-when-deleted discipline behind them
- **[Methodology & reproduction](docs/METHODOLOGY.md)** — how to read the numbers, what they cannot resolve, and how to reproduce them
- **[Results](docs/RESULTS.md)** — ingest · retrieval · the gate · faithfulness · purpose accuracy · generation
- **[Robustness](docs/ROBUSTNESS.md)** — repository isolation, the adversarial gate, serve-time verification
- **[Interfaces](docs/INTERFACES.md)** — the CLI and the MCP server, with real captured output
- **[Roadmap & status](docs/ROADMAP.md)** — what ships, what does not, and the [open-defect list](docs/REMEDIATION.md)

## Quickstart

Requires **Python 3.12+**.

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

Full setup, the CLI, and the MCP server config are in [Interfaces](docs/INTERFACES.md); reproducing the measurements is in [Methodology](docs/METHODOLOGY.md#reproducing-what-is-in-this-document).

## A note on the numbers

This began as one long document, deliberately — so no headline could be quoted without the caveat that bounds it, a failure an audit of it caught twice. Splitting it into sections keeps that rule rather than breaking it: every number lives in the section that measures it, beside its interval and its limits, and the few results quoted on this page carry their caveat with them. The conventions they obey — `repo@sha` stamping, paired intervals, the calibration floor below which there is no 95% interval here — are in [Methodology](docs/METHODOLOGY.md#how-to-read-the-numbers-in-this-document).
