# Product strategy: open source, commercial product, or both?

This decision should be made after the semantic retrieval milestone, not from the amount of
code already written. The repository is technically substantial, but product value depends on
whether accountable semantic memory changes agent outcomes enough to justify setup, model cost,
and trust concerns.

## Recommendation

Use an **open-core validation strategy now**, with no immediate commitment to a hosted SaaS.

- Keep the indexing engine, graph, local semantic store, CLI, MCP server, schemas, and core
  evaluation harness open source.
- Treat the next release as a developer preview focused on local/private use.
- Productize only after semantic-enabled retrieval demonstrates a material advantage over both
  bare tools and structural-only CodeGraph on correctness or engineering time.
- If the evidence is positive, monetize workflow and operations rather than hiding the graph
  algorithm: managed indexing, team memory, policy controls, review workflows, organization-wide
  search, integrations, and support.

This recommendation is reversible. Starting closed before product-market evidence would reduce
adoption and external validation; starting open does not prevent a later hosted or enterprise
offering if licensing and contribution policy are chosen deliberately.

---

## 1. What the possible product actually is

“Graph RAG for code” is not a sufficient product category. Structural code search is crowded,
and agents already have effective grep, language servers, embeddings, and competing code graphs.

The differentiated product thesis is narrower:

> A durable semantic memory for codebases that can explain why code exists, show exactly what
> evidence supports every explanation, and stop serving explanations when that evidence changes.

Likely valuable use cases:

- Onboarding into large or unfamiliar systems.
- Preserving architectural rationale that is otherwise trapped in authors' heads and commit
  history.
- Giving coding agents reviewed institutional memory.
- Impact analysis and test selection augmented by known invariants and risks.
- Regulated or high-assurance environments where generated explanations need provenance.
- Maintaining context across agent sessions without treating old generated prose as truth.

Weak product positions:

- “Faster code search.” Existing tools are good and CodeGraph is strong here.
- “More resolved edges.” Coverage without precision is not user value.
- “An MCP server with five tools.” Tool count is not a moat.
- “LLM summaries for every function.” These are easy to produce and frequently not useful.

---

## 2. Why open source is a strong default

### Trust and inspection are part of the value proposition

The system asks users to trust generated semantic claims while arguing that blind trust is
unsafe. Open schemas, gates, staleness rules, and evaluation methods reinforce that argument.
A closed core would make the accountability claim harder to verify.

### Source-code privacy favors local deployment

Many likely users cannot upload proprietary repositories or inferred architectural knowledge to
a new hosted service. Local-first operation removes the largest adoption objection and lets the
project build credibility before offering managed deployment.

### Distribution matters more than early capture

Code tooling benefits from broad language, framework, editor, and agent integrations. An open
project can attract resolver fixtures, benchmark repositories, model adapters, and integrations
that would be expensive for a small commercial team to build.

### The current moat is evidence and execution, not secrecy

The differentiators are:

- A coherent provenance model.
- Correctly implemented staleness semantics.
- A high-quality evaluation corpus.
- Operational integration into developer workflows.
- Accumulated trusted team knowledge.

The basic algorithms can be reproduced from papers and source reading. Keeping them closed is
unlikely to create durable defensibility.

---

## 3. Why a commercial product may still exist

Open source solves local indexing. Teams may pay for the coordination and operational layer:

- Shared, access-controlled semantic memory across repositories.
- Central policy for approved generators and judges.
- Human review queues and claim ownership.
- Audit history and compliance exports.
- GitHub/GitLab pull-request integration.
- Automatic stale-claim detection on proposed changes.
- CI gates for violated invariants or unsupported claims.
- Organization-wide dependency and impact views.
- Managed model routing, cost controls, and private inference.
- SSO, RBAC, retention controls, backups, and support.
- Evaluation dashboards showing whether the system helps each repository.

The commercially defensible asset is the team's accumulated reviewed knowledge and its workflow,
not the SQLite schema itself.

---

## 4. Strategy options

### Option A — Pure open-source project

Best if:

- Semantic retrieval provides modest but not transformative gains.
- Most usage is individual and local.
- Users resist any hosted processing.
- Maintainer motivation is research, reputation, or ecosystem influence.
- Support and integration demand is low.

Advantages:

- Fastest trust and adoption path.
- Lowest operational burden.
- Strong fit with local/private repositories.
- Easier research collaboration and reproducibility.

Risks:

- Maintenance burden without reliable funding.
- Competitors can package the work more effectively.
- Complex model and index setup may suppress adoption without commercial polish.

Possible funding:

- Sponsorships.
- Grants.
- Paid support and implementation.
- Dual licensing for embedding in proprietary products.

### Option B — Open core plus hosted/team product

Best if:

- Local developers value the engine.
- Teams want shared reviewed claims and policy.
- Security-conscious buyers need self-hosting.
- The benchmark shows meaningful semantic lift.

Open components:

- Extractors and resolvers.
- Local database and indexer.
- Retrieval and evidence assembly.
- Assertion schema, gate, and staleness engine.
- CLI and MCP server.
- Evaluation harness and public gold sets.

Commercial components or service:

- Multi-user review and approval workflow.
- Cross-repository organization graph.
- Hosted or enterprise control plane.
- RBAC, SSO, audit exports, and policy administration.
- CI and pull-request enforcement.
- Fleet observability, backups, and managed upgrades.
- Premium integrations and support.

Advantages:

- Preserves trust and adoption while creating a plausible business.
- Clear separation between developer engine and team operations.
- Self-hosted enterprise path remains available.

Risks:

- Open-core boundaries can become artificial or adversarial.
- Hosted economics may be poor if customers insist on fully local models.
- Team workflows require much more product design than the current repository contains.

This is the recommended option to validate.

### Option C — Closed commercial product

Best only if:

- There is already a committed design partner with a painful, funded problem.
- Hosted semantic memory produces a clearly superior result.
- The buyer accepts repository processing and model costs.
- Speed of proprietary workflow development matters more than ecosystem adoption.

This is not recommended from current evidence. The system has not yet demonstrated the
semantic advantage end to end, and its likely early adopters are precisely the users most likely
to demand local control and inspectability.

### Option D — Research/reference implementation

Best if the semantic result is scientifically interesting but product use remains weak.

The project could become the reference implementation for:

- Evidence-bound code assertions.
- Claim staleness semantics.
- Code-agent retrieval evaluation.
- Resolver precision measurement.
- Human-calibrated semantic memory.

This would still be a successful outcome. It should not be treated as a failed SaaS attempt.

---

## 5. Decision scorecard

Score each dimension from 0 to 3 after the Phase 2 and Phase 7 milestones.

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Semantic correctness lift | none/negative | anecdotal | measurable | large and repeated |
| User time saved | none | under 10% | 10–25% | over 25% |
| Unsupported-claim reduction | none | unclear | meaningful | decisive |
| Organic repeat usage | none | occasional | weekly | workflow-critical |
| Team sharing demand | none | requested | active pilots | budget attached |
| Willingness to pay | none | hypothetical | design partner | signed commitment |
| Hosted-data acceptance | refused | rare | mixed | common in target market |
| Operational burden | prohibitive | high | manageable | favorable |
| Competitive differentiation | commodity | feature | clear niche | category-defining |
| Maintainer commitment | low | part-time | sustained | funded team |

Interpretation:

- **0–10:** keep it a research/open-source project.
- **11–18:** open-source project with paid support or grants.
- **19–24:** validate open core with design partners.
- **25–30:** invest in a commercial team product.

Do not average away a zero in semantic correctness or willingness to pay. Either zero blocks a
commercial product decision regardless of the total.

---

## 6. Experiments before forming a company or hosted service

### Experiment 1 — Does semantic retrieval change outcomes?

Build the first two implementation phases and run four benchmark arms:

- Bare agent.
- CodeGraph.
- Code Learner structural-only.
- Code Learner with semantic retrieval.

Commercial go signal:

- At least a 15% relative improvement in correctness or a 20% reduction in time/cost on
  semantic task strata, reproduced across at least three repositories, without a higher
  unsupported-claim rate.

These thresholds are business judgments and should be frozen before the run.

### Experiment 2 — Do humans value the result?

Recruit 5–10 expert developers unfamiliar with selected repositories. Give them onboarding,
debugging, and impact tasks. Observe behavior rather than asking only for opinions.

Go signals:

- At least half voluntarily use semantic claims after the first task.
- At least half inspect citations and say the evidence changes their trust.
- At least three ask to run it on their own repositories.
- At least two want shared/team memory rather than only local search.

### Experiment 3 — Is there a buyer?

Interview engineering leaders in organizations with large, long-lived repositories. Avoid
generic “would you use AI documentation?” questions.

Ask about:

- Onboarding time.
- Architectural knowledge loss.
- Incidents caused by misunderstood invariants.
- Agent adoption barriers.
- Review and compliance requirements.
- Repository privacy constraints.
- Existing spend on code intelligence.

Go signal: three design partners willing to provide repositories and engineering time, with at
least one prepared to pay for a pilot or enterprise features.

### Experiment 4 — Can it operate economically?

Measure per repository:

- Initial indexing cost.
- Incremental update cost.
- Claim generation cost.
- Adjudication cost.
- Storage growth.
- Query latency.
- Human review time.

A system that must regenerate and judge every symbol after every edit will not support healthy
hosted economics. Incremental invalidation and selective regeneration are product requirements,
not optimizations.

---

## 7. Recommended positioning

Avoid positioning against CodeGraph as “the same thing plus semantics.” That invites comparison
on CodeGraph's strongest surface and understates the differentiated workflow.

Candidate positioning:

> Code Learner gives coding agents durable architectural memory. Every inferred purpose,
> invariant, and risk is linked to exact source evidence, independently reviewable, and
> automatically withheld when that evidence changes.

Short version:

> Accountable memory for coding agents.

The word “memory” captures persistence across sessions. “Accountable” captures evidence,
verdicts, and staleness. “Graph RAG” should remain an architectural description, not the main
customer promise.

---

## 8. Likely initial users

Prioritize:

1. Teams adopting coding agents in mature Python systems.
2. Infrastructure, financial, security, or distributed-systems teams with important invariants.
3. Consultancies and internal platform teams that repeatedly onboard into unfamiliar code.
4. Regulated organizations that need provenance for generated engineering knowledge.
5. Maintainers of large open-source Python projects who can provide public evaluation data.

Deprioritize initially:

- Small greenfield repositories.
- Polyglot monorepos requiring broad language coverage immediately.
- Users wanting only fast symbol lookup.
- Teams unwilling to generate, review, or maintain semantic knowledge.

---

## 9. Licensing and governance

Before actively recruiting contributors:

- Choose a permissive core license if adoption and integrations are the priority. Apache-2.0
  offers an explicit patent grant; MIT is simpler and aligns with CodeGraph's license.
- Add contribution guidelines and a developer certificate of origin or contributor agreement,
  depending on whether dual licensing is anticipated.
- Make benchmark data provenance and repository redistribution rights explicit.
- Separate model-generated fixtures from copyrighted source excerpts where necessary.
- Document telemetry as off by default.
- Publish a security policy and supported-version window.

If open core remains likely, define the boundary early around multi-user operations and policy,
not by removing essential local correctness or safety features from the open edition.

---

## 10. Twelve-month staged strategy

### Months 0–2: prove integration

- Complete source-rich and semantic retrieval.
- Ship a local developer preview.
- Finish the controlled benchmark.
- Recruit public-repository users.

Decision: if semantic lift is absent, remain a research/open-source project and focus on the
accountability primitives rather than building a company.

### Months 2–4: validate workflow

- Add adjudication policy and incremental refresh.
- Run supervised design-partner pilots.
- Learn whether users want reviewed shared memory, CI integration, or only better local search.

Decision: if repeat use exists but team demand does not, pursue open source plus support rather
than SaaS.

### Months 4–8: build the team layer only if demanded

- Review queue and ownership.
- Repository and policy administration.
- Pull-request stale-claim checks.
- Access controls and audit exports.
- Self-hosted deployment path.

Decision: require at least one paid pilot before committing to generalized enterprise
infrastructure.

### Months 8–12: commercialize the repeated pattern

- Harden the integrations actually used by pilots.
- Price against onboarding, review, or incident-cost savings rather than tokens indexed.
- Maintain the local engine as the adoption funnel and trust anchor.
- Add languages only when demanded by committed users and backed by precision fixtures.

---

## 11. Final decision rule

The project is already worth continuing as open source because its assertion provenance and
evaluation machinery are distinctive and useful research artifacts.

It is worth productizing only if the next integrated version proves all three:

1. **Outcome:** semantic retrieval materially improves correct task completion.
2. **Behavior:** developers repeatedly choose it and trust it because of the citations.
3. **Buyer:** teams will pay for shared governance, review, or operational integration.

Until those conditions are met, invest in the open local engine and the experiment—not in a
hosted control plane, broad language support, or a sales narrative.

