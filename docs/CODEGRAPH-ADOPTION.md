# What to take from codegraph, and what not to

A read of `@colbymchenry/codegraph` v1.5.0 (MIT), installed at
`~/.nvm/versions/node/v22.22.0/lib/node_modules/@colbymchenry/codegraph`, judged
against code-learner's thesis: **a claim about what code is FOR is only worth
storing if it cites hashed bytes, and only worth serving if those bytes still
hash.**

codegraph stores no LLM claims at all. That makes it a substrate project, not a
competitor, and it makes the adoption question narrow: *does this make our menus
better, and does it cost us the ability to hold a claim accountable?*

---

## What is actually readable

The npm package's `dist/` is **type declarations only**. The implementation lives
in the platform package,
`node_modules/@colbymchenry/codegraph-linux-x64/lib/dist/*.js` — 171 files of
`tsc` output with **every doc comment preserved**, including issue numbers and
the measurements behind individual constants. The source maps carry no
`sourcesContent`, but the compiled JS is close enough to the TypeScript to read
as source. Every anchor below is to that tree; paths are given relative to it.

The Rust kernel (`lib/kernel/codegraph-kernel.node`, 34 MB) is a binary. It does
tree-sitter extraction only. **Nothing in this document is inferred from the
binary** — resolution, ranking, response assembly, MCP surface, and sync are all
in the readable JS layer.

Two things worth knowing before the lists:

- **codegraph has no embeddings.** No vector store, no ANN index, no cosine
  anything. A `VectorError` class exists in `errors.js` and is never constructed.
  Retrieval is FTS5/BM25 → `LIKE` substring → bounded Levenshtein, plus roughly
  fifteen hand-tuned additive and multiplicative rescoring passes.
- **codegraph resolves ambiguity by picking a winner and burying the doubt.**
  `resolution/index.js:1028` is literally
  `candidates.reduce((best, curr) => curr.confidence > best.confidence ? curr : best)`.
  The confidence survives — but into a **JSON `metadata` blob**
  (`resolution/index.js:1062-1089`: `metadata: { confidence, resolvedBy, refName }`),
  not a column. The `edges` insert (`db/queries.js:1370`) is
  `(source, target, kind, metadata, line, col, provenance)`. Nothing in the query
  path, the ranking, the impact traversal or the MCP response reads that
  confidence, and there is **no threshold anywhere** — a 0.4 guess is persisted,
  traversed and rendered identically to a 0.95 import resolution. The losing
  candidates and the candidate *count* are discarded outright: nothing in the DB
  records that an edge was a 1-of-47 pick. Unresolved references are kept, in a
  separate `unresolved_refs` table with a `pending`/`failed` status and a retry
  path on the next sync (`db/migrations.js:37-39`, `resolution/index.js:1728`).

That second point matters for every judgement below, and it is the hinge of the
whole comparison. Our `edges` row holds the call site (`dst_name`, always
populated) and the binding (`dst_symbol_id`, `confidence`, `resolver`) together
as **queryable columns**, so an unresolved edge is honestly represented and a
low-confidence one is *filterable*. Their higher resolution rate is measured on
a store where the doubt is unreadable; ours is measured on a store that records
what it declined to bind and how sure it was about the rest.

---

# ADOPT

Ranked. Everything here strengthens the structural substrate without touching
the citation gate.

### A1 — Return verbatim, line-numbered source in the tool response

**What it is.** `codegraph_explore` returns one markdown document whose body is
the current on-disk bytes of the relevant symbols, prefixed `<n>\t<line>` — the
exact `cat -n` shape the Read tool emits — grouped by file, with elisions marked
`... (gap) ...`.

- `mcp/tools.js:290` `numberSourceLines()` — eight lines, the whole formatter.
- `mcp/tools.js:3041-3044` — the preamble that tells the agent this is a Read it
  has already performed.
- `mcp/tools.js:3379` — `GAP_MARKER`, deliberately language-neutral (no `//`,
  which is not a comment in Python).
- `mcp/tools.js:3063-3072` — the bytes are re-read **from disk on the call**, not
  served from the index.

**Why this is the first thing to do, and why it is not merely a token
optimisation.** Our `search_code` returns `qualname`, `path`, a line range,
score, modality and `content_hash`. It returns **no source**. The docstring on
`_symbol_hashes` (`codelearner/server/app.py:844-846`) states the ambition
exactly:

> "the retrieval → citation → gate loop only closes if the hash travels with the
> result."

The hash travels. **The bytes do not.** So the loop does not close: an agent
that wants to `submit_assertion` must go and Read the file, at which point the
bytes it is citing came from Read and the hash we handed it is an unchecked
coincidence it is asked to trust. Shipping the bytes alongside the hash makes
`submit_assertion` a pure function of what we served — *cite what you read*
becomes literally true instead of aspirational.

**Cost.** Small. `store.py:451` already builds a citation for
`path[byte_start:byte_end]` by hashing off disk, and `symbols` carries
`byte_start`/`byte_end`/`content_hash` per row. A first version is a `source`
field on each `search_code` hit and on `get_symbol.symbol`, sliced by byte span,
line-numbered, with a per-response character budget.

**What it displaces / risks.**

1. Response size. Our JSON grows from ~200 bytes per hit to several KB. This
   needs a budget (see D2) and it needs the truncation discipline in A2.
2. **A new secrets surface.** A response that ships file bytes can leak what a
   response of locations cannot. codegraph hit this and guards it —
   `mcp/tools.js:2686-2691` skips config-leaf nodes during rendering, and
   `mcp/tools.js:3814-3822` refuses to dump a config/data file at all, showing
   keys with values withheld. We index only Python, so the exposure is smaller
   (a `settings.py` with a literal, a `conftest.py` with a token), but it is not
   zero and it is a *new* risk we take on the day we adopt this.
3. Staleness. The bytes are true at serve time and stale a second later. Our
   answer is better than theirs (see R1) but the response must say which it is.

**Idea or implementation?** Idea, plus `numberSourceLines` which is trivial
enough to reimplement without borrowing. The *policy* — what to include, where
to elide — is the valuable part and is A2.

---

### A2 — Never cut through a symbol body; drop whole sections instead

**What it is.** codegraph's assembler will drop an entire file section rather
than emit half a method, at three separate places:

- `mcp/tools.js:3239-3245` — a whole-file render that does not fit is **skipped
  entirely**: *"Don't slice a whole file mid-method: … Half a file forces the
  Read this is meant to prevent."*
- `mcp/tools.js:3487-3491` — an oversize single cluster renders in **full**,
  because *"half a method is useless (the agent just Reads the rest for the
  other half), which is the very fallback explore exists to prevent."*
- `mcp/tools.js:3602-3613` — the final hard ceiling cuts at the last
  **file-section header** (`FILE_SECTION_PREFIX`, `**\``), falling back to a line
  boundary only in the degenerate single-giant-section case.

**Why it matters to us specifically.** For codegraph a truncated body costs a
round-trip. For us it costs *correctness of citation*: a claim generated from
half a function is a claim whose evidence span, if it is honest, covers bytes the
generator never saw the meaning of. Our chunker already refuses fixed-window
splits for exactly this reason (`schema.sql`, the `chunks` comment: *"the
embedding of half a function is not the embedding of anything a question would
ever be about"*). Extending that rule from the chunker to the *response* is the
same principle applied one layer out, and it is the discipline that makes A1
safe.

**Cost.** It is a rule, not code — perhaps 30 lines of budget accounting.

**Risk.** A pathological single symbol (a 900-line function) blows the budget.
codegraph bounds this by windowing an oversize *spine* method to ±28 lines around
its next-hop call site plus a signature head (`mcp/tools.js:3386-3403`) — a
principled exception, since a window around a call site is still a complete
statement even if it is not a complete body. We would need the same escape hatch,
and the response must mark it.

**Idea or implementation?** Idea.

---

### A3 — Rewrite the tool descriptions and server instructions as a behavioural playbook

**What it is.** `mcp/server-instructions.js:23-73` is a 50-line playbook, not an
API description. It has a "How to query" section mapping question shapes to
calls, an explicit "Anti-patterns" section, and a "Limitations" section. Every
tool description is written to *steer*, e.g. `codegraph_search`'s description ends
"Use codegraph_explore instead to get the actual source" (`mcp/tools.js:405`).

They also ship a second, smaller block into `CLAUDE.md`/`AGENTS.md`
(`installer/instructions-template.js:44-52`) and the header explains why with a
measurement (`:14-24`): subagents receive the project instructions file but **not**
the MCP `initialize` instructions, and *"measured on a forced-delegation flow
question … subagents loaded + used codegraph in ~1 of 9 runs without this block,
and consistently with it."*

**Why it is high-value here.** The pilot's 5/5 vs 1/4 tool-call gap is the
headline observation, and the tempting conclusion is "they have one tool, we have
five." Reading the source, that conclusion is wrong (see D3). They invested far
more in *description* than in *tool count*, and they measured the description's
effect in isolation. Our `INSTRUCTIONS` (`codelearner/server/app.py:189-210`) is
excellent prose about *epistemics* — tiers, hashes, refusal codes — and contains
**no guidance at all about when to call which tool**. It tells an agent what our
answers mean; it never tells it to ask.

**Cost.** Two hours. It is prose in one string constant and five docstrings.

**Risk / what it displaces.** The current INSTRUCTIONS text is load-bearing —
it is where the tier contract is stated. A playbook must be *added around* it,
not written over it. And one specific line of theirs must **not** be copied; see
R2.

**Idea or implementation?** Idea. Their text is MIT and could be adapted, but
almost every sentence in it is about a tool we do not have.

---

### A4 — First-class impact / blast radius, computed from `edges` today

**What it is.** Two things, and the smaller one is the better one.

*The tool*: `codegraph_impact` is a reverse BFS over incoming edges
(`graph/traversal.js:418-478`), depth-capped, that (a) **excludes `contains`**
when climbing — *"a container 'contains' its members but does not depend on
them, so following it upward would climb to the parent class and then re-expand
every sibling member"* (`:469-473`) — and (b) descends *into* container children
so callers of a class's methods appear in the class's impact (`:448-467`). The
output is a flat list grouped by file (`mcp/tools.js:4422-4444`). Multiple
distinct definitions of a name get **separate** blast radii rather than a merged
one (`mcp/tools.js:1565-1576`).

*The better one*: `buildBlastRadiusSection` (`mcp/tools.js:2255-2305`) is
**always-on and unrequested** — every explore response carries, for its entry
symbols, a one-line-per-symbol caller count, up to four caller *files*, the test
files among them, and — when there are none —
`⚠️ no covering tests found`. Locations only, no source. It skips symbols with no
dependents so a leaf query stays clean.

**Why adopt.** We have the call graph and no impact analysis at all. This is
producible from `edges` today with one recursive CTE, and *our version is better
grounded than theirs in two ways*:

- They detect test files by path regex (`search/query-utils.js:300-348`, which
  also sweeps in `sample|example|fixture|benchmark|demo` directories). We store
  `files.is_test` as a **tier-0 fact** derived at index time, and the schema
  already argues why that distinction is worth a column.
- Their impact traverses edges whose confidence was discarded. Ours can report
  the confidence and the resolver per hop, so "6 callers" can be
  "4 callers at confidence 1.0, 2 at 0.6 via the alias resolver."

**Cost.** Half a day for the always-on section attached to `get_symbol`; a day
if it becomes its own tool.

**What it displaces.** Nothing. It adds a section to an existing response.

**Risk.** The `⚠️ no covering tests found` flag is a claim about the world made
from an absence, and absences are where a resolver's misses show up as facts. It
must be phrased as "no *resolved* caller in a test file", not "untested".

**Idea or implementation?** Idea. The traversal is ~40 lines and we would write
it against our own schema.

---

### A5 — Announce dynamic-dispatch boundaries instead of guessing the edge

**What it is.** When the flow among an agent's named symbols does not connect,
codegraph scans **those symbols' bodies only, at query time** for dispatch sites
and reports the site rather than inventing an edge. `mcp/dynamic-boundaries.js:11-13`
states the position outright:

> "Guessing the missing edge was rejected (silent beats wrong — a wrong edge
> poisons the map and teaches abandonment). Instead, explore ANNOUNCES the
> boundary honestly: the exact site where the static path ends, the dispatch
> form, and — when a key is statically visible — that key."

Mechanically it is deterministic regex over comment- and string-blanked bodies
(`blankStringContents`, `:191-220`, blanks contents while preserving offsets so
snippets can be sliced from the *original*), with a table of nine forms
(`:40-146`) plus a hand-written Python `getattr` scanner (`:283-330`) that exists
because real `getattr` arguments span lines and nest calls in a way a regex
argument class cannot bound. Zero graph mutation; a connected flow never triggers
a scan; capped at 3 matches per body, 8 bodies, 200 KB scanned per call
(`mcp/tools.js:1995-1997`).

**This is philosophically our design, arrived at independently.** It is the same
move as `edges.dst_name` being populated whether or not resolution succeeded:
represent the limit of what you know as a first-class answer instead of dropping
it or filling it in. It is the honest half of the thing R3 rejects.

**The Python-relevant forms are the ones we would implement**, and they are few:
`getattr(obj, name)(...)` and its assigned variant, `importlib.import_module` /
`__import__`, the computed member call `handlers[key](...)`, and `.emit`/
`.dispatch`/`.publish` with a non-literal key. And **we can do this better than
they can**: they regex over stripped text because they have no AST at query time.
We already parse Python, so this is an `ast` walk with no false-positive class at
all — no comment stripping, no string blanking, no `MAX_GETATTR_ARGS` cap.

**Cost.** One to two days, done against `ast`. Cheaper than their version.

**What it displaces / risks.** It adds a section to `get_symbol`. The risk is
noise: fired on every symbol rather than only on a broken flow, `getattr` is
common enough in Python to become wallpaper. Their gating — *only* when the flow
the agent asked about failed to connect — is essential and is the part to copy.
We do not currently have a "flow" concept to gate on, so this may need to wait
for one, or gate on "this symbol has ≥1 unresolved call and you asked about it."

**Idea or implementation?** Idea. The Python `getattr` scanner is the only piece
whose logic transfers, and against an AST we would not use its logic.

---

# ADAPT

Things worth taking in changed form, where the change is the point.

### D1 — Guess the ambiguous bare name, at a low confidence we actually store

**This is the design that produces the resolution gap, and it is the most
important finding in this document.** It is not the framework resolvers and it
is emphatically not the dynamic-dispatch reporting.

**What it is.** codegraph runs about ten precise strategies first — file-path
match, qualified-name match, language-specific call-chain matchers, typed method
resolution — and then, when all of them decline, it **always guesses**.
`matchByExactName` (`resolution/name-matcher.js:367-416`):

```js
const candidates = applyLanguageGate(context.getNodesByName(ref.referenceName), ref)
    .filter((n) => n.kind !== 'import')
    .filter((n) => isLexicallyReachable(n, ref, context));
if (candidates.length === 1) { … confidence: 0.9 … }
if (candidates.length > AMBIGUOUS_NAME_CEILING) return null;   // 500
const bestMatch = findBestMatch(ref, candidates, context);
const proximity = computePathProximity(ref.filePath, bestMatch.filePath);
const confidence = proximity >= 30 ? 0.7 : 0.4;
```

`findBestMatch` (`name-matcher.js:1875-1932`) is a locality score, and its
components are the interesting part:

| Signal | Value |
|---|---|
| Same file as the reference | **+100** |
| Shared directory segments | +15 each, capped 80 |
| Same language / different language | +50 / **−80** |
| `calls` ref → function/method target | +25 |
| `instantiates` ref → class/struct target | +25 |
| Target is exported | +10 |
| Same file, line proximity | `max(0, 20 − |Δline|/10)` |

It is seeded at `bestScore = -1` and has **no decline branch**: with 2–500
candidates it always returns a winner. The 500 ceiling is documented as an
anti-noise and anti-O(K²) measure, not a correctness one — *"Real repos top out
near ~40 same-named methods, so a normal codebase never reaches this."*

**Why this is the answer to "what design produces their recall".** Our Phase 0
findings (`docs/PHASE0-FINDINGS.md`, §2) measured the alternative and named it:

> "The first resolver bound a name only when it was unique repo-wide. Result:
> **98 of 34,013 call edges (0.3%)**. … name exists but is **ambiguous** —
> 17,121, 50.3%."

Read those two rows together and the point sharpens: 49.4% of call sites name
something outside the repo entirely, so of the references that *could* be bound,
**essentially all of them are ambiguous** (17,121 of the ~17,219 in-repo). A
resolver that declines ambiguity is not declining an edge case; it is declining
the entire problem. codegraph binds it by scoring locality and committing; we
decline most of it.

**This is not an argument to abandon our approach — it is an argument for a
floor underneath it.** Phase 0's own conclusion was that *"import- and
scope-aware resolution isn't a refinement, it's the entire mechanism,"* and that
is right: a correct binding derived from an import and a scope is worth more than
a locality guess and should always win. What D1 proposes is a **last-resort band**
that runs only where scope-aware resolution has already declined, at a confidence
that says so. Their mistake is not that they guess; it is that their guess is
indistinguishable from their knowledge.
That difference, not the synthesisers, is where the gap lives — and for a
**Python-only** repo it is *nearly the whole story*, because two Python-specific
facts amplify it:

- **`self.foo()` loses its receiver at extraction.**
  `extraction/tree-sitter.js:4366-4374` has
  `SKIP_RECEIVERS = new Set(['self','this','cls','super'])`, and for those the
  callee name becomes the **bare method name**. So `self.foo()` is
  indistinguishable from a module-level `foo()` by the time the resolver sees
  it, and it lands in `matchByExactName` — where the **+100 same-file bonus**
  usually gets it right, because the class body is in the caller's file. There
  is no class-scope check at all: two classes defining `foo` in one file are
  separated only by the line-proximity tiebreak.
- **`obj.bar()` binds on a unique method name with no receiver validation.**
  `name-matcher.js:1683-1690`: if exactly one same-language `method` named `bar`
  exists in the repo, the edge is emitted at **0.7** — the receiver name is not
  consulted on that branch. With several, it needs a camelCase word of the
  receiver to overlap the class name (`score >= 2`) and otherwise declines.

The genuinely precise Python machinery is thin: two regexes for local receiver
types (`name-matcher.js:1103-1107` — `x = Logger(...)` and PEP 526 `x: Logger`),
scanned backward to the enclosing `def`, and **validated** against the inferred
type before emitting at 0.9 (`resolveMethodOnType`, `name-matcher.js:501`;
*"a mis-inference produces no edge"*). That part is excellent and narrow. It does
not carry the number. The bare-name guess does.

**Why ADAPT, and why the adaptation is the entire value.** Their version puts a
0.4 guess and a 0.95 import resolution in the same table, in the same shape,
with the deciding number in a JSON blob nothing reads. An agent cannot tell them
apart; neither can their own impact traversal. **We have exactly the columns
that fix this**, and they were designed for this case — `schema.sql` on `edges`:

> "seeing the call site `foo()` is a tier-0 fact, but deciding *which* `foo` it
> refers to is a tier-1 resolution that can be wrong. Both live in one row …
> so an unresolved edge is honestly represented rather than dropped or guessed."

A locality-scored best-match resolver landing as
`resolver='locality-bestmatch', confidence=0.4` is not a betrayal of that
sentence — it is the case the sentence was written for. It raises recall,
it is filterable at query time, it is recallable if the resolver turns out to be
bad, and `facts_only` already has the seam to exclude it. The *dishonest* version
is theirs: bind at 0.4 and render it as a fact.

**Concretely, three things worth trying, in order:**

1. **Same-file preference for the bare name.** The +100 bonus is the single
   highest-yield component and is nearly free. In Python it mostly recovers
   `self.foo()` where `foo` is on the same class — which we currently decline.
2. **Directory proximity as the tiebreak** beyond the same file, at a lower
   confidence band.
3. **The unique-method-name rule for `obj.bar()`**, at a confidence low enough
   to be honest about the missing receiver check — this is the one I would gate
   hardest, because it binds with zero evidence about the receiver.

**Cost.** Days, not hours, and the cost is mostly *measurement* — we have gold
sets and a resolution-rate metric, so each band can be scored rather than
asserted. **Risk / displacement.** A wrong tier-1 edge is not free: graph
expansion traverses only resolved edges (`retrieve/graph.py`), so a bad binding
becomes a bad retrieval vote, and our own weight sweep shows the graph modality
is already the fragile one. Every band must be measured against retrieval
quality, not just against the resolution percentage.

**Idea or implementation?** Idea. `findBestMatch`'s weight table is worth reading
as a starting hypothesis and nothing more — its constants are tuned for 20
languages, and ours would be tuned for one, against gold sets they do not have.

**Note the boundary with the sibling investigation.** Whether their extra edges
are *correct* is being measured separately. Nothing above claims they are. What
is claimed is only what the source shows: the mechanism is an ungated
locality-scored guess, and it is available to us in an accountable form that is
not available to them.

---

### D2 — An output budget tiered by repo size, with a hard ceiling below the host's inline cap

**What it is.** `getExploreOutputBudget` (`mcp/tools.js:132-230`) returns a
different budget object per project size (<150, <500, <5000, <15000, else
files): `maxOutputChars`, `defaultMaxFiles`, `maxCharsPerFile`, `gapThreshold`,
and boolean feature flags that turn *sections of the response off* on small
repos. The comments record the iterations, including reversals (`ITER3: revert
iter2's aggressive body shrink (forced Read fallback…)`).

**The one genuinely non-obvious empirical finding in the whole package** is at
`:136-143` and `:3591-3600`:

> "it MUST stay under the agent's INLINE tool-result cap (~25K chars). Above
> that, the host externalizes the result to a file the agent then Reads back —
> re-introducing a read AND the cache-write cost — which is exactly what a 35K
> vscode explore did in the n=4 A/B."

A response that is *too helpful* gets spilled to a file and re-read, converting
your saving into a loss. Every tier therefore caps at ~24K with an absolute
25K hard stop, and a bigger repo gets **more calls** (`getExploreBudget`,
`:121-131`) rather than a bigger response.

**Why adapt rather than adopt.** The 25K threshold is a property of the *host*,
not of codegraph, so it applies to us identically the moment we ship source (A1).
Take the ceiling. Do **not** take their tier boundaries or their per-tier
constants — those are tuned against their response shape on their repos, and our
response carries hashes and tier labels theirs does not. Tier on symbol count,
and measure our own numbers with the bench harness that already exists.

**Cost.** Trivial to add, expensive to *tune honestly* — and the harness for
tuning it already exists in `bench/`.

**Risk.** Turning sections off by repo size is how a response quietly stops
carrying the tier labels or the `notes` array on small repos. Any budget we add
must treat the accountability fields as non-negotiable and only trim source.

---

### D3 — Shrink the *listed* tool surface, not the implemented one

**What it is.** codegraph implements eight MCP tools and lists **one**.
`DEFAULT_MCP_TOOLS = new Set(['explore'])` (`mcp/tools.js:669`), and the comment
(`:658-668`) gives the reasoning:

> "Pared to ONLY `codegraph_explore` — the single tool that reliably earns its
> place… Every other tool is a narrower slice of what explore already does, and
> **presence itself steers mis-picks**, so they are no longer LISTED to agents.
> The other defined tools … remain fully functional — handlers stay, the library
> API and CLI are untouched, and `CODEGRAPH_MCP_TOOLS=explore,node,…` re-enables
> any of them."

**So the answer to "is a single wide tool better than five narrow ones" is: that
is not the experiment they ran.** They did not collapse five tools into one. They
built one tool that subsumes the others and then *hid* the others behind an env
var, keeping every handler, the CLI, and the library API. The claim is about
**menu length as a source of mis-picks**, and it is cheap and reversible.

**What ours would collapse into — and what would be lost.**

We cannot go to one tool, and not for a small reason: **`submit_assertion` is a
write.** codegraph is read-only end to end and advertises it
(`READ_ONLY_ANNOTATIONS`, `mcp/tools.js:387-392`, which some clients gate on).
Our surface has two verbs, and folding a write into a read tool would be
indefensible.

Of the four read tools:

- `search_code` and `get_symbol` **are** collapsible in codegraph's sense — a
  wide "explore" that takes a query or a qualname and returns hits *with source,
  callers, callees, unresolved calls and servable assertions* would subsume
  both. That is A1 + A4 landing in one place.
- `get_symbol` **must survive as a name**, because it is the only surface on
  which tier-2 content appears at all, and its `facts_only` flag is the only
  place that flag changes an answer (`app.py:1362-1368`, `tier.py:73-88`).
  Burying the T2 surface inside a general-purpose tool would make the project's
  central feature something an agent reaches by accident.
- `index_stats` and `reading_path` are the two that should stop being *listed*.
  `index_stats` is diagnostics (codegraph's own `codegraph_status` description
  is "Skip unless debugging", `mcp/tools.js:569`). `reading_path` is a genuinely
  good tool that answers a question agents rarely ask unprompted.

**Recommended adaptation:** keep all five implemented and CLI-reachable; list
three by default (`explore`-shaped search, `get_symbol`, `submit_assertion`);
gate `reading_path` and `index_stats` behind an env var exactly as they do. This
is reversible in one line and measurable with the existing harness.

**Cost.** An afternoon. **Risk:** `reading_path` is arguably our most
differentiated read tool and hiding it may be the wrong trade — which is exactly
why it should be behind a flag we can flip during a bench run rather than a
deletion.

---

### D4 — Use graph connectivity to *gate* results, not to generate candidates

**What it is.** `computeGraphRelevance` (`mcp/tools.js:2306-2387`) is
Random-Walk-with-Restart (personalised PageRank) from the query's matched seeds,
**undirected**, α=0.25, 25 power iterations, bounded to the already-gathered
subgraph. Its docstring:

> "This is the ranking signal text search (FTS/bm25) CANNOT provide… A file whose
> symbols are call-connected to the matched cluster accrues walk mass and ranks
> high; a lone TEXT match — e.g. `LensSwitcher.swift` matched the word 'switch'
> from `switchOrganization`, but calls none of `setUser`/`fetchUser` — gets only
> its own restart probability and ranks ~0."

The crucial detail is **where it is applied**: not as a retriever, but (a) to
score files after gathering (`:2773-2777`) and (b) as a **gate** (`:2843-2851`) —
a file survives only if its graph mass is ≥6% of the top, or it is central, or it
defines a named symbol, or it matches ≥2 distinct query terms. Guarded so it
never prunes below two files.

**Why this is worth reading against our own measurement.** We reached the same
conclusion from the opposite direction and wrote it down.
`retrieve/fuse.py:DEFAULT_WEIGHTS` records a sweep: graph weight 0.3 → recall@5
0.646, 0.6 → 0.615, 1.0 → 0.552, 1.5 → 0.354, and at the original unweighted
default the whole hybrid scored 0.385 against 0.573 for lexical+dense alone. The
diagnosis in that comment is exact:

> "graph expansion has no query representation. It contributes symbols that text
> retrieval missed, which raises recall … but every vote it casts is evidence
> about the CODE rather than about the QUESTION."

codegraph's design is that diagnosis taken to its conclusion: put the graph
signal where it *cannot* cast a vote about the question — after retrieval, as a
re-rank and a gate. And `rerank.py`'s finding (with the cross-encoder on, turning
graph off scores identically on all four metrics) says our graph modality is
currently earning nothing at the top of the list.

**What to adapt.** Not the RWR — we have spreading activation already
(`retrieve/graph.py`), and ours is *directed and asymmetrically weighted*
(`OUT_WEIGHT` 1.0 vs `IN_WEIGHT` 0.35), which is better reasoned than their
undirected adjacency; their own comment for why they went undirected is only
"reachable either direction." What to adapt is **the gate**: after fusion, drop
a hit whose structural connection to the seed cluster is ~0 *and* whose lexical
support is a single term. That is a precision lever we do not have, aimed at
exactly the failure our own numbers show.

**Cost.** Small — the activation scores already exist in `expand()`.
**Risk.** Our reranker may already be doing this job; the honest expectation is
that this buys little on top of a cross-encoder, and it should be run as an
ablation before it is shipped. Do not do this before A1–A4.

---

### D5 — Narrow, declared framework resolvers for Python — with our confidence column

**What it is.** `resolution/frameworks/python.js` handles Django/Flask/FastAPI. It
is gated by a `detect()` that reads `requirements.txt`/`setup.py`/`pyproject.toml`
for "django" or checks for `manage.py` (`:12-24`). Its useful half is `extract()`
(`:63-133`), which regexes route registrations out of source and synthesises
`route` nodes with an edge to the handler:

- `path('url', HandlerView.as_view())` / `re_path` / `url` → route → handler.
- DRF `router.register(r'articles', ArticleViewSet)` → route → the ViewSet class,
  gated on a *string* first argument (to exclude `admin.site.register(Model, Admin)`)
  and a `View`/`ViewSet` suffix.
- `include('app.urls')` → the target `urls.py` file.

These are real edges a naive AST walk misses, and they are exactly the shape of
edge that makes a Django codebase navigable at all — a URLconf is the entry point
and nothing calls it.

**Why ADAPT and not ADOPT.** The same file's `resolve()` (`:25-53`) contains this:

```js
if (ref.referenceName.endsWith('Model') || /^[A-Z][a-z]+$/.test(ref.referenceName)) {
    const result = resolveByNameAndKind(ref.referenceName, CLASS_KINDS, MODEL_DIRS, context);
    if (result) return { ..., confidence: 0.8, resolvedBy: 'framework' };
}
```

*Any single-word PascalCase reference* — `User`, `Order`, `Cart`, `Item` — gets
bound to a class found in a `models/` directory at confidence 0.8. The tie-break
inside `resolveByNameAndKind` (`:384-399`) is `return kindFiltered[0].id`:
**first-indexed wins, no ambiguity check, no locality check.** And because the
framework strategy runs first and 0.8 beats the name-matcher's proximity-correct
0.7, on a Django project this rule *outranks* the one signal that would have got
it right. `_iterable_class` is separately hard-wired to `ModelIterable.__iter__`
(`:47-51`, `:137-149`), with the over-approximation admitted in the comment.
These are conventions asserted as facts, at a confidence nothing downstream
reads (see the preamble).

**The adaptation is the whole point.** We have the columns they lack:
`edges.confidence` and `edges.resolver`. A route-registration resolver we write
lands as `resolver='django-urlconf', confidence=0.9` — recallable, auditable, and
excludable by a caller who wants only what the parser saw. Their design cannot
express the difference between "the AST said so" and "Django convention says so";
our schema was built to.

**Cost.** Real — a day per framework, plus a detect step, plus the ongoing
maintenance surface. **What it displaces:** engineering time that A1–A4 and D1
want first. **Recommendation:** not now, and probably not at all unless we index
a Django repo we care about. The `extract()` half is worth ~200 lines if we do;
the `resolve()` half is a worked example of what D1 must not become — a guess
whose confidence outranks a better signal and which nothing downstream can
filter.

---

# REJECT

Each with the specific reason. "We didn't need it" is not on this list.

### R1 — File watchers and auto-sync

**What it is.** `sync/watcher.js`: recursive `fs.watch` on macOS/Windows,
per-directory inotify on Linux, adaptive debounce (300 ms for ≤2 pending files,
2000 ms for a burst), file-level re-parse gated by stat then content hash, a
scoped fast path for ≤500 pending files, exponential backoff on failure, and a
one-way `degrade()` latch. It is competently built.

**The reason to reject is not cost. It is that the watcher does not deliver the
guarantee, and our serve-time check does.**

codegraph's own response format admits this. Every read tool response can carry
`formatStaleBanner` (`mcp/tools.js:325-337`) — *"⚠️ Some files referenced below
were edited since the last index sync… For accurate content of those specific
files, Read them directly"* — and a second, rarer
`formatDegradedBanner` (`:363-368`) for when watching has stopped entirely and
`getPendingFiles()` has gone empty so the first banner *cannot fire on a frozen
index*. There is a third mechanism, `catchUpSync` (`mcp/engine.js:298-326`), a
filesystem-vs-DB reconcile gating the first tool call, which exists because the
watcher populates `pendingFiles` and *catch-up does not*, so a call racing past
it "returns rows for files that no longer exist on disk." And a fourth, a
`needsFullScan` latch for deleted directories, whose child deletions may never
produce their own events.

Four mechanisms, three of them compensating for the first. The composite
guarantee is *file-level*, *best-effort*, and *advisory*: it tells an agent which
files might be wrong and asks it to go Read them.

Ours is **span-level, mandatory, and verified at the moment of use.** An
assertion is re-hashed against `evidence_spans.content_hash` before it is served
(`schema.sql`, `assertions/stale.py`), with a stat()-based fast path
(`span_verifications`) so the steady state is one stat per cited file per query
and zero reads. The schema comment for `evidence_spans.content_hash` explains
why per-span beats per-file:

> "an unrelated edit elsewhere in a 2,000-line module must not expire a claim
> about one function in it, or staleness becomes noise and the first thing anyone
> does with noise is stop reading it."

That is precisely the failure mode of a per-file staleness banner.

**And there is an active cost, not just a redundancy.** A watcher is a background
*writer* against the one SQLite file whose atomicity the entire gate depends on.
Our server already treats a mid-session rebuild as a first-class refusal —
`index_replaced`, raised "before writing anything if the rebuild is already
visible, after writing if it lands mid-call" (`app.py:1416-1420`). Adding a
process that rewrites `symbols` rows on a 300 ms timer turns that rare, honest
refusal into a routine event, and re-indexing replaces symbol rows wholesale,
which is the exact scenario `assertions.subject_symbol_id`'s `ON DELETE SET NULL`
was written to survive. We would be manufacturing the condition our schema was
carefully designed to tolerate.

**Reject. What we should take instead is one small piece of it** — see the note
under "smaller things" below.

### R2 — "Trust codegraph's results — don't re-verify them with grep"

**What it is.** A single line in `mcp/server-instructions.js:62`, under
"Anti-patterns":

> "**Trust codegraph's results — don't re-verify them with grep.** They come from
> a full AST parse; re-checking with grep is slower, less accurate, and wastes
> context."

**Why this specifically must be named as a reject, given that A3 recommends
copying the *form* of this document.** It is a direct instruction to an agent to
disable its own verification. For codegraph that is defensible-ish: their output
is parsed structure and verbatim bytes, and both are checkable in principle. For
us it would be self-refuting. code-learner's product is that a claim can be
checked, and the INSTRUCTIONS we already ship
(`app.py:198-203`) say the opposite — *"Cite what you actually read"*.

The version we want is the inverse and is stronger: **you do not need to re-read
what we served, because we served the bytes and their hash — and you should
re-check anything we did not.** A playbook adopted wholesale would import this
line by default. Adopt the structure, drop this bullet, and replace it with the
tier guidance the structure has no slot for.

### R3 — Synthesised ("heuristic") edges stored in the graph

**What it is.** codegraph writes edges that correspond to no literal call site,
tagged `provenance: 'heuristic'` and carrying a `synthesizedBy` metadata key. The
catalogue is legible from `synthEdgeNote` (`mcp/tools.js:1584-1668`): `callback`,
`event-emitter`, `react-render` (a `setState` → `render()` edge), `jsx-render`,
`vue-handler`, `interface-impl`, `closure-collection`, `fn-pointer-dispatch`,
`goframe-route`, plus a generic fallback for "redux-thunk,
gin-middleware-chain, flutter-build, …". Dedicated synthesiser modules exist
(`resolution/callback-synthesizer.js`, `c-fnptr-synthesizer.js`,
`goframe-synthesizer.js`), and the README describes Expo modules as producing
"synthetic method nodes [that] resolve via existing name-match."

**Why reject, in the form they ship it.** A synthesised edge is an *inference
about control flow* with no citation, stored in the same table as parsed facts,
distinguished only by a nullable `provenance` string, with the confidence that
produced it discarded. That is precisely tier-2 content wearing tier-0 clothes.
Our schema has no honest place for it: `edges.dst_name` is documented as *"The
name as written at the call/import/base-class site. ALWAYS populated — this is
the tier-0 fact and it never depends on resolution succeeding."* A synthesised
edge has no call site, so `dst_name` would have to be a fiction, and the one
column the whole tier model rests on would start carrying invented values.

**And rejecting it costs us almost no recall, which I did not expect.** The
synthesiser registry is 38 passes (`resolution/callback-synthesizer.js:3784-3844`),
and each carries a language gate. **For a Python-only repo exactly one fires**:
`celeryEdges`, `gate: (has) => has('python')` (`:3827`), which invents
`enclosing-fn --calls--> task-fn` for `X.delay(...)` / `X.apply_async(...)` where
`X` resolves to a function decorated `@shared_task`/`@task`/`@app.task`. The
highest-volume shapes are gated *away* from Python — `interfaceOverrideEdges`
(base method "calls" every override) lists java, kotlin, csharp, swift, scala,
go, rust, arkts and the JS family, and not Python (`:3800`); `closureCollEdges`
is hard-gated to Swift and Kotlin (`:52`); the `ALWAYS`-gated passes are
JS/Swift-shaped and produce ~0 on Python source.

So the honest statement is: for our target language this is a clean reject with
no recall left on the table. The design that actually produces their number is
D1, and it is a guess, not a synthesis. Were we ever to want the celery edge, the
**accountable form exists and is not this one:** a `.delay()` call site *is*
bytes on disk, so the hop is expressible as a tier-2 assertion of a new `kind`
(`"dispatch"`) whose `evidence_spans` cite the dispatch line and the decorator —
hashed, expiring, refusable by a judge. What is rejected here is the storage
decision, not the information.

### R4 — The 20-language Rust kernel and the framework resolver zoo

**What it is.** `extraction/kernel/` plus 26 language modules and ~25
`resolution/frameworks/*` modules (Astro, Cargo workspaces, CICS, Drupal, Expo,
Express, Fabric, GoFrame, Laravel, NestJS, Play, React Native, Svelte, Swift/ObjC
bridging, Terraform…).

**The specific reason.** This is not a design insight; it is a body of work whose
size is proportional to the number of framework ecosystems in the world, and its
maintenance cost recurs every time one of them changes a convention. We index one
language. The transferable content is D5 and it is three route shapes.

There is a second, sharper reason to leave it: their per-language resolvers
increasingly encode *negative* rules — Nix cross-file name matches are refused
outright, PHP include paths must not fall back to the name matcher (`#660`:
"a wrong edge is worse than none"), Terraform is directory-scoped by language
semantics. Those refusals are the accumulated scar tissue of a name-matching
resolver applied across 20 languages. A Python-only resolver does not accrue that
debt and should not import the architecture that requires it.

### R5 — Their query planning, as a system

**What it is.** `context/index.js:433-905` plus `db/queries.js:967-1049` and
`search/query-utils.js`. Pure lexical, with roughly fifteen stacked rescoring
passes: +80 exact name, +60 exact token, +15 all-camel-terms, kind bonus by
symbol kind, path-relevance +10/+5/+3, −15 test penalty then ×0.3 test
multiplier, +20 per co-located query symbol, +25 "core directory" boost when one
file holds ≥3× the next file's intra-file edge count, a brevity bonus favouring
shorter class names, multi-term multiplicative boost ×(1 + 0.5·terms), and a ×0.3
demotion for exact matches on common words. Three separate hand-rolled
tokenizers with **two different, disagreeing stopword lists**
(`query-utils.js:102-120` vs `context/index.js:118-144`).

**The specific reason to reject.** Not that it is ugly — much of it is
well-reasoned, and every constant has an anecdote attached. It is that **the
anecdotes are the evaluation.** Each boost is justified by one repo and one
query (sinatra's `base.rb`, elasticsearch's `TransportSearchAction`, cosmos's
`expected_keepers_mocks.go`). There is no held-out set, no reported metric, no
ablation. We have 170 hand gold queries across three repos, a measured RRF
weight sweep, and a reranker whose contribution was measured and whose
end-to-end number was **thrown away rather than reported** when the GPU was
contended (`retrieve/search.py:86-90`). Importing a tuned-on-anecdote scorer into
a measured pipeline would trade a number we can defend for a number we cannot.

The individual *ideas* worth noting are already ours in stronger form:
co-location and multi-term corroboration are what RRF's consensus-across-modalities
computes from first principles, and their brevity bonus is a proxy for the
implementation-over-test preference we measured directly
(`prefer_implementation`, nDCG@10 +0.181).

The one exception is D4, which is not about how they compute a signal but where
they apply it.

---

## Smaller things, noted without a section

- **The catch-up gate** (`mcp/engine.js:298-326`, `mcp/tools.js:741-770`): a
  one-shot filesystem-vs-index reconcile whose promise gates the *first* tool
  call, time-boxed to 3 s so a huge repo cannot hang it. This is the one piece of
  R1 worth having, and it is not a watcher — it is a "the world may have moved
  since we last looked" check at session start. We have drift detection; wiring
  it to gate the first call is cheap and honest.
- **Their staleness bias**, stated at `sync/watcher.js:772-775`: *"We prefer
  false positives ('shown stale, actually fresh' → at worst one extra Read) over
  false negatives ('shown fresh, actually stale' → misleads the agent)."* That is
  our `facts_only` fail-closed rule (`tier.py:66-70`) in different words.
  Agreement worth recording.
- **Refusals returned as success-shaped results, not errors**
  (`mcp/tools.js:30-42`): *"an `isError: true` early in a session teaches the
  agent the toolset is broken and it stops calling codegraph entirely (observed
  repeatedly)."* We reached the same conclusion — `app.py:5-10`, *"A traceback
  crossing an MCP boundary tells the agent that the tool is broken, which is the
  one conclusion that stops it trying again."* Independent confirmation of a rule
  we already hold.

---

## What we have that they don't — judged coldly

**Genuinely ours, and load-bearing:**

- **The unresolved reference and its binding in one row.** `edges.dst_name` is
  always populated; `get_symbol` returns `unresolved_calls` as tier-0
  (`app.py:1036-1042`). codegraph files unresolved references in a *different
  table* with a `pending`/`failed` status. Ours is the reason a resolution-rate
  comparison between the two systems is auditable at all, and it is a better
  design, not just a different one.
- **`confidence` and `resolver` as queryable columns.** They compute both and
  bury them in a JSON blob no code path reads. Ours makes "find every edge a bad
  resolver bound" a query rather than a re-index — the same recall property
  `assertions.generator` gives us for claims. **This is also the thing that makes
  D1 available to us and not to them:** we can raise recall by guessing, because
  we can say how much of a guess it was, and a caller can decline it. That is the
  clearest case in this whole study of the accountability apparatus paying for
  itself in capability rather than costing us any.
- **The assertion store, the verdicts table, and the retained rejections.**
  Nothing in codegraph is remotely comparable. The decision to keep rejected and
  stale assertions rather than delete them is the only thing that makes a pass
  rate mean anything.
- **`files.is_test` as a parsed fact** vs their path regex, which sweeps in
  `sample|example|fixture|benchmark|demo`.
- **1,014 tests and a negative-control apparatus.** No visible equivalent.

**Cold verdicts, where the honest answer is "less than it looks":**

- **The `content_hash` on every hit buys an agent nothing today.** It is a token
  it cannot act on without reading the file itself — at which point it could hash
  the bytes and never needed ours. The retrieval → citation loop is *half built*,
  and A1 is the missing half. Until then this is architecture, not a feature.
- **`facts_only` on `search_code` is admitted inert**, in the code and in the
  docstring (`tier.py:73-88`, `app.py:1333-1339`). It is honestly labelled, which
  is to its credit, but an agent gains nothing from passing it. On `get_symbol`
  it does real work — and `get_symbol` is the tool the pilot agent called least.
- **The tier model buys a code-writing agent very little.** An agent fixing a bug
  does not care whether the edge that led it to a function was T0 or T1; it cares
  whether the function is the right one. The tier model pays off for a *second*
  agent — one auditing the first, or one deciding whether a stored claim is
  admissible. That is a real and valuable audience, and it is not the audience
  the bench pilot measured. Worth saying plainly rather than assuming the tiers
  explain the tool-call gap. They do not; A1 and A3 do. The one place the tier
  model earns its keep for a *first* agent is D1 — and it has not been cashed in
  yet.
- **`reading_path` is differentiated and unasked-for.** Its output is good. No
  agent in the pilot reached for it. That is a description problem (A3) before it
  is a design problem, but it may also be that agents do not want a tour.

---

## What I would do first

1. **A1 + A2 — verbatim line-numbered source, with whole-section truncation.**
   ~1–2 days. Highest value on both axes at once: it is the cheapest fix for the
   tool-call gap *and* it is the thing that finally closes the retrieval →
   citation → gate loop the code already claims to have closed. Needs the
   config-value guard from A1's risk list on day one, not later.
2. **A3 — rewrite INSTRUCTIONS and the five tool docstrings as a playbook.**
   ~2 hours, no code, no test churn. It is the single highest ratio of measured
   effect to effort in this whole document, and codegraph measured it in
   isolation (`installer/instructions-template.js:14-24`). Explicitly excluding
   R2's line.
3. **A4 — blast radius attached to `get_symbol`.** ~half a day. Produced from
   `edges` with one recursive CTE, and better grounded than theirs because
   `files.is_test` is a fact and `edges.confidence` exists.

Then **D1**, which is the largest single item in this document and the only one
that moves the number the benchmark measured. It is deliberately fourth rather
than first: it is days of work gated on measurement, it touches
`codelearner/ingest/**` where a sibling is currently working, and every band of
it must be scored against retrieval quality as well as resolution rate. A1–A3
are cheap, independent, and unblock the bench harness that D1 needs.

A5 next if there is appetite; D4 only as an ablation, after the bench harness has
re-measured with A1 in place.

---

## What I expected to be worth taking, and wasn't

This is the most useful part of the read.

- **The one-tool design.** I went in expecting the lesson to be "one wide tool
  beats five narrow ones." It isn't, because that is not what they built. They
  implement eight tools and *list* one; every handler stays, the CLI stays, the
  library API stays, and `CODEGRAPH_MCP_TOOLS` re-enables any of them
  (`mcp/tools.js:650-669`). The claim in the comment is narrower and more
  interesting than the architecture suggested — *"presence itself steers
  mis-picks"* — and it is a **listing** decision, cheap and reversible. Meanwhile
  they invested far more effort in the *description* (a 50-line playbook, a
  second block written into `CLAUDE.md`, and a measurement of that block's effect
  on subagent uptake) than in the tool count. The advantage is mostly the
  description. And we cannot collapse to one tool regardless: `submit_assertion`
  is a write, and codegraph is read-only end to end.

- **Auto-sync.** This looked like a clear gap — they have watchers, we have
  detection. Reading the source inverts it. The watcher needs *three additional
  mechanisms* to be safe (a per-file staleness banner, a whole-index frozen
  banner for when the first banner cannot fire, and a catch-up reconcile gate
  because the watcher does not populate what catch-up changes), and the composite
  guarantee it delivers is file-level and advisory: *go Read these files
  yourself*. Our serve-time re-hash is span-level and mandatory. We are not
  behind here; we are ahead, and adding their design would put a background
  writer against the file our gate depends on.

- **Dynamic-dispatch boundaries as the source of their recall advantage.** The
  framing "hops grep can't follow" made this sound like the resolution mechanism.
  It is not, and the source says so outright: it runs **at query time**, scans
  only the bodies of symbols the agent named, and **mutates the graph zero
  times** — *"Guessing the missing edge was rejected"*
  (`mcp/dynamic-boundaries.js:11-13`). It cannot contribute a single edge to a
  resolution-rate measurement. It is a *reporting* feature, and a good one (A5),
  but the recall gap is somewhere else entirely — see D1.

- **Their ranking.** I expected to find a retriever worth learning from. There
  are **no embeddings at all** — a `VectorError` class that is never constructed
  is the only trace of the idea. What exists is BM25 plus fifteen hand-tuned
  boosts, three tokenizers, and two stopword lists that disagree with each other,
  every constant justified by a single repo and a single query, with no held-out
  set anywhere. Our RRF + dense + cross-encoder is the more principled system by
  a wide margin, and it is *measured*. The one thing worth taking is not how they
  compute a graph signal but **where they put it** (D4) — as a post-retrieval
  gate rather than a candidate-generating modality, which is exactly the
  conclusion our own weight sweep argues for from the other side.

- **Confidence handling.** I assumed a system resolving 81% of references on
  swarm-sync must be modelling ambiguity carefully. It computes a confidence per
  candidate, takes the max, writes it into a JSON `metadata` blob, and then
  **never reads it again** — no column, no index, no threshold, no filter, and
  no record of how many candidates it beat. They do not reason about ambiguity
  better than we do. They reason about it less, and commit harder.

- **And the biggest one: I expected the recall gap to be a mechanism we could
  not have.** A Rust kernel, twenty languages, framework conventions, synthesised
  dispatch edges. It is none of those. For Python it is a locality-scored guess
  at an ambiguous bare name (D1), plus a receiver that gets *thrown away at
  extraction* so `self.foo()` arrives as bare `foo` and is rescued by a +100
  same-file bonus. That is a strategy we declined on principle in Phase 0 —
  correctly, since uniqueness-only bound 0.3% — and it is one our schema can
  express *more honestly than theirs can*, because we have a confidence column
  and they have a JSON blob. The gap is closable, and closing it costs the
  accountability thesis nothing.

---

## Licence note

codegraph is MIT, so code could be borrowed with attribution. **For every item
above the recommendation is an idea, not an implementation.** The two smallest
exceptions — `numberSourceLines` (8 lines) and the reverse-BFS impact traversal
(~40 lines) — are short enough that reimplementing them against our own schema is
cheaper than carrying an attribution and a TypeScript-to-Python translation. No
codegraph code is proposed for inclusion.

---

*This document changes no code — it is the only file this study wrote, and it is
Markdown. Baseline verified at HEAD `b093f55` before writing: `pytest tests/ -q`
passed and `ruff check .` reported "All checks passed!". Re-run afterwards, the
working tree had picked up concurrent sibling edits to `codelearner/server/`,
`codelearner/ingest/` and `tests/`, and `ruff` reports one F822 in
`codelearner/server/app.py` from that in-flight work. Nothing here touched it.*
