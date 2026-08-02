"""Retrieval gold mined from commit prose, with its selection bias measured and shipped.

`ablation` scores retrieval against `gold/swarm_sync.json`: 16 hand-labelled queries,
11 of them carrying exactly one relevant symbol. That set cannot resolve the
comparisons it is asked to make -- one query flipping moves `hit@5` by 6.25 points,
and with 11 single-relevant queries `recall@k` and `hit@k` are very nearly the same
number computed twice. More queries are needed, and hand-writing them costs an
afternoon per sixteen.

`gold_from_history` already established where free natural-language descriptions of
code come from: **the commit message.** It is prose about the code, written before
anyone was thinking about an eval, and it is *provably outside the indexed corpus* --
the index is built from the working tree, and no commit message is in the working
tree. That module mines one label per symbol for a PURPOSE eval. This one mines
queries for a RETRIEVAL eval, and the difference in direction changes almost every
decision:

  - **Commit-first, not symbol-first.** `gold_from_history` asks, per symbol, "which
    sentences of my introducing commit name me?" That yields exactly one label per
    symbol and therefore exactly one relevant symbol per query -- reproducing the
    defect of the hand set. Here the unit of mining is a *sentence*, and its relevant
    set is every symbol that sentence names. A sentence that names three symbols is a
    genuinely multi-relevant query, and it is multi-relevant because the author
    described three things together, not because a padding rule found near-misses.
  - **Not restricted to the introducing commit.** A purpose label has to come from the
    commit that WROTE the lines or it is prose about someone else's code. A retrieval
    query does not: "the reaper now takes the lease lock before expiring" is a
    perfectly good query for `reap_once` whether it introduced the function or changed
    it. Dropping the line-log requirement is also what makes this affordable -- the
    line log is one `git log -L` per symbol, and kalshi-bot has 2,700 of them.
  - **Attribution is by file touch.** A sentence is only allowed to name symbols in
    files that same commit changed. Without it, a commit that says "unlike
    `_reverse_dep_files`, this walks files" would mint a query for a symbol it did not
    touch, and a leaf name that collides across two modules would mint two.

Everything about the leak boundary is inherited rather than re-derived: `find_leaks`,
the `copied_into_source` rejection, and its cross-symbol twin all come from
`gold_from_history` and are applied here unchanged. What follows is what is *new*,
because retrieval gold has a leak the purpose eval does not care about.

## The lexical-freebie problem, and why it is REPORTED rather than filtered

A query that is largely its target's own identifier tokens is retrieved by BM25 for
free. Score a modality comparison on a set full of those and lexical wins by
construction, and every conclusion drawn from the table is about the gold set rather
than the retriever.

The obvious response is a token-overlap threshold: measure

    overlap = |content tokens of query INTERSECT content tokens of target source|
              / |content tokens of query|

and reject above some line. **That was measured before it was adopted, against the
hand-labelled set that would be the standard to hold mined queries to** -- and the
hand set does not meet it. Its 16 queries, scored against their own relevant symbols'
source at swarm-sync's working tree:

    min 0.00   median 0.71   mean 0.61   max 1.00   (2 of 16 at exactly 1.00)

"creating a new git worktree for an agent" scores 1.00: every content word of that
query appears in `add_worktree`'s source. It is still a good query -- describing what
code does in the vocabulary the code uses is what a user actually types, not a defect.
So a threshold anywhere useful would reject the benchmark it is meant to extend, and
picking one above the hand set's maximum would reject nothing.

The decision, therefore: `source_overlap` is **computed per query, stored on every
query, and summarised in the gold file's metadata** -- next to the hand set's own
distribution, so a consumer can see what "high" means here -- and no query is rejected
for it. A loader that wants a low-overlap stratum can filter on the field; a loader
that wants to report the overlap of the queries it scored can do that too. What is NOT
acceptable is an unmeasured set, and that is the failure this closes.

One thing IS rejected outright, and it is a different claim: `REJECT_NAME_ONLY`, a
query with fewer than `MIN_BLIND_TOKENS` content tokens left after its targets' name
tokens are removed. "Make get_first and get_last safe" leaves `make safe`. That is not
a low-information query, it is an identifier restatement, and the name-blind variant
of it (below) cannot be built at all. The threshold is a count rather than a ratio
because the question is whether the query is *able* to say anything its targets'
names do not, and `format_report` prints the sensitivity so the choice can be checked
rather than believed.

## Name blinding: both, marked, as separate rows

The mention rule guarantees every mined query contains its targets' identifiers --
that is how the query was found. So "should retrieval gold be name-blind" cannot be
answered by mining differently; it has to be answered by deciding what to emit.

Blinding everything is wrong. A real user query very often *does* contain the function
name -- "what does `reap_once` do" is the single most common thing anyone types at a
code search -- and a gold set with no name-bearing queries measures a retrieval mode
its users do not use. Blinding nothing is also wrong: name-bearing queries are exactly
where lexical retrieval gets its freebie, and a comparison run only on them is the
lexical benchmark this module exists to avoid.

So every candidate is emitted **twice**, as two rows sharing a `pair_id`:

  - `VARIANT_VERBATIM` -- the author's sentence, unmodified. `name_bearing` is true.
  - `VARIANT_NAME_BLIND` -- the same sentence with every dotted component of every
    relevant qualname, and every relevant path stem, removed by `blind_terms` and
    `_blind`. `name_bearing` is false.

They carry different `source` values, so a loader that scores sources as separate rows
reports them as separate rows without being told to, and the difference between the
two rows is a direct measurement of how much of a modality's score is name matching.
The blind variant is blinded by the UNION over all relevant symbols, not per symbol:
a query with two targets that blinded only one would still name the other.

One exception, and it is a correctness one rather than a convenience: two candidates
can blind to the SAME text with DIFFERENT gold answers -- facefusion's "Fix blank
screen in `replace_audio()`" and "Fix blank screen in `restore_audio()`" both become
`fix blank screen in`. One query string with two contradictory gold rows scores every
retriever wrong on at least one of them regardless of what it returns, so the blind
row is dropped there and the verbatim row is kept unpaired. `blind_rows_dropped` says
how often, and a consumer must not assume a `pair_id` appears exactly twice.

Two honest costs. `_blind` is a tokeniser, so the blind variant is a lowercased,
punctuation-free token sequence rather than a sentence -- it is the same
transformation `label_retrieval_validity` already applies, which is what makes the
numbers comparable to that function's, but it is not what a user would type. And the
two rows are not independent: they are the same sentence about the same symbols, so
1,000 queries across the two variants is 500 items, and any interval must be
clustered on `pair_id` (or on `commit`, which is coarser and also carried).

## The bias, which is the reason this file has a `bias` block

The mention rule selects symbols a commit message happened to name. The audit of
`gold_from_history` measured what that selects for on swarm-sync -- 100% documented
against 66% of the population, 14% private against 44%, classes over-represented
2.5x, methods under-represented 3.6x -- and this set inherits all of it, because it
uses the same rule.

`symbol_bias` recomputes that table for whatever was actually produced, per repo, and
`to_gold_json` puts it in the file. It is not a footnote: a retrieval set drawn only
from documented, public, class-shaped symbols will overstate any retriever that reads
docstrings, and dense retrieval reads docstrings. A gold set whose bias is stated is
usable -- a consumer can weight it, stratify it, or discount a result that lands
inside it. One whose bias is implicit is a trap, and the trap is baited with a
number that looks like it came from the whole codebase.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..ingest.indexer import is_test_path
from ..ingest.types import Symbol
from .gold_from_history import (
    _blind,
    _git,
    _tokens,
    blind_terms,
    find_leaks,
    is_boilerplate,
    mentions_symbol,
    repo_symbols,
    split_units,
    strip_trailers,
)

#: The two rows every candidate becomes. Distinct `source` values so a loader that
#: scores by source separates them without being configured to.
VARIANT_VERBATIM = "verbatim"
VARIANT_NAME_BLIND = "name_blind"
SOURCE_VERBATIM = "mined_verbatim"
SOURCE_NAME_BLIND = "mined_name_blind"

#: A sentence shorter than this is not a query. Lower than `gold_from_history`'s
#: `MIN_LABEL_WORDS = 6` on purpose: a purpose LABEL has to be a statement, while a
#: retrieval query legitimately is not ("removing a git worktree when the git command
#: fails" is 9 words, but half the hand set is under 8).
MIN_QUERY_WORDS = 5

#: Content tokens that must survive name blinding. Below this the blind variant is a
#: one- or two-word bag and the pair degenerates into the verbatim row twice. See the
#: module docstring; `format_report` prints what 2, 3, 4 and 5 would each have cost.
MIN_BLIND_TOKENS = 3

#: A sentence naming more than this many symbols is an enumeration, not a description
#: -- "BarFeed, SetupDetector ABC + SetupAlert dataclass + @register_setup" names four
#: things and describes none of them. Multi-relevant queries are the point of this
#: module, so the cap is set where a sentence stops being prose rather than where the
#: relevant set stops being convenient.
MAX_RELEVANT = 5

#: Code-punctuation characters per WORD above which a "sentence" is really a snippet.
#: 1.0 is not a tuned number: it says the unit has more brackets and equals signs than
#: words, which English cannot sustain and `x = foo(bar)` exceeds immediately.
#:
#: It was tuned once, badly, and the correction is worth recording. The first version
#: was an absolute count (>= 6 such characters), which rejected 18 of swarm-sync's 121
#: candidates -- and inspecting all 18 found not one snippet among them. They were
#: dense engineering prose: "reconcile_orphaned_integrations(): reads the projection --
#: O(open), not O(history)." is eleven words and six of those characters. Commit prose
#: in these repos is punctuation-heavy, so an absolute count measures verbosity rather
#: than code. At the shipped ratio the highest-scoring real unit across all five repos
#: is 0.75, so this filter rejects NOTHING here -- it is a guard for repos that embed
#: unfenced code, and `format_report` prints its zero rather than hiding it.
CODE_PUNCT_PER_WORD = 1.0

#: Fenced blocks only. An earlier version also stripped 4-space-indented lines as
#: code, which cost real prose: swarm-sync writes indented continuation paragraphs and
#: kalshi-bot indents wrapped list items, and the stripper was deleting 84 and 59 lines
#: of English from those two repos respectively. Indented CODE is caught downstream by
#: `looks_like_code`, which is the right place for a judgement about content.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_CODE_PUNCT = re.compile(r"[=(){}\[\]]")

#: Record/field separators for `--format`, from the C0 controls -- the same choice and
#: the same reason as `gold_from_history._LOG_FORMAT`: commit prose contains every
#: printable character, so a printable delimiter is not a delimiter.
_FS = "\x01"
_RS = "\x02"

# Rejection reasons. As in `gold_from_history`, these are values rather than booleans
# because the funnel is the result and not a diagnostic -- a miner that reports only
# what it kept can report any yield it likes.
REJECT_TOO_SHORT = "too_short"
REJECT_CODE_LIKE = "code_like"
REJECT_TOO_MANY = "too_many_relevant"
REJECT_NAME_ONLY = "name_only"
REJECT_COPIED_INTO_SOURCE = "copied_into_source"
REJECT_COPIED_INTO_SIBLING = "copied_into_sibling"
REJECT_DUPLICATE = "duplicate"
REJECT_NOT_IN_INDEX = "not_in_index"

# Commit-level exclusions, counted before any candidate exists.
SKIP_EMPTY_PROSE = "empty_prose"
SKIP_BOILERPLATE = "boilerplate_subject"
SKIP_NO_LIVE_FILES = "no_live_python_files"
SKIP_NO_SYMBOLS = "no_symbols_in_touched_files"


@dataclass(frozen=True)
class Commit:
    """One commit's prose and the files it touched. Never handed to a retriever."""

    sha: str
    subject: str
    body: str
    files: tuple[str, ...]


@dataclass
class QueryCandidate:
    """One mined query, usable or rejected with a reason.

    `relevant` is ordered for stability, not by relevance -- the scorers treat the set
    as unordered and an accidental ordering would be read as a ranking.
    """

    query: str
    relevant: list[str]
    paths: list[str]
    commit: str
    subject: str
    unit_index: int
    source_overlap: float = 0.0
    blind_tokens: int = 0
    reject: str | None = None
    #: False when the name-blind row would be ambiguous -- see
    #: `_drop_ambiguous_blind_rows`. The verbatim row survives; only the pair does not.
    blind_ok: bool = True

    @property
    def usable(self) -> bool:
        return self.reject is None

    @property
    def pair_id(self) -> str:
        """Repo-local identity of the verbatim/blind pair. Qualified by repo on emit.

        Commit plus sentence index, which is unique within a repo because a sentence
        has one position in one message. `to_gold_json` prefixes the repo name, since
        a loader that concatenates several gold files needs the id to stay unique
        across them -- and two repos CAN share a short sha.
        """
        return f"{self.commit[:10]}:{self.unit_index}"

    def blinded(self) -> str:
        return blind_query(self.query, self.relevant, self.paths)


@dataclass
class MinedGoldReport:
    """The funnel, the queries, and the bias -- the three things a gold set owes.

    `considered` counts *candidates*, meaning sentences that named at least one symbol
    in a file their own commit touched. The commit-level losses above that are in
    `skipped`, kept separate because "the commit said nothing" and "the commit said
    something that did not survive the filters" are different findings.
    """

    repo: str
    head: str = ""
    commits_in_history: int = 0
    commits_used: int = 0
    considered: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    candidates: list[QueryCandidate] = field(default_factory=list)
    index_path: str = ""

    @property
    def usable(self) -> list[QueryCandidate]:
        return [c for c in self.candidates if c.usable]

    @property
    def usable_fraction(self) -> float:
        return len(self.usable) / self.considered if self.considered else 0.0

    def rejects(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cand in self.candidates:
            if cand.reject:
                counts[cand.reject] = counts.get(cand.reject, 0) + 1
        return counts

    def relevant_sizes(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for cand in self.usable:
            counts[len(cand.relevant)] = counts.get(len(cand.relevant), 0) + 1
        return dict(sorted(counts.items()))

    @property
    def multi_relevant(self) -> int:
        return sum(1 for c in self.usable if len(c.relevant) > 1)

    @property
    def blind_rows_dropped(self) -> int:
        """Usable queries whose name-blind row was ambiguous. See `blind_ok`."""
        return sum(1 for c in self.usable if not c.blind_ok)

    def target_symbols(self) -> set[str]:
        return {q for c in self.usable for q in c.relevant}

    def commit_sharing(self) -> dict[str, int]:
        """Usable candidates per commit -- the cluster sizes any interval must use."""
        counts: dict[str, int] = {}
        for cand in self.usable:
            counts[cand.commit] = counts.get(cand.commit, 0) + 1
        return counts


# --------------------------------------------------------------------------------
# Reading history
# --------------------------------------------------------------------------------


def iter_commits(repo: Path, timeout: int = 300) -> list[Commit]:
    """Every non-merge commit with its message and its changed paths.

    Two `git log` passes rather than one. A single pass would have to emit `%B` and
    `--name-only` into the same record, and since a body is arbitrary multi-line text
    there is no way to tell where it ends and the file list begins -- the parse would
    be wrong exactly on the commits with the richest prose, which are the ones this
    module is here for.

    Merges are excluded: a merge commit's message describes a branch, its file list is
    the union of everything on that branch, and pairing the two would attribute a
    one-line summary to a hundred symbols.
    """
    out = _git(repo, "log", "--no-merges", f"--format={_RS}%H{_FS}%s{_FS}%B", timeout=timeout)
    messages: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    for raw in (out or "").split(_RS):
        if not raw.strip():
            continue
        parts = raw.split(_FS)
        if len(parts) >= 3:
            sha = parts[0].strip()
            messages[sha] = (parts[1].strip(), _FS.join(parts[2:]))
            order.append(sha)
    out2 = _git(repo, "log", "--no-merges", f"--format={_RS}%H", "--name-only", timeout=timeout)
    touched: dict[str, list[str]] = {}
    for raw in (out2 or "").split(_RS):
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if lines:
            touched[lines[0]] = lines[1:]
    return [
        Commit(
            sha=sha,
            subject=messages[sha][0],
            body=messages[sha][1],
            files=tuple(touched.get(sha, ())),
        )
        for sha in order
    ]


def head_sha(repo: Path) -> str:
    out = _git(repo, "rev-parse", "HEAD")
    return out.strip() if out else ""


def strip_code_blocks(prose: str) -> str:
    """Remove fenced code from commit prose, before splitting.

    A code block that survives into a query is the worst case this module has: it is
    not a description of the code, it *is* the code, so its token overlap with the
    target is 1.0 by construction and any retriever that indexes the file finds it.
    Stripped rather than rejected downstream so that the prose AROUND a snippet -- the
    sentence explaining it, which is a real query -- is still available. On
    polymarket-official-client, whose bodies embed whole README examples, this alone
    takes the candidate count from 16 to 5 and every one of the 11 it removed was a
    line of Python.
    """
    return _FENCE.sub("\n", prose)


def looks_like_code(unit: str) -> bool:
    """Whether a "sentence" is really a fragment of source.

    A ratio of bracket and assignment characters to words, which is crude and is meant
    to be: the precise version would be to parse it, and prose that parses as Python is
    vanishingly rare next to prose that merely contains `foo(bar)`. See
    `CODE_PUNCT_PER_WORD` for what an absolute count got wrong.
    """
    words = len(unit.split())
    if not words:
        return False
    return len(_CODE_PUNCT.findall(unit)) / words >= CODE_PUNCT_PER_WORD


# --------------------------------------------------------------------------------
# Mining
# --------------------------------------------------------------------------------


def _symbol_index(
    repo: Path, include_tests: bool = False
) -> tuple[dict[str, list[Symbol]], dict[str, Symbol]]:
    """(path -> symbols, qualname -> symbol) for the repo's working tree.

    Built from `repo_symbols`, which reads git-TRACKED python files. That matters for
    more than tidiness: a commit can have touched a file that is now untracked or
    gitignored, and such a file is not in the index either, so a query pointing at it
    would be unresolvable gold. Restricting the candidate pool to this map makes that
    impossible up front rather than at validation time.
    """
    by_path: dict[str, list[Symbol]] = {}
    by_qualname: dict[str, Symbol] = {}
    for rel, sym in repo_symbols(repo, include_tests=include_tests):
        by_path.setdefault(rel, []).append(sym)
        by_qualname[sym.qualname] = sym
    return by_path, by_qualname


def _symbol_source(repo: Path, rel: str, sym: Symbol, cache: dict[str, bytes]) -> str:
    """A symbol's own source text, sliced on BYTE offsets.

    Byte offsets, not character offsets, because the extractor's spans are byte spans
    and slicing a `str` by them shifts silently on any file containing a non-ASCII
    character. Commit prose and docstrings in these repos are full of em dashes.
    """
    if rel not in cache:
        try:
            cache[rel] = (repo / rel).read_bytes()
        except OSError:
            cache[rel] = b""
    return cache[rel][sym.byte_start : sym.byte_end].decode("utf-8", "replace")


def source_overlap(query: str, sources: Iterable[str]) -> float:
    """Fraction of the query's content tokens that occur in its targets' source.

    The lexical-freebie measure. 1.0 means BM25 can match every content word of the
    query against the code it is supposed to find; 0.0 means the query and the code
    share no vocabulary and only a semantic retriever can bridge them. Reported, never
    filtered on -- see the module docstring for the measurement that settled that.
    """
    qtokens = set(_tokens(query))
    if not qtokens:
        return 0.0
    stokens: set[str] = set()
    for text in sources:
        stokens |= set(_tokens(text))
    return len(qtokens & stokens) / len(qtokens)


def blind_query(query: str, relevant: Sequence[str], paths: Sequence[str]) -> str:
    """`query` with every target's name tokens removed, by the union over all targets.

    Per-symbol blinding would leave a two-target query still naming one of its
    targets, which is the whole leak on exactly the queries this module added.
    """
    terms: set[str] = set()
    for qualname, path in zip(relevant, paths, strict=True):
        terms |= set(blind_terms(qualname, path))
    return _blind(query, frozenset(terms))


def mine_queries(
    repo: Path,
    include_tests: bool = False,
    max_relevant: int = MAX_RELEVANT,
    min_blind_tokens: int = MIN_BLIND_TOKENS,
) -> MinedGoldReport:
    """Mine retrieval queries from a repo's commit prose. Every candidate is reported.

    The loop is commit -> sentence -> symbols, and the file-touch constraint is applied
    where the candidate pool is built rather than as a filter afterwards, so a sentence
    can only ever name symbols its own commit could plausibly have been about.
    """
    repo = Path(repo)
    by_path, by_qualname = _symbol_index(repo, include_tests=include_tests)
    commits = iter_commits(repo)
    report = MinedGoldReport(
        repo=str(repo), head=head_sha(repo), commits_in_history=len(commits)
    )
    byte_cache: dict[str, bytes] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    used_commits: set[str] = set()

    for commit in commits:
        prose = strip_trailers(commit.body)
        if not prose:
            report.skipped[SKIP_EMPTY_PROSE] = report.skipped.get(SKIP_EMPTY_PROSE, 0) + 1
            continue
        if is_boilerplate(commit.subject):
            report.skipped[SKIP_BOILERPLATE] = report.skipped.get(SKIP_BOILERPLATE, 0) + 1
            continue
        paths = [
            f
            for f in commit.files
            if f.endswith(".py")
            and f in by_path
            and (include_tests or not is_test_path(f))
        ]
        if not paths:
            report.skipped[SKIP_NO_LIVE_FILES] = report.skipped.get(SKIP_NO_LIVE_FILES, 0) + 1
            continue
        pool = [(rel, sym) for rel in paths for sym in by_path[rel]]
        if not pool:
            report.skipped[SKIP_NO_SYMBOLS] = report.skipped.get(SKIP_NO_SYMBOLS, 0) + 1
            continue
        for i, unit in enumerate(split_units(strip_code_blocks(prose))):
            hits = [(rel, sym) for rel, sym in pool if mentions_symbol(unit, sym.name)]
            if not hits:
                continue
            used_commits.add(commit.sha)
            report.considered += 1
            hits.sort(key=lambda pair: pair[1].qualname)
            relevant = [sym.qualname for _, sym in hits]
            hit_paths = [rel for rel, _ in hits]
            sources = [
                _symbol_source(repo, rel, sym, byte_cache) for rel, sym in hits
            ]
            cand = QueryCandidate(
                query=unit,
                relevant=relevant,
                paths=hit_paths,
                commit=commit.sha,
                subject=commit.subject,
                unit_index=i,
                source_overlap=source_overlap(unit, sources),
                blind_tokens=len(_tokens(blind_query(unit, relevant, hit_paths))),
            )
            report.candidates.append(_classify(cand, sources, max_relevant, min_blind_tokens, seen))

    report.commits_used = len(used_commits)
    _reject_cross_symbol_copies(report, by_path, by_qualname, repo, byte_cache)
    _drop_ambiguous_blind_rows(report)
    return report


def _drop_ambiguous_blind_rows(report: MinedGoldReport) -> None:
    """Clear `blind_ok` where two candidates blind to the same text but differ in gold.

    facefusion has "Fix blank screen in `replace_audio()`" and "Fix blank screen in
    `restore_audio()`" in two different commits. Both are fine verbatim queries with
    different correct answers. Blinded, both become `fix blank screen in` -- one query
    string carrying two contradictory gold rows, so any retriever is scored wrong on at
    least one of them no matter what it returns. That is not a hard query, it is a
    broken one, and it would show up as a retrieval failure the retriever could not
    have avoided.

    Only the BLIND row is dropped. The verbatim rows are unaffected and stay paired
    with nothing, which is why `pair_id` is not assumed to appear exactly twice: a
    consumer pairing rows must tolerate a verbatim row with no blind partner, and
    `funnel.blind_rows_dropped` says how often that happens.

    Collisions whose relevant sets are IDENTICAL are a different case -- the same
    question asked twice -- and only the first is kept, as a duplicate rather than an
    ambiguity.
    """
    groups: dict[str, list[int]] = {}
    for i, cand in enumerate(report.candidates):
        if cand.usable:
            groups.setdefault(" ".join(cand.blinded().split()), []).append(i)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        sets = {tuple(report.candidates[i].relevant) for i in indices}
        keep = indices[0] if len(sets) == 1 else None
        for i in indices:
            if i != keep:
                report.candidates[i] = replace(report.candidates[i], blind_ok=False)


def _classify(
    cand: QueryCandidate,
    sources: Sequence[str],
    max_relevant: int,
    min_blind_tokens: int,
    seen: set[tuple[str, tuple[str, ...]]],
) -> QueryCandidate:
    """Apply every per-candidate filter, in the order the funnel should read.

    Order is not arbitrary. Shape checks come first so that "this was never a
    sentence" is not reported as a leak, and the duplicate check comes LAST so that a
    repeated sentence is only recorded as a duplicate of something that was itself
    usable -- otherwise the first occurrence could be rejected as code-like and the
    second reported as a duplicate of nothing.
    """
    if len(cand.query.split()) < MIN_QUERY_WORDS:
        return replace(cand, reject=REJECT_TOO_SHORT)
    if looks_like_code(cand.query):
        return replace(cand, reject=REJECT_CODE_LIKE)
    if len(cand.relevant) > max_relevant:
        return replace(cand, reject=REJECT_TOO_MANY)
    if cand.blind_tokens < min_blind_tokens:
        return replace(cand, reject=REJECT_NAME_ONLY)
    # Inherited verbatim from `gold_from_history`: prose copied into the source it
    # describes is not held out, and here it is also a query the retriever can match
    # character for character.
    for text in sources:
        if find_leaks(text, [cand.query]):
            return replace(cand, reject=REJECT_COPIED_INTO_SOURCE)
    key = (" ".join(cand.query.lower().split()), tuple(cand.relevant))
    if key in seen:
        return replace(cand, reject=REJECT_DUPLICATE)
    seen.add(key)
    return cand


def _reject_cross_symbol_copies(
    report: MinedGoldReport,
    by_path: dict[str, list[Symbol]],
    by_qualname: dict[str, Symbol],
    repo: Path,
    byte_cache: dict[str, bytes],
) -> None:
    """Reject a query copied verbatim into some OTHER selected symbol's source.

    The per-candidate copy check compares a query to its own targets and is blind to
    this by construction -- the same argument `gold_from_history` makes for
    `REJECT_COPIED_INTO_SIBLING`, where the real finding on swarm-sync was a clause
    from one symbol's label sitting in a different symbol's docstring, from a
    different commit.

    It matters more here than there. A query whose text is inside symbol B's docstring
    while its gold answer is symbol A is not merely un-held-out: it is a query the
    lexical retriever will answer with B, and every modality will then be scored on a
    query whose best lexical match is a symbol the gold set calls WRONG. That is worse
    than a leak; it is an inverted label.

    Scoped to the symbols this gold set actually names, not to the whole repo. A copy
    into a symbol no query points at cannot invert any label, and scanning every symbol
    in kalshi-bot against every candidate is a 2,700 x 105 substring sweep for findings
    that could not affect a score.
    """
    targets = report.target_symbols()
    if len(targets) < 2:
        return
    path_of = {
        sym.qualname: rel for rel, syms in by_path.items() for sym in syms
    }
    sources: dict[str, str] = {}
    for qualname in targets:
        sym = by_qualname.get(qualname)
        rel = path_of.get(qualname)
        if sym is not None and rel is not None:
            sources[qualname] = _symbol_source(repo, rel, sym, byte_cache)
    for i, cand in enumerate(report.candidates):
        if not cand.usable:
            continue
        own = set(cand.relevant)
        for qualname, text in sources.items():
            if qualname in own:
                continue
            if find_leaks(text, [cand.query]):
                report.candidates[i] = replace(cand, reject=REJECT_COPIED_INTO_SIBLING)
                break


# --------------------------------------------------------------------------------
# Index validation
# --------------------------------------------------------------------------------


def index_qualnames(conn: sqlite3.Connection, include_tests: bool = False) -> set[str]:
    """Every symbol qualname an index holds, which is the set gold may point at."""
    sql = "SELECT s.qualname FROM symbols s JOIN files f ON f.id = s.file_id"
    if not include_tests:
        sql += " WHERE f.is_test = 0"
    return {row[0] for row in conn.execute(sql)}


def validate_against_index(
    report: MinedGoldReport, known: set[str], index_path: str = ""
) -> list[str]:
    """Reject every candidate naming a qualname the index does not hold. Returns them.

    A gold entry pointing at a symbol that is not in the index is not a hard query, it
    is an unanswerable one, and it drags every metric down by a fixed amount that looks
    like a retrieval failure. **All-or-nothing per query**: a two-target query with one
    missing target is not silently narrowed to one target, because that would change
    what the query is asking without saying so.
    """
    report.index_path = index_path
    missing: list[str] = []
    for i, cand in enumerate(report.candidates):
        if not cand.usable:
            continue
        gone = [q for q in cand.relevant if q not in known]
        if gone:
            missing.extend(gone)
            report.candidates[i] = replace(cand, reject=REJECT_NOT_IN_INDEX)
    return sorted(set(missing))


# --------------------------------------------------------------------------------
# Bias
# --------------------------------------------------------------------------------


def _profile(symbols: Sequence[Symbol]) -> dict:
    """The three properties the `gold_from_history` audit found the rule skews."""
    total = len(symbols)
    if not total:
        return {"n": 0, "documented": 0.0, "private": 0.0, "kinds": {}}
    kinds: dict[str, int] = {}
    for sym in symbols:
        kinds[sym.kind] = kinds.get(sym.kind, 0) + 1
    return {
        "n": total,
        "documented": sum(1 for s in symbols if (s.docstring or "").strip()) / total,
        "private": sum(1 for s in symbols if s.name.startswith("_")) / total,
        "kinds": {k: kinds.get(k, 0) / total for k in sorted(kinds)},
    }


def symbol_bias(
    repo: Path, report: MinedGoldReport, include_tests: bool = False
) -> dict:
    """Population vs selected, on documentation, privacy and kind.

    The number that carries information is the RATIO, not either level: a repo where
    80% of symbols are documented and a gold set where 100% are is a mild skew, and
    one where 20% are documented is a severe one, and the two look identical if only
    the gold set's level is printed. `over_representation` is selected/population per
    kind, which is the form the audit reported and the form that is comparable across
    repos of different shapes.
    """
    by_path, by_qualname = _symbol_index(repo, include_tests=include_tests)
    population = [sym for syms in by_path.values() for sym in syms]
    chosen = [by_qualname[q] for q in sorted(report.target_symbols()) if q in by_qualname]
    pop = _profile(population)
    sel = _profile(chosen)
    over = {
        kind: (sel["kinds"].get(kind, 0.0) / pop["kinds"][kind])
        for kind in pop["kinds"]
        if pop["kinds"][kind]
    }
    return {
        "note": (
            "The mention rule selects symbols a commit message happened to name. This "
            "set inherits that bias in full; it is measured here, not corrected. "
            "'over_representation' is the selected share of a kind divided by its "
            "population share -- 1.0 is neutral, above 1.0 is over-represented."
        ),
        "population": pop,
        "selected": sel,
        "over_representation": {k: round(v, 3) for k, v in sorted(over.items())},
    }


# --------------------------------------------------------------------------------
# Emitting
# --------------------------------------------------------------------------------


def _overlap_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        return round(ordered[min(n - 1, int(p * n))], 3)

    buckets: dict[str, int] = {}
    for value in ordered:
        lo = min(int(value * 10) * 10, 90)
        buckets[f"{lo}-{lo + 10}%"] = buckets.get(f"{lo}-{lo + 10}%", 0) + 1
    return {
        "n": n,
        "min": round(ordered[0], 3),
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / n, 3),
        "deciles": dict(sorted(buckets.items())),
    }


#: The hand-labelled set's own query -> source token overlap, measured at
#: swarm-sync's working tree with `source_overlap`. Recorded as a CONSTANT and
#: shipped in every mined gold file because it is the number that makes a mined
#: query's overlap readable: without it, "median 0.66" invites a reader to conclude
#: the set is lexically contaminated, when the hand set a reviewer would compare it
#: to sits at 0.71 with two queries at 1.00.
HAND_SET_OVERLAP = {
    "gold": "swarm_sync.json (16 hand-written queries)",
    "min": 0.0,
    "median": 0.71,
    "mean": 0.61,
    "max": 1.0,
    "at_1.0": 2,
}

LABELLING_RULE = (
    "Relevant = every symbol named AS CODE by one sentence of a commit message, where "
    "that commit also changed the file the symbol lives in. 'As code' is "
    "gold_from_history.mentions_symbol: either the identifier is distinctive enough "
    "that a bare occurrence cannot be English, or it appears in backticks, called, "
    "attribute-accessed, or assigned. The sentence, unmodified, is the query. "
    "Excluded: commits with boilerplate subjects, merge commits, sentences under "
    f"{MIN_QUERY_WORDS} words, code snippets, sentences naming more than {MAX_RELEVANT} "
    f"symbols, sentences leaving fewer than {MIN_BLIND_TOKENS} content tokens once "
    "their targets' name tokens are removed, sentences found verbatim (32-char run) in "
    "any selected symbol's source, exact duplicates, and any query naming a qualname "
    "the index does not hold. NOT excluded: high token overlap with the target's "
    "source -- it is measured per query and reported, because the hand-labelled set's "
    "own overlap is a median 0.71 and a threshold would reject it too. Every query is "
    "emitted twice, verbatim and name-blinded, as separate rows sharing a pair_id -- "
    "except where two queries blind to the same text with different relevant sets, "
    "where the blind row is dropped as ambiguous and the verbatim row is left "
    "unpaired. The two rows of a pair are the same item and must be clustered, not "
    "counted twice."
)


def to_gold_json(report: MinedGoldReport, bias: dict) -> dict:
    """The gold file: a superset of `gold/swarm_sync.json`'s shape.

    `repo`, `commit_note`, `labelling_rule` and `queries: [{query, relevant}]` are
    present and mean what they mean there, so an existing loader reads this file
    without knowing anything about mining. Everything else is additive, and every
    additive field is either provenance (which commit, which sha, which rule) or a
    number a consumer needs in order to read the set honestly (`source_overlap`,
    `bias`, `pair_id`).

    Each row carries an explicit `id` of `<repo>:<sha10>:<sentence>:<variant>` rather
    than leaving one to be derived. A loader that slugs the query text collides on
    mined prose in a way it never does on hand-written queries: two sentences of the
    same commit routinely share their first sixty characters ("It serializes
    concurrent POST /integrate requests inside one ..."), which is a duplicate id and a
    hard schema error. Commit plus sentence index cannot collide, is stable across
    regenerations at the same sha, and says where the row came from.

    Per-query `repo` and `source` are duplicated onto every row deliberately. They are
    redundant with the file header, and that redundancy is what lets a loader
    concatenate several gold files into one list and still score each source as its
    own row without carrying a parallel structure to say where each query came from.
    """
    queries = []
    repo_name = Path(report.repo).name
    for cand in report.usable:
        common = {
            "relevant": list(cand.relevant),
            "repo": repo_name,
            "pair_id": f"{repo_name}:{cand.pair_id}",
            "commit": cand.commit[:10],
            "n_relevant": len(cand.relevant),
            "source_overlap": round(cand.source_overlap, 3),
        }
        queries.append(
            {
                "id": f"{common['pair_id']}:{VARIANT_VERBATIM}",
                "query": cand.query,
                "source": SOURCE_VERBATIM,
                "variant": VARIANT_VERBATIM,
                "name_bearing": True,
                **common,
            }
        )
        if cand.blind_ok:
            queries.append(
                {
                    "id": f"{common['pair_id']}:{VARIANT_NAME_BLIND}",
                    "query": cand.blinded(),
                    "source": SOURCE_NAME_BLIND,
                    "variant": VARIANT_NAME_BLIND,
                    "name_bearing": False,
                    **common,
                }
            )
    overlaps = [c.source_overlap for c in report.usable]
    return {
        # The repo NAME, never the absolute path: a shipped artifact that records
        # someone's home directory leaks something other than gold labels.
        "repo": Path(report.repo).name,
        "source": "mined",
        "generated_by": "codelearner.eval.mined_gold.mine_queries",
        "mined_at_head": report.head,
        "commit_note": (
            f"MINED from commit prose, not hand-labelled: {len(report.usable)} queries "
            f"from {report.considered} candidate sentences across "
            f"{report.commits_used} of {report.commits_in_history} commits, emitted as "
            f"{len(queries)} rows (verbatim + name-blind per query). Snapshot at "
            f"{report.head[:10]} -- a mined set is a function of history and moves with "
            "the next commit; regenerate rather than trusting this file."
        ),
        "labelling_rule": LABELLING_RULE,
        "funnel": {
            "commits_in_history": report.commits_in_history,
            "commits_contributing": report.commits_used,
            "commits_skipped": dict(sorted(report.skipped.items())),
            "candidates_considered": report.considered,
            "rejected": dict(sorted(report.rejects().items())),
            "usable_queries": len(report.usable),
            "usable_fraction": round(report.usable_fraction, 3),
            "distinct_target_symbols": len(report.target_symbols()),
            "relevant_set_sizes": {str(k): v for k, v in report.relevant_sizes().items()},
            "multi_relevant": report.multi_relevant,
            "largest_commit_cluster": max(report.commit_sharing().values(), default=0),
            "blind_rows_dropped": report.blind_rows_dropped,
            "rows_emitted": len(queries),
        },
        "source_overlap": {
            "definition": (
                "|content tokens of query & content tokens of relevant symbols' source| "
                "/ |content tokens of query|. Reported, never filtered on."
            ),
            "mined": _overlap_summary(overlaps),
            "hand_labelled_reference": HAND_SET_OVERLAP,
        },
        "bias": bias,
        "queries": queries,
    }


def write_gold(report: MinedGoldReport, bias: dict, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_gold_json(report, bias), indent=2) + "\n")
    return path


def iter_gold_files(gold_dir: Path) -> Iterator[Path]:
    """Every mined gold file. The `mined_` prefix is this module's namespace."""
    return iter(sorted(Path(gold_dir).glob("mined_*.json")))


def blind_sensitivity(
    report: MinedGoldReport, counts: Sequence[int] = (2, 3, 4, 5)
) -> dict[int, int]:
    """How many candidates `MIN_BLIND_TOKENS` would reject at each setting.

    Printed rather than reasoned about. `COPY_RUN_CHARS` in `gold_from_history` earns
    its threshold by showing it rejects the same labels at 24, 32, 48 and 64; this is
    the same courtesy for a threshold that does NOT turn out to be flat, so a reader
    can see what the choice cost instead of taking the constant on trust.
    """
    shaped = [
        c
        for c in report.candidates
        if c.reject
        not in (REJECT_TOO_SHORT, REJECT_CODE_LIKE, REJECT_TOO_MANY)
    ]
    return {n: sum(1 for c in shaped if c.blind_tokens < n) for n in counts}


def format_report(report: MinedGoldReport, bias: dict | None = None) -> str:
    """The funnel, the multi-relevance split, and the bias -- in that order."""
    lines = [
        f"repo: {report.repo}",
        f"head: {report.head[:10]}   commits: {report.commits_in_history}"
        f"   contributing: {report.commits_used}",
        "",
        "COMMIT FUNNEL",
    ]
    for reason in (SKIP_EMPTY_PROSE, SKIP_BOILERPLATE, SKIP_NO_LIVE_FILES, SKIP_NO_SYMBOLS):
        lines.append(f"  skipped: {reason:<28} {report.skipped.get(reason, 0):>5}")
    lines += ["", "CANDIDATE FUNNEL", f"  sentences naming a symbol      {report.considered:>5}"]
    for reason in (
        REJECT_TOO_SHORT,
        REJECT_CODE_LIKE,
        REJECT_TOO_MANY,
        REJECT_NAME_ONLY,
        REJECT_COPIED_INTO_SOURCE,
        REJECT_COPIED_INTO_SIBLING,
        REJECT_DUPLICATE,
        REJECT_NOT_IN_INDEX,
    ):
        lines.append(f"  rejected: {reason:<22} {report.rejects().get(reason, 0):>5}")
    lines += [
        f"  USABLE                         {len(report.usable):>5}"
        f"   ({report.usable_fraction:.1%} of considered)",
        f"  distinct target symbols        {len(report.target_symbols()):>5}",
        f"  multi-relevant                 {report.multi_relevant:>5}"
        f"   of {len(report.usable)}",
        f"  relevant-set sizes             {report.relevant_sizes()}",
        f"  largest commit cluster         {max(report.commit_sharing().values(), default=0):>5}",
        f"  blind rows dropped (ambiguous) {report.blind_rows_dropped:>5}",
        f"  rows emitted                   {2 * len(report.usable) - report.blind_rows_dropped:>5}",
        f"  MIN_BLIND_TOKENS sensitivity   {blind_sensitivity(report)}",
        "",
        "SOURCE OVERLAP (reported, not filtered)",
        f"  mined: {_overlap_summary([c.source_overlap for c in report.usable])}",
        f"  hand:  {HAND_SET_OVERLAP}",
    ]
    if bias:
        pop, sel = bias["population"], bias["selected"]
        lines += [
            "",
            "SELECTION BIAS (population -> selected)",
            f"  n                  {pop['n']:>6} -> {sel['n']:>5}",
            f"  documented         {pop['documented']:>6.1%} -> {sel['documented']:>5.1%}",
            f"  private            {pop['private']:>6.1%} -> {sel['private']:>5.1%}",
        ]
        for kind in sorted(pop["kinds"]):
            lines.append(
                f"  {kind:<18} {pop['kinds'][kind]:>6.1%} -> "
                f"{sel['kinds'].get(kind, 0.0):>5.1%}"
                f"   ({bias['over_representation'].get(kind, 0.0):.2f}x)"
            )
    return "\n".join(lines)
