"""Gold labels for *purpose* accuracy, mined from git history.

Retrieval can be evaluated against a hand-labelled gold set because "is the right
symbol in the top ten" is cheap to label -- `gold/swarm_sync.json` is 16 queries over
23 symbols, and it took an afternoon. Purpose is not like that. Labelling "what is
this function for" by hand costs a paragraph of careful prose per symbol, and a repo
has a thousand symbols. That is the reason the inference layer has no eval.

The trick this module tests: **the commit that introduced a symbol usually states, in
prose, why it was written.** If that is true, then every repo with real history is
already a labelled purpose corpus -- free, unlimited, and written before anyone was
thinking about an eval, which makes it leak-free by construction *provided the
generator never sees it*.

That proviso is the whole design, so it is enforced in code rather than promised:

  - The generator is handed a `SourceView` -- a frozen record built from the working
    tree and nothing else. `source_view()` never invokes git; `test_gold_from_history`
    proves it by making `subprocess.run` raise and calling it anyway.
  - `find_leaks()` walks a view's whole object graph looking for the label text. The
    scoring harness runs it on every view before it scores anything and raises
    `LeakDetected` rather than reporting a number.
  - `suspect_tokens()` checks the other direction: a rare word that appears in the
    held-out prose and in the generator's OUTPUT but nowhere in the input it was
    given is either a coincidence or a leak, and is counted and reported either way.

An eval whose ground truth is reachable from its input measures nothing, and it
measures nothing *silently* -- the scores look better, which is exactly the direction
that does not prompt anyone to check.

## What was measured, on swarm-sync (93 commits, 316 non-test symbols)

The technique yields a usable label for **13.3% of symbols** (42 of 316), and the
funnel says where the other 274 went. One rejection reason accounts for essentially
all of it: 272 symbols were introduced by a commit whose prose never names them.

  - 163 symbols (52%) trace to the single initial commit, whose entire message is a
    602-character project summary. One label cannot describe 163 symbols.
  - The rest come from work-package-sized commits (2 to 57 files, median 6). Their
    prose describes a *change*, and usually several changes.

The prior going in was that commit messages would be too *low-quality* to use ("fix",
"wip", "address review"). On this corpus that prior was wrong and the boilerplate
filter rejected **zero** symbols -- these commit messages are excellent, with a median
body of 1,021 characters. The problem is not quality, it is **attribution**: excellent
prose about a work package is not a purpose statement about a symbol. The filter that
does the work is therefore the mention rule, and it costs 87% of the corpus.

So the honest summary is that this pays off as a *sample*, not as a labelling. 42
labels for zero labelling effort is nearly twice the hand-labelled retrieval gold set,
and it covers 13% of the symbols one would want to measure. It gets no cheaper on a
smaller repo: run against code-learner itself (7 commits, 231 symbols) the yield is
**3 labels, 1.3%**. The technique needs history that is fine-grained, not merely
present.

## Does the mined prose actually describe its symbol?

Two independent checks, because "the commit named the symbol" is not the same claim.

**Purpose agreement** (`score_purposes`, name-blind token-F1, n=42): a docstring-first-
sentence generator scores 0.159 against 0.023 for the shuffled control; a body-
identifier bag scores 0.208 against 0.047; name-and-signature-only scores 0.020
against 0.002. Every condition clears its control by 4-7x, so the labels do carry
symbol-specific signal. Absolute values are low, and the reasons are limits 2 and 4
below.

Swap token-F1 for `Qwen3-Embedding-0.6B` cosine and the ordering is identical:
docstring 0.631 vs 0.427 control (lift 0.203), body identifiers 0.675 vs 0.480 (0.194),
body doc-blind 0.557 vs 0.420 (0.137), name-and-signature 0.451 vs 0.416 (0.035). Two
unrelated similarity measures ranking four conditions the same way is the most
reassuring result here -- it is evidence about the labels rather than about either
metric. Note the cosine floor: any two English texts score ~0.42, so the raw numbers
look far better than the token-F1 ones while carrying the same information, and the
name echo keeps a residual 0.035 lift that token-F1 puts at 0.018. Neither measure
blinds a name perfectly.

**Label validity** (`label_retrieval_validity`, lexical, name-blind): use each label
as a search query and see whether it retrieves the symbol it was mined from. MRR
0.288, hit@5 0.452, hit@10 0.500. So half the mined labels do not put their own symbol
in the top ten of a lexical search -- those are labels whose vocabulary is about a
work package. For scale, the hand-labelled gold set scored on the same modality and
the same measure reaches MRR 0.221 / hit@10 0.435, on the *easier* criterion of
several acceptable symbols per query. Mined prose is, if anything, a slightly better
retrieval query than a hand-written question. That is a statement about both sets, and
it is the one number here comparable to existing work in this repo.

The two gold sets barely overlap: only 5 of the hand set's 23 symbols got a mined
label. They are complementary rather than alternative, and no per-symbol agreement
between them can be computed from five cases.

## Four limits that bound every number above

1. **The label is not independent of the source.** One author wrote the docstring and
   the commit message in the same sitting. A docstring-reading generator therefore
   scores high partly through shared authorship, not through inference. Two things
   respond to this: labels found verbatim in their own symbol's source are rejected
   outright (2 of 44 on swarm-sync), and `score_purposes` runs a `docstring_blind`
   condition -- which costs the body-identifier generator a third of its lift (0.161
   -> 0.089). All 42 usable-labelled symbols in swarm-sync have a docstring, so this
   is not a corner case on this corpus, it is the whole corpus.

2. **A mention is not a purpose statement.** For a symbol introduced by a bug-fix
   commit, the prose that names it often describes the bug rather than the symbol's
   job. Those labels are kept -- filtering them would take the judgement the eval is
   supposed to be measuring -- and they are the main reason absolute similarity stays
   low even for a good generator.

3. **42 labels are not 42 independent measurements.** They come from 17 distinct
   commits, and one commit supplies 9 of them. Treat differences of a few points as
   noise, exactly as `ablation.py` says of its 16 queries.

4. **Token-F1 is a weak similarity.** It is deterministic and model-free, which is why
   the tests use it, and it rewards vocabulary overlap that carries no meaning -- it
   cannot tell "opens the connection" from "closes the connection" (there is a test
   pinning that). The shuffled control is the answer: it is the score available from
   vocabulary alone, and real signal has to clear it. Read the *gap*, never the score.
"""
from __future__ import annotations

import math
import random
import re
import subprocess
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from ..ingest.indexer import is_test_path, iter_python_files
from ..ingest.python_extract import extract_file
from ..ingest.types import KIND_CLASS, KIND_FUNCTION, KIND_METHOD, Symbol

# Symbol kinds a purpose label is meaningful for. Modules are excluded: a module's
# introducing commit is by definition the commit that created the file, so every
# module label is the file-add message and carries no per-symbol information.
LABELLED_KINDS = frozenset({KIND_CLASS, KIND_FUNCTION, KIND_METHOD})

# Attribution methods, reported per label so a caller can stratify by them.
METHOD_LINE_LOG = "line-log"
METHOD_FILE_ADD = "file-add"

# Rejection reasons. Every mined candidate carries exactly one or is usable; the
# counts form the funnel in `format_report`. Reasons are values rather than booleans
# because "how many did we lose and to what" is the result, not a diagnostic.
REJECT_NO_PROVENANCE = "no_provenance"
REJECT_EMPTY_PROSE = "empty_prose"
REJECT_BOILERPLATE = "boilerplate"
REJECT_NO_MENTION = "no_mention"
REJECT_TOO_SHORT = "too_short"
# The label prose is copied verbatim into the symbol's own source. Not a harness bug
# -- the author wrote the docstring and the commit message in one sitting and reused a
# clause -- but such a label is not HELD OUT: the answer is sitting in the input, and
# a docstring-copying generator scores 1.0 on it for no reason. Found by running the
# leak check on swarm-sync, where it fires on real symbols.
REJECT_COPIED_INTO_SOURCE = "copied_into_source"

# A trailer block at the END of a commit message: `Key: value` lines, possibly
# several. Stripped because `Co-Authored-By: Claude ...` is in most commits in both
# repos here and would contribute the same tokens to every label -- which inflates
# any overlap-based similarity uniformly, including the shuffled control, and so
# quietly compresses the gap the eval is trying to see.
_TRAILER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:[ \t]")

# Subjects that say nothing about purpose. Matched against the whole subject after
# stripping a conventional-commit prefix, so `fix(leases): wip` is caught too.
_BOILERPLATE_SUBJECT = re.compile(
    r"^(wip|fixup|squash|fix|fixes|fixed|bug ?fix|update|updates|updated|cleanup|clean ?up"
    r"|refactor|refactoring|misc|minor|tweak|tweaks|nit|nits|typo|typos|lint|format"
    r"|formatting|style|rename|address (review|comments|feedback)|review (fixes|feedback)"
    r"|pr feedback|comments|bump|bump (version|deps|dependencies)|version bump|release"
    r"|initial commit|first commit|init|initial|checkpoint|save|temp|tmp|test|tests"
    r"|more tests|add tests|docs|doc|documentation|readme|chore|revert.*|merge.*"
    r"|v?\d+(\.\d+)*)$",
    re.IGNORECASE,
)

# Conventional-commit prefix: `fix:`, `feat(scope):`, `WP4.5:`. Removed before the
# boilerplate check so the check sees the actual sentence.
_CONVENTIONAL_PREFIX = re.compile(r"^[A-Za-z][\w.]*(\([^)]*\))?!?:\s*")

# Bullet markers, so a bullet-list body splits into one unit per bullet rather than
# one unit for the whole block. Measured on swarm-sync: bodies in the WP-series
# commits are bullet lists where each bullet is about a different module, so
# splitting on sentences alone produces labels four modules wide.
_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+")

# Sentence boundary: `. ` / `! ` / `? ` / newline-newline, not preceded by a
# single-letter initial or a common abbreviation. Deliberately crude -- prose in
# commit messages is full of `e.g.` and `db.py`, and a perfect splitter is not worth
# the dependency.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(—-])")

MIN_LABEL_WORDS = 6
MAX_LABEL_UNITS = 3

# Length of shared text, whitespace-normalised, that counts as COPIED rather than
# coincidental. 32 characters is five or six words -- a clause, which is longer than
# two people describing the same function land on by accident.
#
# The threshold turns out not to be a delicate choice: on swarm-sync the copy filter
# rejects the SAME 2 labels at 24, 32, 48 and 64 characters. Copying here is not a
# borderline phenomenon -- either the author moved a whole sentence from the commit
# message into the docstring or they wrote independently, with nothing in between.
COPY_RUN_CHARS = 32

# Tokens too generic to carry meaning in a similarity score. Kept short and English
# plus the handful of software words that appear in nearly every commit body here;
# a long stoplist starts encoding the answer.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how if in into is it its more must no not of on one only or other our out over
    own same so some such than that the their them then there these they this those
    to too under up use used uses using was we were what when where which while who
    why will with would you your it's don't
    now new also both each every not
    """.split()
)


class LeakDetected(Exception):
    """The generator's input contained text from the held-out label.

    Raised rather than logged. A leak does not degrade the measurement, it voids it,
    and a voided measurement that still prints a number is worse than a crash.
    """


@dataclass(frozen=True)
class SourceView:
    """Everything a purpose generator is allowed to see: source, and nothing else.

    Frozen, and built only by `source_view()`, which reads the working tree. There is
    deliberately no `commit`, no `message`, and no `provenance` field -- the boundary
    is the absence of a place to put them, not a rule about not looking.
    """

    qualname: str
    kind: str
    path: str
    line_start: int
    line_end: int
    signature: str | None
    docstring: str | None
    source: str

    def without_docstring(self) -> SourceView:
        """The same view with the docstring removed from every field it appears in.

        The harder condition, and the more honest one for a generator that claims to
        *infer* purpose rather than relay it. On swarm-sync ALL 42 usable-labelled
        symbols have a docstring, so without this condition every reported number
        would be measuring how well the author documented their own code.
        """
        doc = (self.docstring or "").strip()
        source = self.source
        if doc:
            # Strip the docstring literal, not just the text, so the triple quotes do
            # not leave the body syntactically odd for a generator that parses it.
            source = re.sub(
                r'("""|\'\'\')' + re.escape(doc) + r'\1',
                '""""""',
                source,
                count=1,
            )
            if doc in source:
                source = source.replace(doc, "", 1)
        return SourceView(
            qualname=self.qualname,
            kind=self.kind,
            path=self.path,
            line_start=self.line_start,
            line_end=self.line_end,
            signature=self.signature,
            docstring=None,
            source=source,
        )


@dataclass(frozen=True)
class Provenance:
    """The commit a symbol's lines came from. NEVER handed to a generator."""

    sha: str
    subject: str
    body: str
    method: str
    files_touched: int


@dataclass(frozen=True)
class MinedLabel:
    """One candidate gold label, usable or not.

    Rejected candidates are kept for the same reason the assertion store keeps
    rejected claims: a miner that discards what it filtered can report any yield it
    likes. `reject` is None exactly when the label is usable.
    """

    qualname: str
    kind: str
    path: str
    prose: str
    commit: str
    subject: str
    method: str
    files_touched: int
    units: int
    reject: str | None = None

    @property
    def usable(self) -> bool:
        return self.reject is None


@dataclass
class MineReport:
    """The funnel. `usable_fraction` is the headline number of this module."""

    repo: str
    considered: int = 0
    commits_in_history: int = 0
    labels: list[MinedLabel] = field(default_factory=list)

    @property
    def usable(self) -> list[MinedLabel]:
        return [lab for lab in self.labels if lab.usable]

    @property
    def usable_fraction(self) -> float:
        return len(self.usable) / self.considered if self.considered else 0.0

    def rejects(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lab in self.labels:
            if lab.reject:
                counts[lab.reject] = counts.get(lab.reject, 0) + 1
        return counts

    def label_sharing(self) -> dict[str, int]:
        """How many usable labels each introducing commit accounts for.

        Attribution dilution, made visible: two symbols whose label came from the
        same commit are not two independent measurements of that commit's prose.
        """
        counts: dict[str, int] = {}
        for lab in self.usable:
            counts[lab.commit] = counts.get(lab.commit, 0) + 1
        return counts


# --------------------------------------------------------------------------------
# Mining
# --------------------------------------------------------------------------------


def _git(repo: Path, *args: str, timeout: int = 60) -> str | None:
    """Run a read-only git command in `repo`. None on any failure.

    Returns None rather than raising because a repo with no history, a path that is
    not a repo, and a git that is not installed are all "there is no label here",
    and the funnel should record that as a rejection instead of aborting the run.
    """
    try:
        # S603/S607: `git` from PATH on purpose (see indexer._git_tracked_python_files).
        # Fixed argument vector, no shell. Every subcommand used here is read-only.
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "-c", "log.showSignature=false", *args],  # noqa: S607
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def commit_count(repo: Path) -> int:
    out = _git(repo, "rev-list", "--count", "HEAD")
    return int(out.strip()) if out and out.strip().isdigit() else 0


# Record/field separators for `--format`. Chosen from the C0 controls because commit
# prose can contain anything printable, including every punctuation character.
_FS = "\x01"
_RS = "\x02"
_LOG_FORMAT = f"%H{_FS}%s{_FS}%B{_RS}"


def _parse_log(out: str) -> list[tuple[str, str, str]]:
    records = []
    for raw in out.split(_RS):
        if not raw.strip():
            continue
        parts = raw.lstrip("\n").split(_FS)
        if len(parts) >= 3:
            records.append((parts[0].strip(), parts[1], _FS.join(parts[2:])))
    return records


def introducing_commit(
    repo: Path, path: str, line_start: int, line_end: int
) -> Provenance | None:
    """The commit that wrote a symbol's lines, via git's line log.

    **Why `git log -L<a>,<b>:<file> --reverse` and not `--diff-filter=A -- <file>`.**
    File-add attribution gives every symbol in a file the same commit, so a symbol
    added to an existing module gets the message that introduced the module -- prose
    about a different piece of code. The line log traces the range backwards through
    history and its earliest entry is the commit that actually wrote those lines.
    Measured on swarm-sync: the two disagree for 83 of 316 symbols (26%), and in
    every one of those the line log is the later, more specific commit.

    File-add remains the fallback for the case the line log cannot answer -- an
    untracked file, or a range git will not follow -- and the choice is recorded in
    `Provenance.method` so results can be stratified rather than pooled.

    Two limits worth stating. The line log follows the range as it is TODAY, so for a
    symbol later rewritten in place, the earliest commit touching that range may be
    one that only altered a neighbouring line inside the span. And it does not follow
    a symbol across a file rename by default, which attributes a moved symbol to its
    move commit.
    """
    out = _git(
        repo,
        "log",
        "--reverse",
        "-s",
        f"--format={_LOG_FORMAT}",
        f"-L{line_start},{line_end}:{path}",
    )
    method = METHOD_LINE_LOG
    records = _parse_log(out) if out else []
    if not records:
        out = _git(
            repo,
            "log",
            "--diff-filter=A",
            "--reverse",
            f"--format={_LOG_FORMAT}",
            "--",
            path,
        )
        method = METHOD_FILE_ADD
        records = _parse_log(out) if out else []
    if not records:
        return None
    sha, subject, body = records[0]
    return Provenance(
        sha=sha,
        subject=subject.strip(),
        body=body,
        method=method,
        files_touched=_files_touched(repo, sha),
    )


def _files_touched(repo: Path, sha: str) -> int:
    """How many files the introducing commit changed.

    Not a filter -- a wide commit can still name a symbol and say why it exists --
    but it is the single best predictor of how diluted a label is, so it is recorded
    and reported.
    """
    out = _git(repo, "show", "--name-only", "--format=", "--first-parent", sha)
    if not out:
        return 0
    return len([n for n in out.split("\n") if n.strip()])


def strip_trailers(message: str) -> str:
    """The prose of a commit message: no trailer block, no diff-stat noise."""
    lines = message.rstrip().split("\n")
    while lines:
        last = lines[-1]
        if not last.strip() or _TRAILER_LINE.match(last.strip()):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def is_boilerplate(subject: str) -> bool:
    """Whether a subject line says nothing about why any code exists."""
    stripped = _CONVENTIONAL_PREFIX.sub("", subject.strip()).strip()
    stripped = stripped.rstrip(".").strip()
    if not stripped:
        return True
    if _BOILERPLATE_SUBJECT.match(stripped):
        return True
    # Two words or fewer cannot state a purpose. `fix reaper` is not a label.
    return len(stripped.split()) <= 2


def split_units(prose: str) -> list[str]:
    """Split prose into bullets and sentences -- the granularity a label is chosen at.

    A commit body here is typically several paragraphs about several modules. The
    unit, not the message, is what gets matched against a symbol name, because a
    whole body as a label is a label for the commit and not for the symbol.
    """
    units: list[str] = []
    for block in re.split(r"\n[ \t]*\n", prose):
        block = block.strip()
        if not block:
            continue
        pieces = [p for p in _BULLET.split(block) if p.strip()]
        for piece in pieces:
            flat = " ".join(piece.split())
            units.extend(s.strip() for s in _SENTENCE_SPLIT.split(flat) if s.strip())
    return units


def _identifier_is_distinctive(name: str) -> bool:
    """Whether a bare occurrence of `name` in prose must be a code reference.

    `_write_settings_atomically` and `BlackboardUnreachable` cannot be English.
    `money` can be, and is: swarm-sync's initial commit says "all 5 demo money shots
    verified", which a bare substring match reads as prose about
    `sample_repo.formats.money`. That false attribution is what this rules out.
    """
    if "_" in name.strip("_"):
        return True
    core = name.strip("_")
    return bool(core[1:] and any(c.isupper() for c in core[1:]))


def mentions_symbol(unit: str, name: str) -> bool:
    """Whether `unit` refers to the symbol `name` *as code*.

    Either the identifier is distinctive enough that a bare occurrence cannot be
    English, or it occurs in an unmistakably code-ish context: in backticks, called,
    attribute-accessed, or assigned. Requiring one of the two is what turns
    commit-wide prose into a symbol-specific label, and it is also the filter that
    rejects the great majority of candidates.
    """
    # The lookbehind excludes an identifier character but NOT a dot: `events.tail` is
    # the single most common way a commit body refers to a method, and excluding it --
    # as the first version of this did -- silently discarded every qualified mention.
    # Found by a test, not by reading. The cost is that a mention of some OTHER
    # module's `tail` also matches, which is the same basename ambiguity the tier-1
    # resolver lives with.
    bare = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    if not bare.search(unit):
        return False
    if _identifier_is_distinctive(name):
        return True
    code_ish = re.compile(
        r"(?:`[^`]*\b" + re.escape(name) + r"\b[^`]*`)"       # `name`, `mod.name(x)`
        r"|(?:\b" + re.escape(name) + r"\s*\()"                 # name(
        r"|(?:\.\s*" + re.escape(name) + r"\b)"                 # mod.name
        r"|(?:\b" + re.escape(name) + r"\s*=[^=])"              # name=
    )
    return bool(code_ish.search(unit))


def extract_label(message: str, name: str) -> tuple[str, int]:
    """The units of a commit message that refer to `name`, joined. ("", 0) if none.

    The subject is included when it mentions the symbol, and only then: a subject
    that does not name the symbol is prose about the commit.
    """
    prose = strip_trailers(message)
    if not prose:
        return "", 0
    units = split_units(prose)
    hits = [u for u in units if mentions_symbol(u, name)]
    if not hits:
        return "", 0
    chosen = hits[:MAX_LABEL_UNITS]
    return " ".join(chosen), len(hits)


def repo_symbols(
    repo: Path, include_tests: bool = False
) -> Iterator[tuple[str, Symbol]]:
    """(relative path, symbol) for every labellable symbol in a repo's working tree.

    Reads the working tree only -- no index, no git. A caller with an index can pass
    its own symbols to `mine_labels` instead.
    """
    for file_path in iter_python_files(repo):
        rel = file_path.relative_to(repo).as_posix()
        if not include_tests and is_test_path(rel):
            continue
        try:
            extract = extract_file(file_path, repo)
        except OSError:
            continue
        for sym in extract.symbols:
            if sym.kind in LABELLED_KINDS:
                yield rel, sym


def mine_labels(
    repo: Path,
    symbols: Iterable[tuple[str, Symbol]] | None = None,
    include_tests: bool = False,
) -> MineReport:
    """Mine a purpose label per symbol from the commit that introduced it.

    Every candidate lands in the report, usable or rejected with a reason. The
    rejected ones are the measurement: `usable_fraction` is only interpretable next
    to what it lost and why.
    """
    repo = Path(repo)
    report = MineReport(repo=str(repo), commits_in_history=commit_count(repo))
    pairs = list(symbols) if symbols is not None else list(repo_symbols(repo, include_tests))
    file_bytes: dict[str, bytes] = {}
    for rel, sym in pairs:
        report.considered += 1
        prov = introducing_commit(repo, rel, sym.line_start, sym.line_end)
        if prov is None:
            report.labels.append(_candidate(rel, sym, None, reject=REJECT_NO_PROVENANCE))
            continue
        if not strip_trailers(prov.body):
            report.labels.append(_candidate(rel, sym, prov, reject=REJECT_EMPTY_PROSE))
            continue
        if is_boilerplate(prov.subject):
            # Checked before the mention rule so the funnel separates "the commit
            # said nothing" from "the commit said something, but not about this".
            report.labels.append(_candidate(rel, sym, prov, reject=REJECT_BOILERPLATE))
            continue
        prose, units = extract_label(prov.body, sym.name)
        if not prose:
            report.labels.append(_candidate(rel, sym, prov, reject=REJECT_NO_MENTION))
            continue
        if len(prose.split()) < MIN_LABEL_WORDS:
            report.labels.append(
                _candidate(rel, sym, prov, prose, units, reject=REJECT_TOO_SHORT)
            )
            continue
        if rel not in file_bytes:
            try:
                file_bytes[rel] = (repo / rel).read_bytes()
            except OSError:
                file_bytes[rel] = b""
        # Sliced on BYTE offsets, decoded after -- the extractor's spans are byte
        # spans, and slicing a str by them silently shifts on any file with a non-ASCII
        # character. swarm-sync's prose is full of em dashes, so this is not academic.
        symbol_source = file_bytes[rel][sym.byte_start : sym.byte_end].decode(
            "utf-8", "replace"
        )
        if find_leaks(symbol_source, [prose]):
            # The answer is in the input. Dropped rather than scored: see
            # REJECT_COPIED_INTO_SOURCE.
            report.labels.append(
                _candidate(rel, sym, prov, prose, units, reject=REJECT_COPIED_INTO_SOURCE)
            )
            continue
        report.labels.append(_candidate(rel, sym, prov, prose, units))
    return report


def _candidate(
    rel: str,
    sym: Symbol,
    prov: Provenance | None,
    prose: str = "",
    units: int = 0,
    reject: str | None = None,
) -> MinedLabel:
    """One row of the funnel, usable or rejected.

    A free function rather than a closure over the mining loop: a closure that
    captures the loop variable is a live foot-gun the moment anything defers the call,
    and ruff's B023 is right to say so.
    """
    return MinedLabel(
        qualname=sym.qualname,
        kind=sym.kind,
        path=rel,
        prose=prose,
        commit=prov.sha if prov else "",
        subject=prov.subject if prov else "",
        method=prov.method if prov else "",
        files_touched=prov.files_touched if prov else 0,
        units=units,
        reject=reject,
    )


# --------------------------------------------------------------------------------
# The leak boundary
# --------------------------------------------------------------------------------


def source_view(repo: Path, path: str, sym: Symbol) -> SourceView:
    """Build the generator's input from the working tree.

    This function must never touch git, and the test that proves it makes
    `subprocess.run` raise before calling it. That is a stronger guarantee than
    review: any future edit that reaches for a commit message here fails a test
    rather than quietly improving the scores.
    """
    raw = (Path(repo) / path).read_bytes()
    return SourceView(
        qualname=sym.qualname,
        kind=sym.kind,
        path=path,
        line_start=sym.line_start,
        line_end=sym.line_end,
        signature=sym.signature,
        docstring=sym.docstring,
        source=raw[sym.byte_start : sym.byte_end].decode("utf-8", "replace"),
    )


def _walk_strings(obj: object, seen: set[int] | None = None) -> Iterator[tuple[str, str]]:
    """Every string reachable from `obj`, with the path that reached it.

    Recursive rather than a `repr()` scan on purpose: `repr` truncates, and a leak
    hidden behind a lazy property or a nested container is exactly the leak a repr
    scan would miss.
    """
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    stack: list[tuple[str, object]] = [("", obj)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, str):
            yield path or "<root>", node
        elif isinstance(node, (bytes, bytearray)):
            yield path or "<root>", node.decode("utf-8", "replace")
        elif isinstance(node, (int, float, bool, type(None))):
            continue
        elif isinstance(node, dict):
            for key, value in node.items():
                stack.append((f"{path}[{key!r}]", key))
                stack.append((f"{path}[{key!r}]", value))
        elif isinstance(node, (list, tuple, set, frozenset)):
            for i, value in enumerate(node):
                stack.append((f"{path}[{i}]", value))
        elif is_dataclass(node) and not isinstance(node, type):
            for f in fields(node):
                stack.append((f"{path}.{f.name}", getattr(node, f.name, None)))
        elif hasattr(node, "__dict__"):
            for key, value in vars(node).items():
                stack.append((f"{path}.{key}", value))


def find_leaks(
    view: object, secrets: Sequence[str], min_run: int = COPY_RUN_CHARS
) -> list[str]:
    """Paths within `view` where any of `secrets` is reachable.

    Matching is on a normalised whitespace-collapsed substring of at least `min_run`
    characters, so a label and a docstring that merely share a word do not read as a
    leak while a copied clause does. See `COPY_RUN_CHARS` for why the default is 32.
    """
    needles = []
    for secret in secrets:
        flat = " ".join(secret.split())
        if len(flat) >= min_run:
            needles.append(flat)
    if not needles:
        return []
    hits: list[str] = []
    for path, text in _walk_strings(view):
        flat = " ".join(text.split())
        if len(flat) < min_run:
            continue
        for needle in needles:
            # Any window of the needle long enough to be a clause.
            for start in range(0, max(1, len(needle) - min_run + 1)):
                window = needle[start : start + min_run]
                if window in flat:
                    hits.append(f"{path}: {window!r}")
                    break
            else:
                continue
            break
    return hits


def assert_no_leak(view: object, secrets: Sequence[str]) -> None:
    """Raise `LeakDetected` if any held-out prose is reachable from `view`.

    The second of the two gates, and the weaker one: it is a text search, so it fires
    on a clause the author happened to copy from the commit message into the
    docstring as readily as on a harness bug that routed the label into the input.
    Those cases are removed from the corpus at mining time
    (`REJECT_COPIED_INTO_SOURCE`), which is what makes this gate meaningful on a
    mined set -- anything it still finds is not authorship, it is plumbing.
    """
    hits = find_leaks(view, secrets)
    if hits:
        raise LeakDetected(
            f"{len(hits)} leak(s) of held-out label prose into generator input: "
            + "; ".join(hits[:3])
        )


def assert_view_is_source_only(repo: Path, view: SourceView) -> None:
    """Raise `LeakDetected` unless every byte of `view` came out of the working tree.

    The primary gate, and the reason it is structural rather than a text search:
    commit prose that a text search would flag can legitimately be in the source (an
    author quoting their own commit message in a docstring), while a harness bug that
    put a commit message into the view produces text that is *not in the file* --
    which is exactly what this checks and what no substring search can distinguish.

    Concretely: `view.source` must be the file's bytes at the symbol's span, and the
    docstring and signature must occur inside those bytes. There is no field on
    `SourceView` for anything else, so a view that passes this has no room left to
    carry a label.
    """
    file_path = Path(repo) / view.path
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LeakDetected(f"{view.qualname}: view cannot be checked -- {exc}") from exc
    if view.source not in raw:
        raise LeakDetected(
            f"{view.qualname}: view.source is not a substring of {view.path} "
            "-- something other than the working tree wrote it"
        )
    for field_name in ("docstring", "signature"):
        value = getattr(view, field_name)
        if value and " ".join(value.split()) not in " ".join(raw.split()):
            raise LeakDetected(
                f"{view.qualname}: view.{field_name} is not present in {view.path}"
            )


def audit_leak_boundary(
    repo: Path, labels: Sequence[MinedLabel]
) -> tuple[int, list[str]]:
    """Run both gates over a whole mined set, cross-checking EVERY label against EVERY view.

    `score_purposes` checks a view against its own label, which is the leak that
    would matter. This checks the full cross product, which catches the subtler
    version: a harness that built views from the wrong symbols, or a `path` field
    pointing somewhere that happens to contain another symbol's held-out prose.
    Returns (views checked, findings) rather than raising, because this is a report.
    """
    prose = [lab.prose for lab in labels]
    findings: list[str] = []
    checked = 0
    for lab in labels:
        try:
            extract = extract_file(Path(repo) / lab.path, Path(repo))
        except OSError as exc:
            findings.append(f"{lab.qualname}: {exc}")
            continue
        sym = next((s for s in extract.symbols if s.qualname == lab.qualname), None)
        if sym is None:
            continue
        view = source_view(Path(repo), lab.path, sym)
        checked += 1
        try:
            assert_view_is_source_only(Path(repo), view)
        except LeakDetected as exc:
            findings.append(str(exc))
        findings.extend(f"{lab.qualname}: {hit}" for hit in find_leaks(view, prose))
    return checked, findings


def suspect_tokens(inferred: str, label: str, view: SourceView) -> list[str]:
    """Rare tokens shared by the output and the held-out label but absent from the input.

    The other direction of the boundary check. A generator can only legitimately
    produce vocabulary it was shown (or generic English); a distinctive word that
    appears in the answer and in the model's output but nowhere in what it was given
    is either a coincidence worth knowing about or a leak worth stopping.

    Counted and reported rather than raised: short tokens collide constantly and a
    hard failure on them would be noise. `assert_no_leak` is the hard gate.
    """
    seen = set(_tokens(" ".join(str(v) for v in vars(view).values())))
    out = set(_tokens(inferred))
    answer = set(_tokens(label))
    return sorted(t for t in (out & answer) - seen if len(t) >= 6)


# --------------------------------------------------------------------------------
# Generators -- source-only, by construction
# --------------------------------------------------------------------------------

Generator = Callable[[SourceView], str]


def _split_identifier(name: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    words: list[str] = []
    for part in parts:
        words.extend(w for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", part))
    return [w.lower() for w in words if w]


def _tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    """Content words, with identifiers split into their parts.

    `bind_managed_root` and "binds the managed root" have to tokenise to the same
    thing or no similarity between prose and code can ever be non-zero.
    """
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        out.extend(_split_identifier(raw))
    if drop_stopwords:
        out = [t for t in out if t not in _STOPWORDS and len(t) > 1]
    return out


def docstring_purpose(view: SourceView) -> str:
    """The docstring's first sentence, falling back to the name and signature.

    The strongest source-only baseline and NOT an inference system: it copies the
    documentation the author already wrote. It is here as the upper reference -- a
    real generator that cannot beat a docstring copier on documented symbols is not
    earning its forward pass, and one that scores well only where docstrings exist
    has not been tested on the case that matters.
    """
    doc = (view.docstring or "").strip()
    if doc:
        first = _SENTENCE_SPLIT.split(" ".join(doc.split()))
        return first[0] if first else doc
    return name_purpose(view)


def name_purpose(view: SourceView) -> str:
    """The symbol's name and signature as words. The floor.

    Anything that cannot beat this is not reading the code. Reported because a
    similarity metric that puts this level with a real generator is a metric with no
    resolution, which is a finding about the metric rather than about the generator.
    """
    words = _split_identifier(view.qualname.rsplit(".", 1)[-1])
    if view.signature:
        params = re.search(r"\(([^)]*)\)", view.signature)
        if params:
            for part in params.group(1).split(","):
                words.extend(_split_identifier(part.split(":")[0].split("=")[0]))
    return " ".join(words)


def body_purpose(view: SourceView) -> str:
    """Identifiers used in the body, as words. A source-only bag-of-code baseline.

    Deliberately dumb: it has no idea what the symbol is FOR. It exists to show what
    part of any score is available from vocabulary co-occurrence alone, without any
    understanding -- the same role the shuffled control plays, from the other side.
    """
    return " ".join(dict.fromkeys(_tokens(view.source)))


# --------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------


def token_f1(a: str, b: str) -> float:
    """Harmonic mean of token precision and recall, on content words. In [0, 1].

    Deterministic and model-free, which is why the tests use it. It measures
    vocabulary agreement and nothing else: `token_f1` cannot tell "opens a WAL
    connection" from "closes a WAL connection". Read it as a ranking of conditions,
    never as a percentage of correctness.
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if not overlap:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(embedder: object) -> Callable[[str, str], float]:
    """Cosine similarity from an `Embedder`. Better than token-F1, less reproducible.

    The right measure for this job -- it can tell a paraphrase from a word-overlap
    coincidence, which is most of what separates a good purpose statement from a bad
    one. It is not the default because its numbers move with the model version, and
    an eval whose absolute values shift under a dependency bump cannot be compared
    across runs.
    """

    def similarity(a: str, b: str) -> float:
        vecs = embedder.encode_documents([a, b])  # type: ignore[attr-defined]
        va, vb = list(vecs[0]), list(vecs[1])
        dot = sum(x * y for x, y in zip(va, vb, strict=True))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        return dot / (na * nb) if na and nb else 0.0

    return similarity


def _derangement(n: int, seed: int = 20250729) -> list[int]:
    """A permutation with no fixed point, deterministic for a given n and seed.

    The shuffled control needs every view paired with a label that is NOT its own.
    A rotation would do it in one line but pairs each symbol with its file
    neighbour, and neighbours share vocabulary -- which would overstate the control
    and understate the signal.
    """
    if n < 2:
        return list(range(n))
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    order = list(range(n))
    for _ in range(1000):
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            return order
    return [(i + 1) % n for i in range(n)]


@dataclass
class PurposeScorecard:
    """One condition's result: the score, its control, and the gap between them.

    `gold` alone is uninterpretable. A generator emitting the twenty most common
    words in the repo's commit log scores well above zero against any label, so the
    number that carries information is `lift` -- how much better the generator does
    against the RIGHT label than against a wrong one.
    """

    name: str
    scores: list[float] = field(default_factory=list)
    control: list[float] = field(default_factory=list)
    suspect: int = 0
    empty_output: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def gold(self) -> float:
        return _mean(self.scores)

    @property
    def shuffled(self) -> float:
        return _mean(self.control)

    @property
    def lift(self) -> float:
        return self.gold - self.shuffled

    def row(self) -> str:
        return (
            f"{self.name:<30} {self.n:>4} {self.gold:>7.3f} {self.shuffled:>9.3f} "
            f"{self.lift:>7.3f} {self.suspect:>8} {self.empty_output:>6}"
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_purposes(
    repo: Path,
    labels: Sequence[MinedLabel],
    generator: Generator,
    name: str,
    similarity: Callable[[str, str], float] = token_f1,
    docstring_blind: bool = False,
    name_blind: bool = True,
) -> PurposeScorecard:
    """Score a generator's inferred purpose against held-out commit prose.

    The leak boundary is enforced here, on every symbol, before any score is
    computed: the view is built from the working tree by `source_view`, checked
    against the label prose by `assert_no_leak`, and only then passed to the
    generator. A generator that somehow received the prose would raise rather than
    produce a good number.

    `name_blind` removes the symbol's own name tokens from both sides, and the default
    is not a detail. The mention rule selected each label BECAUSE it contains the
    symbol's name, so the name is the one token every generator is guaranteed to share
    with every label. Measured on swarm-sync: scored raw, `name + signature only`
    reaches 0.210 against `docstring first sentence`'s 0.225 -- i.e. the metric cannot
    tell a name echo from reading the documentation, and a generator that printed the
    function's own name would look like a working purpose inferrer. Name-blinding drops
    the echo to 0.020 while the docstring condition keeps 0.159, which is the
    resolution the eval needs to be worth running.
    """
    card = PurposeScorecard(name=name)
    by_path: dict[str, dict[str, Symbol]] = {}
    views: list[SourceView] = []
    kept: list[MinedLabel] = []
    for lab in labels:
        if lab.path not in by_path:
            try:
                extract = extract_file(Path(repo) / lab.path, Path(repo))
            except OSError:
                by_path[lab.path] = {}
            else:
                by_path[lab.path] = {s.qualname: s for s in extract.symbols}
        sym = by_path[lab.path].get(lab.qualname)
        if sym is None:
            continue
        view = source_view(Path(repo), lab.path, sym)
        # Both gates, on every symbol, before anything is scored. The structural one
        # is the guarantee; the text one is the belt. `audit_leak_boundary` runs the
        # same pair across the full label x view cross product.
        assert_view_is_source_only(Path(repo), view)
        assert_no_leak(view, [lab.prose])
        if docstring_blind:
            view = view.without_docstring()
        views.append(view)
        kept.append(lab)

    if not views:
        return card

    order = _derangement(len(views))
    for i, (view, lab) in enumerate(zip(views, kept, strict=True)):
        inferred = generator(view)
        if not inferred.strip():
            card.empty_output += 1
        blind = view.qualname.rsplit(".", 1)[-1] if name_blind else None
        card.scores.append(similarity(_blind(inferred, blind), _blind(lab.prose, blind)))
        other = kept[order[i]]
        card.control.append(
            similarity(_blind(inferred, blind), _blind(other.prose, blind))
        )
        card.suspect += len(suspect_tokens(inferred, lab.prose, view))
    return card


def _blind(text: str, name: str | None) -> str:
    """Remove a symbol name's tokens from `text`, for name-blind scoring."""
    if not name:
        return text
    drop = set(_split_identifier(name))
    return " ".join(t for t in _tokens(text, drop_stopwords=False) if t not in drop)


#: The four conditions worth reporting, and why each is in the table.
#:
#: `docstring_purpose` on a doc-blind view is deliberately ABSENT: it degenerates to
#: `name_purpose` by construction (that is its fallback), so the row would duplicate
#: the floor and read as an independent measurement.
DEFAULT_CONDITIONS: tuple[tuple[str, Generator, bool], ...] = (
    ("docstring first sentence", docstring_purpose, False),
    ("name + signature only", name_purpose, False),
    ("body identifiers", body_purpose, False),
    ("body identifiers, doc-blind", body_purpose, True),
)


def run_purpose_eval(
    repo: Path,
    similarity: Callable[[str, str], float] = token_f1,
    include_tests: bool = False,
    conditions: Sequence[tuple[str, Generator, bool]] = DEFAULT_CONDITIONS,
) -> tuple[MineReport, list[PurposeScorecard]]:
    """Mine a repo's history and score every default condition against it."""
    report = mine_labels(repo, include_tests=include_tests)
    usable = report.usable
    cards = [
        score_purposes(
            repo, usable, generator, name, similarity=similarity, docstring_blind=blind
        )
        for name, generator, blind in conditions
    ]
    return report, cards


@dataclass
class LabelValidity:
    """Whether a mined label is *about* its symbol, checked without a judge.

    The independent validity test, and the one that does not depend on any generator:
    feed each label's prose to the retrieval pipeline as a query and see whether it
    finds the symbol it was mined from. Prose that describes a symbol should retrieve
    it; prose that describes a whole work package should not, and will land on
    whichever of the commit's symbols the vocabulary happens to favour.

    This is measurable against the hand-labelled gold set on the same modality and at
    the same k, which makes it the one number in this module that can be compared to
    existing work in the repo rather than only to its own control.
    """

    n: int = 0
    ranks: list[int | None] = field(default_factory=list)

    @property
    def mrr(self) -> float:
        return _mean([1.0 / r if r else 0.0 for r in self.ranks])

    def hit_at(self, k: int) -> float:
        return _mean([1.0 if (r and r <= k) else 0.0 for r in self.ranks])


def label_retrieval_validity(
    conn: object, labels: Sequence[MinedLabel], k: int = 10, name_blind: bool = True
) -> LabelValidity:
    """Use each label's prose as a query; record the rank of the symbol it labels.

    Lexical retrieval only, deliberately. It needs no model, so this number is
    reproducible on any machine, and it is the modality the ablation reports first --
    so `LabelValidity.mrr` here and the `lexical only` row there are the same measure
    on two different gold sets.

    **`name_blind` defaults to True and the default is the whole point.** Every mined
    label contains the symbol's name -- the mention rule is what selected it -- and
    FTS5 will happily match that name against the symbol's own chunk. Scored raw, this
    function largely measures "does searching for a name find the thing with that
    name", which is true and worthless. Blinding removes the name's tokens from the
    query, so what is left is whether the *description* finds the symbol.

    Two further asymmetries with the ablation, both of which make these numbers NOT
    interchangeable with its rows: there, one query has several relevant symbols and
    any of them counts; here, one label has exactly one correct symbol. And a mined
    label is a paragraph of the author's own vocabulary, where a gold query is a short
    question a reader would type -- longer queries give lexical retrieval more to
    match on.
    """
    from ..retrieve.lexical import search_lexical  # local: keeps import graph flat

    out = LabelValidity()
    for lab in labels:
        name = lab.qualname.rsplit(".", 1)[-1] if name_blind else None
        query = _blind(lab.prose, name) if name else lab.prose
        hits = search_lexical(conn, query, k=k)  # type: ignore[arg-type]
        rank: int | None = None
        for i, hit in enumerate(hits, start=1):
            if hit.qualname == lab.qualname:
                rank = i
                break
        out.n += 1
        out.ranks.append(rank)
    return out


#: The labelling rule, stated in the same place and the same terms as the hand-labelled
#: gold set's `labelling_rule`. A gold set whose rule is not written down cannot be
#: disagreed with, and a rule that is not next to the labels is not really published.
LABELLING_RULE = (
    "A label is the sentence(s) of the commit that introduced a symbol's lines "
    "(git log -L, earliest entry) which refer to that symbol AS CODE -- either the "
    "identifier is distinctive enough that a bare occurrence cannot be English, or it "
    "occurs in backticks, called, attribute-accessed, or assigned. Commits with "
    "boilerplate subjects are excluded, as are labels shorter than 6 words and labels "
    "found verbatim in the symbol's own source (not held out). The generator sees only "
    "a SourceView built from the working tree; it never sees any commit message. "
    "The label describes why the code was WRITTEN, which for a symbol introduced by a "
    "bug-fix commit is often the bug rather than the symbol's standing purpose -- a "
    "known and unfiltered weakness of the rule."
)


def to_gold_json(report: MineReport, head: str = "") -> dict:
    """The mined set in the shape of `gold/swarm_sync.json`, for human inspection.

    A SNAPSHOT, and labelled as one. The hand gold set is a fixed artifact; a mined set
    is a function of a repo's history and changes with the next commit, so anything
    checked in has to carry the sha it was mined at or it will quietly become a claim
    about code that no longer exists. Nothing in the test suite asserts against a
    snapshot for exactly that reason -- the tests use a purpose-built fixture.
    """
    return {
        # The name, not the absolute path: a shipped artifact that records someone's
        # home directory is a leak of a different kind.
        "repo": Path(report.repo).name,
        "mined_at_head": head,
        "commit_note": (
            f"MINED, not hand-labelled: {len(report.usable)} usable labels from "
            f"{report.considered} symbols ({report.usable_fraction:.1%}) across "
            f"{report.commits_in_history} commits. Snapshot -- regenerate with "
            f"codelearner.eval.mine_labels rather than trusting this file."
        ),
        "labelling_rule": LABELLING_RULE,
        "rejected": report.rejects(),
        "labels": [
            {
                "qualname": lab.qualname,
                "kind": lab.kind,
                "path": lab.path,
                "commit": lab.commit,
                "subject": lab.subject,
                "files_touched": lab.files_touched,
                "attribution": lab.method,
                "purpose": lab.prose,
            }
            for lab in report.usable
        ],
    }


def format_report(report: MineReport, cards: Sequence[PurposeScorecard]) -> str:
    """The funnel and the scorecards, in the shape the ablation table uses."""
    lines = [
        f"repo: {report.repo}",
        f"commits in history: {report.commits_in_history}",
        "",
        "MINING FUNNEL",
        f"  symbols considered           {report.considered:>6}",
    ]
    for reason in (
        REJECT_NO_PROVENANCE,
        REJECT_EMPTY_PROSE,
        REJECT_BOILERPLATE,
        REJECT_NO_MENTION,
        REJECT_TOO_SHORT,
        REJECT_COPIED_INTO_SOURCE,
    ):
        lines.append(f"  rejected: {reason:<20} {report.rejects().get(reason, 0):>6}")
    lines.append(
        f"  USABLE                       {len(report.usable):>6}"
        f"   ({report.usable_fraction:.1%} of considered)"
    )
    sharing = report.label_sharing()
    if sharing:
        widest = max(sharing.values())
        lines += [
            f"  distinct introducing commits {len(sharing):>6}",
            f"  most symbols from one commit {widest:>6}",
        ]
    methods: dict[str, int] = {}
    for lab in report.usable:
        methods[lab.method] = methods.get(lab.method, 0) + 1
    if methods:
        lines.append(
            "  attribution: "
            + ", ".join(f"{m}={n}" for m, n in sorted(methods.items()))
        )
    if cards:
        header = (
            f"{'condition':<30} {'n':>4} {'gold':>7} {'shuffled':>9} {'lift':>7} "
            f"{'suspect':>8} {'empty':>6}"
        )
        lines += ["", "PURPOSE AGREEMENT (name-blind token-F1)", header, "-" * len(header)]
        lines += [c.row() for c in cards]
    return "\n".join(lines)
