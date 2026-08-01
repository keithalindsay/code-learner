"""The generation pipeline: symbols in, admitted tier-2 claims out.

This is the part that was missing. `assertions/store.py` has been able to admit an
evidence-bound claim since Phase 4, and nothing in the repo produced one -- every
assertion in every test was hand-written. So the store's rules were proven and the
path that would actually feed it was not, which is precisely the gap in which a
convenience gets added and quietly defeats them.

The one that would defeat them is not hard to name, because it is the obvious
engineering instinct. A model drafts a claim about `leases.acquire`, cites reference
`[7]` when the menu only went up to five, and the resolved evidence comes back empty.
`write_assertion` then raises `EvidenceRequired` and the run loses a claim that was
probably fine. The repair suggests itself: the subject's own span is right there, it
is known-good, and attaching it would turn a lost claim into a stored one. **Nothing
in this module may ever do that.** A claim citing a span the generator did not choose
is a claim whose citation was written by the pipeline and attributed to the model. It
verifies forever, because the subject's bytes are real; it passes the faithfulness
judge about as often as chance, because the subject's source usually does mention
whatever the claim was about; and the only thing it destroys is the one property the
whole tier rests on -- that a citation records what the claim was actually derived
from. There would be no way to find these afterwards and no signal that they existed.
So a draft whose references all miss produces no row, the miss is counted, and the
count is reported. Refuse, never repair.

**What this module is allowed to construct, and from what.** Every `Offer.span` comes
from `store.span_for_symbol` -- the index's own byte range and its own sha256, for a
symbol row that already exists. Nothing here builds a span from a path, an offset or a
line number that came out of a generator, because there is no route by which one could:
a `Draft` carries `tuple[int, ...]`, and an int can only name a menu entry or miss.
`Draft.resolve` does the mapping and hands back the misses, and this module's job is to
count them rather than smooth them over -- `LearnReport.invalid_refs` is the direct
measurement of whether the reference-number design is holding, and it is the number to
watch when a generator is swapped.

**An outage is not a result.** `GeneratorUnavailable` from a single symbol aborts the
whole run, uncaught. Every other exception from a generator is counted per-symbol and
the walk continues, because "this one symbol broke the model" is a fact about that
symbol. An unreachable backend is not: the next four hundred symbols will fail the same
way, and a report reading `400 considered, 3 admitted, 397 generator errors` looks like
a bad generator rather than an ollama that was never started. Worse, it looks like a
completed measurement, so the missing 397 would silently bias every comparison built on
it. There is no partial-credit reading of a dead backend, so there is no report.

**Re-running is resumption, not duplication.** `skip_existing=True` by default: a
symbol that already has an *active* assertion from this same generator is not asked
again. The store never deletes, so a second run without this would double the store
permanently, and every rate computed over it afterwards -- faithfulness, rejection,
staleness -- would be weighted by how many times a symbol happened to get re-drafted.
The same default is what makes a four-hour run over a local model resumable after a
crash. Note what it does NOT skip: a claim that went stale is no longer active, so its
symbol becomes a candidate again on the next run, which is the behaviour worth having
-- the repo invalidated that claim and the pipeline re-derives it.

**Determinism, because a run is only useful next to the previous one.** Candidate
selection (`candidate_symbols`) and reference numbering (`build_offers`) are both
stable for a given index, so two runs over an unchanged repo consider the same symbols
in the same order and hand out the same numbers. A selection that drifted would make
every before/after comparison a comparison of two different samples.

Nothing here prints. Long runs are observable through `on_progress` and through this
module's logger, and a library that writes to stdout is one that cannot be embedded in
the MCP server without corrupting its protocol stream.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import db
from ..assertions import store
from ..ingest.types import KIND_FUNCTION, KIND_METHOD, content_hash
from ..retrieve.graph import _neighbours
from .types import ClaimGenerator, Draft, GeneratorUnavailable, Offer

__all__ = [
    "DEFAULT_KINDS",
    "DEFAULT_MAX_OFFERS",
    "DEFAULT_MAX_OFFER_BYTES",
    "DEFAULT_MIN_LINES",
    "OUTCOME_ADMITTED",
    "OUTCOME_EMPTY_CLAIM",
    "OUTCOME_ERROR",
    "OUTCOME_NO_CITATION",
    "OUTCOME_NO_OFFERS",
    "OUTCOME_SKIPPED_EXISTING",
    "PHASE_DONE",
    "PHASE_START",
    "ROLE_CALLEE",
    "ROLE_CALLER",
    "ROLE_SUBJECT",
    "Candidate",
    "LearnProgress",
    "LearnReport",
    "LearnResult",
    "build_offers",
    "candidate_symbols",
    "learn",
]

logger = logging.getLogger(__name__)

# How many menu entries a generator is shown for one symbol, INCLUDING the subject.
# Bounded for two separate reasons and both of them bite: a menu that grows with a
# hub's fan-in blows the context window of a local 8k model, and reference numbers
# that depend on how many callers a symbol happens to have today are not stable
# between runs -- `[4]` meaning a different span this week than last is the quietest
# possible way for a stored citation to stop being reproducible. Twelve is a guess
# calibrated to nothing more than "the subject plus a handful of each side fits
# comfortably"; it is an argument so a caller with a bigger context can raise it.
DEFAULT_MAX_OFFERS = 12

# Size cap on ONE neighbour's menu entry, in bytes of source. A menu entry the model
# cannot read in full is one it would cite without having read, which is the failure
# the offer mechanism exists to prevent -- so an oversized neighbour is dropped rather
# than truncated. Truncating would be worse than dropping: the model would read the
# first 4KB and cite a span covering all 40KB, and the citation would verify.
#
# The SUBJECT is exempt. Refusing to describe a symbol because it is long would bias
# the store toward short functions, and the subject's span is the one span the claim is
# definitely about.
DEFAULT_MAX_OFFER_BYTES = 4_000

# A symbol shorter than this is not worth a claim. A one-line function's purpose is
# already stated by its signature, and an inferred restatement of a signature is a
# tier-2 row carrying a tier-0 fact -- it costs a judge call, occupies the store, and
# tells a reader nothing they could not read faster from the code.
DEFAULT_MIN_LINES = 3

# Which symbol kinds get claims. Functions and methods only, and the omissions are the
# interesting part: a `module` symbol's byte span is the ENTIRE file and a `class`
# symbol's span is every method it contains, so a claim about either cites a span as
# wide as the file. A wide citation is one a faithfulness judge will support almost
# anything from, and one a human cannot check by looking -- which makes it a citation
# in form only. `chunk/chunker.py` reached the same conclusion from the retrieval side
# and gives classes and modules summary chunks rather than bodies.
DEFAULT_KINDS: tuple[str, ...] = (KIND_FUNCTION, KIND_METHOD)

# What a menu entry IS, told to the model in the entry's own label. A model shown four
# unlabelled code blocks will cite whichever one contains the word it is describing;
# saying which is the subject and which are its neighbours is what lets it cite the
# span its claim actually rests on.
ROLE_SUBJECT = "the subject"
ROLE_CALLEE = "callee"
ROLE_CALLER = "caller"

# Per-symbol outcomes. Deliberately not collapsed into "admitted / not admitted": a
# run that refuses everything because the claims come back empty and a run that
# refuses everything because the references all miss are the same number and entirely
# different repairs, and only the split says which one happened.
OUTCOME_ADMITTED = "admitted"
OUTCOME_EMPTY_CLAIM = "refused_empty_claim"
OUTCOME_NO_CITATION = "refused_no_citation"
OUTCOME_NO_OFFERS = "no_offers"
OUTCOME_SKIPPED_EXISTING = "skipped_existing"
OUTCOME_ERROR = "generator_error"

# Why a candidate span never reached the menu. Split for the same reason as the
# outcomes above, and for a measured one: these were a single counter until a real run
# reported "96 offers dropped as unreadable" against a repository in which all 1,345
# symbols hashed clean. Every one of the 96 was oversize. `DROP_OVERSIZE` is the menu
# staying inside its context budget and is expected on any repo with long functions;
# `DROP_UNREADABLE` means the index and the working tree disagree, which is a reason to
# stop and re-index. Averaging them produces a number that cries wolf on every healthy
# repo, and would therefore be ignored on the one where it mattered.
DROP_OVERSIZE = "oversize"
DROP_UNREADABLE = "unreadable"

# Progress phases. `start` fires BEFORE the generator call, which is the only one that
# helps: the wait is the model, so a caller that only heard about a symbol after it
# finished would sit through a forty-second silence with nothing on screen.
PHASE_START = "start"
PHASE_DONE = "done"


@dataclass(frozen=True)
class Candidate:
    """One symbol selected to be claimed about, with what selection judged it on.

    Carries the fields the decision was made from rather than just an id, so a caller
    that wants to log or re-filter the set does not have to go back to the index and
    risk applying a different rule than the one that chose it.
    """

    symbol_id: int
    qualname: str
    kind: str
    path: str
    line_start: int
    line_end: int

    @property
    def lines(self) -> int:
        return self.line_end - self.line_start + 1


@dataclass(frozen=True)
class LearnResult:
    """What happened to one symbol, kept whether or not anything was admitted.

    The refusals are the point. A run's admitted count says how much the generator
    produced; the refusals say what was wrong with the rest, and `invalid_refs` on a
    per-symbol basis is what turns "the generator cites off the menu 8% of the time"
    into a list of the specific drafts where it did.
    """

    symbol_id: int
    qualname: str
    outcome: str
    assertion_id: int | None = None
    claim: str = ""
    citations: tuple[str, ...] = ()
    invalid_refs: tuple[int, ...] = ()
    offered: int = 0
    error: str = ""

    @property
    def admitted(self) -> bool:
        return self.outcome == OUTCOME_ADMITTED

    def detail(self) -> str:
        """One symbol's outcome in the form a human diagnoses a bad run with."""
        head = f"[{self.outcome}] {self.qualname}"
        parts = [head]
        if self.claim:
            parts.append(f"  claim:   {self.claim}")
        if self.citations:
            parts.append(f"  cited:   {', '.join(self.citations)}")
        if self.invalid_refs:
            parts.append(
                f"  off-menu refs: {', '.join(str(r) for r in self.invalid_refs)} "
                f"(menu had {self.offered})"
            )
        if self.error:
            parts.append(f"  error:   {self.error}")
        return "\n".join(parts)


@dataclass(frozen=True)
class LearnProgress:
    """One tick of a long run. Passed to `on_progress`, never printed from here.

    `result` is None on `PHASE_START` -- there is nothing to report yet, and that is
    the tick worth having, because the silence a caller needs filling is the model
    call that has not returned.
    """

    phase: str
    index: int  # 1-based position in the walk
    total: int
    candidate: Candidate
    result: LearnResult | None = None


@dataclass
class LearnReport:
    """What a run did, in the terms that say whether the reference design is holding.

    Follows `stale.RefreshReport` and `eval.faithfulness.FaithfulnessReport`: counts
    plus the per-item detail behind them, never a bare number. The counters are kept
    apart on purpose, and the identity worth knowing is

        considered = skipped_existing + symbols_without_offers + drafts_requested
        drafts_requested = admitted + refused_empty_claim + refused_no_citation
                           + generator_errors

    so a run whose numbers do not add up has lost drafts somewhere, which is exactly
    the shape of the bug a pipeline like this gets: a swallowed exception that turns
    into a symbol nobody notices was never asked about.

    `invalid_refs` is the one to watch. It counts off-menu references, which is the
    direct measure of whether numbered citation is working -- a generator that cites
    `[9]` against a five-entry menu is one whose evidence is arriving by accident, and
    every claim it does land is suspect even when the reference happens to resolve.
    """

    generator: str
    considered: int = 0
    skipped_existing: int = 0
    symbols_without_offers: int = 0
    drafts_requested: int = 0
    admitted: int = 0
    refused_empty_claim: int = 0
    refused_no_citation: int = 0
    invalid_refs: int = 0
    drafts_citing_off_menu: int = 0
    offers_dropped_unreadable: int = 0
    offers_dropped_oversize: int = 0
    generator_errors: int = 0
    results: list[LearnResult] = field(default_factory=list)

    @property
    def admitted_ids(self) -> list[int]:
        return [r.assertion_id for r in self.results if r.assertion_id is not None]

    @property
    def refused(self) -> list[LearnResult]:
        """Every draft that was requested and did not become a row.

        `skipped_existing` and `no_offers` are excluded: neither reached a generator,
        so neither is evidence about one.
        """
        refusals = (OUTCOME_EMPTY_CLAIM, OUTCOME_NO_CITATION, OUTCOME_ERROR)
        return [r for r in self.results if r.outcome in refusals]

    @property
    def admission_rate(self) -> float | None:
        """Admitted / drafts requested. None over an empty run, never 1.0.

        The same rule as `FaithfulnessReport.score`, for the same reason: "every draft
        was admitted" is trivially true of no drafts, and this repo has already been
        bitten once by a vacuous truth reading as success (the `no_evidence` guard in
        `store.py`). A run that asked for nothing has to say so here.
        """
        if not self.drafts_requested:
            return None
        return self.admitted / self.drafts_requested

    def summary(self) -> str:
        rate = "n/a (nothing drafted)" if self.admission_rate is None else (
            f"{self.admission_rate:.3f}"
        )
        return (
            f"admitted {self.admitted}/{self.drafts_requested} ({rate})  "
            f"generator={self.generator}\n"
            f"  considered={self.considered} "
            f"skipped_existing={self.skipped_existing} "
            f"no_offers={self.symbols_without_offers}\n"
            f"  refused: empty_claim={self.refused_empty_claim} "
            f"no_valid_citation={self.refused_no_citation} "
            f"generator_errors={self.generator_errors}\n"
            f"  off-menu refs={self.invalid_refs} "
            f"across {self.drafts_citing_off_menu} draft(s); "
            f"offers dropped: oversize={self.offers_dropped_oversize} "
            f"unreadable={self.offers_dropped_unreadable}"
        )

    def format_report(self) -> str:
        """The summary plus every draft that did not become a row.

        A refusal count with no attached detail cannot be acted on. Whether the
        generator is returning empty claims, citing numbers that do not exist, or
        raising, the repair is different in each case and the only way to tell is to
        read the ones that failed.
        """
        lines = [self.summary()]
        failed = self.refused
        if failed:
            lines.append("")
            lines.append(f"-- {len(failed)} draft(s) refused --")
            lines += [r.detail() for r in failed]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# symbol selection
# --------------------------------------------------------------------------


def candidate_symbols(
    conn: sqlite3.Connection,
    *,
    kinds: Sequence[str] = DEFAULT_KINDS,
    include_tests: bool = False,
    include_private: bool = False,
    min_lines: int = DEFAULT_MIN_LINES,
    limit: int | None = None,
) -> list[Candidate]:
    """The symbols worth claiming about, in a stable order. Deterministic by design.

    A named function rather than a query inside `learn` because the eval has to score
    the SAME set the run produced. If the pipeline and the measurement each decided for
    themselves which symbols were interesting, a coverage number would be a ratio of
    two different denominators and nobody would be able to tell.

    The ordering is `(path, line_start, id)` -- file order, which is stable across runs
    for an unchanged index and reads in the order a human would. `limit` applies after
    ordering, so `limit=50` is the same fifty symbols every time; a limit applied to an
    unordered scan gives a different sample per run and quietly makes two runs
    incomparable while looking identical in the report.

    The four exclusions, each because the claim it would produce is worse than no claim:

    * **kinds** -- see `DEFAULT_KINDS`. Modules and classes span whole files.
    * **tests** -- `files.is_test`. A test's purpose is stated in its own name, and a
      store full of "this test tests X" crowds out the claims about X. The convention
      that decides this is `ingest.indexer.is_test_path`, applied once at index time.
    * **private** -- a leading underscore. Not because private code is unimportant, but
      because it is where the trivial helpers are, and the budget on a local model is
      real. `include_private=True` for a run that wants them.
    * **trivial** -- fewer than `min_lines` lines.

    None of these is a judgement about correctness, so none of them belongs in the
    store's gate; they are a sampling policy, which is why they live here and are all
    arguments.
    """
    if not kinds:
        raise ValueError("candidate_symbols needs at least one symbol kind")
    placeholders = ",".join("?" * len(kinds))
    sql = (
        "SELECT s.id, s.qualname, s.kind, s.name, f.path, s.line_start, s.line_end "  # noqa: S608
        "FROM symbols s JOIN files f ON f.id = s.file_id "
        f"WHERE s.kind IN ({placeholders}) "
        "  AND (s.line_end - s.line_start + 1) >= ? "
    )
    params: list[object] = [*kinds, int(min_lines)]
    if not include_tests:
        sql += "  AND f.is_test = 0 "
    if not include_private:
        # On the bare name, not the qualname: a public method of a private class is
        # still reachable through the public one, and excluding on qualname would drop
        # every symbol in a module whose package happens to start with an underscore.
        sql += "  AND s.name NOT LIKE '\\_%' ESCAPE '\\' "
    sql += "ORDER BY f.path, s.line_start, s.id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [
        Candidate(
            symbol_id=int(row["id"]),
            qualname=str(row["qualname"]),
            kind=str(row["kind"]),
            path=str(row["path"]),
            line_start=int(row["line_start"]),
            line_end=int(row["line_end"]),
        )
        for row in conn.execute(sql, tuple(params))
    ]


# --------------------------------------------------------------------------
# the menu
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    """A menu entry before it has a number. Refs are assigned in one place, at the end."""

    symbol_id: int
    span: store.EvidenceSpan
    text: str
    label: str


@dataclass(frozen=True)
class _Menu:
    """A built menu plus what building it discarded.

    The drop count travels with the offers because it is a measurement, not a detail:
    an index that disagrees with disk drops entries silently, and a run whose menus
    were quietly half empty would look like a run against a generator that had nothing
    to say.
    """

    offers: list[Offer]
    dropped_unreadable: int = 0
    dropped_oversize: int = 0


def _read_source(root: Path, path: str, cache: dict[str, bytes | None]) -> bytes | None:
    """Read a file once per menu. None if it cannot be read.

    The `is_file` test is not redundant with the `except OSError`. OSError covers
    every way a read fails loudly; a FIFO fails quietly, by blocking inside
    `read_bytes` until some other process opens the write end. A generation run
    against a repo holding one would stop building menus and never raise, never log,
    and never finish -- indistinguishable, from outside, from a slow model.
    `is_file()` is False for a FIFO, a directory, a socket and a device node, and True
    for a regular file or a symlink to one, so one test covers the class.

    None drops the offer as `DROP_UNREADABLE`, which is the disposition already given
    to a file the index disagrees with, and is the right one here: what the guard
    knows is "these bytes cannot be had", not that anything is corrupt.

    Near-verbatim in `assertions.store._read_source` and `assertions.stale._read_file`,
    and deliberately not factored out -- a private cross-package import to share four
    lines is the worse coupling. It also does not close the window between the test
    and the read; a regular file swapped for a FIFO in between still blocks, and
    closing that needs an fd-based `os.open(..., O_NONBLOCK)`.
    """
    if path not in cache:
        target = root / path
        if not target.is_file():
            cache[path] = None
            return cache[path]
        try:
            cache[path] = target.read_bytes()
        except OSError:
            cache[path] = None
    return cache[path]


def _item_for(
    conn: sqlite3.Connection,
    root: Path,
    symbol_id: int,
    role: str,
    cache: dict[str, bytes | None],
    *,
    max_bytes: int | None,
) -> tuple[_Item | None, str | None]:
    """Build one menu entry from the index, or explain which kind of nothing it is.

    Returns `(item, None)` on success and `(None, reason)` otherwise, where the reason
    separates a benign budget decision from a broken index. That split is not
    cosmetic: it was reported as one number on the first real run, and 96 offers came
    back "dropped as unreadable" on a repository whose 1,345 symbols every one hashed
    clean. All 96 were `DROP_OVERSIZE`. The counter said the index disagreed with disk
    -- which is a data-integrity failure worth stopping for -- when what had actually
    happened was that a neighbour was longer than `max_offer_bytes`, which is the menu
    working as designed. One name for two conditions sends a reader hunting for
    corruption that is not there, and would equally hide corruption that is behind a
    number nobody reads twice.

    Four ways this declines, and the third is the one that matters:

    1. The symbol is not in the index (a stale edge; `span_for_symbol` raises `KeyError`).
    2. Its file cannot be read right now.
    3. **The bytes on disk do not hash to what the index published for this symbol.**
       The index is behind the working tree. Offering it anyway would show the model
       the CURRENT text while attaching the STALE hash to the citation -- so the claim
       would be about what the model read, and the citation would be about bytes nobody
       looked at, which is exactly the drift `Offer` carries `span` and `text` together
       to prevent. It would also expire on the first serve, with a `hash_mismatch` that
       blames the repo for an inconsistency this function introduced.
    4. It is over `max_bytes` (neighbours only -- see `DEFAULT_MAX_OFFER_BYTES`).

    The span is `store.span_for_symbol`, always. There is no branch in this module that
    builds a span from anything else, and that is the invariant the whole tier rests on.
    """
    try:
        span = store.span_for_symbol(conn, symbol_id)
    except KeyError:
        return None, DROP_UNREADABLE
    row = conn.execute(
        "SELECT qualname, kind FROM symbols WHERE id = ?", (symbol_id,)
    ).fetchone()
    if row is None:
        return None, DROP_UNREADABLE
    if max_bytes is not None and (span.byte_end - span.byte_start) > max_bytes:
        return None, DROP_OVERSIZE
    source = _read_source(root, span.path, cache)
    if source is None or span.byte_end > len(source):
        return None, DROP_UNREADABLE
    raw = source[span.byte_start : span.byte_end]
    if content_hash(raw) != span.content_hash:
        return None, DROP_UNREADABLE
    return (
        _Item(
            symbol_id=symbol_id,
            span=span,
            # Exactly the cited bytes, and nothing around them. Not the chunk text: a
            # chunk carries a generated header that is NOT inside the span, so a model
            # citing on the strength of the header would be citing bytes that do not
            # contain what convinced it.
            text=raw.decode("utf-8", "replace"),
            label=f"{role}: {row['kind']} {row['qualname']}",
        ),
        None,
    )


def _neighbour_ids(
    conn: sqlite3.Connection, symbol_id: int, *, include_callers: bool
) -> tuple[list[int], list[int]]:
    """Direct callees and callers of `symbol_id`, deduplicated and stably ordered.

    Traversal is `retrieve.graph._neighbours` rather than a second call-graph query in
    this package. One place decides what a neighbour is, which is what keeps "the menu"
    and "what the graph modality would have retrieved" the same notion of adjacency;
    `expand` itself does not fit -- it is multi-hop spreading activation that ranks by
    accumulated score, drops its own seeds, and returns `Hit`s, and a menu needs the
    one-hop neighbours of one symbol in an order that does not depend on scoring.

    Two inherited limits, stated rather than hidden. `_neighbours` caps at
    `graph.MAX_FANOUT` EDGE ROWS per direction, and `edges` holds one row per call
    site, so a symbol that calls the same helper twenty times can fill its outbound cap
    with twenty rows naming one neighbour. And the cap is applied with
    `ORDER BY confidence DESC`, which leaves ties to SQLite's scan order -- so for a
    hub with more equally-confident neighbours than the cap, WHICH ones survive is
    stable for a given database file but is not guaranteed by anything. Everything
    after the cap is sorted by `(qualname, id)` here, so the ordering of what does
    survive is ours and is stable.
    """
    callees: list[int] = []
    callers: list[int] = []
    for neighbour_id, direction in _neighbours(conn, symbol_id, include_callers):
        (callees if direction == "calls" else callers).append(int(neighbour_id))
    return _stable(conn, callees), _stable(conn, callers)


def _stable(conn: sqlite3.Connection, ids: list[int]) -> list[int]:
    """Deduplicate and order by `(qualname, id)`. Empty in, empty out."""
    unique = sorted(set(ids))
    if not unique:
        return []
    placeholders = ",".join("?" * len(unique))
    rows = conn.execute(
        f"SELECT id, qualname FROM symbols WHERE id IN ({placeholders})",  # noqa: S608
        tuple(unique),
    ).fetchall()
    return [int(r["id"]) for r in sorted(rows, key=lambda r: (str(r["qualname"]), int(r["id"])))]


def _build_menu(
    conn: sqlite3.Connection,
    repo_root: db.StrPath,
    symbol_id: int,
    *,
    max_offers: int,
    include_callers: bool,
    max_offer_bytes: int | None,
) -> _Menu:
    """`build_offers` plus the drop count, which the public signature has no room for."""
    if max_offers < 1:
        raise ValueError(
            f"max_offers={max_offers} leaves no room for the subject itself. A menu "
            "with no entries cannot be cited, and a claim that cites nothing is not "
            "admitted -- so this would silently produce a run that refuses everything."
        )
    root = Path(str(repo_root))
    cache: dict[str, bytes | None] = {}
    dropped_unreadable = 0
    dropped_oversize = 0

    subject, _reason = _item_for(conn, root, symbol_id, ROLE_SUBJECT, cache, max_bytes=None)
    if subject is None:
        # No subject means no menu at all. Offering a symbol's callers as evidence for
        # a claim about the symbol, while the symbol's own bytes could not be read, is
        # how a claim ends up cited entirely on code it is not about.
        #
        # Always `unreadable`, never `oversize`: the subject is passed `max_bytes=None`,
        # so the size branch cannot fire for it. A subject that vanished here really is
        # an index/disk disagreement.
        return _Menu(offers=[], dropped_unreadable=1)

    callee_ids, caller_ids = _neighbour_ids(
        conn, symbol_id, include_callers=include_callers
    )
    taken = {symbol_id}  # self-recursion offers the subject twice, wasting a number
    budget = max_offers - 1

    # Callees are filled first, up to half the budget, then callers, then whichever
    # list still has entries spends the slack. Not an arbitrary split: callees are
    # what the subject DOES and callers are what it is FOR, and the default claim kind
    # is `purpose`, so when the budget binds the "what it does" side is the one worth
    # keeping. A hub with forty callers would otherwise crowd its own body out of its
    # menu.
    first_share = budget if not caller_ids else (budget + 1) // 2
    items: list[_Item] = [subject]

    def take(ids: Sequence[int], role: str, room: int) -> list[int]:
        nonlocal dropped_unreadable, dropped_oversize
        leftover: list[int] = []
        for neighbour_id in ids:
            if neighbour_id in taken:
                continue
            if len(items) - 1 >= room:
                leftover.append(neighbour_id)
                continue
            item, reason = _item_for(
                conn, root, neighbour_id, role, cache, max_bytes=max_offer_bytes
            )
            if item is None:
                if reason == DROP_OVERSIZE:
                    dropped_oversize += 1
                else:
                    dropped_unreadable += 1
                continue
            taken.add(neighbour_id)
            items.append(item)
        return leftover

    remaining_callees = take(callee_ids, ROLE_CALLEE, min(first_share, budget))
    remaining_callers = take(caller_ids, ROLE_CALLER, budget)
    take(remaining_callees, ROLE_CALLEE, budget)
    take(remaining_callers, ROLE_CALLER, budget)

    # `take` is the ONE place the bound is applied. A second `[:max_offers]` truncation
    # here would look like belt and braces and is worse than nothing: it can never fire
    # while `take` is correct, so it cannot be tested, and a future edit that broke the
    # room check would be masked by it rather than caught.
    offers = [
        Offer(ref=i, span=item.span, text=item.text, label=item.label)
        for i, item in enumerate(items, start=1)
    ]
    return _Menu(
        offers=offers,
        dropped_unreadable=dropped_unreadable,
        dropped_oversize=dropped_oversize,
    )


def build_offers(
    conn: sqlite3.Connection,
    repo_root: db.StrPath,
    symbol_id: int,
    *,
    max_offers: int = DEFAULT_MAX_OFFERS,
    include_callers: bool = True,
    max_offer_bytes: int | None = DEFAULT_MAX_OFFER_BYTES,
) -> list[Offer]:
    """The numbered menu a generator gets for one symbol: the subject, its callees, its
    callers.

    Every span here is `store.span_for_symbol` -- the index's own byte range and its
    own sha256 for a row that exists. Nothing on this menu can be constructed from
    anything a model said, which is the property that makes a fabricated citation
    unrepresentable rather than merely discouraged (see `generate/types.py`).

    Refs are 1-based (a model told to cite `[0]` cites `[1]` anyway) and the subject is
    always ref 1. Ordering is deterministic for a given index: subject, then callees,
    then callers, each side sorted by `(qualname, symbol id)` -- so the same symbol
    yields the same numbers on a re-run, and a stored claim's reference `[3]` still
    means what it meant. See `_neighbour_ids` for the one place that stability is
    inherited rather than enforced.

    Empty list means the subject itself could not be offered -- it is not in the index,
    its file is unreadable, or the index's hash for it disagrees with disk. The caller
    must treat that as "do not ask about this symbol", never as "ask with a smaller
    menu": a claim about a symbol, cited entirely on its neighbours, is cited on code
    it is not about.

    `max_offer_bytes=None` removes the per-neighbour size cap for a caller with a large
    context window. `max_offers` bounds the whole menu including the subject.
    """
    return _build_menu(
        conn,
        repo_root,
        symbol_id,
        max_offers=max_offers,
        include_callers=include_callers,
        max_offer_bytes=max_offer_bytes,
    ).offers


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _already_claimed(conn: sqlite3.Connection, generator: str) -> set[str]:
    """Subjects that already hold an ACTIVE claim from this generator.

    Active, not any status. A stale claim's evidence moved and it should be re-derived;
    a rejected one was refuted and re-asking is how a generator gets a second attempt
    at a claim a judge already threw out -- which is a defensible thing to want and is
    what `skip_existing=False` is for, rather than something to do by default.

    Matched on `generator = ?`, so claims written by a different generator (or by hand,
    with a NULL generator) never suppress this one. Two generators over one repo is the
    comparison the `generator` column exists to make possible.
    """
    return {
        str(row["subject_qualname"])
        for row in conn.execute(
            "SELECT DISTINCT subject_qualname FROM assertions "
            "WHERE status = ? AND generator = ?",
            (store.STATUS_ACTIVE, generator),
        )
    }


def _tick(
    on_progress: Callable[[LearnProgress], None] | None,
    phase: str,
    index: int,
    total: int,
    candidate: Candidate,
    result: LearnResult | None = None,
) -> None:
    """Report one step of the walk to the caller's callback, if it wanted one.

    A free function rather than a closure over the loop variables, because a closure
    defined inside a loop captures them by reference and fires later holding whatever
    the loop moved on to -- which for a progress display is a report of the wrong
    symbol, and is exactly the kind of bug nobody looks for in output they are only
    watching scroll past.
    """
    if on_progress is None:
        return
    on_progress(
        LearnProgress(
            phase=phase, index=index, total=total, candidate=candidate, result=result
        )
    )


def _admit(
    conn: sqlite3.Connection,
    candidate: Candidate,
    generator_name: str,
    draft: Draft,
    offers: Sequence[Offer],
) -> LearnResult:
    """Resolve one draft's references and put it through the gate. One symbol's worth.

    The order of checks is fixed and the precedence is deliberate, so that the outcome
    counters partition the drafts rather than overlapping: refs are resolved FIRST (an
    off-menu reference is counted even on a draft that is about to be refused for some
    other reason -- it is a fact about the generator either way), then an empty claim,
    then the gate.

    Nothing here inspects `spans` before calling `write_assertion`. That is on purpose:
    `EvidenceRequired` is the store's decision and this catches it, so there is exactly
    one place in the codebase that decides an uncited claim is inadmissible. A
    pre-check here would be a second implementation of that rule, free to drift, and
    the direction it would drift is toward "well, we could just attach the subject's
    span".
    """
    spans, invalid = draft.resolve(offers)
    claim = draft.claim.strip()

    def outcome(
        name: str, *, assertion_id: int | None = None, text: str = "",
        citations: tuple[str, ...] = (),
    ) -> LearnResult:
        return LearnResult(
            symbol_id=candidate.symbol_id,
            qualname=candidate.qualname,
            outcome=name,
            assertion_id=assertion_id,
            claim=text,
            citations=citations,
            # Counted on every route out of here, including the refusals. An off-menu
            # reference is a fact about the generator whether or not the draft carrying
            # it was going to land.
            invalid_refs=tuple(invalid),
            offered=len(offers),
        )

    if not claim:
        # An empty claim with perfectly good citations is still nothing. Storing it
        # would put a row in the store that a judge would have to adjudicate and a
        # reader would have to read, and both would find no statement in it.
        return outcome(OUTCOME_EMPTY_CLAIM)
    try:
        assertion_id = store.write_assertion(
            conn,
            subject_qualname=candidate.qualname,
            kind=draft.kind,
            claim=claim,
            spans=spans,
            subject_symbol_id=candidate.symbol_id,
            generator=generator_name,
            confidence=draft.confidence,
        )
    except store.EvidenceRequired:
        # Counted, and that is ALL that happens. The temptation at this exact line is
        # to fall back to the subject's span, which is sitting in `offers[0]` and is
        # known-good; doing so would attribute a pipeline-authored citation to the
        # generator and there would be no way to find them again. See the module
        # docstring.
        return outcome(OUTCOME_NO_CITATION, text=claim)
    return outcome(
        OUTCOME_ADMITTED,
        assertion_id=assertion_id,
        text=claim,
        citations=tuple(span.citation for span in spans),
    )


def learn(
    conn: sqlite3.Connection,
    repo_root: db.StrPath | None,
    generator: ClaimGenerator,
    *,
    candidates: Sequence[Candidate] | None = None,
    limit: int | None = None,
    max_offers: int = DEFAULT_MAX_OFFERS,
    include_callers: bool = True,
    max_offer_bytes: int | None = DEFAULT_MAX_OFFER_BYTES,
    skip_existing: bool = True,
    on_progress: Callable[[LearnProgress], None] | None = None,
) -> LearnReport:
    """Walk the candidate symbols, draft a claim about each, admit what cites the menu.

    The whole pipeline: `candidate_symbols` -> `build_offers` -> `generator.draft` ->
    `Draft.resolve` -> `store.write_assertion`. Every claim that lands carries
    `generator=generator.name`, which is what makes "find everything that model wrote"
    a query and a two-generator comparison possible at all.

    **`GeneratorUnavailable` is not caught, and one symbol raising it stops the run.**
    Every other exception is per-symbol and counted; this one is not, because a backend
    is not a property of a symbol. If ollama is down for `leases.acquire` it is down
    for the four hundred symbols after it, and absorbing it would produce a report that
    is indistinguishable from a completed run against a generator that mostly failed --
    same shape, same counters, and a coverage hole biased toward whenever the outage
    started. There is no honest partial reading of "no measurement happened", so the
    report is not returned. The claims admitted before it are real and stay in the
    store; `skip_existing` means the next run picks up from there.

    **Re-running.** By default a symbol that already has an active claim from this same
    generator is skipped -- see `_already_claimed`. `skip_existing=False` drafts again
    regardless, which duplicates rows if the repo has not changed; that is the honest
    behaviour for a deliberate second opinion and the wrong default for everything else,
    because the store never deletes and every rate computed over it afterwards would
    silently weight symbols by how often they were re-drafted.

    **Writes are per-claim, not one batch.** Each `write_assertion` is its own
    transaction (`store._atomic` joins a caller's if there is one), so a run
    interrupted after three hours keeps the claims it reached instead of losing all of
    them -- the same choice `eval.faithfulness.adjudicate` makes about verdicts. A
    caller that genuinely needs all-or-nothing can wrap this in `db.transaction`.

    `candidates` overrides selection entirely, which is how the eval scores the exact
    set a run used rather than re-deriving a set that might differ. `limit` applies
    after ordering either way.
    """
    root = store._repo_root(conn, repo_root)
    walk = list(candidate_symbols(conn) if candidates is None else candidates)
    if limit is not None:
        walk = walk[:limit]
    report = LearnReport(generator=generator.name)
    seen = _already_claimed(conn, generator.name) if skip_existing else set()
    total = len(walk)

    for position, candidate in enumerate(walk, start=1):
        report.considered += 1

        if candidate.qualname in seen:
            report.skipped_existing += 1
            result = LearnResult(
                symbol_id=candidate.symbol_id,
                qualname=candidate.qualname,
                outcome=OUTCOME_SKIPPED_EXISTING,
            )
            report.results.append(result)
            _tick(on_progress, PHASE_DONE, position, total, candidate, result)
            continue

        menu = _build_menu(
            conn,
            root,
            candidate.symbol_id,
            max_offers=max_offers,
            include_callers=include_callers,
            max_offer_bytes=max_offer_bytes,
        )
        report.offers_dropped_unreadable += menu.dropped_unreadable
        report.offers_dropped_oversize += menu.dropped_oversize
        if not menu.offers:
            report.symbols_without_offers += 1
            result = LearnResult(
                symbol_id=candidate.symbol_id,
                qualname=candidate.qualname,
                outcome=OUTCOME_NO_OFFERS,
                error="the subject's own bytes could not be offered as evidence",
            )
            report.results.append(result)
            logger.debug("no menu for %s; not asking", candidate.qualname)
            _tick(on_progress, PHASE_DONE, position, total, candidate, result)
            continue

        _tick(on_progress, PHASE_START, position, total, candidate)
        report.drafts_requested += 1
        try:
            draft = generator.draft(subject=candidate.qualname, offered=menu.offers)
        except GeneratorUnavailable:
            # Uncaught by design. See the docstring: no measurement happened, so there
            # is no report to return, and the counters so far describe a run that was
            # cut short rather than one that completed.
            logger.error(
                "generator %s became unreachable at %s (%d of %d); aborting the run "
                "with %d claim(s) already admitted",
                generator.name, candidate.qualname, position, total, report.admitted,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not end the walk
            report.generator_errors += 1
            result = LearnResult(
                symbol_id=candidate.symbol_id,
                qualname=candidate.qualname,
                outcome=OUTCOME_ERROR,
                offered=len(menu.offers),
                error=f"{type(exc).__name__}: {exc}",
            )
            report.results.append(result)
            logger.warning("generator failed on %s: %s", candidate.qualname, exc)
            _tick(on_progress, PHASE_DONE, position, total, candidate, result)
            continue

        result = _admit(conn, candidate, generator.name, draft, menu.offers)
        if result.invalid_refs:
            report.invalid_refs += len(result.invalid_refs)
            report.drafts_citing_off_menu += 1
            logger.warning(
                "%s cited %s off a menu of %d",
                candidate.qualname,
                list(result.invalid_refs),
                len(menu.offers),
            )
        if result.outcome == OUTCOME_ADMITTED:
            report.admitted += 1
        elif result.outcome == OUTCOME_EMPTY_CLAIM:
            report.refused_empty_claim += 1
        elif result.outcome == OUTCOME_NO_CITATION:
            report.refused_no_citation += 1
        report.results.append(result)
        logger.debug("%s -> %s", candidate.qualname, result.outcome)
        _tick(on_progress, PHASE_DONE, position, total, candidate, result)

    return report
