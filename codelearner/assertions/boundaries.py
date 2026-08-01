"""Citation boundaries: the claims whose evidence is right and whose EDGES are wrong.

Every other expiry in this package is a finding about bytes. A file went, an edit
landed, a tail was truncated -- something on disk stopped matching what was cited, the
hash says so, and the repair is to re-derive the claim or follow the rename. This
module answers a question none of those can, because on this failure the bytes are
perfect: **is the citation pointing at the whole of what it claims to be about?**

The defect it finds is WP8's, surviving inside carried data. Before schema v6 a
decorated symbol's span was taken from the inner `function_definition`, which begins
at `def` and not at the `@`, so every citation of a decorated symbol silently excluded
its own decorators. WP8 widened the symbol and bumped the version; `--carry-assertions`
then carries the tier-2 store across that rebuild, and a carried claim keeps the span
it was written with. Those bytes never moved, so the claim verifies, stays `active`,
and is served -- correctly, by the rules as written, and carrying the exact fail-open
exposure the widening existed to close. Measured on a real upgraded index (swarm-sync,
150 carried assertions), 10 active claims were in this state, one of them a claim about
an endpoint whose citation stopped one line short of

    @app.post("/intent", dependencies=[Depends(require_token)])

Delete `Depends(require_token)` from that line and the claim still verifies fresh, on
both verifiers, forever. Nothing else in this package can see that.

**Precision over coverage, deliberately.** The obvious rule -- expire any span that is
a strict suffix of a symbol -- is wrong, and wrong in the direction that punishes the
good case. An agent quoting three lines of a function body is making a NARROWER and
therefore stronger citation than one quoting the whole function, and expiring it would
teach the generator to cite as widely as possible, which is the opposite of what the
evidence gate wants. On the same real index, 15 active spans are strict suffixes of a
symbol and only 10 of them are this defect; the other 5 are a claim about a method that
happens to be the last method in its class, so the method's span ends exactly where the
class's does. Those must survive, and they do.

**The rule is exact, not heuristic.** A span is a pre-v6 narrowed citation when its end
equals a symbol's end AND its start equals the start of the definition INSIDE that
symbol's `decorated_definition` node -- asked of tree-sitter, which is the same parser
that produced both spans, rather than inferred from the prefix text. There is no
pattern-match to be defeated: `@` inside a leading comment or docstring, a decorator
with a multi-line argument list, a comment sitting between two decorators, `async def`
whose definition node starts at `async` -- each is answered by the parse, not by the
shape of the bytes. See `python_extract.decorated_body_start`.

The one thing that is NOT proven is intent: a post-v6 agent could deliberately cite a
decorated function from `def` onwards and be caught by this. That claim is expired, and
it should be -- it is byte-for-byte the same citation as the pre-v6 one and carries the
same fail-open exposure, whoever wrote it and whenever. There is no version stamp on a
span, and there should not be one; the boundary is the defect.

**Marked stale, never rewritten.** Widening the stored span to match the symbol would
make the numbers look right and would fabricate a citation the generator never made --
it would assert that a claim was derived from bytes no generator ever read, which is
the single thing `generate/pipeline.py` forbids most explicitly. Stale is the honest
outcome: the claim, its text, its verdicts and its log row all stay, and what it needs
is a redraft against the whole symbol.
"""
from __future__ import annotations

import sqlite3

from .. import db
from ..ingest.python_extract import decorated_body_start
from .store import (
    REASON_DECORATORS_EXCLUDED,
    STATUS_ACTIVE,
    EvidenceSpan,
    _load_assertions,
    _read_source,
    _repo_root,
    _Unread,
    mark_stale,
)

__all__ = ["REASON_DECORATORS_EXCLUDED", "expire_narrowed_citations"]


def _symbol_ends(
    conn: sqlite3.Connection, paths: set[str]
) -> dict[tuple[str, int], list[int]]:
    """`(path, byte_end) -> [symbol byte_start, ...]` for the cited files only.

    Keyed on the end because that is the half a narrowed citation shares with its
    symbol; the start is the half that moved. Restricted to files something actually
    cites so this costs one query rather than a scan of every symbol in the repo.
    """
    ends: dict[tuple[str, int], list[int]] = {}
    if not paths:
        return ends
    placeholders = ",".join("?" * len(paths))
    for row in conn.execute(
        "SELECT f.path AS path, s.byte_start, s.byte_end FROM symbols s "  # noqa: S608
        f"JOIN files f ON f.id = s.file_id WHERE f.path IN ({placeholders})",
        tuple(sorted(paths)),
    ):
        ends.setdefault((row["path"], row["byte_end"]), []).append(row["byte_start"])
    return ends


def _is_narrowed(
    source: bytes, span: EvidenceSpan, starts: list[int]
) -> bool:
    """Whether `span` is a symbol's span with the decorators cut off the front.

    `starts` are the symbol starts that share this span's end. A symbol starting AT or
    AFTER the span's start is not a wider citation of anything and is skipped without
    a parse -- that is the ordinary case where a claim cites a whole symbol, and it
    must stay cheap because it is nearly all of them.
    """
    for symbol_start in starts:
        if symbol_start >= span.byte_start:
            continue
        if decorated_body_start(source, symbol_start, span.byte_end) == span.byte_start:
            return True
    return False


def expire_narrowed_citations(
    conn: sqlite3.Connection, repo_root: db.StrPath | None = None
) -> int:
    """Expire every active claim whose citation excludes its symbol's decorators.
    Returns how many transitioned.

    Run on the carry path, after the ordinary verification. Order matters and is not
    interchangeable: a claim whose bytes ALSO moved is a `hash_mismatch` first, that
    is the more urgent finding, and `mark_stale` refuses to expire something already
    expired -- so the second sweep silently declines it rather than writing a second
    log row that would make one claim look like two failures.

    **Only `active` claims are loaded.** A `rejected` claim was refused by a judge on
    evidence that was correct at the time, and re-expiring it here would overwrite an
    adjudication with a boundary complaint; an already-`stale` one has its reason
    recorded and does not need a second. Both are enforced twice, once by the query
    here and once by `mark_stale`'s `AND status = ?`, because this is the sweep most
    likely to be called from somewhere new.

    A file that cannot be read is skipped in silence, with no status change and no log
    row: an unreadable file establishes nothing about where a symbol starts, and "we
    could not look" is not grounds for expiry here any more than it is on the serve
    path. The next carry over a healthy filesystem finds it.

    Neither hash column is written. The cited bytes hash to exactly what was cited --
    that is the whole finding -- so an `expected_hash` and an `observed_hash` here
    would be two identical strings inviting the reading "nothing changed, why did this
    expire". `span_id` names the citation that is too narrow and the reason names the
    defect, which is the entire content of the event.
    """
    root = _repo_root(conn, repo_root)
    active = _load_assertions(conn, "status = ?", (STATUS_ACTIVE,))
    if not active:
        return 0

    ends = _symbol_ends(
        conn, {span.path for a in active for span in a.spans}
    )
    cache: dict[str, bytes | _Unread] = {}
    expired = 0
    for assertion in active:
        for span in assertion.spans:
            starts = ends.get((span.path, span.byte_end))
            if not starts:
                continue
            source = _read_source(root, span.path, cache)
            if isinstance(source, _Unread):
                continue
            if not _is_narrowed(source, span, starts):
                continue
            if mark_stale(
                conn,
                assertion.id,
                REASON_DECORATORS_EXCLUDED,
                span_id=span.id,
            ):
                expired += 1
            # One claim, one expiry. A second narrowed span on the same claim is the
            # same finding and would only add a log row for an assertion that has
            # already left `active`.
            break
    return expired
