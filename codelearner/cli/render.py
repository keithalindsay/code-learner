"""Turning results into text a person reads, and into JSON a program parses.

Both renderings live in one module on purpose. The human table and the `--json`
object drift apart the moment they are derived in two places, and a `--json` shape
that disagrees with what the tool just printed is worse than having no `--json` at
all -- one of them is wrong and nothing says which.
"""
from __future__ import annotations

from typing import Any

from ..ingest.types import TIER_FACT, TIER_INFERRED, TIER_RESOLVED
from ..retrieve import Hit

TIER_LABELS = {TIER_FACT: "T0", TIER_RESOLVED: "T1", TIER_INFERRED: "T2"}

# Which tier a hit's evidence rests on, keyed by the modality that produced it.
#
# A lexical or dense hit is T0: the text that matched was parsed straight out of the
# source, and nothing had to be decided to reach it. A graph hit is T1, because
# graph expansion traverses ONLY resolved edges (see retrieve/graph.py) -- reaching
# it depended on a name binding that carries a confidence below 1 and can be wrong.
# That difference is the entire point of the tier model, and a caller who asked for
# facts should not be handed the output of a resolver's guess without being told.
MODALITY_TIER = {
    "lexical": TIER_FACT,
    "dense": TIER_FACT,
    "graph": TIER_RESOLVED,
    # Phase 4's assertion layer will retrieve under this name. Listed now, before it
    # exists, so `--facts-only` is a filter that already works rather than a promise
    # that something later will remember to wire it up.
    "inferred": TIER_INFERRED,
}


def tier_of(hit: Hit) -> int:
    """The tier of the strongest evidence that reached `hit`.

    Fusion joins the contributing modalities with `+`, so a symbol found by text
    search AND by graph expansion arrives as `dense+graph`. The MINIMUM tier is the
    right answer there: the tier records what had to be believed to surface this
    symbol, and if a text modality matched it directly then no resolution was
    involved, whatever else also voted for it.

    An unrecognised modality is treated as inferred. `--facts-only` is a promise
    about provenance, so it has to fail closed -- an unknown source is exactly the
    thing a caller asking for facts only is trying to exclude.
    """
    parts = [p for p in hit.modality.split("+") if p]
    if not parts:
        return TIER_INFERRED
    return min(MODALITY_TIER.get(part, TIER_INFERRED) for part in parts)


def facts_only(hits: list[Hit]) -> list[Hit]:
    """Drop anything above T1 -- parsed facts and resolved names, nothing asserted.

    Today this removes nothing, because nothing yet retrieves at T2. It is written
    as a real filter rather than a no-op so that the flag does not quietly start
    lying the day the inference layer lands. When it does, the better place for
    this is before fusion, so the filtered slots get refilled instead of shortening
    the result list.
    """
    return [hit for hit in hits if tier_of(hit) <= TIER_RESOLVED]


def hit_json(hit: Hit, rank: int) -> dict[str, Any]:
    """One hit as a stable JSON object.

    `via` is always present, empty string when the hit was not reached by graph
    expansion. A consumer should not have to probe for a key to find out whether
    an explanation exists.
    """
    tier = tier_of(hit)
    return {
        "rank": rank,
        "tier": TIER_LABELS[tier],
        "tier_n": tier,
        "symbol_id": hit.symbol_id,
        "qualname": hit.qualname,
        "kind": hit.kind,
        "path": hit.path,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "score": round(float(hit.score), 6),
        "modality": hit.modality,
        "is_test": hit.is_test,
        "via": hit.via,
    }


def format_hit(hit: Hit, rank: int) -> str:
    """One hit as two or three indented lines.

    Location is `path:start-end` rather than a bare path because the answer to
    "where is this" is a place in a file, and a reader should be able to paste it
    into an editor without opening the file to go hunting.
    """
    lines = [
        f"{rank:>3}  {TIER_LABELS[tier_of(hit)]}  {hit.qualname}",
        f"        {hit.kind}  {hit.path}:{hit.line_start}-{hit.line_end}"
        f"  score {hit.score:.4f}  [{hit.modality}]"
        + ("  (test)" if hit.is_test else ""),
    ]
    if hit.via:
        # The only account of why graph expansion surfaced this symbol. Retrieval
        # that cannot say why it returned something is hard to trust and harder to
        # improve, so it is shown rather than kept for the debugger.
        lines.append(f"        via {hit.via}")
    return "\n".join(lines)


def count_line(label: str, value: int, width: int = 10) -> str:
    return f"  {label:<{width}} {value:>9,}"
