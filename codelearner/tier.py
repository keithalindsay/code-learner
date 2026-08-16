"""The tier model: what had to be believed to surface a result.

The project's central design claim lives here. It was written in `cli/render.py`,
whose own docstring says "everything below this package is an API. This is the part a
person types" -- so the claim that both surfaces are supposed to agree on was defined
inside one of them, and `server/app.py` imported *upward* into the CLI to reach it.
An MCP tool answering a `facts_only` request through the CLI's presentation layer is
not a layering nit: it means the rule has a home, and the home is the wrong one.

A leaf, deliberately. It imports `ingest.types` for the tier constants only. Its local
protocols describe the retrieval fields it reads, so assigning a tier never makes the
foundational tier model depend on a retrieval package.

`cli/render.py` re-exports every name below, so existing imports keep working. New
callers on either surface should import from here.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

from .ingest.types import TIER_FACT, TIER_INFERRED, TIER_RESOLVED

__all__ = [
    "MODALITY_TIER",
    "TIER_LABELS",
    "facts_only",
    "hit_json",
    "tier_of",
]

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


class _TieredHit(Protocol):
    @property
    def modality(self) -> str: ...


class _RenderableHit(_TieredHit, Protocol):
    # Read-only members, like `modality` above: every hit this renders is a frozen
    # dataclass, and a protocol that declares a settable attribute is not satisfied
    # by one. Declaring them mutable made `hit_json(hit)` a type error at both call
    # sites while the runtime behaviour was correct.
    @property
    def symbol_id(self) -> int: ...
    @property
    def qualname(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def line_start(self) -> int: ...
    @property
    def line_end(self) -> int: ...
    @property
    def score(self) -> float: ...
    @property
    def is_test(self) -> bool: ...
    @property
    def via(self) -> str: ...


_HitT = TypeVar("_HitT", bound=_TieredHit)


def tier_of(hit: _TieredHit) -> int:
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


def facts_only(hits: list[_HitT]) -> list[_HitT]:
    """Drop anything above T1 -- parsed facts and resolved names, nothing asserted.

    Over RETRIEVAL results this removes nothing, and cannot: no modality in
    `MODALITY_TIER` maps to `TIER_INFERRED` except `"inferred"`, and nothing in the
    codebase emits it. It is written as a real filter rather than a no-op so that the
    flag does not quietly start lying the day the inference layer lands. When it
    does, the better place for this is before fusion, so the filtered slots get
    refilled instead of shortening the result list.

    Where the flag is not inert today is `get_symbol`, which returns stored tier-2
    assertions alongside the parsed facts. That path filters on the assertion rows
    directly rather than through this function, because assertions are not `Hit`s --
    but it is the same promise, and it is the surface where T2 actually appears.
    """
    return [hit for hit in hits if tier_of(hit) <= TIER_RESOLVED]


def hit_json(hit: _RenderableHit, rank: int) -> dict[str, Any]:
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
