"""Turning results into text a person reads, and into JSON a program parses.

Both renderings live in one module on purpose. The human table and the `--json`
object drift apart the moment they are derived in two places, and a `--json` shape
that disagrees with what the tool just printed is worse than having no `--json` at
all -- one of them is wrong and nothing says which.

What is NOT here any more is the tier model itself. `TIER_LABELS`, `MODALITY_TIER`,
`tier_of`, `facts_only` and `hit_json` were defined in this file, which meant the
project's central design claim -- what had to be believed to surface a result -- was
owned by the CLI's presentation layer, and `server/app.py` reached *upward* into it
to answer MCP calls. They now live in the leaf `codelearner.tier`, which both
surfaces import as a peer. They are re-exported below so existing imports keep
working; new callers should import from `codelearner.tier`.
"""
from __future__ import annotations

from ..retrieve import Hit
from ..retrieve.types import Candidate, SourceCandidate
from ..tier import MODALITY_TIER, TIER_LABELS, facts_only, hit_json, tier_of

__all__ = [
    # Re-exported from `codelearner.tier` for compatibility. This module is not their
    # home; it is a surface that happens to need them.
    "MODALITY_TIER",
    "TIER_LABELS",
    "count_line",
    "facts_only",
    "format_candidate",
    "format_hit",
    "hit_json",
    "tier_of",
]


def format_hit(hit: Hit | SourceCandidate, rank: int) -> str:
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


def format_candidate(candidate: Candidate, rank: int) -> str:
    """One ranked candidate, with a claim never rendered as if it were a symbol.

    The word `semantic` is on the first line of every tier-2 result on purpose. A
    reader scanning a result list should not have to notice a tier code to tell a
    claim somebody made from a line of code that is actually there, and the
    supporting verdict and freshness are printed with it because a claim without
    them is not something this tool is willing to stand behind.
    """
    if isinstance(candidate, SourceCandidate):
        return format_hit(candidate, rank)

    verdicts = ", ".join(
        f"{verdict.verdict} by {verdict.judge}" for verdict in candidate.verdicts
    ) or "no supporting verdict"
    freshness = "evidence verified" if candidate.freshness.verified else "unverified"
    citations = ", ".join(span.citation for span in candidate.spans)
    lines = [
        f"{rank:>3}  {TIER_LABELS[candidate.tier]}  semantic  "
        f"{candidate.subject_qualname}  ({candidate.kind})",
        f"        {candidate.claim}",
        f"        {verdicts}  |  {freshness}  |  score {candidate.score:.4f}  "
        f"[{candidate.modality}]",
    ]
    if citations:
        lines.append(f"        cites {citations}")
    if candidate.conflict:
        # Two eligible claims of the same kind about the same symbol disagree.
        # Retrieval does not pick a winner, so the reader is told there is one.
        lines.append("        conflicting claim of this kind about this subject")
    return "\n".join(lines)


def count_line(label: str, value: int, width: int = 10) -> str:
    return f"  {label:<{width}} {value:>9,}"
