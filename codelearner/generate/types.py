"""The seam between a model that writes claims and the store that admits them.

Nothing in this module calls a model, and nothing in it writes to the database. It
exists to fix one decision that every other part of the generation path depends on:
**how a generator cites.**

The obvious design is to let the model name its evidence -- a path and a byte range,
or a path and two line numbers. It is also the one design that cannot be made safe.
A model asked for `codelearner/db.py[4120:4380]` will produce something in that shape
whether or not it read those bytes, and an offset it invented does not fail loudly:
it lands somewhere inside a real file, hashes to something stable, and verifies
forever while pointing at nothing the claim is about. The gate in
`assertions.store` would admit it, `servable_assertions` would keep re-hashing it
happily, and the only signal that anything was wrong would be a human eventually
following the citation and finding an unrelated function.

So the model is never given the chance. It is handed a numbered menu of `Offer`s --
each one an `EvidenceSpan` the *index* built, off bytes already on disk -- and it
answers with reference numbers. `Draft.cited_refs` is a tuple of ints, and the only
thing an int can do is name one of the spans that were offered or fall outside the
menu, where it is dropped and counted. Citing a span it was never shown is not
discouraged here, it is unrepresentable.

That moves the failure mode from "invents a citation" to "picks the wrong one from a
real list", which is a strictly better problem: it is visible to the faithfulness
judge, it is attributable to the generator rather than to the store, and the cited
bytes are the bytes a later reader will see.

`Draft` is deliberately not an `Assertion`. A draft has not been through the gate; it
has no id, no status, and no guarantee that any of its references were valid. The
conversion happens in the pipeline, through `write_assertion`, which is the only
function permitted to admit a claim -- so a generator that returns an empty
`cited_refs` produces no row rather than an uncited one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..assertions.store import EvidenceSpan

__all__ = [
    "ClaimGenerator",
    "Draft",
    "GeneratorUnavailable",
    "Offer",
]


class GeneratorUnavailable(RuntimeError):
    """The generator could not be reached, and so did not answer.

    The sibling of `JudgeUnavailable`, and it exists for the same reason. A backend
    that is down must not be mistaken for a backend that declined to make a claim:
    the first is an outage the operator has to fix, the second is a result. Returning
    an empty `Draft` here would write "this symbol has no describable purpose" into
    the run report for every symbol in the repo because ollama was not running.
    """


@dataclass(frozen=True)
class Offer:
    """One numbered candidate citation, built by the index and shown to the model.

    `span` is the real thing -- a hashed `EvidenceSpan` off disk -- and `text` is the
    same bytes rendered for a prompt. They are carried together rather than derived
    from each other at use time so that what the model read and what a reader will
    later verify cannot drift apart.

    `ref` is 1-based because it appears in a prompt, and a model that is told to cite
    `[0]` will reliably cite `[1]` instead.
    """

    ref: int
    span: EvidenceSpan
    text: str
    label: str

    @property
    def citation(self) -> str:
        """`path:start-end`, matching `EvidenceSpan.citation`."""
        return self.span.citation


@dataclass(frozen=True)
class Draft:
    """A claim a generator wants to make, and the offers it says establish it.

    Unadmitted by construction. The pipeline maps `cited_refs` back through the menu
    it handed out, discards any reference that was not on it, and only then calls
    `write_assertion` -- which raises rather than store a claim whose references all
    turned out to be invalid.
    """

    claim: str
    cited_refs: tuple[int, ...]
    kind: str = "purpose"
    confidence: float | None = None

    def resolve(self, offered: Sequence[Offer]) -> tuple[list[EvidenceSpan], list[int]]:
        """Map references onto the spans that were offered.

        Returns the resolved spans and the references that could not be resolved, so
        a caller reports how often the generator cited off the menu instead of
        silently thinning the evidence. Duplicates collapse: a model that cites the
        same span three times has made one citation, and counting it three times
        would make a thinly-evidenced claim look well supported.
        """
        by_ref = {offer.ref: offer.span for offer in offered}
        spans: list[EvidenceSpan] = []
        seen: set[int] = set()
        invalid: list[int] = []
        for ref in self.cited_refs:
            span = by_ref.get(ref)
            if span is None:
                invalid.append(ref)
            elif ref not in seen:
                seen.add(ref)
                spans.append(span)
        return spans, invalid


class ClaimGenerator(Protocol):
    """The swappable backend, mirroring `eval.faithfulness.Judge`.

    Same contract and same motive: every test in this repo runs against a
    deterministic fake, no test calls a model, and `name` becomes the value written
    to `assertions.generator` so a store holding claims from two generators can still
    tell them apart. That column is what makes a before/after comparison possible at
    all -- without it, re-running with a better model overwrites the evidence that
    the worse one was worse.
    """

    @property
    def name(self) -> str: ...

    def draft(self, *, subject: str, offered: Sequence[Offer]) -> Draft: ...
