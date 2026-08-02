"""Negative controls for the tier-2 assertion gate: what it must refuse, measured.

The gate -- `server.app._verify_span` plus `assertions.store.write_assertion` -- is the
load-bearing claim of this project. An inferred claim is shippable only if it cites
evidence that exists and hash-matches. `tests/test_mcp.py` proves that on a handful of
hand-written cases. This module is the other half: an adversarial corpus generated from
every symbol in an indexed repository, one instance per attack per symbol, so the number
reported is a rejection RATE over hundreds of submissions rather than a few examples
somebody happened to think of.

Three properties make that rate mean something, and each is here because its absence is
invisible:

**Positive controls.** A gate that refuses everything scores 100% against a
negative-only corpus. So every symbol is also submitted with the hash the index
published for it, and with the exact lines quoted off disk, and those MUST be admitted.
The positive pass rate is reported beside the rejection rate; either number alone is
unfalsifiable.

**Attribution.** Each negative control names the refusal code it expects. A harness
pointed at the wrong repo root refuses every submission with `file_missing` and scores
a perfect 100%; requiring `bad_range` from the beyond-EOF attack and `hash_mismatch`
from the stale one makes that failure surface as a wrong code rather than a right rate.

**Mutation.** Every family names the gate rule it targets, as the textual edit that
removes it. `run_mutation` copies the package to a temp tree, applies that edit to the
COPY, and re-runs the family in a subprocess importing off PYTHONPATH. A control that
keeps holding when its own rule is deleted is decoration, and the run says so by name.
The working tree is never edited: mutating an installed package in place leaves a window
in which every other process importing it sees the hole, and a crash mid-run leaves the
hole behind for good.

**Two surfaces, because there are two doors.** This corpus used to import
`server.app` and nothing else, so every rate it published described the MCP path
alone -- and `codelearner learn`, the eval harness, and every other library caller
reach the store without passing through it. The same corpus now runs against
`store.write_assertion` directly as well, and the report has two columns. The columns
are not expected to agree, and where they differ the difference is the finding:

* the same attack is often refused by a DIFFERENT rule at each door (a citation of a
  file the index never parsed is `file_missing` at the server and `evidence_stale` at
  the store), so acceptable codes and mutations are declared per gate rather than per
  family -- collapsing them into a union would let a refusal for the wrong reason
  score as attribution, which is the one thing the codes exist to prevent;
* one attack is expressible only at the store (`zero_length_span`: the server is given
  line numbers and derives the bytes itself, so no caller can ask it for an empty
  range); and
* one attack the server refused was **not refused at the store at all**. Running the
  corpus at the second door found `escaping_path` admitted, stored `active`, and
  reported servable on every one of 12,803 generated instances -- which, measured the
  way this module now measures (see "the resolution of a rate"), is ONE probe repeated
  12,803 times: the gate did the same thing to all of them. It was recorded as an
  `Unenforced` gate entry -- controls still generated, still submitted, still scored as
  the failures they were -- because a corpus that declines to generate a control
  reports a hole as an absence. `store.SpanEscapesRepo` closed it, and closing it
  FAILED the test that pinned the gap set exactly, which is what that entry is for: a
  declared gap is a liability the suite makes you settle, not a footnote it lets you
  keep. `Unenforced` stays in the vocabulary for the next one.

**Sized, not just measured.** The pooled figure this module used to lead with --
`12,266 attacks / 1.0000` -- was never wrong and was reported at a resolution the
corpus cannot support. It is fifteen attack SHAPES instantiated once per symbol, and
for most of them the gate does literally the same thing on every instance: the same
lines run, the same bytes (usually none) are read, the same `raise` fires. Four
significant figures over a numerator that can only be 0 or n does not describe twelve
thousand independent adversarial probes; it describes a handful of probes photocopied.

So the per-family table is the output and the pooled figure is a caption under it, each
family carries the number of DISTINCT gate executions its instances produced, and every
bound is computed on that rather than on the submission count. The classification is
measured at the gate and not guessed from the family name -- `Execution` explains how
and what it does and does not establish, and it is the only thing that can say that
`unverifiable_span` is one probe at the store and n probes at the server, which is true
and which no reading of the family table would give you.

The attacks are the ways a confident model gets a citation wrong, not arbitrary
malformed input -- content it remembers from the wrong file, a range that runs one line
past the end, the hash of the symbol next door, and, the one that matters most, a hash
that WAS correct before the file changed. That last is the only attack whose evidence
was ever real, so it is the only one that a gate trusting its own stored hashes instead
of the bytes on disk will admit. It is also the only one that cannot be caught by
looking at the submission alone.

What this does NOT measure: whether a span supports the claim it is attached to. That is
adjudication (`store.record_verdict`), it needs a judge, and no amount of arithmetic
substitutes for it. Everything here is arithmetic, which is exactly why a confident
model cannot argue with it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import db
from ..ingest import index_repo
from ..ingest.types import content_hash

# sha256 of nothing. A blank line hashes to this, which is why an empty span must be
# refused by range rather than by hash: it verifies forever while pointing at nothing.
EMPTY_SHA = content_hash(b"")

REFUSED = "refused"
ACCEPTED = "accepted"
RAISED = "raised"

# Where the mutation runner finds the two files that hold the gate, relative to the
# `codelearner` package root.
GATE_MODULE = "server/app.py"
STORE_MODULE = "assertions/store.py"

# The two gates, and the three ways of reaching them.
#
# A GATE is a body of rules: `server.app` (`_verify_span` plus `_submit_body`) and
# `assertions.store` (`write_assertion`). A SURFACE is a way in. `direct` calls the
# tool body under `_guard`; `tool` goes through the registered MCP tool as a client
# would; `store` calls `write_assertion` with `EvidenceSpan`s, which is what
# `codelearner learn` and every other library caller does. The first two share a gate
# and must agree control for control -- that is a test. The third does not share it,
# and the whole reason this module now has a surface axis is that until WP4 the third
# door had one lock on it.
GATE_SERVER = "server"
GATE_STORE = "store"

SURFACE_DIRECT = "direct"
SURFACE_TOOL = "tool"
SURFACE_STORE = "store"

SURFACES: tuple[str, ...] = (SURFACE_DIRECT, SURFACE_TOOL, SURFACE_STORE)

# Which body of rules each surface actually reaches. `tool` maps to `server` rather
# than getting an entry of its own on purpose: an attack refused by a different code
# depending on whether the call arrived over the transport would be a contract bug,
# not a second gate, and giving it its own code set is exactly how such a bug would
# stop being visible.
SURFACE_GATES: dict[str, str] = {
    SURFACE_DIRECT: GATE_SERVER,
    SURFACE_TOOL: GATE_SERVER,
    SURFACE_STORE: GATE_STORE,
}


def gate_of(surface: str) -> str:
    """Which gate a surface reaches. Raises on a surface nobody declared.

    A `KeyError` here is better than a default, because the default would be
    `GATE_SERVER` and a mistyped surface would then be silently scored against the
    wrong rule set -- reporting attribution for codes the door it actually used could
    never produce.
    """
    return SURFACE_GATES[surface]


class VacuousCorpus(Exception):
    """A rate was asked of an empty set.

    Refusing to answer is the point. `refused / len(negatives)` is 1.0 for an empty
    corpus under every convention that does not raise, and a fixture that silently
    generated no controls would then report a perfect gate. This project has already
    shipped three tests that passed while asserting nothing; this is the one place where
    that failure is cheap to make structurally impossible.
    """


class MutationFailed(Exception):
    """A mutation could not be applied, or was applied to the wrong tree.

    Distinct from "the mutation was not detected". A snippet that no longer matches the
    source means the rule moved and the control is now aimed at nothing -- which reads
    exactly like a passing mutation check if it is allowed to pass silently.
    """


# ---------------------------------------------------------------------------
# the gate, imported late
# ---------------------------------------------------------------------------

def gate_module() -> Any:
    """The MCP server module, imported at call time rather than at import time.

    Two reasons, both load-bearing. `mcp` is an optional extra (see pyproject: a user
    who wants `codelearner search` in a shell should not install an ASGI stack), so
    importing `codelearner.eval` must not require it. And the mutation runner depends on
    nothing here having bound the installed module before the child process gets to
    choose a copied one off PYTHONPATH.
    """
    from ..server import app

    return app


def store_module() -> Any:
    """The store, which is the gate the `store` surface measures.

    Imported at call time for the second of `gate_module`'s two reasons only -- the
    mutation runner must not find this module already bound to the installed package
    when the child process is meant to import a copied one. The first reason does not
    apply: `assertions.store` has no optional dependency, which is the entire point of
    measuring it separately. A gate reachable without `mcp` installed is a gate whose
    rate cannot be reported by a corpus that needs `mcp` to run.
    """
    from ..assertions import store

    return store


# Every way the store can refuse an admission, and the code the corpus scores it by.
# Deliberately a SECOND copy of `server.app._STORE_REFUSAL_CODES` rather than an import
# of it: the store surface exists to be measurable without the MCP extra installed, and
# importing the server to name the store's refusals would make the measurement depend on
# the very thing it is measuring around. Two copies can drift, so `test_gate_controls`
# asserts they are identical -- which is a cheaper coupling than the import, and a
# louder one than a shared constant, because it fails by name rather than by absence.
def store_refusal_codes() -> dict[type[BaseException], str]:
    store = store_module()
    return {
        store.EvidenceRequired: "evidence_required",
        store.EmptyClaim: "empty_claim",
        store.InvalidSpan: "invalid_span",
        store.EvidenceUnverifiable: "evidence_unverifiable",
        store.UnknownSubject: "unknown_subject",
        store.EvidenceStale: "evidence_stale",
        store.SpanEscapesRepo: "span_escapes_repo",
    }


# ---------------------------------------------------------------------------
# the shapes fixture
# ---------------------------------------------------------------------------

# Small on purpose, and not arbitrary: between them these two files contain every shape
# where a symbol's stored bytes are NOT its lines' bytes -- a decorated method (`@memoize`
# puts the symbol's first byte a line above its `def`), a property (four columns in from
# the start of its line), and two modules (whose span runs one line past the last line
# anything is written on).
#
# That population is not an edge case. Measured over this repository, tests included,
# with `--repo .`: 217 of 850 symbols, 25.5%, split as every method (143/143, because an
# indented symbol begins at its `def` and its line begins at column 0), every module
# (51/51), 19 of 584 functions and 4 of 72 classes (the decorated ones). Those are the
# symbols a too-narrow gate FALSELY REJECTS while still refusing every attack, which is
# the failure that teaches an agent the gate is noise -- and the reason the positive
# controls below are not a formality. Both files also hold a blank line and more than one
# symbol, which the blank-range and foreign-hash attacks need.
SHAPES: dict[str, str] = {
    "core.py": (
        'def frobnicate_widgets():\n'
        '    """Frobnicate every widget on the tray."""\n'
        '    return _plumbing()\n'
        '\n'
        '\n'
        'def _plumbing():\n'
        '    """Detail."""\n'
        '    return 42\n'
    ),
    "tray.py": (
        'import functools\n'
        '\n'
        '\n'
        'def memoize(fn):\n'
        '    """Cache a nullary method."""\n'
        '    return functools.cache(fn)\n'
        '\n'
        '\n'
        'class Tray:\n'
        '    """A tray of widgets."""\n'
        '\n'
        '    @property\n'
        '    def widgets(self):\n'
        '        return self._widgets\n'
        '\n'
        '    @memoize\n'
        '    def count(self):\n'
        '        return len(self._widgets)\n'
    ),
}

# Written OUTSIDE the indexed repo, one directory up. The escaping-path attack cites it
# with its real hash, so the only thing standing between the gate and a claim about the
# host filesystem is the repo-containment check.
OUTSIDE_NAME = "outside_the_repo.py"
OUTSIDE_SOURCE = 'SECRET = "hunter2"\nAPI_KEY = "sk-not-a-real-key"\n'


# ---------------------------------------------------------------------------
# facts about the indexed repository
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolFact:
    """One indexed symbol, exactly as retrieval would hand it to an agent."""

    qualname: str
    kind: str
    path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    content_hash: str


@dataclass
class FileFact:
    """One indexed file as it reads on disk, plus the symbols the index put in it."""

    path: str
    source: bytes
    symbols: list[SymbolFact] = field(default_factory=list)

    @property
    def line_starts(self) -> list[int]:
        starts = [0]
        idx = self.source.find(b"\n")
        while idx != -1:
            starts.append(idx + 1)
            idx = self.source.find(b"\n", idx + 1)
        return starts

    @property
    def line_count(self) -> int:
        starts = self.line_starts
        return len(starts) - 1 if self.source.endswith(b"\n") else len(starts)

    def line_bytes(self, line_start: int, line_end: int) -> tuple[int, int] | None:
        """Byte range of an inclusive 1-based line range, or None if it is not one.

        Deliberately a second implementation of `app._line_bytes` rather than a call to
        it. This is what decides whether a control is adversarial -- whether the decoy
        hash it cites is genuinely not one the gate should accept -- and a guard that
        asked the gate that question would inherit whatever the gate got wrong.
        """
        starts = self.line_starts
        if line_start < 1 or line_end < line_start or line_end > self.line_count:
            return None
        byte_start = starts[line_start - 1]
        byte_end = starts[line_end] - 1 if line_end < len(starts) else len(self.source)
        return byte_start, byte_end

    def text_at(self, line_start: int, line_end: int) -> str | None:
        span = self.line_bytes(line_start, line_end)
        if span is None:
            return None
        return self.source[span[0]:span[1]].decode()

    def blank_lines(self) -> list[int]:
        """1-based line numbers whose bytes are empty -- nothing, not even a space."""
        return [
            n
            for n in range(1, self.line_count + 1)
            if (span := self.line_bytes(n, n)) is not None and span[0] == span[1]
        ]

    def accepted_hashes(self, line_start: int, line_end: int) -> set[str]:
        """Every hash the gate is entitled to accept for this line range, right now.

        Both honest readings: any indexed symbol occupying exactly these lines, and the
        whole lines themselves. A negative control whose cited hash lands in this set is
        not an attack -- it is a positive control wearing an attack's name, and it would
        pass while proving nothing.
        """
        hashes = {
            content_hash(self.source[s.byte_start:s.byte_end])
            for s in self.symbols
            if s.line_start == line_start and s.line_end == line_end
        }
        span = self.line_bytes(line_start, line_end)
        if span is not None and span[0] < span[1]:
            hashes.add(content_hash(self.source[span[0]:span[1]]))
        return hashes


def _load_facts(conn: sqlite3.Connection, repo: Path) -> dict[str, FileFact]:
    files: dict[str, FileFact] = {}
    rows = conn.execute(
        "SELECT s.qualname, s.kind, f.path, s.line_start, s.line_end, s.byte_start, "
        "       s.byte_end, s.content_hash "
        "FROM symbols s JOIN files f ON f.id = s.file_id ORDER BY s.id"
    ).fetchall()
    for row in rows:
        path = str(row["path"])
        if path not in files:
            files[path] = FileFact(path=path, source=(repo / path).read_bytes())
        files[path].symbols.append(
            SymbolFact(
                qualname=str(row["qualname"]),
                kind=str(row["kind"]),
                path=path,
                line_start=int(row["line_start"]),
                line_end=int(row["line_end"]),
                byte_start=int(row["byte_start"]),
                byte_end=int(row["byte_end"]),
                content_hash=str(row["content_hash"]),
            )
        )
    return files


# ---------------------------------------------------------------------------
# the rules, and the edits that remove them
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edit:
    """One textual substitution in one file of a COPY of the package."""

    target: str
    old: str
    new: str


@dataclass(frozen=True)
class Mutation:
    """A gate rule, named, plus the edit(s) that delete it from a copied tree."""

    rule: str
    edits: tuple[Edit, ...]


@dataclass(frozen=True)
class Rule:
    """What ONE gate refuses this attack with, and the edit(s) that delete it there.

    Codes and mutation travel together because they are the same claim stated twice:
    "this door says no, by this name, because of this code". Splitting them would
    allow a family to name a code it has no mutation for, which is a control that
    asserts an outcome without establishing that anything produces it.
    """

    codes: frozenset[str]
    mutation: Mutation


@dataclass(frozen=True)
class Unenforced:
    """This gate does NOT refuse this attack. A measured hole, named on purpose.

    The controls are still generated and still submitted, and they still score as
    failures -- because they are failures. The alternative, declining to generate them
    for this surface, is how a corpus reports a gap as absent: the family would simply
    not appear in that column, and a reader comparing the two columns would see a
    shorter list rather than a hole. `detail` says what gets through and where the rule
    it needs would have to live.
    """

    detail: str


@dataclass(frozen=True)
class Inexpressible:
    """This attack cannot be PUT to this gate at all, and here is why.

    Distinct from `Unenforced` in the direction that matters. Unenforced means the
    submission arrives and is admitted; inexpressible means there is no submission to
    make, because the surface's vocabulary cannot state the attack -- the MCP tool
    takes line numbers and derives the byte range itself, so no caller can hand it an
    empty range. Recording the reason is the whole point: amendment #2 of the
    remediation plan is that this corpus can only test itself against situations it
    already knows how to generate, so the situations it CANNOT generate have to be
    written down rather than inferred from an absence.
    """

    reason: str


GateEntry = Rule | Unenforced | Inexpressible


@dataclass(frozen=True)
class Family:
    """One attack (or one legitimate submission), and what each gate does with it.

    `gates` must name every gate, with no default. A family that simply omitted a gate
    would be scored against nothing there and would vanish from that column -- the
    exact failure `Unenforced` and `Inexpressible` exist to make loud. The
    completeness check is a test, because a dataclass cannot enforce it.
    """

    name: str
    expect: str
    attack: str
    gates: dict[str, GateEntry]

    def at(self, surface: str) -> GateEntry:
        """What the gate behind `surface` does with this attack."""
        return self.gates[gate_of(surface)]

    def rule(self, surface: str) -> Rule:
        """The rule this attack meets at `surface`, or a refusal to pretend there is one."""
        entry = self.at(surface)
        if not isinstance(entry, Rule):
            raise MutationFailed(
                f"{self.name!r} has no rule at the {gate_of(surface)!r} gate "
                f"({type(entry).__name__}: {getattr(entry, 'detail', None) or entry.reason}). "  # type: ignore[union-attr]
                "There is nothing to delete, so there is nothing to detect."
            )
        return entry

    def codes(self, surface: str) -> frozenset[str]:
        """The refusal codes this family accepts at `surface`. Empty where nothing refuses."""
        entry = self.at(surface)
        return entry.codes if isinstance(entry, Rule) else frozenset()


def _mut(rule: str, *edits: Edit) -> Mutation:
    return Mutation(rule=rule, edits=edits)


# ---------------------------------------------------------------------------
# the edits, named once and shared
# ---------------------------------------------------------------------------
#
# Hoisted out of the family table because after WP4 several rules have TWO homes and
# several families target ONE rule, so the table would otherwise repeat the same
# snippet four times and a drift fix would have to find every copy. Each is a rule, not
# a line: `E_STORE_VERIFY` deletes the store's whole re-verification step, which is the
# single rule that catches four different attacks at that door.

E_STORE_EVIDENCE_REQUIRED = Edit(
    target=STORE_MODULE,
    old=(
        "    spans = tuple(spans)\n"
        "    if not spans:\n"
        "        raise EvidenceRequired(\n"
        "            f\"assertion about {subject_qualname!r} cites no evidence spans and was \"\n"
        "            \"not written. An uncited claim cannot be adjudicated, cannot expire, \"\n"
        "            \"and cannot be checked by a reader -- it is indistinguishable from a \"\n"
        "            \"good one at every stage after this.\"\n"
        "        )\n"
    ),
    new="    spans = tuple(spans)\n",
)

E_STORE_EMPTY_CLAIM = Edit(
    target=STORE_MODULE,
    old=(
        "    if not claim.strip():\n"
        "        raise EmptyClaim(\n"
        "            f\"assertion about {subject_qualname!r} carries no claim text and was not \"\n"
        "            \"written. Its citations may be perfect; there is still no proposition \"\n"
        "            \"here for a judge to adjudicate or a reader to disagree with, and every \"\n"
        "            \"check after this one would pass.\"\n"
        "        )\n"
    ),
    new="",
)

E_STORE_EMPTY_RANGE = Edit(
    target=STORE_MODULE,
    old="        if not 0 <= span.byte_start < span.byte_end:",
    new="        if False:  # mutated: the write gate's empty-span rule is gone",
)

E_STORE_SPAN_FOR_RANGE = Edit(
    target=STORE_MODULE,
    old="    if not 0 <= byte_start < byte_end <= len(source):",
    new="    if not 0 <= byte_start <= byte_end <= len(source):",
)

E_STORE_NO_HASH = Edit(
    target=STORE_MODULE,
    old="        if not span.content_hash.strip():",
    new="        if False:  # mutated: the store's unverifiable-span rule is gone",
)

E_STORE_ESCAPES_REPO = Edit(
    target=STORE_MODULE,
    old="        if _escapes_repo(span.path):",
    new="        if False:  # mutated: the store's containment rule is gone",
)

E_STORE_UNKNOWN_SUBJECT = Edit(
    target=STORE_MODULE,
    old="    if not allow_unindexed_subject:",
    new="    if False:  # mutated: the store's subject-existence rule is gone",
)

E_STORE_VERIFY = Edit(
    target=STORE_MODULE,
    old="    if verify:",
    new="    if False:  # mutated: the store no longer re-reads the cited bytes",
)

E_STORE_CITED_BYTES = Edit(
    target=STORE_MODULE,
    old="        observed = content_hash(source[span.byte_start:span.byte_end])",
    new="        observed = content_hash(source)  # mutated: the whole file, not the span",
)

E_STORE_EVERY_SPAN = Edit(
    target=STORE_MODULE,
    old="                for s in spans\n",
    new="                for s in spans[:1]  # mutated: only the first span is stored\n",
)

E_SERVER_UNVERIFIABLE = Edit(
    target=GATE_MODULE,
    old=(
        "    else:\n"
        "        raise ToolError(\n"
        "            \"evidence_unverifiable\",\n"
        "            f\"the span {candidates[-1].citation} carries neither content_hash nor \"\n"
        "            \"text, so there is nothing to check it against. Pass the content_hash \"\n"
        "            \"that search_code or get_symbol returned for this symbol, or the exact \"\n"
        "            \"source text you read at those lines.\",\n"
        "            path=raw.path,\n"
        "            line_start=raw.line_start,\n"
        "            line_end=raw.line_end,\n"
        "        )\n"
    ),
    new="    else:\n        cited = candidates[-1].content_hash  # mutated: nothing to check is fine\n",
)


# Every `old` below is checked for EXACTLY ONE occurrence before it is applied, and a
# miss raises rather than skipping. A snippet that has drifted out of the source means
# the rule moved and the control now points at nothing -- which looks identical to a
# control that cannot detect its rule being deleted, unless the harness refuses to
# proceed.
FAMILIES: dict[str, Family] = {
    "zero_evidence": Family(
        name="zero_evidence",
        expect=REFUSED,
        attack="a claim with no citations at all -- the plausible-sounding summary",
        gates={
            # One rule, one home, and it was already at the chokepoint before WP4 --
            # which is why both doors refuse it identically. The family is the
            # baseline the other columns are read against: where the two gates
            # disagree below, the disagreement is about where a rule lives, not about
            # what the corpus submitted.
            gate: Rule(
                codes=frozenset({"evidence_required"}),
                mutation=_mut(
                    "store.write_assertion refuses an empty span list before opening "
                    "a transaction",
                    E_STORE_EVIDENCE_REQUIRED,
                ),
            )
            for gate in (GATE_SERVER, GATE_STORE)
        },
    ),
    "empty_claim": Family(
        name="empty_claim",
        expect=REFUSED,
        attack="perfect citations carrying no statement -- whitespace where the claim goes",
        gates={
            # Added by WP5 against a rule WP4 created. Every arithmetic check the gate
            # makes passes on this submission: the span exists, it hash-matches, the
            # subject is real. Before WP4 it stored `active`, reported servable, and
            # was handed back next to the code it was allegedly about, saying nothing.
            # One home, in the store, so the server column measures the store's rule
            # reached through the tool -- and that is worth stating rather than hiding,
            # because it means `_submit_body` contributes nothing here and a reader
            # comparing the columns should not conclude the server has a rule of its own.
            gate: Rule(
                codes=frozenset({"empty_claim"}),
                mutation=_mut(
                    "store.write_assertion refuses a claim that says nothing, however "
                    "good its citations",
                    E_STORE_EMPTY_CLAIM,
                ),
            )
            for gate in (GATE_SERVER, GATE_STORE)
        },
    ),
    "unverifiable_span": Family(
        name="unverifiable_span",
        expect=REFUSED,
        attack="a citation with nothing to check it against -- no hash, no quoted text",
        gates={
            # The vacuous-truth failure one level down from `servable_assertions`'s
            # `no_evidence` guard: a span that asserts nothing about what is at those
            # bytes can never be found to be wrong, so it reports fresh evidence
            # forever. TWO homes, and this is the family amendment #3 predicts: the
            # server refuses it in `_verify_span` and the store refuses it again in
            # `write_assertion`, so the server-gate mutation deletes both. Deleting
            # only the server's leaves the store refusing the same attack -- though
            # not, as it happens, with the same code, because by then the server has
            # substituted the observed hash and the store's rule cannot fire at all.
            # That is a worse trap than the identical-code case WP4 hit: the rule is
            # bypassed by construction rather than duplicated, so the single-edit
            # mutation WOULD flip and the second home would go unmeasured.
            GATE_SERVER: Rule(
                codes=frozenset({"evidence_unverifiable"}),
                mutation=_mut(
                    "_verify_span refuses a span carrying neither content_hash nor "
                    "text, and write_assertion refuses one carrying no hash",
                    E_SERVER_UNVERIFIABLE,
                    E_STORE_NO_HASH,
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset({"evidence_unverifiable"}),
                mutation=_mut(
                    "write_assertion refuses a span with no content hash to check it "
                    "against",
                    # Deleting this does NOT admit the attack: the empty hash then
                    # falls through to the re-verification step and is refused as
                    # `evidence_stale`. The control still flips -- a code outside this
                    # family's set is not a hold -- and the flip is what the mutation
                    # measures. Recorded here because "detected" and "admitted" are
                    # not the same outcome, and a reader of the mutation table is
                    # entitled to know which one this row is.
                    E_STORE_NO_HASH,
                ),
            ),
        },
    ),
    "absent_file": Family(
        name="absent_file",
        expect=REFUSED,
        attack="the right content cited in a file that does not exist",
        gates={
            # Still exactly one code, and that is now load-bearing rather than
            # incidental. WP2 put an index-membership check ahead of the read, so this
            # control is refused before anything is stat'd -- and it is refused with
            # the SAME code and the same shape as a path that is present on disk but
            # unindexed. A second code would have made the refusal answer "does this
            # file exist", which is the oracle WP2 exists to close. If this set ever
            # grows, check that the new code is not distinguishing two paths a caller
            # is not entitled to tell apart.
            GATE_SERVER: Rule(
                codes=frozenset({"file_missing"}),
                mutation=_mut(
                    "_verify_span refuses a citation it cannot, or will not, read off "
                    "disk",
                    # Four edits, one rule. Defence in depth means no single deletion
                    # admits the attack, so a mutation that removed only one guard
                    # would report this control as undetectable when it is in fact
                    # over-defended. Each `old` is unique in the module, so drift in
                    # the surrounding comments cannot silently turn this into a no-op
                    # -- the harness raises on a snippet that no longer matches
                    # exactly once.
                    Edit(
                        target=GATE_MODULE,
                        old='    if conn.execute("SELECT 1 FROM files WHERE path = ?", (raw.path,)).fetchone() is None:',
                        new="    if False:  # mutated: the index-membership guard is gone",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old="    if not target.is_file():",
                        new="    if False:  # mutated: the regular-file guard is gone",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old="        size = target.stat().st_size",
                        new="        size = 0  # mutated: the size ceiling is gone",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    except OSError as exc:\n"
                            "        raise ToolError(\n"
                            "            \"file_missing\",\n"
                            "            f\"cannot read {raw.path!r} ({exc}). Cite a file that exists in the \"\n"
                            "            \"indexed repository.\",\n"
                            "            path=raw.path,\n"
                            "        ) from exc"
                        ),
                        new=(
                            "    except OSError:\n"
                            "        return store.EvidenceSpan(\n"
                            "            path=raw.path,\n"
                            "            line_start=raw.line_start,\n"
                            "            line_end=raw.line_end,\n"
                            "            byte_start=0,\n"
                            "            byte_end=1,\n"
                            "            content_hash=raw.content_hash\n"
                            "            or content_hash((raw.text or \"\").encode()),\n"
                            "        )"
                        ),
                    ),
                    # The store's own copy of the refusal, without which the four
                    # edits above leave `write_assertion` refusing the fabricated span
                    # as `evidence_stale` -- a flip, but of a rule this family does not
                    # name at this gate.
                    E_STORE_VERIFY,
                ),
            ),
            # A DIFFERENT rule and a different code at the other door, and the gap is
            # the interesting part. The store has no notion of "a file this index
            # parsed" -- it has never seen the `files` table -- so an unreadable
            # citation is caught only by the re-read, as `evidence_stale`. That is
            # weaker than `file_missing` in one specific way worth writing down: it
            # cannot distinguish "this path was never indexed" from "these bytes
            # moved", so a library caller gets a message blaming an edit for a path
            # that never existed.
            GATE_STORE: Rule(
                codes=frozenset({"evidence_stale"}),
                mutation=_mut(
                    "write_assertion re-reads every cited span off disk before "
                    "admitting it, and a file it cannot read is a span that does not "
                    "verify",
                    E_STORE_VERIFY,
                ),
            ),
        },
    ),
    "escaping_path": Family(
        name="escaping_path",
        expect=REFUSED,
        attack="a real file, really hashed, from outside the indexed repository",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"path_escapes_repo"}),
                mutation=_mut(
                    "the gate refuses a path that leaves the indexed repository -- in "
                    "_verify_span, which resolves it against the root, and again in "
                    "store.write_assertion, which reads it lexically at the door",
                    # THREE edits, one rule, and the third bite of amendment #3. This
                    # one is not a duplicated rule, like `unknown_subject`, nor a
                    # bypassed one, like `unverifiable_span`. It is STACKED: deleting
                    # the containment check alone leaves the attack refused as
                    # `file_missing` by WP2's index-membership guard -- measured, not
                    # assumed, at 8 of 8 -- so the family would report "detected" on
                    # the strength of a rule it does not name. The two cannot be
                    # separated by any submission, because every path that escapes the
                    # repository is by construction a path the index did not parse.
                    #
                    # So the mutation removes both server guards and the store's copy,
                    # and the attack is then genuinely ADMITTED. That is a weaker claim
                    # than "containment did the work" and it is the strongest claim
                    # available here: what it establishes is that the rule doing the
                    # work is inside this set, and that nothing outside it catches the
                    # attack. `absent_file` makes the same trade for the same reason.
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    if not target.is_relative_to(root.resolve()):\n"
                            "        raise ToolError(\n"
                            "            \"path_escapes_repo\",\n"
                            "            f\"{raw.path!r} resolves outside the indexed repository. Citations must \"\n"
                            "            \"be repo-root-relative paths.\",\n"
                            "            path=raw.path,\n"
                            "        )\n"
                        ),
                        new="",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old='    if conn.execute("SELECT 1 FROM files WHERE path = ?", (raw.path,)).fetchone() is None:',
                        new="    if False:  # mutated: the index-membership guard is gone",
                    ),
                    E_STORE_ESCAPES_REPO,
                ),
            ),
            # THE hole this work package existed to find, and it was not a hypothesis:
            # submitted directly to `write_assertion`, a claim citing
            # `../outside_the_repo.py` with that file's real hash was ADMITTED, stored
            # `active`, and reported servable. WP4 moved five rules to the chokepoint
            # and repo containment was not among them, so it was a rule the MCP caller
            # met and `codelearner learn` did not.
            #
            # It was recorded here as an `Unenforced` entry first -- controls generated,
            # submitted, and scored as the failures they were, because a corpus that
            # declines to generate a control reports a hole as an absence. Closing it
            # then failed the test asserting `STORE_GAPS` exactly, which is the
            # behaviour that entry was built for: a declared gap is a liability the
            # suite makes you settle, not a footnote it lets you keep.
            #
            # The store's rule is deliberately LEXICAL where the server's resolves. It
            # runs at the door, and `verify=False` is a supported call, so a containment
            # check that needed the disk would be a rule only the re-reading caller met
            # -- which is the shape of the bug being fixed. A symlink out of the tree
            # still passes it and is caught downstream at verification; that residue is
            # stated rather than papered over.
            GATE_STORE: Rule(
                codes=frozenset({"span_escapes_repo"}),
                mutation=_mut(
                    "write_assertion refuses a citation that leaves the repository",
                    # One edit here, and the asymmetry with the server's three is the
                    # measurement: nothing else at this door catches the attack. Delete
                    # this line and the claim is admitted, stored `active`, and
                    # reported servable -- which is the state the corpus found and
                    # which this mutation now re-creates on demand.
                    E_STORE_ESCAPES_REPO,
                ),
            ),
        },
    ),
    "past_eof": Family(
        name="past_eof",
        expect=REFUSED,
        attack="a range running one line past the end, quoting the last line that exists",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"bad_range"}),
                mutation=_mut(
                    "_line_bytes refuses a line range the file does not have (rather "
                    "than clamping it)",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    if line_start < 1 or line_end < line_start or line_end > line_count:\n"
                            "        raise ToolError(\n"
                            "            \"bad_range\",\n"
                            "            f\"lines {line_start}-{line_end} are not a valid range in a \"\n"
                            "            f\"{line_count}-line file.\",\n"
                            "            line_count=line_count,\n"
                            "        )"
                        ),
                        new="    line_end = min(line_end, line_count)",
                    ),
                ),
            ),
            # The store is handed bytes, not lines, so "one line past the end" reaches
            # it as a byte range whose end is past the end of the file. There is no
            # line-range rule to fire; the truncation branch inside `_first_failure`
            # catches it, which is why the code is `evidence_stale` and why the
            # mutation is the whole re-verification step rather than that one branch:
            # deleting the branch alone leaves the short slice hashing to something
            # that is simply not the cited hash, and the attack is refused anyway.
            GATE_STORE: Rule(
                codes=frozenset({"evidence_stale"}),
                mutation=_mut(
                    "write_assertion re-reads the cited bytes, and a range that runs "
                    "off the end of the file does not verify",
                    E_STORE_VERIFY,
                ),
            ),
        },
    ),
    "blank_range": Family(
        name="blank_range",
        expect=REFUSED,
        attack="a blank line cited as evidence, with the hash of nothing",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"bad_range"}),
                mutation=_mut(
                    "the store refuses an empty byte range -- in span_for, which "
                    "builds the citation, and again at the write gate, which admits it",
                    # Two edits, one rule, for the reason `absent_file` gives at
                    # length: after WP4 the empty-span rule lives in two places, and a
                    # mutation that deleted only `span_for`'s copy would leave
                    # `write_assertion` refusing the attack with `invalid_span`. That
                    # still flips the control -- a code outside this family's set is
                    # not a hold -- but it would be measuring the wrong rule, and the
                    # family would quietly become a test of the write gate under the
                    # name of a test of the constructor.
                    E_STORE_SPAN_FOR_RANGE,
                    E_STORE_EMPTY_RANGE,
                ),
            ),
            # `span_for` never runs on this path -- the corpus hands the store the
            # byte range directly -- so only the write gate's copy is in play, and it
            # answers with its own code rather than the server's `bad_range`.
            GATE_STORE: Rule(
                codes=frozenset({"invalid_span"}),
                mutation=_mut(
                    "write_assertion refuses a span that covers no bytes",
                    E_STORE_EMPTY_RANGE,
                ),
            ),
        },
    ),
    "zero_length_span": Family(
        name="zero_length_span",
        expect=REFUSED,
        attack="an empty byte range at a line that is NOT blank -- sha256 of nothing, "
               "cited as evidence for a symbol that really is there",
        gates={
            # Reproduced before WP4 as admitted, servable, and permanently fresh: an
            # empty range hashes to a stable value, so it re-verifies against the file
            # as it is, as it becomes, and as it would be after the symbol it points
            # at is deleted. Not merely unfalsifiable -- it positively reports `fresh`
            # on every read, which is the strongest evidence the store can express.
            #
            # Distinct from `blank_range`, which cites a line that HAS no bytes. This
            # one cites a line full of code and asks for none of it, which is what a
            # caller computing offsets wrongly actually produces.
            GATE_SERVER: Inexpressible(
                reason=(
                    "the MCP surface is given line numbers and derives the byte range "
                    "itself, so no caller can ask it for an empty range at a non-blank "
                    "line -- the only empty range expressible there is a blank line, "
                    "which is the `blank_range` family. This attack reaches the store "
                    "only from a caller that constructs EvidenceSpan directly, which "
                    "is how it was reproduced and is why the rule had to move."
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset({"invalid_span"}),
                mutation=_mut(
                    "write_assertion refuses a span that covers no bytes, whatever is "
                    "at the line it names",
                    E_STORE_EMPTY_RANGE,
                ),
            ),
        },
    ),
    "decoy_content_hash": Family(
        name="decoy_content_hash",
        expect=REFUSED,
        attack="the right lines cited with the hash of other content in the same file",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"hash_mismatch"}),
                mutation=_mut(
                    "_verify_span compares the cited hash to the bytes that are there",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    for span in candidates:\n"
                            "        if cited == span.content_hash:\n"
                            "            return span\n"
                        ),
                        new="    if candidates:\n        return candidates[-1]\n",
                    ),
                    # Without this the server hands `write_assertion` a span carrying
                    # the bytes' real hash, so the store's re-read agrees and the
                    # attack is admitted anyway -- the mutation would flip, but the
                    # store's copy of the rule would never have been exercised.
                    E_STORE_VERIFY,
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset({"evidence_stale"}),
                mutation=_mut(
                    "write_assertion hashes the bytes on disk and compares them to "
                    "what was cited",
                    E_STORE_VERIFY,
                ),
            ),
        },
    ),
    "stale_but_once_valid": Family(
        name="stale_but_once_valid",
        expect=REFUSED,
        attack="a hash that was correct before the file changed under it",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"hash_mismatch"}),
                mutation=_mut(
                    "_verify_span hashes the bytes on disk NOW, never the index's "
                    "stored hash",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    for span in candidates:\n"
                            "        if cited == span.content_hash:\n"
                            "            return span\n"
                        ),
                        new=(
                            "    stored = {\n"
                            "        r[\"content_hash\"]\n"
                            "        for r in conn.execute(\n"
                            "            \"SELECT s.content_hash FROM symbols s \"\n"
                            "            \"JOIN files f ON f.id = s.file_id WHERE f.path = ?\",\n"
                            "            (raw.path,),\n"
                            "        )\n"
                            "    }\n"
                            "    for span in candidates:\n"
                            "        if cited == span.content_hash or cited in stored:\n"
                            "            return span\n"
                        ),
                    ),
                    E_STORE_VERIFY,
                ),
            ),
            # The attack that matters most, and the only one whose evidence was ever
            # real -- so it is also the only one the store could not possibly catch by
            # inspecting the submission. Both doors refuse it by re-reading, which is
            # the property being measured; they differ only in what they call it.
            GATE_STORE: Rule(
                codes=frozenset({"evidence_stale"}),
                mutation=_mut(
                    "write_assertion re-reads the bytes at admission time, so a "
                    "citation that was correct yesterday is refused today",
                    E_STORE_VERIFY,
                ),
            ),
        },
    ),
    "foreign_symbol_hash": Family(
        name="foreign_symbol_hash",
        expect=REFUSED,
        attack="the hash of a DIFFERENT indexed symbol in the same file",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"hash_mismatch"}),
                mutation=_mut(
                    "_symbol_bytes_at admits only symbols occupying EXACTLY the cited "
                    "lines",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "            \"WHERE f.path = ? AND s.line_start = ? AND s.line_end = ? ORDER BY s.id\",\n"
                            "            (path, line_start, line_end),"
                        ),
                        new=(
                            "            \"WHERE f.path = ? ORDER BY s.id\",\n"
                            "            (path,),"
                        ),
                    ),
                ),
            ),
            # The store never looks a symbol up, so there is no widened lookup to
            # exploit and nothing here for `_symbol_bytes_at`'s rule to do. What
            # catches it is the same re-read that catches every other wrong hash --
            # which is worth stating plainly, because it means this family measures a
            # distinct rule at one door and a shared one at the other.
            GATE_STORE: Rule(
                codes=frozenset({"evidence_stale"}),
                mutation=_mut(
                    "write_assertion hashes the cited byte range, so a sibling "
                    "symbol's hash does not verify against it",
                    E_STORE_VERIFY,
                ),
            ),
        },
    ),
    "unknown_subject": Family(
        name="unknown_subject",
        expect=REFUSED,
        attack="a perfectly cited claim about a symbol that does not exist",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"unknown_subject"}),
                mutation=_mut(
                    "the gate refuses a subject_qualname that names no indexed symbol "
                    "-- in _submit_body, which can say what to search for, and again "
                    "in store.write_assertion, which is the door every caller comes "
                    "through",
                    # Four edits, one rule, and the first is the one WP4 made
                    # necessary. Before it, deleting the server's check left
                    # `write_assertion`'s copy refusing the attack with the SAME code,
                    # so this family reported its own rule as undeletable while the
                    # rule it was actually measuring had moved underneath it. That is
                    # precisely the failure the mutation harness exists to catch,
                    # arriving as a false negative in the harness itself.
                    E_STORE_UNKNOWN_SUBJECT,
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    if subject is None:\n"
                            "        raise ToolError(\n"
                            "            \"unknown_subject\",\n"
                            "            f\"no symbol named {subject_qualname!r} in this index, so there is nothing \"\n"
                            "            \"for this claim to be about. Qualnames are dotted paths from the module \"\n"
                            "            \"root -- use search_code or get_symbol to find the exact one. Only what \"\n"
                            "            \"this index parsed can be the subject of a stored claim.\",\n"
                            "            subject_qualname=subject_qualname,\n"
                            "        )\n"
                        ),
                        new="",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old="            subject_symbol_id=int(subject[\"id\"]),",
                        new="            subject_symbol_id=None if subject is None else int(subject[\"id\"]),",
                    ),
                    Edit(
                        target=GATE_MODULE,
                        old="        \"subject_symbol_id\": int(subject[\"id\"]),",
                        new="        \"subject_symbol_id\": None if subject is None else int(subject[\"id\"]),",
                    ),
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset({"unknown_subject"}),
                mutation=_mut(
                    "store.write_assertion refuses a subject this index has never "
                    "parsed",
                    E_STORE_UNKNOWN_SUBJECT,
                ),
            ),
        },
    ),
    # --- positive controls ------------------------------------------------
    "published_hash": Family(
        name="published_hash",
        expect=ACCEPTED,
        attack="the loop the design rests on: cite the hash retrieval handed you",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "_verify_span checks the cited hash against the SYMBOL's bytes as "
                    "well as the lines' bytes",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    return [\n"
                            "        (int(r[\"byte_start\"]), int(r[\"byte_end\"]))\n"
                            "        for r in conn.execute(\n"
                            "            \"SELECT s.byte_start, s.byte_end FROM symbols s JOIN files f ON f.id = s.file_id \"\n"
                            "            \"WHERE f.path = ? AND s.line_start = ? AND s.line_end = ? ORDER BY s.id\",\n"
                            "            (path, line_start, line_end),\n"
                            "        )\n"
                            "    ]"
                        ),
                        new="    return []",
                    ),
                ),
            ),
            # A positive control detects over-strictness, and the store has a
            # different way to be over-strict than the server does. The two readings
            # of a line range do not exist here -- the caller supplies the bytes -- so
            # what this family pins at this door is that verification is scoped to the
            # CITED range and not to the file containing it. Widening it to the whole
            # file refuses every correct citation in a module that has more than one
            # symbol in it, which is every module.
            GATE_STORE: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "write_assertion verifies the cited byte range, not the file it "
                    "sits in",
                    E_STORE_CITED_BYTES,
                ),
            ),
        },
    ),
    "quoted_lines": Family(
        name="quoted_lines",
        expect=ACCEPTED,
        attack="the other honest reading: the exact lines, copied out of the file",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "_verify_span checks the cited hash against the whole lines' bytes "
                    "as well as the symbol's",
                    Edit(
                        target=GATE_MODULE,
                        old=(
                            "    if line_range is not None and line_range not in ranges:\n"
                            "        ranges.append(line_range)"
                        ),
                        new="    pass",
                    ),
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "write_assertion verifies the cited byte range, not the file it "
                    "sits in",
                    E_STORE_CITED_BYTES,
                ),
            ),
        },
    ),
    "multi_span": Family(
        name="multi_span",
        expect=ACCEPTED,
        attack="two good citations on one claim, both of which must be stored",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "_submit_body verifies and stores EVERY submitted span, not the "
                    "first one",
                    Edit(
                        target=GATE_MODULE,
                        old="    spans = [_verify_span(conn, root, raw) for raw in evidence_spans]",
                        new="    spans = [_verify_span(conn, root, raw) for raw in evidence_spans[:1]]",
                    ),
                ),
            ),
            GATE_STORE: Rule(
                codes=frozenset(),
                mutation=_mut(
                    "write_assertion inserts a row for every span it was given",
                    E_STORE_EVERY_SPAN,
                ),
            ),
        },
    ),
}

NEGATIVE_FAMILIES = tuple(f for f, spec in FAMILIES.items() if spec.expect == REFUSED)
POSITIVE_FAMILIES = tuple(f for f, spec in FAMILIES.items() if spec.expect == ACCEPTED)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cite:
    """One citation stated in BOTH vocabularies the two gates speak.

    The MCP surface is given line numbers and derives the byte range itself; the store
    is given the byte range and told the lines. A corpus holding only one of the two
    would have to reconstruct the other at submission time, and that reconstruction
    would be a third implementation of the gate's own arithmetic sitting between the
    corpus and the thing it measures -- inheriting whatever the gate got wrong, which
    is the mistake `FileFact.line_bytes` already exists to avoid.

    So both readings are computed once, in `build_corpus`, where the attack's MEANING
    is known and only there. "One line past the end of the file" is a line range the
    file does not have and a byte range that runs off the end of it; nothing but the
    generator of that control knows those are the same attack.

    `content_hash` and `text` are both optional and both may be absent -- that is the
    `unverifiable_span` attack, and it is why neither can be required here.
    """

    path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    content_hash: str | None = None
    text: str | None = None

    def cited_hash(self) -> str:
        """What the submission asserts is at those bytes. Empty when it asserts nothing.

        The empty string is not a sentinel for "unset" here; it is the citation's
        actual content, and `write_assertion` is entitled to refuse it as such.
        `EvidenceSpan.content_hash` is typed `str`, so an absent hash has to arrive as
        something, and anything other than the empty string would be a value the corpus
        invented on the caller's behalf.
        """
        if self.content_hash is not None:
            return self.content_hash
        if self.text is not None:
            return content_hash(self.text.encode())
        return ""

    def as_input(self) -> dict[str, Any]:
        """The kwargs for `server.app.EvidenceSpanInput`. Byte offsets are dropped.

        Dropped rather than passed, because the server derives them and a corpus that
        supplied them would be measuring its own arithmetic on that surface.
        """
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
            "text": self.text,
        }


@dataclass(frozen=True)
class Control:
    """One submission the gate must refuse (or admit), and what makes it adversarial."""

    name: str
    family: str
    subject_qualname: str
    claim: str
    spans: tuple[Cite, ...]
    # (repo-relative path, replacement bytes) written before submitting and restored
    # after. Only the stale attack needs it: its evidence was real when it was read.
    edit: tuple[str, bytes] | None = None

    @property
    def expect(self) -> str:
        return FAMILIES[self.family].expect


def _flip_a_letter(source: bytes, byte_start: int, byte_end: int) -> bytes | None:
    """Change one ascii letter inside a byte range, leaving every offset intact.

    Letters only, so the file keeps its line count and every other symbol keeps its
    span: the stale attack has to be about content changing under a citation, not about
    the whole file shifting. Returns None if the range holds no letter to change.
    """
    window = source[byte_start:byte_end]
    for offset, char in enumerate(window):
        if 97 <= char <= 122:  # a-z
            replacement = 98 if char != 98 else 99  # 'b', or 'c' if it already was 'b'
            return source[:byte_start + offset] + bytes([replacement]) + source[byte_start + offset + 1:]
    return None


def _cite(symbol: SymbolFact, **overrides: Any) -> Cite:
    """The citation retrieval would hand an agent for `symbol`, before any tampering.

    Both vocabularies come from the same index row, so an attack built by overriding
    one field of this differs from a correct citation in exactly that field. A control
    that also drifted in some second field would be refused for a reason nobody asked
    about, and a whole family can pass that way.
    """
    return Cite(**{
        "path": symbol.path,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "byte_start": symbol.byte_start,
        "byte_end": symbol.byte_end,
        "content_hash": symbol.content_hash,
        **overrides,
    })


def build_corpus(
    files: dict[str, FileFact],
    *,
    limit: int | None = None,
    surface: str = SURFACE_DIRECT,
) -> tuple[list[Control], list[tuple[str, str, str]]]:
    """Generate the corpus from what the index actually holds.

    One instance per attack per symbol, wherever the attack is constructible. Where it
    is not -- a file with no blank line has nothing to cite blankly -- the instance is
    SKIPPED WITH A REASON and returned, not dropped. A skip that vanishes is how a
    family quietly becomes empty, and an empty family is how a rejection rate becomes
    a statement about nothing.

    `surface` drops the families the surface cannot state at all, and each one leaves a
    skip carrying the reason from its `Inexpressible` entry. A family that is merely
    UNENFORCED at a surface is still generated and still submitted: not being refused
    is the measurement.

    `limit` caps instances per family, for a fast run. It is applied per family rather
    than by truncating the symbol list, so no family can be emptied by it.
    """
    controls: list[Control] = []
    skips: list[tuple[str, str, str]] = []
    per_family: dict[str, int] = dict.fromkeys(FAMILIES, 0)
    gate = gate_of(surface)

    unstatable = {
        name: entry.reason
        for name, spec in FAMILIES.items()
        if isinstance(entry := spec.gates[gate], Inexpressible)
    }
    for name, reason in unstatable.items():
        skips.append((name, f"surface={surface}", reason))

    def add(control: Control) -> None:
        if control.family in unstatable:
            return
        if limit is not None and per_family[control.family] >= limit:
            return
        per_family[control.family] += 1
        controls.append(control)

    outside_line = OUTSIDE_SOURCE.split("\n")[0]

    for fact in files.values():
        blanks = fact.blank_lines()
        symbols = fact.symbols
        for symbol in symbols:
            qual = symbol.qualname
            accepted = fact.accepted_hashes(symbol.line_start, symbol.line_end)

            add(Control(
                name=f"zero_evidence/{qual}",
                family="zero_evidence",
                subject_qualname=qual,
                claim=f"{qual} is the entry point for the whole subsystem",
                spans=(),
            ))
            add(Control(
                name=f"empty_claim/{qual}",
                family="empty_claim",
                subject_qualname=qual,
                # Whitespace rather than "": a caller that sends the empty string has
                # obviously sent nothing, and a rule that only caught that one would
                # pass every generator whose template emitted a newline.
                claim="   \n\t ",
                spans=(_cite(symbol),),
            ))
            add(Control(
                name=f"unverifiable_span/{qual}",
                family="unverifiable_span",
                subject_qualname=qual,
                claim=f"{qual} is described by these lines, take my word for what is in them",
                spans=(_cite(symbol, content_hash=None),),
            ))
            add(Control(
                name=f"zero_length_span/{qual}",
                family="zero_length_span",
                subject_qualname=qual,
                claim=f"{qual} is proved by nothing at all, cited where its code begins",
                # The line range is the symbol's own and is perfectly valid; only the
                # byte range is empty. That is what separates this from `blank_range`
                # and what made it survive every check the store had before WP4.
                spans=(_cite(
                    symbol, byte_end=symbol.byte_start, content_hash=EMPTY_SHA
                ),),
            ))
            add(Control(
                name=f"absent_file/{qual}",
                family="absent_file",
                subject_qualname=qual,
                claim=f"{qual} is defined in a file this index has never seen",
                spans=(_cite(symbol, path=f"{symbol.path}.absent"),),
            ))
            add(Control(
                name=f"escaping_path/{qual}",
                family="escaping_path",
                subject_qualname=qual,
                claim=f"{qual} reads a secret from outside the repository",
                spans=(Cite(
                    path=f"../{OUTSIDE_NAME}",
                    line_start=1,
                    line_end=1,
                    byte_start=0,
                    byte_end=len(outside_line),
                    content_hash=content_hash(outside_line.encode()),
                ),),
            ))
            add(Control(
                name=f"unknown_subject/{qual}",
                family="unknown_subject",
                subject_qualname=f"{qual}_that_does_not_exist",
                claim=f"{qual}_that_does_not_exist does the work {qual} is credited with",
                spans=(_cite(symbol),),
            ))

            # --- the hash attacks, each guarded against being accidentally correct ---
            decoy = _decoy_hash(fact, accepted)
            if decoy is None:
                skips.append(("decoy_content_hash", qual, "no other content in this file hashes differently"))
            else:
                add(Control(
                    name=f"decoy_content_hash/{qual}",
                    family="decoy_content_hash",
                    subject_qualname=qual,
                    claim=f"{qual} contains the code that is actually elsewhere in this file",
                    spans=(_cite(symbol, content_hash=decoy),),
                ))

            foreign = next(
                (
                    other.content_hash
                    for other in symbols
                    if other.qualname != qual and other.content_hash not in accepted
                ),
                None,
            )
            if foreign is None:
                skips.append(("foreign_symbol_hash", qual, "no sibling symbol with a different hash"))
            else:
                add(Control(
                    name=f"foreign_symbol_hash/{qual}",
                    family="foreign_symbol_hash",
                    subject_qualname=qual,
                    claim=f"{qual} is the symbol whose body the sibling hash describes",
                    spans=(_cite(symbol, content_hash=foreign),),
                ))

            edited = _flip_a_letter(fact.source, symbol.byte_start, symbol.byte_end)
            if edited is None:
                skips.append(("stale_but_once_valid", qual, "no ascii letter inside the symbol to change"))
            elif symbol.content_hash in _hashes_after(fact, edited, symbol):
                skips.append(("stale_but_once_valid", qual, "the edit did not move this symbol's bytes"))
            else:
                add(Control(
                    name=f"stale_but_once_valid/{qual}",
                    family="stale_but_once_valid",
                    subject_qualname=qual,
                    claim=f"{qual} still does what it did when this hash was taken",
                    spans=(_cite(symbol),),
                    edit=(fact.path, edited),
                ))

            # --- positives ------------------------------------------------
            add(Control(
                name=f"published_hash/{qual}",
                family="published_hash",
                subject_qualname=qual,
                claim=f"{qual} is cited by the hash this index published for it",
                spans=(_cite(symbol),),
            ))
            quoted = fact.text_at(symbol.line_start, symbol.line_end)
            line_span = fact.line_bytes(symbol.line_start, symbol.line_end)
            if quoted is None or line_span is None:
                skips.append((
                    "quoted_lines",
                    qual,
                    "the symbol's stored line range is not a valid line range -- a "
                    "module ends one line past its last written line",
                ))
            elif not quoted.strip():
                skips.append(("quoted_lines", qual, "the symbol's lines are blank"))
            else:
                add(Control(
                    name=f"quoted_lines/{qual}",
                    family="quoted_lines",
                    subject_qualname=qual,
                    claim=f"{qual} is cited by the exact lines quoted off disk",
                    # The WHOLE-LINES reading, not the symbol's: that is the point of
                    # this family, and for the 25.5% of symbols where the two differ
                    # the byte range here is deliberately not `symbol.byte_start`.
                    spans=(Cite(
                        path=symbol.path,
                        line_start=symbol.line_start,
                        line_end=symbol.line_end,
                        byte_start=line_span[0],
                        byte_end=line_span[1],
                        text=quoted,
                    ),),
                ))

        # --- per-file attacks ---------------------------------------------
        if not symbols:
            skips.append(("past_eof", fact.path, "no indexed symbol to be the subject"))
            continue
        subject = symbols[0].qualname
        last_line = fact.text_at(fact.line_count, fact.line_count)
        last_span = fact.line_bytes(fact.line_count, fact.line_count)
        if last_line is None or last_span is None:
            skips.append(("past_eof", fact.path, "the file has no last line"))
        else:
            add(Control(
                name=f"past_eof/{fact.path}",
                family="past_eof",
                subject_qualname=subject,
                claim=f"{subject} continues past the end of {fact.path}",
                # "One line past the end" in bytes is one byte past the end. Not a
                # clamp and not the last valid offset: the attack is a range the file
                # does not contain, and encoding it as one it does contain would turn
                # the control into a correct citation with an attack's name on it.
                spans=(Cite(
                    path=fact.path,
                    line_start=fact.line_count,
                    line_end=fact.line_count + 1,
                    byte_start=last_span[0],
                    byte_end=len(fact.source) + 1,
                    text=last_line,
                ),),
            ))
        if not blanks:
            skips.append(("blank_range", fact.path, "the file has no blank line"))
        else:
            blank = blanks[0]
            blank_span = fact.line_bytes(blank, blank)
            assert blank_span is not None  # noqa: S101 - blank_lines() only yields real lines
            add(Control(
                name=f"blank_range/{fact.path}#text",
                family="blank_range",
                subject_qualname=subject,
                claim=f"{subject} is documented on line {blank} of {fact.path}",
                spans=(Cite(
                    path=fact.path, line_start=blank, line_end=blank,
                    byte_start=blank_span[0], byte_end=blank_span[1], text="",
                ),),
            ))
            add(Control(
                name=f"blank_range/{fact.path}#hash",
                family="blank_range",
                subject_qualname=subject,
                claim=f"{subject} is documented on line {blank} of {fact.path}",
                spans=(Cite(
                    path=fact.path, line_start=blank, line_end=blank,
                    byte_start=blank_span[0], byte_end=blank_span[1],
                    content_hash=EMPTY_SHA,
                ),),
            ))
        if len(symbols) < 2:
            skips.append(("multi_span", fact.path, "only one symbol in the file"))
        else:
            first, second = symbols[0], symbols[1]
            add(Control(
                name=f"multi_span/{fact.path}",
                family="multi_span",
                subject_qualname=first.qualname,
                claim=f"{first.qualname} and {second.qualname} are cited together",
                spans=(_cite(first), _cite(second)),
            ))
    return controls, skips


def _decoy_hash(fact: FileFact, accepted: set[str]) -> str | None:
    """The hash of content that IS in this file but is not what was cited.

    Tried widest-first, and every candidate is checked against `accepted` -- the two
    readings the gate is entitled to admit for the cited lines. For a module symbol the
    whole file IS the symbol, so the whole-file hash would be a correct citation; that is
    exactly the case this loop exists to walk past.
    """
    ranges = [
        (0, len(fact.source)),
        *[
            span
            for span in (
                fact.line_bytes(1, 1),
                fact.line_bytes(1, min(2, fact.line_count)),
                fact.line_bytes(fact.line_count, fact.line_count),
            )
            if span is not None
        ],
    ]
    for byte_start, byte_end in ranges:
        if byte_start >= byte_end:
            continue
        candidate = content_hash(fact.source[byte_start:byte_end])
        if candidate not in accepted:
            return candidate
    return None


def _hashes_after(fact: FileFact, edited: bytes, symbol: SymbolFact) -> set[str]:
    """What the stale attack's cited lines would hash to AFTER the edit.

    The guard on the stale control: if the edit left this symbol's bytes (or its lines'
    bytes) hashing to the same thing, the citation is still valid and the control is a
    positive control with an attack's name on it.
    """
    after = FileFact(path=fact.path, source=edited, symbols=fact.symbols)
    return after.accepted_hashes(symbol.line_start, symbol.line_end)


# ---------------------------------------------------------------------------
# the resolution of a rate
# ---------------------------------------------------------------------------
#
# WP14. The pooled number was never wrong; it was SIZED wrong, and the sizing is what
# persuades. `1.0000` over 12,266 submissions reads as twelve thousand independent
# adversarial probes. It is fifteen attack shapes instantiated once per symbol, and for
# more than half of them the gate does literally the same thing on every instance --
# same lines, same bytes, same refusal -- so the extra instances are replications, not
# coverage. Four significant figures over a numerator that can only be 0 or n, on a
# corpus whose independent breadth is fifteen, is the most over-precise number here.
#
# Three things follow, and all three are printed rather than documented: the per-family
# table is the measurement, the denominator is the number of distinct gate EXECUTIONS,
# and every family carries an upper bound that says what its n actually buys.

ALPHA = 0.05

# What a family's instances turned out to be, measured rather than assumed. See
# `_GateObserver` for how, and `FamilyStat.shape` for what each one licenses.
REPLICATED = "replicated"
VARYING = "varying"
UNMEASURED = "unmeasured"

INTERVAL_NOTE = (
    "ub95 is a one-sided 95% Clopper-Pearson upper bound on this family's FAILURE rate.\n"
    "Exact, not approximate: with zero failures it is 1 - 0.05**(1/n), which is what the\n"
    "rule of three (3/n) approximates and which, unlike 3/n, is still a probability at\n"
    "n=1 and n=2 -- and several families here have single-figure probe counts. It assumes\n"
    "n independent Bernoulli trials. The replicated families do not supply them: instances\n"
    "that executed the gate identically are one trial repeated, so the bound is computed\n"
    "on `probes`, and ub(n) is printed beside it only to show the size of the difference.\n"
    "Even ub(probes) is optimistic -- the probes are one template with one field varied,\n"
    "not a sample from the space of attacks."
)

BREADTH_LIMIT = (
    "WHAT THIS CANNOT SHOW\n"
    "  Every control above is generated from FAMILIES, so this corpus can only ever\n"
    "  expose a rule some family already names.\n"
    "  It has found no attack nobody had enumerated.\n"
    "  The holes it did find were found some other way: two auditors found two by\n"
    "  probing outside the family list, and adding a second door found a third --\n"
    "  escaping_path was refused at the server and admitted at the store, stored\n"
    "  active and reported servable, on every instance. Adding doors is a second axis\n"
    "  with the same property. A rate of 100.0% here is a statement about the\n"
    "  enumerated attack shapes and about nothing else."
)


def binomial_at_most(successes: int, n: int, p: float) -> float:
    """P(X <= successes) for X ~ Binomial(n, p). In log space, so n=12,266 is fine.

    Public because the bound below is only worth anything if it can be checked against
    its own definition rather than against itself -- which is what
    `test_the_upper_bound_is_exact_rather_than_a_rule_of_thumb` does.
    """
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if successes >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(n + 1)
    total = 0.0
    for k in range(min(successes, n) + 1):
        total += math.exp(
            log_n - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * log_p + (n - k) * log_q
        )
    return min(total, 1.0)


def clopper_pearson_upper(failures: int, n: int, alpha: float = ALPHA) -> float:
    """The one-sided (1-alpha) upper bound on a failure rate, exact.

    Clopper-Pearson rather than the rule of three, and the reason is small n rather
    than taste. `3/n` is the large-n approximation to this for a zero numerator -- they
    agree to 0.2% by n=50 -- but it stops being a probability below n=3, and this
    corpus has families whose honest denominator is 1. `1 - alpha**(1/n)` is closed,
    needs no library, and answers 0.95 at n=1, which is the true and useful thing to
    say about a single probe.

    Non-zero numerators are solved by bisection on the binomial CDF, which is the same
    definition: the bound is the p at which a result this good has probability alpha.
    Nothing in a passing run needs that branch, and it is here because a bound that
    silently only works at zero would be a bound nobody could trust the first time
    something failed.
    """
    if n <= 0:
        raise VacuousCorpus(
            "an upper bound was asked over no trials. Every convention that does not "
            "raise here returns something between 0 and 1 for a measurement that never "
            "happened."
        )
    if failures >= n:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / n)
    low, high = failures / n, 1.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if binomial_at_most(failures, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


# ---------------------------------------------------------------------------
# watching what the gate actually did
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Execution:
    """One traversal of the gate: which of its own lines ran, and what they consumed.

    This is the instrument behind `replicated` vs `varying`, and it is deliberately an
    observation of the CODE rather than an inference from the family name or from the
    submission. Two claims a name-based rule cannot make and this one can:

    * `unverifiable_span` is the same attack at both doors and is replicated at one and
      varying at the other -- the store refuses it in the span loop before opening
      anything, while the server reads the file and hashes both readings of the cited
      lines before finding there is no hash to compare them against;
    * `absent_file` submits a different path and a different byte range for every
      symbol, and is still one probe, because the gate refuses it at the index-membership
      check without ever reaching them.

    `code_path` is the digest of the (module-relative file, line) sequence executed
    inside the package, excluding `eval` -- the corpus watching itself would swamp the
    signal. `evidence_read` is the digest of what the gate pulled out of the world on
    the way: the bytes returned by every `Path.read_bytes` and the bytes handed to every
    `content_hash` call the gate made. Reads alone are too coarse -- the store reads a
    whole file to refuse a decoy hash, so all of a file's symbols would look alike --
    and the hashed bytes are what the comparison actually turns on.

    What identity here establishes is narrow and worth stating: it is a post-hoc fact
    about this run, not a proof about the submissions. n instances that produced the
    identical execution carry the information of one, because the refusal code is
    decided by which `raise` ran and that is inside `code_path`. It does NOT establish
    that a different symbol could not have taken a different branch.
    """

    code_path: str
    evidence_read: str

    @property
    def digest(self) -> str:
        return f"{self.code_path}/{self.evidence_read}"


class _GateObserver:
    """A `sys.settrace` hook that records one `Execution`. Cheap, and off by default.

    Nothing is monkey-patched. The two boundary calls whose ARGUMENTS matter -- a file
    read and a content hash -- are picked out by code-object identity, so a rename
    cannot silently turn this into a line counter, and `content_hash` calls made from
    `eval` are skipped by looking at the calling frame: the corpus computes the cited
    hash of a quoted-text control itself, and counting that as evidence the GATE
    consumed would make every such family look varying for the corpus's own reasons.
    """

    def __init__(self) -> None:
        root = str(Path(__file__).resolve().parents[1]) + os.sep
        self._root = root
        self._skip = root + "eval" + os.sep
        self._relative: dict[str, str | None] = {}
        self._locals: dict[str, Any] = {}
        self._path = hashlib.sha256()
        self._evidence = hashlib.sha256()
        self._hash_code = content_hash.__code__
        self._read_code = getattr(Path.read_bytes, "__code__", None)

    def _rel(self, filename: str) -> str | None:
        try:
            return self._relative[filename]
        except KeyError:
            pass
        rel: str | None = None
        if filename.startswith(self._root) and not filename.startswith(self._skip):
            rel = filename[len(self._root):]
        self._relative[filename] = rel
        return rel

    def _local_for(self, rel: str) -> Any:
        tracer = self._locals.get(rel)
        if tracer is None:
            prefix = rel.encode()
            update = self._path.update

            def tracer(frame: Any, event: str, arg: Any) -> Any:  # noqa: ANN401
                if event == "line":
                    update(b"%s:%d;" % (prefix, frame.f_lineno))
                return tracer

            self._locals[rel] = tracer
        return tracer

    def _read_return(self, frame: Any, event: str, arg: Any) -> Any:  # noqa: ANN401
        if event == "return" and isinstance(arg, bytes):
            self._evidence.update(b"read:" + hashlib.sha256(arg).digest())
        return self._read_return

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:  # noqa: ANN401
        if event != "call":
            return None
        code = frame.f_code
        if code is self._hash_code:
            back = frame.f_back
            if back is None or self._rel(back.f_code.co_filename) is not None:
                data = frame.f_locals.get("source")
                if isinstance(data, bytes):
                    self._evidence.update(b"hash:" + hashlib.sha256(data).digest())
            return None
        if code is self._read_code:
            return self._read_return
        rel = self._rel(code.co_filename)
        return None if rel is None else self._local_for(rel)

    def result(self) -> Execution:
        return Execution(
            code_path=self._path.hexdigest(), evidence_read=self._evidence.hexdigest()
        )


@contextmanager
def _observing() -> Iterator[list[Execution]]:
    """Install the observer for one submission and hand back what it saw.

    The previous trace function is saved and restored rather than assumed absent: a
    coverage tool holding the slot would otherwise be silently uninstalled for the rest
    of the process by a measurement that has nothing to do with it.
    """
    seen: list[Execution] = []
    observer = _GateObserver()
    previous = sys.gettrace()
    sys.settrace(observer)
    try:
        yield seen
    finally:
        sys.settrace(previous)
        seen.append(observer.result())


# ---------------------------------------------------------------------------
# running the corpus against the real gate
# ---------------------------------------------------------------------------

@dataclass
class Harness:
    """A disposable repo, an index over it, and the gate bound to both."""

    repo: Path
    index_path: Path
    files: dict[str, FileFact]
    surface: str = SURFACE_DIRECT
    corpus_name: str = ""
    # The revision of the tree the corpus was generated from, when there is one. Carried
    # so the caption under the instance count can name what N symbols were counted at:
    # "12,266 instances" is a different claim at every commit, and a pooled figure
    # quoted without one is a number nobody can re-derive.
    revision: str = ""
    pristine: dict[str, bytes] = field(default_factory=dict)
    dirty: set[str] = field(default_factory=set)
    _source: Any = None
    _server: Any = None
    _conn: Any = None

    @property
    def gate(self) -> str:
        return gate_of(self.surface)

    @property
    def source(self) -> Any:
        if self._source is None:
            self._source = gate_module().IndexSource(path=self.index_path)
        return self._source

    @property
    def conn(self) -> sqlite3.Connection:
        """The connection the submission will be made through.

        The store surface opens its own with `db.connect` rather than borrowing
        `IndexSource`, and that is not tidiness. `codelearner learn` reaches
        `write_assertion` over a plain connection with no MCP anywhere in the process,
        and this surface exists to measure that path -- a harness that imported
        `server.app` to get a cursor would make the store's rate unreportable on a
        machine without the optional extra, which is exactly the population whose gate
        was one lock short until WP4.
        """
        if self.gate == GATE_STORE:
            if self._conn is None:
                self._conn = db.connect(self.index_path)
            conn: sqlite3.Connection = self._conn
            return conn
        return self.source.connect()  # type: ignore[no-any-return]

    def gate_path(self) -> str:
        """The file that holds the rules this surface is being measured against."""
        module = store_module() if self.gate == GATE_STORE else gate_module()
        return str(Path(module.__file__ or "?").resolve())

    def rows(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM assertions").fetchone()[0])

    def warm(self) -> None:
        """Build everything lazy BEFORE the first control is watched.

        `IndexSource`, the MCP server object and the store's connection are all built on
        first use, and first use is inside a submission. Under observation that
        construction lands in the first control's `Execution` and nowhere else, so a
        family whose 1,343 instances are otherwise identical would report two distinct
        probes -- one of them an artefact of import order. The warm-up is the whole fix
        and it costs one query.
        """
        self.rows()
        if self.surface == SURFACE_TOOL and self._server is None:
            from ..server import build_server

            self._server = build_server(self.index_path)

    def observe(
        self, control: Control
    ) -> tuple[dict[str, Any] | None, BaseException | None, Execution]:
        """Submit one control and record what the gate did with it.

        Separate from `submit` rather than folded into it, because `submit` is also how
        the individual tests put one control through the gate, and instrumenting those
        would trace pytest's frames for no benefit. The trace is installed around the
        submission only.

        A raise is returned rather than propagated, so that the execution which produced
        it survives. A gate that crashes is a finding, and a finding whose code path was
        thrown away with the exception is one nobody can compare against the run's other
        crashes.
        """
        error: BaseException | None = None
        payload: dict[str, Any] | None = None
        with _observing() as seen:
            try:
                payload = self.submit(control)
            except Exception as exc:  # a raise IS the finding; never let it end the run
                error = exc
        return payload, error, seen[0]

    def apply(self, control: Control) -> None:
        if control.edit is None:
            return
        rel, replacement = control.edit
        (self.repo / rel).write_bytes(replacement)
        self.dirty.add(rel)

    def restore(self) -> None:
        """Put every file a control edited back the way it was found.

        Between controls, not at the end. The stale attack changes a file under a
        citation, and a later control that inherited that change would be refused for a
        reason nobody asked about -- a whole family passing for the wrong reason, which
        looks exactly like a family passing.
        """
        for rel in sorted(self.dirty):
            (self.repo / rel).write_bytes(self.pristine[rel])
        self.dirty.clear()

    def close(self) -> None:
        if self._source is not None and self._source._conn is not None:
            self._source._conn.close()
            self._source._conn = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def submit(self, control: Control) -> dict[str, Any]:
        """Put one control through the gate, on the surface an agent would reach.

        `direct` calls the tool body under `_guard`, which is the whole gate and the
        whole error contract. `tool` goes through the registered MCP tool as a client
        would, so the same corpus can prove that a refusal arrives as data rather than
        as a traceback -- the module's other rule, and one that a direct call cannot
        check. Both must return the same verdict for every control; that is a test.

        `store` calls `write_assertion` with the byte ranges the corpus already holds,
        which is what every library caller does. It is the surface the README's
        headline rate did NOT cover, and the one where an attack that the other two
        refuse is currently admitted.
        """
        if self.surface == SURFACE_STORE:
            return self._via_store(control)
        app = gate_module()
        spans = [app.EvidenceSpanInput(**span.as_input()) for span in control.spans]
        if self.surface == SURFACE_TOOL:
            return self._via_tool(control, spans)
        payload = app._guard(
            self.source,
            app._submit_body,
            subject_qualname=control.subject_qualname,
            claim=control.claim,
            evidence_spans=spans,
            kind="purpose",
            generator="gate-controls/v1",
            confidence=None,
        )
        return dict(payload)

    def _via_store(self, control: Control) -> dict[str, Any]:
        """`store.write_assertion`, called the way a library caller calls it.

        Nothing here checks anything. The adapter's only job is to turn the corpus's
        byte ranges into `EvidenceSpan`s and turn the store's exceptions into the same
        payload shape the server surface produces -- because an adapter that declined
        to submit a citation it thought was bad would be a third gate, sitting between
        the corpus and the gate under test and quietly answering for it. That is the
        failure `Harness` is otherwise built to avoid, and it is the easiest one to
        introduce here: every one of these submissions looks obviously wrong.

        `evidence` is read back out of `evidence_spans` rather than echoed from the
        argument. Echoing it would make the positive controls' "every submitted span
        was stored" conjunct trivially true, which is the same conjunct the server
        surface checks against a response the server built from what it verified.
        """
        store = store_module()
        refusals = store_refusal_codes()
        spans = [
            store.EvidenceSpan(
                path=cite.path,
                line_start=cite.line_start,
                line_end=cite.line_end,
                byte_start=cite.byte_start,
                byte_end=cite.byte_end,
                content_hash=cite.cited_hash(),
            )
            for cite in control.spans
        ]
        conn = self.conn
        try:
            assertion_id = store.write_assertion(
                conn,
                subject_qualname=control.subject_qualname,
                kind="purpose",
                claim=control.claim,
                spans=spans,
                generator="gate-controls/v1",
            )
        except tuple(refusals) as exc:
            code = refusals.get(type(exc))
            if code is None:
                # A SUBCLASS of a named refusal, which nothing declares a code for.
                # Re-raised so it lands as RAISED rather than being scored under its
                # parent's code -- a new rule inheriting an old one's name is the
                # quietest way for a refusal to stop being attributable.
                raise
            return {"ok": False, "error": {"code": code, "message": str(exc)}}
        stored = int(
            conn.execute(
                "SELECT count(*) FROM evidence_spans WHERE assertion_id = ?",
                (assertion_id,),
            ).fetchone()[0]
        )
        return {
            "ok": True,
            "assertion_id": assertion_id,
            "servable": store.is_servable(conn, assertion_id),
            "evidence": [None] * stored,
        }

    def _via_tool(self, control: Control, spans: list[Any]) -> dict[str, Any]:
        import asyncio

        from ..server import build_server

        if self._server is None:
            self._server = build_server(self.index_path)
        result = asyncio.run(
            self._server.call_tool(
                "submit_assertion",
                {
                    "subject_qualname": control.subject_qualname,
                    "claim": control.claim,
                    "evidence_spans": [s.model_dump() for s in spans],
                    "kind": "purpose",
                    "generator": "gate-controls/v1",
                },
            )
        )
        if result.is_error or result.structured_content is None:
            # A traceback across the transport is a distinct failure from a refusal,
            # and collapsing the two would let it pass as a rejection.
            return {"ok": False, "error": {"code": "raised_into_transport",
                                           "message": str(result.content)}}
        return dict(result.structured_content)


@dataclass(frozen=True)
class Outcome:
    """What the gate did with one control, and whether that is what it promised."""

    control: str
    family: str
    verdict: str
    code: str | None
    rows_added: int
    evidence: int
    expected_evidence: int
    servable: bool | None
    detail: str = ""
    # Which door this control went through. Defaulted so that every existing
    # construction still means what it meant, and load-bearing rather than
    # informational: `held` reads the acceptable codes out of the family's entry FOR
    # THIS GATE, and the two gates refuse the same attack under different names. An
    # outcome that lost its surface would be scored against whichever code set the
    # default named, which for a store-surface refusal is a set it can never produce.
    surface: str = SURFACE_DIRECT
    # What the gate DID to produce this outcome (see `Execution`). Both default to the
    # empty string, which means NOT WATCHED rather than "watched and consumed nothing":
    # `FamilyStat` reads that difference and refuses to call an unwatched family
    # replicated, because shrinking a denominator on the strength of a measurement that
    # never happened is the same move as reporting a rate over an empty corpus.
    code_path: str = ""
    evidence_read: str = ""

    @property
    def execution(self) -> str | None:
        """This outcome's execution digest, or None when nobody was watching."""
        if not self.code_path:
            return None
        return f"{self.code_path}/{self.evidence_read}"

    @property
    def expect(self) -> str:
        return FAMILIES[self.family].expect

    @property
    def known_gap(self) -> bool:
        """Whether the gate this control went through has no rule for it at all.

        Never a pass. It is here so a reader of the failure list can tell the two
        apart: a control failing because a declared rule stopped working is a
        regression, and one failing because nothing was ever there is the measurement
        this corpus exists to publish.
        """
        return isinstance(FAMILIES[self.family].at(self.surface), Unenforced)

    @property
    def held(self) -> bool:
        """Whether this control's rule did its whole job.

        For a refusal that means the expected code AND no row: a gate that says no and
        writes the row anyway has refused nothing. For an admission it means servable,
        exactly one row, and every submitted span stored -- an accepted claim carrying
        half its evidence stands on less than its author thought it did, and nothing
        downstream would record that the rest was dropped.

        A gate with no rule for this attack can never hold, whatever it returns. The
        alternative -- scoring an `Unenforced` entry as "held, as declared" -- would
        let a hole raise the rate that is supposed to expose it, which is the
        rejection-rate-of-an-empty-corpus failure wearing a different hat.
        """
        spec = FAMILIES[self.family]
        entry = spec.at(self.surface)
        if spec.expect == REFUSED:
            if not isinstance(entry, Rule):
                return False
            return self.verdict == REFUSED and self.code in entry.codes and self.rows_added == 0
        return (
            self.verdict == ACCEPTED
            and self.servable is True
            and self.rows_added == 1
            and self.evidence == self.expected_evidence
        )

    @property
    def refused(self) -> bool:
        return self.verdict == REFUSED


@dataclass(frozen=True)
class FamilyStat:
    """One family at one door, at the resolution the run actually supports.

    `instances` is what was submitted. `probes` is how many DISTINCT things the gate
    did with them, and it is the denominator every bound here is computed on. The two
    differ by an order of magnitude for more than half the families, and printing only
    the first is what made `1.0000` read as twelve thousand independent adversarial
    tests.
    """

    family: str
    expect: str
    surface: str
    instances: int
    probes: int
    paths: int
    held: int
    shape: str
    codes: dict[str, int]
    enforced: bool

    @property
    def hold_rate(self) -> float:
        if not self.instances:
            raise VacuousCorpus(f"family {self.family!r} generated no controls")
        return self.held / self.instances

    @property
    def upper_bound(self) -> float:
        """95% upper bound on this family's failure rate, over its PROBES.

        The honest one. For a replicated family it is 0.95 however many instances were
        submitted, which is the correct and uncomfortable answer: one probe repeated
        1,343 times establishes what one probe establishes.
        """
        return clopper_pearson_upper(self.instances - self.held, self.probes)

    @property
    def naive_upper_bound(self) -> float:
        """The same bound over INSTANCES. Printed only to show the size of the lie."""
        return clopper_pearson_upper(self.instances - self.held, self.instances)

    def row(self) -> str:
        codes = ", ".join(f"{k}={v}" for k, v in sorted(self.codes.items()))
        if not self.enforced:
            codes = "NO RULE AT THIS GATE -- " + codes
        return (
            f"{self.family:<22} {self.expect:<8} {self.instances:>6} {self.probes:>6} "
            f"{self.paths:>5} {self.held:>6} {self.hold_rate * 100:>7.1f}% "
            f"{self.naive_upper_bound * 100:>7.2f}% {self.upper_bound * 100:>7.2f}%  "
            f"{self.shape:<10} {codes}"
        )


FAMILY_HEADER = (
    f"{'family':<22} {'expect':<8} {'n':>6} {'probes':>6} {'paths':>5} {'held':>6} "
    f"{'rate':>8} {'ub95(n)':>8} {'ub95(pr)':>8}  {'shape':<10} codes"
)


@dataclass
class GateReport:
    """The measurement: a per-family table, and a pooled figure that summarises it."""

    corpus: str
    symbols: int
    outcomes: list[Outcome]
    skips: list[tuple[str, str, str]]
    gate_path: str
    surface: str = SURFACE_DIRECT
    revision: str = ""

    @property
    def gate(self) -> str:
        return gate_of(self.surface)

    @property
    def known_gaps(self) -> list[Outcome]:
        """Controls whose gate has no rule for them. Failures with a documented cause.

        Reported apart from the rest of `failures` for one reason: a run against the
        store surface is red today and will stay red until repo containment moves to
        the chokepoint, and a permanently red run is a run nobody reads. Splitting them
        keeps the regression signal usable without letting the hole out of the rate --
        `rejection_rate` and `attributed_rate` still count these as the refusals that
        did not happen.
        """
        return [o for o in self.outcomes if o.known_gap]

    @property
    def unexpected_failures(self) -> list[Outcome]:
        """Failures that are NOT already declared. The number that should be zero."""
        return [o for o in self.failures if not o.known_gap]

    @property
    def negatives(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.expect == REFUSED]

    @property
    def positives(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.expect == ACCEPTED]

    @property
    def rejection_rate(self) -> float:
        """Fraction of attacks the gate refused. Must be 1.0."""
        negatives = self.negatives
        if not negatives:
            raise VacuousCorpus(
                "no negative controls were generated, so there is no rejection rate. "
                "An empty corpus divides to 1.0 under every convention that does not "
                "raise here, and would report a perfect gate."
            )
        return sum(1 for o in negatives if o.refused) / len(negatives)

    @property
    def attributed_rate(self) -> float:
        """Fraction refused BY THE RULE THEY TARGET, with no row left behind.

        Reported separately from `rejection_rate` because they fail differently. A gate
        pointed at the wrong repo root refuses everything as `file_missing`: rejection
        rate 1.0, attributed rate near zero.
        """
        negatives = self.negatives
        if not negatives:
            raise VacuousCorpus("no negative controls were generated")
        return sum(1 for o in negatives if o.held) / len(negatives)

    @property
    def positive_pass_rate(self) -> float:
        positives = self.positives
        if not positives:
            raise VacuousCorpus(
                "no positive controls were generated. Without them a gate that refuses "
                "every submission scores a perfect rejection rate, which is the one "
                "failure a negative-only suite cannot see."
            )
        return sum(1 for o in positives if o.held) / len(positives)

    def family(self, name: str) -> list[Outcome]:
        return [o for o in self.outcomes if o.family == name]

    def hold_rate(self, name: str) -> float:
        group = self.family(name)
        if not group:
            raise VacuousCorpus(f"family {name!r} generated no controls")
        return sum(1 for o in group if o.held) / len(group)

    def stat(self, name: str) -> FamilyStat:
        """This family's row, at the resolution the run supports.

        The probe count is the number of DISTINCT executions observed, and an outcome
        that carries none makes the whole family `unmeasured` -- every instance counts
        as its own probe and nothing is claimed. That default is the conservative one on
        purpose: the alternative, treating an absent observation as evidence of sameness,
        would shrink a denominator using data nobody collected.
        """
        group = self.family(name)
        if not group:
            raise VacuousCorpus(f"family {name!r} generated no controls")
        executions = [o.execution for o in group]
        if any(e is None for e in executions):
            shape, probes, paths = UNMEASURED, len(group), len(group)
        else:
            probes = len(set(executions))
            paths = len({o.code_path for o in group})
            shape = REPLICATED if probes == 1 else VARYING
        return FamilyStat(
            family=name,
            expect=FAMILIES[name].expect,
            surface=self.surface,
            instances=len(group),
            probes=probes,
            paths=paths,
            held=sum(1 for o in group if o.held),
            shape=shape,
            codes=self.codes(name),
            enforced=isinstance(FAMILIES[name].gates[self.gate], Rule),
        )

    def family_stats(self) -> list[FamilyStat]:
        """Every family that produced controls here. The primary output."""
        return [self.stat(name) for name in FAMILIES if self.family(name)]

    @property
    def negative_probes(self) -> int:
        """Distinct gate executions across the attacks. The honest denominator.

        Summed per family rather than taken over the whole run: two families that
        happened to execute the gate identically would be two attacks nobody had
        collapsed, and the point of the number is the replication INSIDE a family, which
        is where the generator put it.
        """
        return sum(s.probes for s in self.family_stats() if s.expect == REFUSED)

    @property
    def positive_probes(self) -> int:
        return sum(s.probes for s in self.family_stats() if s.expect == ACCEPTED)

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.held]

    def _rate(self, name: str) -> float | None:
        """A rate for output, or None when there is nothing to rate.

        Only for serialisation and display. The properties themselves still raise,
        because the failure being guarded against is a rate REPORTED as 1.0 for an empty
        set -- and `null` is not 1.0. A family-filtered run (which is how every mutation
        is measured) legitimately has no positive controls in it, and refusing to
        serialise that would leave the mutation runner with nothing to read.
        """
        try:
            return float(getattr(self, name))
        except VacuousCorpus:
            return None

    def codes(self, name: str) -> dict[str, int]:
        counted: dict[str, int] = {}
        for outcome in self.family(name):
            key = outcome.code or outcome.verdict
            counted[key] = counted.get(key, 0) + 1
        return counted

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "revision": self.revision,
            "surface": self.surface,
            "gate": self.gate,
            "symbols": self.symbols,
            "gate_path": self.gate_path,
            "attack_shapes": len(FAMILIES),
            "negatives": len(self.negatives),
            "positives": len(self.positives),
            "negative_probes": self.negative_probes,
            "positive_probes": self.positive_probes,
            "rejection_rate": self._rate("rejection_rate"),
            "attributed_rate": self._rate("attributed_rate"),
            "positive_pass_rate": self._rate("positive_pass_rate"),
            "families": {
                stat.family: {
                    "expect": stat.expect,
                    # `n` keeps meaning INSTANCES, and `instances` is its name spelled
                    # out beside it. Reusing `n` for the probe count would have been the
                    # tidier schema and would have silently changed what every existing
                    # reader of this payload was comparing.
                    "n": stat.instances,
                    "instances": stat.instances,
                    "probes": stat.probes,
                    "paths": stat.paths,
                    "shape": stat.shape,
                    "held": stat.held,
                    "hold_rate": stat.hold_rate,
                    "upper_bound_95": stat.upper_bound,
                    "naive_upper_bound_95": stat.naive_upper_bound,
                    "codes": stat.codes,
                    "enforced": stat.enforced,
                }
                for stat in self.family_stats()
            },
            "failures": [
                {"control": o.control, "family": o.family, "verdict": o.verdict,
                 "code": o.code, "rows_added": o.rows_added, "detail": o.detail,
                 "known_gap": o.known_gap}
                for o in self.failures
            ],
            "known_gaps": sorted({o.family for o in self.known_gaps}),
            "skips": [{"family": f, "subject": s, "reason": r} for f, s, r in self.skips],
        }

    def _missing_row(self, name: str) -> str:
        entry = FAMILIES[name].gates[self.gate]
        note = "(not expressible here)" if isinstance(entry, Inexpressible) else "(no instances)"
        return (
            f"{name:<22} {FAMILIES[name].expect:<8} {0:>6} {'-':>6} {'-':>5} {'-':>6} "
            f"{'-':>8} {'-':>8} {'-':>8}  {'-':<10} {note}"
        )

    def caption(self) -> str:
        """What the pooled count is a count OF. Printed where the count is.

        `12,266 attacks` is the sentence a reader takes away as twelve thousand
        independent adversarial probes, and the fix is not a smaller number -- it is the
        denominator being described. Fifteen shapes, instantiated per symbol, against a
        stated number of symbols in a named tree.
        """
        at = self.revision or self.corpus
        return (
            f"{len(FAMILIES)} attack shapes, instantiated per symbol against "
            f"{self.symbols} symbols at {at}."
        )

    def format_table(self) -> str:
        pooled = "-" if self._rate("rejection_rate") is None else (
            f"{self.rejection_rate * 100:.1f}%"
        )
        lines = [
            f"corpus: {self.corpus}  ({self.symbols} symbols, surface={self.surface})",
            f"gate:   {self.gate_path}",
            "",
            "PER FAMILY -- this is the measurement. Everything below it is a summary.",
            "",
            FAMILY_HEADER,
        ]
        for name in FAMILIES:
            group = self.family(name)
            lines.append(self.stat(name).row() if group else self._missing_row(name))

        def show(name: str) -> str:
            rate = self._rate(name)
            return "     -" if rate is None else f"{rate * 100:.1f}%"

        lines += [
            "",
            self.caption(),
            "",
            f"pooled   refused {show('rejection_rate')}  attributed "
            f"{show('attributed_rate')}  over {len(self.negatives)} instances "
            f"({self.negative_probes} distinct gate executions)",
            f"         positive {show('positive_pass_rate')}  over "
            f"{len(self.positives)} legitimate submissions "
            f"({self.positive_probes} distinct gate executions)",
            f"  {pooled} pooled is an EXISTENCE CLAIM about this run:",
            "  no instance of any enumerated attack was admitted at this door.",
            "  It is not an estimate of the probability an attack gets through -- most of",
            "  that denominator is a handful of probes repeated once per symbol. Read it",
            "  beside the positive pass rate, which is the only thing stopping a gate that",
            "  refuses everything from scoring perfectly, and read the per-family bounds",
            "  for what any of it is worth.",
            "",
            INTERVAL_NOTE,
            "",
            BREADTH_LIMIT,
        ]
        gaps = self.known_gaps
        if gaps:
            lines += [
                "",
                f"KNOWN GAPS ({len(gaps)} controls): this gate has no rule for "
                + ", ".join(sorted({o.family for o in gaps}))
                + ".",
            ]
            lines += [
                f"  {name}: {FAMILIES[name].gates[self.gate].detail}"  # type: ignore[union-attr]
                for name in sorted({o.family for o in gaps})
            ]
        unexpected = self.unexpected_failures
        if unexpected:
            lines.append("")
            lines.append("FAILURES:")
            lines += [
                f"  {o.control} -> {o.verdict} {o.code or ''} rows={o.rows_added} {o.detail}"
                for o in unexpected[:40]
            ]
        return "\n".join(lines)


def _revision_of(repo: Path) -> str:
    """The commit the corpus was generated from, or "" when there is not one.

    Best effort, and silent when it fails. A caption naming a revision the reader can
    check is worth a subprocess; a corpus that refuses to run outside a git checkout is
    not, and the harness deliberately indexes a COPY with `.git` stripped, so the
    question can only be asked here, of the tree that was pointed at.
    """
    git = shutil.which("git")
    if git is None:
        return ""
    try:
        proc = subprocess.run(  # noqa: S603
            [git, "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except OSError:  # pragma: no cover - a git that exists and will not start
        return ""
    revision = proc.stdout.strip()
    return revision if proc.returncode == 0 and revision else ""


def build_harness(
    workdir: Path,
    *,
    files: dict[str, str] | None = None,
    repo: Path | None = None,
    surface: str = SURFACE_DIRECT,
) -> Harness:
    """Index a throwaway copy of a repository and bind the gate to it.

    Always a copy, never the tree it was pointed at. The stale attack has to change a
    file under a citation that was valid when it was read, and doing that to a real
    working tree -- one an editor or another process might be holding -- is not a risk
    worth taking to measure a rate.
    """
    root = workdir / "repo"
    revision = ""
    if repo is not None:
        revision = _revision_of(repo)
        shutil.copytree(
            repo,
            root,
            ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv", ".codelearner",
                                          "*.db", "node_modules"),
        )
        corpus_name = f"copy of {repo}"
    else:
        root.mkdir(parents=True, exist_ok=True)
        for rel, text in (files or SHAPES).items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        corpus_name = "shapes fixture"
    (workdir / OUTSIDE_NAME).write_text(OUTSIDE_SOURCE)

    index_path = workdir / "index.db"
    conn, _ = index_repo(root, index_path=index_path)
    facts = _load_facts(conn, root)
    conn.close()
    return Harness(
        repo=root,
        index_path=index_path,
        files=facts,
        surface=surface,
        corpus_name=corpus_name,
        revision=revision,
        pristine={path: fact.source for path, fact in facts.items()},
    )


def run_controls(
    harness: Harness,
    *,
    only: Sequence[str] | None = None,
    limit: int | None = None,
) -> GateReport:
    """Submit every control and score it. The measurement, in one pass."""
    controls, skips = build_corpus(harness.files, limit=limit, surface=harness.surface)
    if only:
        wanted = set(only)
        controls = [c for c in controls if c.family in wanted]
        skips = [s for s in skips if s[0] in wanted]
    outcomes: list[Outcome] = []
    harness.warm()
    for control in controls:
        harness.restore()
        harness.apply(control)
        before = harness.rows()
        payload, error, execution = harness.observe(control)
        if error is not None or payload is None:
            outcomes.append(Outcome(
                control=control.name, family=control.family, verdict=RAISED,
                code=type(error).__name__, rows_added=harness.rows() - before,
                evidence=0, expected_evidence=len(control.spans), servable=None,
                detail=f"{type(error).__name__}: {error}"[:200],
                surface=harness.surface,
                code_path=execution.code_path, evidence_read=execution.evidence_read,
            ))
            continue
        rows_added = harness.rows() - before
        accepted = bool(payload.get("ok"))
        outcomes.append(Outcome(
            control=control.name,
            family=control.family,
            verdict=ACCEPTED if accepted else REFUSED,
            code=None if accepted else str(payload.get("error", {}).get("code")),
            rows_added=rows_added,
            evidence=len(payload.get("evidence", ())),
            expected_evidence=len(control.spans),
            servable=payload.get("servable") if accepted else None,
            surface=harness.surface,
            code_path=execution.code_path,
            evidence_read=execution.evidence_read,
        ))
    harness.restore()
    return GateReport(
        corpus=harness.corpus_name or str(harness.repo),
        symbols=sum(len(f.symbols) for f in harness.files.values()),
        outcomes=outcomes,
        skips=skips,
        gate_path=harness.gate_path(),
        surface=harness.surface,
        revision=harness.revision,
    )


# ---------------------------------------------------------------------------
# both doors at once
# ---------------------------------------------------------------------------

@dataclass
class SurfaceComparison:
    """The same corpus, scored at every door, side by side.

    The point of the shape is that the columns are compared rather than concatenated.
    Two rates printed in two separate runs are two facts nobody puts next to each
    other; one table with a family per row makes "refused here, admitted there" the
    thing a reader sees first, and it is the only thing in this module that could have
    told anyone the README's headline described one of two doors.
    """

    reports: dict[str, GateReport]

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(self.reports)

    def divergent(self) -> list[str]:
        """Families whose HOLD RATE is not the same at every surface.

        The number that should be empty, and was not: `escaping_path` sat here at
        1.000 against 0.000 until the store gained a containment rule. Every entry is a
        place where a single headline rate would have been a lie about at least one
        door.

        An empty list is the goal state and is not the same as the doors agreeing --
        see `differently_named`, which stays non-empty by design.
        """
        names = []
        for name in FAMILIES:
            rates = {
                report.hold_rate(name)
                for report in self.reports.values()
                if report.family(name)
            }
            if len(rates) > 1:
                names.append(name)
        return names

    def differently_named(self) -> dict[str, dict[str, list[str]]]:
        """Families every door refuses, under different names.

        Not a defect and deliberately not folded into `divergent`: `past_eof` is
        `bad_range` at the server and `evidence_stale` at the store, and both are
        correct. It is reported because it is the thing a reader of two columns most
        needs and least expects -- the store's vocabulary is coarser, so four distinct
        attacks arrive there as one code, and a library caller gets a message blaming
        an edit for a path that was never in the index. Keeping this visible is also
        what stops `divergent()` returning to zero from reading as "the two doors are
        the same gate now".
        """
        differing: dict[str, dict[str, list[str]]] = {}
        for name in FAMILIES:
            by_surface = {
                surface: sorted(report.codes(name))
                for surface, report in self.reports.items()
                if report.family(name)
            }
            if len({tuple(codes) for codes in by_surface.values()}) > 1:
                differing[name] = by_surface
        return differing

    def to_json(self) -> dict[str, Any]:
        return {
            "surfaces": {name: report.to_json() for name, report in self.reports.items()},
            "divergent_families": self.divergent(),
            "differently_named_families": self.differently_named(),
        }

    def format_table(self) -> str:
        width = 22
        head = f"{'family':<{width}} {'expect':<8}"
        sub = f"{'':<{width}} {'':<8}"
        for surface in self.surfaces:
            head += f"  {surface:^37}"
            sub += f"  {'n':>6} {'probes':>6} {'rate':>7} {'ub95(pr)':>8} {'shape':<5}"
        lines = ["the same corpus, at each door it can be reached through", "", head, sub]
        for name in FAMILIES:
            row = f"{name:<{width}} {FAMILIES[name].expect:<8}"
            for surface in self.surfaces:
                report = self.reports[surface]
                if not report.family(name):
                    entry = FAMILIES[name].gates[report.gate]
                    mark = "n/a" if isinstance(entry, Inexpressible) else "-"
                    row += f"  {0:>6} {mark:>6} {'-':>7} {'-':>8} {'-':<5}"
                else:
                    stat = report.stat(name)
                    row += (
                        f"  {stat.instances:>6} {stat.probes:>6} "
                        f"{stat.hold_rate * 100:>6.1f}% {stat.upper_bound * 100:>7.2f}% "
                        f"{stat.shape[:4]:<5}"
                    )
            lines.append(row)
        lines.append("")
        for surface, report in self.reports.items():
            def show(report: GateReport, name: str) -> str:
                rate = report._rate(name)
                return "    -" if rate is None else f"{rate * 100:.1f}%"

            lines.append(
                f"{surface:<8} refused {show(report, 'rejection_rate')}  "
                f"attributed {show(report, 'attributed_rate')}  "
                f"positive {show(report, 'positive_pass_rate')}  over "
                f"{len(report.negatives)} instances / {report.negative_probes} distinct "
                f"gate executions  gate={Path(report.gate_path).name}"
            )
            lines.append(f"{'':<8} {report.caption()}")
            for name in sorted({o.family for o in report.known_gaps}):
                lines.append(f"{'':<8}   KNOWN GAP: {name} is not refused here at all")
        divergent = self.divergent()
        lines.append("")
        if divergent:
            lines.append(
                f"families that do not score the same at every door: {', '.join(divergent)}"
            )
        else:
            lines.append("every family scores the same at every door.")
        differing = self.differently_named()
        if differing:
            lines.append("refused at every door, under different names:")
            lines += [
                "  " + name + ": " + ", ".join(
                    f"{surface}={'/'.join(codes)}" for surface, codes in by_surface.items()
                )
                for name, by_surface in differing.items()
            ]
        lines += ["", INTERVAL_NOTE, "", BREADTH_LIMIT]
        return "\n".join(lines)


def compare_surfaces(
    workdir: Path,
    *,
    surfaces: Sequence[str] = (SURFACE_DIRECT, SURFACE_STORE),
    repo: Path | None = None,
    limit: int | None = None,
    only: Sequence[str] | None = None,
) -> SurfaceComparison:
    """Run the corpus once per surface, each against its OWN index.

    Its own index, not a shared one, because the surfaces write. A control admitted at
    one door would be a row the next door's `rows_added` arithmetic had to know about,
    and the stale family rewrites files under the repo; sharing either would make the
    second column a measurement of the first column's leftovers.
    """
    reports: dict[str, GateReport] = {}
    for surface in surfaces:
        with _temp_harness(workdir / f"surface-{surface}", repo=repo, surface=surface) as harness:
            reports[surface] = run_controls(harness, only=only, limit=limit)
    return SurfaceComparison(reports=reports)


# ---------------------------------------------------------------------------
# mutation: does each control detect its own rule being deleted?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationResult:
    """One family, its rule removed from a copied tree, re-measured."""

    family: str
    rule: str
    baseline_rate: float
    mutant_rate: float
    baseline_n: int
    mutant_n: int
    gate_path: str
    flipped: tuple[str, ...]
    # Which door was measured. A family whose rule has two homes has two mutations and
    # two results, and a table that did not say which was which would report the same
    # family twice with no way to tell a detected server rule from a detected store one.
    surface: str = SURFACE_DIRECT

    @property
    def detected(self) -> bool:
        """Whether removing the rule changed this family's verdict.

        Strictly lower, not merely different. A family whose hold rate is unchanged when
        its own rule is deleted is not testing that rule.
        """
        return self.mutant_rate < self.baseline_rate

    @property
    def flip_fraction(self) -> float:
        """What proportion of the family's instances the deletion actually flipped.

        The distinction `12/12 mutation-verified` threw away. Deleting a NEGATIVE rule
        admits the attack on every instance -- the mutant rate is exactly 0.0, and the
        claim is as strong as a mutation claim gets. Deleting the leniency a POSITIVE
        family names is only partially detected by construction: `published_hash` cites
        the symbol reading, and removing it falsely rejects only the symbols whose
        stored bytes are not their lines' bytes, leaving the rest admitted. Both are
        `detected`; pooling them states the weaker result over the stronger one.
        """
        if self.baseline_n <= 0:
            raise VacuousCorpus(f"{self.family!r} had no baseline instances to flip")
        return len(self.flipped) / self.baseline_n

    @property
    def partial(self) -> bool:
        return self.detected and 0.0 < self.flip_fraction < 1.0

    def row(self) -> str:
        if not self.detected:
            mark = "NOT DETECTED"
        elif self.partial:
            mark = f"detected (partial: {len(self.flipped)}/{self.baseline_n} flipped)"
        else:
            mark = f"detected ({len(self.flipped)}/{self.baseline_n} flipped)"
        return (
            f"{self.family:<24} {self.surface:<7} {self.baseline_rate:>8.3f} -> "
            f"{self.mutant_rate:<8.3f} {self.mutant_n:>5}  {mark}"
        )


def mutate_tree(dest: Path, mutation: Mutation) -> Path:
    """Copy the package into `dest` and remove one gate rule from the COPY.

    Every `old` must appear exactly once. A snippet that no longer matches means the
    rule has moved and this mutation now tests nothing -- indistinguishable, if it were
    allowed to pass, from a control that cannot detect its own rule being deleted.
    """
    package = Path(__file__).resolve().parents[1]
    tree = dest / "codelearner"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(package, tree, ignore=shutil.ignore_patterns("__pycache__"))
    for edit in mutation.edits:
        target = tree / edit.target
        text = target.read_text()
        found = text.count(edit.old)
        if found != 1:
            raise MutationFailed(
                f"{edit.target}: the snippet for {mutation.rule!r} appears {found} times, "
                f"not once. The rule moved; this mutation tests nothing until the "
                f"snippet is updated.\n---\n{edit.old[:200]}"
            )
        target.write_text(text.replace(edit.old, edit.new))
    return tree


def _child_env(tree_parent: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree_parent)
    # Re-mutating within one clock second at the same byte size makes CPython reuse a
    # cached .pyc, which looks exactly like a mutation the tests failed to detect.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONHOME", None)
    return env


def _run_child(tree_parent: Path, args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "codelearner.eval.gate_controls", "--json", *args],
        capture_output=True,
        text=True,
        env=_child_env(tree_parent),
        cwd=str(tree_parent),
        check=False,
    )
    # The exit code is deliberately NOT the pass/fail signal here: a mutant run exits
    # non-zero precisely because controls failed, which is the outcome being measured.
    # What must not be tolerated is a run that produced no measurement at all -- a
    # mutation that crashes the gate is a broken mutation, not a detected one, and the
    # two look identical if a missing report is allowed to count as a failed family.
    try:
        return dict(json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        raise MutationFailed(
            f"the child produced no report (exit {proc.returncode}). A mutation that "
            f"crashes the gate has measured nothing.\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-2000:]}"
        ) from exc


def run_mutation(
    family: str,
    workdir: Path,
    *,
    baseline: GateReport | None = None,
    surface: str = SURFACE_DIRECT,
) -> MutationResult:
    """Delete the rule this family targets AT THIS DOOR, in a copy, and re-measure it.

    `surface` is not a display detail. After WP4 several rules have two homes, and the
    edit that removes one of them changes nothing at the other door -- so a mutation
    run only at the default surface would report a rule as controlled on the strength
    of a control that never went near it. Every family with a `Rule` at both gates has
    to be run twice, and `baseline` must be a report measured at the SAME surface or
    the comparison is between two different columns.
    """
    spec = FAMILIES[family]
    rule = spec.rule(surface)
    if baseline is None:
        with _temp_harness(workdir / f"base-{family}", surface=surface) as harness:
            baseline = run_controls(harness, only=[family])
    if baseline.surface != surface:
        raise MutationFailed(
            f"the baseline for {family!r} was measured at surface "
            f"{baseline.surface!r} and the mutant at {surface!r}. The two doors refuse "
            "the same attack under different codes, so this would compare a rate to a "
            "rate about something else."
        )
    tree_parent = workdir / f"mutant-{family}-{surface}"
    tree_parent.mkdir(parents=True, exist_ok=True)
    mutate_tree(tree_parent, rule.mutation)
    payload = _run_child(tree_parent, ["--family", family, "--surface", surface])

    gate_path = str(payload["gate_path"])
    if not gate_path.startswith(str(tree_parent.resolve())):
        # The whole method rests on this. If the child imported the installed package,
        # the mutation was never in play and "not detected" would be a lie in the safe
        # direction -- or, worse, a later refactor could make it a lie in the other one.
        raise MutationFailed(
            f"the child imported {gate_path}, which is not inside the mutant tree "
            f"{tree_parent}. PYTHONPATH did not win over the installed package."
        )
    families = payload["families"]
    if family not in families:
        raise MutationFailed(f"the mutant run generated no {family!r} controls")
    mutant = families[family]
    return MutationResult(
        family=family,
        rule=rule.mutation.rule,
        baseline_rate=baseline.hold_rate(family),
        mutant_rate=float(mutant["hold_rate"]),
        baseline_n=len(baseline.family(family)),
        mutant_n=int(mutant["n"]),
        gate_path=gate_path,
        flipped=tuple(
            f["control"] for f in payload["failures"] if f["family"] == family
        ),
        surface=surface,
    )


def run_unmutated_copy(workdir: Path, *, surface: str = SURFACE_DIRECT) -> dict[str, Any]:
    """Run the whole corpus in a copied, UNmutated tree.

    The control on the mutation method itself. If a copied tree scores differently from
    the working one, then a mutant's low score says something about copying rather than
    about the rule that was deleted.
    """
    tree_parent = workdir / "unmutated"
    tree_parent.mkdir(parents=True, exist_ok=True)
    mutate_tree(tree_parent, Mutation(rule="none", edits=()))
    return _run_child(tree_parent, ["--surface", surface])


def mutable_families(surface: str) -> tuple[str, ...]:
    """The families that HAVE a rule to delete at this door.

    The rest are named rather than filtered silently: `escaping_path` at the store has
    nothing to delete because nothing is there, and `zero_length_span` at the server
    has nothing to delete because nothing can be submitted. Both are findings, and a
    runner that quietly skipped them would report a shorter, greener table.
    """
    return tuple(
        name for name, spec in FAMILIES.items() if isinstance(spec.at(surface), Rule)
    )


@dataclass(frozen=True)
class MutationCensus:
    """How many rules this corpus can delete, by door and by polarity.

    `12/12 mutation-verified` was one door and one polarity, and it is now wrong in
    three ways at once: there are two doors, several rules have a home at each of them,
    and the positive families' mutations are only partially detected. The counts are
    derived from `FAMILIES` rather than listed, so a family that gains a rule at a door
    it did not have one at is counted from the next run without anybody remembering.
    """

    negatives: tuple[tuple[str, str], ...]
    positives: tuple[tuple[str, str], ...]
    unmutable: tuple[tuple[str, str], ...]

    def cases(self) -> tuple[tuple[str, str], ...]:
        return self.negatives + self.positives

    def summary(self) -> str:
        """What there is to mutate. Structure only -- `mutation_summary` measures it.

        Deliberately not a result. `12/12 mutation-verified` conflated "there are twelve
        rules" with "twelve mutations were detected", and the two drifted apart the
        moment a rule gained a second home. This counts rules; running them counts
        detections.
        """
        doors = len({s for _, s in self.cases()})
        head = (
            f"{len(self.negatives)} negative and {len(self.positives)} positive "
            f"(family, door) rules have an edit that deletes them, over "
            f"{doors} door{'' if doors == 1 else 's'}."
        )
        if not self.unmutable:
            return head + " Every family has a rule at every door counted."
        return head + f" {len(self.unmutable)} pair(s) have nothing to delete: " + ", ".join(
            f"{f}@{s}" for f, s in self.unmutable
        )


def mutation_summary(results: Sequence[MutationResult]) -> str:
    """The measured replacement for `12/12 mutation-verified`, polarities apart.

    Three counts, because they are three different strengths of claim and pooling them
    states the weakest. A NEGATIVE rule's deletion admits or misattributes the attack on
    every instance -- the mutant hold rate is exactly 0.000, which is as strong as a
    mutation result gets. A POSITIVE rule's deletion is detected, but often only
    partially: `published_hash` names the symbol reading of a line range, and deleting
    it falsely rejects only the symbols whose stored bytes are not their lines' bytes.
    Reporting `detected` for both loses the difference; reporting the partial ones as
    the headline understates the negative result, which is the best-verified thing here.
    """
    negatives = [r for r in results if FAMILIES[r.family].expect == REFUSED]
    positives = [r for r in results if FAMILIES[r.family].expect == ACCEPTED]
    total = [r for r in negatives if r.detected and r.mutant_rate == 0.0]
    detected = [r for r in positives if r.detected]
    partial = [r for r in detected if r.partial]
    return (
        f"{len(total)}/{len(negatives)} negative rules produce an admitted or "
        f"misattributed attack on EVERY instance when deleted (mutant hold rate 0.000); "
        f"{len(detected)}/{len(positives)} positive rules are detected, {len(partial)} of "
        f"them only partially -- deleting one reading of a legitimate citation flips the "
        f"instances that needed that reading and leaves the rest admitted "
        + (
            "(" + ", ".join(
                f"{r.family}@{r.surface} {len(r.flipped)}/{r.baseline_n}" for r in partial
            ) + ")"
            if partial
            else "(none partial in this run)"
        )
    )


def mutation_census(
    surfaces: Sequence[str] = (SURFACE_DIRECT, SURFACE_STORE),
) -> MutationCensus:
    """Count the mutations this corpus can run, at every door, polarity kept apart."""
    negatives, positives, unmutable = [], [], []
    for surface in surfaces:
        for name, spec in FAMILIES.items():
            if not isinstance(spec.at(surface), Rule):
                unmutable.append((name, surface))
            elif spec.expect == REFUSED:
                negatives.append((name, surface))
            else:
                positives.append((name, surface))
    return MutationCensus(
        negatives=tuple(negatives), positives=tuple(positives), unmutable=tuple(unmutable)
    )


def run_mutations(
    workdir: Path,
    *,
    families: Sequence[str] | None = None,
    surface: str = SURFACE_DIRECT,
) -> list[MutationResult]:
    names = [n for n in (families or FAMILIES) if n in mutable_families(surface)]
    results = []
    for name in names:
        results.append(run_mutation(name, workdir, surface=surface))
    return results


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

@contextmanager
def _temp_harness(path: Path | None = None, **kwargs: Any) -> Iterator[Harness]:
    """A harness in a throwaway directory, closed and removed on the way out."""
    owned = None if path is not None else tempfile.mkdtemp(prefix="gate-controls-")
    root = Path(owned) if owned is not None else path
    if root is None:  # pragma: no cover - unreachable, kept for the type narrowing
        raise ValueError("no directory to build a harness in")
    root.mkdir(parents=True, exist_ok=True)
    harness = build_harness(root, **kwargs)
    try:
        yield harness
    finally:
        harness.close()
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codelearner.eval.gate_controls",
        description="Adversarial controls for the tier-2 assertion gate.",
    )
    parser.add_argument("--repo", type=Path, default=None,
                        help="index a COPY of this tree instead of the shapes fixture")
    parser.add_argument("--family", action="append", default=None,
                        help="restrict to one attack family (repeatable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap instances per family")
    parser.add_argument("--surface", choices=SURFACES, default=SURFACE_DIRECT,
                        help="the tool body, the registered MCP tool, or "
                             "store.write_assertion as a library caller reaches it")
    parser.add_argument("--compare", action="store_true",
                        help="run the corpus at BOTH doors and print two columns")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--mutations", action="store_true",
                        help="also delete each rule in a copied tree and re-measure")
    args = parser.parse_args(argv)

    if args.compare:
        tmp = Path(tempfile.mkdtemp(prefix="gate-surfaces-"))
        try:
            comparison = compare_surfaces(
                tmp, repo=args.repo, limit=args.limit, only=args.family
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps(comparison.to_json()) if args.json else comparison.format_table())
        return 0 if not any(
            r.unexpected_failures for r in comparison.reports.values()
        ) else 1

    with _temp_harness(repo=args.repo, surface=args.surface) as harness:
        report = run_controls(harness, only=args.family, limit=args.limit)
    if args.json:
        print(json.dumps(report.to_json()))
    else:
        print(report.format_table())

    if args.mutations:
        tmp = Path(tempfile.mkdtemp(prefix="gate-mutations-"))
        try:
            results = run_mutations(tmp, families=args.family, surface=args.surface)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if args.json:
            print(json.dumps([r.__dict__ for r in results]))
        else:
            print("")
            print(f"{'family':<24} {'surface':<7} {'baseline':>8}    {'mutant':<8} "
                  f"{'n':>5}  verdict")
            for result in results:
                print(result.row())
            for name in FAMILIES:
                if name not in mutable_families(args.surface):
                    entry = FAMILIES[name].at(args.surface)
                    why = getattr(entry, "detail", None) or entry.reason  # type: ignore[union-attr]
                    print(f"{name:<24} {args.surface:<7} NO MUTATION -- {why}")
            print("")
            print(mutation_census([args.surface]).summary())
            print(mutation_summary(results))
        if any(not r.detected for r in results):
            return 1
    # A known gap is a measurement, not a regression, and a command that exits non-zero
    # every single time it is run is a command whose exit code stops being read. The
    # gap is still printed, still counted in the rate, and still a failure in the
    # report -- it is only the process's verdict that distinguishes "a declared hole is
    # still open" from "something that used to hold has stopped".
    return 0 if not report.unexpected_failures else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
