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
import json
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
class Family:
    """One attack (or one legitimate submission), the rule it targets, and its verdict."""

    name: str
    expect: str
    attack: str
    codes: frozenset[str]
    mutation: Mutation


def _mut(rule: str, *edits: Edit) -> Mutation:
    return Mutation(rule=rule, edits=edits)


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
        codes=frozenset({"evidence_required"}),
        mutation=_mut(
            "store.write_assertion refuses an empty span list before opening a "
            "transaction",
            Edit(
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
                    "    with _atomic(conn):"
                ),
                new="    spans = tuple(spans)\n    with _atomic(conn):",
            ),
        ),
    ),
    "absent_file": Family(
        name="absent_file",
        expect=REFUSED,
        attack="the right content cited in a file that does not exist",
        # Still exactly one code, and that is now load-bearing rather than incidental.
        # WP2 put an index-membership check ahead of the read, so this control is
        # refused before anything is stat'd -- and it is refused with the SAME code and
        # the same shape as a path that is present on disk but unindexed. A second code
        # would have made the refusal answer "does this file exist", which is the
        # oracle WP2 exists to close. If this set ever grows, check that the new code
        # is not distinguishing two paths a caller is not entitled to tell apart.
        codes=frozenset({"file_missing"}),
        mutation=_mut(
            "_verify_span refuses a citation it cannot, or will not, read off disk",
            # Three edits, one rule. Defence in depth means no single deletion admits
            # the attack, so a mutation that removed only one guard would report this
            # control as undetectable when it is in fact over-defended. Each `old` is
            # a single line and each is unique in the module, so drift in the
            # surrounding comments cannot silently turn this into a no-op -- the
            # harness raises on a snippet that no longer matches exactly once.
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
        ),
    ),
    "escaping_path": Family(
        name="escaping_path",
        expect=REFUSED,
        attack="a real file, really hashed, from outside the indexed repository",
        codes=frozenset({"path_escapes_repo"}),
        mutation=_mut(
            "_verify_span refuses a path that resolves outside the repo root",
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
        ),
    ),
    "past_eof": Family(
        name="past_eof",
        expect=REFUSED,
        attack="a range running one line past the end, quoting the last line that exists",
        codes=frozenset({"bad_range"}),
        mutation=_mut(
            "_line_bytes refuses a line range the file does not have (rather than "
            "clamping it)",
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
    "blank_range": Family(
        name="blank_range",
        expect=REFUSED,
        attack="a blank line cited as evidence, with the hash of nothing",
        codes=frozenset({"bad_range"}),
        mutation=_mut(
            "store.span_for refuses an empty byte range",
            Edit(
                target=STORE_MODULE,
                old="    if not 0 <= byte_start < byte_end <= len(source):",
                new="    if not 0 <= byte_start <= byte_end <= len(source):",
            ),
        ),
    ),
    "decoy_content_hash": Family(
        name="decoy_content_hash",
        expect=REFUSED,
        attack="the right lines cited with the hash of other content in the same file",
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
        ),
    ),
    "stale_but_once_valid": Family(
        name="stale_but_once_valid",
        expect=REFUSED,
        attack="a hash that was correct before the file changed under it",
        codes=frozenset({"hash_mismatch"}),
        mutation=_mut(
            "_verify_span hashes the bytes on disk NOW, never the index's stored hash",
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
        ),
    ),
    "foreign_symbol_hash": Family(
        name="foreign_symbol_hash",
        expect=REFUSED,
        attack="the hash of a DIFFERENT indexed symbol in the same file",
        codes=frozenset({"hash_mismatch"}),
        mutation=_mut(
            "_symbol_bytes_at admits only symbols occupying EXACTLY the cited lines",
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
    "unknown_subject": Family(
        name="unknown_subject",
        expect=REFUSED,
        attack="a perfectly cited claim about a symbol that does not exist",
        codes=frozenset({"unknown_subject"}),
        mutation=_mut(
            "_submit_body refuses a subject_qualname that names no indexed symbol",
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
    # --- positive controls ------------------------------------------------
    "published_hash": Family(
        name="published_hash",
        expect=ACCEPTED,
        attack="the loop the design rests on: cite the hash retrieval handed you",
        codes=frozenset(),
        mutation=_mut(
            "_verify_span checks the cited hash against the SYMBOL's bytes as well as "
            "the lines' bytes",
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
    "quoted_lines": Family(
        name="quoted_lines",
        expect=ACCEPTED,
        attack="the other honest reading: the exact lines, copied out of the file",
        codes=frozenset(),
        mutation=_mut(
            "_verify_span checks the cited hash against the whole lines' bytes as well "
            "as the symbol's",
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
    "multi_span": Family(
        name="multi_span",
        expect=ACCEPTED,
        attack="two good citations on one claim, both of which must be stored",
        codes=frozenset(),
        mutation=_mut(
            "_submit_body verifies and stores EVERY submitted span, not the first one",
            Edit(
                target=GATE_MODULE,
                old="    spans = [_verify_span(conn, root, raw) for raw in evidence_spans]",
                new="    spans = [_verify_span(conn, root, raw) for raw in evidence_spans[:1]]",
            ),
        ),
    ),
}

NEGATIVE_FAMILIES = tuple(f for f, spec in FAMILIES.items() if spec.expect == REFUSED)
POSITIVE_FAMILIES = tuple(f for f, spec in FAMILIES.items() if spec.expect == ACCEPTED)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Control:
    """One submission the gate must refuse (or admit), and what makes it adversarial."""

    name: str
    family: str
    subject_qualname: str
    claim: str
    spans: tuple[dict[str, Any], ...]
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


def build_corpus(
    files: dict[str, FileFact],
    *,
    limit: int | None = None,
) -> tuple[list[Control], list[tuple[str, str, str]]]:
    """Generate the corpus from what the index actually holds.

    One instance per attack per symbol, wherever the attack is constructible. Where it
    is not -- a file with no blank line has nothing to cite blankly -- the instance is
    SKIPPED WITH A REASON and returned, not dropped. A skip that vanishes is how a
    family quietly becomes empty, and an empty family is how a rejection rate becomes
    a statement about nothing.

    `limit` caps instances per family, for a fast run. It is applied per family rather
    than by truncating the symbol list, so no family can be emptied by it.
    """
    controls: list[Control] = []
    skips: list[tuple[str, str, str]] = []
    per_family: dict[str, int] = dict.fromkeys(FAMILIES, 0)

    def add(control: Control) -> None:
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
            published = {
                "path": symbol.path,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "content_hash": symbol.content_hash,
            }
            accepted = fact.accepted_hashes(symbol.line_start, symbol.line_end)

            add(Control(
                name=f"zero_evidence/{qual}",
                family="zero_evidence",
                subject_qualname=qual,
                claim=f"{qual} is the entry point for the whole subsystem",
                spans=(),
            ))
            add(Control(
                name=f"absent_file/{qual}",
                family="absent_file",
                subject_qualname=qual,
                claim=f"{qual} is defined in a file this index has never seen",
                spans=({**published, "path": f"{symbol.path}.absent"},),
            ))
            add(Control(
                name=f"escaping_path/{qual}",
                family="escaping_path",
                subject_qualname=qual,
                claim=f"{qual} reads a secret from outside the repository",
                spans=({
                    "path": f"../{OUTSIDE_NAME}",
                    "line_start": 1,
                    "line_end": 1,
                    "content_hash": content_hash(outside_line.encode()),
                },),
            ))
            add(Control(
                name=f"unknown_subject/{qual}",
                family="unknown_subject",
                subject_qualname=f"{qual}_that_does_not_exist",
                claim=f"{qual}_that_does_not_exist does the work {qual} is credited with",
                spans=(dict(published),),
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
                    spans=({**published, "content_hash": decoy},),
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
                    spans=({**published, "content_hash": foreign},),
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
                    spans=(dict(published),),
                    edit=(fact.path, edited),
                ))

            # --- positives ------------------------------------------------
            add(Control(
                name=f"published_hash/{qual}",
                family="published_hash",
                subject_qualname=qual,
                claim=f"{qual} is cited by the hash this index published for it",
                spans=(dict(published),),
            ))
            quoted = fact.text_at(symbol.line_start, symbol.line_end)
            if quoted is None:
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
                    spans=({
                        "path": symbol.path,
                        "line_start": symbol.line_start,
                        "line_end": symbol.line_end,
                        "text": quoted,
                    },),
                ))

        # --- per-file attacks ---------------------------------------------
        if not symbols:
            skips.append(("past_eof", fact.path, "no indexed symbol to be the subject"))
            continue
        subject = symbols[0].qualname
        last_line = fact.text_at(fact.line_count, fact.line_count)
        if last_line is None:
            skips.append(("past_eof", fact.path, "the file has no last line"))
        else:
            add(Control(
                name=f"past_eof/{fact.path}",
                family="past_eof",
                subject_qualname=subject,
                claim=f"{subject} continues past the end of {fact.path}",
                spans=({
                    "path": fact.path,
                    "line_start": fact.line_count,
                    "line_end": fact.line_count + 1,
                    "text": last_line,
                },),
            ))
        if not blanks:
            skips.append(("blank_range", fact.path, "the file has no blank line"))
        else:
            blank = blanks[0]
            add(Control(
                name=f"blank_range/{fact.path}#text",
                family="blank_range",
                subject_qualname=subject,
                claim=f"{subject} is documented on line {blank} of {fact.path}",
                spans=({"path": fact.path, "line_start": blank, "line_end": blank, "text": ""},),
            ))
            add(Control(
                name=f"blank_range/{fact.path}#hash",
                family="blank_range",
                subject_qualname=subject,
                claim=f"{subject} is documented on line {blank} of {fact.path}",
                spans=({
                    "path": fact.path,
                    "line_start": blank,
                    "line_end": blank,
                    "content_hash": EMPTY_SHA,
                },),
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
                spans=(
                    {
                        "path": first.path,
                        "line_start": first.line_start,
                        "line_end": first.line_end,
                        "content_hash": first.content_hash,
                    },
                    {
                        "path": second.path,
                        "line_start": second.line_start,
                        "line_end": second.line_end,
                        "content_hash": second.content_hash,
                    },
                ),
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
# running the corpus against the real gate
# ---------------------------------------------------------------------------

@dataclass
class Harness:
    """A disposable repo, an index over it, and the gate bound to both."""

    repo: Path
    index_path: Path
    files: dict[str, FileFact]
    surface: str = "direct"
    corpus_name: str = ""
    pristine: dict[str, bytes] = field(default_factory=dict)
    dirty: set[str] = field(default_factory=set)
    _source: Any = None
    _server: Any = None

    @property
    def source(self) -> Any:
        if self._source is None:
            self._source = gate_module().IndexSource(path=self.index_path)
        return self._source

    @property
    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection = self.source.connect()
        return conn

    def rows(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM assertions").fetchone()[0])

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

    def submit(self, control: Control) -> dict[str, Any]:
        """Put one control through the gate, on the surface an agent would reach.

        `direct` calls the tool body under `_guard`, which is the whole gate and the
        whole error contract. `tool` goes through the registered MCP tool as a client
        would, so the same corpus can prove that a refusal arrives as data rather than
        as a traceback -- the module's other rule, and one that a direct call cannot
        check. Both must return the same verdict for every control; that is a test.
        """
        app = gate_module()
        spans = [app.EvidenceSpanInput(**span) for span in control.spans]
        if self.surface == "tool":
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

    @property
    def expect(self) -> str:
        return FAMILIES[self.family].expect

    @property
    def held(self) -> bool:
        """Whether this control's rule did its whole job.

        For a refusal that means the expected code AND no row: a gate that says no and
        writes the row anyway has refused nothing. For an admission it means servable,
        exactly one row, and every submitted span stored -- an accepted claim carrying
        half its evidence stands on less than its author thought it did, and nothing
        downstream would record that the rest was dropped.
        """
        spec = FAMILIES[self.family]
        if spec.expect == REFUSED:
            return self.verdict == REFUSED and self.code in spec.codes and self.rows_added == 0
        return (
            self.verdict == ACCEPTED
            and self.servable is True
            and self.rows_added == 1
            and self.evidence == self.expected_evidence
        )

    @property
    def refused(self) -> bool:
        return self.verdict == REFUSED


@dataclass
class GateReport:
    """The measurement: a rejection rate, a positive pass rate, and what failed."""

    corpus: str
    symbols: int
    outcomes: list[Outcome]
    skips: list[tuple[str, str, str]]
    gate_path: str
    surface: str = "direct"

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
            "surface": self.surface,
            "symbols": self.symbols,
            "gate_path": self.gate_path,
            "negatives": len(self.negatives),
            "positives": len(self.positives),
            "rejection_rate": self._rate("rejection_rate"),
            "attributed_rate": self._rate("attributed_rate"),
            "positive_pass_rate": self._rate("positive_pass_rate"),
            "families": {
                name: {
                    "expect": FAMILIES[name].expect,
                    "n": len(self.family(name)),
                    "held": sum(1 for o in self.family(name) if o.held),
                    "hold_rate": self.hold_rate(name) if self.family(name) else None,
                    "codes": self.codes(name),
                }
                for name in FAMILIES
                if self.family(name)
            },
            "failures": [
                {"control": o.control, "family": o.family, "verdict": o.verdict,
                 "code": o.code, "rows_added": o.rows_added, "detail": o.detail}
                for o in self.failures
            ],
            "skips": [{"family": f, "subject": s, "reason": r} for f, s, r in self.skips],
        }

    def format_table(self) -> str:
        lines = [
            f"corpus: {self.corpus}  ({self.symbols} symbols, surface={self.surface})",
            f"gate:   {self.gate_path}",
            "",
            f"{'family':<24} {'expect':<9} {'n':>5} {'held':>5} {'rate':>7}  codes",
        ]
        for name in FAMILIES:
            group = self.family(name)
            if not group:
                lines.append(f"{name:<24} {FAMILIES[name].expect:<9} {0:>5} {'-':>5} {'-':>7}  (no instances)")
                continue
            held = sum(1 for o in group if o.held)
            codes = ", ".join(f"{k}={v}" for k, v in sorted(self.codes(name).items()))
            lines.append(
                f"{name:<24} {FAMILIES[name].expect:<9} {len(group):>5} {held:>5} "
                f"{self.hold_rate(name):>7.3f}  {codes}"
            )
        def show(name: str) -> str:
            rate = self._rate(name)
            return "     -" if rate is None else f"{rate:.4f}"

        lines += [
            "",
            f"rejection rate      {show('rejection_rate')}  ({len(self.negatives)} attacks)",
            f"attributed rate     {show('attributed_rate')}  (refused by the rule it targets, "
            "no row written)",
            f"positive pass rate  {show('positive_pass_rate')}  ({len(self.positives)} "
            "legitimate submissions)",
        ]
        if self.failures:
            lines.append("")
            lines.append("FAILURES:")
            lines += [
                f"  {o.control} -> {o.verdict} {o.code or ''} rows={o.rows_added} {o.detail}"
                for o in self.failures[:40]
            ]
        return "\n".join(lines)


def build_harness(
    workdir: Path,
    *,
    files: dict[str, str] | None = None,
    repo: Path | None = None,
    surface: str = "direct",
) -> Harness:
    """Index a throwaway copy of a repository and bind the gate to it.

    Always a copy, never the tree it was pointed at. The stale attack has to change a
    file under a citation that was valid when it was read, and doing that to a real
    working tree -- one an editor or another process might be holding -- is not a risk
    worth taking to measure a rate.
    """
    root = workdir / "repo"
    if repo is not None:
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
        pristine={path: fact.source for path, fact in facts.items()},
    )


def run_controls(
    harness: Harness,
    *,
    only: Sequence[str] | None = None,
    limit: int | None = None,
) -> GateReport:
    """Submit every control and score it. The measurement, in one pass."""
    controls, skips = build_corpus(harness.files, limit=limit)
    if only:
        wanted = set(only)
        controls = [c for c in controls if c.family in wanted]
        skips = [s for s in skips if s[0] in wanted]
    app = gate_module()
    outcomes: list[Outcome] = []
    for control in controls:
        harness.restore()
        harness.apply(control)
        before = harness.rows()
        try:
            payload = harness.submit(control)
        except Exception as exc:  # a raise IS the finding; never let it end the run
            outcomes.append(Outcome(
                control=control.name, family=control.family, verdict=RAISED,
                code=type(exc).__name__, rows_added=harness.rows() - before,
                evidence=0, expected_evidence=len(control.spans), servable=None,
                detail=f"{type(exc).__name__}: {exc}"[:200],
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
        ))
    harness.restore()
    return GateReport(
        corpus=harness.corpus_name or str(harness.repo),
        symbols=sum(len(f.symbols) for f in harness.files.values()),
        outcomes=outcomes,
        skips=skips,
        gate_path=str(Path(app.__file__ or "?").resolve()),
        surface=harness.surface,
    )


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

    @property
    def detected(self) -> bool:
        """Whether removing the rule changed this family's verdict.

        Strictly lower, not merely different. A family whose hold rate is unchanged when
        its own rule is deleted is not testing that rule.
        """
        return self.mutant_rate < self.baseline_rate

    def row(self) -> str:
        mark = "detected" if self.detected else "NOT DETECTED"
        return (
            f"{self.family:<24} {self.baseline_rate:>8.3f} -> {self.mutant_rate:<8.3f} "
            f"{self.mutant_n:>4}  {mark}"
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


def run_mutation(family: str, workdir: Path, *, baseline: GateReport | None = None) -> MutationResult:
    """Delete the rule this family targets, in a copy, and re-measure the family."""
    spec = FAMILIES[family]
    if baseline is None:
        with _temp_harness(workdir / f"base-{family}") as harness:
            baseline = run_controls(harness, only=[family])
    tree_parent = workdir / f"mutant-{family}"
    tree_parent.mkdir(parents=True, exist_ok=True)
    mutate_tree(tree_parent, spec.mutation)
    payload = _run_child(tree_parent, ["--family", family])

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
        rule=spec.mutation.rule,
        baseline_rate=baseline.hold_rate(family),
        mutant_rate=float(mutant["hold_rate"]),
        baseline_n=len(baseline.family(family)),
        mutant_n=int(mutant["n"]),
        gate_path=gate_path,
        flipped=tuple(
            f["control"] for f in payload["failures"] if f["family"] == family
        ),
    )


def run_unmutated_copy(workdir: Path) -> dict[str, Any]:
    """Run the whole corpus in a copied, UNmutated tree.

    The control on the mutation method itself. If a copied tree scores differently from
    the working one, then a mutant's low score says something about copying rather than
    about the rule that was deleted.
    """
    tree_parent = workdir / "unmutated"
    tree_parent.mkdir(parents=True, exist_ok=True)
    mutate_tree(tree_parent, Mutation(rule="none", edits=()))
    return _run_child(tree_parent, [])


def run_mutations(workdir: Path, *, families: Sequence[str] | None = None) -> list[MutationResult]:
    names = list(families or FAMILIES)
    results = []
    for name in names:
        results.append(run_mutation(name, workdir))
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
    parser.add_argument("--surface", choices=("direct", "tool"), default="direct",
                        help="call the tool body, or the registered MCP tool")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--mutations", action="store_true",
                        help="also delete each rule in a copied tree and re-measure")
    args = parser.parse_args(argv)

    with _temp_harness(repo=args.repo, surface=args.surface) as harness:
        report = run_controls(harness, only=args.family, limit=args.limit)
    if args.json:
        print(json.dumps(report.to_json()))
    else:
        print(report.format_table())

    if args.mutations:
        tmp = Path(tempfile.mkdtemp(prefix="gate-mutations-"))
        try:
            results = run_mutations(tmp, families=args.family)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if args.json:
            print(json.dumps([r.__dict__ for r in results]))
        else:
            print("")
            print(f"{'family':<24} {'baseline':>8}    {'mutant':<8} {'n':>4}  verdict")
            for result in results:
                print(result.row())
        if any(not r.detected for r in results):
            return 1
    return 0 if not report.failures else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
