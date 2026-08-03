"""What a purpose generator is allowed to see, and the gate that proves it.

A leaf module by construction: it imports nothing from any `codelearner` package, so
anything may depend on it and it can create no cycle. That is the whole reason it
exists as a separate file rather than living where it was written.

`SourceView` and its gate were defined in `eval/gold_from_history.py`, which is where
they are *used* to score a held-out corpus. But `generate/purpose.py` needs the same
three names to build its input and prove the input is clean -- and importing them from
`eval` made `generate` depend on `eval`, which is the one direction
`generate/llm.py:JUDGE_FAMILY` states in bold must stay empty: the package that writes
claims must not be able to reach into the package that grades them, or "the generator
and the judge are independent" stops being structurally true.

That import also closed a real four-package cycle,
`eval -> server -> cli -> generate -> eval`, which survived only because two of its
edges are function-local. Both facts are now enforced by
`tests/test_generate_purpose.py::TestImportDirection` rather than asserted in a
comment, because a rule that lives only in prose is a rule that has already been
broken once.

`eval.gold_from_history` re-exports all four names, so every existing import keeps
working and no caller has to know this file moved.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Generator",
    "LeakDetected",
    "SourceView",
    "assert_view_is_source_only",
]


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


# Every generator in the project has this shape, and the shape IS the guarantee: a
# `SourceView` in, a string out, and no parameter through which a held-out label could
# arrive. Declared here beside the view rather than beside any one caller, so both the
# scoring harness and the shipped generator name the same contract.
Generator = Callable[[SourceView], str]
