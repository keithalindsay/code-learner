"""Source evidence rendering helpers."""

from __future__ import annotations


def number_source(source: str, line_start: int) -> str:
    if line_start < 1:
        raise ValueError("line_start must be >= 1")
    return "".join(
        f"{line_number} | {line}"
        for line_number, line in enumerate(source.splitlines(keepends=True), line_start)
    )


def content_bytes(text: str) -> int:
    return len(text.encode("utf-8"))
