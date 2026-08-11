"""Immutable source evidence response types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSection:
    symbol_id: int
    qualname: str
    path: str
    line_start: int
    line_end: int
    content_hash: str
    source: str
    content_bytes: int


@dataclass(frozen=True)
class EvidenceBundle:
    sections: tuple[EvidenceSection, ...]
    budget_bytes: int
    used_bytes: int
    sections_omitted: int
    omitted_symbol_ids: tuple[int, ...]

    @property
    def truncated(self) -> bool:
        return self.sections_omitted > 0

    def as_json(self) -> dict[str, object]:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "truncated": self.truncated,
            "sections_omitted": self.sections_omitted,
            "omitted_symbol_ids": list(self.omitted_symbol_ids),
            "sections": [
                {
                    "symbol_id": section.symbol_id,
                    "qualname": section.qualname,
                    "path": section.path,
                    "line_start": section.line_start,
                    "line_end": section.line_end,
                    "content_hash": section.content_hash,
                    "content_bytes": section.content_bytes,
                    "source": section.source,
                }
                for section in self.sections
            ],
        }
