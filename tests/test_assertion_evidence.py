"""Bounded, all-or-nothing evidence for mixed source and assertion candidates."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from codelearner.assertions import store
from codelearner.evidence import (
    MAX_EVIDENCE_BYTES,
    EvidenceError,
    assemble_candidate_evidence,
)
from codelearner.ingest import index_repo
from codelearner.retrieve.lexical import Hit
from codelearner.retrieve.types import (
    AssertionCandidate,
    Freshness,
    SourceCandidate,
    VerdictSummary,
)

CITATION_SOURCE = '2 | """Coordinates lease renewal \u03bb."""'

SOURCE = '''def acquire():
    """Coordinates lease renewal λ."""
    return 1


def caller():
    return acquire()
'''


@pytest.fixture()
def repo(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _symbol_id(conn, qualname: str) -> int:
    return int(
        conn.execute(
            "SELECT id FROM symbols WHERE qualname = ?", (qualname,)
        ).fetchone()["id"]
    )


def _source_candidate(conn, qualname: str) -> SourceCandidate:
    row = conn.execute(
        "SELECT s.id, s.kind, s.qualname, s.line_start, s.line_end, f.path "
        "FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
        (qualname,),
    ).fetchone()
    return SourceCandidate.from_hit(
        Hit(
            symbol_id=int(row["id"]),
            qualname=str(row["qualname"]),
            kind=str(row["kind"]),
            path=str(row["path"]),
            line_start=int(row["line_start"]),
            line_end=int(row["line_end"]),
            score=1.0,
            modality="lexical",
            header="",
        )
    )


def _span(root: Path, needle: str) -> store.EvidenceSpan:
    raw = (root / "leases.py").read_bytes()
    start = raw.index(needle.encode("utf-8"))
    return store.span_for(root, "leases.py", start, start + len(needle.encode("utf-8")))


def _admit(
    root: Path,
    conn,
    *,
    claim: str = "Renews ownership during long work.",
    spans: tuple[store.EvidenceSpan, ...] | None = None,
    subject: str = "leases.acquire",
    verdict: str | None = store.VERDICT_SUPPORTED,
) -> int:
    assertion_id = store.write_assertion(
        conn,
        subject_qualname=subject,
        subject_symbol_id=_symbol_id(conn, subject),
        kind="purpose",
        claim=claim,
        spans=spans or (_span(root, '"""Coordinates lease renewal λ."""'),),
        generator="test/v1",
    )
    if verdict is not None:
        store.record_verdict(conn, assertion_id, "judge/v1", verdict, "cited bytes say so")
    return assertion_id


def _assertion_candidate(conn, assertion_id: int, **overrides) -> AssertionCandidate:
    assertion = store.load_assertions_by_ids(conn, [assertion_id])[0]
    verdicts = tuple(
        VerdictSummary(
            judge=str(row["judge"]),
            verdict=str(row["verdict"]),
            rationale=None if row["rationale"] is None else str(row["rationale"]),
        )
        for row in store.verdicts_for(conn, assertion_id)
        if row["verdict"] == store.VERDICT_SUPPORTED
    )
    fields = {
        "assertion_id": assertion.id,
        "subject_symbol_id": assertion.subject_symbol_id,
        "subject_qualname": assertion.subject_qualname,
        "kind": assertion.kind,
        "claim": assertion.claim,
        "generator": assertion.generator,
        "status": assertion.status,
        "verdicts": verdicts,
        "freshness": Freshness(verified=True, method="hash"),
        "spans": assertion.spans,
        "score": 1.0,
        "modality": "assertion_lexical",
        "conflict": False,
        "contributions": (),
    }
    fields.update(overrides)
    return AssertionCandidate(**fields)


def test_source_and_assertion_candidates_stay_tagged_and_ordered(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    candidates = [
        _assertion_candidate(conn, assertion_id),
        _source_candidate(conn, "leases.caller"),
    ]

    bundle = assemble_candidate_evidence(conn, root, candidates, budget_bytes=10_000)

    assert [(r.candidate_type, r.candidate_key, r.rank) for r in bundle.results] == [
        ("assertion", f"assertion:{assertion_id}", 1),
        ("source", f"source:{_symbol_id(conn, 'leases.caller')}", 2),
    ]
    assert bundle.results[0].section is None
    assert bundle.results[1].assertion is None
    assert bundle.results[0].assertion is not None
    assert bundle.results[1].section is not None
    assert bundle.omitted == ()


def test_assertion_evidence_carries_claim_verdicts_and_citations(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    evidence = bundle.results[0].assertion
    assert evidence.claim == "Renews ownership during long work."
    assert evidence.kind == "purpose"
    assert evidence.subject_qualname == "leases.acquire"
    assert [(v.judge, v.verdict) for v in evidence.verdicts] == [
        ("judge/v1", "supported")
    ]
    assert evidence.freshness.verified is True
    citation = evidence.citations[0]
    assert citation.path == "leases.py"
    assert citation.line_start == 2
    assert citation.line_end == 2
    assert citation.source == CITATION_SOURCE
    assert citation.content_hash == evidence.citations[0].content_hash


def test_assertion_evidence_includes_the_subject_symbol_source(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    subject = bundle.results[0].assertion.subject
    assert subject.qualname == "leases.acquire"
    assert subject.source.startswith("1 | def acquire():")
    assert "return 1" in subject.source


def test_assertion_evidence_reports_live_lines_after_an_equal_byte_prefix_edit(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    original = (root / "leases.py").read_bytes()
    start = original.index(b'"""Coordinates')
    # Same byte length, same cited bytes: only the newline count before them moves.
    moved = original[:start].replace(b"\n    ", b"\n\n   ", 1)
    assert len(moved) == start
    (root / "leases.py").write_bytes(moved + original[start:])

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    citation = bundle.results[0].assertion.citations[0]
    assert citation.line_start == 3
    assert citation.source.startswith("3 |")


def test_multiple_citations_are_ordered_by_path_then_byte_offset(repo):
    root, conn = repo
    assertion_id = _admit(
        root,
        conn,
        spans=(_span(root, "return 1"), _span(root, "def acquire():")),
    )

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    citations = bundle.results[0].assertion.citations
    assert [c.byte_start for c in citations] == sorted(c.byte_start for c in citations)
    assert citations[0].source.endswith("def acquire():")


def test_assertion_evidence_reports_subject_callers_and_callees(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    related = bundle.results[0].assertion.related
    assert ("caller", "leases.caller") in [(r.relation, r.qualname) for r in related]


def test_subject_section_is_not_repeated_when_a_citation_already_covers_it(repo):
    root, conn = repo
    whole = (root / "leases.py").read_bytes()
    assertion_id = _admit(
        root, conn, spans=(store.span_for(root, "leases.py", 0, len(whole)),)
    )

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    evidence = bundle.results[0].assertion
    assert evidence.subject is None
    assert evidence.subject_reason == "covered_by_citation"
    assert evidence.citations[0].source.startswith("1 | def acquire():")
    assert bundle.used_bytes == evidence.citations[0].content_bytes


def test_a_citation_contained_in_another_citation_is_rendered_once(repo):
    root, conn = repo
    whole = (root / "leases.py").read_bytes()
    assertion_id = _admit(
        root,
        conn,
        spans=(
            store.span_for(root, "leases.py", 0, len(whole)),
            _span(root, "return 1"),
        ),
    )

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    citations = bundle.results[0].assertion.citations
    assert len(citations) == 2
    rendered = [c for c in citations if c.source is not None]
    assert len(rendered) == 1
    contained = next(c for c in citations if c.source is None)
    assert contained.duplicate is True
    assert contained.content_bytes == 0
    assert contained.line_start == 3


def test_citation_coordinates_are_utf8_bytes_not_characters(repo):
    root, conn = repo
    assertion_id = _admit(root, conn, spans=(_span(root, "renewal λ"),))

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    citation = bundle.results[0].assertion.citations[0]
    assert citation.byte_end - citation.byte_start == len("renewal λ".encode())
    assert citation.source == "2 | renewal λ"


def test_a_changed_citation_withholds_the_whole_semantic_candidate(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    (root / "leases.py").write_text(SOURCE.replace("renewal λ", "renewal now"))

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    assert bundle.results == ()
    assert [(o.key, o.reason) for o in bundle.omitted] == [
        (f"assertion:{assertion_id}", "citation_changed")
    ]
    assert "Renews ownership" not in json.dumps(bundle.as_json())


def test_a_missing_cited_file_withholds_the_whole_semantic_candidate(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    (root / "leases.py").unlink()

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10_000
    )

    assert bundle.results == ()
    assert [o.reason for o in bundle.omitted] == ["citation_file_missing"]


def test_a_semantic_candidate_that_cannot_fit_its_citations_is_omitted_whole(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)

    bundle = assemble_candidate_evidence(
        conn, root, [_assertion_candidate(conn, assertion_id)], budget_bytes=10
    )

    assert bundle.results == ()
    assert [(o.key, o.reason) for o in bundle.omitted] == [
        (f"assertion:{assertion_id}", "budget")
    ]
    assert bundle.used_bytes == 0


def test_a_subject_that_does_not_fit_is_dropped_without_dropping_the_claim(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    citation_bytes = len(CITATION_SOURCE.encode())

    bundle = assemble_candidate_evidence(
        conn,
        root,
        [_assertion_candidate(conn, assertion_id)],
        budget_bytes=citation_bytes,
    )

    evidence = bundle.results[0].assertion
    assert evidence.citations[0].source is not None
    assert evidence.subject is None
    assert evidence.subject_reason == "budget"
    assert bundle.used_bytes == citation_bytes


def test_later_candidates_are_omitted_deterministically_when_the_budget_runs_out(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)
    acquire = _source_candidate(conn, "leases.acquire")
    caller = _source_candidate(conn, "leases.caller")
    citation_bytes = len(CITATION_SOURCE.encode())

    bundle = assemble_candidate_evidence(
        conn,
        root,
        [_assertion_candidate(conn, assertion_id), acquire, caller],
        budget_bytes=citation_bytes,
    )

    assert [r.candidate_key for r in bundle.results] == [f"assertion:{assertion_id}"]
    assert [(o.key, o.reason) for o in bundle.omitted] == [
        (acquire.key, "budget"),
        (caller.key, "budget"),
    ]


def test_source_candidate_evidence_still_refuses_edited_indexed_source(repo):
    root, conn = repo
    (root / "leases.py").write_text(SOURCE.replace("return 1", "return 2"))

    with pytest.raises(EvidenceError) as error:
        assemble_candidate_evidence(
            conn,
            root,
            [_source_candidate(conn, "leases.acquire")],
            budget_bytes=10_000,
        )

    assert error.value.code == "source_changed"


def test_bundle_json_is_serializable_and_never_exposes_absolute_paths(repo):
    root, conn = repo
    assertion_id = _admit(root, conn)

    bundle = assemble_candidate_evidence(
        conn,
        root,
        [_assertion_candidate(conn, assertion_id), _source_candidate(conn, "leases.caller")],
        budget_bytes=10_000,
    )

    payload = json.dumps(bundle.as_json())
    assert str(root) not in payload
    assert str(root.resolve()) not in payload
    assert "leases.py" in payload
    restored = json.loads(payload)
    assert restored["results"][0]["candidate_type"] == "assertion"
    assert restored["results"][0]["assertion"]["claim"] == (
        "Renews ownership during long work."
    )
    assert restored["budget_bytes"] == 10_000


def test_budget_is_clamped_to_the_server_ceiling(repo):
    root, conn = repo

    bundle = assemble_candidate_evidence(
        conn, root, [], budget_bytes=MAX_EVIDENCE_BYTES + 1
    )

    assert bundle.budget_bytes == MAX_EVIDENCE_BYTES
    assert bundle.results == ()


def test_a_negative_budget_is_rejected(repo):
    root, conn = repo

    with pytest.raises(ValueError, match="budget_bytes must be >= 0"):
        assemble_candidate_evidence(conn, root, [], budget_bytes=-1)
