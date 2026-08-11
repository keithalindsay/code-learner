import subprocess
from pathlib import Path

import pytest

from codelearner.evidence import (
    MAX_EVIDENCE_BYTES,
    MAX_SOURCE_FILE_BYTES,
    EvidenceBundle,
    EvidenceError,
    EvidenceSection,
    assemble_evidence,
    content_bytes,
    number_source,
)
from codelearner.ingest import index_repo
from codelearner.retrieve.lexical import Hit, search_lexical

SOURCE = '''def alpha():
    """A unicode λ docstring."""
    return 1


def beta():
    return alpha()
'''


@pytest.fixture()
def indexed_repo(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sample.py").write_text(SOURCE)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _hit(conn, qualname: str) -> Hit:
    row = conn.execute(
        "SELECT s.id, s.kind, s.qualname, s.line_start, s.line_end, f.path "
        "FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
        (qualname,),
    ).fetchone()
    return Hit(
        symbol_id=row["id"], qualname=row["qualname"], kind=row["kind"],
        path=row["path"], line_start=row["line_start"], line_end=row["line_end"],
        score=1.0, modality="lexical", header="",
    )


def test_assemble_evidence_hydrates_one_complete_indexed_symbol(indexed_repo):
    root, conn = indexed_repo
    hits = search_lexical(conn, "alpha", k=1)

    bundle = assemble_evidence(conn, root, hits, budget_bytes=10_000)

    assert len(bundle.sections) == 1
    section = bundle.sections[0]
    assert section.source == (
        '1 | def alpha():\n2 |     """A unicode λ docstring."""\n3 |     return 1'
    )
    assert "beta" not in section.source
    expected_hash = conn.execute(
        "SELECT content_hash FROM symbols WHERE id = ?", (section.symbol_id,)
    ).fetchone()["content_hash"]
    assert section.content_hash == expected_hash
    assert bundle.used_bytes == len(section.source.encode("utf-8"))


def test_assemble_evidence_omits_an_oversized_first_section(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")

    bundle = assemble_evidence(conn, root, [alpha], budget_bytes=1)

    assert bundle.sections == ()
    assert bundle.used_bytes == 0
    assert bundle.omitted_symbol_ids == (alpha.symbol_id,)


def test_assemble_evidence_keeps_a_later_section_that_fits(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    beta = _hit(conn, "sample.beta")

    bundle = assemble_evidence(conn, root, [alpha, beta], budget_bytes=40)

    assert [section.symbol_id for section in bundle.sections] == [beta.symbol_id]
    assert bundle.omitted_symbol_ids == (alpha.symbol_id,)


def test_assemble_evidence_zero_budget_omits_every_hit(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    beta = _hit(conn, "sample.beta")

    bundle = assemble_evidence(conn, root, [alpha, beta], budget_bytes=0)

    assert bundle.sections == ()
    assert bundle.omitted_symbol_ids == (alpha.symbol_id, beta.symbol_id)


def test_assemble_evidence_rejects_a_negative_budget(indexed_repo):
    root, conn = indexed_repo

    with pytest.raises(ValueError, match="budget_bytes must be >= 0"):
        assemble_evidence(conn, root, [], budget_bytes=-1)


def test_assemble_evidence_clamps_the_budget_to_the_server_ceiling(indexed_repo):
    root, conn = indexed_repo

    bundle = assemble_evidence(conn, root, [], budget_bytes=MAX_EVIDENCE_BYTES + 1)

    assert bundle.budget_bytes == MAX_EVIDENCE_BYTES


def test_assemble_evidence_deduplicates_hits_at_the_first_occurrence(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")

    bundle = assemble_evidence(conn, root, [alpha, alpha], budget_bytes=10_000)

    assert [section.symbol_id for section in bundle.sections] == [alpha.symbol_id]


def test_assemble_evidence_refuses_an_edited_indexed_symbol(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    (root / "sample.py").write_text(SOURCE.replace("return 1", "return 2"))

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "source_changed"
    assert error.value.symbol_id == alpha.symbol_id


def test_assemble_evidence_refuses_a_symlink_replacing_indexed_source(indexed_repo, tmp_path):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    outside = tmp_path / "outside.py"
    outside.write_text(SOURCE)
    indexed_file = root / "sample.py"
    indexed_file.unlink()
    indexed_file.symlink_to(outside)

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "file_not_regular"


def test_assemble_evidence_refuses_an_invalid_span_before_slicing(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    conn.execute("UPDATE symbols SET byte_end = 1000000 WHERE id = ?", (alpha.symbol_id,))

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "invalid_span"


def test_assemble_evidence_refuses_a_non_numeric_indexed_coordinate(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    conn.execute("UPDATE symbols SET byte_end = ? WHERE id = ?", ("not-an-int", alpha.symbol_id))

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "invalid_span"
    assert error.value.symbol_id == alpha.symbol_id


@pytest.mark.parametrize(
    ("select_coordinate", "update_coordinate"),
    [
        ("SELECT byte_start FROM symbols WHERE id = ?", "UPDATE symbols SET byte_start = ? WHERE id = ?"),
        ("SELECT byte_end FROM symbols WHERE id = ?", "UPDATE symbols SET byte_end = ? WHERE id = ?"),
        ("SELECT line_start FROM symbols WHERE id = ?", "UPDATE symbols SET line_start = ? WHERE id = ?"),
        ("SELECT line_end FROM symbols WHERE id = ?", "UPDATE symbols SET line_end = ? WHERE id = ?"),
    ],
)
def test_assemble_evidence_refuses_real_indexed_coordinates(
    indexed_repo,
    select_coordinate,
    update_coordinate,
):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    coordinate = conn.execute(select_coordinate, (alpha.symbol_id,)).fetchone()[0]
    conn.execute(update_coordinate, (float(coordinate) + 0.5, alpha.symbol_id))

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "invalid_span"
    assert error.value.symbol_id == alpha.symbol_id


def test_assemble_evidence_refuses_a_path_outside_the_repository(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    conn.execute(
        "UPDATE files SET path = ? WHERE id = (SELECT file_id FROM symbols WHERE id = ?)",
        ("../../outside.py", alpha.symbol_id),
    )

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "path_escapes_repo"


def test_assemble_evidence_refuses_an_absolute_indexed_path_inside_the_repository(
    indexed_repo,
):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    conn.execute(
        "UPDATE files SET path = ? WHERE id = (SELECT file_id FROM symbols WHERE id = ?)",
        (str((root / "sample.py").resolve()), alpha.symbol_id),
    )

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "path_escapes_repo"
    assert error.value.symbol_id == alpha.symbol_id


def test_assemble_evidence_refuses_a_missing_indexed_source(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    (root / "sample.py").unlink()

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "file_missing"


def test_assemble_evidence_refuses_an_indexed_source_above_the_size_limit(indexed_repo):
    root, conn = indexed_repo
    alpha = _hit(conn, "sample.alpha")
    (root / "sample.py").write_bytes(b"x" * (MAX_SOURCE_FILE_BYTES + 1))

    with pytest.raises(EvidenceError) as error:
        assemble_evidence(conn, root, [alpha], budget_bytes=10_000)

    assert error.value.code == "file_too_large"


def test_number_source_uses_original_one_based_lines_and_preserves_content():
    assert number_source("def f():\n    return 1\n", 7) == (
        "7 | def f():\n8 |     return 1\n"
    )


def test_content_bytes_counts_encoded_bytes_not_characters():
    assert content_bytes("λ\n") == 3


def test_number_source_returns_empty_for_empty_source():
    assert number_source("", 1) == ""


def test_number_source_preserves_final_line_without_newline():
    assert number_source("return 1", 4) == "4 | return 1"


def test_number_source_rejects_zero_line_start():
    import pytest

    with pytest.raises(ValueError, match="line_start must be >= 1"):
        number_source("return 1", 0)


def test_bundle_json_has_stable_explicit_omission_metadata():
    section = EvidenceSection(
        symbol_id=3,
        qualname="pkg.f",
        path="pkg.py",
        line_start=7,
        line_end=8,
        content_hash="abc",
        source="7 | def f():\n8 |     return 1\n",
        content_bytes=36,
    )
    bundle = EvidenceBundle(
        sections=(section,), budget_bytes=100, used_bytes=36,
        sections_omitted=2, omitted_symbol_ids=(8, 13),
    )
    assert bundle.as_json() == {
        "budget_bytes": 100,
        "used_bytes": 36,
        "truncated": True,
        "sections_omitted": 2,
        "omitted_symbol_ids": [8, 13],
        "sections": [{
            "symbol_id": 3,
            "qualname": "pkg.f",
            "path": "pkg.py",
            "line_start": 7,
            "line_end": 8,
            "content_hash": "abc",
            "content_bytes": 36,
            "source": "7 | def f():\n8 |     return 1\n",
        }],
    }
