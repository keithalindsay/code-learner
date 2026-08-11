from codelearner.evidence import EvidenceBundle, EvidenceSection, content_bytes, number_source


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
