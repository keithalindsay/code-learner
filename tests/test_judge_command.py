from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codelearner.assertions import store
from codelearner.ingest import index_repo

SRC = '''def clamp(value):
    """Clamp."""
    if value < 0:
        value = 0
    return value
'''


@pytest.fixture()
def indexed(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "m.py").write_text(SRC)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    conn, _ = index_repo(root, index_path=tmp_path / "i.db")
    return root, conn


def _submit(root, conn, claim="clamp forces value to be non-negative", generator="gen/v1"):
    row = conn.execute(
        "SELECT s.id, s.byte_start, s.byte_end, f.path FROM symbols s "
        "JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
        ("m.clamp",),
    ).fetchone()
    return store.write_assertion(
        conn,
        subject_qualname="m.clamp",
        subject_symbol_id=int(row["id"]),
        kind="invariant",
        claim=claim,
        spans=[store.span_for(root, row["path"], row["byte_start"], row["byte_end"])],
        generator=generator,
        repo_root=root,
    )


def test_unjudged_returns_active_claims_without_a_supporting_verdict(indexed):
    root, conn = indexed
    pending = _submit(root, conn)
    judged = _submit(root, conn, claim="clamp returns an int")
    store.record_verdict(conn, judged, "judge/v1", store.VERDICT_SUPPORTED, "ok")

    ids = [a.id for a in store.unjudged_assertions(conn)]
    assert pending in ids
    assert judged not in ids


def test_unjudged_does_not_mutate(indexed):
    root, conn = indexed
    _submit(root, conn)
    before = conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"]
    store.unjudged_assertions(conn, limit=1)
    after = conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"]
    assert before == after
