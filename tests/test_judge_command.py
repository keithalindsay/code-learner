from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from codelearner import db
from codelearner.adjudicate import Judgement
from codelearner.assertions import store
from codelearner.ingest import index_repo
from codelearner.retrieve.mixed import search_candidates
from codelearner.retrieve.types import AssertionCandidate

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


class _FakeJudge:
    name = "fake/judge-b"

    def __init__(self, label):
        self._label = label

    def judge(self, *, claim, evidence, subject):
        from codelearner.adjudicate import CAUSE_JUDGED

        return Judgement(label=self._label, reasoning="fake", judge=self.name, cause=CAUSE_JUDGED)


def _judge_args(root: Path, index_path: Path, **overrides):
    fields = {
        "repo": root,
        "index_path": index_path,
        "limit": None,
        "model": None,
        "subject": None,
        "allow_same_family": False,
        "dry_run": False,
        "json": False,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_judge_makes_a_submitted_claim_servable(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    # Before judging: the claim is withheld.
    before = search_candidates(conn, root, "non-negative clamp", k=5)
    assert not [c for c in before.candidates if isinstance(c, AssertionCandidate)]

    args = _judge_args(root, tmp_path / "i.db")
    assert commands.cmd_judge(args, factory=None) == 0

    # A fresh connection, deliberately -- `conn` above already ran a read (the
    # `before` search) and cmd_judge wrote its verdict through a SEPARATE
    # connection it opened, committed, and closed. Re-querying through `conn`
    # would depend on when SQLite happens to refresh its WAL snapshot rather
    # than on cmd_judge having actually committed, so the after-assertion goes
    # through a connection that was never open before the write landed.
    after_conn = db.connect(tmp_path / "i.db")
    try:
        after = search_candidates(after_conn, root, "non-negative clamp", k=5)
    finally:
        after_conn.close()
    assert [c for c in after.candidates if isinstance(c, AssertionCandidate)]
