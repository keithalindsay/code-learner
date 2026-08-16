from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from codelearner import db
from codelearner.adjudicate import Judgement
from codelearner.assertions import store
from codelearner.cli import main
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


def _judge_args(root: Path, index_path: Path, *, dry_run=False, as_json=False, **overrides):
    fields = {
        "repo": root,
        "index_path": index_path,
        "limit": None,
        "model": None,
        "subject": None,
        "allow_same_family": False,
        "dry_run": dry_run,
        "json": as_json,
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


def test_same_family_judge_is_skipped_by_default(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    # generator family "qwen"; judge name "qwen3.5:9b" -> family "qwen" -> match.
    _submit(root, conn, generator="qwen3-coder:7b")
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))
    monkeypatch.setattr(_FakeJudge, "name", "qwen3.5:9b")

    args = _judge_args(root, tmp_path / "i.db")
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 0


def test_allow_same_family_records_the_verdict(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn, generator="qwen3-coder:7b")
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))
    monkeypatch.setattr(_FakeJudge, "name", "qwen3.5:9b")

    args = _judge_args(root, tmp_path / "i.db", allow_same_family=True)
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 1


def test_dry_run_records_nothing(indexed, monkeypatch, tmp_path):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    args = _judge_args(root, tmp_path / "i.db", dry_run=True)
    commands.cmd_judge(args, factory=None)
    assert conn.execute("SELECT count(*) c FROM verdicts").fetchone()["c"] == 0


def test_json_emits_per_claim_verdicts(indexed, monkeypatch, tmp_path, capsys):
    from codelearner.adjudicate import LABEL_SUPPORTED
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _FakeJudge(LABEL_SUPPORTED))

    args = _judge_args(root, tmp_path / "i.db", as_json=True)
    commands.cmd_judge(args, factory=None)
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["supported"] == 1
    assert payload["results"][0]["verdict"] == "supported"


class _UnavailableJudge:
    """A judge that behaves exactly like `OllamaJudge` when ollama is unreachable --
    the real, most-likely failure `cmd_judge` hits in practice."""

    name = "ollama/unreachable"

    def judge(self, *, claim, evidence, subject):
        from codelearner.adjudicate import JudgeUnavailable

        raise JudgeUnavailable(
            "could not reach the judge at http://localhost:11434 (connection "
            "refused). Start it (`ollama serve`), pull the model, and re-run."
        )


def test_judge_unavailable_is_a_clean_error_not_a_traceback(indexed, monkeypatch, tmp_path, capsys):
    """`OllamaJudge` raises `JudgeUnavailable` -- a bare `RuntimeError` -- the moment
    ollama is unreachable. Left uncaught that would print as a stack trace instead
    of the repo's `codelearner: <message>` line, breaking `main`'s own promise to
    never raise for a predictable failure. Invoked through `main`, the same path a
    real invocation takes, so the assertion covers the translation all the way to
    the process exit code and stderr -- not just that `cmd_judge` raises something."""
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    monkeypatch.setattr(commands, "_build_judge", lambda args: _UnavailableJudge())

    exit_code = main(["judge", str(root), "--index", str(tmp_path / "i.db")])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Traceback" not in captured.err
    assert captured.err.startswith("codelearner: ")
    assert "could not reach the judge" in captured.err


class _EvidenceCapturingJudge:
    """Records the evidence text handed to it, so a test can assert the judge saw the
    real cited bytes rather than an unreadable-span placeholder."""

    name = "capture/judge-x"

    def __init__(self):
        self.seen: list[str] = []

    def judge(self, *, claim, evidence, subject):
        from codelearner.adjudicate import CAUSE_JUDGED, LABEL_SUPPORTED, Judgement

        self.seen.append(evidence)
        return Judgement(label=LABEL_SUPPORTED, reasoning="ok", judge=self.name,
                         cause=CAUSE_JUDGED)


def test_judge_reads_evidence_against_the_bound_root_not_args_repo(indexed, monkeypatch, tmp_path):
    """`cmd_judge` must read cited evidence from the tree the index is BOUND to, not
    from `args.repo` (which defaults to the cwd). Judging an index from any other
    directory used to make every span unreadable -- `render_evidence` returns a
    'could not read this span: not a regular file' placeholder -- and the judge, shown
    no code, refuted every claim. Here `args.repo` points at an empty unrelated
    directory while the index is bound to `root`; the evidence handed to the judge must
    still be the real source."""
    from codelearner.cli import commands

    root, conn = indexed
    _submit(root, conn)
    cap = _EvidenceCapturingJudge()
    monkeypatch.setattr(commands, "_build_judge", lambda args: cap)

    wrong = tmp_path / "unrelated"
    wrong.mkdir()
    args = _judge_args(wrong, tmp_path / "i.db")  # repo=wrong dir, index=the real one
    assert commands.cmd_judge(args, factory=None) == 0

    assert cap.seen, "the judge was never called"
    joined = "\n".join(cap.seen)
    assert "could not read this span" not in joined
    assert "clamp" in joined  # the real cited source reached the judge
