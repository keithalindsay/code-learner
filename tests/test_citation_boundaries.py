"""Pre-v6 narrowed citations: the claims a carry keeps active and should not.

WP8 widened a decorated symbol's span to start at its outermost `@` and bumped the
schema to v6. `--carry-assertions` carries the tier-2 store across that rebuild, and a
carried claim keeps the span it was written with -- bytes that are unchanged on disk,
so the claim verifies, stays `active`, and is served, carrying the exact fail-open
exposure the widening existed to close. Every test here is about the difference
between that claim and a legitimately narrow one, because the rule is only worth
having if it can tell them apart: expiring an agent's deliberate three-line citation
of a function body would punish the stronger of the two citations.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from codelearner import db
from codelearner.assertions import boundaries, store
from codelearner.cli import main
from codelearner.cli.commands import INDEX_RELPATH
from codelearner.ingest.python_extract import decorated_body_start


class FakeEmbedder:
    """Deterministic stand-in; no test here embeds anything, but `main` wants a factory."""

    @property
    def dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "fake/v1"

    def encode(self, texts):  # pragma: no cover - never reached, nothing embeds
        return [[0.0] for _ in texts]


def fake_factory(model_name: str) -> FakeEmbedder:
    return FakeEmbedder()


# `guarded` is the shape the real defect was found in: a decorated symbol whose
# decorator carries the security-relevant part. `plain` is the control -- an
# undecorated function long enough to cite a genuine sub-range of.
APP_PY = (
    'import functools\n'
    '\n'
    '\n'
    'def require_token():\n'
    '    """The auth dependency the decorator below installs."""\n'
    '    return True\n'
    '\n'
    '\n'
    '@functools.lru_cache(maxsize=None)\n'
    'def guarded(key):\n'
    '    """Look up a key, with the token check applied by the decorator."""\n'
    '    return require_token() and key\n'
    '\n'
    '\n'
    'def plain(key):\n'
    '    """Count the characters in a key."""\n'
    '    total = 0\n'
    '    for ch in key:\n'
    '        total += 1\n'
    '    return total\n'
)


def _mkrepo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(APP_PY)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)  # noqa: S603, S607
    return root


@pytest.fixture()
def indexed(tmp_path, capsys) -> tuple[Path, Path]:
    """A repo with a v6 index, capsys drained so the report under test parses."""
    repo = _mkrepo(tmp_path / "repo")
    assert main(["index", str(repo)], embedder_factory=fake_factory) == 0
    capsys.readouterr()
    return repo, repo / INDEX_RELPATH


def _symbol(index_path: Path, qualname: str) -> sqlite3.Row:
    conn = db.connect(index_path, check_schema=False)
    try:
        row = conn.execute(
            "SELECT f.path AS path, s.byte_start, s.byte_end FROM symbols s "
            "JOIN files f ON f.id = s.file_id WHERE s.qualname = ?",
            (qualname,),
        ).fetchone()
        assert row is not None, f"fixture has no symbol {qualname!r}"
        return row
    finally:
        conn.close()


def _admit(
    index_path: Path,
    repo: Path,
    subject: str,
    claim: str,
    *,
    byte_start: int,
    byte_end: int,
    status: str = store.STATUS_ACTIVE,
) -> int:
    """Admit one claim through the real gate, citing exactly these bytes."""
    conn = db.connect(index_path)
    try:
        span = store.span_for(repo, "app.py", byte_start, byte_end)
        return store.write_assertion(
            conn,
            subject_qualname=subject,
            kind="purpose",
            claim=claim,
            spans=[span],
            repo_root=repo,
            status=status,
        )
    finally:
        conn.close()


def _pre_v6_span(index_path: Path, qualname: str, keyword: str) -> tuple[int, int]:
    """The span pre-v6 code would have written for a decorated symbol.

    Derived by finding the `def`/`class` keyword rather than by hardcoding an offset,
    so an edit to the fixture cannot quietly turn this into a citation of something
    else.
    """
    symbol = _symbol(index_path, qualname)
    inner = APP_PY.encode().index(keyword.encode())
    assert symbol["byte_start"] < inner, "fixture symbol is not decorated"
    return inner, symbol["byte_end"]


def _rows(index_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = db.connect(index_path, check_schema=False)
    try:
        return list(conn.execute(sql, params))
    finally:
        conn.close()


def _git_add(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)  # noqa: S603, S607


def _carry(repo: Path, capsys) -> dict:
    """Rebuild carrying the store, returning the `--json` payload."""
    _git_add(repo)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions", "--json"],
        embedder_factory=fake_factory,
    ) == 0
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# the detection primitive
# ---------------------------------------------------------------------------

def test_decorated_body_start_finds_the_definition_a_naive_prefix_scan_would_miss():
    """The reason this asks tree-sitter instead of reading the prefix text.

    Each case below defeats `lstrip().startswith("@")` in one direction or the other:
    a decorator whose argument list spans lines, a comment between two decorators, an
    `@` inside a leading string, and `async def`, whose definition node starts at
    `async` rather than at `def`. A rule that got any of these wrong would either
    expire a good claim or -- worse -- leave a narrowed one active."""
    cases = {
        "multi-line decorator arguments": (
            '@retry(\n    attempts=3,\n    backoff=2,\n)\ndef fetch():\n    return 1\n',
            'def fetch():',
        ),
        "a comment between two decorators": (
            '@outer\n# why the order matters\n@inner\ndef fetch():\n    return 1\n',
            'def fetch():',
        ),
        "an at-sign inside the docstring": (
            '@guard\ndef fetch():\n    """Mail bugs to a@b.example."""\n    return 1\n',
            'def fetch():',
        ),
        "async def": (
            '@guard\nasync def fetch():\n    return 1\n',
            'async def fetch():',
        ),
        "a decorated class": (
            '@dataclass\nclass Box:\n    x: int = 0\n',
            'class Box:',
        ),
    }
    for label, (text, keyword) in cases.items():
        source = text.encode()
        found = decorated_body_start(source, 0, len(source.rstrip(b"\n")))
        assert found == source.index(keyword.encode()), label


def test_decorated_body_start_says_none_for_anything_that_is_not_a_decorated_definition():
    """None is the answer that keeps the boundary test exact rather than suggestive.

    An undecorated function, a module, and a range that merely OVERLAPS a decorated
    definition all have to be indistinguishable from each other here: each one means
    "these bytes are not a symbol with decorators in front of them", and a caller that
    got an offset for any of them would expire a citation on no evidence."""
    source = b'@guard\ndef fetch():\n    return 1\n\n\ndef plain():\n    return 2\n'
    plain = source.index(b'def plain():')
    assert decorated_body_start(source, plain, len(source.rstrip(b"\n"))) is None
    assert decorated_body_start(source, 0, len(source)) is None, "the whole module"
    assert decorated_body_start(source, 1, source.index(b'return 1') + 8) is None


# ---------------------------------------------------------------------------
# the carry path
# ---------------------------------------------------------------------------

def test_a_carried_claim_citing_a_decorated_symbol_from_its_def_comes_back_stale(
    indexed, capsys
):
    """The defect, end to end. The claim's bytes never move, so every hash in this
    package reports it fresh; what is wrong is that those bytes stop one line short of
    the decorator, and the claim would go on being served after that decorator was
    deleted."""
    repo, index_path = indexed
    byte_start, byte_end = _pre_v6_span(index_path, "app.guarded", "def guarded(key):")
    admitted = _admit(
        index_path, repo, "app.guarded", "looks a key up",
        byte_start=byte_start, byte_end=byte_end,
    )

    payload = _carry(repo, capsys)

    row = _rows(index_path, "SELECT * FROM assertions WHERE id = ?", (admitted,))[0]
    assert row["status"] == "stale"
    # Kept, not deleted and not rewritten: the claim text and the too-narrow span both
    # survive, because widening the stored span would fabricate a citation no
    # generator ever made.
    assert row["claim"] == "looks a key up"
    span = _rows(
        index_path, "SELECT * FROM evidence_spans WHERE assertion_id = ?", (admitted,)
    )[0]
    assert span["byte_start"] == byte_start

    events = _rows(
        index_path, "SELECT * FROM staleness_log WHERE assertion_id = ?", (admitted,)
    )
    assert [e["reason"] for e in events] == ["decorators_excluded"]
    assert events[0]["span_id"] == span["id"]
    # Not a hash finding, and it must not read as one: two identical hashes here would
    # invite "nothing changed, why did this expire".
    assert events[0]["expected_hash"] is None
    assert events[0]["observed_hash"] is None
    assert payload["tier2"]["narrowed_citations"] == 1
    # Counted apart from the claims whose bytes actually moved, which need a different
    # remedy.
    assert payload["tier2"]["expired_by_rebuild"] == 0


def test_a_sub_range_citation_of_a_function_body_survives_the_carry(indexed, capsys):
    """The control that stops this over-firing, and the reason the rule is not "any
    strict suffix of a symbol".

    An agent citing the last three lines of a function is making a NARROWER and
    therefore stronger citation than one citing the whole function. This span ends
    exactly where `app.plain` ends and starts well inside it, so a suffix rule would
    expire it -- and expiring it would teach a generator to cite as widely as it can,
    which is the opposite of what the evidence gate is for."""
    repo, index_path = indexed
    symbol = _symbol(index_path, "app.plain")
    body_start = APP_PY.encode().index(b"    for ch in key:")
    assert symbol["byte_start"] < body_start < symbol["byte_end"]
    admitted = _admit(
        index_path, repo, "app.plain", "counts the characters",
        byte_start=body_start, byte_end=symbol["byte_end"],
    )

    payload = _carry(repo, capsys)

    row = _rows(index_path, "SELECT * FROM assertions WHERE id = ?", (admitted,))[0]
    assert row["status"] == "active"
    assert _rows(
        index_path, "SELECT * FROM staleness_log WHERE assertion_id = ?", (admitted,)
    ) == []
    assert payload["tier2"]["narrowed_citations"] == 0


def test_an_undecorated_symbols_whole_span_citation_is_untouched(indexed, capsys):
    """The ordinary case, which is nearly all of them. Nothing about this claim's
    boundary changed at v6, and a sweep that moved it would make every rebuild of
    every index expire claims at random."""
    repo, index_path = indexed
    symbol = _symbol(index_path, "app.plain")
    admitted = _admit(
        index_path, repo, "app.plain", "counts the characters",
        byte_start=symbol["byte_start"], byte_end=symbol["byte_end"],
    )

    payload = _carry(repo, capsys)

    row = _rows(index_path, "SELECT * FROM assertions WHERE id = ?", (admitted,))[0]
    assert row["status"] == "active"
    assert payload["tier2"]["narrowed_citations"] == 0
    assert _rows(index_path, "SELECT * FROM staleness_log") == []


def test_a_rejected_claim_with_a_narrowed_citation_keeps_its_status(indexed, capsys):
    """A judge refused this claim on evidence that was correct at the time, and its
    citation being too narrow does not overturn that. Re-expiring something already
    out of `active` also writes a second log row for one failure, which makes the
    table's growth rate -- documented as a real signal about how fast this repo
    invalidates its inferences -- into a count of how many sweeps have run."""
    repo, index_path = indexed
    byte_start, byte_end = _pre_v6_span(index_path, "app.guarded", "def guarded(key):")
    admitted = _admit(
        index_path, repo, "app.guarded", "looks a key up",
        byte_start=byte_start, byte_end=byte_end,
        status=store.STATUS_REJECTED,
    )

    payload = _carry(repo, capsys)

    row = _rows(index_path, "SELECT * FROM assertions WHERE id = ?", (admitted,))[0]
    assert row["status"] == "rejected"
    assert _rows(
        index_path, "SELECT * FROM staleness_log WHERE assertion_id = ?", (admitted,)
    ) == []
    assert payload["tier2"]["narrowed_citations"] == 0


def test_a_second_carry_does_not_re_expire_what_the_first_one_did(indexed, capsys):
    """One expiry, one log row, however many times the sweep runs. `mark_stale`
    enforces this and the sweep is checked against it here, because a rebuild is the
    command an operator repeats."""
    repo, index_path = indexed
    byte_start, byte_end = _pre_v6_span(index_path, "app.guarded", "def guarded(key):")
    admitted = _admit(
        index_path, repo, "app.guarded", "looks a key up",
        byte_start=byte_start, byte_end=byte_end,
    )

    assert _carry(repo, capsys)["tier2"]["narrowed_citations"] == 1
    assert _carry(repo, capsys)["tier2"]["narrowed_citations"] == 0
    assert len(_rows(
        index_path, "SELECT * FROM staleness_log WHERE assertion_id = ?", (admitted,)
    )) == 1


def test_the_carry_summary_names_the_reason_and_says_a_re_index_will_not_help(
    indexed, capsys
):
    """The number is useless without the remedy attached to it. Re-indexing is the
    reflex when a rebuild reports a count, and it is the one action that cannot repair
    this: the bytes are unchanged and the symbol table is already right."""
    repo, index_path = indexed
    byte_start, byte_end = _pre_v6_span(index_path, "app.guarded", "def guarded(key):")
    _admit(
        index_path, repo, "app.guarded", "looks a key up",
        byte_start=byte_start, byte_end=byte_end,
    )

    _git_add(repo)
    assert main(
        ["index", str(repo), "--force", "--carry-assertions"],
        embedder_factory=fake_factory,
    ) == 0
    out = capsys.readouterr().out

    assert store.REASON_DECORATORS_EXCLUDED in out
    assert "decorators" in out
    assert "re-indexing will not repair them" in out
    assert "redrafted" in out


# ---------------------------------------------------------------------------
# the sweep, called directly
# ---------------------------------------------------------------------------

def test_the_sweep_skips_a_file_it_cannot_read_without_touching_any_status(
    indexed, tmp_path
):
    """An unreadable file establishes nothing about where a symbol starts, so "we
    could not look" is not grounds for expiry here any more than it is on the serve
    path. The claim comes back on the next sweep over a healthy filesystem."""
    repo, index_path = indexed
    byte_start, byte_end = _pre_v6_span(index_path, "app.guarded", "def guarded(key):")
    admitted = _admit(
        index_path, repo, "app.guarded", "looks a key up",
        byte_start=byte_start, byte_end=byte_end,
    )

    conn = db.connect(index_path)
    try:
        # A directory where the module was: readable metadata, no bytes.
        (repo / "app.py").unlink()
        (repo / "app.py").mkdir()
        assert boundaries.expire_narrowed_citations(conn, repo) == 0
        assert conn.execute(
            "SELECT status FROM assertions WHERE id = ?", (admitted,)
        ).fetchone()["status"] == "active"
    finally:
        conn.close()


def test_a_carried_staleness_history_is_not_counted_as_this_rebuilds_work(indexed, capsys):
    """REGRESSION. `staleness_log` is one of the carried tables, and the baseline count
    used to be read BEFORE the carried rows landed -- so a store arriving with prior
    expiries had its own history attributed to this rebuild.

    The number it produced was not obviously wrong, which is why it survived: a repo
    that had never gone stale reported 0, and one that had gone stale last week
    reported last week's number under a heading saying this rebuild expired them. An
    operator reading `expired 3` would go looking for three edits that never happened."""
    repo, index_path = indexed
    sym = _symbol(index_path, "app.plain")
    aid = _admit(
        index_path,
        repo,
        "app.plain",
        "counts the characters in a key",
        byte_start=sym["byte_start"],
        byte_end=sym["byte_end"],
    )

    # Manufacture a PRIOR expiry: the claim goes stale before any rebuild happens.
    conn = db.connect(index_path, check_schema=False)
    try:
        store.mark_stale(conn, aid, reason=store.REASON_HASH_MISMATCH)
        assert conn.execute("SELECT count(*) c FROM staleness_log").fetchone()["c"] == 1
    finally:
        conn.close()

    payload = _carry(repo, capsys)

    # This rebuild expired nothing: the one log row it carried was already there.
    assert payload["tier2"]["expired_by_rebuild"] == 0, payload["tier2"]
    assert _rows(index_path, "SELECT * FROM staleness_log") != []
