"""Every repo-relative read refuses a non-regular file instead of blocking on it.

A FIFO is the hazard these tests are about, and it is not the hazard it looks like.
Reading a missing file raises promptly; reading a directory raises promptly; reading a
FIFO does neither. `open()` on a FIFO blocks until some other process opens the write
end, so `read_bytes` never returns, nothing is raised, nothing is logged, and the
caller simply stops. Wave 0 closed that in `server/app.py` and `assertions/store.py`
after `mkfifo victim/pipe.py` wedged the single-threaded MCP server indefinitely. Four
more unguarded reads on repo-relative paths survived that pass; this file pins all
four.

`is_file()` is the whole guard: False for a FIFO, a directory, a socket and a device
node, True for a regular file or a symlink to one. It does NOT close the window
between the test and the read -- a regular file swapped for a FIFO in between still
blocks -- and none of these tests claim it does. That residual needs an fd-based
`os.open(..., O_NONBLOCK)` to close, and is stated at each call site rather than
carried quietly.

**How a regression here behaves.** It hangs. That is the point: the defect under test
is a call that never returns, so a test that merely calls the function would take the
suite down with it. The first attempt at this harness elsewhere used SIGALRM and did
not work -- signals are delivered to the main thread, and the main thread is not the
one that is blocked, so an unfixed run burned the full wall clock. `_completes` runs
the call on a daemon thread and abandons it on timeout, so a regression costs one
loud failed test and one leaked thread rather than the whole run.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from codelearner.assertions import stale, store
from codelearner.chunk import build_chunks
from codelearner.eval.faithfulness import render_evidence
from codelearner.generate import pipeline
from codelearner.ingest import index_repo

T = TypeVar("T")

# Generous by three orders of magnitude. Every guarded call under test is a `stat()`
# and a return; the unguarded one blocks forever. There is no in-between duration to
# be careful about, so this is set for a loaded machine, not for precision.
_TIMEOUT_S = 10.0

SOURCE = 'def acquire(parcel_id):\n    """Take a lease."""\n    return True\n'


def _completes(what: str, call: Callable[[], T]) -> T:
    """Run `call` and return its result, failing loudly if it does not come back.

    The thread is a daemon and is never joined a second time. On timeout it is
    abandoned still blocked inside `open()`, which is deliberate: it cannot be
    cancelled (no writer will ever appear to release it, and there is no portable way
    to interrupt a blocking open), and the alternative to leaking it is hanging the
    session. A leaked daemon thread costs one thread until the interpreter exits and
    is not joined at shutdown; a blocked main thread costs the whole suite.

    Exceptions are re-raised on the calling thread rather than swallowed -- a guard
    that turned a hang into an unexpected traceback is a different bug, and must not
    be reported as a pass.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised below, not swallowed
            box["error"] = exc

    worker = threading.Thread(target=_run, name=f"read-guard:{what}", daemon=True)
    worker.start()
    worker.join(_TIMEOUT_S)
    if worker.is_alive():
        pytest.fail(
            f"{what} did not return within {_TIMEOUT_S}s on a FIFO. This is the "
            "regression, not a slow machine: the read is blocked inside open() "
            "waiting for a writer that will never arrive, and would have hung this "
            "process forever. The thread is abandoned; the rest of the suite is "
            "unaffected.",
            pytrace=False,
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A one-module repo whose `leases.py` has been replaced by a FIFO.

    Written as a real file first and then swapped, because that is the realistic
    shape: the index, the store and the menu all hold paths that were regular files
    when they were recorded. It is also the only shape that reaches `build_chunks`,
    since `iter_python_files` already declines to hand a FIFO to the parser.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(SOURCE)
    return root


def _fifo(root: Path, rel: str = "leases.py") -> None:
    target = root / rel
    if target.exists():
        target.unlink()
    os.mkfifo(target)
    assert not target.is_file(), "test is meaningless if this is a regular file"


# --------------------------------------------------------------------------
# assertions/stale.py -- the serve-path read
# --------------------------------------------------------------------------


def test_stale_read_file_refuses_a_fifo_instead_of_blocking_the_serve_path(repo):
    """`_read_file` is the seam through which staleness verification touches disk, and
    it runs on serve. A cited path that is a FIFO would block it with no exception to
    catch and no log line to find -- on the MCP path that is the entire server, which
    has no second thread to notice. It is treated as unreadable, which is what it is.

    What this test does NOT assert is that the claim's fate is right. None here means
    `file_missing`, which expires the claim permanently and irreversibly, and a FIFO
    is not an absence. That is the audit's "transient read failures permanently expire
    claims" finding and it is WP10's to fix; conflating it with this guard would put
    two different repairs in one place.
    """
    _fifo(repo)
    assert _completes("stale._read_file", lambda: stale._read_file(repo, "leases.py")) is None


def test_stale_read_file_still_reads_an_ordinary_file(repo):
    """The guard must not have been bought by refusing everything."""
    assert stale._read_file(repo, "leases.py") == SOURCE.encode()


# --------------------------------------------------------------------------
# generate/pipeline.py -- the menu read
# --------------------------------------------------------------------------


def test_pipeline_read_source_refuses_a_fifo_instead_of_blocking_the_menu(repo):
    """Near-verbatim the function Wave 0 fixed in `store.py`, and it was missed
    because it lives in another package. A generation run against a repo holding one
    pipe would stop building menus without raising, which from outside is
    indistinguishable from a slow model.

    None is the existing "cannot read these bytes" disposition -- `_item_for` turns it
    into `DROP_UNREADABLE` -- and it is the right one: the guard knows the bytes
    cannot be had, not that anything is corrupt.
    """
    _fifo(repo)
    cache: dict[str, bytes | None] = {}
    got = _completes(
        "pipeline._read_source", lambda: pipeline._read_source(repo, "leases.py", cache)
    )
    assert got is None
    # Cached, so a menu of twenty claims about this file stats it once rather than
    # twenty times -- the negative result has to be as cacheable as the positive one.
    assert cache == {"leases.py": None}


def test_pipeline_read_source_still_reads_an_ordinary_file(repo):
    assert pipeline._read_source(repo, "leases.py", {}) == SOURCE.encode()


# --------------------------------------------------------------------------
# chunk/chunker.py -- the index-time read
# --------------------------------------------------------------------------


def test_build_chunks_refuses_a_fifo_instead_of_blocking_the_index(repo, tmp_path):
    """Chunking reads each indexed file once, from a path list the database supplies.
    A file that was regular at ingest and is a pipe at chunk time -- a re-chunk against
    a moved working tree, which is exactly what `build_chunks` is called directly for
    -- would hang the build with no traceback.

    Empty bytes, matching what an unreadable file already gets: every symbol whose
    chunk needed those bytes collapses to its own header and is dropped by the
    existing empty-chunk rule rather than put in the retrieval set as a guaranteed
    false positive. The module chunk survives, and correctly -- "defines: acquire" is
    assembled from the index, not from disk, so it is not evidence the read happened.
    """
    conn, _ = index_repo(repo, index_path=tmp_path / "i.db")
    before = conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    assert before > 1
    _fifo(repo)

    stats = _completes("chunker.build_chunks", lambda: build_chunks(conn, repo))

    assert stats.skipped_empty > 0
    assert stats.chunks < before
    texts = [r["text"] for r in conn.execute("SELECT text FROM chunks")]
    assert not any("return True" in t for t in texts), "no file bytes should survive"


# --------------------------------------------------------------------------
# eval/faithfulness.py -- evidence rendering for the judge
# --------------------------------------------------------------------------


def test_render_evidence_refuses_a_fifo_instead_of_blocking_the_judge(repo):
    """One cited pipe would stall an entire adjudication run: no verdict for this
    claim, none for any claim after it, no partial report, and nothing raised to say
    why.

    Labelled rather than skipped. A span the judge cannot see must still appear to it
    as a span it cannot see -- dropping it silently would let a claim be graded on
    whichever of its citations happened to be readable, which is the one way a claim
    resting on nothing scores as supported.
    """
    span = store.EvidenceSpan(
        path="leases.py", line_start=1, line_end=3,
        byte_start=0, byte_end=len(SOURCE), content_hash="0" * 64,
    )
    assertion = store.Assertion(
        id=1, subject_qualname="leases.acquire", subject_symbol_id=None,
        kind="purpose", claim="c", status=store.STATUS_ACTIVE, generator=None,
        confidence=None, created_at="", spans=(span,),
    )
    _fifo(repo)

    rendered = _completes(
        "faithfulness.render_evidence", lambda: render_evidence(repo, assertion)
    )

    assert "could not read this span" in rendered
    assert span.citation in rendered
