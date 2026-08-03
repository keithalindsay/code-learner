"""VRAM contention: read who holds the card, ask for it back, and VERIFY.

**The failure this module exists to prevent.** `index/embed.py` and
`retrieve/rerank.py` both catch a CUDA OOM on model load and fall back to CPU with a
warning. That is the right default -- a 10GB card is shared and it is not this tool's
place to evict whoever else is on it -- but it means a measurement run can quietly
become a different run: ten times slower, possibly interrupted, and reported under the
same name as the fast one. A warning in a log that scrolls is not a control.

**What was measured on ollama 0.17.5, and what it says about the design.** Every line
here was checked by hand against a wedged instance before any of it was written:

  * `GET /api/ps` lists loaded models with `size_vram` and `expires_at`. It is
    authoritative about what ollama BELIEVES it holds.
  * `POST /api/generate {"model": M, "keep_alive": 0}` returns
    `{"done_reason": "unload"}` **and does not free VRAM.** The model was still listed
    twelve seconds later with `expires_at` forty-one seconds in the past, and
    `nvidia-smi` free memory had not moved.
  * `ollama stop M`, the supported CLI, behaved the same way: returned cleanly,
    freed nothing.
  * The VRAM is held by a child `ollama runner` process, not by `ollama serve`. Both
    run as user `ollama`. A `kill` from another user fails with EPERM -- *silently*,
    if stderr was redirected, which is how the first diagnosis missed it.

So the single rule this module encodes: **the polite path reports success while
changing nothing.** An unload request is a request. `release()` therefore treats the
API's answer as worthless and polls `/api/ps` and `nvidia-smi` until the memory
actually comes back or a deadline passes, and it distinguishes *freed* from *asked,
and it did not work* -- because the second is the common outcome and reporting it as
the first is the whole bug.

**Both signals are checked, because either one alone can lie.** Ollama can go on
listing a model whose runner is gone, and a runner can go on holding 9 GiB of a model
ollama has forgotten. `/api/ps` says what ollama thinks; `nvidia-smi` says what the
driver knows. Only agreement counts as freed.

**Resident is not the same as in use, and only the first was visible at first.** The
version of this module that shipped the paragraphs above printed "Free it with
`codelearner gpu --free`" at anything ollama held. Polled three times six seconds
apart, a live instance showed `expires_at` advancing by exactly the poll interval --
ollama resets the keep-alive countdown on every request, so something was actively
CALLING that model. Following the advice would have unloaded it mid-request and taken
a running job with it: the same harm as evicting a peer's model, arrived at by
recommending it rather than by doing it.

Two `/api/ps` reads a moment apart separate the two (`classify_usage`), and everything
downstream of them is built to under-claim. Unknown is the default and is never
resolved to idle; `--free` refuses a MEASURED in-use model and needs `--force`; a
`keep_alive` pinned past `USAGE_PINNED_HORIZON_S` is reported unknown rather than idle,
because a countdown that was never going to move is not evidence that nothing happened.
And it is a sample, not a lock -- an idle model can be called a millisecond later,
which `USAGE_CAVEAT` says on the line where the verdict prints rather than here.

**This module never kills anything, and that is a decision rather than an omission.**
See `advice()`. It identifies the runner processes and their owner and hands the human
an exact command; it does not send the signal itself. A tool that infers a pid and
then signals it is strictly worse than one that reports honestly and says what to run:
the inference can be wrong, pids are reused, and a signal cannot be taken back. The
one case where killing would succeed -- ollama running as the invoking user -- is
also the case where the human can run one line themselves.

Nothing here is required for the tool to work. `nvidia-smi` absent, ollama absent,
`/proc` absent: each degrades to "unknown" and says so, because a VRAM report that
guesses is worse than one that admits what it could not see.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess  # noqa: S404 - nvidia-smi is read-only and its path is not user input
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Short by design. Every call here is to localhost and every one of them is on the
# path of a human waiting at a terminal or of an exception message being built. A
# long timeout would turn "ollama is not running" into a hang.
DEFAULT_TIMEOUT_S = 5.0

# How long `release()` waits for VRAM to actually come back. Ollama's own shutdown is
# not instant even when it works -- the runner has to exit and the driver has to
# reclaim -- and the observed failure held for minutes, so a short wait cannot tell
# "slow" from "never". Thirty seconds is long enough that a success is a success and
# short enough to sit through.
RELEASE_WAIT_S = 30.0
RELEASE_POLL_S = 1.0

# Whether a resident model is being CALLED right now, which is a different question
# from whether it is resident and the only one that decides whether freeing it is
# safe. `expires_at` is a countdown that ollama RESETS on every request, so two reads
# a moment apart tell them apart: an expiry that moved forward means something asked
# the model for something in between, an expiry that stayed put means it is idle and
# ticking down towards its own unload.
#
# Found by using the tool rather than by writing it. The first version printed "Free
# it with `codelearner gpu --free`" at a model whose `expires_at` was advancing by
# exactly the poll interval -- following that advice would have killed a running job
# mid-request, which is the precise harm this module's docstring argues against.
USAGE_UNKNOWN = "unknown"
USAGE_IDLE = "idle"
USAGE_IN_USE = "in-use"

# Gap between the two samples. Long enough that a request landing in the window moves
# the expiry visibly, short enough to sit through at a prompt. It is the whole cost of
# the check, which is why no library caller pays it unless it asks.
USAGE_SAMPLE_GAP_S = 1.5

# An expiry further out than this is not a countdown, it is a pin (`keep_alive: -1`,
# or an hours-long value). A pinned model's `expires_at` does not move when it is
# called, so the signal this check relies on is simply absent -- and reporting "idle"
# because a clock that was never going to move did not move is exactly the confident
# wrong answer the whole module exists to avoid. Unknown is the honest label.
USAGE_PINNED_HORIZON_S = 3600.0

# Printed wherever a usage verdict is, and worded the way `cli.commands.drift_note`
# words its own limit: the check is right when it speaks and incomplete when it is
# silent, and it says so on the line rather than in a docstring nobody reading the
# output will open. Idle is a sample, not a lock -- a model can be called a
# millisecond after this printed, and nothing here holds it still.
USAGE_CAVEAT = (
    f"Usage is a {USAGE_SAMPLE_GAP_S:g}s sample, not a lock: an idle model can be "
    "called a millisecond after this was printed."
)

# Fraction of the VRAM ollama reported holding that must come back before a release is
# called freed.
#
# Not 1.0, and the slack is load-bearing in both directions. `size_vram` is ollama's
# own accounting and need not match the driver's byte-for-byte, and another process
# can take the freed memory inside the polling window -- either would make a strict
# comparison cry wolf. Half is loose enough to survive both and tight enough that the
# observed failure, where 9.1 GiB stayed exactly where it was, cannot pass.
VRAM_RECOVERY_FRACTION = 0.5

BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * 1024 * 1024

# release() outcomes. Strings rather than an enum because they are printed, put in
# `--json`, and asserted against in tests; an enum would be three conversions to buy
# nothing.
OUTCOME_NO_OLLAMA = "no-ollama"
OUTCOME_NOTHING_LOADED = "nothing-loaded"
OUTCOME_FREED = "freed"
OUTCOME_NOT_FREED = "not-freed"
# Declined, not attempted and not failed. Something is calling the model; unloading it
# would succeed and take a running job with it.
OUTCOME_IN_USE = "in-use"

# Why a release failed. The distinction matters to the human: the first means the
# request never took, the second means it took and the memory stayed anyway -- which
# is the one that no amount of asking will fix.
REASON_STILL_LISTED = "ollama still lists the model as loaded"
REASON_VRAM_HELD = "ollama unloaded the model but the VRAM did not come back"
REASON_REFUSED = "ollama would not accept the unload request"
REASON_OLLAMA_VANISHED = (
    "ollama stopped answering after the unload request, so what it holds is unknown"
)


class OllamaUnreachable(RuntimeError):
    """Ollama did not answer. Not a failure to free -- there was nothing to free."""


class CpuFallbackRefused(RuntimeError):
    """A caller said CPU was not acceptable for this run, and CPU is what it got.

    Lives here rather than in `index/embed.py` or `retrieve/rerank.py` because both
    raise it and a caller wrapping a whole measurement run needs ONE `except` clause
    to catch either. That is the difference from `_is_oom`, which those two modules
    deliberately duplicate: a four-line predicate can be copied without consequence,
    an exception type cannot -- two of them would be two things to catch and one of
    them would eventually be forgotten.
    """


@dataclass(frozen=True)
class LoadedModel:
    """One entry from `/api/ps`: a model ollama believes is resident."""

    name: str
    size_vram_bytes: int = 0
    expires_at: str | None = None
    # Defaults to unknown, and that default is the point: a single read of `/api/ps`
    # cannot tell idle from in-use, so anything built from one sample says so rather
    # than guessing the safe-looking answer.
    usage: str = USAGE_UNKNOWN

    @property
    def size_vram_gib(self) -> float:
        return self.size_vram_bytes / BYTES_PER_GIB

    @property
    def in_use(self) -> bool:
        """True only when it was MEASURED to be in use. Unknown is not in-use.

        Read the asymmetry deliberately: this drives what gets printed, while
        `release()` refuses on `usage != USAGE_IDLE`. Unknown is safe to print
        neutrally and not safe to act on.
        """
        return self.usage == USAGE_IN_USE

    def expiry_note(self, now: datetime | None = None) -> str | None:
        """"in 4m21s" / "41s ago", or `None` if ollama did not say or said something
        unparseable.

        Worth printing rather than hiding: an `expires_at` in the PAST beside a model
        that is still listed is the exact signature of the failure this module was
        written for. Ollama has already decided the model should be gone and it is
        still there.
        """
        moment = _parse_timestamp(self.expires_at)
        if moment is None:
            return None
        now = now or datetime.now(UTC)
        seconds = (moment - now).total_seconds()
        if seconds >= 0:
            return f"in {_duration(seconds)}"
        return f"{_duration(-seconds)} ago"


@dataclass(frozen=True)
class GpuMemory:
    """One device as `nvidia-smi` reports it, in MiB."""

    index: int
    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def free_bytes(self) -> int:
        return self.free_mib * BYTES_PER_MIB


@dataclass(frozen=True)
class RunnerProcess:
    """An `ollama runner` process, found by reading `/proc` -- never signalled.

    `owned_by_us` is the whole point of carrying `uid`: it is the difference between
    advice a human can act on directly and advice that needs `sudo`, and getting it
    wrong sends someone off to run a `kill` that will fail with EPERM.
    """

    pid: int
    uid: int
    user: str
    cmdline: str
    vram_bytes: int | None = None

    @property
    def owned_by_us(self) -> bool:
        return self.uid == os.getuid()


@dataclass(frozen=True)
class GpuState:
    """What holds the card right now, and what could not be determined.

    Every "could not be determined" carries its own `*_detail` string. A state object
    that renders an absent `nvidia-smi` as zero free bytes would be worse than one
    that renders it as unknown, because zero free bytes is a claim.
    """

    ollama_reachable: bool = False
    ollama_detail: str | None = None
    models: tuple[LoadedModel, ...] = ()
    devices: tuple[GpuMemory, ...] = ()
    devices_detail: str | None = None
    runners: tuple[RunnerProcess, ...] = ()
    host: str = DEFAULT_OLLAMA_HOST
    usage_sampled: bool = False

    @property
    def held_bytes(self) -> int:
        """VRAM ollama claims for its loaded models."""
        return sum(m.size_vram_bytes for m in self.models)

    @property
    def in_use(self) -> tuple[LoadedModel, ...]:
        """Models measured to be serving requests right now."""
        return tuple(m for m in self.models if m.in_use)

    @property
    def safe_to_free(self) -> bool:
        """Every resident model was MEASURED idle.

        Unknown does not qualify. The default is `USAGE_UNKNOWN`, so a state built
        without the second sample is never "safe to free" -- which makes forgetting to
        sample fail towards caution instead of towards the harm.
        """
        return bool(self.models) and all(m.usage == USAGE_IDLE for m in self.models)

    @property
    def free_bytes(self) -> int | None:
        """Free VRAM across every visible device, or `None` when nvidia-smi could not
        be read. `None` and `0` mean opposite things and are kept distinguishable."""
        if not self.devices:
            return None
        return sum(d.free_bytes for d in self.devices)

    def as_json(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "usage_sampled": self.usage_sampled,
            "safe_to_free": self.safe_to_free,
            "ollama_reachable": self.ollama_reachable,
            "ollama_detail": self.ollama_detail,
            "models": [
                {
                    "name": m.name,
                    "size_vram_bytes": m.size_vram_bytes,
                    "size_vram_gib": round(m.size_vram_gib, 2),
                    "expires_at": m.expires_at,
                    "expires": m.expiry_note(),
                    "usage": m.usage,
                }
                for m in self.models
            ],
            "held_bytes": self.held_bytes,
            "devices": [
                {
                    "index": d.index,
                    "total_mib": d.total_mib,
                    "used_mib": d.used_mib,
                    "free_mib": d.free_mib,
                }
                for d in self.devices
            ],
            "devices_detail": self.devices_detail,
            "free_bytes": self.free_bytes,
            "runners": [
                {
                    "pid": r.pid,
                    "user": r.user,
                    "owned_by_us": r.owned_by_us,
                    "vram_bytes": r.vram_bytes,
                }
                for r in self.runners
            ],
        }


@dataclass(frozen=True)
class ReleaseReport:
    """What was asked, what happened, and -- separately -- whether it worked.

    `outcome` is decided by measurement after the fact, never by what the API
    answered. `asked` records the requests and `refused` the ones that errored out,
    so a report can say "ollama accepted every unload and freed nothing", which is
    the sentence the observed failure needs and the one a success/failure boolean
    cannot produce.
    """

    outcome: str
    before: GpuState
    after: GpuState
    asked: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    reason: str | None = None
    waited_s: float = 0.0
    responses: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @property
    def ok(self) -> bool:
        """True when nothing is being held by ollama at the end of this call.

        `no-ollama` and `nothing-loaded` are successes: a script gating on this is
        asking "is the card clear of ollama", and the answer in both cases is yes.

        `in-use` is NOT one of them, and that is the harder call. Nothing was
        attempted and nothing failed -- but the card is not clear, and a measurement
        script that read this as success would start a run onto a full card. The
        question a gate asks is about the card, not about this function's effort.
        """
        return self.outcome not in (OUTCOME_NOT_FREED, OUTCOME_IN_USE)

    @property
    def declined(self) -> bool:
        """Refused on purpose, rather than tried and beaten.

        Kept separate from `ok` because the two failures have opposite remedies: this
        one resolves itself when the other job finishes, and the other one needs a
        human with sudo. A caller that cannot tell them apart will either wait forever
        or escalate for nothing.
        """
        return self.outcome == OUTCOME_IN_USE

    @property
    def still_loaded(self) -> tuple[LoadedModel, ...]:
        return self.after.models

    @property
    def recovered_bytes(self) -> int | None:
        before, after = self.before.free_bytes, self.after.free_bytes
        if before is None or after is None:
            return None
        return after - before

    def as_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "ok": self.ok,
            "reason": self.reason,
            "asked": list(self.asked),
            "refused": [{"model": m, "error": e} for m, e in self.refused],
            "responses": [{"model": m, "done_reason": r} for m, r in self.responses],
            "waited_s": round(self.waited_s, 2),
            "held_before_bytes": self.before.held_bytes,
            "held_after_bytes": self.after.held_bytes,
            "recovered_bytes": self.recovered_bytes,
            "before": self.before.as_json(),
            "after": self.after.as_json(),
            "advice": advice(self),
        }


# ---------------------------------------------------------------------------
# ollama: the read and the ask
# ---------------------------------------------------------------------------


class OllamaControl:
    """Reads `/api/ps` and asks for unloads. Never kills, never waits.

    Transport lives in `_get`/`_post` as methods rather than as free functions for the
    reason `generate.llm.OllamaClaimGenerator` gives: the tests fake the backend by
    patching those attributes on an instance, and a class whose transport cannot be
    swapped per-instance forces every test through `urlopen` and makes each one a test
    of urllib instead of of this module.
    """

    def __init__(
        self, host: str = DEFAULT_OLLAMA_HOST, timeout: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout

    @property
    def host(self) -> str:
        return self._host

    def _request(self, path: str, payload: dict[str, object] | None) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(  # noqa: S310 - fixed http(s) localhost URL
            f"{self._host}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaUnreachable(f"could not reach ollama at {self._host} ({exc})") from exc
        except json.JSONDecodeError as exc:
            raise OllamaUnreachable(
                f"ollama at {self._host} returned a body that is not JSON ({exc})"
            ) from exc
        if not isinstance(body, dict):
            raise OllamaUnreachable(
                f"ollama at {self._host} returned {type(body).__name__}, not an object"
            )
        return body

    def _get(self, path: str) -> dict[str, Any]:
        return self._request(path, None)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        return self._request(path, payload)

    def loaded(self) -> tuple[LoadedModel, ...]:
        """Everything `/api/ps` lists. Raises `OllamaUnreachable` if it cannot ask.

        Deliberately tolerant of the entry shape: a missing `size_vram` becomes 0
        rather than an exception, because a model listed with no size is still a model
        listed, and refusing to report it would hide exactly the thing being looked
        for.
        """
        body = self._get("/api/ps")
        entries = body.get("models")
        if not isinstance(entries, list):
            return ()
        models = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("model") or entry.get("name") or "").strip()
            if not name:
                continue
            models.append(
                LoadedModel(
                    name=name,
                    size_vram_bytes=_as_int(entry.get("size_vram")),
                    expires_at=_as_str(entry.get("expires_at")),
                )
            )
        return tuple(models)

    def ask_unload(self, model: str) -> str:
        """Request that `model` be unloaded now. Returns ollama's `done_reason`.

        The return value is diagnostic ONLY. `"unload"` was observed against a model
        that then stayed resident for minutes, so nothing in this module may branch on
        it -- it is printed so a human can see that ollama said yes, next to the
        evidence that it did not mean it.

        `/api/generate` with `keep_alive: 0` rather than `ollama stop`: same effect,
        measured, and it needs no subprocess and no CLI on PATH.
        """
        body = self._post("/api/generate", {"model": model, "keep_alive": 0})
        return _as_str(body.get("done_reason")) or "(no done_reason)"


# ---------------------------------------------------------------------------
# nvidia-smi and /proc: the ground truth, both optional
# ---------------------------------------------------------------------------


def _nvidia_smi(query: str, timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    """Run one `nvidia-smi` query, or return `None` if it cannot be run.

    Absent binary, non-zero exit, a driver that has fallen over, a timeout: all of
    them mean "the driver could not be read", which is a thing to report rather than
    an error to raise. Nothing in this module requires a GPU to be present.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def nvidia_smi_memory(timeout: float = DEFAULT_TIMEOUT_S) -> tuple[GpuMemory, ...] | None:
    """Per-device total/used/free in MiB, or `None` when nvidia-smi cannot be read.

    `None` rather than an empty tuple for the unreadable case: an empty tuple would be
    the honest answer for a machine with a driver and no devices, and collapsing "no
    card" into "could not look" loses the only distinction a reader cares about.
    """
    out = _nvidia_smi("gpu=memory.total,memory.used,memory.free", timeout)
    if out is None:
        return None
    devices = []
    for index, line in enumerate(out.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total, used, free = (int(p) for p in parts)
        except ValueError:
            # "[Insufficient Permissions]" and "[N/A]" both land here. A line that
            # cannot be parsed is skipped rather than guessed at.
            continue
        devices.append(GpuMemory(index=index, total_mib=total, used_mib=used, free_mib=free))
    return tuple(devices)


def nvidia_smi_compute_apps(timeout: float = DEFAULT_TIMEOUT_S) -> dict[int, int]:
    """`{pid: bytes}` for processes the driver says hold memory. Empty when unknown.

    Empty rather than `None` because this is only ever used to ANNOTATE a runner that
    was already found in `/proc`. Nothing depends on it, so "we could not attribute
    the bytes" and "there were none to attribute" need not be told apart here.
    """
    out = _nvidia_smi("compute-apps=pid,used_memory", timeout)
    if out is None:
        return {}
    apps: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            apps[int(parts[0])] = int(parts[1]) * BYTES_PER_MIB
        except ValueError:
            continue
    return apps


def find_runners(proc_root: Path | str = "/proc") -> tuple[RunnerProcess, ...]:
    """`ollama runner` processes, read out of `/proc`. Read-only, always.

    This is the identification step that `advice()` turns into a command for a human.
    It exists BECAUSE this module will not kill anything: knowing that the memory is
    held by pid 12345 owned by `ollama` while you are `keith` is the difference
    between "restart the service" and "your kill will fail with EPERM", and that was
    the finding the original hand diagnosis took longest to reach.

    Matched on the cmdline containing both "ollama" and "runner", which is what the
    child process is actually called. A miss here costs a line of advice, never
    correctness -- nothing else in this module consults the result.
    """
    root = Path(proc_root)
    found: list[RunnerProcess] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            uid = entry.stat().st_uid
        except OSError:
            # The process exited between listing and reading. Normal, not an error.
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        lowered = cmdline.lower()
        if "ollama" not in lowered or "runner" not in lowered:
            continue
        found.append(
            RunnerProcess(pid=int(entry.name), uid=uid, user=_username(uid), cmdline=cmdline)
        )
    return tuple(found)


def _username(uid: int) -> str:
    try:
        import pwd

        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


# ---------------------------------------------------------------------------
# reading the state
# ---------------------------------------------------------------------------


def classify_usage(
    first: LoadedModel, second: LoadedModel, now: datetime | None = None
) -> str:
    """Idle or in-use, from the same model seen twice.

    Ollama resets `expires_at` on every request, so a later expiry in the second
    sample means a request landed between them. Anything the two samples cannot
    settle -- an unparseable stamp, a missing one, or an expiry pinned so far out that
    it would not move under load anyway -- is `USAGE_UNKNOWN`, never the convenient
    answer.
    """
    before = _parse_timestamp(first.expires_at)
    after = _parse_timestamp(second.expires_at)
    if before is None or after is None:
        return USAGE_UNKNOWN
    now = now or datetime.now(UTC)
    if (after - now).total_seconds() > USAGE_PINNED_HORIZON_S:
        return USAGE_UNKNOWN
    return USAGE_IN_USE if after > before else USAGE_IDLE


def _annotate_usage(
    first: tuple[LoadedModel, ...], second: tuple[LoadedModel, ...]
) -> tuple[LoadedModel, ...]:
    """The second sample, labelled against the first.

    The SECOND is returned rather than the first because it is the fresher truth about
    what is resident; the first is only a baseline for the comparison. A model present
    in the second and absent from the first has no baseline and stays unknown.
    """
    baseline = {m.name: m for m in first}
    out = []
    for model in second:
        earlier = baseline.get(model.name)
        usage = classify_usage(earlier, model) if earlier else USAGE_UNKNOWN
        out.append(replace(model, usage=usage))
    return tuple(out)


def read_state(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = DEFAULT_TIMEOUT_S,
    control: OllamaControl | None = None,
    with_runners: bool = True,
    usage_gap_s: float | None = None,
    sleep: Any = time.sleep,
) -> GpuState:
    """Who holds the card, from every source that will answer.

    Never raises. Ollama down, nvidia-smi missing, `/proc` unreadable -- each becomes
    a field on the result that says so. A caller building an error message or gating a
    long run must be able to call this without a `try`, or it will not be called from
    the places it is most needed.

    `usage_gap_s` opts into the second `/api/ps` sample that separates idle from
    in-use, and it is **off by default because it is the one thing here that costs
    real time** -- a second and a half against milliseconds for everything else. That
    default is chosen for the library, not for the CLI: `warn_if_contended` runs ahead
    of every `index --embed` and must stay cheap, while `codelearner gpu` is a human
    waiting for an answer that is worthless without it, so the command opts in and
    offers `--no-usage-check` to opt back out. Off, every model is labelled
    `USAGE_UNKNOWN`, which is true rather than optimistic.
    """
    control = control or OllamaControl(host, timeout)
    reachable: bool = True
    detail: str | None = None
    models: tuple[LoadedModel, ...] = ()
    sampled = False
    try:
        models = control.loaded()
        if usage_gap_s and models:
            sleep(usage_gap_s)
            models = _annotate_usage(models, control.loaded())
            sampled = True
    except OllamaUnreachable as exc:
        reachable, detail = False, str(exc)

    devices = nvidia_smi_memory(timeout)
    devices_detail = None
    if devices is None:
        devices, devices_detail = (), "nvidia-smi could not be run; VRAM totals unknown"
    elif not devices:
        # A driver that answered and listed nothing. Distinct from the line above and
        # not worth collapsing into it: one says "we could not look", the other says
        # "we looked and there is no card here", and only the first leaves open the
        # possibility that something is holding memory we cannot see.
        devices_detail = "nvidia-smi reported no devices"

    runners: tuple[RunnerProcess, ...] = ()
    if with_runners:
        runners = find_runners()
        apps = nvidia_smi_compute_apps(timeout) if runners else {}
        if apps:
            runners = tuple(
                RunnerProcess(r.pid, r.uid, r.user, r.cmdline, apps.get(r.pid)) for r in runners
            )

    return GpuState(
        ollama_reachable=reachable,
        ollama_detail=detail,
        models=models,
        devices=devices,
        devices_detail=devices_detail,
        runners=runners,
        host=control.host,
        usage_sampled=sampled,
    )


# ---------------------------------------------------------------------------
# releasing, and proving it
# ---------------------------------------------------------------------------


def release(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = DEFAULT_TIMEOUT_S,
    wait_s: float = RELEASE_WAIT_S,
    poll_interval: float = RELEASE_POLL_S,
    control: OllamaControl | None = None,
    force: bool = False,
    usage_gap_s: float | None = USAGE_SAMPLE_GAP_S,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
) -> ReleaseReport:
    """Ask ollama to drop every IDLE model, then poll until the VRAM is back or the
    deadline passes.

    The polling is the entire feature. `ask_unload` returns `done_reason="unload"`
    against a model that stays resident, so a version of this function that returned
    after the POST would report success on the exact case it was written to catch.

    Freed requires BOTH signals to agree: `/api/ps` no longer lists the models, and --
    when nvidia-smi can be read -- at least `VRAM_RECOVERY_FRACTION` of the memory
    ollama claimed has come back. Either alone is forgeable. Where nvidia-smi cannot
    be read the driver check is skipped rather than assumed, which is stated on the
    report rather than left to be inferred from a missing field.

    **It refuses by default when something is using the model, and refusing is not
    the same as trying and failing.** This is the one place where the ask is likely to
    WORK, which is exactly why it must not be made: unloading a model mid-request
    succeeds and takes the caller's job down with it. `index/embed.py` argues that
    evicting another process is not this tool's decision, and a running job is the
    strongest form of that claim, so the default answer is no and `force=True` is how a
    caller who knows whose job it is says otherwise. The alternative -- ask anyway and
    report what happened -- gets the harm done first and narrates it afterwards.

    Unknown usage does NOT block. Where the sample cannot settle the question (a pinned
    `keep_alive`, an absent `expires_at`, `usage_gap_s=None`), refusing would make the
    command useless on every such instance and would be a guess in its own right; the
    report says the usage was not established and the human decides. Only a MEASURED
    in-use stops it.

    `sleep` and `clock` are parameters so the timeout path is testable in
    microseconds. A test that had to wait thirty seconds to assert what happens after
    thirty seconds would not be written, and this is the branch that matters most.
    """
    control = control or OllamaControl(host, timeout)
    before = read_state(
        control=control, timeout=timeout, usage_gap_s=usage_gap_s, sleep=sleep
    )

    if not before.ollama_reachable:
        return ReleaseReport(
            outcome=OUTCOME_NO_OLLAMA, before=before, after=before, reason=before.ollama_detail
        )
    if not before.models:
        return ReleaseReport(outcome=OUTCOME_NOTHING_LOADED, before=before, after=before)

    busy = before.in_use
    if busy and not force:
        return ReleaseReport(
            outcome=OUTCOME_IN_USE,
            before=before,
            after=before,
            reason=_in_use_reason(busy),
        )

    asked: list[str] = []
    refused: list[tuple[str, str]] = []
    responses: list[tuple[str, str]] = []
    for model in before.models:
        try:
            responses.append((model.name, control.ask_unload(model.name)))
            asked.append(model.name)
        except OllamaUnreachable as exc:
            # Ollama answered `/api/ps` and then would not take the unload. Recorded
            # per model rather than aborting: unloading three of four models is a
            # different outcome from unloading none, and the report has to be able to
            # say which.
            refused.append((model.name, str(exc)))

    started = clock()
    after = before
    while True:
        after = read_state(control=control, timeout=timeout)
        if _looks_freed(before, after):
            break
        if clock() - started >= wait_s:
            break
        sleep(min(poll_interval, max(0.0, wait_s - (clock() - started))))

    waited = clock() - started
    if _looks_freed(before, after):
        return ReleaseReport(
            outcome=OUTCOME_FREED,
            before=before,
            after=after,
            asked=tuple(asked),
            refused=tuple(refused),
            waited_s=waited,
            responses=tuple(responses),
        )
    return ReleaseReport(
        outcome=OUTCOME_NOT_FREED,
        before=before,
        after=after,
        asked=tuple(asked),
        refused=tuple(refused),
        reason=_failure_reason(before, after, refused),
        waited_s=waited,
        responses=tuple(responses),
    )


def _looks_freed(before: GpuState, after: GpuState) -> bool:
    """Both signals, or the one that is available."""
    if after.models:
        return False
    if not after.ollama_reachable:
        # Ollama went away mid-poll. It is no longer holding anything through an API
        # we can see, but the runner may well still be resident -- so this only counts
        # as freed if the driver says the memory came back.
        return _vram_recovered(before, after) is True
    return _vram_recovered(before, after) is not False


def _vram_recovered(before: GpuState, after: GpuState) -> bool | None:
    """`True`/`False`/`None` for recovered / not recovered / could not tell."""
    free_before, free_after = before.free_bytes, after.free_bytes
    if free_before is None or free_after is None or before.held_bytes <= 0:
        return None
    return (free_after - free_before) >= before.held_bytes * VRAM_RECOVERY_FRACTION


def _failure_reason(
    before: GpuState, after: GpuState, refused: list[tuple[str, str]]
) -> str:
    if refused and len(refused) == len(before.models):
        return REASON_REFUSED
    if after.models:
        return REASON_STILL_LISTED
    if not after.ollama_reachable:
        # `/api/ps` is empty because nobody answered, which is not the same evidence
        # as ollama answering and listing nothing. Saying "the VRAM did not come back"
        # here would attribute a measurement to a witness that never spoke.
        return REASON_OLLAMA_VANISHED
    return REASON_VRAM_HELD


def _in_use_reason(busy: tuple[LoadedModel, ...]) -> str:
    names = ", ".join(m.name for m in busy)
    return (
        f"{names} {'is' if len(busy) == 1 else 'are'} serving requests right now "
        f"(expires_at advanced between two reads {USAGE_SAMPLE_GAP_S:g}s apart)"
    )


def advice(report: ReleaseReport) -> list[str]:
    """What a human should do next, given what actually happened.

    This is where the decision not to kill anything is paid back. The module knows the
    pid, knows who owns it, and knows whether that is the caller -- so it can hand over
    the one correct command instead of guessing at a signal. Empty when there is
    nothing to advise, so a caller can print it unconditionally.
    """
    if report.declined:
        return [
            "Nothing was unloaded. Something is calling this model, and unloading it "
            "would succeed -- taking that caller's request down with it.",
            "Wait for it to finish (ollama unloads it on its own when the keep_alive "
            "countdown runs out), or re-run with --force if the caller is yours.",
            USAGE_CAVEAT,
        ]
    if report.ok:
        return []

    lines = [
        "The VRAM is held by a child `ollama runner` process, not by `ollama serve`, "
        "and an unload request cannot make it exit.",
    ]
    ours = [r for r in report.after.runners if r.owned_by_us]
    theirs = [r for r in report.after.runners if not r.owned_by_us]

    if ours:
        pids = " ".join(str(r.pid) for r in ours)
        lines.append(f"You own the runner. End it yourself:  kill {pids}")
    if theirs:
        owners = ", ".join(sorted({r.user for r in theirs}))
        pids = ", ".join(str(r.pid) for r in theirs)
        lines.append(
            f"The runner (pid {pids}) belongs to {owners}, not to you -- `kill` will "
            f"fail with EPERM, silently if you redirect stderr."
        )
    if theirs or not report.after.runners:
        lines.append("Restart the service:  sudo systemctl restart ollama")
    if not report.after.runners:
        lines.append(
            "No runner process was visible from /proc, so the holder could not be "
            "identified -- check `nvidia-smi` for what else is on the card."
        )
    lines.append(
        "This tool will not kill a process it only inferred, and does not ask for "
        "privileges it was not given. Run one of the above and re-check with "
        "`codelearner gpu`."
    )
    return lines


# ---------------------------------------------------------------------------
# the pre-flight check
# ---------------------------------------------------------------------------


def contention_note(
    *, host: str = DEFAULT_OLLAMA_HOST, timeout: float = 2.0, min_free_bytes: int = 0
) -> str | None:
    """One sentence naming what holds the card, or `None` if nothing does.

    Best-effort to the point of paranoia: it is called from an exception path and from
    the top of a long run, and in both places a failure to diagnose must cost nothing.
    Every exception is swallowed, deliberately and with the reason stated -- a
    diagnostic that can itself fail the thing it is diagnosing is a liability.

    `min_free_bytes` lets a caller say how much it needs; 0 means "report any ollama
    model that is resident, and let the human decide".
    """
    try:
        state = read_state(host=host, timeout=timeout, with_runners=False)
    except Exception:  # noqa: BLE001 - a diagnostic must never break its caller
        return None
    if not state.ollama_reachable or not state.models:
        return None
    free = state.free_bytes
    if min_free_bytes and free is not None and free >= min_free_bytes:
        return None
    names = ", ".join(f"{m.name} ({m.size_vram_gib:.1f} GiB)" for m in state.models)
    free_note = f", {free / BYTES_PER_GIB:.1f} GiB free" if free is not None else ""
    return f"ollama is holding {names}{free_note}"


def refusal_message(*, what: str, cause: str, host: str = DEFAULT_OLLAMA_HOST) -> str:
    """The body of a `CpuFallbackRefused`, with a live diagnosis attached if one can
    be had.

    Shared by the embedder and the reranker so the two refusals read the same and both
    end with the same next step. The diagnosis is best-effort by construction -- an
    exception message that could itself raise, or hang, would be worse than a vaguer
    one.
    """
    note = contention_note(host=host)
    tail = f" {note} -- free it with `codelearner gpu --free`." if note else ""
    return (
        f"{what} would run on CPU ({cause}) and this run set strict_device=True. "
        f"CPU embedding is roughly ten times slower, so what came back would be a "
        f"different run from the one asked for, under the same name.{tail}"
    )


def warn_if_contended(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = 2.0,
    min_free_bytes: int = 0,
    log: logging.Logger | None = None,
) -> str | None:
    """Log a warning before a long GPU run if ollama is sitting on the card.

    **Warn, never evict.** `index/embed.py` argues that it is not this tool's place to
    take VRAM from another process, and that argument is not weakened by the run being
    long. What IS worth doing is saying so BEFORE the sixty seconds of model loading
    that ends in a CPU fallback, and naming the command that fixes it -- the existing
    warning fires after the cost has already been paid, from inside a library, into a
    log a user is not necessarily watching.
    """
    note = contention_note(host=host, timeout=timeout, min_free_bytes=min_free_bytes)
    if note is None:
        return None
    (log or logger).warning(
        "%s. This run may fall back to CPU and take roughly ten times as long. "
        "Free the card with `codelearner gpu --free` and re-run.",
        note,
    )
    return note


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def format_state(state: GpuState) -> str:
    """The `codelearner gpu` report, as lines."""
    out = []
    if state.ollama_reachable:
        out.append(f"ollama     {state.host}  reachable")
    else:
        out.append(f"ollama     {state.host}  NOT REACHABLE")
        out.append(f"           {state.ollama_detail}")

    if state.devices:
        for device in state.devices:
            out.append(
                f"gpu {device.index}      {device.total_mib:,} MiB total, "
                f"{device.used_mib:,} used, {device.free_mib:,} free"
            )
    else:
        out.append(f"gpu        unknown -- {state.devices_detail}")

    out.append("")
    out.append("models resident in ollama")
    if state.models:
        for model in state.models:
            expiry = model.expiry_note()
            suffix = f"  expires {expiry}" if expiry else ""
            label = {
                USAGE_IN_USE: "IN USE",
                USAGE_IDLE: "idle",
            }.get(model.usage, "usage unknown")
            out.append(
                f"  {model.name:<28} {model.size_vram_gib:6.2f} GiB  "
                f"{label:<14}{suffix}"
            )
        out.append(f"  {'held':<28} {state.held_bytes / BYTES_PER_GIB:6.2f} GiB")
    elif state.ollama_reachable:
        out.append("  none. ollama is not holding any VRAM.")
    else:
        out.append("  unknown. ollama did not answer, so it is holding nothing we can see.")

    if state.runners:
        out.append("")
        out.append("runner processes on the card")
        for runner in state.runners:
            vram = f"{runner.vram_bytes / BYTES_PER_GIB:.2f} GiB" if runner.vram_bytes else "?"
            owner = "you" if runner.owned_by_us else f"user {runner.user}, not you"
            out.append(f"  pid {runner.pid:<8} {vram:>10}   {owner}")
    return "\n".join(out)


def next_step(state: GpuState) -> list[str]:
    """What to do about what `format_state` just printed. Empty when there is nothing
    to do.

    Four different answers where the first version of this module had one. It printed
    "Free it with `codelearner gpu --free`" at anything resident, which is wrong advice
    at a model that is mid-request and wrong advice at a model whose usage was never
    established -- and advice that is confidently wrong is worse than no advice,
    because it gets followed.
    """
    if not state.models:
        return []
    busy = state.in_use
    if busy:
        names = ", ".join(m.name for m in busy)
        return [
            f"{names} {'is' if len(busy) == 1 else 'are'} IN USE right now -- freeing "
            "would interrupt whatever is calling it.",
            "`codelearner gpu --free` refuses while that is true; --force overrides it.",
            USAGE_CAVEAT,
        ]
    if state.safe_to_free:
        return ["Idle. Free it with `codelearner gpu --free`.", USAGE_CAVEAT]
    if not state.usage_sampled:
        return [
            "Whether anything is CALLING these models was not checked. Re-run without "
            "--no-usage-check before freeing -- unloading a model mid-request succeeds "
            "and takes the caller's job with it."
        ]
    return [
        "Whether these models are in use could not be established -- no usable "
        "`expires_at`, or a `keep_alive` pinned far enough out that the countdown "
        "carries no signal. `--free` will proceed anyway; check who is on the card "
        "first."
    ]


def format_release(report: ReleaseReport) -> str:
    """The `--free` report. Says what was asked and, separately, what is true now.

    Those two are printed as separate facts on purpose. Against a wedged instance the
    first says `unload` and the second says the model is still there, and a reader who
    sees only one of them draws the wrong conclusion either way round.
    """
    out = []
    if report.outcome == OUTCOME_NO_OLLAMA:
        out.append(f"ollama is not reachable at {report.before.host}; nothing to release.")
        out.append(f"  {report.reason}")
        return "\n".join(out)
    if report.outcome == OUTCOME_NOTHING_LOADED:
        out.append("ollama is holding no models; nothing to release.")
        return "\n".join(out)
    if report.outcome == OUTCOME_IN_USE:
        held = report.before.held_bytes / BYTES_PER_GIB
        out.append(f"held          {', '.join(m.name for m in report.before.models)} -- {held:.2f} GiB")
        out.append(f"usage         {report.reason}")
        out.append("")
        out.append("DECLINED. Nothing was asked to unload.")
        for line in advice(report):
            out.append(f"  {line}")
        return "\n".join(out)

    held = report.before.held_bytes / BYTES_PER_GIB
    names = ", ".join(m.name for m in report.before.models)
    out.append(f"held before   {names} -- {held:.2f} GiB")
    for model, done_reason in report.responses:
        out.append(f"asked         {model}  keep_alive=0 -> done_reason={done_reason}")
    for model, error in report.refused:
        out.append(f"refused       {model}  {error}")
    out.append(f"waited        {report.waited_s:.1f}s, polling /api/ps and nvidia-smi")

    recovered = report.recovered_bytes
    if recovered is None:
        out.append("vram          unknown -- nvidia-smi could not be read")
    else:
        out.append(f"vram          {recovered / BYTES_PER_GIB:+.2f} GiB free after the request")

    if report.outcome == OUTCOME_FREED:
        out.append("")
        out.append("FREED. ollama lists nothing and the memory came back.")
        return "\n".join(out)

    still = ", ".join(f"{m.name} ({m.size_vram_gib:.2f} GiB)" for m in report.after.models)
    out.append("")
    out.append(f"NOT FREED. {report.reason}.")
    if still:
        out.append(f"  still resident: {still}")
    for line in advice(report):
        out.append(f"  {line}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

# Ollama stamps `expires_at` with nanosecond precision, which `fromisoformat` will not
# take. Trimming to microseconds is lossy in a way nothing here can notice -- the
# field is used to print "41s ago".
_FRACTION = re.compile(r"(\.\d{6})\d+")


def _parse_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(_FRACTION.sub(r"\1", text))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
