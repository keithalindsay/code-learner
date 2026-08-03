"""VRAM contention: reading the card, asking for it back, and refusing to believe the
answer.

No test here reaches a network or a GPU, and both are enforced rather than trusted.
`_no_network` patches `urlopen` for every test in the file -- the same fixture and the
same reason as `tests/test_faithfulness.py` and `tests/test_generate_llm.py` -- and
`_no_nvidia_smi` patches `subprocess.run`, so a code path that shells out to a driver
this machine may not have fails the suite instead of quietly passing on the one
workstation with a card in it.

The case that matters most is the one that cannot be produced on demand: ollama
answering an unload request with `done_reason="unload"` and then holding the memory
anyway. `FakeOllama` reproduces it exactly -- `unload` on the way out, the model still
in `/api/ps` on the way back -- because that is what the real thing did, twice, for
minutes. Every other test in this file exists to make sure that one cannot be reported
as a success.

Every test names a rule that would otherwise fail silently, and each was checked by
deleting the behaviour it names and confirming the test went red.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request

import pytest

from codelearner import gpu
from codelearner.gpu import (
    OUTCOME_FREED,
    OUTCOME_IN_USE,
    OUTCOME_NO_OLLAMA,
    OUTCOME_NOT_FREED,
    OUTCOME_NOTHING_LOADED,
    REASON_STILL_LISTED,
    REASON_VRAM_HELD,
    CpuFallbackRefused,
    GpuMemory,
    GpuState,
    LoadedModel,
    OllamaControl,
    OllamaUnreachable,
    advice,
    find_runners,
    format_release,
    format_state,
    nvidia_smi_compute_apps,
    nvidia_smi_memory,
    read_state,
    release,
    warn_if_contended,
)

GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024

# The measurement that started this module: qwen3:14b resident on a 10GB card.
QWEN_VRAM = 9_756_000_000


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach ollama. Enforced, not assumed.

    A module whose entire job is talking to a local daemon is the easiest one in the
    repo to make machine-dependent: one un-faked control object and the suite passes
    on the workstation with ollama running and hangs on every other machine.
    """

    def _refuse(*args, **kwargs):
        raise urllib.error.URLError("tests must not reach ollama")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    """No test may shell out to a real driver either.

    The default is "the binary is not there", which is also the state of most CI
    machines and of the laptop this has to keep working on. Tests that want a driver
    install their own fake over the top.
    """

    def _absent(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _absent)


def _smi(monkeypatch, outputs: dict[str, str], returncode: int = 0):
    """Fake `nvidia-smi`, keyed by the `--query-...` argument it was given.

    Faked at `subprocess.run` rather than at `nvidia_smi_memory` on purpose: the
    parsing of the CSV -- including the lines that cannot be parsed -- is behaviour
    this module owns, and a fake installed above it would test nothing but the fake.
    """
    seen: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = ""

    def _run(argv, **kwargs):
        seen.append(list(argv))
        key = next((a for a in argv if a.startswith("--query-")), "")
        for needle, out in outputs.items():
            if needle in key:
                return Completed(out)
        return Completed("")

    monkeypatch.setattr(subprocess, "run", _run)
    return seen


class FakeOllama:
    """A scripted ollama: a queue of `/api/ps` answers and a recorded unload.

    Modelled on `tests/test_generate_llm.py`'s fake of the same name and patched in
    the same way -- over `_get`/`_post` on an instance -- so a test can see exactly
    what a real daemon would have received.

    `ps_sequence` is a LIST of answers rather than one, because the whole subject of
    this module is what `/api/ps` says over time. A fake that could only hold one
    answer could not express "still there after thirty seconds", which is the failure
    being tested.
    """

    def __init__(
        self,
        ps_sequence: list[list[LoadedModel]] | None = None,
        *,
        done_reason: str = "unload",
        unreachable: bool = False,
        unload_error: str | None = None,
    ) -> None:
        self.ps_sequence = ps_sequence if ps_sequence is not None else [[]]
        self.done_reason = done_reason
        self.unreachable = unreachable
        self.unload_error = unload_error
        self.ps_calls = 0
        self.posts: list[tuple[str, dict]] = []

    def get(self, path: str) -> dict:
        assert path == "/api/ps"
        if self.unreachable:
            raise OllamaUnreachable("could not reach ollama at http://fake (refused)")
        index = min(self.ps_calls, len(self.ps_sequence) - 1)
        self.ps_calls += 1
        return {
            "models": [
                {
                    "model": m.name,
                    "name": m.name,
                    "size_vram": m.size_vram_bytes,
                    "expires_at": m.expires_at,
                }
                for m in self.ps_sequence[index]
            ]
        }

    def post(self, path: str, payload: dict) -> dict:
        self.posts.append((path, payload))
        if self.unload_error:
            raise OllamaUnreachable(self.unload_error)
        return {"done_reason": self.done_reason}


def _control(fake: FakeOllama, monkeypatch, host: str = "http://fake") -> OllamaControl:
    control = OllamaControl(host)
    monkeypatch.setattr(control, "_get", fake.get)
    monkeypatch.setattr(control, "_post", fake.post)
    return control


class Clock:
    """A monotonic clock that only moves when someone sleeps.

    The timeout branch is the one this module was written for and it is thirty
    seconds long. A test that waited thirty seconds to assert what happens after
    thirty seconds is a test nobody runs, so time is a parameter of `release()` and
    this is what gets passed.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _released(fake: FakeOllama, monkeypatch, **kwargs):
    """Drive `release()` with a fake clock, usage sampling OFF unless a test asks.

    Off by default here and ON by default in `release()` itself, which is not an
    inconsistency: the usage sample costs a second `/api/ps` read, so leaving it on
    would shift every `ps_sequence` index in this file by one and make each test about
    release mechanics quietly also a test of the sampler. The tests that mean to
    exercise sampling pass `usage_gap_s` explicitly, and
    `test_release_samples_usage_by_default` pins the real default so this convenience
    cannot drift away from it.
    """
    kwargs.setdefault("usage_gap_s", None)
    clock = Clock()
    report = release(
        control=_control(fake, monkeypatch),
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )
    return report, clock


# --------------------------------------------------------------------------
# reading the state: every source is optional and says so when it is absent
# --------------------------------------------------------------------------


def test_no_ollama_is_a_reported_state_and_not_an_exception():
    """`read_state` must be callable without a `try`.

    It is called from an exception message being built and from the top of a long
    run, and in both places a diagnostic that can itself raise is worse than no
    diagnostic. The `urlopen` fixture makes ollama unreachable, so this exercises the
    real transport's failure path rather than a fake's."""
    state = read_state(host="http://127.0.0.1:1", timeout=0.01)
    assert state.ollama_reachable is False
    assert state.models == ()
    assert "could not reach ollama" in (state.ollama_detail or "")


def test_ollama_running_with_nothing_loaded_is_not_the_same_as_ollama_absent(monkeypatch):
    """Reachable-and-empty and unreachable both show zero models, and a reader who
    cannot tell them apart will go looking for a daemon that is running fine."""
    state = read_state(control=_control(FakeOllama([[]]), monkeypatch))
    assert state.ollama_reachable is True
    assert state.models == ()
    assert state.held_bytes == 0
    assert "none. ollama is not holding any VRAM." in format_state(state)


def test_loaded_models_are_read_with_their_vram_and_expiry(monkeypatch):
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM, "2026-08-02T11:00:00Z")]])
    state = read_state(control=_control(fake, monkeypatch))
    assert [m.name for m in state.models] == ["qwen3:14b"]
    assert state.held_bytes == QWEN_VRAM
    assert 9.0 < state.models[0].size_vram_gib < 9.2


def test_a_ps_entry_without_a_size_is_still_reported(monkeypatch):
    """Tolerance in the right direction. A model listed with no `size_vram` is still
    a model listed, and dropping it would hide the exact thing being looked for."""
    fake = FakeOllama()
    monkeypatch.setattr(fake, "get", lambda path: {"models": [{"model": "m", "size_vram": None}]})
    state = read_state(control=_control(fake, monkeypatch))
    assert [m.name for m in state.models] == ["m"]
    assert state.held_bytes == 0


def test_a_ps_body_that_is_not_the_expected_shape_yields_no_models(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(fake, "get", lambda path: {"models": "not a list"})
    assert read_state(control=_control(fake, monkeypatch)).models == ()


def test_missing_nvidia_smi_is_unknown_and_never_zero(monkeypatch):
    """`free_bytes` is None, not 0.

    Zero free bytes is a CLAIM -- it is what a full card looks like. Rendering "we
    could not look" as "there is none left" would turn an absent binary into a
    diagnosis, and the release check would then read a card it never saw as one that
    never gave anything back."""
    state = read_state(control=_control(FakeOllama(), monkeypatch))
    assert state.devices == ()
    assert state.free_bytes is None
    assert "nvidia-smi could not be run" in (state.devices_detail or "")
    assert "unknown" in format_state(state)


def test_nvidia_smi_output_is_parsed_per_device(monkeypatch):
    _smi(monkeypatch, {"gpu=memory": "10240, 9800, 440\n"})
    state = read_state(control=_control(FakeOllama(), monkeypatch))
    assert state.devices == (GpuMemory(index=0, total_mib=10240, used_mib=9800, free_mib=440),)
    assert state.free_bytes == 440 * MIB


def test_unparseable_nvidia_smi_lines_are_skipped_not_guessed(monkeypatch):
    """`[Insufficient Permissions]` and `[N/A]` are both real nvidia-smi output.

    Coercing them to a number would invent a measurement; skipping the line reports
    only the devices that actually answered."""
    _smi(monkeypatch, {"gpu=memory": "[Insufficient Permissions]\n10240, 100, 10140\n"})
    devices = nvidia_smi_memory()
    assert devices is not None
    assert [d.free_mib for d in devices] == [10140]


def test_nvidia_smi_failing_is_none_and_not_an_empty_tuple(monkeypatch):
    """A driver that will not answer and a machine with no devices are different
    facts, and only one of them is a reason to stop trusting the release check."""
    _smi(monkeypatch, {}, returncode=9)
    assert nvidia_smi_memory() is None


def test_compute_apps_are_empty_rather_than_none_when_unreadable(monkeypatch):
    """The one place the distinction does NOT matter, and it is worth saying why:
    compute-apps only ever annotates a runner already found in /proc, so nothing
    branches on it."""
    _smi(monkeypatch, {}, returncode=9)
    assert nvidia_smi_compute_apps() == {}


# --------------------------------------------------------------------------
# /proc: identifying the holder, which is what replaces killing it
# --------------------------------------------------------------------------


def _fake_proc(tmp_path, pid: int, cmdline: str):
    entry = tmp_path / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(cmdline.replace(" ", "\x00").encode())
    return entry


def test_runner_processes_are_identified_by_cmdline(tmp_path):
    _fake_proc(tmp_path, 4242, "/usr/local/bin/ollama runner --model /var/lib/x.gguf")
    _fake_proc(tmp_path, 4243, "/usr/local/bin/ollama serve")
    _fake_proc(tmp_path, 4244, "/usr/bin/python3 -m pytest")
    (tmp_path / "not-a-pid").mkdir()

    runners = find_runners(tmp_path)
    assert [r.pid for r in runners] == [4242]
    assert "runner" in runners[0].cmdline


def test_an_unreadable_proc_is_no_runners_rather_than_a_crash(tmp_path):
    """A miss here costs a line of advice and nothing else -- nothing in the module
    consults the result to decide an outcome."""
    assert find_runners(tmp_path / "does-not-exist") == ()


def test_a_process_that_exits_mid_scan_is_skipped(tmp_path):
    """`/proc/<pid>` disappearing between the listing and the read is normal, not an
    error, and it is the single likeliest thing to happen while scanning for a
    process that is being asked to exit."""
    (tmp_path / "5150").mkdir()  # no cmdline file: the process is already gone
    _fake_proc(tmp_path, 5151, "ollama runner")
    assert [r.pid for r in find_runners(tmp_path)] == [5151]


# --------------------------------------------------------------------------
# release: the ask is not the answer
# --------------------------------------------------------------------------


def test_release_with_no_ollama_is_not_a_failure(monkeypatch):
    """Nothing to free is not a failure to free. A script gating on this is asking
    "is the card clear of ollama", and with no ollama the answer is yes."""
    report, _ = _released(FakeOllama(unreachable=True), monkeypatch)
    assert report.outcome == OUTCOME_NO_OLLAMA
    assert report.ok is True
    assert report.asked == ()


def test_release_with_nothing_loaded_asks_for_nothing(monkeypatch):
    """No models means no unload requests. Posting `keep_alive: 0` at a daemon
    holding nothing is a request that can only do harm to a model loaded a
    millisecond later."""
    fake = FakeOllama([[]])
    report, _ = _released(fake, monkeypatch)
    assert report.outcome == OUTCOME_NOTHING_LOADED
    assert report.ok is True
    assert fake.posts == []


def test_release_asks_every_loaded_model_to_unload_with_keep_alive_zero(monkeypatch):
    fake = FakeOllama([[LoadedModel("a", GIB), LoadedModel("b", GIB)], []])
    report, _ = _released(fake, monkeypatch)
    assert report.asked == ("a", "b")
    assert [p for _, p in fake.posts] == [
        {"model": "a", "keep_alive": 0},
        {"model": "b", "keep_alive": 0},
    ]
    assert all(path == "/api/generate" for path, _ in fake.posts)


def test_a_model_that_goes_away_is_freed(monkeypatch):
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)], []])
    report, _ = _released(fake, monkeypatch)
    assert report.outcome == OUTCOME_FREED
    assert report.ok is True
    assert report.still_loaded == ()
    assert "FREED" in format_release(report)


def test_a_model_still_listed_after_the_timeout_is_NOT_freed(monkeypatch):
    """The observed failure, reproduced exactly.

    Ollama answers the unload request with `done_reason="unload"` and goes on listing
    the model. This is the case the whole module exists for, and a version of
    `release()` that returned after the POST -- or that believed `done_reason` --
    reports it as a success. That is the bug."""
    loaded = [LoadedModel("qwen3:14b", QWEN_VRAM, "2026-08-02T10:00:00Z")]
    fake = FakeOllama([loaded] * 60, done_reason="unload")
    report, clock = _released(fake, monkeypatch, wait_s=30.0, poll_interval=1.0)

    assert report.outcome == OUTCOME_NOT_FREED
    assert report.ok is False
    assert report.reason == REASON_STILL_LISTED
    assert [m.name for m in report.still_loaded] == ["qwen3:14b"]
    # It asked, and ollama said yes. Both facts are on the report, because either one
    # alone is a misleading account of what happened.
    assert report.responses == (("qwen3:14b", "unload"),)
    assert clock.now >= 30.0


def test_release_polls_rather_than_answering_from_the_first_look(monkeypatch):
    """A model that takes a few seconds to actually go is a success, not a failure.

    Without polling the only way to pass the previous test is to fail this one."""
    loaded = [LoadedModel("qwen3:14b", QWEN_VRAM)]
    fake = FakeOllama([loaded, loaded, loaded, []])
    report, clock = _released(fake, monkeypatch, wait_s=30.0, poll_interval=1.0)
    assert report.outcome == OUTCOME_FREED
    assert 0 < clock.now < 30.0


def test_ollama_forgetting_the_model_while_the_vram_stays_is_NOT_freed(monkeypatch):
    """The same lie one layer down.

    `/api/ps` going empty is ollama's opinion. If the driver says the memory never
    came back, the runner is still on the card and the next model load will still OOM
    -- so believing `/api/ps` alone would report exactly the situation this tool is
    for as resolved."""
    _smi(monkeypatch, {"gpu=memory": "10240, 9800, 440\n"})  # free never moves
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)], []])
    report, _ = _released(fake, monkeypatch, wait_s=5.0, poll_interval=1.0)

    assert report.outcome == OUTCOME_NOT_FREED
    assert report.reason == REASON_VRAM_HELD
    assert report.recovered_bytes == 0


def test_vram_coming_back_confirms_the_release(monkeypatch):
    """Both signals agreeing is the only thing that counts as freed."""
    frames = iter(["10240, 9800, 440\n", "10240, 300, 9940\n"])
    last = {"value": "10240, 9800, 440\n"}

    class Completed:
        returncode = 0

        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    def _run(argv, **kwargs):
        key = next((a for a in argv if a.startswith("--query-")), "")
        if "gpu=memory" not in key:
            return Completed("")
        last["value"] = next(frames, last["value"])
        return Completed(last["value"])

    monkeypatch.setattr(subprocess, "run", _run)
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)], []])
    report, _ = _released(fake, monkeypatch, wait_s=5.0, poll_interval=1.0)

    assert report.outcome == OUTCOME_FREED
    assert report.recovered_bytes == (9940 - 440) * MIB


def test_without_nvidia_smi_the_driver_check_is_skipped_not_assumed(monkeypatch):
    """No driver means `/api/ps` is the only witness available, and the report says
    the vram delta is unknown rather than printing a number it did not measure."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)], []])
    report, _ = _released(fake, monkeypatch)
    assert report.outcome == OUTCOME_FREED
    assert report.recovered_bytes is None
    assert "unknown -- nvidia-smi could not be read" in format_release(report)


def test_an_unload_ollama_will_not_accept_is_recorded_per_model(monkeypatch):
    """Refusing the request and ignoring it are different failures with different
    remedies, so they are not collapsed into one."""
    fake = FakeOllama(
        [[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60, unload_error="ollama refused"
    )
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)
    assert report.outcome == OUTCOME_NOT_FREED
    assert report.asked == ()
    assert report.refused == (("qwen3:14b", "ollama refused"),)
    assert report.reason == gpu.REASON_REFUSED


# --------------------------------------------------------------------------
# what a human is told when it did not work
# --------------------------------------------------------------------------


def test_advice_is_empty_on_success(monkeypatch):
    report, _ = _released(FakeOllama([[LoadedModel("a", GIB)], []]), monkeypatch)
    assert advice(report) == []


def test_advice_names_the_owner_and_the_eperm_when_the_runner_is_not_ours(monkeypatch):
    """The finding the original hand diagnosis took longest to reach, encoded.

    `kill` against another user's process fails with EPERM -- silently, if stderr was
    redirected. Advice that says "kill the runner" without saying whose it is sends
    someone round that loop again."""
    monkeypatch.setattr(
        gpu,
        "find_runners",
        lambda *a, **k: (
            gpu.RunnerProcess(pid=99, uid=999, user="ollama", cmdline="ollama runner"),
        ),
    )
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60)
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)

    text = "\n".join(advice(report))
    assert "pid 99" in text
    assert "ollama" in text and "EPERM" in text
    assert "sudo systemctl restart ollama" in text
    assert "will not kill a process it only inferred" in text
    # It must not tell a user to run a kill that cannot work.
    assert "kill 99" not in text


def test_advice_offers_the_kill_when_the_runner_is_ours(monkeypatch):
    """Where a signal WOULD work, the human is handed the exact command -- and still
    sends it themselves. That is the whole trade: identification, not action."""
    import os

    monkeypatch.setattr(
        gpu,
        "find_runners",
        lambda *a, **k: (
            gpu.RunnerProcess(pid=77, uid=os.getuid(), user="keith", cmdline="ollama runner"),
        ),
    )
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60)
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)
    assert "kill 77" in "\n".join(advice(report))


def test_the_failure_report_prints_the_ask_and_the_truth_as_separate_facts(monkeypatch):
    """A reader who sees only "done_reason=unload" concludes it worked; a reader who
    sees only "still resident" concludes nothing was tried. The report has to carry
    both or it misleads whichever half it drops."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60)
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)
    text = format_release(report)
    assert "done_reason=unload" in text
    assert "NOT FREED" in text
    assert "still resident: qwen3:14b" in text


def test_the_report_is_json_serialisable_for_a_script(monkeypatch):
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60)
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)
    payload = json.loads(json.dumps(report.as_json()))
    assert payload["ok"] is False
    assert payload["outcome"] == OUTCOME_NOT_FREED
    assert payload["advice"]


def test_an_expiry_in_the_past_beside_a_resident_model_is_printed():
    """The smoking gun from the original diagnosis: ollama had already decided the
    model should be gone forty-one seconds earlier and it was still there."""
    from datetime import UTC, datetime, timedelta

    stamp = (datetime.now(UTC) - timedelta(seconds=41)).isoformat()
    model = LoadedModel("qwen3:14b", QWEN_VRAM, stamp)
    assert (model.expiry_note() or "").endswith("ago")

    state = GpuState(ollama_reachable=True, models=(model,))
    assert "expires" in format_state(state) and "ago" in format_state(state)


def test_nanosecond_timestamps_are_parsed_rather_than_dropped():
    """Ollama stamps `expires_at` with nine fractional digits, which
    `fromisoformat` will not take. Left unhandled the field silently disappears from
    every report -- the one field that proves the unload was already due."""
    model = LoadedModel("m", 0, "2030-01-01T00:00:00.123456789+00:00")
    assert model.expiry_note() is not None


def test_an_unparseable_expiry_is_omitted_rather_than_invented():
    assert LoadedModel("m", 0, "whenever").expiry_note() is None


# --------------------------------------------------------------------------
# the pre-flight check: warns, never acts
# --------------------------------------------------------------------------


def test_warn_if_contended_says_nothing_when_ollama_holds_nothing(monkeypatch, caplog):
    monkeypatch.setattr(gpu, "read_state", lambda **k: GpuState(ollama_reachable=True))
    assert warn_if_contended() is None
    assert caplog.records == []


def test_warn_if_contended_names_the_model_and_the_remedy(monkeypatch, caplog):
    monkeypatch.setattr(
        gpu,
        "read_state",
        lambda **k: GpuState(
            ollama_reachable=True, models=(LoadedModel("qwen3:14b", QWEN_VRAM),)
        ),
    )
    with caplog.at_level("WARNING"):
        note = warn_if_contended()
    assert note is not None and "qwen3:14b" in note
    assert "codelearner gpu --free" in caplog.text


def test_the_preflight_check_never_evicts_anything(monkeypatch):
    """It reads and it warns. If it ever posted an unload it would be doing, from a
    library, the thing `index/embed.py` argues at length is not this tool's call."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]])
    monkeypatch.setattr(gpu, "OllamaControl", lambda *a, **k: _control(fake, monkeypatch))
    warn_if_contended()
    assert fake.posts == []


def test_a_diagnostic_that_cannot_run_costs_nothing(monkeypatch):
    """Called from an exception path and from the top of a long run. A failure to
    diagnose must never become a failure of the thing being diagnosed."""

    def _explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gpu, "read_state", _explode)
    assert gpu.contention_note() is None
    assert warn_if_contended() is None


# --------------------------------------------------------------------------
# strict device: the opt-in that turns a warning into a refusal
# --------------------------------------------------------------------------


class FakeSentenceTransformer:
    """Raises a CUDA OOM on cuda and succeeds on cpu, which is the shape of the real
    failure: the fallback is what makes the run slow, not what makes it fail."""

    def __init__(self, model_name, device=None, **kwargs):
        if device == "cuda":
            raise RuntimeError("CUDA error: out of memory")
        self.device = device
        self.max_seq_length = 0

    def get_sentence_embedding_dimension(self):
        return 3


def _install_torch(monkeypatch, cuda: bool):
    """Fake `torch` so the device decision is testable on a machine with no card."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_st(monkeypatch, factory=FakeSentenceTransformer):
    import sys
    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_the_default_still_falls_back_to_cpu_and_only_warns(monkeypatch, caplog):
    """The default must not change. `index/embed.py` argues that evicting another
    process is not this tool's place and that reasoning stands -- strictness is an
    opt-in for measurement runs, not a new default."""
    from codelearner.index.embed import SentenceTransformerEmbedder

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch)
    with caplog.at_level("WARNING"):
        embedder = SentenceTransformerEmbedder("fake/model")
    assert embedder.device == "cpu"
    assert "falling back to CPU" in caplog.text


def test_strict_device_turns_the_oom_fallback_into_an_exception(monkeypatch):
    from codelearner.index.embed import SentenceTransformerEmbedder

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch)
    with pytest.raises(CpuFallbackRefused) as excinfo:
        SentenceTransformerEmbedder("fake/model", strict_device=True)
    assert "out of memory" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None  # the original OOM is preserved


def test_strict_device_refuses_a_machine_with_no_cuda_at_all(monkeypatch):
    """The case a strictness built only around the OOM would miss, and the commonest
    one: an eval launched where `torch.cuda.is_available()` is False never OOMs. It
    just quietly takes all night on CPU."""
    from codelearner.index.embed import SentenceTransformerEmbedder

    _install_torch(monkeypatch, cuda=False)
    _install_st(monkeypatch)
    with pytest.raises(CpuFallbackRefused, match="no CUDA device"):
        SentenceTransformerEmbedder("fake/model", strict_device=True)


def test_strict_device_refuses_an_explicit_cpu_request(monkeypatch):
    """`strict_device=True, device="cpu"` is a contradiction, and resolving it in
    favour of the device would make strictness silently inert."""
    from codelearner.index.embed import SentenceTransformerEmbedder

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch)
    with pytest.raises(CpuFallbackRefused, match="CPU was requested"):
        SentenceTransformerEmbedder("fake/model", device="cpu", strict_device=True)


def test_strict_device_is_satisfied_by_a_gpu(monkeypatch):
    """Strictness must not be a refusal to work. Where cuda holds, the run proceeds
    exactly as before."""
    from codelearner.index.embed import SentenceTransformerEmbedder

    class Loads(FakeSentenceTransformer):
        def __init__(self, model_name, device=None, **kwargs):
            self.device = device
            self.max_seq_length = 0

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch, Loads)
    assert SentenceTransformerEmbedder("fake/model", strict_device=True).device == "cuda"


def test_the_reranker_default_still_degrades_to_cpu(monkeypatch, caplog):
    from codelearner.retrieve.rerank import CrossEncoderReranker

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch)
    with caplog.at_level("WARNING"):
        reranker = CrossEncoderReranker("fake/model", warmup=False)
    assert reranker.device == "cpu"
    assert "falling back to CPU" in caplog.text


def test_strict_device_makes_the_reranker_refuse_too(monkeypatch):
    """One exception type across both models, which is why `CpuFallbackRefused` lives
    in `gpu` rather than being defined twice: a caller wrapping a measurement run
    catches it once."""
    from codelearner.retrieve.rerank import CrossEncoderReranker

    _install_torch(monkeypatch, cuda=True)
    _install_st(monkeypatch)
    with pytest.raises(CpuFallbackRefused):
        CrossEncoderReranker("fake/model", warmup=False, strict_device=True)


def test_load_reranker_still_returns_none_by_default(monkeypatch):
    """Reranking is optional and never fatal. Unchanged."""
    from codelearner.retrieve import rerank

    _install_torch(monkeypatch, cuda=False)

    def _explode(*a, **k):
        raise RuntimeError("no weights")

    monkeypatch.setattr(rerank, "CrossEncoderReranker", _explode)
    assert rerank.load_reranker() is None


def test_load_reranker_propagates_the_refusal_under_strict(monkeypatch):
    """Swallowing it and returning `None` would be the silent quality loss strictness
    exists to make impossible -- a measurement run that reports "reranked" having
    never reranked."""
    from codelearner.retrieve import rerank

    def _refuse(*a, **k):
        raise CpuFallbackRefused("no gpu")

    monkeypatch.setattr(rerank, "CrossEncoderReranker", _refuse)
    with pytest.raises(CpuFallbackRefused):
        rerank.load_reranker(strict_device=True)


def test_strict_does_not_substitute_the_fallback_model(monkeypatch):
    """`bge-reranker-base` is a different, much weaker model, and none of the numbers
    in `rerank.py`'s docstring were measured with it. Quietly swapping it into a
    measurement run is the sin that module already names: a measurement attributed to
    the wrong model is worse than no measurement."""
    from codelearner.retrieve import rerank

    tried: list[str] = []

    def _record(name, **kwargs):
        tried.append(name)
        raise CpuFallbackRefused("no gpu")

    monkeypatch.setattr(rerank, "CrossEncoderReranker", _record)
    with pytest.raises(CpuFallbackRefused):
        rerank.load_reranker(strict_device=True)
    assert tried == [rerank.DEFAULT_MODEL]

    tried.clear()

    def _fail(name, **kwargs):
        tried.append(name)
        raise RuntimeError("no weights")

    monkeypatch.setattr(rerank, "CrossEncoderReranker", _fail)
    assert rerank.load_reranker() is None
    assert tried == [rerank.DEFAULT_MODEL, rerank.FALLBACK_MODEL]


def test_a_driver_that_answers_with_no_devices_is_not_a_driver_that_failed(monkeypatch):
    """Both render as "no VRAM figures", and only one of them leaves open that
    something is holding memory we could not see."""
    _smi(monkeypatch, {"gpu=memory": "\n"})
    state = read_state(control=_control(FakeOllama(), monkeypatch))
    assert state.devices == ()
    assert state.devices_detail == "nvidia-smi reported no devices"


def test_ollama_vanishing_mid_poll_is_not_reported_as_vram_held(monkeypatch):
    """An empty `/api/ps` because nobody answered is not evidence of anything.

    Reporting it as "unloaded but the VRAM stayed" would attribute a measurement to a
    witness that never spoke."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]])

    calls = {"n": 0}
    original = fake.get

    def _then_vanish(path):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OllamaUnreachable("could not reach ollama at http://fake (refused)")
        return original(path)

    monkeypatch.setattr(fake, "get", _then_vanish)
    report, _ = _released(fake, monkeypatch, wait_s=2.0, poll_interval=1.0)
    assert report.outcome == OUTCOME_NOT_FREED
    assert report.reason == gpu.REASON_OLLAMA_VANISHED


# --------------------------------------------------------------------------
# idle vs in-use: the state a single read cannot see
# --------------------------------------------------------------------------
#
# Found by using the tool, not by writing it. `/api/ps` polled three times six seconds
# apart showed `expires_at` advancing by exactly the poll interval -- something was
# CALLING the model -- while the report cheerfully printed "Free it with `codelearner
# gpu --free`". Following that advice would have unloaded a model mid-request and taken
# a running job with it, which is the exact harm the module docstring argues against.
#
# The signal is that ollama RESETS the keep_alive countdown on every request, so two
# reads a moment apart separate idle from in-use. Everything below is about not
# over-claiming from that signal.


def _at(offset_s: float) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) + timedelta(seconds=offset_s)).isoformat()


def _usage_fake(first_expiry: float, second_expiry: float) -> FakeOllama:
    """Two `/api/ps` answers for the same model with the expiries a test dictates."""
    return FakeOllama(
        [
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(first_expiry))],
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(second_expiry))],
        ]
    )


def test_an_advancing_expiry_means_the_model_is_being_called(monkeypatch):
    """The observed case, reproduced: the countdown moved forward between two reads,
    so a request landed in between."""
    state = read_state(
        control=_control(_usage_fake(240, 300), monkeypatch), usage_gap_s=1.5, sleep=lambda s: None
    )
    assert state.usage_sampled is True
    assert [m.usage for m in state.models] == [gpu.USAGE_IN_USE]
    assert state.in_use
    assert state.safe_to_free is False


def test_a_receding_expiry_means_the_model_is_idle(monkeypatch):
    """Nothing called it, so the countdown just ticked down towards its own unload."""
    state = read_state(
        control=_control(_usage_fake(300, 294), monkeypatch), usage_gap_s=1.5, sleep=lambda s: None
    )
    assert [m.usage for m in state.models] == [gpu.USAGE_IDLE]
    assert state.safe_to_free is True


def test_a_single_sample_says_unknown_rather_than_guessing_idle(monkeypatch):
    """The default, and it must not be optimistic.

    One read cannot tell the two apart, and "unknown" is the only true answer. If it
    resolved to idle, every caller that skipped the sample -- including
    `warn_if_contended` on a hot path -- would be handing out permission to unload
    someone else's running job."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM, _at(240))]])
    state = read_state(control=_control(fake, monkeypatch))
    assert state.usage_sampled is False
    assert [m.usage for m in state.models] == [gpu.USAGE_UNKNOWN]
    assert state.safe_to_free is False
    assert fake.ps_calls == 1  # and it cost exactly one read


def test_a_pinned_keep_alive_is_unknown_and_not_idle(monkeypatch):
    """`keep_alive: -1` puts `expires_at` years out, where it does not move under load
    either. Calling that idle would be reporting a clock that was never going to tick
    as evidence that nothing happened."""
    state = read_state(
        control=_control(_usage_fake(86_400, 86_400), monkeypatch),
        usage_gap_s=1.5,
        sleep=lambda s: None,
    )
    assert [m.usage for m in state.models] == [gpu.USAGE_UNKNOWN]


def test_an_unparseable_expiry_is_unknown_usage(monkeypatch):
    fake = FakeOllama(
        [
            [LoadedModel("m", GIB, "whenever")],
            [LoadedModel("m", GIB, "whenever")],
        ]
    )
    state = read_state(control=_control(fake, monkeypatch), usage_gap_s=1.5, sleep=lambda s: None)
    assert [m.usage for m in state.models] == [gpu.USAGE_UNKNOWN]


def test_a_model_that_appeared_between_samples_has_no_baseline(monkeypatch):
    """Loaded after the first read, so there is nothing to compare it against. The
    second sample is what gets reported -- it is the fresher truth about what is
    resident -- and the newcomer is honestly unlabelled."""
    fake = FakeOllama(
        [
            [LoadedModel("a", GIB, _at(300))],
            [LoadedModel("a", GIB, _at(294)), LoadedModel("b", GIB, _at(300))],
        ]
    )
    state = read_state(control=_control(fake, monkeypatch), usage_gap_s=1.5, sleep=lambda s: None)
    assert {m.name: m.usage for m in state.models} == {
        "a": gpu.USAGE_IDLE,
        "b": gpu.USAGE_UNKNOWN,
    }


def test_the_usage_sample_costs_exactly_one_extra_read(monkeypatch):
    """It is the only thing in this module that costs real time, which is the whole
    reason it is opt-in for library callers."""
    fake = _usage_fake(300, 294)
    slept: list[float] = []
    read_state(control=_control(fake, monkeypatch), usage_gap_s=1.5, sleep=slept.append)
    assert fake.ps_calls == 2
    assert slept == [1.5]


def test_the_sample_is_skipped_entirely_when_nothing_is_loaded(monkeypatch):
    """No models, nothing to classify, so nobody pays the second and a half."""
    fake = FakeOllama([[]])
    slept: list[float] = []
    read_state(control=_control(fake, monkeypatch), usage_gap_s=1.5, sleep=slept.append)
    assert slept == []


# --------------------------------------------------------------------------
# release refuses a model in use
# --------------------------------------------------------------------------


def test_release_declines_to_unload_a_model_that_is_in_use(monkeypatch):
    """Refusing, not attempting and failing.

    This is the one case where the unload would WORK, which is exactly why it must not
    be made: it succeeds and takes the caller's request with it. Asking anyway and
    reporting the outcome does the harm first and narrates it afterwards."""
    fake = _usage_fake(240, 300)
    report, _ = _released(fake, monkeypatch, usage_gap_s=1.5)

    assert report.outcome == OUTCOME_IN_USE
    assert report.declined is True
    assert report.ok is False
    assert fake.posts == []  # nothing was asked, which is the assertion that matters
    assert "serving requests right now" in (report.reason or "")


def test_force_overrides_the_refusal(monkeypatch):
    """An escape hatch for the caller who knows the job is theirs -- explicit, never
    implicit."""
    fake = FakeOllama(
        [
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(240))],
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(300))],
            [],
        ]
    )
    report, _ = _released(fake, monkeypatch, usage_gap_s=1.5, force=True)
    assert report.outcome == OUTCOME_FREED
    assert [p for _, p in fake.posts] == [{"model": "qwen3:14b", "keep_alive": 0}]


def test_an_idle_model_is_released_without_force(monkeypatch):
    """The check must not become a blanket refusal, or the command stops working."""
    fake = FakeOllama(
        [
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(300))],
            [LoadedModel("qwen3:14b", QWEN_VRAM, _at(294))],
            [],
        ]
    )
    report, _ = _released(fake, monkeypatch, usage_gap_s=1.5)
    assert report.outcome == OUTCOME_FREED


def test_unknown_usage_does_not_block_the_release(monkeypatch):
    """Deliberately asymmetric with in-use.

    Refusing on unknown would make the command useless against every instance with a
    pinned keep_alive or an absent `expires_at`, and would itself be a guess. Only a
    MEASURED in-use stops it; unknown is reported and the human decides."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM, None)], []])
    report, _ = _released(fake, monkeypatch)
    assert report.outcome == OUTCOME_FREED


def test_release_samples_usage_by_default(monkeypatch):
    """Pins the real default, which the helper in this file overrides for convenience.

    Without this, `_released` setting `usage_gap_s=None` could quietly become the
    production default and every refusal test above would still pass."""
    import inspect

    assert inspect.signature(release).parameters["usage_gap_s"].default == gpu.USAGE_SAMPLE_GAP_S
    assert inspect.signature(release).parameters["force"].default is False


def test_declining_is_told_apart_from_failing(monkeypatch):
    """Opposite remedies: this one clears itself when the other job ends, the other
    needs a human with sudo. A caller that cannot tell them apart either waits forever
    or escalates for nothing."""
    declined, _ = _released(_usage_fake(240, 300), monkeypatch, usage_gap_s=1.5)
    stuck, _ = _released(
        FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM)]] * 60), monkeypatch,
        wait_s=2.0, poll_interval=1.0,
    )
    assert (declined.declined, declined.ok) == (True, False)
    assert (stuck.declined, stuck.ok) == (False, False)


def test_the_declined_report_says_nothing_was_asked(monkeypatch):
    report, _ = _released(_usage_fake(240, 300), monkeypatch, usage_gap_s=1.5)
    text = format_release(report)
    assert "DECLINED. Nothing was asked to unload." in text
    assert "--force" in text
    assert "sudo systemctl restart ollama" not in text  # wrong advice for this case


def test_every_usage_verdict_carries_the_sample_caveat(monkeypatch):
    """Idle is a sample, not a lock, and it says so where it prints -- the same rule
    `cli.commands.drift_note` follows for its mtime comparison."""
    idle = read_state(
        control=_control(_usage_fake(300, 294), monkeypatch), usage_gap_s=1.5, sleep=lambda s: None
    )
    busy = read_state(
        control=_control(_usage_fake(240, 300), monkeypatch), usage_gap_s=1.5, sleep=lambda s: None
    )
    assert gpu.USAGE_CAVEAT in " ".join(gpu.next_step(idle))
    assert gpu.USAGE_CAVEAT in " ".join(gpu.next_step(busy))
    assert "not a lock" in gpu.USAGE_CAVEAT


def test_the_next_step_is_not_free_it_when_the_model_is_in_use(monkeypatch):
    """The bug this section exists for. The first version printed one hint at anything
    resident, and at a busy model that hint was an instruction to destroy a running
    job."""
    busy = read_state(
        control=_control(_usage_fake(240, 300), monkeypatch), usage_gap_s=1.5, sleep=lambda s: None
    )
    text = " ".join(gpu.next_step(busy))
    assert "IN USE" in text
    assert "refuses" in text
    assert "Free it with" not in text


def test_an_unsampled_state_is_told_to_sample_before_freeing(monkeypatch):
    """Neither "free it" nor "in use" -- the third answer, which the single-hint
    version could not give."""
    fake = FakeOllama([[LoadedModel("qwen3:14b", QWEN_VRAM, _at(240))]])
    state = read_state(control=_control(fake, monkeypatch))
    text = " ".join(gpu.next_step(state))
    assert "was not checked" in text
    assert "Free it with" not in text


def test_nothing_resident_needs_no_next_step():
    assert gpu.next_step(GpuState(ollama_reachable=True)) == []


def test_warn_if_contended_does_not_pay_for_the_usage_sample(monkeypatch):
    """The hot path stays a single read. It runs ahead of every `index --embed`, and a
    second and a half there is a second and a half of every indexing run."""
    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return GpuState(ollama_reachable=True, models=(LoadedModel("m", GIB),))

    monkeypatch.setattr(gpu, "read_state", _record)
    warn_if_contended()
    assert seen.get("usage_gap_s") is None
