"""Three-arm agent benchmark: does a code index help an agent, and by how much?

The arms differ in exactly one thing -- which MCP servers `claude -p` is given:

    bare          no MCP servers at all
    codegraph     the codegraph MCP server (structure only, stores no LLM claims)
    code-learner  this project's MCP server (structure PLUS accountable claims)

All three keep the built-in tools, because the question is whether an index HELPS,
not whether it replaces Bash. Removing the built-ins from the bare arm would
manufacture the win.

## What a run records, and why all of it

Every run writes one self-contained JSON to `results/`. The list is chosen so that
nothing downstream ever needs to re-run the agent:

*The full answer text, verbatim.* An efficiency win on a wrong answer is not a win,
and correctness is judged separately against ground truth a sibling authored. If the
answer were not stored, a judging pass would mean paying for the whole matrix again
and comparing against different samples. It is stored uncompressed and untruncated.

*Tools offered vs. tools used.* This is the trap the whole harness is built around. A
run where our MCP server failed to start is byte-for-byte indistinguishable, in the
usage numbers, from a run where the agent simply chose not to call it -- and it looks
like a *cheap* run, so it biases the arm in its own favour. `claude` emits a `system`
event with `subtype == "init"` carrying both the wired-up tool list and an
`mcp_servers` status array; `assert_index_offered` voids a run whose server is absent
or `failed` rather than averaging it in. `index_tool_calls` then says whether the agent
used what it was given.

## Two confounds found by measurement, both live

*MCP tools are denied by default under `--permission-mode dontAsk`.* The first probe
run of the code-learner arm came back with "Calling `index_stats` was denied by the
current permission mode". The agent had the tool, tried to use it, and was refused --
which in the usage numbers is indistinguishable from an index that did not help. Every
arm therefore passes `ALLOWED`, naming both MCP servers explicitly. A benchmark that
skipped this step would have measured its own permission config and published it as an
index result.

*A slow-starting MCP server's tools are DEFERRED, not offered.* `init` fires before
slow servers finish connecting. Measured here: codegraph, a compiled binary, reports
`status: connected` at init and `mcp__codegraph__codegraph_explore` is first-class;
code-learner reports `status: pending` and its five tools land in the deferred pool,
costing the agent two to three extra `ToolSearch` calls before it can call one. About
0.6s of the gap is the reference `mcp` Python SDK's own import. Confirmed causal by
`shim/delay.py`: putting a 1.2s delay in front of codegraph reproduces the deferral
exactly, and on a short task the agent then never reached the tool at all.

That is a real property of the two servers and it is reported. It is not a property of
the two indexes, so `--defer-parity` swaps in `mcp/codegraph-delayed.json` to put both
servers on the same footing, `toolsearch_calls` is recorded per run so the overhead is
subtractable either way, and the report gives both numbers.

*`cache_read_input_tokens`, separately from input tokens.* A warm prompt cache makes
whichever arm ran second look cheaper for reasons that have nothing to do with the
index. Arm order is counterbalanced (see `arm_orders`) and the cache split is recorded
per run so the ordering effect can be measured instead of assumed away.

*Wall-clock, recorded but flagged.* It is the noisiest of the three cost metrics and
codegraph's own README concedes a floor effect on small repos. `analyze.py` reports it
last and with wider intervals.

## Environment control

`claude` inherits `CLAUDE_*` variables from any parent Claude Code session, and a
benchmark launched from inside one is not the same environment as a benchmark launched
from a shell. `SCRUB_ENV` names what is stripped. Measured on this machine: with the
parent's variables inherited, the child session came up with a *different built-in
tool set* than with them stripped, which would have been a silent confound.

`DISALLOWED` pins the built-in tool set identically across all three arms. It is
applied to every arm including bare, so the bare arm cannot be accidentally
handicapped -- `verify_arms` asserts the three offered sets are identical modulo the
MCP tools, which is the one difference the benchmark is measuring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
MCP_DIR = BENCH_DIR / "mcp"
SHIM_DIR = BENCH_DIR / "shim"
TASKS_DIR = BENCH_DIR / "tasks"
RESULTS_DIR = BENCH_DIR / "results"

CLAUDE_BIN = os.environ.get("BENCH_CLAUDE_BIN", "/home/keith/.claude/local/claude")

#: Inherited from a parent Claude Code session and NOT part of the environment a real
#: user's agent runs in. Stripping them is the difference between measuring the index
#: and measuring the harness. See the module docstring.
SCRUB_ENV = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_BRIDGE_SESSION_ID",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "AI_AGENT",
    "ANTHROPIC_MODEL",
)

#: Applied identically to every arm. Write tools are removed because every task is a
#: question and a benchmark must not mutate the repos it measures; `Task` is removed
#: because a sub-agent's tool calls are accounted separately and would make
#: `tool_calls` mean different things in different runs; network tools are removed
#: because an answer fetched from the web is not an answer from the index.
DISALLOWED = (
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "Task",
    "WebFetch",
    "WebSearch",
)

#: Auto-approved so no arm can lose a tool call to a permission prompt it never sees.
#: The MCP entries are server-level (`mcp__<server>`) and are present in EVERY arm's
#: command line including bare, where they match nothing -- keeping the argv identical
#: across arms removes one more thing that could differ. See the module docstring.
ALLOWED = ("Bash", "Read", "ToolSearch", "mcp__codegraph", "mcp__codelearner")

#: Tools Claude Code injects because a connected MCP server DECLARED a capability, not
#: because they are part of the built-in set. They do not carry the `mcp__` prefix, so
#: without this list they would be counted as built-ins and `verify_arms` would report
#: the arms as having different built-in sets.
#:
#: Found by measurement once the code-learner server started connecting in time: the
#: reference `mcp` Python SDK advertises `resources` and `prompts` capabilities whether
#: or not a server implements either, so Claude Code adds these three. codegraph
#: declares `{"tools": {}}` only and gets none of them. code-learner's `resources/list`
#: is EMPTY, so the three tools can do nothing -- they are pure system-prompt weight,
#: and an asymmetry in the arms' favour of nobody. Reported per arm rather than hidden.
MCP_COMPANION_TOOLS = frozenset({
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "ReadMcpResourceDirTool",
})

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT_S = 900

ARMS: dict[str, dict[str, Any]] = {
    "bare": {"mcp_config": MCP_DIR / "bare.json", "servers": []},
    "codegraph": {"mcp_config": MCP_DIR / "codegraph.json", "servers": ["codegraph"]},
    "code-learner": {"mcp_config": MCP_DIR / "code-learner.json", "servers": ["codelearner"]},
}
ARM_NAMES = tuple(ARMS)

#: Used unless `--no-fast-handshake`. Runs the same server behind `shim/fastmcp.py`,
#: which answers the four handshake methods from a captured snapshot and forwards every
#: tool call to the real server untouched.
#:
#: On by DEFAULT because without it the code-learner arm does not measure an index.
#: Claude Code opens with `initialize`, `tools/list`, `prompts/list`, `resources/list`
#: against a server declaring `prompts` and `resources` capabilities -- four serialised
#: round trips behind the reference `mcp` Python SDK's ~600 ms import -- and is still
#: `pending` when `init` fires, so the tools land in the deferred pool. codegraph
#: declares only `{"tools": {}}` and is asked for `tools/list` alone, so it is
#: `connected` and first-class every time. Measured consequence: across 6 pilot runs
#: with the plain config the agent called a code-learner tool ZERO times and the arm
#: scored WORSE than bare. With the relay the server reports `connected` and all five
#: tools are first-class.
#:
#: `--no-fast-handshake` reproduces the out-of-the-box behaviour, which is the honest
#: answer to "what happens if I install this today" and a different question from "does
#: this index help". Both belong in the report.
FAST_CONFIGS = {"code-learner": MCP_DIR / "code-learner-fast.json"}

#: Snapshots the relay serves. `verify` re-derives and diffs them: a stale snapshot
#: would hand the agent a tool list the server no longer has, and that must void the
#: comparison rather than pass silently.
SNAPSHOTS = {"code-learner": SHIM_DIR / "snapshots" / "codelearner.json"}

#: With `--defer-parity`, codegraph is started behind `shim/delay.py` so both index
#: arms are `pending` at init and pay the same deferred-tool overhead.
PARITY_CONFIGS = {"codegraph": MCP_DIR / "codegraph-delayed.json"}

#: Appended identically to every arm by `--nudge-discovery`, including bare, where it
#: finds nothing and costs the same call.
#:
#: Needed because of a measured failure, not a theory. In the pilot the code-learner
#: arm had its server running and its five tools loadable, and the agent never called
#: one: they were in the deferred pool, nothing pointed at them, and it fell back to
#: Bash. Scoring that run as "the index did not help" would be a false negative about
#: the INDEX caused entirely by a startup-latency artifact -- the connection window is
#: under ~420ms, codegraph's compiled server clears it at 116ms and a Python MCP server
#: at 670ms cannot.
#:
#: The wording names no server, no tool and no strategy, so it cannot tell an agent to
#: PREFER the index -- only that a list exists and can be read. `bare` gets the same
#: sentence and the same opportunity to waste a call on it. Off by default: the report
#: gives the out-of-the-box number and the discoverable number side by side, because
#: which one matters depends on whether the reader is asking "is this index useful" or
#: "what happens if I install it today".
DISCOVERY_NUDGE = (
    "Some tools in this session may not be listed in your initial tool set; "
    "additional tools can be discovered with ToolSearch. Check what is available "
    "before deciding how to approach the task."
)


# ---------------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------------


@dataclass
class Task:
    """One question, plus where to ask it. Ground truth rides along untouched.

    The field aliases exist because `tasks/*.json` is authored by a different agent and
    a benchmark that silently drops a task whose author wrote `question` instead of
    `prompt` is worse than one that refuses to start. `ground_truth` is carried into
    every result verbatim and is never read by this module -- judging is a separate
    pass and must not be able to leak into the run.
    """

    id: str
    prompt: str
    repo: Path
    ground_truth: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> Task:
        known = {"id", "task_id", "name", "prompt", "question", "query", "repo",
                 "repo_path", "ground_truth", "answer", "expected"}
        tid = data.get("id") or data.get("task_id") or data.get("name") or source.stem
        prompt = data.get("prompt") or data.get("question") or data.get("query")
        if not prompt:
            raise ValueError(f"{source}: task {tid!r} has no prompt/question/query")
        repo = data.get("repo") or data.get("repo_path")
        if not repo:
            raise ValueError(f"{source}: task {tid!r} has no repo/repo_path")
        repo_path = Path(repo)
        if not repo_path.is_absolute():
            repo_path = Path("/home/keith/projects") / repo
        gt = data.get("ground_truth", data.get("answer", data.get("expected")))
        return cls(
            id=str(tid),
            prompt=str(prompt),
            repo=repo_path,
            ground_truth=gt,
            extra={k: v for k, v in data.items() if k not in known},
        )


def load_tasks(tasks_dir: Path = TASKS_DIR) -> list[Task]:
    """Every task in `tasks/*.json`, accepting either one object or a list per file."""
    tasks: list[Task] = []
    for path in sorted(tasks_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        items = payload if isinstance(payload, list) else payload.get("tasks", [payload])
        for item in items:
            tasks.append(Task.from_dict(item, path))
    seen: dict[str, Path] = {}
    for t in tasks:
        if t.id in seen:
            raise ValueError(f"duplicate task id {t.id!r}")
        seen[t.id] = t.repo
    return tasks


# ---------------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------------


@dataclass
class RunResult:
    """Everything one `claude -p` invocation produced. Serialised as-is to JSON."""

    run_id: str
    task_id: str
    arm: str
    repo: str
    model: str
    rep: int
    #: Position of this arm within its counterbalanced block, 0-based. Recorded so a
    #: cache-warming effect shows up as a position effect rather than an arm effect.
    arm_position: int
    started_at: float
    prompt: str
    ground_truth: Any

    ok: bool = False
    error: str | None = None

    #: THE POINT. Verbatim, untruncated, judged later against `ground_truth`.
    answer: str = ""

    tools_offered: list[str] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    #: Per-server status from the init event: "connected", "pending", "failed", ...
    mcp_status: dict[str, str] = field(default_factory=dict)
    #: The arm's server was wired up at all. False means the run is VOID, not cheap.
    index_offered: bool = False
    #: The arm's tools were in init's tool list rather than the deferred pool. When
    #: this is False the agent must spend `ToolSearch` calls to reach them, which is a
    #: property of server startup time and not of the index. See the module docstring.
    index_first_class: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    n_tool_calls: int = 0
    #: Subset of `n_tool_calls` that went to this arm's MCP server.
    index_tool_calls: int = 0
    #: Calls spent locating deferred tools. Subtract to compare like with like.
    toolsearch_calls: int = 0
    #: An MCP call that came back an error -- a denied permission looks exactly like
    #: an index that did not help, so it is counted rather than left in the noise.
    index_tool_errors: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    wall_s: float = 0.0
    duration_api_ms: int = 0
    num_turns: int = 0
    session_id: str = ""
    stop_reason: str = ""
    transcript_path: str = ""
    exit_code: int | None = None
    #: Which MCP config file this run actually used, and whether parity was on.
    extra_config: dict[str, Any] = field(default_factory=dict)


def resolve_config(arm: str, *, defer_parity: bool = False,
                   fast_handshake: bool = True) -> Path:
    """Which MCP config file an arm actually runs with, given the two switches.

    One function so `run_one`, `probe_server` and `verify_arms` can never disagree
    about what was launched -- a verify that proved a config the runs did not use would
    be worse than no verify at all. The chosen path is also written into every result's
    `extra_config`, so a run record says what it ran rather than what the defaults were
    at the time.
    """
    if defer_parity and arm in PARITY_CONFIGS:
        return PARITY_CONFIGS[arm]
    if fast_handshake and arm in FAST_CONFIGS:
        return FAST_CONFIGS[arm]
    return Path(ARMS[arm]["mcp_config"])


def check_snapshots(repo: Path, fast_handshake: bool = True) -> dict[str, Any]:
    """Re-derive every relay snapshot and report drift, per repo.

    Per repo rather than once, because a snapshot is only trustworthy where it was
    taken: a server whose tool list depended on the working directory would otherwise
    be served a list from somewhere else. Drift is a hard failure in `verify` -- the
    relay would hand the agent tools the server no longer has, and every number
    downstream would be describing a server that does not exist.
    """
    out: dict[str, Any] = {}
    if not fast_handshake:
        return {"skipped": "fast_handshake off; no snapshot is served"}
    for arm, snap in SNAPSHOTS.items():
        if not snap.exists():
            out[arm] = {"ok": False, "error": f"missing snapshot {snap}"}
            continue
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [sys.executable, str(SHIM_DIR / "fastmcp.py"), "--snapshot", str(snap),
             "--check", "--cwd", str(repo)],
            capture_output=True, text=True, timeout=180,
        )
        out[arm] = {
            "ok": proc.returncode == 0,
            "detail": (proc.stdout or proc.stderr).strip()[:500],
        }
    return out


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
    env.update(
        {
            "DO_NOT_TRACK": "1",
            "CODEGRAPH_TELEMETRY": "0",
            "CODEGRAPH_NO_UPDATE_CHECK": "1",
            "NO_COLOR": "1",
        }
    )
    return env


def _mcp_prefixes(arm: str) -> tuple[str, ...]:
    return tuple(f"mcp__{s}__" for s in ARMS[arm]["servers"])


def _parse_stream(lines: list[str], result: RunResult) -> None:
    """Fold the stream-json transcript into `result`.

    Answer text is assembled from assistant text blocks and then overridden by the
    terminal `result` event when it carries one -- the event's `result` field is what
    `--output-format json` would have returned, so the two formats agree, and the
    per-block assembly is the fallback for a truncated stream.
    """
    texts: list[str] = []
    prefixes = _mcp_prefixes(result.arm)
    servers = set(ARMS[result.arm]["servers"])
    mcp_use_ids: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mtype = msg.get("type")

        if mtype == "system" and msg.get("subtype") == "init":
            result.tools_offered = list(msg.get("tools") or [])
            result.mcp_servers = list(msg.get("mcp_servers") or [])
            result.mcp_status = {
                str(s.get("name")): str(s.get("status")) for s in result.mcp_servers
            }
            result.session_id = msg.get("session_id", result.session_id)
            # "Offered" means wired up, NOT necessarily first-class: a `pending` server
            # connects moments later and its tools arrive deferred. Only an absent or
            # failed server voids the run.
            result.index_offered = not servers or all(
                result.mcp_status.get(s, "absent") not in ("absent", "failed", "error")
                for s in servers
            )
            result.index_first_class = not prefixes or any(
                t.startswith(prefixes) for t in result.tools_offered
            )

        elif mtype == "assistant":
            for block in msg.get("message", {}).get("content", []) or []:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    result.tool_calls.append(
                        {"name": name, "input": block.get("input", {})}
                    )
                    result.tool_call_counts[name] = result.tool_call_counts.get(name, 0) + 1
                    if name == "ToolSearch":
                        result.toolsearch_calls += 1
                    if prefixes and name.startswith(prefixes):
                        result.index_tool_calls += 1
                        mcp_use_ids.add(str(block.get("id")))

        elif mtype == "user":
            for block in msg.get("message", {}).get("content", []) or []:
                if (
                    block.get("type") == "tool_result"
                    and block.get("is_error")
                    and str(block.get("tool_use_id")) in mcp_use_ids
                ):
                    result.index_tool_errors += 1

        elif mtype == "result":
            if msg.get("result"):
                texts = [str(msg["result"])]
            usage = msg.get("usage") or {}
            result.input_tokens = usage.get("input_tokens", 0)
            result.output_tokens = usage.get("output_tokens", 0)
            result.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
            result.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
            result.total_cost_usd = float(msg.get("total_cost_usd") or 0.0)
            result.duration_api_ms = int(msg.get("duration_api_ms") or 0)
            result.num_turns = int(msg.get("num_turns") or 0)
            result.stop_reason = str(msg.get("stop_reason") or "")
            result.session_id = msg.get("session_id", result.session_id)
            if msg.get("is_error"):
                result.error = f"claude reported is_error (subtype={msg.get('subtype')})"

    result.answer = "".join(texts).strip()
    result.n_tool_calls = len(result.tool_calls)
    result.total_tokens = (
        result.input_tokens
        + result.output_tokens
        + result.cache_creation_input_tokens
        + result.cache_read_input_tokens
    )


def run_one(
    task: Task,
    arm: str,
    *,
    rep: int = 0,
    arm_position: int = 0,
    model: str = DEFAULT_MODEL,
    results_dir: Path = RESULTS_DIR,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    assert_index_offered: bool = True,
    defer_parity: bool = False,
    nudge_discovery: bool = False,
    fast_handshake: bool = True,
) -> RunResult:
    """Run one task in one arm and write one durable JSON.

    `assert_index_offered` defaults to True and is the reason this function can be
    trusted: a run whose MCP server never came up is marked `ok=False` and excluded by
    `analyze.py` rather than being averaged in as a fast, cheap, index-free success.
    Turn it off only to deliberately measure a broken server.

    `--output-format stream-json` rather than `json`: the single-object `json` format
    reports usage and the final text but not the tool calls, and "89% fewer tool calls"
    is the claim under test. The stream's terminal `result` event is byte-identical to
    what `json` would have emitted, so nothing is given up by taking the superset.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {list(ARMS)}")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{task.id}__{arm}__r{rep}__{uuid.uuid4().hex[:8]}"

    result = RunResult(
        run_id=run_id,
        task_id=task.id,
        arm=arm,
        repo=str(task.repo),
        model=model,
        rep=rep,
        arm_position=arm_position,
        started_at=time.time(),
        prompt=task.prompt,
        ground_truth=task.ground_truth,
    )

    mcp_config = resolve_config(arm, defer_parity=defer_parity,
                                fast_handshake=fast_handshake)
    result.extra_config = {
        "mcp_config": str(mcp_config),
        "defer_parity": defer_parity,
        "nudge_discovery": nudge_discovery,
        "fast_handshake": fast_handshake,
    }

    cmd = [
        CLAUDE_BIN,
        "-p",
        task.prompt,
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        *ALLOWED,
        "--disallowedTools",
        *DISALLOWED,
    ]
    if nudge_discovery:
        cmd += ["--append-system-prompt", DISCOVERY_NUDGE]

    transcript = results_dir / f"{run_id}.stream.jsonl"
    start = time.perf_counter()
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            cmd,
            cwd=str(task.repo),
            env=_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        result.wall_s = time.perf_counter() - start
        result.exit_code = proc.returncode
        transcript.write_text(proc.stdout)
        result.transcript_path = str(transcript)
        _parse_stream(proc.stdout.splitlines(), result)
        if proc.returncode != 0:
            result.error = (result.error or "") + f" exit={proc.returncode}: {proc.stderr[:2000]}"
    except subprocess.TimeoutExpired:
        result.wall_s = time.perf_counter() - start
        result.error = f"timeout after {timeout_s}s"

    if result.error is None and assert_index_offered and not result.index_offered:
        result.error = (
            f"arm {arm!r} expected MCP server(s) {ARMS[arm]['servers']} but init "
            f"reported {result.mcp_status or 'no servers'} -- the server did not "
            f"start. This run is VOID, not cheap; excluding it."
        )
    result.ok = result.error is None and bool(result.answer)

    (results_dir / f"{run_id}.json").write_text(json.dumps(asdict(result), indent=2, default=str))
    return result


# ---------------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------------


def arm_orders(n_blocks: int, arms: list[str], seed: int = 20260803) -> list[list[str]]:
    """One randomised arm permutation per block, so cache warmth cannot favour an arm.

    Each block runs every arm once, which keeps the design PAIRED -- the analysis
    compares arms within a block, not across the whole matrix. Within a block the order
    is a fresh permutation, so across enough blocks every arm sits in every position
    about equally often and any prompt-cache advantage from running second is spread
    evenly instead of accruing to whichever arm was listed last.

    The seed is fixed so a re-analysis lands on the same schedule as the run it is
    re-analysing.
    """
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    orders = []
    for _ in range(n_blocks):
        block = list(arms)
        rng.shuffle(block)
        orders.append(block)
    return orders


def probe_server(arm: str, repo: Path, *, defer_parity: bool = False,
                 fast_handshake: bool = True, timeout_s: int = 60) -> dict[str, Any]:
    """Speak MCP to the arm's server directly and list its tools.

    Independent of `claude` entirely, and that is the point: it separates "the server
    is broken" from "the server was still connecting when `init` fired". Without it a
    `pending` status is ambiguous between the two, and the ambiguous case is exactly
    the one that biases the result. Also the only place the tool SCHEMAS can be read,
    which is half of what characterising a control arm means.
    """
    cfg = resolve_config(arm, defer_parity=defer_parity, fast_handshake=fast_handshake)
    spec = json.loads(Path(cfg).read_text()).get("mcpServers", {})
    out: dict[str, Any] = {"arm": arm, "config": str(cfg), "servers": {}}
    for name, s in spec.items():
        env = _env()
        env.update(s.get("env") or {})
        argv = [s["command"], *(s.get("args") or [])]
        entry: dict[str, Any] = {"command": argv}
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(  # noqa: S603 -- argv from a config file in this repo
                argv, cwd=str(repo), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except OSError as exc:
            entry["error"] = f"spawn failed: {exc}"
            out["servers"][name] = entry
            continue

        def send(obj: dict[str, Any], p: subprocess.Popen = proc) -> None:
            assert p.stdin is not None  # noqa: S101 -- Popen(stdin=PIPE) guarantees it
            p.stdin.write(json.dumps(obj) + "\n")
            p.stdin.flush()

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "bench-probe", "version": "0"}}})
            deadline = time.time() + timeout_s
            assert proc.stdout is not None  # noqa: S101 -- Popen(stdout=PIPE)
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == 1:
                    entry["server_info"] = msg.get("result", {}).get("serverInfo")
                    entry["init_s"] = round(time.perf_counter() - started, 3)
                    send({"jsonrpc": "2.0", "method": "notifications/initialized",
                          "params": {}})
                    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
                elif msg.get("id") == 2:
                    tools = msg.get("result", {}).get("tools", [])
                    entry["tools"] = [
                        {"name": t.get("name"),
                         "description": (t.get("description") or "")[:400],
                         "input_schema": t.get("inputSchema")}
                        for t in tools
                    ]
                    entry["n_tools"] = len(tools)
                    break
            entry.setdefault("error", None if entry.get("tools") else "no tools/list reply")
        finally:
            if proc.stdin:
                proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        entry["total_s"] = round(time.perf_counter() - started, 3)
        out["servers"][name] = entry
    return out


def verify_arms(
    repo: Path,
    model: str = DEFAULT_MODEL,
    results_dir: Path | None = None,
    defer_parity: bool = False,
    fast_handshake: bool = True,
) -> dict[str, Any]:
    """Prove the three arms differ ONLY in their MCP tools, before spending on runs.

    Two ways a three-arm comparison silently becomes meaningless: a bare arm that is
    missing a built-in the other arms have (which would fake a large win), and an MCP
    server that never starts (which would fake a cheap one). Both are checked here with
    a trivial prompt, against the real config files, in the real working directory.

    Returns the per-arm offered sets and a `same_builtins` verdict. Cheap enough to run
    before every matrix and worth doing every time -- an MCP server that worked
    yesterday is not evidence it is on PATH today.
    """
    probe = Task(id="_verify", prompt="Reply with the single word: ok", repo=repo)
    out: dict[str, Any] = {"repo": str(repo), "defer_parity": defer_parity,
                           "fast_handshake": fast_handshake,
                           "snapshots": check_snapshots(repo, fast_handshake), "arms": {}}
    builtins: dict[str, set[str]] = {}
    for arm in ARM_NAMES:
        direct = probe_server(arm, repo, defer_parity=defer_parity,
                              fast_handshake=fast_handshake)
        r = run_one(
            probe,
            arm,
            model=model,
            results_dir=results_dir or (RESULTS_DIR / "_verify"),
            timeout_s=300,
            assert_index_offered=False,
            defer_parity=defer_parity,
            fast_handshake=fast_handshake,
        )
        offered = set(r.tools_offered)
        mcp_tools = {t for t in offered if t.startswith("mcp__")}
        companions = offered & MCP_COMPANION_TOOLS
        builtins[arm] = offered - mcp_tools - companions
        out["arms"][arm] = {
            "mcp_companion_tools": sorted(companions),
            "reachable": {
                n: {"n_tools": s.get("n_tools", 0), "init_s": s.get("init_s"),
                    "server_info": s.get("server_info"), "error": s.get("error")}
                for n, s in direct["servers"].items()
            },
            "tool_schemas": {
                n: s.get("tools", []) for n, s in direct["servers"].items()
            },
            "index_offered": r.index_offered,
            "index_first_class": r.index_first_class,
            "mcp_status_at_init": r.mcp_status,
            "mcp_tools_first_class": sorted(mcp_tools),
            "n_builtins": len(builtins[arm]),
            "error": r.error,
        }
    ref = builtins[ARM_NAMES[0]]
    out["same_builtins"] = all(builtins[a] == ref for a in ARM_NAMES)
    out["builtins"] = sorted(ref)
    out["builtin_diffs"] = {
        a: {"missing": sorted(ref - builtins[a]), "extra": sorted(builtins[a] - ref)}
        for a in ARM_NAMES
        if builtins[a] != ref
    }
    #: Not a defect in this harness, but it changes what the comparison means: this
    #: Claude Code build has no Grep or Glob tool at all -- search is done through Bash
    #: (ripgrep is on PATH). A published "N% fewer tool calls vs. grep+read" was
    #: measured against a baseline that HAD a first-class Grep, and is not this one.
    out["has_grep_tool"] = "Grep" in ref
    #: True when every arm's server is reachable AND its tools are first-class, i.e.
    #: no arm is paying a `ToolSearch` tax the others avoid. False is not fatal --
    #: `toolsearch_calls` makes it subtractable and `--defer-parity` equalises it --
    #: but it must be known before the numbers are read.
    #: Only arms that HAVE a server can be deferred; bare is trivially first-class and
    #: including it would report a false asymmetry whenever both index arms agree.
    index_arms = [a for a in ARM_NAMES if ARMS[a]["servers"]]
    out["deferral_symmetric"] = len(
        {out["arms"][a]["index_first_class"] for a in index_arms}
    ) <= 1
    #: The companion tools are useless here (code-learner's resource list is empty) but
    #: they are not free -- they occupy the system prompt in one arm and not the others.
    #: Not fatal, and not silent either.
    out["companion_tools_symmetric"] = len(
        {tuple(out["arms"][a]["mcp_companion_tools"]) for a in ARM_NAMES}
    ) == 1
    return out


# ---------------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------------


def run_matrix(
    tasks: list[Task],
    arms: list[str] | None = None,
    *,
    reps: int = 1,
    model: str = DEFAULT_MODEL,
    results_dir: Path = RESULTS_DIR,
    seed: int = 20260803,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    defer_parity: bool = False,
    nudge_discovery: bool = False,
    fast_handshake: bool = True,
    on_result: Any = None,
) -> list[RunResult]:
    """Every task x every arm x `reps`, in counterbalanced blocks.

    A block is one (task, rep) pair and holds one run of every arm, so the pairing the
    analysis relies on survives an interrupted matrix: stopping halfway leaves whole
    blocks rather than an arm-skewed prefix.
    """
    arms = list(arms or ARM_NAMES)
    blocks = [(t, rep) for rep in range(reps) for t in tasks]
    orders = arm_orders(len(blocks), arms, seed=seed)
    results: list[RunResult] = []
    for (task, rep), order in zip(blocks, orders, strict=True):
        for pos, arm in enumerate(order):
            r = run_one(
                task,
                arm,
                rep=rep,
                arm_position=pos,
                model=model,
                results_dir=results_dir,
                timeout_s=timeout_s,
                defer_parity=defer_parity,
                nudge_discovery=nudge_discovery,
                fast_handshake=fast_handshake,
            )
            results.append(r)
            if on_result:
                on_result(r)
    return results


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Every run JSON on disk. The analysis never re-runs the agent."""
    out = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _fmt(r: RunResult) -> str:
    flag = "ok " if r.ok else "VOID"
    return (
        f"[{flag}] {r.task_id:<24} {r.arm:<13} pos={r.arm_position} "
        f"calls={r.n_tool_calls:<3} (index={r.index_tool_calls} "
        f"search={r.toolsearch_calls} err={r.index_tool_errors}) "
        f"fc={int(r.index_first_class)} "
        f"tok={r.total_tokens:<8,} cache_read={r.cache_read_input_tokens:<8,} "
        f"${r.total_cost_usd:.4f} {r.wall_s:6.1f}s"
        + (f"\n       {r.error[:300]}" if r.error else "")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="prove the arms differ only in MCP tools")
    v.add_argument("--repo", type=Path, default=Path("/home/keith/projects/swarm-sync"))
    v.add_argument("--model", default=DEFAULT_MODEL)
    v.add_argument("--defer-parity", action="store_true")
    v.add_argument("--no-fast-handshake", action="store_true",
                   help="use the servers exactly as shipped, without shim/fastmcp.py")
    v.add_argument("--schemas", action="store_true", help="print full tool schemas")

    r = sub.add_parser("run", help="run the task x arm matrix")
    r.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    r.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    r.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--seed", type=int, default=20260803)
    r.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    r.add_argument("--only", nargs="*", default=None, help="restrict to these task ids")
    r.add_argument(
        "--defer-parity",
        action="store_true",
        help="start codegraph behind shim/delay.py so both index arms pay the same "
             "deferred-tool overhead",
    )
    r.add_argument(
        "--nudge-discovery",
        action="store_true",
        help="append an arm-neutral note that tools may be deferred, so an index the "
             "agent never SEES is not scored as an index that did not HELP",
    )
    r.add_argument(
        "--no-fast-handshake",
        action="store_true",
        help="use the servers exactly as shipped. Reproduces the out-of-the-box "
             "condition, in which code-learner's tools are deferred and the agent was "
             "measured never calling them -- a deployment fact, not an index result",
    )

    args = ap.parse_args(argv)

    if args.cmd == "verify":
        report = verify_arms(args.repo, model=args.model,
                             defer_parity=args.defer_parity,
                             fast_handshake=not args.no_fast_handshake)
        shown = json.loads(json.dumps(report))
        if not args.schemas:
            for a in shown["arms"].values():
                a.pop("tool_schemas", None)
        print(json.dumps(shown, indent=2))
        if not report["same_builtins"]:
            print("\nFAIL: arms do not share a built-in tool set.", file=sys.stderr)
            return 1
        unreachable = [
            f"{a}/{n}"
            for a, d in report["arms"].items()
            for n, s in d["reachable"].items()
            if s["error"] or not s["n_tools"]
        ]
        if unreachable:
            print(f"\nFAIL: MCP server unreachable: {unreachable}.", file=sys.stderr)
            return 1
        print("\nOK: arms differ only in their MCP tools; every server answers "
              "tools/list directly.")
        if not report["has_grep_tool"]:
            print(
                "NOTE: this Claude Code build exposes no Grep/Glob tool; search runs "
                "through Bash. Tool-call counts are not comparable to a published "
                "benchmark whose baseline had Grep."
            )
        if not report["companion_tools_symmetric"]:
            comp = {a: report["arms"][a]["mcp_companion_tools"] for a in ARM_NAMES}
            print(
                f"NOTE: Claude Code injects MCP companion tools unevenly ({comp}). They "
                f"appear because a server declared a `resources`/`prompts` capability; "
                f"code-learner's resource list is EMPTY, so they can do nothing but "
                f"still occupy that arm's system prompt."
            )
        if not report["deferral_symmetric"]:
            fc = {
                a: report["arms"][a]["index_first_class"]
                for a in ARM_NAMES
                if ARMS[a]["servers"]
            }
            print(
                f"NOTE: tool deferral is ASYMMETRIC across arms ({fc}). An arm whose "
                f"server is still `pending` at init pays extra ToolSearch calls for "
                f"its dependency's startup time, not for its index. Re-run with "
                f"--defer-parity, or subtract `toolsearch_calls`."
            )
        return 0

    tasks = load_tasks(args.tasks_dir)
    if args.only:
        tasks = [t for t in tasks if t.id in set(args.only)]
    if not tasks:
        print(f"no tasks in {args.tasks_dir}", file=sys.stderr)
        return 1
    print(f"{len(tasks)} tasks x {len(args.arms)} arms x {args.reps} reps "
          f"= {len(tasks) * len(args.arms) * args.reps} runs")
    run_matrix(
        tasks,
        args.arms,
        reps=args.reps,
        model=args.model,
        results_dir=args.results_dir,
        seed=args.seed,
        timeout_s=args.timeout,
        defer_parity=args.defer_parity,
        nudge_discovery=args.nudge_discovery,
        fast_handshake=not args.no_fast_handshake,
        on_result=lambda r: print(_fmt(r), flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
