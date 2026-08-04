"""Answer the MCP handshake instantly, so a slow server's tools are not made invisible.

## The problem this exists to remove

Claude Code emits its `init` event -- the one that decides whether a tool is FIRST-CLASS
or lands in the deferred pool behind `ToolSearch` -- a few hundred milliseconds after
launch, without waiting for slow MCP servers. Measured on this machine by bracketing
with `delay.py`: codegraph's compiled server answers `initialize` in **116 ms** and is
first-class; adding a 300 ms delay is already enough to make it `pending`. The window is
under ~420 ms.

`codelearner.server` answers in **670 ms**, of which about 600 ms is the reference `mcp`
Python SDK's own import. It misses the window every time. There is no `MCP_TIMEOUT` or
equivalent that helps: that variable caps how long a connection MAY take, it does not
make `init` wait.

The consequence is not a few extra tool calls. Across the pilot matrix the code-learner
arm had its server running and its five tools loadable, and the agent called one in
**zero of six runs** -- the tools were in the deferred pool, nothing pointed at them,
and it used `Bash` instead. Appending a note that deferred tools exist did not fix it
either: in four more runs the agent never even spent the `ToolSearch`. An arm in that
state does not measure an index. It measures `bare` plus a second of startup latency,
and it scores WORSE than bare, which would be a false negative about the index caused
entirely by a dependency's import time.

## What this does instead

A relay small enough to start fast: stdlib only, no `mcp` import, ~30 ms to first byte.
It spawns the real server immediately in the background, answers `initialize` and
`tools/list` from a snapshot captured beforehand, and forwards every other message --
including every actual tool call -- to the real server untouched.

So the server's real behaviour is fully intact. The ONLY thing removed from the critical
path is the interpreter and SDK import, which is not part of what the benchmark is
comparing. Nothing about retrieval quality, tool semantics, or response content passes
through a cache.

## Why a snapshot rather than a faster server

Making `codelearner.server` start in under 400 ms would mean changing package code, and
the benchmark must not modify the thing it is measuring. A snapshot is auditable
instead: `--capture` writes it, `--check` re-derives it and diffs, and the file's hash
goes into the run record. If the tool list ever drifts from the snapshot the comparison
is void and the diff says so, rather than the agent being quietly served a stale list.

## Its limits, stated plainly

The snapshot is trusted for exactly two methods. If a server's tool list depended on the
working directory or on runtime state, the snapshot could serve a list the real server
would not -- so `--check` is run per repo, not once. And a server that fails to start is
still detectable: forwarded calls error out, `index_tool_errors` counts them, and
`harness.probe_server` talks to the real server directly with no relay involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

SHIM_DIR = Path(__file__).resolve().parent

#: Methods answered from the snapshot: exactly the handshake, and nothing else.
#: Every `tools/call` goes to the real server untouched.
#:
#: All four are here because all four are on the critical path, which took a packet
#: capture to establish. Claude Code's opening sequence against a server that declares
#: `prompts` and `resources` capabilities is:
#:
#:     initialize -> notifications/initialized -> tools/list -> prompts/list
#:                -> resources/list
#:
#: codegraph declares only `{"tools": {}}`, so it is asked for `tools/list` alone --
#: two round trips against a 116 ms server. The reference `mcp` Python SDK declares
#: `prompts` and `resources` whether or not a server implements any, so
#: `codelearner.server` is asked for four, serialised behind a ~600 ms SDK import.
#: Caching only `initialize` and `tools/list` was measured and did NOT help: the two
#: uncached list calls were forwarded, blocked on the real server's startup, and the
#: connection was still `pending` when `init` fired.
CACHED_METHODS = ("initialize", "tools/list", "prompts/list", "resources/list")


def _read_message(stream: Any) -> dict | None:
    """One newline-delimited JSON-RPC message, or None at EOF.

    Only the line framing is implemented, because that is what both servers in this
    benchmark speak. A Content-Length transport would need a different reader and this
    would fail loudly rather than silently mangling it.
    """
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def _write_message(stream: Any, msg: dict) -> None:
    stream.write(json.dumps(msg) + "\n")
    stream.flush()


def _empty_result(method: str) -> dict:
    """What "this server has none of those" looks like for a list method."""
    return {method.split("/")[0]: []}


def capture(argv: list[str], cwd: Path, timeout_s: int = 120) -> dict:
    """Run the real server once and record its handshake replies.

    Separated from serving so the snapshot is a reviewable artifact rather than
    something conjured at run time: capture happens before the matrix, `--check`
    re-derives it afterwards, and any drift between them voids the comparison.
    """
    proc = subprocess.Popen(  # noqa: S603 -- argv comes from a config file in this repo
        argv, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    snapshot: dict[str, Any] = {"command": argv, "cwd": str(cwd)}
    try:
        _write_message(proc.stdin, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "bench-fastmcp", "version": "0"}}})
        while True:
            msg = _read_message(proc.stdout)
            if msg is None:
                raise RuntimeError("server closed before initialize reply")
            if msg.get("id") == 1:
                snapshot["initialize"] = msg.get("result")
                _write_message(proc.stdin,
                               {"jsonrpc": "2.0", "method": "notifications/initialized",
                                "params": {}})
                for i, method in enumerate(CACHED_METHODS[1:], start=2):
                    _write_message(proc.stdin,
                                   {"jsonrpc": "2.0", "id": i, "method": method,
                                    "params": {}})
            elif isinstance(msg.get("id"), int) and 2 <= msg["id"] <= len(CACHED_METHODS):
                method = CACHED_METHODS[msg["id"] - 1]
                # A server may legitimately not implement prompts/resources and answer
                # with an error. Store an empty list rather than the error: the relay
                # must reply something well-formed inside the init window, and an empty
                # list is what "this server has none" means to the client.
                snapshot[method] = msg.get("result") or _empty_result(method)
                if all(m in snapshot for m in CACHED_METHODS):
                    break
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    body = json.dumps(snapshot.get("tools/list"), sort_keys=True)
    snapshot["tools_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return snapshot


def serve(snapshot_path: Path) -> int:
    """Relay stdio to the real server, answering the handshake from the snapshot."""
    snapshot = json.loads(snapshot_path.read_text())
    argv = snapshot["command"]

    proc = subprocess.Popen(  # noqa: S603 -- argv from the snapshot this repo captured
        argv, cwd=os.getcwd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sys.stderr, text=True, bufsize=1,
    )
    ready = threading.Event()

    def handshake_real() -> None:
        """Bring the real server to the same state the client thinks it is in.

        Runs off-thread so the client's `initialize` is answered from the snapshot
        without waiting for the import. Ids 900001/900002 are outside the range a
        client would use, so the replies cannot be confused with forwarded traffic.
        """
        try:
            _write_message(proc.stdin, {
                "jsonrpc": "2.0", "id": 900001, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "bench-fastmcp", "version": "0"}}})
            while True:
                msg = _read_message(proc.stdout)
                if msg is None:
                    break
                if msg.get("id") == 900001:
                    _write_message(proc.stdin,
                                   {"jsonrpc": "2.0",
                                    "method": "notifications/initialized",
                                    "params": {}})
                    break
        except (BrokenPipeError, OSError):
            pass
        finally:
            ready.set()

    threading.Thread(target=handshake_real, daemon=True).start()

    def pump_downstream() -> None:
        """Real server -> client, once the private handshake is out of the way."""
        ready.wait(timeout=120)
        while True:
            msg = _read_message(proc.stdout)
            if msg is None:
                break
            _write_message(sys.stdout, msg)

    threading.Thread(target=pump_downstream, daemon=True).start()

    while True:
        msg = _read_message(sys.stdin)
        if msg is None:
            break
        method = msg.get("method")
        if method in CACHED_METHODS and msg.get("id") is not None:
            _write_message(sys.stdout,
                           {"jsonrpc": "2.0", "id": msg["id"],
                            "result": snapshot[method]})
            continue
        if method == "notifications/initialized":
            continue  # already sent to the real server by `handshake_real`
        ready.wait(timeout=120)
        try:
            _write_message(proc.stdin, msg)
        except (BrokenPipeError, OSError):
            break

    if proc.stdin:
        try:
            proc.stdin.close()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--capture", nargs=argparse.REMAINDER,
                    help="capture a snapshot by running: --capture <command> [args...]")
    ap.add_argument("--check", action="store_true",
                    help="re-derive the snapshot and fail if the tool list drifted")
    ap.add_argument("--cwd", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)

    if args.capture:
        snap = capture(args.capture, args.cwd)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snap, indent=2))
        n = len((snap.get("tools/list") or {}).get("tools", []))
        print(f"captured {n} tools -> {args.snapshot} (sha256 {snap['tools_sha256'][:16]})")
        return 0

    if args.check:
        old = json.loads(args.snapshot.read_text())
        new = capture(old["command"], args.cwd)
        if new["tools_sha256"] != old["tools_sha256"]:
            print(
                f"SNAPSHOT DRIFT in {args.snapshot}\n"
                f"  captured: {old['tools_sha256']}\n"
                f"  now:      {new['tools_sha256']}\n"
                f"  The relay would serve a tool list the server no longer has. Any "
                f"comparison using this snapshot is void; re-capture it.",
                file=sys.stderr,
            )
            return 1
        print(f"snapshot current ({new['tools_sha256'][:16]}, "
              f"{len(new['tools/list']['tools'])} tools)")
        return 0

    return serve(args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
