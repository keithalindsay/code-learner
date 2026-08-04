"""Start an MCP stdio server after a fixed delay, to equalise connection timing.

Exists because of a measured harness artifact, not a hypothesis. Claude Code emits its
`init` event -- the one that decides whether a tool is FIRST-CLASS or lands in the
deferred pool behind `ToolSearch` -- before slow MCP servers finish connecting. On this
machine codegraph (a compiled binary) reports `connected` at init and its tool is
first-class; code-learner reports `pending` and its tools cost the agent two to three
extra `ToolSearch` calls before it can use them at all. About 0.6s of that gap is the
reference `mcp` Python SDK's own import, which no amount of benchmark hygiene removes.

That is a real property of the two servers and worth reporting. It is NOT a property of
the two INDEXES, which is what the benchmark is measuring, so an arm comparison that
leaves it in charges code-learner two or three tool calls per run for its dependency's
import time. This shim puts the faster server on the slower one's footing so the
deferred-tool overhead is a constant across arms and cancels in the paired analysis.

Deliberately the crude fix: delaying the fast server rather than accelerating the slow
one keeps every byte of both servers' real behaviour intact. `--defer-parity` turns it
on; the default is off, and the report gives both numbers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> int:
    delay = float(os.environ.get("BENCH_MCP_DELAY_S", "1.2"))
    if len(sys.argv) < 2:
        print("usage: delay.py <command> [args...]", file=sys.stderr)
        return 2
    time.sleep(delay)
    # exec so the child owns stdin/stdout directly: the MCP transport is this process's
    # stdio and a relay would add a copy loop plus a place for framing to go wrong.
    os.execvp(sys.argv[1], sys.argv[1:])  # noqa: S606 -- fixed argv from the config file
    return subprocess.call(sys.argv[1:])  # noqa: S603 -- unreachable; keeps mypy honest


if __name__ == "__main__":
    raise SystemExit(main())
