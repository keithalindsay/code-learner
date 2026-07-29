"""The `codelearner-mcp` command: resolve an index path, then serve.

Starting is deliberately unconditional. An MCP client launches its servers at
session start and marks one that exits non-zero as failed, often permanently until
the user notices -- so a server that refused to start because the index had not been
built yet would take the whole integration down over a condition that is fixed by
one command. Instead it starts, and the first tool call answers `no_index` with the
command to run. `codelearner.server.app` holds the same policy for every other
predictable failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cli.commands import resolve_index_path
from .app import build_server

EXIT_OK = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codelearner-mcp",
        description=(
            "Serve a code-learner index over MCP, so a coding agent can search it, "
            "walk its call graph, and submit inferences that are gated on citations."
        ),
    )
    parser.add_argument(
        "repo",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="repository root whose index to serve (default: the current directory)",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="serve this index file instead of <repo>/.codelearner/index.db",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help=(
            "stdio (default) is what a local MCP client launches as a subprocess; "
            "streamable-http serves over a port for a remote or shared index"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    server = build_server(index_path)
    # Blocks until the client disconnects. Nothing may be printed to stdout on the
    # stdio transport -- stdout IS the protocol channel, and one stray line makes the
    # session unparseable for the client rather than merely noisy.
    server.run(transport=args.transport)
    return EXIT_OK
