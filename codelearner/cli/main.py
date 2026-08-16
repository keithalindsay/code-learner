"""Argument parsing and dispatch for the `codelearner` command.

The parser is built by a function rather than at import time so tests can construct
one per case, and `main` takes `argv` and an embedder factory as parameters rather
than reading `sys.argv` and constructing a model itself. Both are the same decision:
the CLI is a library with a thin shell around it, so its behaviour can be asserted
in milliseconds instead of by shelling out and grepping text.

The embedder factory in particular is the seam that keeps the test suite off the
GPU. Loading `Qwen3-Embedding-0.6B` takes tens of seconds and ~1.2GB of VRAM that
another process may be holding; a fake that returns three floats proves the wiring
just as well, and proves it on a laptop with no card at all.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .. import db
from ..index import DEFAULT_MODEL, Embedder
from .commands import (
    CliError,
    EmbedderFactory,
    cmd_gpu,
    cmd_index,
    cmd_judge,
    cmd_learn,
    cmd_search,
    cmd_stats,
)

DEFAULT_K = 10


class _RepoPathAction(argparse.Action):
    """Record that --repo came from the caller rather than the cwd default."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.repo_explicit = True

# Exit codes. 0 success, 1 a condition the tool predicted and explained, 2 a usage
# error (argparse's own convention, kept rather than re-invented). The distinction
# matters to a script: 2 means the command line was wrong, 1 means the world was.
EXIT_OK = 0
EXIT_ERROR = 1


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {value}")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or greater, got {value}")
    return value


def _default_embedder(model_name: str) -> Embedder:
    """Build the real embedder. Imported late -- the import pulls in torch."""
    from ..index import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codelearner",
        description="GraphRAG over a codebase: index it, then ask it questions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="parse a repository into an index")
    p_index.add_argument("repo", type=Path, help="repository root to index")
    p_index.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="where to write the index (default: <repo>/.codelearner/index.db)",
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="delete and rebuild an existing index. Always discards its embeddings; "
        "if it holds a tier-2 store (assertions, verdicts, staleness events) the "
        "rebuild refuses and names the counts, because only the embeddings are "
        "re-derivable -- pick --carry-assertions or --discard-assertions",
    )
    # Mutually exclusive because the two answers to "what happens to the store" are
    # opposites, and a command line asserting both is one whose author believed
    # something untrue about at least one of them.
    tier2 = p_index.add_mutually_exclusive_group()
    tier2.add_argument(
        "--carry-assertions",
        action="store_true",
        help="with --force: carry the tier-2 store across the rebuild. Subjects are "
        "re-resolved by qualname, a claim whose subject is gone keeps a NULL link, "
        "and a claim whose cited bytes moved comes back stale with a log row",
    )
    tier2.add_argument(
        "--discard-assertions",
        action="store_true",
        help="with --force: destroy the tier-2 store along with the index. "
        "Irreversible -- verdicts and the rejected set cannot be re-derived from "
        "source the way embeddings can",
    )
    p_index.add_argument(
        "--embed",
        action="store_true",
        help="also build dense vectors (slow; needs the [embed] extra and a GPU to be quick)",
    )
    p_index.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"embedding model, with --embed (default: {DEFAULT_MODEL})",
    )
    p_index.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="hybrid search over an index")
    p_search.add_argument("query", help="what to look for, in plain words")
    _add_index_location(p_search)
    p_search.add_argument(
        "-k", "--k", type=_positive_int, default=DEFAULT_K, help=f"results to return (default: {DEFAULT_K})"
    )
    p_search.add_argument(
        "--facts-only",
        action="store_true",
        help="return T0/T1 only -- parsed facts and resolved names, nothing inferred",
    )
    # The three modality switches exist for the same reason `search()` has them:
    # "which modality actually carries retrieval" is a question that can only be
    # answered by turning each one off.
    p_search.add_argument("--no-lexical", action="store_true", help="disable BM25 lexical search")
    p_search.add_argument("--no-dense", action="store_true", help="disable vector search")
    p_search.add_argument("--no-graph", action="store_true", help="disable graph expansion")
    p_search.add_argument(
        "--no-assertions",
        action="store_true",
        help="disable tier-2 semantic retrieval -- the ablation that says what "
        "stored claims are worth against source alone",
    )
    p_search.add_argument(
        "--debug-scores",
        action="store_true",
        help="show the per-modality rank contributions behind each fused score",
    )
    # Opt-in, not on by default: it downloads ~3.4GB of weights on first use and
    # costs a model forward pass per candidate. Off, `search` behaves exactly as it
    # did before Phase 3b; on and unavailable, it says so and still answers.
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="reorder results with a cross-encoder that reads the query (slow, downloads a model)",
    )
    p_search.add_argument(
        "--include-source",
        action="store_true",
        help="include complete, current source for returned symbols",
    )
    p_search.add_argument(
        "--evidence-budget",
        type=_nonnegative_int,
        default=16_384,
        metavar="BYTES",
        help="source-evidence byte budget when --include-source is set (default: 16384)",
    )
    p_search.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="what is in an index")
    _add_index_location(p_stats)
    p_stats.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_stats.set_defaults(func=cmd_stats)

    p_learn = sub.add_parser(
        "learn",
        help="draft tier-2 claims with a local model and admit the ones that cite evidence",
    )
    _add_index_location(p_learn)
    p_learn.add_argument(
        "--model",
        default=None,
        help=(
            "ollama model to draft with (default: llama3.1:8b, chosen because it is NOT "
            "the judge's family)"
        ),
    )
    p_learn.add_argument(
        "--host", default="http://localhost:11434", help="ollama host (default: localhost)"
    )
    p_learn.add_argument(
        "--limit", type=_positive_int, default=None, help="stop after this many symbols"
    )
    p_learn.add_argument(
        "--max-offers",
        type=_positive_int,
        default=12,
        help="how many evidence spans to put on the menu (default: 12)",
    )
    p_learn.add_argument(
        "--no-callers",
        action="store_true",
        help="offer only the subject and its callees; a symbol's purpose is usually "
        "visible from its callers, so this makes the task harder on purpose",
    )
    p_learn.add_argument(
        "--redo",
        action="store_true",
        help="re-draft symbols that already hold an active claim from this generator "
        "(default: skip them, so a long run is resumable)",
    )
    p_learn.add_argument("--quiet", action="store_true", help="no per-symbol progress")
    p_learn.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_learn.set_defaults(func=cmd_learn)

    p_judge = sub.add_parser(
        "judge",
        help="adjudicate unjudged claims so serving can use them",
    )
    p_judge.add_argument("repo", type=Path, nargs="?", default=Path("."))
    # `dest="index_path"` on purpose, matching every other command's internal name
    # for this value (`resolve_index_path` and `open_index` both take `index_path`)
    # even though the flag here is spelled `--index` rather than `--index-path`.
    p_judge.add_argument(
        "--index",
        type=Path,
        default=None,
        dest="index_path",
        help="index file to use (default: <repo>/.codelearner/index.db)",
    )
    p_judge.add_argument(
        "--limit", type=_positive_int, default=None, help="judge at most this many claims"
    )
    p_judge.add_argument("--model", default=None, help="ollama judge model tag")
    p_judge.add_argument(
        "--subject", default=None, help="only judge claims about this qualname"
    )
    p_judge.add_argument(
        "--allow-same-family",
        action="store_true",
        help="judge a claim even when the judge and its generator share a model "
        "family (by default such claims are skipped, not judged, and counted as "
        "skipped_same_family)",
    )
    p_judge.add_argument(
        "--dry-run",
        action="store_true",
        help="call the judge and report verdicts without writing them to the store",
    )
    p_judge.add_argument("--json", action="store_true", dest="json", help="emit JSON instead of a table")
    p_judge.set_defaults(func=cmd_judge)

    # No --repo and no --index-path, and that absence is the design. "What is holding
    # my card" is asked from wherever the terminal happens to be, often before an
    # index exists at all; requiring one would make the diagnostic unavailable in
    # exactly the situation it diagnoses.
    p_gpu = sub.add_parser(
        "gpu",
        help="what is holding VRAM, and (with --free) getting it back",
        description=(
            "Report what holds the GPU. With --free, ask ollama to unload every "
            "resident model and then VERIFY it happened -- an unload request returns "
            "success without freeing anything, so the check is the point. Exits 1 if "
            "a release was attempted and the memory did not come back, and 3 if it "
            "was declined because something is using the model -- so a script can "
            "gate a measurement run on it, and can tell 'wait' from 'fetch a human'."
        ),
    )
    p_gpu.add_argument(
        "--free",
        action="store_true",
        help="ask ollama to unload its IDLE models, then poll until the VRAM is "
        "actually back or --wait expires. Refuses if a model is serving requests; "
        "never kills a process; prints what to run if the polite path fails",
    )
    p_gpu.add_argument(
        "--force",
        action="store_true",
        help="with --free: unload even a model that is currently serving requests. "
        "This will interrupt whoever is calling it -- only use it when that caller "
        "is yours",
    )
    p_gpu.add_argument(
        "--no-usage-check",
        action="store_true",
        help="skip the second /api/ps sample that tells an idle model from one being "
        "called. Saves ~1.5s and gives up the only signal that makes freeing safe",
    )
    p_gpu.add_argument(
        "--host", default="http://localhost:11434", help="ollama host (default: localhost)"
    )
    p_gpu.add_argument(
        "--wait",
        type=float,
        default=30.0,
        help="seconds to wait for the VRAM to come back, with --free (default: 30)",
    )
    p_gpu.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_gpu.set_defaults(func=cmd_gpu)

    return parser


def _add_index_location(parser: argparse.ArgumentParser) -> None:
    """The two ways every read-only command finds its index.

    Defaulting `--repo` to the working directory is what makes `codelearner search
    "..."` work with no arguments from inside a repo, which is the shape of the
    command people actually type.
    """
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        action=_RepoPathAction,
        help="repository whose index to use (default: the working directory)",
    )
    parser.set_defaults(repo_explicit=False)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="index file to use (default: <repo>/.codelearner/index.db)",
    )


def main(argv: list[str] | None = None, embedder_factory: EmbedderFactory | None = None) -> int:
    """Run one command. Returns the process exit code; never raises for a
    predictable failure."""
    args = build_parser().parse_args(argv)
    factory: EmbedderFactory = embedder_factory or _default_embedder
    try:
        return int(args.func(args, factory))
    except CliError as exc:
        print(f"codelearner: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        # `open_index` and `cmd_index` both turn these into a `CliError` with a
        # remedy attached, and this clause exists for the paths that do not go
        # through either -- a library call reached from a command, a connection
        # opened deeper in. They are `RuntimeError` subclasses, so without a clause
        # of their own they land in Python's default handler as a traceback, which
        # is the one thing this function promises never to print.
        print(f"codelearner: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, sqlite3.Error) as exc:
        # The environmental failures -- unreadable path, full disk, locked database.
        # Predictable in kind if not in detail, and a traceback tells the user
        # nothing they can act on.
        print(f"codelearner: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("codelearner: interrupted", file=sys.stderr)
        return 130
