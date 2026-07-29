"""The three commands: index, search, stats.

Every failure here is somebody's Tuesday afternoon, so the rule this module follows
is that a user must never see a traceback for a condition the tool could have
predicted. A missing index, an index without embeddings, a model that does not match
the vectors in the file -- these are all normal states of the world, and each one
gets a sentence that says what happened and what to do about it. `CliError` is the
carrier for exactly that: raised here, printed without a stack by `main`.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import db
from ..index import Embedder, embed_chunks
from ..ingest import index_repo
from ..retrieve import load_reranker, search, stored_embed_model
from .render import count_line, facts_only, format_hit, hit_json

# Kept in step with `indexer.index_repo`'s own default. One file per repo is what
# makes cross-repo contamination structurally impossible, so the CLI must not
# invent a second convention for where that file lives.
INDEX_RELPATH = Path(".codelearner") / "index.db"

EmbedderFactory = Callable[[str], Embedder]


class CliError(RuntimeError):
    """A condition with a known remedy. Printed as one line, never as a traceback."""


def resolve_index_path(repo: Path, index_path: Path | None) -> Path:
    """Where this invocation's index lives: explicit if given, else the default."""
    if index_path is not None:
        return index_path.expanduser()
    return repo / INDEX_RELPATH


def open_index(index_path: Path) -> sqlite3.Connection:
    """Open an EXISTING index, or explain how to make one.

    `db.connect` happily creates an empty SQLite file at any path, which is how a
    typo'd path becomes "0 results" instead of "no such index". Checking for the
    file first is the difference between a wrong answer and an error message.
    """
    if not index_path.exists():
        raise CliError(
            f"no index at {index_path}. Build one with "
            f"`codelearner index <repo>`, or point at an existing one with "
            f"--index-path."
        )
    try:
        return db.connect(index_path)
    except sqlite3.Error as exc:
        raise CliError(f"could not open the index at {index_path}: {exc}") from exc


def build_embedder(factory: EmbedderFactory, model_name: str) -> Embedder:
    """Construct an embedder, turning every way that can fail into one sentence.

    Loading a model reaches for torch, a GPU, and ~1.2GB of weights on disk. Any of
    the three can be absent, and none of them produce an error a user can act on
    without being told what was being attempted.
    """
    try:
        return factory(model_name)
    except ImportError as exc:
        raise CliError(
            f"embedding needs the optional dependencies: {exc}. "
            'Install them with `pip install -e ".[embed]"`.'
        ) from exc
    except Exception as exc:  # noqa: BLE001 - the CLI's job is to never traceback
        raise CliError(f"could not load the embedding model {model_name!r}: {exc}") from exc


def _delete_index(index_path: Path) -> None:
    """Remove an index file and its WAL sidecars.

    The `-wal` and `-shm` files are not incidental: deleting only the main file
    leaves a write-ahead log that SQLite will replay into the fresh database,
    resurrecting rows from the index that was supposed to be gone.
    """
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(index_path) + suffix)
        if candidate.exists():
            candidate.unlink()


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def cmd_index(args: Any, factory: EmbedderFactory) -> int:
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise CliError(f"{repo} is not a directory, so there is nothing to index.")

    index_path = resolve_index_path(repo, args.index_path)
    if index_path.exists():
        if not args.force:
            raise CliError(
                f"an index already exists at {index_path}. There is no incremental "
                "update yet, so re-indexing means rebuilding from scratch. Re-run "
                "with --force to delete and rebuild it -- note that this discards "
                "any embeddings, which are the expensive part -- or use "
                "--index-path to build a second index elsewhere."
            )
        _delete_index(index_path)

    try:
        conn, stats = index_repo(repo, index_path=index_path)
    except (db.SchemaVersionError, db.RepoRootMismatchError) as exc:
        raise CliError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise CliError(f"indexing failed while writing {index_path}: {exc}") from exc

    embed_info: dict[str, Any] | None = None
    if args.embed:
        embedder = build_embedder(factory, args.model)
        try:
            estats = embed_chunks(conn, embedder)
        except RuntimeError as exc:
            # sqlite-vec missing or unloadable. The structural half of the index is
            # already written and useful, so this reports rather than unwinds.
            raise CliError(str(exc)) from exc
        embed_info = {
            "model": estats.model,
            "dim": estats.dim,
            "embedded": estats.embedded,
            "skipped_unchanged": estats.skipped_unchanged,
        }

    rstats = stats.resolve
    in_repo = rstats.total - rstats.external
    payload = {
        "repo": str(repo),
        "index": str(index_path),
        "files": stats.files,
        "symbols": stats.symbols,
        "edges": stats.edges,
        "chunks": stats.chunks,
        "skipped": stats.skipped,
        "resolution": {
            "total": rstats.total,
            "resolved": rstats.resolved,
            "external": rstats.external,
            "ambiguous": rstats.ambiguous,
            "in_repo": in_repo,
            "rate": round(rstats.rate, 6),
            "rate_of_internal": round(rstats.rate_of_internal, 6),
            "by_resolver": dict(sorted(rstats.by_resolver.items())),
        },
        "embeddings": embed_info,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"indexed {repo}")
    print(f"  index      {index_path}")
    print(count_line("files", stats.files))
    print(count_line("symbols", stats.symbols))
    print(count_line("edges", stats.edges))
    print(count_line("chunks", stats.chunks))
    if stats.skipped:
        print(count_line("skipped", stats.skipped))
    # Two denominators, because only one of them is honest. Roughly half the calls
    # in real code target stdlib or third-party code and are CORRECTLY unresolvable;
    # counting those as failures makes a working resolver look broken.
    print(
        f"  resolved   {rstats.resolved:>9,}  "
        f"{rstats.rate_of_internal:.1%} of {in_repo:,} in-repo references "
        f"({rstats.external:,} target code outside this repo)"
    )
    if embed_info is not None:
        print(
            f"  embedded   {embed_info['embedded']:>9,}  "
            f"chunks with {embed_info['model']} ({embed_info['dim']}-dim)"
        )
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args: Any, factory: EmbedderFactory) -> int:
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    conn = open_index(index_path)

    use_lexical = not args.no_lexical
    use_dense = not args.no_dense
    use_graph = not args.no_graph
    notes: list[str] = []
    embedder: Embedder | None = None

    if use_dense:
        stored = stored_embed_model(conn)
        if stored is None:
            # The common case, and not an error: an index without embeddings still
            # answers with lexical and graph. Degrading loudly beats failing.
            use_dense = False
            notes.append(
                "dense retrieval unavailable: this index has no embeddings. Build "
                "them with `codelearner index <repo> --embed --force`."
            )
        else:
            embedder = build_embedder(factory, stored)
            if embedder.name != stored:
                # Vectors from two models are not comparable. Querying anyway
                # returns results that look plausible and mean nothing, which is
                # strictly worse than returning none.
                notes.append(
                    f"dense retrieval disabled: this index was embedded with "
                    f"{stored!r} but the loaded model is {embedder.name!r}, and "
                    "vectors from two models are not comparable."
                )
                use_dense = False
                embedder = None

    if not use_lexical and not use_dense:
        # Graph expansion cannot run alone. It has no query representation of its
        # own -- it is seeded by the text modalities -- so with both of them off it
        # would return nothing at all, for every query, silently.
        raise CliError(
            "no text modality is available, so there is nothing to search with. "
            "Graph expansion has no query representation of its own; it is seeded "
            "by lexical and dense results and cannot run alone. Drop --no-lexical, "
            "or build embeddings with `codelearner index <repo> --embed --force`."
        )

    reranker = None
    if getattr(args, "rerank", False):
        reranker = load_reranker(conn=conn)
        if reranker is None:
            # Asked for and not available. Say so and keep going -- the fused order
            # is the result every release before Phase 3b returned, and refusing the
            # query would be a strictly worse answer than a slightly worse ranking.
            notes.append(
                "reranking unavailable: no cross-encoder could be loaded (no model "
                "weights, or not enough memory). Returning the fused order."
            )

    result = search(
        conn,
        args.query,
        k=args.k,
        embedder=embedder,
        use_lexical=use_lexical,
        use_dense=use_dense,
        use_graph=use_graph,
        reranker=reranker,
    )
    hits = facts_only(result.hits) if args.facts_only else list(result.hits)

    # Notes go to stderr unconditionally so that `--json` on stdout stays a single
    # parseable document and a shell pipeline does not have to strip warnings.
    for note in notes:
        print(f"codelearner: {note}", file=sys.stderr)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "index": str(index_path),
                    "k": args.k,
                    "facts_only": args.facts_only,
                    "modalities": {
                        "lexical": use_lexical,
                        "dense": use_dense,
                        "graph": use_graph,
                    },
                    "count": len(hits),
                    "hits": [hit_json(hit, i) for i, hit in enumerate(hits, start=1)],
                },
                indent=2,
            )
        )
        return 0

    enabled = [
        name
        for name, on in (("lexical", use_lexical), ("dense", use_dense), ("graph", use_graph))
        if on
    ]
    if not hits:
        print(f"no results for {args.query!r}  [{'+'.join(enabled)}]")
        return 0
    print(f"{len(hits)} result(s) for {args.query!r}  [{'+'.join(enabled)}, k={args.k}]")
    for rank, hit in enumerate(hits, start=1):
        print(format_hit(hit, rank))
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return 0 if row is None else int(row[0])


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def _classify_unresolved(conn: sqlite3.Connection) -> tuple[int, int]:
    """Split unresolved edges into (external, ambiguous), mirroring `resolve_all`.

    Recomputed from the stored graph rather than remembered from index time, so
    `stats` tells the truth about the file in front of it even if the resolver was
    re-run since. External means "no symbol in this repo even shares the basename",
    which is the only honest way to say that a call to `json.dumps` is not a
    resolution failure.
    """
    names = {row["name"] for row in conn.execute("SELECT DISTINCT name FROM symbols")}
    external = ambiguous = 0
    for row in conn.execute("SELECT dst_name FROM edges WHERE dst_symbol_id IS NULL"):
        base = str(row["dst_name"]).rsplit(".", 1)[-1]
        if base in names:
            ambiguous += 1
        else:
            external += 1
    return external, ambiguous


def _embedding_info(conn: sqlite3.Connection) -> dict[str, Any]:
    """What vectors this index holds, if any, and from which model."""
    model = stored_embed_model(conn)
    dim = _meta(conn, "embed_dim")
    try:
        vectors = _scalar(conn, "SELECT count(*) FROM vec_chunks")
    except sqlite3.OperationalError:
        # No vec table, or sqlite-vec is not loadable on this handle. Either way
        # there is nothing to report and nothing to fail about.
        vectors = 0
    return {
        "present": bool(model and vectors),
        "model": model,
        "dim": int(dim) if dim is not None else None,
        "vectors": vectors,
    }


def cmd_stats(args: Any, factory: EmbedderFactory) -> int:
    del factory  # stats never loads a model; the stored name is all it reports
    index_path = resolve_index_path(args.repo.expanduser().resolve(), args.index_path)
    conn = open_index(index_path)

    try:
        counts = {
            "files": _scalar(conn, "SELECT count(*) FROM files"),
            "symbols": _scalar(conn, "SELECT count(*) FROM symbols"),
            "edges": _scalar(conn, "SELECT count(*) FROM edges"),
            "chunks": _scalar(conn, "SELECT count(*) FROM chunks"),
        }
        tier_rows = conn.execute(
            "SELECT tier, count(*) AS n FROM edges GROUP BY tier"
        ).fetchall()
        kind_rows = conn.execute(
            "SELECT kind, count(*) AS n FROM symbols GROUP BY kind ORDER BY n DESC"
        ).fetchall()
        resolver_rows = conn.execute(
            "SELECT resolver, count(*) AS n, avg(confidence) AS conf FROM edges "
            "WHERE dst_symbol_id IS NOT NULL GROUP BY resolver ORDER BY n DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise CliError(
            f"{index_path} does not look like a code-learner index ({exc}). "
            "Remedy: point --index-path at the right file, or re-index."
        ) from exc

    by_tier = {int(r["tier"]): int(r["n"]) for r in tier_rows}
    resolved = _scalar(conn, "SELECT count(*) FROM edges WHERE dst_symbol_id IS NOT NULL")
    external, ambiguous = _classify_unresolved(conn)
    in_repo = counts["edges"] - external
    rate = resolved / counts["edges"] if counts["edges"] else 0.0
    rate_of_internal = resolved / in_repo if in_repo else 0.0
    embeddings = _embedding_info(conn)

    payload = {
        "index": str(index_path),
        "repo_root": db.stored_repo_root(conn),
        "schema_version": _meta(conn, "schema_version"),
        "counts": counts,
        # The tier column lives on edges: 0 is the call site as written, 1 is that
        # site bound to a symbol. Symbols themselves are all T0 by construction --
        # they were parsed, not decided.
        "tiers": {
            "T0": by_tier.get(0, 0),
            "T1": by_tier.get(1, 0),
            "T2": by_tier.get(2, 0),
        },
        "symbol_kinds": {str(r["kind"]): int(r["n"]) for r in kind_rows},
        "resolution": {
            "total": counts["edges"],
            "resolved": resolved,
            "external": external,
            "ambiguous": ambiguous,
            "in_repo": in_repo,
            "rate": round(rate, 6),
            "rate_of_internal": round(rate_of_internal, 6),
            "by_resolver": {
                str(r["resolver"]): {
                    "count": int(r["n"]),
                    "confidence": round(float(r["conf"]), 4) if r["conf"] is not None else None,
                }
                for r in resolver_rows
            },
        },
        "embeddings": embeddings,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"index      {index_path}")
    print(f"repo       {payload['repo_root'] or '(unbound)'}")
    print(f"schema     v{payload['schema_version'] or '?'}")
    print()
    print("counts")
    for label in ("files", "symbols", "edges", "chunks"):
        print(count_line(label, counts[label]))
    print()
    print("edges by tier")
    print(count_line("T0 FACT", by_tier.get(0, 0), width=12) + "  call site as written, unbound")
    print(count_line("T1 RESOLVED", by_tier.get(1, 0), width=12) + "  bound to a symbol, with confidence")
    print(count_line("T2 INFERRED", by_tier.get(2, 0), width=12) + "  the inference layer is not built yet")
    print()
    print("symbol kinds")
    for row in kind_rows:
        print(count_line(str(row["kind"]), int(row["n"]), width=12))
    print()
    print("resolution")
    print(
        f"  {resolved:,} of {counts['edges']:,} edges resolved "
        f"-- {rate_of_internal:.1%} of {in_repo:,} in-repo "
        f"references ({external:,} external, {ambiguous:,} ambiguous)"
    )
    for row in resolver_rows:
        conf = row["conf"]
        suffix = f"  confidence {float(conf):.2f}" if conf is not None else ""
        print(count_line(str(row["resolver"]), int(row["n"]), width=24) + suffix)
    print()
    print("embeddings")
    if embeddings["present"]:
        print(
            f"  {embeddings['vectors']:,} vectors from {embeddings['model']} "
            f"({embeddings['dim']}-dim)"
        )
    else:
        print(
            "  none. Dense retrieval is unavailable on this index; build vectors "
            "with `codelearner index <repo> --embed --force`."
        )
    return 0
