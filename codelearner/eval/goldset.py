"""Multi-source gold sets: provenance, validation, and the strata scoring must respect.

The 16-query hand-labelled set was one file from one repo written by one person, so a
gold "set" could be a bare list of queries and nothing was lost. Enlarging it breaks
that. Gold now arrives from at least three places -- hand-written questions, questions
mined from commit prose, and (eventually) other repos -- and those three have DIFFERENT
biases, not merely different sizes:

  - **hand-written** questions are written by someone who knows the codebase, so they
    use the vocabulary the code already uses. That flatters lexical retrieval.
  - **mined from commit prose** inherits the committer's vocabulary, which describes the
    CHANGE rather than the code, and is systematically further from the identifiers.
    `gold_from_history` already measured this asymmetry: mined prose scored 0.280 MRR
    against hand-written's 0.309 on the same retriever.
  - **another repo** changes the index, the naming conventions, and the density of the
    call graph all at once.

Pooling those into one mean is not an average of the same quantity measured three
times; it is a mean whose value depends on the MIX. Change the mix -- add 40 mined
queries to 16 hand-written ones -- and every number moves without the retriever
changing at all. So `GoldSet` carries `source` and `repo` on every query, scoring
produces a row per stratum, and the pooled row is labelled POOLED so that nobody reads
it as if it were a measurement of one thing.

## The schema

A gold file is JSON. File-level keys supply defaults; per-query keys override them:

```json
{
  "repo": "swarm-sync",
  "source": "handwritten",
  "commit_note": "free text: what the index looked like when this was labelled",
  "labelling_rule": "free text: what 'relevant' was taken to mean",
  "queries": [
    {
      "id": "lease-expiry",
      "query": "how does a lease expire and get reclaimed",
      "relevant": ["swarmsync.coordinator.reaper.reap_once"],
      "source": "handwritten",
      "repo": "swarm-sync",
      "notes": "free text"
    }
  ]
}
```

`query` and `relevant` are required and `relevant` must be non-empty -- a query with no
relevant symbol scores 0 on every configuration and can only drag every row down by the
same amount, which is not a measurement. `id` defaults to a stable slug of the query
text, so per-query results can be joined across runs and across configurations.

`source` and `repo` are required for new files, but a file that omits them loads with
`source="unspecified"` and the diagnostics say so out loud, because `swarm_sync.json`
predates this schema and rewriting it is not this module's business.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

GOLD_DIR = Path(__file__).parent / "gold"

#: What a file that predates the provenance schema gets. Deliberately not "handwritten":
#: guessing the provenance of someone else's labels is how a bias becomes invisible.
UNSPECIFIED_SOURCE = "unspecified"

#: How many missing qualnames an error message lists before it truncates. Enough to see
#: the pattern -- a stale module prefix, a rename -- without printing a whole gold file.
_MAX_REPORTED = 12


class GoldError(Exception):
    """Base for every way a gold set can be unusable."""


class GoldSchemaError(GoldError):
    """The file is not a gold file: missing keys, wrong types, empty `relevant`."""


class GoldIndexMismatch(GoldError):
    """The gold names symbols the index being scored against does not contain.

    This is the failure the audit found scoring silently as 0.000 across every row of
    the table, complete with a `[0.000, 0.000]` bootstrap interval that reads as
    certainty rather than as "nothing matched". A gold set and an index that disagree
    do not produce a bad measurement, they produce NO measurement, and the difference
    has to be visible.
    """

    def __init__(self, missing: Sequence[str], index_size: int, total: int) -> None:
        self.missing = list(missing)
        self.index_size = index_size
        self.total = total
        shown = ", ".join(self.missing[:_MAX_REPORTED])
        if len(self.missing) > _MAX_REPORTED:
            shown += f", ... (+{len(self.missing) - _MAX_REPORTED} more)"
        super().__init__(
            f"{len(self.missing)} of {total} gold qualnames are not in this index "
            f"({index_size} symbols): {shown}. "
            "Scoring would have returned zeros for every configuration. Either the "
            "index is stale/for another repo, or the gold refers to renamed symbols."
        )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


@dataclass(frozen=True)
class GoldQuery:
    """One labelled question, with the provenance that decides which rows it belongs in."""

    query: str
    relevant: tuple[str, ...]
    source: str = UNSPECIFIED_SOURCE
    repo: str = ""
    query_id: str = ""
    notes: str = ""

    @property
    def is_single_relevant(self) -> bool:
        return len(self.relevant) == 1


@dataclass
class GoldSet:
    """Queries plus the provenance needed to know which of them may be averaged together."""

    name: str
    queries: list[GoldQuery] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[GoldQuery]:
        return iter(self.queries)

    def sources(self) -> list[str]:
        return sorted({q.source for q in self.queries})

    def repos(self) -> list[str]:
        return sorted({q.repo for q in self.queries})

    def subset(self, *, source: str | None = None, repo: str | None = None) -> GoldSet:
        picked = [
            q
            for q in self.queries
            if (source is None or q.source == source) and (repo is None or q.repo == repo)
        ]
        bits = [b for b in (source, repo) if b]
        return GoldSet(
            name=f"{self.name}[{'/'.join(bits)}]" if bits else self.name,
            queries=picked,
            files=list(self.files),
        )

    # -- the diagnostics that explain why two metrics were duplicates ----------
    def single_relevant(self) -> int:
        """Queries with exactly ONE relevant symbol.

        This is the quantity that made `recall@k` and `hit@k` the same measurement on
        the old set: when a query has one relevant symbol, `recall@k` can only be 0 or
        1, which is precisely what `hit@k` is. 11 of 16 meant two thirds of the table's
        two recall columns were a restatement of its hit column.
        """
        return sum(1 for q in self.queries if q.is_single_relevant)

    def mean_relevant(self) -> float:
        if not self.queries:
            return 0.0
        return sum(len(q.relevant) for q in self.queries) / len(self.queries)

    def cluster_labels(self) -> list[str]:
        """The repo of each query, in order -- the clustering unit for the bootstrap."""
        return [q.repo for q in self.queries]

    def qualnames(self) -> set[str]:
        return {name for q in self.queries for name in q.relevant}


def parse_gold(payload: dict, *, filename: str = "<dict>") -> list[GoldQuery]:
    """Turn one loaded gold file into `GoldQuery` records, validating the schema.

    Raises rather than skipping a malformed entry: a gold file that silently drops
    three of its queries changes `n`, and `n` is the denominator of every number the
    table prints.
    """
    if not isinstance(payload, dict):
        raise GoldSchemaError(f"{filename}: top level must be an object")
    raw = payload.get("queries")
    if not isinstance(raw, list) or not raw:
        raise GoldSchemaError(f"{filename}: 'queries' must be a non-empty list")

    file_source = payload.get("source", UNSPECIFIED_SOURCE)
    file_repo = payload.get("repo", "")
    out: list[GoldQuery] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        where = f"{filename}[{i}]"
        if not isinstance(entry, dict):
            raise GoldSchemaError(f"{where}: each query must be an object")
        text = entry.get("query")
        if not isinstance(text, str) or not text.strip():
            raise GoldSchemaError(f"{where}: 'query' must be a non-empty string")
        relevant = entry.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise GoldSchemaError(
                f"{where}: 'relevant' must be a non-empty list. A query with no "
                "relevant symbol scores 0 for every configuration and measures nothing."
            )
        if not all(isinstance(r, str) and r for r in relevant):
            raise GoldSchemaError(f"{where}: every entry in 'relevant' must be a qualname string")
        if len(set(relevant)) != len(relevant):
            raise GoldSchemaError(
                f"{where}: 'relevant' repeats a qualname, which would inflate recall's "
                "denominator without any retriever being able to satisfy it twice"
            )
        qid = entry.get("id") or _slug(text)
        if qid in seen_ids:
            raise GoldSchemaError(f"{where}: duplicate query id {qid!r}")
        seen_ids.add(qid)
        out.append(
            GoldQuery(
                query=text,
                relevant=tuple(relevant),
                source=str(entry.get("source", file_source)),
                repo=str(entry.get("repo", file_repo)),
                query_id=qid,
                notes=str(entry.get("notes", "")),
            )
        )
    return out


def load_gold_set(
    names: str | Iterable[str] = "swarm_sync",
    *,
    gold_dir: Path | None = None,
) -> GoldSet:
    """Load one or more gold files into a single provenance-carrying set.

    Loading several files pools them into one `GoldSet`, which is safe because nothing
    downstream averages a `GoldSet` without first asking it for its strata. The pooling
    that is NOT safe -- reporting one mean over sources whose biases differ -- is
    prevented at the reporting layer, where the pooled row is labelled as pooled.
    """
    directory = gold_dir or GOLD_DIR
    if isinstance(names, str):
        names = [names]
    wanted = list(names)
    if not wanted:
        raise GoldSchemaError("no gold files requested")

    queries: list[GoldQuery] = []
    files: list[str] = []
    for name in wanted:
        path = directory / f"{name}.json"
        if not path.exists():
            available = sorted(p.stem for p in directory.glob("*.json"))
            raise GoldSchemaError(
                f"no gold file {name!r} in {directory}. Available: {available or 'none'}"
            )
        payload = json.loads(path.read_text())
        queries.extend(parse_gold(payload, filename=path.name))
        files.append(path.name)

    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise GoldSchemaError(
            f"query ids collide across {files}: {dupes[:_MAX_REPORTED]}. Two files "
            "labelling the same question would double-weight it in every mean."
        )
    return GoldSet(name="+".join(wanted), queries=queries, files=files)


def index_qualnames(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT qualname FROM symbols")}


def validate_against_index(gold: GoldSet, conn: sqlite3.Connection) -> None:
    """Fail loudly when the gold and the index disagree, instead of scoring zeros.

    `run_ablation` used to score whatever it was handed. A gold set naming symbols the
    index does not contain produced a full table of 0.000 with `[0.000, 0.000]`
    intervals -- output that looks like a finished measurement and is in fact the
    absence of one. There is no configuration of a retriever that can find a symbol
    that is not indexed, so this is never a result about retrieval.
    """
    if not gold.queries:
        raise GoldSchemaError(f"gold set {gold.name!r} is empty")
    present = index_qualnames(conn)
    if not present:
        raise GoldIndexMismatch(sorted(gold.qualnames()), 0, len(gold.qualnames()))
    wanted = gold.qualnames()
    missing = sorted(wanted - present)
    if missing:
        raise GoldIndexMismatch(missing, len(present), len(wanted))
