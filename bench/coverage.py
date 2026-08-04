"""Coverage metrics for code-learner and codegraph, computed the SAME way on both.

The point of this module is one row of the report and nothing else: an
apples-to-apples coverage comparison. It exists because the two projects publish
numbers under the same word and they measure different things.

codegraph's README reports a *resolution rate* whose denominator is FILE-LEVEL and
INBOUND: the share of symbol-bearing source files that have at least one resolved
cross-file dependent. code-learner reports a *resolution rate* whose denominator is
REFERENCE-LEVEL and OUTBOUND: the share of in-repo references that bound to a symbol.

A file with one incoming import and four hundred unresolved calls counts as fully
covered under the first and as 0.2% covered under the second. Reporting them side by
side as if they were the same quantity would be the single most misleading thing this
benchmark could do, so this module computes BOTH definitions against BOTH indexes and
the report prints four cells, not two.

Two asymmetries are handled explicitly rather than papered over:

*External references.* code-learner classifies an unresolved reference as `external`
when no symbol in the repo even shares its basename -- a call to `json.dumps` is not a
resolution failure. codegraph's `unresolved_refs` table carries no such distinction, so
its failures lump the standard library in with genuinely-missed in-repo calls. This
module applies code-learner's own rule (`codelearner.cli.commands._classify_unresolved`)
to codegraph's unresolved rows, so both systems get the same denominator. The all-
references variant is reported too, because the basename rule is a heuristic and the
reader should be able to see both bounds.

*Structural edges.* codegraph's `contains` edge (file contains function) is
containment, not a reference, and code-learner has no equivalent. Counting it would
inflate codegraph's resolved numerator with edges that never had to be resolved. It is
excluded, and `DEP_KINDS` names exactly what is counted.

## The file-level metric is dominated by one definitional choice

Measured here, not argued: on swarm-sync the file-inbound rate for codegraph's own
index is **48.4%** over all symbol-bearing files and **96.3%** over non-test files.
Nothing about the resolver changed between those two numbers -- a pytest module is
imported by nothing by construction, so every test file is a guaranteed miss, and the
rate ends up reporting each repo's test-to-source ratio more than its resolver. That
48-point swing is larger than any difference this benchmark could plausibly measure
between the two systems, so both variants are always printed and neither is called
"the" coverage number. The published 86-100% figures are consistent with a non-test
denominator; that is an inference from the arithmetic, not something the README states.

Test classification uses `codelearner.ingest.indexer.is_test_path` for BOTH systems.
Reaching into one project's helper to judge the other is deliberate: the alternative is
two heuristics, and then the split itself becomes a place the comparison could be
tilted. Both indexes store repo-root-relative POSIX paths and they match byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from codelearner.ingest.indexer import is_test_path

#: codegraph edge kinds that represent a REFERENCE from one symbol to another.
#: `contains` is deliberately absent -- it is structural containment, always
#: trivially "resolved", and has no counterpart in code-learner's edge table.
DEP_KINDS = ("calls", "references", "imports", "instantiates", "extends", "decorates")

#: What counts as a symbol for the "symbol-bearing file" denominator. codegraph also
#: emits `variable` and `import` nodes and code-learner emits a `module` node per file;
#: none of those are the kind of thing either README means by "symbol", and including
#: them would move the denominator for one system only.
SYMBOL_KINDS = ("function", "class", "method")


@dataclass
class Coverage:
    """Both denominators for one index, plus the raw counts behind each."""

    system: str
    repo: str
    index_path: str

    # -- reference-level, outbound (code-learner's published denominator) ----------
    refs_total: int = 0
    refs_resolved: int = 0
    refs_external: int = 0
    refs_in_repo: int = 0
    #: resolved / in_repo -- the headline. NOT comparable to `file_inbound_rate`.
    ref_resolution_rate: float = 0.0
    #: resolved / total, including references to code outside the repo. The lower
    #: bound, reported so the basename heuristic is visible rather than load-bearing.
    ref_resolution_rate_all: float = 0.0

    # -- file-level, inbound (codegraph's published denominator) -------------------
    files_total: int = 0
    files_symbol_bearing: int = 0
    files_with_inbound: int = 0
    file_inbound_rate: float = 0.0
    #: Same metric with test files dropped from the denominator. A test module is
    #: imported by nothing, so it is a guaranteed miss and its share of the repo, not
    #: the resolver, decides the headline. See the module docstring.
    files_symbol_bearing_nontest: int = 0
    files_with_inbound_nontest: int = 0
    file_inbound_rate_nontest: float = 0.0

    notes: list[str] = field(default_factory=list)


def _rate(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _file_level(bearing: set[str], with_inbound: set[str], cov: Coverage) -> None:
    """Fill the four file-level fields from two path sets, for either system."""
    hit = bearing & with_inbound
    cov.files_symbol_bearing = len(bearing)
    cov.files_with_inbound = len(hit)
    cov.file_inbound_rate = _rate(len(hit), len(bearing))

    bearing_nt = {p for p in bearing if not is_test_path(p)}
    hit_nt = bearing_nt & with_inbound
    cov.files_symbol_bearing_nontest = len(bearing_nt)
    cov.files_with_inbound_nontest = len(hit_nt)
    cov.file_inbound_rate_nontest = _rate(len(hit_nt), len(bearing_nt))


# ---------------------------------------------------------------------------------
# codegraph
# ---------------------------------------------------------------------------------


def codegraph_coverage(repo: Path) -> Coverage:
    """Both metrics over a `.codegraph/codegraph.db`.

    The resolved numerator is dependency edges whose target is a real definition.
    Edges into codegraph's `import` nodes are counted separately and excluded: an
    import node stands for the import *statement*, so an edge landing there is a
    reference that reached the local binding rather than the definition, and treating
    it as resolved would credit codegraph for a hop it did not make. The count is
    reported in `notes` so the choice is auditable rather than silent.
    """
    db = repo / ".codegraph" / "codegraph.db"
    cov = Coverage(system="codegraph", repo=str(repo), index_path=str(db))
    if not db.exists():
        cov.notes.append(f"no codegraph index at {db}")
        return cov

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(DEP_KINDS))

        # Resolved dependency edges, split by whether the target is a definition.
        rows = conn.execute(
            f"SELECT n.kind AS target_kind, COUNT(*) AS c "  # noqa: S608 -- kinds are module constants
            f"FROM edges e JOIN nodes n ON n.id = e.target "
            f"WHERE e.kind IN ({placeholders}) GROUP BY 1",
            DEP_KINDS,
        ).fetchall()
        by_target_kind = {r["target_kind"]: r["c"] for r in rows}
        into_imports = by_target_kind.get("import", 0)
        resolved = sum(c for k, c in by_target_kind.items() if k != "import")

        # code-learner's own external rule, applied to codegraph's failures: a
        # reference is external when no node in this index shares its basename.
        names = {r[0] for r in conn.execute("SELECT DISTINCT name FROM nodes")}
        external = ambiguous = 0
        for (ref_name,) in conn.execute("SELECT reference_name FROM unresolved_refs"):
            base = str(ref_name).rsplit(".", 1)[-1]
            if base in names:
                ambiguous += 1
            else:
                external += 1

        cov.refs_resolved = resolved
        cov.refs_total = resolved + external + ambiguous
        cov.refs_external = external
        cov.refs_in_repo = resolved + ambiguous
        cov.ref_resolution_rate = _rate(resolved, cov.refs_in_repo)
        cov.ref_resolution_rate_all = _rate(resolved, cov.refs_total)
        cov.notes.append(
            f"{into_imports} dependency edges target an `import` node (the local "
            f"binding, not a definition) and are excluded from the numerator"
        )

        # File-level inbound.
        cov.files_total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        kind_ph = ",".join("?" * len(SYMBOL_KINDS))
        bearing = {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT file_path FROM nodes WHERE kind IN ({kind_ph})",  # noqa: S608
                SYMBOL_KINDS,
            )
        }
        with_inbound = {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT tgt.file_path "  # noqa: S608
                f"FROM edges e "
                f"JOIN nodes src ON src.id = e.source "
                f"JOIN nodes tgt ON tgt.id = e.target "
                f"WHERE e.kind IN ({placeholders}) "
                f"AND tgt.kind IN ({kind_ph}) "
                f"AND src.file_path != tgt.file_path",
                (*DEP_KINDS, *SYMBOL_KINDS),
            )
        }
        _file_level(bearing, with_inbound, cov)
    finally:
        conn.close()
    return cov


# ---------------------------------------------------------------------------------
# code-learner
# ---------------------------------------------------------------------------------


def codelearner_coverage(repo: Path, index_path: Path | None = None) -> Coverage:
    """Both metrics over a `.codelearner/index.db`.

    The reference-level half reproduces `codelearner stats` from the stored graph
    rather than importing it, so this module can be pointed at any index file --
    including the throwaway ones the timing runs build -- without the CLI's repo
    conventions. The file-level half is new: code-learner has never published it,
    which is the whole reason it has to be computed here.
    """
    db = index_path or (repo / ".codelearner" / "index.db")
    cov = Coverage(system="code-learner", repo=str(repo), index_path=str(db))
    if not db.exists():
        cov.notes.append(f"no code-learner index at {db}")
        return cov

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cov.refs_total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        cov.refs_resolved = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE dst_symbol_id IS NOT NULL"
        ).fetchone()[0]

        names = {r[0] for r in conn.execute("SELECT DISTINCT name FROM symbols")}
        external = 0
        for (dst_name,) in conn.execute(
            "SELECT dst_name FROM edges WHERE dst_symbol_id IS NULL"
        ):
            if str(dst_name).rsplit(".", 1)[-1] not in names:
                external += 1
        cov.refs_external = external
        cov.refs_in_repo = cov.refs_total - external
        cov.ref_resolution_rate = _rate(cov.refs_resolved, cov.refs_in_repo)
        cov.ref_resolution_rate_all = _rate(cov.refs_resolved, cov.refs_total)

        cov.files_total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        kind_ph = ",".join("?" * len(SYMBOL_KINDS))
        bearing = {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT f.path FROM symbols s "  # noqa: S608
                f"JOIN files f ON f.id = s.file_id WHERE s.kind IN ({kind_ph})",
                SYMBOL_KINDS,
            )
        }
        with_inbound = {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT tf.path "  # noqa: S608
                f"FROM edges e "
                f"JOIN symbols src ON src.id = e.src_symbol_id "
                f"JOIN symbols tgt ON tgt.id = e.dst_symbol_id "
                f"JOIN files tf ON tf.id = tgt.file_id "
                f"WHERE tgt.kind IN ({kind_ph}) AND src.file_id != tgt.file_id",
                SYMBOL_KINDS,
            )
        }
        _file_level(bearing, with_inbound, cov)
    finally:
        conn.close()
    return cov


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------

_WARNING = (
    "The two columns below measure DIFFERENT THINGS and a system may lead on one and "
    "trail on the other. Read across a row, never down a column into the other metric."
)


def format_table(covs: list[Coverage]) -> str:
    """The four-cell row this module exists to print."""
    lines = [
        _WARNING,
        "",
        f"{'repo':<12} {'system':<13} "
        f"{'ref-level (outbound)':>22} | "
        f"{'file-level, all files':>22} | "
        f"{'file-level, non-test':>22}",
    ]
    for c in covs:
        repo = Path(c.repo).name
        lines.append(
            f"{repo:<12} {c.system:<13} "
            f"{c.refs_resolved:>7,}/{c.refs_in_repo:<7,} {c.ref_resolution_rate:>6.1%} | "
            f"{c.files_with_inbound:>7,}/{c.files_symbol_bearing:<7,} "
            f"{c.file_inbound_rate:>6.1%} | "
            f"{c.files_with_inbound_nontest:>7,}/{c.files_symbol_bearing_nontest:<7,} "
            f"{c.file_inbound_rate_nontest:>6.1%}"
        )
    lines += [
        "",
        "ref-level  = outbound, per reference: does this call/import bind to a symbol? "
        "Denominator drops references whose basename matches nothing in the repo "
        "(stdlib, third party) -- code-learner's own `external` rule, applied to both.",
        "file-level = inbound, per file: does ANY other file resolve into this one? One "
        "resolved import makes a file with 400 unresolved calls count as covered.",
        "The two file-level columns differ only in whether test files are in the "
        "denominator. A test module is imported by nothing, so it is a guaranteed miss; "
        "the gap between those columns is the repo's test ratio, not resolver quality.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repos", nargs="+", type=Path, help="repository roots to measure")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument(
        "--codelearner-index",
        type=Path,
        action="append",
        default=None,
        help="explicit code-learner index path, repeatable, positional with `repos`",
    )
    args = ap.parse_args(argv)

    covs: list[Coverage] = []
    for i, repo in enumerate(args.repos):
        idx = None
        if args.codelearner_index and i < len(args.codelearner_index):
            idx = args.codelearner_index[i]
        covs.append(codelearner_coverage(repo, idx))
        covs.append(codegraph_coverage(repo))

    if args.json:
        print(json.dumps({"warning": _WARNING, "rows": [asdict(c) for c in covs]}, indent=2))
    else:
        print(format_table(covs))
        for c in covs:
            for note in c.notes:
                print(f"  note [{Path(c.repo).name}/{c.system}]: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
