"""The hand-written gold sets must name symbols the index actually has.

`run_ablation` scores a query by membership: `qualname in spec["relevant"]`. A gold
entry naming a symbol that is not in the index can therefore never be retrieved, so
the query contributes a silent zero to every metric on every row. A whole gold FILE
pointed at the wrong repo produces an all-zeros table that looks like a measurement
and is actually a typo -- that is the failure this module exists to catch, and it is
the one an audit already found once.

Two tiers of check, deliberately:

* the SHAPE checks (`test_schema_*`) need no index and run everywhere. They pin the
  superset schema the multi-source loader consumes -- `source`/`repo` per query, the
  measured `overlap`, the `name_bearing` flag -- so a hand file that drifts out of
  that shape fails here rather than in whatever reads it next.
* the RESOLUTION check (`test_every_relevant_qualname_exists_in_the_index`) needs the
  repo's `.codelearner/index.db`. Those live outside this repo and are not fixtures,
  so the test SKIPS when the index is absent and FAILS when it is present and a
  qualname is missing. Skipping on absence is the only honest option (we cannot
  validate against a database that is not there); passing on absence would defeat the
  point, so the skip reason names the index it wanted.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from codelearner.eval.ablation import GOLD_DIR

#: Coverage codes a hand gold file may use. Anything else is a typo, and a typo in a
#: coverage tag silently removes a query from whatever slice the tag was meant to
#: define -- the same class of quiet loss as an unresolvable qualname.
KNOWN_COVERAGE_CODES = frozenset(
    {
        "undocumented",
        "private",
        "generic_name",
        "cross_module",
        "error_path",
        "hard_negative",
    }
)

REQUIRED_OVERLAP_KEYS = frozenset(
    {
        "name_overlap",
        "source_overlap",
        "rare_source_overlap",
        "idf_weighted_source_overlap",
    }
)

HAND_GOLD_FILES = sorted(GOLD_DIR.glob("hand_*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_there_is_at_least_one_hand_gold_file():
    """A glob that matches nothing makes every parametrised test below vacuous.

    Without this, deleting or renaming every `hand_*.json` turns the rest of this
    module into zero collected tests -- a green run that checked nothing.
    """
    assert HAND_GOLD_FILES, f"no hand_*.json files under {GOLD_DIR}"


@pytest.mark.parametrize("path", HAND_GOLD_FILES, ids=lambda p: p.stem)
def test_schema_top_level(path: Path):
    """The file-level keys the multi-source loader and the ablation both read."""
    gold = _load(path)
    for key in ("source", "repo", "repo_path", "index_path", "commit_note",
                "labelling_rule", "queries"):
        assert key in gold, f"{path.name}: missing top-level key {key!r}"
    assert gold["source"] == "hand", (
        f"{path.name}: `source` must be 'hand' for a hand_*.json file "
        f"(got {gold['source']!r}) -- the loader routes on this, so a wrong value "
        "silently files these queries under the mined set"
    )
    assert gold["queries"], f"{path.name}: `queries` is empty"


@pytest.mark.parametrize("path", HAND_GOLD_FILES, ids=lambda p: p.stem)
def test_schema_per_query(path: Path):
    """Every query carries its provenance, its measurement, and a non-empty label set."""
    gold = _load(path)
    seen: set[str] = set()
    for i, spec in enumerate(gold["queries"]):
        where = f"{path.name}[{i}] {spec.get('query', '<no query>')!r}"

        assert spec.get("query", "").strip(), f"{where}: empty query text"
        assert spec["query"] not in seen, (
            f"{where}: duplicate query text -- two identical queries are one query "
            "scored twice, which quietly doubles its weight in every mean"
        )
        seen.add(spec["query"])

        assert spec.get("relevant"), f"{where}: `relevant` is empty"
        assert len(set(spec["relevant"])) == len(spec["relevant"]), (
            f"{where}: duplicate qualname in `relevant` -- recall@k divides by "
            "len(relevant), so a duplicate deflates the score for a symbol that "
            "can only be retrieved once"
        )

        assert spec.get("source") == gold["source"], f"{where}: per-query `source` mismatch"
        assert spec.get("repo") == gold["repo"], f"{where}: per-query `repo` mismatch"

        assert isinstance(spec.get("hard_negative"), bool), f"{where}: `hard_negative` missing"
        assert isinstance(spec.get("name_bearing"), bool), f"{where}: `name_bearing` missing"

        unknown = set(spec.get("coverage", [])) - KNOWN_COVERAGE_CODES
        assert not unknown, f"{where}: unknown coverage code(s) {sorted(unknown)}"

        overlap = spec.get("overlap")
        assert isinstance(overlap, dict), f"{where}: `overlap` missing"
        missing = REQUIRED_OVERLAP_KEYS - set(overlap)
        assert not missing, f"{where}: overlap missing {sorted(missing)}"
        for key in REQUIRED_OVERLAP_KEYS:
            value = overlap[key]
            assert isinstance(value, (int, float)), f"{where}: overlap[{key!r}] not numeric"
            assert 0.0 <= value <= 1.0, f"{where}: overlap[{key!r}]={value} outside [0,1]"


@pytest.mark.parametrize("path", HAND_GOLD_FILES, ids=lambda p: p.stem)
def test_name_bearing_flag_agrees_with_the_measurement(path: Path):
    """`name_bearing` is DERIVED from `name_overlap`, so it cannot be asserted by hand.

    The flag exists so name-leaking queries can be scored as their own row rather
    than silently inflating the lexical modality. A flag that disagreed with the
    number it summarises would put a leaky query back in the clean bucket, which is
    exactly the inflation it was added to prevent.
    """
    gold = _load(path)
    for spec in gold["queries"]:
        expected = spec["overlap"]["name_overlap"] > 0
        assert spec["name_bearing"] is expected, (
            f"{path.name}: {spec['query']!r} has name_overlap="
            f"{spec['overlap']['name_overlap']} but name_bearing={spec['name_bearing']}"
        )


@pytest.mark.parametrize("path", HAND_GOLD_FILES, ids=lambda p: p.stem)
def test_hard_negative_flag_agrees_with_the_coverage_tag(path: Path):
    gold = _load(path)
    for spec in gold["queries"]:
        tagged = "hard_negative" in spec.get("coverage", [])
        assert spec["hard_negative"] is tagged, (
            f"{path.name}: {spec['query']!r} hard_negative={spec['hard_negative']} "
            f"but coverage={spec.get('coverage')}"
        )


@pytest.mark.parametrize("path", HAND_GOLD_FILES, ids=lambda p: p.stem)
def test_every_relevant_qualname_exists_in_the_index(path: Path):
    """THE check. A qualname the index does not have can never be retrieved.

    `run_ablation` compares retrieved qualnames against `relevant` by string
    membership and has no way to tell "the retriever missed it" from "this string
    names nothing". So a stale or mistyped entry is scored as a permanent miss, and a
    file pointed at the wrong repo scores as an all-zeros table -- a number that
    reads like a finding.
    """
    gold = _load(path)
    index_path = Path(gold["index_path"])
    if not index_path.exists():
        pytest.skip(
            f"{path.name}: index {index_path} is absent (it lives in the target "
            "repo, not in this one) -- cannot validate qualnames against it"
        )

    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        known = {row[0] for row in conn.execute("SELECT qualname FROM symbols")}
    finally:
        conn.close()

    assert known, f"{path.name}: {index_path} has no symbols at all"

    missing = sorted(
        {
            qualname
            for spec in gold["queries"]
            for qualname in spec["relevant"]
            if qualname not in known
        }
    )
    assert not missing, (
        f"{path.name}: {len(missing)} qualname(s) are not in {index_path}. Each one "
        f"is an unreachable gold entry that scores a silent zero on every metric: "
        f"{missing}"
    )
