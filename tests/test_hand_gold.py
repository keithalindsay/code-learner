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
* the KIND checks, which apply only to `hand_tests_*.json` -- the test-seeking gold,
  where the correct answer IS a test symbol. Those files are the only place the
  evaluation can see test-seeking retrieval at all, and a file that drifted into
  naming implementations instead would keep passing every check above while quietly
  ceasing to measure the thing it was written for. So `answer_kind` is asserted
  against `files.is_test` rather than trusted, on the same skip-when-absent terms.
"""
from __future__ import annotations

import json
import re
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

#: The TEST-SEEKING subset: gold whose correct answer IS a test symbol. A strict subset
#: of HAND_GOLD_FILES (the glob above matches them too), so they get every shape and
#: resolution check in this module and then the extra ones below.
#:
#: They exist because the rest of the gold could not see a whole capability die. Across
#: every other gold file in this repo, 0 of 978 relevant labels is a test symbol, so an
#: experiment that dropped tests from the embedding corpus scored a clean, significant
#: nDCG win while taking test-seeking retrieval from 100% to 0%. A metric that improves
#: while a capability silently disappears is the failure this project exists to prevent.
TEST_SEEKING_FILES = sorted(GOLD_DIR.glob("hand_tests_*.json"))

#: What `answer_kind` may say. DERIVED from `files.is_test` over `relevant`, never
#: asserted by hand -- see `test_answer_kind_agrees_with_the_index`.
KNOWN_ANSWER_KINDS = frozenset({"test_only", "test_and_impl"})


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


# ---------------------------------------------------------------------------
# The test-seeking gold: extra checks the other hand files cannot make.
#
# These files are the ONLY place in the evaluation where the right answer is a test.
# If they quietly drifted into naming implementations instead, they would keep passing
# every check above -- schema, overlap, resolution -- and stop closing the hole they
# were written for, which is the same shape of silent loss as an unresolvable qualname.
# So the property is asserted, against the index rather than against a hand-typed flag.
# ---------------------------------------------------------------------------


def test_there_is_at_least_one_test_seeking_gold_file():
    """Without this, deleting every `hand_tests_*.json` collects zero tests below.

    Same reason as `test_there_is_at_least_one_hand_gold_file`: a vacuous parametrise is
    a green run that checked nothing, and here it would also mean the measurement blind
    spot these files close had silently reopened.
    """
    assert TEST_SEEKING_FILES, f"no hand_tests_*.json files under {GOLD_DIR}"


def test_test_seeking_files_are_covered_by_the_shared_hand_checks():
    """The `hand_tests_*` prefix must stay inside the `hand_*` glob.

    Renaming these out of that glob would drop them from every schema, derived-flag and
    resolution check above while leaving the file present and apparently maintained.
    """
    assert set(TEST_SEEKING_FILES) <= set(HAND_GOLD_FILES)


@pytest.mark.parametrize("path", TEST_SEEKING_FILES, ids=lambda p: p.stem)
def test_schema_answer_kind_present(path: Path):
    """`answer_kind` is required here and meaningless elsewhere, so it is checked here."""
    gold = _load(path)
    for i, spec in enumerate(gold["queries"]):
        where = f"{path.name}[{i}] {spec.get('query', '<no query>')!r}"
        kind = spec.get("answer_kind")
        assert kind in KNOWN_ANSWER_KINDS, f"{where}: answer_kind={kind!r} not in {sorted(KNOWN_ANSWER_KINDS)}"


@pytest.mark.parametrize("path", TEST_SEEKING_FILES, ids=lambda p: p.stem)
def test_the_query_text_never_says_the_word_test(path: Path):
    """The one leak that cannot be reworded away query-by-query.

    Every pytest function is named `test_*`, so the bare token "test" is in the name of
    every symbol these files can possibly target. A query phrased as "the test for X"
    therefore names its own answer no matter how the rest of it is worded -- it would be
    scored as a clean row while being a pure lexical hit. `name_bearing` already catches
    it via `name_overlap`, but only after the fact; this states the rule.
    """
    gold = _load(path)
    for spec in gold["queries"]:
        words = re.findall(r"[a-z]+", spec["query"].lower())
        assert "test" not in words and "tests" not in words, (
            f"{path.name}: {spec['query']!r} contains the word 'test', which is a name "
            "token of EVERY possible target in this file"
        )


@pytest.mark.parametrize("path", TEST_SEEKING_FILES, ids=lambda p: p.stem)
def test_answer_kind_agrees_with_the_index(path: Path):
    """THE check these files exist for: the answers really are tests.

    `files.is_test` is derived deterministically from path conventions at index time, so
    it is a tier-0 fact and the right thing to assert against -- not a hand flag, which
    could be edited to match a drifted label set.

    Two claims, because they fail differently:

    * every query names at least one test symbol. A test-seeking file whose answers had
      all become implementations would score fine and measure nothing it was written for.
    * `answer_kind` says which shape each query is. 'test_only' means the whole relevant
      set is tests; 'test_and_impl' is the honest multi-relevant case -- someone asking
      "how is this behaviour verified" wants the code AND the case that pins it -- and it
      must genuinely span both, or it is a mislabelled pure-test row.

    Skips when the index is absent, for the same reason as the resolution check above:
    those databases live in the target repos, not in this one.
    """
    gold = _load(path)
    index_path = Path(gold["index_path"])
    if not index_path.exists():
        pytest.skip(
            f"{path.name}: index {index_path} is absent (it lives in the target "
            "repo, not in this one) -- cannot classify qualnames against it"
        )

    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        is_test = {
            row[0]: bool(row[1])
            for row in conn.execute(
                "SELECT s.qualname, f.is_test FROM symbols s JOIN files f ON f.id = s.file_id"
            )
        }
    finally:
        conn.close()

    assert any(is_test.values()), (
        f"{path.name}: {index_path} has no test symbols at all, so this file could not "
        "be satisfied by any label -- wrong index, or an index built with tests excluded"
    )

    for spec in gold["queries"]:
        where = f"{path.name}: {spec['query']!r}"
        # Unresolvable qualnames are the other test's job; skip them here so a missing
        # symbol reports as one failure there rather than two unrelated ones.
        known = [q for q in spec["relevant"] if q in is_test]
        if len(known) != len(spec["relevant"]):
            continue
        tests = [q for q in known if is_test[q]]
        impls = [q for q in known if not is_test[q]]

        assert tests, (
            f"{where}: no relevant qualname is a test symbol. This file is the only "
            "place the evaluation can see test-seeking retrieval; a query whose answers "
            f"are all implementations quietly stops closing that hole. Got: {known}"
        )
        expected = "test_only" if not impls else "test_and_impl"
        assert spec["answer_kind"] == expected, (
            f"{where}: answer_kind={spec['answer_kind']!r} but the index says "
            f"{expected!r} ({len(tests)} test symbol(s), {len(impls)} implementation(s): "
            f"{impls})"
        )
