"""Mined retrieval gold: the mining rules, and the three ways this set could be a lie.

Same standard as `test_gold_from_history.py`: every test names a rule, and deleting
the rule has to turn the test red. The fixture is a purpose-built git repo, built
fresh per test, because real history is not reproducible -- the shipped gold files are
snapshots of six repos at six shas and nothing here asserts against their contents
beyond the structural guarantees that must hold at any sha.

Three failures would make this gold set worse than no gold set, so each gets a test
that can be seen to fail rather than a test that passes vacuously:

* **A query attributed to a symbol its commit never touched.** The mention rule alone
  would mint one for every `foo` in the repo the moment a message says `foo`.
  `test_file_touch_attribution_excludes_an_untouched_symbol` plants exactly that.

* **A blind row with two contradictory answers.** Two sentences that differ only in
  the identifier collapse to the same text once blinded, and a retriever is then
  scored wrong on one of them whatever it returns.

* **A query pointing at a qualname the index does not hold.** Unanswerable, and it
  reads as a retrieval failure. The validator's ALL-OR-NOTHING behaviour is the part
  worth pinning: silently narrowing a two-target query to its one surviving target
  would change the question without saying so.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from codelearner.eval import mined_gold as mg
from codelearner.eval.ablation import GOLD_DIR
from codelearner.eval.mined_gold import (
    MAX_RELEVANT,
    REJECT_COPIED_INTO_SOURCE,
    REJECT_NAME_ONLY,
    REJECT_NOT_IN_INDEX,
    REJECT_TOO_SHORT,
    SOURCE_NAME_BLIND,
    SOURCE_VERBATIM,
    blind_query,
    index_qualnames,
    iter_commits,
    looks_like_code,
    mine_queries,
    source_overlap,
    strip_code_blocks,
    symbol_bias,
    to_gold_json,
    validate_against_index,
)

# A clause that exists only in commit prose in the fixture, never in a source file.
# The copy tests plant it on both sides to show the detector can fire.
COPIED_CLAUSE = "the reaper reclaims a lease whose heartbeat stopped arriving"


def _git(repo, *args):
    # S603/S607: fixed argument vector, no shell, `git` from PATH -- the same trade the
    # indexer documents. The only interpolated value is a pytest tmp_path.
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
    )


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c", "user.name=Fixture",
        "-c", "user.email=fixture@example.invalid",
        "commit", "-q", "--no-gpg-sign", "-m", message,
    )


@pytest.fixture
def repo(tmp_path):
    """A git repo whose history makes one mining rule decidable per commit."""
    root = tmp_path / "fixture"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")

    leases = root / "pkg" / "leases.py"
    worktree = root / "pkg" / "worktree.py"

    # 1. Two symbols, one sentence: the multi-relevant case this module exists for.
    leases.write_text(
        '"""Leases."""\n\n\n'
        "def acquire_lease(parcel, ttl):\n"
        '    """Take it."""\n'
        "    return True\n\n\n"
        "def release_lease(parcel):\n"
        '    """Drop it."""\n'
        "    return False\n"
    )
    _commit(
        root,
        "Leases become symmetric\n\n"
        "`acquire_lease` and `release_lease` are now a matched pair, so a caller that "
        "takes a parcel always has a way to give it back without waiting for expiry.\n",
    )

    # 2. A commit touching only worktree.py that NAMES a leases.py symbol. The mention
    #    rule alone would attribute it; the file-touch rule must not.
    worktree.write_text(
        '"""Worktrees."""\n\n\n'
        "def prune_stale_worktree(path):\n"
        '    """Remove it."""\n'
        "    return path\n"
    )
    _commit(
        root,
        "Prune worktrees left behind by a crash\n\n"
        "`prune_stale_worktree` removes a directory whose owning process is gone, "
        "which is the same liveness question `acquire_lease` answers for parcels.\n",
    )

    # 3. The commit sentence is copied verbatim into the symbol's own docstring.
    leases.write_text(
        leases.read_text()
        + "\n\n"
        "def reap_expired(now):\n"
        f'    """{COPIED_CLAUSE}."""\n'
        "    return now\n"
    )
    _commit(
        root,
        "Reclaim leases whose owner died\n\n"
        f"`reap_expired` is the sweep where {COPIED_CLAUSE}, so a parcel does not stay "
        "locked forever behind a process that is no longer running.\n",
    )

    # 4. Two units in one body: one too short to be a query at all, one long enough
    #    but made entirely of its target's own name tokens.
    leases.write_text(
        leases.read_text()
        + "\n\n"
        "def bind_managed_root(path):\n"
        '    """Bind."""\n'
        "    return path\n"
    )
    _commit(
        root,
        "Bind the root\n\n"
        "`bind_managed_root` now binds.\n\n"
        "The managed root bind that `bind_managed_root` does here.\n",
    )

    return root


def test_a_sentence_naming_two_symbols_becomes_one_multi_relevant_query(repo):
    """Multi-relevant queries come from prose, not from padding a single-target one."""
    report = mine_queries(repo)
    multi = [c for c in report.usable if len(c.relevant) > 1]
    assert multi, "the fixture's first commit describes two symbols in one sentence"
    assert {"pkg.leases.acquire_lease", "pkg.leases.release_lease"} <= set(multi[0].relevant)
    assert report.multi_relevant == len(multi)


def test_file_touch_attribution_excludes_an_untouched_symbol(repo):
    """A commit may only mint queries for symbols in files it changed.

    Commit 2 touches `worktree.py` and names `acquire_lease` in its prose. Without the
    file-touch rule that sentence would carry `pkg.leases.acquire_lease` as gold -- a
    query about worktree pruning whose correct answer is a lease function.
    """
    report = mine_queries(repo)
    for cand in report.usable:
        if "prune_stale_worktree" in cand.query:
            assert cand.relevant == ["pkg.worktree.prune_stale_worktree"]
            break
    else:
        pytest.fail("the worktree commit produced no query at all")


def test_prose_copied_into_the_symbols_own_source_is_rejected(repo):
    """Inherited from `gold_from_history`: a query sitting in its answer is not gold.

    Guarded against passing vacuously -- the same clause is asserted to be present in
    the source, so a detector that never fires cannot read as a clean result.
    """
    report = mine_queries(repo)
    assert COPIED_CLAUSE in (repo / "pkg" / "leases.py").read_text()
    rejected = [c for c in report.candidates if c.reject == REJECT_COPIED_INTO_SOURCE]
    assert rejected, "the planted copy was not detected"
    assert all(COPIED_CLAUSE in c.query for c in rejected)
    assert not any("reap_expired" in c.query for c in report.usable)


def test_a_query_that_is_only_its_targets_name_is_rejected(repo):
    """`REJECT_NAME_ONLY`: nothing survives blinding, so the pair would degenerate."""
    report = mine_queries(repo)
    name_only = [c for c in report.candidates if c.reject == REJECT_NAME_ONLY]
    assert any("bind_managed_root" in c.query for c in name_only)


def test_merge_commits_are_not_mined(repo, tmp_path):
    """A merge's message describes a branch and its file list is the whole branch."""
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "pkg" / "extra.py").write_text("def side_helper():\n    return 1\n")
    _commit(repo, "Add a side helper\n\n`side_helper` exists to be merged in a moment.\n")
    _git(repo, "checkout", "-q", "main")
    _git(
        repo,
        "-c", "user.name=Fixture",
        "-c", "user.email=fixture@example.invalid",
        "merge", "-q", "--no-ff", "--no-gpg-sign",
        "-m", "Merge side\n\n`side_helper` arrives on main here and is now available.\n",
        "side",
    )
    shas = {c.sha for c in iter_commits(repo)}
    subjects = {c.subject for c in iter_commits(repo)}
    assert "Merge side" not in subjects
    assert shas, "the non-merge history is still there"


def test_validation_drops_a_whole_query_when_one_target_is_missing():
    """All-or-nothing: a two-target query is never silently narrowed to one target."""
    report = mg.MinedGoldReport(repo="fixture")
    report.considered = 1
    report.candidates.append(
        mg.QueryCandidate(
            query="the pair of functions that take and give back a parcel",
            relevant=["pkg.leases.acquire_lease", "pkg.leases.gone"],
            paths=["pkg/leases.py", "pkg/leases.py"],
            commit="0" * 40,
            subject="s",
            unit_index=0,
        )
    )
    missing = validate_against_index(report, {"pkg.leases.acquire_lease"})
    assert missing == ["pkg.leases.gone"]
    assert report.candidates[0].reject == REJECT_NOT_IN_INDEX
    assert report.usable == []


def test_blind_query_removes_every_targets_name_not_only_the_first():
    """A two-target query blinded per symbol would still name the other target."""
    blinded = blind_query(
        "acquire_lease and release_lease are now a matched pair",
        ["pkg.leases.acquire_lease", "pkg.leases.release_lease"],
        ["pkg/leases.py", "pkg/leases.py"],
    )
    assert "acquire" not in blinded
    assert "release" not in blinded
    assert "matched pair" in blinded


def test_an_ambiguous_blind_row_is_dropped_and_its_verbatim_row_is_kept(tmp_path):
    """Two sentences differing only in the identifier blind to one contradictory query."""
    root = tmp_path / "amb"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    audio = root / "pkg" / "audio.py"
    audio.write_text(
        "def replace_audio(x):\n    return x\n\n\ndef restore_audio(x):\n    return x\n"
    )
    _commit(root, "Fix the first one\n\nFix a blank screen shown in `replace_audio` now.\n")
    audio.write_text(audio.read_text() + "\n\ndef unrelated(x):\n    return x\n")
    _commit(root, "Fix the second one\n\nFix a blank screen shown in `restore_audio` now.\n")

    report = mine_queries(root)
    pair = [c for c in report.usable if "blank screen" in c.query]
    assert len(pair) == 2, "both sentences should survive as verbatim queries"
    assert {c.blinded() for c in pair} == {"fix a blank screen shown in now"}
    assert report.blind_rows_dropped == 2

    gold = to_gold_json(report, {})
    blind_rows = [q for q in gold["queries"] if q["source"] == SOURCE_NAME_BLIND]
    assert not any("blank screen" in q["query"] for q in blind_rows)
    verbatim = [q for q in gold["queries"] if q["source"] == SOURCE_VERBATIM]
    assert sum(1 for q in verbatim if "blank screen" in q["query"]) == 2


def test_fenced_code_is_stripped_but_indented_prose_survives():
    """The indented-block stripper was deleting English; fences are the real snippets."""
    prose = (
        "Here is the usage:\n\n"
        "```python\nclient = ClobClient(host, key=key)\n```\n\n"
        "    a continuation paragraph indented four spaces, which is prose\n"
    )
    out = strip_code_blocks(prose)
    assert "ClobClient(host" not in out
    assert "continuation paragraph" in out


def test_looks_like_code_does_not_fire_on_dense_prose():
    """Regression on a tuned constant: an absolute punctuation count rejected prose.

    The first version rejected 18 of swarm-sync's 121 candidates and not one of them
    was a snippet. The line below is a real one of those 18.
    """
    prose = "reconcile_orphaned_integrations(): reads the projection -- O(open), not O(history)."
    assert not looks_like_code(prose)
    assert looks_like_code("order = OrderArgs(price=0.0005, size=20, side=BUY)")


def test_source_overlap_is_the_query_side_fraction():
    """1.0 means BM25 can match every content word of the query against the code."""
    assert source_overlap("acquire the parcel lease", ["def acquire(parcel, lease): ..."]) == 1.0
    assert source_overlap("acquire the parcel lease", ["def unrelated(): ..."]) == 0.0
    assert source_overlap("", ["anything"]) == 0.0


def test_a_sentence_naming_too_many_symbols_is_rejected(repo):
    """An enumeration is not a description, and MAX_RELEVANT is where prose stops."""
    many = repo / "pkg" / "many.py"
    many.write_text(
        "".join(f"def widget_{i}(x):\n    return x\n\n\n" for i in range(MAX_RELEVANT + 1))
    )
    _commit(
        root := repo,
        "Add the widgets\n\nNew helpers: "
        + ", ".join(f"`widget_{i}`" for i in range(MAX_RELEVANT + 1))
        + " each wrap a value and return it unchanged for now.\n",
    )
    assert root
    report = mine_queries(repo)
    assert any(c.reject == mg.REJECT_TOO_MANY for c in report.candidates)
    assert all(len(c.relevant) <= MAX_RELEVANT for c in report.usable)


def test_short_sentences_are_rejected(repo):
    """A four-word mention is not a query. Guarded: the fixture plants one."""
    report = mine_queries(repo)
    short = [c for c in report.candidates if c.reject == REJECT_TOO_SHORT]
    assert any("now binds" in c.query for c in short)
    assert all(len(c.query.split()) >= mg.MIN_QUERY_WORDS for c in report.usable)


def test_bias_reports_population_and_selection_not_only_selection(repo):
    """A gold set's documented rate is uninterpretable without the repo's own."""
    report = mine_queries(repo)
    bias = symbol_bias(repo, report)
    assert bias["population"]["n"] > bias["selected"]["n"] > 0
    assert set(bias["over_representation"]) <= set(bias["population"]["kinds"])
    for kind, ratio in bias["over_representation"].items():
        expected = bias["selected"]["kinds"].get(kind, 0.0) / bias["population"]["kinds"][kind]
        assert ratio == pytest.approx(expected, abs=1e-3)


def test_gold_json_is_a_superset_of_the_hand_labelled_schema(repo):
    """An existing loader must read a mined file without knowing about mining."""
    report = mine_queries(repo)
    gold = to_gold_json(report, symbol_bias(repo, report))
    hand = json.loads((GOLD_DIR / "swarm_sync.json").read_text())
    assert set(hand) <= set(gold), "the hand set's keys must all be present"
    for row in gold["queries"]:
        assert row["query"].strip()
        assert row["relevant"] and all(isinstance(q, str) for q in row["relevant"])
        assert row["source"] in (SOURCE_VERBATIM, SOURCE_NAME_BLIND)
        assert row["repo"] == repo.name
        assert 0.0 <= row["source_overlap"] <= 1.0
    assert gold["funnel"]["usable_queries"] == len(report.usable)
    assert gold["bias"]["population"]["n"] > 0


def test_index_qualnames_excludes_tests_by_default(tmp_path):
    """Gold may only point where the default index actually holds symbols."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, is_test INTEGER);"
        "CREATE TABLE symbols (id INTEGER PRIMARY KEY, file_id INTEGER, qualname TEXT);"
        "INSERT INTO files VALUES (1, 0), (2, 1);"
        "INSERT INTO symbols VALUES (1, 1, 'pkg.a'), (2, 2, 'tests.b');"
    )
    assert index_qualnames(conn) == {"pkg.a"}
    assert index_qualnames(conn, include_tests=True) == {"pkg.a", "tests.b"}


# --------------------------------------------------------------------------------
# The shipped snapshots. Structural only -- their CONTENTS move with every commit
# made to the mined repos, so asserting on a count here would be asserting on
# someone else's working tree.
# --------------------------------------------------------------------------------


def _shipped():
    return sorted(GOLD_DIR.glob("mined_*.json"))


def test_shipped_gold_files_carry_their_provenance_and_their_bias():
    files = _shipped()
    assert files, "no mined gold files are shipped"
    for path in files:
        gold = json.loads(path.read_text())
        assert gold["source"] == "mined"
        assert len(gold["mined_at_head"]) == 40, f"{path.name} has no sha"
        assert gold["labelling_rule"] and gold["commit_note"]
        assert gold["bias"]["population"]["n"] >= gold["bias"]["selected"]["n"] > 0
        assert gold["source_overlap"]["mined"]["n"] == gold["funnel"]["usable_queries"]
        assert gold["queries"], f"{path.name} ships no queries"


def test_shipped_gold_rows_are_internally_consistent():
    for path in _shipped():
        gold = json.loads(path.read_text())
        pairs: dict[str, list[dict]] = {}
        ids: set[str] = set()
        for row in gold["queries"]:
            # Explicit, because a loader that slugs the query text collides on two
            # sentences of one commit that share a long prefix. See `to_gold_json`.
            assert row["id"] not in ids, f"{path.name}: duplicate id {row['id']}"
            ids.add(row["id"])
            assert row["id"].startswith(row["pair_id"] + ":"), path.name
            assert row["query"].strip(), path.name
            assert row["relevant"], path.name
            assert len(set(row["relevant"])) == len(row["relevant"]), path.name
            assert row["n_relevant"] == len(row["relevant"]), path.name
            assert row["name_bearing"] is (row["source"] == SOURCE_VERBATIM)
            pairs.setdefault(row["pair_id"], []).append(row)
        for rows in pairs.values():
            # One or two rows -- two normally, one where the blind row was ambiguous.
            assert 1 <= len(rows) <= 2, path.name
            assert len({tuple(r["relevant"]) for r in rows}) == 1, path.name
        emitted = gold["funnel"]["rows_emitted"]
        assert emitted == len(gold["queries"]) == 2 * len(pairs) - gold["funnel"][
            "blind_rows_dropped"
        ], path.name


def test_shipped_gold_files_leak_no_absolute_paths():
    """A shipped artifact that records someone's home directory leaks the wrong thing."""
    for path in _shipped():
        text = path.read_text()
        assert "/home/" not in text, path.name
        assert "\\Users\\" not in text, path.name
