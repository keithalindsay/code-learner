"""Git-history gold labels: the mining rules, and the leak boundary that makes them gold.

Same standard as `test_assertions.py` and `test_stale.py`: every test names a rule, and
deleting the rule has to turn the test red. Three tests in this repo's history have
passed while asserting nothing, so the two rules here that are easiest to test
vacuously get an explicit guard:

* **"the generator's input never contains the answer."** A test that walks a view
  looking for label prose and finds none passes just as happily when the walker is
  broken. `test_find_leaks_catches_a_planted_leak` and
  `test_score_purposes_refuses_to_score_a_leaked_view` exist to show the detector CAN
  fail, so the clean result next to them means something.

* **"`source_view` never invokes git."** An assertion about work not done. Tested by
  making `subprocess.run` raise and calling it anyway -- and then, in the same test,
  calling the miner to prove the patch was actually in force. Without that second
  half, a monkeypatch applied to the wrong module would read as a passing test.

The fixture is a purpose-built four-file git repo with nine commits, built fresh per
test. Real history is not reproducible -- swarm-sync's numbers move with every commit
Keith makes -- so nothing here touches a real repo.
"""
from __future__ import annotations

import subprocess

import pytest

from codelearner.eval import gold_from_history as gh
from codelearner.eval.gold_from_history import (
    METHOD_FILE_ADD,
    METHOD_LINE_LOG,
    REJECT_BOILERPLATE,
    REJECT_COPIED_INTO_SOURCE,
    REJECT_NO_MENTION,
    REJECT_NO_PROVENANCE,
    REJECT_TOO_SHORT,
    LeakDetected,
    MinedLabel,
    SourceView,
    assert_no_leak,
    assert_view_is_source_only,
    audit_leak_boundary,
    body_purpose,
    docstring_purpose,
    extract_label,
    find_leaks,
    introducing_commit,
    is_boilerplate,
    mentions_symbol,
    mine_labels,
    name_purpose,
    score_purposes,
    source_view,
    split_units,
    strip_trailers,
    suspect_tokens,
    token_f1,
)

# A phrase that exists ONLY in commit prose, never in any fixture source file. Every
# leak test keys off it: if it turns up on the generator's side of the boundary, the
# boundary is not holding, and no similarity score computed afterwards means anything.
SENTINEL = "the parcel lock is taken with a compare and swap insert"


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
    """A git repo whose history exercises one mining rule per commit.

    Deliberately NOT a realistic repo. Each commit is the minimal history that makes
    exactly one rule decidable, so a failing test names a rule rather than a symptom.
    """
    root = tmp_path / "fixture"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")

    leases = root / "pkg" / "leases.py"

    # 1. Boilerplate subject. The symbol is named in the body, so ONLY the subject
    #    rule can reject it -- which is what makes this commit a test of that rule.
    leases.write_text(
        '"""Leases."""\n\n\n'
        "def scratch_pad():\n"
        '    """Scratch."""\n'
        "    return 1\n"
    )
    _commit(root, "wip\n\nscratch_pad is a scratch pad for trying things out here.\n")

    # 2. A good label: distinctive identifier, named bare in a body that says why.
    leases.write_text(
        leases.read_text()
        + "\n\n"
        "def acquire_lease(parcel, ttl):\n"
        '    """Take the lease."""\n'
        "    return True\n"
    )
    _commit(
        root,
        "Add lease acquisition\n\n"
        "acquire_lease refuses a second holder while the first is alive, because "
        + SENTINEL
        + " and two agents must never both believe they hold one parcel.\n\n"
        "Co-Authored-By: Fixture <fixture@example.invalid>\n",
    )

    # 3. Prose that never names the symbol. The commit is good; the attribution is not.
    leases.write_text(
        leases.read_text()
        + "\n\n"
        "def reap_expired(conn):\n"
        '    """Sweep."""\n'
        "    return 0\n"
    )
    _commit(
        root,
        "Bound the events surface with retention and compaction\n\n"
        "The reaper now sweeps rows whose time to live has elapsed, so a crashed "
        "agent cannot hold a parcel forever.\n",
    )

    # 4. A name that is also an English word, occurring only as English. The false
    #    attribution that a bare substring match would produce.
    money = root / "pkg" / "formats.py"
    money.write_text(
        '"""Formats."""\n\n\n'
        "def money(amount):\n"
        '    """Format an amount."""\n'
        '    return f"${amount}"\n'
    )
    _commit(
        root,
        "Verified baseline of the whole prototype\n\n"
        "All five demo money shots verified end to end, including concurrent "
        "disjoint edits and crash recovery across the reaper.\n",
    )

    # 5. A non-distinctive name in an unmistakably code-ish context. The other half of
    #    rule 4: the mention rule must not simply reject every short lowercase name.
    events = root / "pkg" / "events.py"
    events.write_text(
        '"""Events."""\n\n\n'
        "def tail(conn, since):\n"
        '    """Read."""\n'
        "    return []\n"
    )
    _commit(
        root,
        "Page the event log from a cursor instead of the beginning\n\n"
        "`tail(conn, since)` returns events newer than a caller's cursor so the "
        "read-the-world step no longer pages the whole log from sequence zero.\n",
    )

    # 6. The label, verbatim, sitting in the docstring. Not held out.
    copied = root / "pkg" / "copied.py"
    copied.write_text(
        '"""Copied."""\n\n\n'
        "def compact_events(conn):\n"
        '    """Drop events below the retention floor so the table cannot grow without bound."""\n'
        "    return 0\n"
    )
    _commit(
        root,
        "Add event compaction\n\n"
        "compact_events will drop events below the retention floor so the table "
        "cannot grow without bound, which is the only thing keeping the log finite.\n",
    )

    # 7. A mention too short to be a purpose statement.
    events.write_text(
        events.read_text()
        + "\n\n"
        "def seq(conn):\n"
        '    """Seq."""\n'
        "    return 0\n"
    )
    _commit(root, "Expose the sequence cursor\n\n`seq()` reads it.\n")

    # 8. Added to an EXISTING file, several commits after that file was created. This
    #    is the commit that makes line-log and file-add attribution disagree.
    leases.write_text(
        leases.read_text()
        + "\n\n"
        "def release_lease(parcel):\n"
        '    """Release."""\n'
        "    return True\n"
    )
    _commit(
        root,
        "Release a lease explicitly instead of waiting for its time to live\n\n"
        "release_lease drops the row immediately so a well behaved agent hands the "
        "parcel back the moment it finishes rather than blocking the next one.\n",
    )
    return root


def _labels(repo):
    return {lab.qualname: lab for lab in mine_labels(repo).labels}


# --------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------


def test_line_log_attributes_a_late_symbol_to_the_commit_that_wrote_it(repo):
    """The rule: attribute to the commit that wrote the LINES, not the file.

    `release_lease` was appended to `pkg/leases.py` seven commits after that file was
    created. File-add attribution would hand it the "wip" message from commit 1 --
    prose about a different function -- which is why the line log is the primary
    method.
    """
    prov = introducing_commit(repo, "pkg/leases.py", 1, 200)
    file_add = introducing_commit(repo, "pkg/formats.py", 1, 6)
    assert prov is not None and file_add is not None

    release = introducing_commit(repo, "pkg/leases.py", *_span(repo, "release_lease"))
    assert release is not None
    assert release.method == METHOD_LINE_LOG
    assert release.subject.startswith("Release a lease explicitly")
    assert "wip" not in release.subject

    # And the file's FIRST symbol still traces to the file-add commit, so the line log
    # is not simply returning HEAD for everything.
    scratch = introducing_commit(repo, "pkg/leases.py", *_span(repo, "scratch_pad"))
    assert scratch is not None
    assert scratch.subject == "wip"
    assert scratch.sha != release.sha


def _span(repo, name):
    """The 1-based line span of a top-level def, read out of the working tree."""
    for path in sorted(repo.rglob("*.py")):
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines, start=1):
            if line.startswith(f"def {name}("):
                end = i
                for j in range(i, len(lines)):
                    if lines[j].strip():
                        end = j + 1
                return i, end
    raise AssertionError(f"no def {name} in fixture")


def test_file_add_is_the_recorded_fallback_when_the_line_log_cannot_answer(repo, monkeypatch):
    """The rule: fall back to file-add attribution, and RECORD that it happened.

    The fallback matters less than the recording. Pooling line-log and file-add labels
    without a marker would let the weaker attribution hide inside the stronger one's
    numbers.
    """
    real_git = gh._git

    def no_line_log(repo_arg, *args, **kwargs):
        if any(a.startswith("-L") for a in args):
            return None
        return real_git(repo_arg, *args, **kwargs)

    monkeypatch.setattr(gh, "_git", no_line_log)
    prov = introducing_commit(repo, "pkg/leases.py", *_span(repo, "release_lease"))
    assert prov is not None
    assert prov.method == METHOD_FILE_ADD
    assert prov.subject == "wip"  # the file-add commit, i.e. the wrong prose


def test_an_untracked_file_yields_no_provenance(repo):
    """The rule: no history, no label -- recorded as a rejection, not an exception.

    The symbols are passed in explicitly because `repo_symbols` walks git's tracked
    file list, so an untracked file never reaches the miner by that route. A caller
    with its own index can and will hand over symbols git has never seen.
    """
    from codelearner.ingest.python_extract import extract_file

    path = repo / "pkg" / "untracked.py"
    path.write_text('"""U."""\n\n\ndef never_committed():\n    return 1\n')
    sym = next(
        s for s in extract_file(path, repo).symbols if s.name == "never_committed"
    )
    report = mine_labels(repo, symbols=[("pkg/untracked.py", sym)])
    assert report.considered == 1
    assert report.labels[0].reject == REJECT_NO_PROVENANCE
    assert report.usable == []


# --------------------------------------------------------------------------------
# Quality filtering
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject",
    [
        "wip", "WIP", "fix", "Fix", "fixes", "update", "cleanup", "refactor",
        "misc", "typo", "lint", "bump deps", "v1.2.3", "Merge pull request #1 from x",
        "address review", "PR feedback", "initial commit", "chore", "docs",
        "fix: wip", "fix(leases): typo", "revert this thing", "",
    ],
)
def test_boilerplate_subjects_are_rejected(subject):
    assert is_boilerplate(subject)


@pytest.mark.parametrize(
    "subject",
    [
        "Add lease acquisition with a compare-and-swap",
        "fix(leases): acquire's CAS must stamp from SQLite's clock",
        "Bound the events surface with retention and compaction",
        "WP4.5: declare the wire contract for parcels and leases",
    ],
)
def test_real_subjects_are_not_rejected_as_boilerplate(subject):
    assert not is_boilerplate(subject)


def test_a_boilerplate_subject_rejects_the_label_even_when_the_body_names_the_symbol(repo):
    """The rule fires on the SUBJECT, and the fixture proves it is the deciding factor.

    Commit 1's body does name `scratch_pad`, so the mention rule would have admitted
    it. Only the subject rule rejects it -- delete that rule and this symbol becomes
    a usable label whose prose is "scratch_pad is a scratch pad for trying things out".
    """
    assert _labels(repo)["pkg.leases.scratch_pad"].reject == REJECT_BOILERPLATE


def test_prose_that_never_names_the_symbol_is_not_a_label(repo):
    """The rule that costs the most and matters the most.

    Commit 3 is a good commit with real prose about a real change. It never writes
    `reap_expired`, so its prose is about the commit and not about the symbol, and
    admitting it would put a work-package description in the gold set as if it were
    a purpose statement.
    """
    assert _labels(repo)["pkg.leases.reap_expired"].reject == REJECT_NO_MENTION


def test_an_english_word_that_happens_to_be_a_name_is_not_a_mention(repo):
    """The false-attribution rule: "money shots" is not prose about `money`.

    Found on the real corpus -- swarm-sync's initial commit says "all 5 demo money
    shots verified", and a bare substring match reads that as a purpose statement for
    `sample_repo.formats.money`. Without this rule the gold set silently acquires
    labels that are about nothing.
    """
    assert _labels(repo)["pkg.formats.money"].reject == REJECT_NO_MENTION
    assert not mentions_symbol("All five demo money shots verified end to end", "money")


def test_a_short_name_in_code_context_does_count_as_a_mention(repo):
    """The other side of that rule: it must not reject every short lowercase name.

    A rule that only accepted distinctive identifiers would throw away `tail`, `emit`,
    `seq` -- the short names that carry most of a codebase's core behaviour.
    """
    label = _labels(repo)["pkg.events.tail"]
    assert label.usable, label.reject
    assert "cursor" in label.prose
    assert mentions_symbol("`tail(conn, since)` returns events", "tail")
    assert mentions_symbol("events.tail is the reader", "tail")
    assert not mentions_symbol("the tail of the log grows", "tail")


def test_a_mention_too_short_to_state_a_purpose_is_rejected(repo):
    assert _labels(repo)["pkg.events.seq"].reject == REJECT_TOO_SHORT


def test_trailers_are_not_part_of_a_label(repo):
    """Trailers are in nearly every commit in both real repos and would contribute
    identical tokens to every label, compressing the gap between signal and control."""
    label = _labels(repo)["pkg.leases.acquire_lease"]
    assert label.usable
    assert "Co-Authored-By" not in label.prose
    assert "fixture@example.invalid" not in label.prose
    assert strip_trailers("Subject\n\nBody.\n\nSigned-off-by: X <y@z>\n") == "Subject\n\nBody."


def test_the_funnel_accounts_for_every_symbol_considered(repo):
    """A yield is only interpretable next to what it rejected and why."""
    report = mine_labels(repo)
    assert report.considered == len(report.labels)
    assert report.considered == len(report.usable) + sum(report.rejects().values())
    assert 0.0 < report.usable_fraction < 1.0
    assert report.commits_in_history == 8


def test_the_gold_dump_carries_its_rule_and_its_sha_and_no_home_directory(repo):
    """A published gold set states the rule it was labelled by and the sha it holds for.

    The hand-labelled set does both (`labelling_rule`, `commit_note`); a mined set that
    did neither could not be disagreed with and could not be told from a stale one. The
    repo field is a name, not a path -- an artifact that records someone's home
    directory is a leak of a different kind.
    """
    report = mine_labels(repo)
    doc = gh.to_gold_json(report, head="deadbeef" * 5)
    assert doc["repo"] == "fixture"
    assert "/" not in doc["repo"]  # no path, therefore no home directory
    assert doc["mined_at_head"] == "deadbeef" * 5
    assert "MINED, not hand-labelled" in doc["commit_note"]
    assert "git log -L" in doc["labelling_rule"]
    assert doc["rejected"][REJECT_NO_MENTION] >= 1
    assert len(doc["labels"]) == len(report.usable)
    assert all(entry["purpose"] for entry in doc["labels"])


def test_mining_is_deterministic(repo):
    first = mine_labels(repo)
    second = mine_labels(repo)
    assert [(x.qualname, x.prose, x.reject) for x in first.labels] == [
        (y.qualname, y.prose, y.reject) for y in second.labels
    ]


def test_units_split_bullets_as_well_as_sentences():
    """A bullet-list body is one unit per bullet.

    Measured on swarm-sync: the work-package commits are bullet lists in which each
    bullet is about a different module, so splitting on sentences alone produces
    labels several modules wide.
    """
    units = split_units("Subject line.\n\n- first about db.py\n- second about cli.py\n")
    assert units == ["Subject line.", "first about db.py", "second about cli.py"]
    assert extract_label(
        "S\n\n- `emit` writes one row.\n- `tail` reads many rows back out.\n", "emit"
    )[0] == "`emit` writes one row."


# --------------------------------------------------------------------------------
# The leak boundary
# --------------------------------------------------------------------------------


def test_source_view_never_invokes_git(repo, monkeypatch):
    """THE property. The generator's input is built from the working tree, full stop.

    Both halves are load-bearing. The first shows `source_view` works with git made
    unusable; the second shows the sabotage was real -- without it, a monkeypatch
    applied to the wrong module would leave this test green and prove nothing.
    """
    from codelearner.ingest.python_extract import extract_file

    extract = extract_file(repo / "pkg" / "leases.py", repo)
    sym = next(s for s in extract.symbols if s.name == "acquire_lease")

    calls: list[object] = []

    def refusing_run(*args, **kwargs):
        calls.append(args)
        raise OSError("git is not available in this test")

    monkeypatch.setattr(gh.subprocess, "run", refusing_run)

    view = source_view(repo, "pkg/leases.py", sym)
    assert "def acquire_lease" in view.source
    assert SENTINEL not in view.source
    assert calls == [], "source_view must not shell out at all"

    # The sabotage bites: the miner, which DOES need git, now gets nothing. Without
    # this half, a monkeypatch applied to the wrong module would leave the assertions
    # above green while proving nothing at all.
    assert introducing_commit(repo, "pkg/leases.py", 1, 5) is None
    assert calls, "the patch was never in force"


def test_source_view_has_nowhere_to_put_a_commit_message():
    """The boundary is structural: there is no field for provenance on the view.

    A rule enforced by the dataclass rather than by discipline. Adding a `commit` or
    `message` field here would make the leak possible again, and this test is what
    would go red.
    """
    names = {f.name for f in SourceView.__dataclass_fields__.values()}
    assert names == {
        "qualname", "kind", "path", "line_start", "line_end",
        "signature", "docstring", "source",
    }
    for banned in ("commit", "message", "body", "subject", "prose", "provenance", "label"):
        assert banned not in names


def test_no_label_prose_is_reachable_from_any_generator_input(repo):
    """The whole mined set, cross-checked: every view against EVERY label.

    The cross product, not just each view against its own label, because the subtler
    failure is a harness that built the right-looking views from the wrong symbols.
    """
    report = mine_labels(repo)
    checked, findings = audit_leak_boundary(repo, report.usable)
    assert checked == len(report.usable) >= 3
    assert findings == []
    # And the sentinel specifically -- the phrase that exists only in commit prose.
    assert any(SENTINEL in lab.prose for lab in report.usable)
    for lab in report.usable:
        text = (repo / lab.path).read_text()
        assert SENTINEL not in text


def test_find_leaks_catches_a_planted_leak(repo):
    """Proof the detector CAN fail. Without this, the clean result above is vacuous."""
    report = mine_labels(repo)
    label = next(lab for lab in report.usable if SENTINEL in lab.prose)
    honest = _view_for(repo, label)
    assert find_leaks(honest, [label.prose]) == []

    leaked = SourceView(
        qualname=honest.qualname,
        kind=honest.kind,
        path=honest.path,
        line_start=honest.line_start,
        line_end=honest.line_end,
        signature=honest.signature,
        docstring=label.prose,          # the answer, handed to the generator
        source=honest.source,
    )
    hits = find_leaks(leaked, [label.prose])
    assert hits, "a view carrying the label prose must be detected"
    assert any(".docstring" in hit for hit in hits)
    with pytest.raises(LeakDetected):
        assert_no_leak(leaked, [label.prose])


def test_find_leaks_walks_into_nested_containers(repo):
    """The walk is recursive, not a repr scan -- a leak one level down still counts."""
    nested = {"view": [{"payload": ("x", SENTINEL + " and more besides")}]}
    assert find_leaks(nested, [SENTINEL])
    assert find_leaks({"a": {"b": [SENTINEL]}}, [SENTINEL])
    assert find_leaks("plain string " + SENTINEL, [SENTINEL])


def test_find_leaks_ignores_incidental_word_overlap():
    """A shared word is not a leak. A rule that fired on one word would reject every
    label and make the boundary check useless rather than strict."""
    assert find_leaks(SourceView(
        qualname="a.b", kind="function", path="a.py", line_start=1, line_end=2,
        signature="b()", docstring="Takes the lease.", source="def b(): pass",
    ), ["acquire_lease refuses a second holder while the first is alive"]) == []


def test_assert_view_is_source_only_rejects_text_the_working_tree_does_not_contain(repo):
    """The structural gate. It cannot false-positive on an author quoting themselves,
    and it cannot be satisfied by anything the file does not actually contain."""
    report = mine_labels(repo)
    label = next(lab for lab in report.usable if SENTINEL in lab.prose)
    honest = _view_for(repo, label)
    assert_view_is_source_only(repo, honest)  # does not raise

    forged = SourceView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature, docstring=honest.docstring,
        source="# " + label.prose,
    )
    with pytest.raises(LeakDetected, match="not a substring"):
        assert_view_is_source_only(repo, forged)

    forged_doc = SourceView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature, docstring=label.prose, source=honest.source,
    )
    with pytest.raises(LeakDetected, match="docstring"):
        assert_view_is_source_only(repo, forged_doc)


def test_score_purposes_refuses_to_score_a_leaked_view(repo):
    """The gate is WIRED INTO the scoring path, not merely available beside it.

    A leak does not degrade the measurement, it voids it, so the harness raises rather
    than returning a better-looking number.
    """
    report = mine_labels(repo)
    label = next(lab for lab in report.usable if SENTINEL in lab.prose)

    # A label whose prose is lifted from the symbol's own source. Mining rejects these
    # (REJECT_COPIED_INTO_SOURCE); this is what happens if one arrives anyway.
    source_text = " ".join(_view_for(repo, label).source.split())
    planted = MinedLabel(
        qualname=label.qualname, kind=label.kind, path=label.path,
        prose=source_text, commit=label.commit, subject=label.subject,
        method=label.method, files_touched=label.files_touched, units=1,
    )
    with pytest.raises(LeakDetected):
        score_purposes(repo, [planted], docstring_purpose, "leaked")

    # The honest label scores without raising, so the exception above is the gate
    # firing and not the harness being broken.
    card = score_purposes(repo, [label], docstring_purpose, "honest")
    assert card.n == 1


def test_score_purposes_runs_the_structural_gate_too(repo, monkeypatch):
    """The structural gate is wired into scoring, and this is the only test that shows it.

    In normal operation every view is legitimate, so the structural gate never fires --
    which means deleting the call from `score_purposes` would break nothing and leave
    an unwired guard reading as a working one. Forging a view at the seam makes the
    wire itself observable. The forged text is absent from the file AND from the label,
    so only the structural gate can catch it.
    """
    report = mine_labels(repo)
    label = report.usable[0]
    honest = _view_for(repo, label)
    forged = SourceView(
        qualname=honest.qualname, kind=honest.kind, path=honest.path,
        line_start=honest.line_start, line_end=honest.line_end,
        signature=honest.signature, docstring=honest.docstring,
        source="# bytes that appear in no fixture file and in no commit message\n",
    )
    monkeypatch.setattr(gh, "source_view", lambda *a, **k: forged)
    with pytest.raises(LeakDetected, match="not a substring"):
        score_purposes(repo, [label], docstring_purpose, "forged")


def test_a_label_copied_into_the_docstring_is_rejected_at_mining_time(repo):
    """Not a harness bug -- an author reusing their own clause -- but not held out.

    The answer is in the input for these, so a docstring-copying generator scores
    perfectly on them for no reason. Dropped, and counted, rather than scored.
    """
    labels = _labels(repo)
    assert labels["pkg.copied.compact_events"].reject == REJECT_COPIED_INTO_SOURCE


def test_suspect_tokens_flags_vocabulary_only_the_answer_contained(repo):
    """The reverse direction: output words that were in the answer but not in the input.

    Counted rather than raised -- short tokens collide constantly -- but a generator
    that had somehow seen the label shows up here even if `assert_no_leak` passes.
    """
    report = mine_labels(repo)
    label = next(lab for lab in report.usable if SENTINEL in lab.prose)
    view = _view_for(repo, label)

    cheating = suspect_tokens(label.prose, label.prose, view)
    assert cheating, "a generator echoing the label must show suspect tokens"

    honest = suspect_tokens(docstring_purpose(view), label.prose, view)
    assert honest == []


def _view_for(repo, label):
    from codelearner.ingest.python_extract import extract_file

    extract = extract_file(repo / label.path, repo)
    sym = next(s for s in extract.symbols if s.qualname == label.qualname)
    return source_view(repo, label.path, sym)


# --------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------


def test_token_f1_bounds():
    assert token_f1("open a wal connection", "open a wal connection") == 1.0
    assert token_f1("open a wal connection", "delete every parcel row") == 0.0
    assert token_f1("", "anything at all here") == 0.0
    assert 0.0 < token_f1("open a wal connection", "open a socket connection") < 1.0


def test_token_f1_splits_identifiers_so_prose_and_code_can_agree():
    """`bind_managed_root` and "bind the managed root" have to tokenise alike or no
    similarity between prose and code is ever non-zero."""
    assert token_f1("bind_managed_root", "bind the managed root") == 1.0
    assert token_f1("BlackboardUnreachable", "the blackboard is unreachable") == 1.0


def test_token_f1_cannot_tell_opposites_apart():
    """A stated limit, pinned by a test so nobody reads the score as correctness.

    This is why the shuffled control and the condition gap are the reported numbers
    rather than the raw similarity.
    """
    assert token_f1("opens the connection", "closes the connection") == token_f1(
        "opens the connection", "opens the connection".replace("opens", "closes")
    )
    assert token_f1("the lease is granted", "the lease is not granted") == 1.0


def test_the_shuffled_control_never_pairs_a_view_with_its_own_label():
    """A control with a fixed point scores itself and understates the signal."""
    for n in (2, 3, 5, 37, 316):
        order = gh._derangement(n)
        assert sorted(order) == list(range(n))
        assert all(i != j for i, j in enumerate(order))
    assert gh._derangement(37) == gh._derangement(37)  # deterministic


def test_name_blind_scoring_gives_a_name_echo_no_credit(repo):
    """Why `name_blind` defaults to True.

    Every mined label contains the symbol's name -- the mention rule selected it for
    that reason -- so a generator that echoes the name is guaranteed an overlap it did
    not earn. Measured on swarm-sync: scored raw, `name + signature only` reaches
    0.210 against `docstring first sentence`'s 0.225, i.e. the metric cannot tell a
    name echo from reading the documentation. Name-blinding drops the echo to 0.020.
    """
    report = mine_labels(repo)
    usable = report.usable

    def echo(view):
        return view.qualname.rsplit(".", 1)[-1]

    blind = score_purposes(repo, usable, echo, "echo", name_blind=True)
    raw = score_purposes(repo, usable, echo, "echo", name_blind=False)
    assert blind.gold == 0.0
    assert raw.gold > 0.0


def test_a_source_reading_generator_beats_the_shuffled_control(repo):
    """The result the module exists to produce: mined prose carries symbol-specific
    signal, or it does not. On the fixture and on swarm-sync it does."""
    report = mine_labels(repo)
    card = score_purposes(repo, report.usable, body_purpose, "body identifiers")
    assert card.n == len(report.usable)
    assert card.gold > card.shuffled
    assert card.suspect == 0


def test_generators_are_source_only_and_produce_no_label_vocabulary(repo):
    """Every shipped generator, run over the whole mined set, stays inside the boundary."""
    report = mine_labels(repo)
    for generator in (docstring_purpose, name_purpose, body_purpose):
        for lab in report.usable:
            view = _view_for(repo, lab)
            out = generator(view)
            assert SENTINEL not in out
            assert suspect_tokens(out, lab.prose, view) == []


def test_without_docstring_removes_the_docstring_from_the_source_too(repo):
    """The doc-blind condition has to blind the BODY as well as the field.

    Removing `view.docstring` alone would leave the same text in `view.source`, and a
    generator that reads the source would still see it -- a doc-blind condition that
    is not blind, which is the failure mode that reports an inference score for a copy.
    """
    report = mine_labels(repo)
    label = next(lab for lab in report.usable if SENTINEL in lab.prose)
    view = _view_for(repo, label)
    assert view.docstring
    blinded = view.without_docstring()
    assert blinded.docstring is None
    assert view.docstring.strip() not in blinded.source
    assert docstring_purpose(blinded) == name_purpose(blinded)
