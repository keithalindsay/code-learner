"""The gate's negative controls, and the controls on the controls.

`tests/test_mcp.py` proves the gate refuses eight hand-written submissions. This file
proves something different and harder to fake: that a generated adversarial corpus is
refused at a measured RATE of 1.0, that legitimate submissions are still admitted at a
measured rate of 1.0, and -- the part that makes the first two numbers worth anything --
that every control detects the deletion of the rule it names.

Three tests in this project have passed while asserting nothing: a clustering fixture
with no clusters in it, a hub that inherited its leaf's centrality, and a timestamp test
that survived deleting the code it was named after. A negative-control suite is the
easiest place in a codebase for that to happen again, because everything about it looks
right when it is measuring nothing: an empty corpus refuses 100% of what it submits, a
harness pointed at the wrong repo refuses everything for the wrong reason, and a decoy
hash that happens to be the correct one passes as a refusal that never happened.

So the tests below are in two halves. The first half measures the gate. The second half
attacks the measurement -- an empty corpus must raise rather than report 1.0, a no-op
mutation must NOT be reported as detected, an unmutated copy must score exactly what the
working tree scores, and a mutation snippet that has drifted out of the source must fail
loudly rather than silently test nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="the gate lives on the MCP surface, an optional extra")

from codelearner.eval.gate_controls import (  # noqa: E402
    ACCEPTED,
    FAMILIES,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    REFUSED,
    Edit,
    Family,
    GateReport,
    Mutation,
    MutationFailed,
    VacuousCorpus,
    build_corpus,
    build_harness,
    gate_module,
    mutate_tree,
    run_controls,
    run_mutation,
    run_unmutated_copy,
)

# The eight attacks the gate is claimed to refuse, named here as a literal so that
# deleting one from the corpus fails a test rather than quietly improving the score.
# `escaping_path` is the ninth: the same rule the README's threat model implies but does
# not spell out -- a real file, really hashed, from outside the indexed repo.
REQUIRED_ATTACKS = {
    "zero_evidence",
    "absent_file",
    "past_eof",
    "decoy_content_hash",
    "stale_but_once_valid",
    "blank_range",
    "foreign_symbol_hash",
    "unknown_subject",
    "escaping_path",
}

# The shapes where a symbol's stored bytes are NOT its lines' bytes: two modules (whose
# span runs one line past the last line anything is written on), a property (four columns
# in from the start of its line) and a decorated method (whose first byte is the `@`, a
# line above its `def`). These are the symbols a too-narrow gate FALSELY REJECTS, so a
# positive-control suite that does not contain them is measuring the easy half.
DISAGREEING_SHAPES = {"core", "tray", "tray.Tray.widgets", "tray.Tray.count"}


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    """One indexed shapes repo, reused. Controls restore what they edit."""
    return build_harness(tmp_path_factory.mktemp("gate-controls"))


@pytest.fixture(scope="module")
def report(harness):
    return run_controls(harness)


@pytest.fixture(scope="module")
def mutant_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("gate-mutations")


# ---------------------------------------------------------------------------
# the fixture itself -- coverage that cannot be assumed
# ---------------------------------------------------------------------------

def test_the_shapes_fixture_still_contains_every_shape_the_corpus_needs(harness):
    """Every attack below is generated FROM the fixture, so a fixture that lost a shape
    silently loses the controls that depend on it and the suite gets greener. Named
    kinds, a blank line and a sibling symbol in every file, and at least one symbol whose
    stored bytes disagree with its lines' bytes."""
    kinds = {s.kind for fact in harness.files.values() for s in fact.symbols}
    assert {"module", "class", "method", "function"} <= kinds, "fixture lost a symbol kind"
    disagreeing = {
        s.qualname
        for fact in harness.files.values()
        for s in fact.symbols
        if fact.line_bytes(s.line_start, s.line_end) != (s.byte_start, s.byte_end)
    }
    assert disagreeing == DISAGREEING_SHAPES
    for fact in harness.files.values():
        assert fact.blank_lines(), f"{fact.path} has no blank line to cite"
        assert len(fact.symbols) >= 2, f"{fact.path} has no sibling symbol"


def test_the_corpus_covers_every_named_attack(report):
    """A family with no instances is a rule nothing tests, and it costs nothing to
    generate an empty one by accident: every attack here is built from a symbol that has
    to satisfy a condition."""
    assert REQUIRED_ATTACKS <= set(NEGATIVE_FAMILIES)
    assert set(POSITIVE_FAMILIES) == {"published_hash", "quoted_lines", "multi_span"}
    for name in FAMILIES:
        assert report.family(name), f"family {name!r} generated no controls"
    # Floors, not exact counts: the point is that each attack is applied to many symbols
    # rather than demonstrated once, which is what makes a rate a rate.
    assert len(report.negatives) >= 60
    assert len(report.positives) >= 15


def test_every_skip_is_a_named_reason_rather_than_a_silent_gap(report):
    """An instance that cannot be constructed is returned with a reason. A skip that
    disappears is how a family becomes empty without anybody's number moving."""
    assert [(family, reason) for family, _, reason in report.skips] == [
        (
            "quoted_lines",
            "the symbol's stored line range is not a valid line range -- a module ends "
            "one line past its last written line",
        ),
    ] * 2


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def test_the_gate_refuses_every_attack_in_the_corpus(report):
    """The load-bearing claim of the project, as a rate. Any single acceptance here is a
    defect in the gate, not a reason to soften the corpus."""
    accepted = [o.control for o in report.negatives if not o.refused]
    assert accepted == [], f"the gate ADMITTED an attack: {accepted}"
    assert report.rejection_rate == 1.0


def test_every_refusal_names_the_rule_that_produced_it(report):
    """Rejection rate alone is not enough. A harness pointed at the wrong repo root
    refuses all 62 submissions with `file_missing` and scores a perfect 1.0; requiring
    `bad_range` from the beyond-EOF attack and `hash_mismatch` from the stale one turns
    that failure into a wrong code instead of a right rate."""
    wrong = [
        (o.control, o.code)
        for o in report.negatives
        if o.code not in FAMILIES[o.family].codes
    ]
    assert wrong == []
    assert report.attributed_rate == 1.0
    # Each attack is refused by its OWN rule, so the codes must not collapse into one.
    codes = {code for name in NEGATIVE_FAMILIES for code in report.codes(name)}
    assert codes == {
        "evidence_required",
        "file_missing",
        "path_escapes_repo",
        "bad_range",
        "hash_mismatch",
        "unknown_subject",
    }


def test_a_refusal_never_leaves_a_row_behind(report):
    """A gate that says no and writes the row anyway has refused nothing -- and every
    later stage would treat that row as admitted evidence."""
    assert [o.control for o in report.negatives if o.rows_added != 0] == []


def test_the_gate_admits_every_legitimate_submission(report):
    """Without this the suite is unfalsifiable: a gate that refuses everything scores a
    perfect rejection rate. The four shapes where a symbol's stored bytes are not its
    lines' bytes are named explicitly, because they are the ones a narrowed gate rejects
    while still refusing every attack."""
    refused = [(o.control, o.code) for o in report.positives if o.verdict != ACCEPTED]
    assert refused == [], f"the gate REFUSED a correct citation: {refused}"
    assert report.positive_pass_rate == 1.0
    admitted = {o.control.split("/", 1)[1] for o in report.family("published_hash")}
    assert DISAGREEING_SHAPES <= admitted


def test_an_admitted_claim_is_servable_and_keeps_all_of_its_evidence(report):
    """Accepted is not the whole promise. The claim has to survive the second
    verification `_submit_body` runs against disk, and it has to store every span it was
    submitted with -- a claim quietly reduced to one of its two citations stands on less
    than its author thought, and nothing downstream would record that."""
    for outcome in report.positives:
        assert outcome.servable is True, outcome.control
        assert outcome.rows_added == 1, outcome.control
        assert outcome.evidence == outcome.expected_evidence, outcome.control
    assert [o.evidence for o in report.family("multi_span")] == [2, 2]


def test_the_controls_drive_the_production_gate_and_not_a_copy_of_it(report):
    """The corpus is worth nothing if it is scored against a reimplementation. The report
    records which module answered, so a future refactor that stubs the gate out of the
    harness fails here rather than passing quietly."""
    assert report.gate_path == str(gate_module().__file__)
    assert report.gate_path.endswith("codelearner/server/app.py")


# ---------------------------------------------------------------------------
# the attacks, individually, where the rate could hide something
# ---------------------------------------------------------------------------

def test_the_stale_attack_was_a_correct_citation_before_the_file_changed(harness):
    """The attack that matters most, and the only one whose evidence was ever real.

    Every other refusal could be explained by the citation being malformed. This one is
    well-formed, cites bytes the index itself published, and must STILL be refused once
    the file moves under it. Proven both ways round: the same submission is accepted with
    the file as it was and refused with the file as it is, so the refusal can only be
    coming from the re-read rather than from anything about the submission."""
    controls, _ = build_corpus(harness.files)
    stale = [c for c in controls if c.family == "stale_but_once_valid"]
    assert len(stale) >= 6, "the stale family lost its instances"
    for control in stale:
        harness.restore()
        before = harness.submit(control)
        assert before["ok"] is True, f"{control.name} was never a valid citation"

        harness.restore()
        harness.apply(control)
        after = harness.submit(control)
        assert after["ok"] is False
        assert after["error"]["code"] == "hash_mismatch"
        assert after["error"]["cited_hash"] != after["error"]["observed_hash"]
    harness.restore()


def test_every_decoy_hash_is_one_the_gate_should_not_accept(harness):
    """The way a negative control turns into a positive one without anybody noticing: if
    a decoy hash happens to be a hash the gate is entitled to accept, the control passes
    as a refusal that never had to happen. Recomputed here off disk, independently of the
    gate, for every hash-based attack."""
    controls, _ = build_corpus(harness.files)
    checked = 0
    for control in controls:
        if control.family not in {"decoy_content_hash", "foreign_symbol_hash"}:
            continue
        span = control.spans[0]
        fact = harness.files[span["path"]]
        accepted = fact.accepted_hashes(span["line_start"], span["line_end"])
        assert accepted, f"{control.name} cites lines with no acceptable hash at all"
        assert span["content_hash"] not in accepted, control.name
        checked += 1
    assert checked >= 12, "the hash families lost their instances"


def test_a_blank_line_is_refused_by_range_rather_than_by_hash(harness):
    """sha256 of nothing is a perfectly stable hash, so an empty span would verify
    forever while pointing at nothing. Both channels are tried -- the empty text and the
    hash of the empty text -- because either one getting in is the same hole."""
    controls, _ = build_corpus(harness.files)
    blank = [c for c in controls if c.family == "blank_range"]
    assert len(blank) == 4
    cited = {c.spans[0].get("text", c.spans[0].get("content_hash")) for c in blank}
    assert cited == {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}


def test_both_surfaces_return_the_same_verdict_for_every_control(tmp_path):
    """The corpus is scored through the tool body; an agent reaches it through the
    registered MCP tool. If those ever disagree, the measurement is about a code path
    nobody calls -- and a refusal that arrives as a traceback rather than as data is a
    refusal the agent reads as a broken server."""
    direct = run_controls(build_harness(tmp_path / "direct"))
    tool = run_controls(build_harness(tmp_path / "tool", surface="tool"))
    assert [(o.control, o.verdict, o.code) for o in direct.outcomes] == [
        (o.control, o.verdict, o.code) for o in tool.outcomes
    ]
    assert tool.rejection_rate == 1.0
    assert tool.positive_pass_rate == 1.0
    assert "raised_into_transport" not in {o.code for o in tool.outcomes}


# ---------------------------------------------------------------------------
# controls on the measurement
# ---------------------------------------------------------------------------

def test_a_rate_over_no_controls_raises_rather_than_reporting_a_perfect_gate():
    """`refused / len(negatives)` is 1.0 for an empty corpus under every convention that
    does not raise. A fixture that silently generated nothing would then report a perfect
    gate -- which is exactly the shape of the three vacuous tests this project has
    already shipped."""
    empty = GateReport(corpus="none", symbols=0, outcomes=[], skips=[], gate_path="?")
    for rate in ("rejection_rate", "attributed_rate", "positive_pass_rate"):
        with pytest.raises(VacuousCorpus):
            getattr(empty, rate)
    with pytest.raises(VacuousCorpus):
        empty.hold_rate("zero_evidence")
    # Serialisation may not raise -- a family-filtered run has no positives by
    # construction -- but it must report absence, never 1.0.
    assert empty.to_json()["rejection_rate"] is None


@pytest.mark.parametrize("family", list(FAMILIES))
def test_each_control_detects_the_deletion_of_the_rule_it_names(family, mutant_dir, report):
    """The one test that decides whether any of the others mean anything.

    The rule each family targets is deleted from a COPY of the package and the family is
    re-measured in a subprocess importing that copy. A control whose verdict does not
    move when its own rule is gone is decoration: it would keep passing through the
    regression it exists to catch.

    The positive families move partway rather than to zero, and that is the correct
    answer: removing the symbol-span reading falsely rejects only the symbols whose
    stored bytes are not their lines' bytes -- half of this fixture, and 217 of 850
    (25.5%) of this repository -- which is precisely the population the rule exists for.
    A mutation that took a positive family to zero would mean the rule was doing
    something broader than it claims."""
    result = run_mutation(family, mutant_dir, baseline=report)
    assert result.baseline_rate == 1.0
    assert result.detected, (
        f"deleting {result.rule!r} did not change the {family} verdict "
        f"({result.baseline_rate} -> {result.mutant_rate}); the control cannot see its "
        "own rule being removed"
    )
    assert result.mutant_n == len(report.family(family))
    assert result.flipped, "detection was reported with no control naming itself"
    if FAMILIES[family].expect == REFUSED:
        assert result.mutant_rate == 0.0, "the attack should succeed with the rule gone"
    else:
        assert set(result.flipped) <= {
            f"{family}/{name}" for name in DISAGREEING_SHAPES
        } or family == "multi_span"


def test_a_no_op_mutation_is_not_reported_as_detected(mutant_dir, report):
    """The mutation harness has to be able to say no. If an empty edit list still comes
    back 'detected', then every detection above is an artefact of copying the tree or of
    running in a subprocess rather than of the rule being gone."""
    family = "past_eof"
    unchanged = Family(**{**FAMILIES[family].__dict__, "mutation": Mutation("no-op", ())})
    saved = FAMILIES[family]
    FAMILIES[family] = unchanged
    try:
        result = run_mutation(family, mutant_dir / "noop", baseline=report)
    finally:
        FAMILIES[family] = saved
    assert result.mutant_rate == result.baseline_rate == 1.0
    assert result.detected is False


def test_an_unmutated_copy_scores_exactly_what_the_working_tree_scores(mutant_dir, report):
    """The other half of the same doubt. A mutant is compared against a baseline measured
    in the working tree, so if copying the package changed the score by itself, every
    mutation result would be measuring the copy."""
    payload = run_unmutated_copy(mutant_dir / "unmutated")
    assert payload["rejection_rate"] == report.rejection_rate == 1.0
    assert payload["positive_pass_rate"] == report.positive_pass_rate == 1.0
    assert payload["failures"] == []
    assert payload["gate_path"].startswith(str(mutant_dir / "unmutated"))


def test_a_mutation_snippet_that_has_drifted_out_of_the_source_raises(mutant_dir):
    """A rule that moved leaves its control pointing at nothing, and a harness that
    skipped the mutation would report that as 'not detected' at best and say nothing at
    worst. Exactly one occurrence, or refuse to run."""
    drifted = Mutation(
        rule="a rule that no longer exists",
        edits=(
            Edit(
                target="server/app.py",
                old="if not 0 <= byte_start < byte_end <= len(source):  # moved away",
                new="pass",
            ),
        ),
    )
    with pytest.raises(MutationFailed, match="appears 0 times"):
        mutate_tree(mutant_dir / "drifted", drifted)


def test_a_mutation_is_measured_against_the_copy_and_never_the_installed_package(mutant_dir, report):
    """The subprocess has to import the mutant. If PYTHONPATH ever stopped winning over
    the editable install, every mutation would silently measure the unmutated gate."""
    result = run_mutation("zero_evidence", mutant_dir / "isolation", baseline=report)
    assert result.gate_path.startswith(str(mutant_dir / "isolation"))
    assert result.gate_path != str(gate_module().__file__)


def test_the_working_tree_is_never_edited_by_a_mutation(mutant_dir, report):
    """Mutating an installed package in place leaves a window in which any other process
    importing it sees the hole, and a crash mid-run leaves the hole behind for good."""
    gate = gate_module().__file__
    before = open(gate, "rb").read()  # noqa: SIM115
    run_mutation("decoy_content_hash", mutant_dir / "untouched", baseline=report)
    assert open(gate, "rb").read() == before  # noqa: SIM115
