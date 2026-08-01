"""The gate's negative controls, and the controls on the controls.

`tests/test_mcp.py` proves the gate refuses eight hand-written submissions. This file
proves something different and harder to fake: that a generated adversarial corpus is
refused at a measured RATE, that legitimate submissions are still admitted at a measured
rate of 1.0, and -- the part that makes the first two numbers worth anything -- that
every control detects the deletion of the rule it names, at every door that rule can be
reached through.

Three tests in this project have passed while asserting nothing: a clustering fixture
with no clusters in it, a hub that inherited its leaf's centrality, and a timestamp test
that survived deleting the code it was named after. A negative-control suite is the
easiest place in a codebase for that to happen again, because everything about it looks
right when it is measuring nothing: an empty corpus refuses 100% of what it submits, a
harness pointed at the wrong repo refuses everything for the wrong reason, and a decoy
hash that happens to be the correct one passes as a refusal that never happened.

**And a fourth way, which is what WP5 added.** A corpus can measure the right thing at
the wrong door. Until now every rate here described `server.app`, while `codelearner
learn` and every library caller reach `store.write_assertion` directly -- so "100%
refused" was a true statement about one of the two ways in. Running the same corpus at
the second door found `escaping_path` admitted there, stored `active` and reported
servable, on every one of 12,803 instances.

Both columns now read 1.000, and the tests below still assert them SEPARATELY. A test
that averaged them, or asserted only the better one, would put the measurement back
where it was -- and the two doors still refuse seven of these attacks under different
names, which `differently_named` reports precisely so that agreement on rates is not
read as one gate wearing two hats.

So the tests are in four parts. The first measures the gate at each door. The second
compares the doors. The third attacks the measurement -- an empty corpus must raise
rather than report 1.0, a no-op mutation must NOT be reported as detected, an unmutated
copy must score exactly what the working tree scores, and a mutation snippet that has
drifted out of the source must fail loudly rather than silently test nothing. The fourth
attacks `Outcome.held`, the single predicate all of those numbers are computed through,
with synthetic outcomes the corpus cannot currently produce.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("mcp", reason="the gate lives on the MCP surface, an optional extra")

from codelearner.eval.gate_controls import (  # noqa: E402
    ACCEPTED,
    FAMILIES,
    GATE_SERVER,
    GATE_STORE,
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    REFUSED,
    SURFACE_DIRECT,
    SURFACE_STORE,
    SURFACE_TOOL,
    Edit,
    Family,
    GateReport,
    Inexpressible,
    Mutation,
    MutationFailed,
    Outcome,
    Rule,
    Unenforced,
    VacuousCorpus,
    build_corpus,
    build_harness,
    compare_surfaces,
    gate_module,
    gate_of,
    mutable_families,
    mutate_tree,
    run_controls,
    run_mutation,
    run_unmutated_copy,
    store_module,
    store_refusal_codes,
)

# The attacks the gate is claimed to refuse, named here as literals so that deleting one
# from the corpus fails a test rather than quietly improving the score. `escaping_path`
# is the one the README's threat model implies but does not spell out -- a real file,
# really hashed, from outside the indexed repo. The last three were added by WP5 against
# rules WP4 had just created, and the corpus had no instance of any of them.
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
    "empty_claim",
    "unverifiable_span",
    "zero_length_span",
}

# The shapes where a symbol's stored bytes are NOT its lines' bytes: two modules (whose
# span runs one line past the last line anything is written on), a property (four columns
# in from the start of its line) and a decorated method (whose first byte is the `@`, a
# line above its `def`). These are the symbols a too-narrow gate FALSELY REJECTS, so a
# positive-control suite that does not contain them is measuring the easy half.
DISAGREEING_SHAPES = {"core", "tray", "tray.Tray.widgets", "tray.Tray.count"}

# Attacks that reach `write_assertion` and are admitted by it. Asserted as an exact set
# in both directions, which is why it is empty rather than deleted: a new hole appearing
# fails, and a declared hole being CLOSED also fails, because closing one changes what
# the store guarantees and must not land without the corpus being told.
#
# It held `escaping_path` and no longer does. That is the entry doing its job -- the fix
# could not be merged until this line was edited, which is the difference between a gap
# the suite tracks and a gap somebody wrote down. Empty is a real assertion: no gate
# currently declines to enforce anything the corpus knows how to submit.
STORE_GAPS: frozenset[str] = frozenset()

# Attacks that cannot be stated at a door at all, as opposed to being unrefused there.
SERVER_INEXPRESSIBLE = {"zero_length_span"}

# A family that exists only while a test is running, so the apparatus for a DECLARED gap
# stays under test now that no real gap is declared.
#
# The alternative designs were worse in the same way. A permanent `Unenforced` entry in
# `FAMILIES` would generate real controls against the real gate and sit inside every
# rejection rate as a fabricated hole -- a fake failure inside the number whose job is to
# count failures, which is the exact pretence this module exists to refuse. Deleting the
# tests would keep the code and drop the property, and the property is the expensive
# half: `Unenforced` is what made the escaping-path hole a liability the suite forced
# somebody to settle rather than a comment. So the machinery is exercised against a spec
# built for the purpose and removed afterwards, which is the same move
# `test_a_no_op_mutation_is_not_reported_as_detected` already makes on `FAMILIES`.
UNENFORCED_PROBE = "unenforced_probe"


@pytest.fixture
def unenforced_probe():
    """Install a family one door declines to refuse, for as long as one test needs it."""
    spec = Family(
        name=UNENFORCED_PROBE,
        expect=REFUSED,
        attack="a fixture, not an attack: a rule at one door and nothing at the other",
        gates={
            GATE_SERVER: Rule(
                codes=frozenset({"path_escapes_repo"}),
                mutation=Mutation(
                    rule="a rule that exists, so this door is mutable",
                    edits=(
                        Edit(
                            target="server/app.py",
                            old="    if not target.is_relative_to(root.resolve()):",
                            new="    if False:  # mutated",
                        ),
                    ),
                ),
            ),
            GATE_STORE: Unenforced(
                detail="nothing at this door refuses it; the controls are submitted "
                       "anyway and are scored as the failures they are",
            ),
        },
    )
    FAMILIES[UNENFORCED_PROBE] = spec
    try:
        yield spec
    finally:
        del FAMILIES[UNENFORCED_PROBE]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    """One indexed shapes repo, reused. Controls restore what they edit."""
    return build_harness(tmp_path_factory.mktemp("gate-controls"))


@pytest.fixture(scope="module")
def report(harness):
    return run_controls(harness)


@pytest.fixture(scope="module")
def store_harness(tmp_path_factory):
    """A SECOND repo and index, for the store surface.

    Never shared with the direct one. Both surfaces write, and the stale family rewrites
    files under the repo, so a shared fixture would make each column a measurement of
    the other column's leftovers.
    """
    return build_harness(tmp_path_factory.mktemp("gate-controls-store"), surface=SURFACE_STORE)


@pytest.fixture(scope="module")
def store_report(store_harness):
    return run_controls(store_harness)


@pytest.fixture(scope="module")
def reports(report, store_report):
    return {SURFACE_DIRECT: report, SURFACE_STORE: store_report}


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


def test_the_corpus_covers_every_named_attack(reports):
    """A family with no instances is a rule nothing tests, and it costs nothing to
    generate an empty one by accident: every attack here is built from a symbol that has
    to satisfy a condition. Checked per surface, because a family can be present at one
    door and unstatable at the other -- which is a fact to record, not to average away."""
    assert REQUIRED_ATTACKS <= set(NEGATIVE_FAMILIES)
    assert set(POSITIVE_FAMILIES) == {"published_hash", "quoted_lines", "multi_span"}
    for surface, one in reports.items():
        expected = {
            name for name in FAMILIES
            if not isinstance(FAMILIES[name].at(surface), Inexpressible)
        }
        assert {name for name in FAMILIES if one.family(name)} == expected, surface
        # Floors, not exact counts: the point is that each attack is applied to many
        # symbols rather than demonstrated once, which is what makes a rate a rate.
        assert len(one.negatives) >= 78, surface
        assert len(one.positives) >= 15, surface


def test_every_family_says_what_each_gate_does_with_it(reports):
    """`Family.gates` has no default, and this is why. A family that named only the
    server would be scored against nothing at the store and would vanish out of that
    column -- looking, to a reader comparing the two, like a shorter list rather than a
    hole. The three answers are a rule, a declared absence of one, and an attack that
    cannot be stated there at all, and every family owes each gate one of them."""
    for name, spec in FAMILIES.items():
        assert set(spec.gates) == {GATE_SERVER, GATE_STORE}, name
        for gate, entry in spec.gates.items():
            assert isinstance(entry, Rule | Unenforced | Inexpressible), (name, gate)
            if isinstance(entry, Rule) and spec.expect == REFUSED:
                assert entry.codes, f"{name} at {gate} names no refusal code"
            if isinstance(entry, Rule):
                assert entry.mutation.edits, f"{name} at {gate} has an empty mutation"
    # And the declared exceptions are exactly these, in both directions. `STORE_GAPS` is
    # empty and is still asserted: an `Unenforced` entry added without editing this line
    # is a hole somebody meant to note and nobody had to accept.
    assert {n for n, s in FAMILIES.items() if isinstance(s.gates[GATE_STORE], Unenforced)} == STORE_GAPS
    assert {
        n for n, s in FAMILIES.items() if isinstance(s.gates[GATE_SERVER], Inexpressible)
    } == SERVER_INEXPRESSIBLE
    assert not [n for n, s in FAMILIES.items() if isinstance(s.gates[GATE_SERVER], Unenforced)]


def test_every_skip_is_a_named_reason_rather_than_a_silent_gap(reports):
    """An instance that cannot be constructed is returned with a reason. A skip that
    disappears is how a family becomes empty without anybody's number moving -- and a
    family the surface cannot express at all is the same failure one level up, which is
    why the inexpressible entry leaves a skip rather than simply not appearing."""
    module_reason = (
        "the symbol's stored line range is not a valid line range -- a module ends "
        "one line past its last written line"
    )
    direct, store = reports[SURFACE_DIRECT], reports[SURFACE_STORE]
    assert [(f, s) for f, s, _ in direct.skips] == [
        ("zero_length_span", f"surface={SURFACE_DIRECT}"),
        ("quoted_lines", "core"),
        ("quoted_lines", "tray"),
    ]
    assert direct.skips[0][2] == FAMILIES["zero_length_span"].gates[GATE_SERVER].reason
    assert [(f, r) for f, _, r in direct.skips[1:]] == [("quoted_lines", module_reason)] * 2
    # The store can state everything, so its only skips are the two module shapes.
    assert [(f, r) for f, _, r in store.skips] == [("quoted_lines", module_reason)] * 2


# ---------------------------------------------------------------------------
# the measurement, at each door
# ---------------------------------------------------------------------------

def test_the_server_gate_refuses_every_attack_in_the_corpus(report):
    """The load-bearing claim of the project as the README states it, and now stated
    with the scope it always had: this is the MCP door. Any single acceptance here is a
    defect in the gate, not a reason to soften the corpus."""
    accepted = [o.control for o in report.negatives if not o.refused]
    assert accepted == [], f"the gate ADMITTED an attack: {accepted}"
    assert report.rejection_rate == 1.0
    assert report.known_gaps == []


def test_the_store_gate_refuses_every_attack_in_the_corpus(store_report):
    """The other door, and the number that was never measured.

    `codelearner learn`, the eval harness and every library caller reach
    `write_assertion` without passing `server.app`, so a rule that lives only there is
    a rule those callers do not meet. Repo containment was one, and this test is the
    reason it is not any more: it read `rejection 0.9070` with `escaping_path` admitted,
    stored `active` and reported servable, and the gap set below is what made that a
    debt rather than a note.

    Asserted as a bare 1.0 now, and the arithmetic that used to subtract the gap is
    gone with it. A rate that has to be corrected before it can be compared is a rate
    somebody will eventually compare uncorrected."""
    admitted = {o.family for o in store_report.negatives if not o.refused}
    assert admitted == set(), f"the store ADMITTED an attack: {sorted(admitted)}"
    assert store_report.known_gaps == []
    assert store_report.unexpected_failures == []
    assert store_report.rejection_rate == 1.0
    assert store_report.attributed_rate == 1.0


def test_the_escaping_path_attack_is_refused_at_both_doors_and_leaves_no_row(
    harness, store_harness
):
    """The rule that was missing, put back and checked through the corpus that found it.

    The store's copy is deliberately LEXICAL where the server's resolves the path, and
    the asymmetry is the point: `verify=False` is a supported call, so a containment
    check that needed the disk would be a rule only the re-reading caller met -- which
    is the shape of the bug being fixed. `tests/test_assertions.py` pins the lexical
    behaviour directly, on paths this corpus does not generate; what is asserted here is
    that the corpus's own escaping control, built from a real file with its real hash,
    is refused at both doors under each door's own name.

    Admitted-and-servable is still reproducible -- under mutation, which is where a
    reproduction of a fixed bug belongs. See the `escaping_path@store` mutation."""
    for surface, one in ((SURFACE_DIRECT, harness), (SURFACE_STORE, store_harness)):
        controls, _ = build_corpus(one.files, surface=surface)
        escaping = [c for c in controls if c.family == "escaping_path"]
        assert escaping, f"{surface}: the escaping-path family lost its instances"
        one.restore()
        before = one.rows()
        payload = one.submit(escaping[0])
        assert payload["ok"] is False, f"{surface} ADMITTED a citation from outside the repo"
        assert payload["error"]["code"] in FAMILIES["escaping_path"].codes(surface)
        assert one.rows() == before
        assert escaping[0].spans[0].path.startswith("../")
    # Each door answers under its OWN name. The store cannot resolve the path -- it does
    # not know the root at the moment it checks -- so its code is not, and must not be,
    # the server's.
    assert FAMILIES["escaping_path"].codes(SURFACE_DIRECT) == frozenset({"path_escapes_repo"})
    assert FAMILIES["escaping_path"].codes(SURFACE_STORE) == frozenset({"span_escapes_repo"})


def test_every_refusal_names_the_rule_that_produced_it_at_the_door_it_used(reports):
    """Rejection rate alone is not enough. A harness pointed at the wrong repo root
    refuses every submission with one code and scores a perfect 1.0; requiring
    `bad_range` from the beyond-EOF attack at the server and `evidence_stale` from it at
    the store turns that failure into a wrong code instead of a right rate.

    The codes are read per gate, and that is the conjunct WP5 had to add. The same
    attack is refused under different names at the two doors, so a single acceptable-code
    set would have had to be the union of both -- under which a server-surface refusal
    carrying a store-only code would score as correctly attributed."""
    for surface, one in reports.items():
        wrong = [
            (o.control, o.code)
            for o in one.negatives
            if o.code not in FAMILIES[o.family].codes(surface) and not o.known_gap
        ]
        assert wrong == [], surface
    direct, store = reports[SURFACE_DIRECT], reports[SURFACE_STORE]
    assert direct.attributed_rate == 1.0
    # Each attack is refused by its OWN rule, so the codes must not collapse into one.
    assert {code for name in NEGATIVE_FAMILIES for code in direct.codes(name)} == {
        "evidence_required",
        "empty_claim",
        "evidence_unverifiable",
        "file_missing",
        "path_escapes_repo",
        "bad_range",
        "hash_mismatch",
        "unknown_subject",
    }
    # The store's vocabulary is SMALLER, and that is a finding rather than a detail: it
    # has no notion of an unindexed file and no line-range rule, so four distinct
    # attacks arrive at one code. A library caller gets a message blaming an edit for a
    # path that was never in the index.
    assert {code for name in NEGATIVE_FAMILIES for code in store.codes(name)} == {
        "evidence_required",
        "empty_claim",
        "evidence_unverifiable",
        "invalid_span",
        "span_escapes_repo",
        "unknown_subject",
        "evidence_stale",
    }


def test_a_refusal_never_leaves_a_row_behind(reports):
    """A gate that says no and writes the row anyway has refused nothing -- and every
    later stage would treat that row as admitted evidence.

    Scoped to the attacks that were actually REFUSED, because at the store one is not:
    the escaping-path control leaves a row precisely because it was admitted, and
    folding that into this assertion would hide a hole behind a rule about refusals.
    That control is asserted separately, as an admission, where it belongs."""
    for surface, one in reports.items():
        leaked = [o.control for o in one.negatives if o.refused and o.rows_added != 0]
        assert leaked == [], surface
        admitted = [o for o in one.negatives if not o.refused]
        assert all(o.rows_added == 1 and o.known_gap for o in admitted), surface


def test_both_gates_admit_every_legitimate_submission(reports):
    """Without this the suite is unfalsifiable: a gate that refuses everything scores a
    perfect rejection rate. The four shapes where a symbol's stored bytes are not its
    lines' bytes are named explicitly, because they are the ones a narrowed gate rejects
    while still refusing every attack."""
    for surface, one in reports.items():
        refused = [(o.control, o.code) for o in one.positives if o.verdict != ACCEPTED]
        assert refused == [], f"{surface}: the gate REFUSED a correct citation: {refused}"
        assert one.positive_pass_rate == 1.0, surface
        admitted = {o.control.split("/", 1)[1] for o in one.family("published_hash")}
        assert DISAGREEING_SHAPES <= admitted, surface


def test_an_admitted_claim_is_servable_and_keeps_all_of_its_evidence(reports):
    """Accepted is not the whole promise. The claim has to survive the second
    verification run against disk, and it has to store every span it was submitted with
    -- a claim quietly reduced to one of its two citations stands on less than its author
    thought, and nothing downstream would record that. At the store surface the span
    count is read back out of `evidence_spans` rather than echoed, or this conjunct would
    be true by construction."""
    for surface, one in reports.items():
        for outcome in one.positives:
            assert outcome.servable is True, (surface, outcome.control)
            assert outcome.rows_added == 1, (surface, outcome.control)
            assert outcome.evidence == outcome.expected_evidence, (surface, outcome.control)
        assert [o.evidence for o in one.family("multi_span")] == [2, 2], surface


def test_each_report_names_the_module_that_actually_answered(reports):
    """The corpus is worth nothing if it is scored against a reimplementation, and worth
    less than nothing if it reports one door's rate under the other's name. The report
    records which file holds the rules it measured."""
    direct, store = reports[SURFACE_DIRECT], reports[SURFACE_STORE]
    assert direct.gate_path == str(gate_module().__file__)
    assert direct.gate_path.endswith("codelearner/server/app.py")
    assert store.gate_path == str(store_module().__file__)
    assert store.gate_path.endswith("codelearner/assertions/store.py")
    assert direct.gate == GATE_SERVER
    assert store.gate == GATE_STORE


def test_the_store_surface_can_be_measured_without_the_mcp_extra_installed():
    """The whole reason the store surface exists is that its callers do not have a
    server. A harness that reached for `server.app` to get a connection would make this
    column unreportable on exactly the machines whose gate was one lock short, and the
    import would be invisible -- everything would still pass, here, where `mcp` is
    installed. Asserted by running the corpus in a fresh interpreter and looking at what
    it imported."""
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys, tempfile, pathlib\n"
            "from codelearner.eval.gate_controls import build_harness, run_controls\n"
            "work = pathlib.Path(tempfile.mkdtemp())\n"
            "h = build_harness(work, surface='store')\n"
            "r = run_controls(h, only=['zero_evidence', 'published_hash'])\n"
            "assert r.positive_pass_rate == 1.0\n"
            "leaked = sorted(m for m in sys.modules if m.startswith(('mcp', 'codelearner.server')))\n"
            "print('LEAKED=' + ','.join(leaked))\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "LEAKED=" in proc.stdout, proc.stdout
    assert proc.stdout.strip().endswith("LEAKED="), (
        f"the store surface imported the server: {proc.stdout.strip()}"
    )


def test_the_two_gates_speak_the_same_refusal_vocabulary():
    """`gate_controls` keeps its own copy of the exception-to-code mapping so the store
    surface does not have to import the server to name a refusal. Two copies can drift,
    and a drift here would show up as a family scoring 0.0 with no explanation -- so the
    copies are compared rather than trusted."""
    app = gate_module()
    assert store_refusal_codes() == app._STORE_REFUSAL_CODES


# ---------------------------------------------------------------------------
# the two doors, compared
# ---------------------------------------------------------------------------

def test_the_same_attack_is_refused_by_different_rules_at_the_two_doors(reports):
    """The finding the two-column report exists to make visible. These four attacks are
    refused at both doors, and by different rules -- the server knows about indexed
    files and line ranges, and the store knows only whether the cited bytes hash to what
    was cited. Naming them here means a future change that collapsed the store's answers
    into the server's (or vice versa) fails a test rather than quietly making the columns
    agree."""
    direct, store = reports[SURFACE_DIRECT], reports[SURFACE_STORE]
    for family, server_code, store_code in [
        ("absent_file", "file_missing", "evidence_stale"),
        ("past_eof", "bad_range", "evidence_stale"),
        ("blank_range", "bad_range", "invalid_span"),
        ("foreign_symbol_hash", "hash_mismatch", "evidence_stale"),
    ]:
        assert set(direct.codes(family)) == {server_code}, family
        assert set(store.codes(family)) == {store_code}, family
        assert direct.hold_rate(family) == store.hold_rate(family) == 1.0, family


def test_the_comparison_reports_agreement_on_rates_and_disagreement_on_names(tmp_path):
    """One table, both columns, from one call -- because two rates printed by two
    separate runs are two facts nobody puts next to each other, and that is precisely how
    "100% refused" survived as a description of one of two doors.

    `divergent()` is empty now and was `['escaping_path']` when this was written: the
    one family whose hold rate differed, at 1.000 against 0.000. Empty is the goal
    state, and it is NOT the same as the two doors being the same gate --
    `differently_named()` stays non-empty by design, and asserting both is what stops
    an empty divergence list reading as "there is only one gate now"."""
    comparison = compare_surfaces(tmp_path / "compare", limit=2)
    assert set(comparison.surfaces) == {SURFACE_DIRECT, SURFACE_STORE}
    assert comparison.divergent() == []
    assert set(comparison.differently_named()) == {
        "absent_file", "escaping_path", "past_eof", "blank_range",
        "decoy_content_hash", "stale_but_once_valid", "foreign_symbol_hash",
    }
    assert comparison.differently_named()["escaping_path"] == {
        SURFACE_DIRECT: ["path_escapes_repo"], SURFACE_STORE: ["span_escapes_repo"],
    }
    table = comparison.format_table()
    assert "every family scores the same at every door." in table
    assert "refused at every door, under different names:" in table
    assert "KNOWN GAP" not in table
    assert comparison.reports[SURFACE_DIRECT].gate_path.endswith("app.py")
    assert comparison.reports[SURFACE_STORE].gate_path.endswith("store.py")


def test_the_tool_surface_returns_the_same_verdict_as_the_tool_body(tmp_path):
    """The corpus is scored through the tool body; an agent reaches it through the
    registered MCP tool. If those ever disagree, the measurement is about a code path
    nobody calls -- and a refusal that arrives as a traceback rather than as data is a
    refusal the agent reads as a broken server. Both map to the same GATE, so unlike the
    store column these two must agree control for control."""
    direct = run_controls(build_harness(tmp_path / "direct"))
    tool = run_controls(build_harness(tmp_path / "tool", surface=SURFACE_TOOL))
    assert gate_of(SURFACE_TOOL) == gate_of(SURFACE_DIRECT)
    assert [(o.control, o.verdict, o.code) for o in direct.outcomes] == [
        (o.control, o.verdict, o.code) for o in tool.outcomes
    ]
    assert tool.rejection_rate == 1.0
    assert tool.positive_pass_rate == 1.0
    assert "raised_into_transport" not in {o.code for o in tool.outcomes}


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
        fact = harness.files[span.path]
        accepted = fact.accepted_hashes(span.line_start, span.line_end)
        assert accepted, f"{control.name} cites lines with no acceptable hash at all"
        assert span.content_hash not in accepted, control.name
        checked += 1
    assert checked >= 12, "the hash families lost their instances"


def test_a_blank_line_is_refused_by_range_rather_than_by_hash(harness):
    """sha256 of nothing is a perfectly stable hash, so an empty span would verify
    forever while pointing at nothing. Both channels are tried -- the empty text and the
    hash of the empty text -- because either one getting in is the same hole."""
    controls, _ = build_corpus(harness.files)
    blank = [c for c in controls if c.family == "blank_range"]
    assert len(blank) == 4
    cited = {c.spans[0].cited_hash() for c in blank}
    assert cited == {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
    assert all(c.spans[0].byte_start == c.spans[0].byte_end for c in blank)


def test_the_zero_length_span_cites_a_line_that_is_full_of_code(store_harness):
    """The attack WP5 added, and the reason it is not a duplicate of `blank_range`.

    `blank_range` cites a line with no bytes on it. This one cites a line with a symbol's
    first byte on it and then asks for none of those bytes -- which is what a caller
    computing offsets wrongly actually produces, and what was reproduced as admitted,
    servable, and permanently fresh before the rule moved to the chokepoint. An empty
    range hashes to a stable value, so it re-verifies against the file as it is, as it
    becomes, and as it would be after the symbol it points at is deleted."""
    controls, _ = build_corpus(store_harness.files, surface=SURFACE_STORE)
    zero = [c for c in controls if c.family == "zero_length_span"]
    assert len(zero) == 8
    for control in zero:
        cite = control.spans[0]
        fact = store_harness.files[cite.path]
        assert cite.byte_start == cite.byte_end
        # The line range it names is one a reader can open, so nothing about the LINES
        # is wrong -- which is what made this survive every check that looked at them.
        assert fact.line_bytes(cite.line_start, cite.line_start) is not None
        # The cited line is not blank: this is not `blank_range` under another name.
        assert cite.line_start not in fact.blank_lines(), control.name
        # And the bytes it declines to cite are really there.
        assert cite.byte_start < len(fact.source)


def test_the_zero_length_span_cannot_be_stated_to_the_server_at_all(harness):
    """Amendment #2, applied to a family this work package added. The MCP surface takes
    line numbers and derives the byte range itself, so no caller can ask it for an empty
    range at a non-blank line -- the only empty range expressible there is a blank line,
    which is a different family. This is recorded as `Inexpressible` with a reason rather
    than as a family that happens to generate nothing, because "the corpus cannot express
    this here" and "the corpus forgot to generate this here" are indistinguishable from
    the outside."""
    entry = FAMILIES["zero_length_span"].at(SURFACE_DIRECT)
    assert isinstance(entry, Inexpressible)
    assert "byte range" in entry.reason
    controls, skips = build_corpus(harness.files, surface=SURFACE_DIRECT)
    assert [c for c in controls if c.family == "zero_length_span"] == []
    assert ("zero_length_span", f"surface={SURFACE_DIRECT}", entry.reason) in skips
    assert "zero_length_span" not in mutable_families(SURFACE_DIRECT)


def test_an_empty_claim_is_refused_at_both_doors_by_the_stores_rule(reports):
    """The claim text is the only part of a submission no arithmetic can check, which is
    why it was the part with no rule: every span hash-matched, the subject was real, and
    the row stored `active` and served, saying nothing. The rule lives only in
    `write_assertion`, so the server column here is measuring the store's rule reached
    through the tool -- worth asserting rather than assuming, because it means
    `_submit_body` contributes nothing and a reader must not conclude otherwise."""
    for surface, one in reports.items():
        assert one.hold_rate("empty_claim") == 1.0, surface
        assert set(one.codes("empty_claim")) == {"empty_claim"}, surface
    assert FAMILIES["empty_claim"].rule(SURFACE_DIRECT).mutation.edits == (
        FAMILIES["empty_claim"].rule(SURFACE_STORE).mutation.edits
    ), "the same rule, so the same edit deletes it at both doors"


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


# Every (family, door) pair that has a rule to delete. Built from the declaration rather
# than listed, so a family gaining a rule at a door it did not have one at is mutated
# from the next run onwards without anybody remembering to add it here -- and a family
# LOSING one shows up in `test_every_family_says_what_each_gate_does_with_it` instead of
# quietly leaving this list.
MUTATION_CASES = [
    pytest.param(family, surface, id=f"{family}@{surface}")
    for surface in (SURFACE_DIRECT, SURFACE_STORE)
    for family in mutable_families(surface)
]

# How far each positive family's hold rate falls when the leniency it names is deleted,
# and WHICH controls flip. Exact sets, not a floor and not a superset: the audit found
# the previous version of this assertion short-circuiting on `or family == "multi_span"`,
# which is the tautology `A <= B or True`. A positive control that flipped everywhere
# would mean the rule it names is doing something broader than it claims, and one that
# flipped nowhere would mean it does nothing -- only the exact set distinguishes those.
EXPECTED_POSITIVE_FLIPS: dict[tuple[str, str], set[str]] = {
    # Removing the SYMBOL reading falsely rejects exactly the symbols whose stored bytes
    # are not their lines' bytes -- half this fixture, and 217 of 850 (25.5%) of the
    # repository. That population is the reason the rule exists.
    ("published_hash", SURFACE_DIRECT): {f"published_hash/{n}" for n in DISAGREEING_SHAPES},
    # Removing the WHOLE-LINES reading rejects only the symbols that have one and
    # disagree: the two modules have no valid line range, so they were never in this
    # family at all.
    ("quoted_lines", SURFACE_DIRECT): {
        "quoted_lines/tray.Tray.widgets", "quoted_lines/tray.Tray.count",
    },
    ("multi_span", SURFACE_DIRECT): {"multi_span/core.py", "multi_span/tray.py"},
    # At the store the leniency is different: verification is scoped to the CITED range,
    # and widening it to the whole file rejects every symbol that is not the whole file.
    # The two modules ARE the whole file, so they survive -- which is what makes 0.250
    # the right answer here and evidence that the mutation removed what it claims to.
    ("published_hash", SURFACE_STORE): {
        f"published_hash/{n}"
        for n in ("core.frobnicate_widgets", "core._plumbing", "tray.memoize",
                  "tray.Tray", "tray.Tray.widgets", "tray.Tray.count")
    },
    ("quoted_lines", SURFACE_STORE): {
        f"quoted_lines/{n}"
        for n in ("core.frobnicate_widgets", "core._plumbing", "tray.memoize",
                  "tray.Tray", "tray.Tray.widgets", "tray.Tray.count")
    },
    ("multi_span", SURFACE_STORE): {"multi_span/core.py", "multi_span/tray.py"},
}


@pytest.mark.parametrize(("family", "surface"), MUTATION_CASES)
def test_each_control_detects_the_deletion_of_the_rule_it_names(
    family, surface, mutant_dir, reports
):
    """The one test that decides whether any of the others mean anything.

    The rule each family targets AT THIS DOOR is deleted from a COPY of the package and
    the family is re-measured in a subprocess importing that copy. A control whose
    verdict does not move when its own rule is gone is decoration: it would keep passing
    through the regression it exists to catch.

    Parametrised over doors as well as families because of amendment #3. WP4's
    `unknown_subject` mutation stopped flipping when the rule gained a second home --
    deleting the server's copy left the store's refusing the same attack with the same
    code, so the family reported its own rule as undeletable. Running one mutation at one
    surface can only ever establish that SOME rule refused the attack; running the
    per-door mutation at each door establishes which."""
    result = run_mutation(family, mutant_dir, baseline=reports[surface], surface=surface)
    assert result.surface == surface
    assert result.baseline_rate == 1.0
    assert result.detected, (
        f"deleting {result.rule!r} at the {surface} door did not change the {family} "
        f"verdict ({result.baseline_rate} -> {result.mutant_rate}); the control cannot "
        "see its own rule being removed"
    )
    assert result.mutant_n == len(reports[surface].family(family))
    assert result.flipped, "detection was reported with no control naming itself"
    if FAMILIES[family].expect == REFUSED:
        assert result.mutant_rate == 0.0, "the attack should succeed with the rule gone"
        assert set(result.flipped) == {
            o.control for o in reports[surface].family(family)
        }
    else:
        assert set(result.flipped) == EXPECTED_POSITIVE_FLIPS[(family, surface)]


def test_every_family_is_mutation_verified_at_every_door_that_has_a_rule():
    """The bookkeeping the parametrisation rests on, asserted rather than assumed.

    A rule with two homes needs a mutation for each, and the failure mode is a family
    that silently has a rule at one door and no case for it -- which reads as thorough
    coverage right up until that door's rule is deleted for real. The two entries that
    are NOT mutation-verified are named individually, with why, because "we could not
    test this" is a finding and "we did not notice" looks identical to it in a passing
    suite."""
    cases = {(family, surface) for family, surface, *_ in [p.values for p in MUTATION_CASES]}
    for name, spec in FAMILIES.items():
        for surface in (SURFACE_DIRECT, SURFACE_STORE):
            if isinstance(spec.at(surface), Rule):
                assert (name, surface) in cases, f"{name}@{surface} has a rule and no mutation"
            else:
                assert (name, surface) not in cases
    assert {(n, s) for n in FAMILIES for s in (SURFACE_DIRECT, SURFACE_STORE)} - cases == {
        # Nothing to submit: the server derives byte ranges from line numbers. The one
        # remaining entry, where there were two -- `escaping_path@store` left this set
        # when the store gained a containment rule, which is the only way an entry
        # should ever leave it.
        ("zero_length_span", SURFACE_DIRECT),
    }
    # Both doors are actually exercised, rather than one door twice.
    assert len({surface for _, surface in cases}) == 2


def test_a_no_op_mutation_is_not_reported_as_detected(mutant_dir, report):
    """The mutation harness has to be able to say no. If an empty edit list still comes
    back 'detected', then every detection above is an artefact of copying the tree or of
    running in a subprocess rather than of the rule being gone."""
    family = "past_eof"
    saved = FAMILIES[family]
    unchanged = Family(
        name=saved.name,
        expect=saved.expect,
        attack=saved.attack,
        gates={**saved.gates, GATE_SERVER: Rule(
            codes=saved.gates[GATE_SERVER].codes, mutation=Mutation("no-op", ())
        )},
    )
    FAMILIES[family] = unchanged
    try:
        result = run_mutation(family, mutant_dir / "noop", baseline=report)
    finally:
        FAMILIES[family] = saved
    assert result.mutant_rate == result.baseline_rate == 1.0
    assert result.detected is False


def test_a_mutation_measured_against_the_other_doors_baseline_refuses_to_run(mutant_dir, store_report):
    """The comparison that would look right and mean nothing. `absent_file` holds at
    1.000 at both doors, so a store-surface mutant scored against the SERVER baseline
    would report `1.000 -> 0.000  detected` while comparing two different rules'
    columns. The rates being equal today is exactly what makes this silent."""
    with pytest.raises(MutationFailed, match="surface"):
        run_mutation(
            "absent_file", mutant_dir / "crossed", baseline=store_report,
            surface=SURFACE_DIRECT,
        )


def test_a_family_with_no_rule_at_a_door_refuses_to_be_mutated(
    mutant_dir, store_report, unenforced_probe
):
    """There is nothing to delete, so there is nothing to detect, and a harness that
    quietly returned `not detected` would say the control is decoration when the truth is
    that the rule is absent. Two different repairs; the message has to distinguish them.

    Run against a family installed for this test, because no real one is unenforced any
    more. Keeping the property is worth more than the family that happened to exercise
    it -- `escaping_path` was the second gap this apparatus was pointed at and it will
    not be the last, and a machine for handling declared gaps that is only tested while
    a gap exists is untested exactly when the next one arrives."""
    assert isinstance(FAMILIES[UNENFORCED_PROBE].at(SURFACE_STORE), Unenforced)
    assert UNENFORCED_PROBE not in mutable_families(SURFACE_STORE)
    assert UNENFORCED_PROBE in mutable_families(SURFACE_DIRECT)
    with pytest.raises(MutationFailed, match="no rule at the"):
        run_mutation(
            UNENFORCED_PROBE, mutant_dir / "nogap", baseline=store_report,
            surface=SURFACE_STORE,
        )


@pytest.mark.parametrize("surface", [SURFACE_DIRECT, SURFACE_STORE])
def test_an_unmutated_copy_scores_exactly_what_the_working_tree_scores(
    surface, mutant_dir, reports
):
    """The other half of the same doubt. A mutant is compared against a baseline measured
    in the working tree, so if copying the package changed the score by itself, every
    mutation result would be measuring the copy."""
    one = reports[surface]
    payload = run_unmutated_copy(mutant_dir / f"unmutated-{surface}", surface=surface)
    assert payload["rejection_rate"] == one.rejection_rate
    assert payload["positive_pass_rate"] == one.positive_pass_rate == 1.0
    assert [f["control"] for f in payload["failures"]] == [o.control for o in one.failures]
    assert payload["known_gaps"] == sorted({o.family for o in one.known_gaps})
    assert payload["gate_path"].startswith(str(mutant_dir / f"unmutated-{surface}"))


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


def test_a_store_surface_mutation_is_measured_against_the_copied_store(mutant_dir, store_report):
    """The same isolation claim for the second door, and it is not implied by the first:
    the store surface reports `assertions/store.py` as its gate path, so a child that
    imported the installed package would fail the containment check on a DIFFERENT file
    than the one the direct surface checks."""
    result = run_mutation(
        "zero_length_span", mutant_dir / "store-isolation", baseline=store_report,
        surface=SURFACE_STORE,
    )
    assert result.gate_path.startswith(str(mutant_dir / "store-isolation"))
    assert result.gate_path.endswith("assertions/store.py")
    assert result.gate_path != str(store_module().__file__)


def test_the_working_tree_is_never_edited_by_a_mutation(mutant_dir, report, store_report):
    """Mutating an installed package in place leaves a window in which any other process
    importing it sees the hole, and a crash mid-run leaves the hole behind for good. Both
    files now carry mutated rules, so both are checked."""
    watched = [gate_module().__file__, store_module().__file__]
    before = [open(path, "rb").read() for path in watched]  # noqa: SIM115
    run_mutation("decoy_content_hash", mutant_dir / "untouched", baseline=report)
    run_mutation(
        "blank_range", mutant_dir / "untouched-store", baseline=store_report,
        surface=SURFACE_STORE,
    )
    assert [open(path, "rb").read() for path in watched] == before  # noqa: SIM115


# ---------------------------------------------------------------------------
# the scoring function, scored
# ---------------------------------------------------------------------------

# One negative family and one positive family, named as literals rather than stubbed:
# `held` reads its expectation and its acceptable codes out of FAMILIES, so both branches
# have to be exercised through real spec entries or the table is testing a fixture.
ATTACK_FAMILY = "past_eof"        # expect REFUSED, codes == {"bad_range"} at the server
LEGITIMATE_FAMILY = "multi_span"  # expect ACCEPTED, two spans submitted, both stored
# expect REFUSED, with a rule at one door and nothing at the other. Installed by the
# `unenforced_probe` fixture rather than named from FAMILIES: it WAS `escaping_path`,
# and the point of these rows is the scoring behaviour, not the family that once had it.
UNGUARDED_FAMILY = UNENFORCED_PROBE


def _attack(**overrides) -> Outcome:
    """An attack the gate refused exactly as promised, before overrides."""
    return Outcome(**{
        "control": f"{ATTACK_FAMILY}/tray.Tray.count",
        "family": ATTACK_FAMILY,
        "verdict": REFUSED,
        "code": "bad_range",
        "rows_added": 0,
        "evidence": 0,
        "expected_evidence": 0,
        "servable": None,
        **overrides,
    })


def _legitimate(**overrides) -> Outcome:
    """A correct citation the gate admitted exactly as promised, before overrides."""
    return Outcome(**{
        "control": f"{LEGITIMATE_FAMILY}/tray.Tray.widgets",
        "family": LEGITIMATE_FAMILY,
        "verdict": ACCEPTED,
        "code": None,
        "rows_added": 1,
        "evidence": 2,
        "expected_evidence": 2,
        "servable": True,
        **overrides,
    })


# Each row is one way a control can fall short of what its family promised, plus the two
# rows that have to hold or an always-False `held` would pass this table. The wrong-code
# and leaked-row rows are not hypothetical shapes: they are the two failures the gate
# report is specifically supposed to distinguish from a clean refusal.
#
# The first twelve rows are unchanged from WP6 and are the reason this table exists.
# `Outcome` gained a `surface` field, defaulted to `direct`, so each of them still
# asserts exactly what it asserted before; the rows after them are the surface axis WP5
# added, and they are additions rather than substitutions.
HELD_TABLE = [
    # -- negative branch: refused, by the named rule, with nothing written -----
    pytest.param(_attack(), True, id="refused-by-its-own-rule-leaving-no-row"),
    pytest.param(
        _attack(rows_added=1), False,
        id="refused-but-a-row-was-written",
    ),
    pytest.param(
        _attack(code="file_missing"), False,
        id="refused-for-the-wrong-reason",
    ),
    pytest.param(
        _attack(code=None), False,
        id="refused-with-no-code-at-all",
    ),
    pytest.param(
        _attack(verdict=ACCEPTED, code="bad_range"), False,
        id="attack-admitted-while-still-carrying-a-refusal-code",
    ),
    # -- positive branch: accepted, servable, one row, every span kept ---------
    pytest.param(_legitimate(), True, id="admitted-servable-and-whole"),
    pytest.param(
        _legitimate(servable=False), False,
        id="admitted-but-not-servable",
    ),
    pytest.param(
        _legitimate(servable=None), False,
        id="admitted-but-servability-never-established",
    ),
    pytest.param(
        _legitimate(evidence=1), False,
        id="admitted-having-silently-dropped-a-span",
    ),
    pytest.param(
        _legitimate(rows_added=0), False,
        id="admitted-but-nothing-was-stored",
    ),
    pytest.param(
        _legitimate(rows_added=2), False,
        id="admitted-and-stored-twice",
    ),
    pytest.param(
        _legitimate(verdict=REFUSED), False,
        id="legitimate-citation-refused",
    ),
    # -- the surface axis: the same code means different things at each door ---
    pytest.param(
        _attack(surface=SURFACE_STORE, code="evidence_stale"), True,
        id="store-refusal-carrying-the-stores-own-code",
    ),
    pytest.param(
        _attack(surface=SURFACE_STORE), False,
        id="store-refusal-carrying-the-servers-code-for-the-same-attack",
    ),
    pytest.param(
        _attack(surface=SURFACE_DIRECT, code="evidence_stale"), False,
        id="server-refusal-carrying-the-stores-code-for-the-same-attack",
    ),
    # -- a door with no rule at all can never hold, whatever it returns --------
    pytest.param(
        _attack(family=UNGUARDED_FAMILY, control=f"{UNGUARDED_FAMILY}/core",
                surface=SURFACE_STORE, verdict=ACCEPTED, code=None, rows_added=1), False,
        id="unenforced-gate-admitting-the-attack",
    ),
    pytest.param(
        _attack(family=UNGUARDED_FAMILY, control=f"{UNGUARDED_FAMILY}/core",
                surface=SURFACE_STORE, code="path_escapes_repo"), False,
        id="unenforced-gate-refusing-it-anyway-still-does-not-hold",
    ),
    pytest.param(
        _attack(family=UNGUARDED_FAMILY, control=f"{UNGUARDED_FAMILY}/core",
                code="path_escapes_repo"), True,
        id="the-same-family-at-the-door-that-does-have-the-rule",
    ),
]


@pytest.mark.parametrize(("outcome", "expected"), HELD_TABLE)
def test_held_is_false_for_every_way_a_control_can_fall_short(
    outcome, expected, unenforced_probe
):
    """`held` is the predicate every number in the gate report is computed through, and
    until this table existed three of its four conjuncts could be deleted with the whole
    suite green.

    `held` feeds `hold_rate`, `attributed_rate` and `positive_pass_rate`, and through
    `MutationResult.detected` it decides whether each family can see its own rule being
    removed. The properties it checks ARE tested directly elsewhere -- a refusal leaves
    no row, a refusal names its rule, an admission is servable and keeps its evidence --
    but those tests run over the corpus as it is today, in which no control has ever
    produced a row-leaking refusal, a wrong code or a non-servable admission. A conjunct
    that never fires on real data is a conjunct no corpus-driven test can pin, so the
    scoring function drifts free of the checks while the reported rate still reads 1.000.
    A future rule that refused and wrote the row anyway would be scored as a clean
    refusal by the very apparatus whose job is to notice.

    The surface rows are the same argument for the conjunct WP5 added. `held` now reads
    the acceptable codes out of the family's entry FOR THE GATE THE CONTROL USED, and
    nothing in a corpus run can exercise the mismatch -- each surface only ever produces
    its own codes. Constructing the crossed pair is the only way to establish that the
    lookup is by door rather than by family.

    This is the failure `assertions/store.py` names as the reason nothing is ever deleted
    -- a pipeline that scores its own rejections can report any pass rate it likes --
    one level up, in the scorer rather than in the store. Hence synthetic outcomes: the
    only way to reach a state the corpus cannot currently reach is to construct it.
    """
    assert outcome.held is expected


def test_the_held_table_exercises_both_branches_and_both_doors(unenforced_probe):
    """The negative and positive branches of `held` share no conjunct except the verdict,
    so a table that drifted to one side would leave the other unscored while still
    looking thorough. The two families are read out of FAMILIES for the same reason: if
    `past_eof` ever stopped expecting a refusal, or its code vocabulary grew to include
    the wrong-reason code used above, the rows built on it would silently stop
    discriminating."""
    assert FAMILIES[ATTACK_FAMILY].expect == REFUSED
    assert FAMILIES[LEGITIMATE_FAMILY].expect == ACCEPTED
    assert "file_missing" not in FAMILIES[ATTACK_FAMILY].codes(SURFACE_DIRECT)
    assert "bad_range" in FAMILIES[ATTACK_FAMILY].codes(SURFACE_DIRECT)
    # The crossed rows only discriminate while the two doors really do disagree here.
    assert FAMILIES[ATTACK_FAMILY].codes(SURFACE_DIRECT) == frozenset({"bad_range"})
    assert FAMILIES[ATTACK_FAMILY].codes(SURFACE_STORE) == frozenset({"evidence_stale"})
    assert isinstance(FAMILIES[UNGUARDED_FAMILY].at(SURFACE_STORE), Unenforced)
    branches = {FAMILIES[case.values[0].family].expect for case in HELD_TABLE}
    assert branches == {REFUSED, ACCEPTED}
    assert {case.values[0].surface for case in HELD_TABLE} == {SURFACE_DIRECT, SURFACE_STORE}
    # Both polarities on both sides: without a held=True row per branch, a `held` that
    # returned False unconditionally would satisfy every other row in the table.
    for expect in (REFUSED, ACCEPTED):
        verdicts = {
            expected
            for case in HELD_TABLE
            for outcome, expected in [case.values]
            if FAMILIES[outcome.family].expect == expect
        }
        assert verdicts == {True, False}


def test_a_report_that_scores_a_leaked_row_as_a_refusal_does_not_reach_1_0():
    """The reason the table above is not merely pedantic. `attributed_rate` is the number
    the README quotes as proof the controls are not vacuous, and it is `held` summed. A
    single attack refused with a row left behind has to move it off 1.0 -- if it does
    not, the headline rate is reporting the gate's intent rather than its behaviour."""
    leaky = GateReport(
        corpus="synthetic",
        symbols=0,
        outcomes=[_attack(), _attack(control="past_eof/core", rows_added=1)],
        skips=[],
        gate_path="?",
    )
    assert leaky.rejection_rate == 1.0, "both were refused, so the crude rate is blind"
    assert leaky.attributed_rate == 0.5
    assert [o.control for o in leaky.failures] == ["past_eof/core"]

    mixed = GateReport(
        corpus="synthetic",
        symbols=0,
        outcomes=[_legitimate(), _legitimate(control="multi_span/core", servable=False)],
        skips=[],
        gate_path="?",
    )
    assert mixed.positive_pass_rate == 0.5


def test_a_known_gap_stays_inside_the_rate_it_is_a_gap_in(unenforced_probe):
    """The one way declaring a hole could make things worse: if `Unenforced` were treated
    as "expected, therefore fine", the rejection rate would read 1.000 with an attack
    walking through. The gap is separated from the OTHER failures for triage and for the
    exit code, and it is separated nowhere else -- it is still a failure, still counted
    against the rate, and still listed.

    That distinction is the reason the store column read 0.9070 rather than 1.000 for as
    long as the containment hole was open, and the reason the hole got fixed."""
    outcomes = [
        _attack(),
        _attack(family=UNGUARDED_FAMILY, control=f"{UNGUARDED_FAMILY}/core",
                surface=SURFACE_STORE, verdict=ACCEPTED, code=None, rows_added=1),
    ]
    one = GateReport(
        corpus="synthetic", symbols=0, outcomes=outcomes, skips=[], gate_path="?",
        surface=SURFACE_STORE,
    )
    assert one.rejection_rate == 0.5
    assert one.attributed_rate == 0.5
    assert [o.control for o in one.known_gaps] == [f"{UNGUARDED_FAMILY}/core"]
    assert one.unexpected_failures == []
    assert [o.control for o in one.failures] == [f"{UNGUARDED_FAMILY}/core"]
