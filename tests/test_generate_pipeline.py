"""The generation pipeline: the menu, the refusals, and what a re-run does.

Driven entirely by deterministic fake `ClaimGenerator`s through the protocol, on a
real temp repo with a real index and the real store -- not a mock of it. Both halves
matter. No test here may reach a model (`_no_network` makes that a failure rather than
a slow machine-dependent pass, the same guard `test_faithfulness.py` uses), and no test
here may assert against a stubbed `write_assertion`, because every rule under test is a
rule about what ends up in the assertion tables.

Same standard as `test_assertions.py`: every test names a rule that would otherwise
fail silently, and deleting the rule has to turn it red. Two of them are the ones the
project cannot afford to get wrong twice:

* a draft whose references all miss admits NOTHING -- no fallback citation, no row;
* no evidence span anywhere in the store can be traced to generator output, asserted
  structurally by comparing every stored span against what the index published.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import shutil
import urllib.error
import urllib.request

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.generate.pipeline import (
    # Private on purpose, and imported anyway: `_OUTCOME_COUNTERS` is the mapping that
    # keeps the report's partition identity true, and the test below is the only thing
    # that can establish it is COMPLETE. Reaching for it here is cheaper than the
    # alternative, which is discovering an outcome with no counter as a run whose totals
    # quietly stop adding up.
    _OUTCOME_COUNTERS,
    OUTCOME_ADMITTED,
    OUTCOME_EMPTY_CLAIM,
    OUTCOME_ERROR,
    OUTCOME_ESCAPING_SPAN,
    OUTCOME_INVALID_SPAN,
    OUTCOME_NO_CITATION,
    OUTCOME_NO_OFFERS,
    OUTCOME_SKIPPED_EXISTING,
    OUTCOME_STALE_EVIDENCE,
    OUTCOME_UNKNOWN_SUBJECT,
    OUTCOME_UNVERIFIABLE,
    PHASE_DONE,
    PHASE_START,
    REFUSAL_OUTCOMES,
    ROLE_CALLEE,
    ROLE_CALLER,
    ROLE_SUBJECT,
    Candidate,
    build_offers,
    candidate_symbols,
    learn,
)
from codelearner.generate.types import Draft, GeneratorUnavailable, Offer
from codelearner.ingest import index_repo
from codelearner.ingest.types import content_hash

# Shaped so that ordering is actually exercised: `acquire` has THREE callees whose
# qualnames do not appear in call order, and TWO callers, one of which is a test. A
# one-callee fixture cannot tell a stable menu from an unstable one.
LEASES = '''\
def acquire(parcel_id):
    """Take a lease."""
    if parcel_id is None:
        return False
    _notify(parcel_id)
    _audit(parcel_id)
    return _record(parcel_id)


def _record(parcel_id):
    """Write the lease down."""
    return True


def _audit(parcel_id):
    """Note that a lease was requested."""
    return None


def _notify(parcel_id):
    """Tell the watchers."""
    return None


def renew(parcel_id):
    """Extend a lease that is already held."""
    if not acquire(parcel_id):
        return False
    return True


def tiny(x):
    return x
'''

TEST_FILE = '''\
from leases import acquire


def test_acquire_takes_a_lease():
    """A test symbol: never a candidate, but a real caller in the graph."""
    assert acquire(1)
'''

# Menu order for `leases.acquire`: the subject, then its callees by qualname, then its
# callers by qualname. Written out rather than derived, because a test that computes
# the expected order the same way the code does cannot catch the code computing it
# differently tomorrow.
ACQUIRE_MENU = [
    (1, "the subject", "leases.acquire"),
    (2, "callee", "leases._audit"),
    (3, "callee", "leases._notify"),
    (4, "callee", "leases._record"),
    (5, "caller", "leases.renew"),
    (6, "caller", "tests.test_leases.test_acquire_takes_a_lease"),
]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach a model. Enforced, not assumed.

    The pipeline's whole point is that the generator is a seam; a test that quietly
    started using a real one would pass on a machine with ollama running and hang
    everywhere else.
    """

    def _refuse(*args, **kwargs):
        raise urllib.error.URLError("tests must not reach a model")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


@pytest.fixture
def repo(tmp_path):
    """A small real repo, indexed. `acquire` calls `_record` and is called by `renew`."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "leases.py").write_text(LEASES)
    (root / "tests").mkdir()
    (root / "tests" / "test_leases.py").write_text(TEST_FILE)
    conn, _ = index_repo(root, index_path=tmp_path / "index.db")
    return root, conn


def _symbol_id(conn, qualname):
    row = conn.execute(
        "SELECT id FROM symbols WHERE qualname = ?", (qualname,)
    ).fetchone()
    assert row is not None, f"{qualname} is not in the index"
    return int(row["id"])


def _count(conn, table):
    return int(conn.execute(f"SELECT count(*) c FROM {table}").fetchone()["c"])  # noqa: S608


# --------------------------------------------------------------------------
# the fakes. Every one of them is deterministic and none of them can produce a span.
# --------------------------------------------------------------------------


class FakeGenerator:
    """Cites a fixed set of reference numbers for every subject."""

    def __init__(self, refs=(1,), claim="the subject takes a lease", *, name="fake/v1",
                 kind="purpose", confidence=0.5):
        self._refs = tuple(refs)
        self._claim = claim
        self._name = name
        self._kind = kind
        self._confidence = confidence
        self.seen: list[tuple[str, tuple[Offer, ...]]] = []

    @property
    def name(self) -> str:
        return self._name

    def draft(self, *, subject: str, offered):
        self.seen.append((subject, tuple(offered)))
        return Draft(
            claim=self._claim,
            cited_refs=self._refs,
            kind=self._kind,
            confidence=self._confidence,
        )


class OutageGenerator:
    """Answers normally until the Nth subject, then the backend goes away."""

    def __init__(self, fail_on: int = 2):
        self._fail_on = fail_on
        self.calls = 0

    @property
    def name(self) -> str:
        return "outage/v1"

    def draft(self, *, subject: str, offered):
        self.calls += 1
        if self.calls >= self._fail_on:
            raise GeneratorUnavailable("ollama is not running")
        return Draft(claim=f"{subject} does something", cited_refs=(1,))


class BrokenGenerator:
    """Raises an ordinary exception on one named subject. Not an outage."""

    def __init__(self, bad_subject: str):
        self._bad = bad_subject

    @property
    def name(self) -> str:
        return "broken/v1"

    def draft(self, *, subject: str, offered):
        if subject == self._bad:
            raise ValueError("the model returned something unparseable")
        return Draft(claim=f"{subject} does something", cited_refs=(1,))


# --------------------------------------------------------------------------
# the happy path: claims that are admitted, and verify
# --------------------------------------------------------------------------


def test_a_good_generator_produces_admitted_claims_whose_spans_verify(repo):
    """End to end. The claims must come back out of `servable_assertions`, which
    re-hashes every cited byte off disk -- an admitted claim that cannot survive that
    was never worth admitting."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(1,)))

    assert report.admitted == 2  # acquire and renew; _record is private, tiny is trivial
    assert report.refused == []
    servable = store.servable_assertions(conn, root)
    assert sorted(a.id for a in servable) == sorted(report.admitted_ids)
    assert {a.subject_qualname for a in servable} == {"leases.acquire", "leases.renew"}
    for assertion in servable:
        assert assertion.spans
        assert assertion.generator == "fake/v1"
        assert assertion.confidence == 0.5
        assert assertion.kind == "purpose"


def test_the_subject_symbol_id_and_claim_text_are_carried_through(repo):
    root, conn = repo
    learn(conn, root, FakeGenerator(refs=(1,), claim="  takes a lease  "))
    row = conn.execute(
        "SELECT subject_symbol_id, claim FROM assertions WHERE subject_qualname = ?",
        ("leases.acquire",),
    ).fetchone()
    assert row["subject_symbol_id"] == _symbol_id(conn, "leases.acquire")
    # Surrounding whitespace is the only normalisation applied, and it is applied to
    # the same string the emptiness check saw.
    assert row["claim"] == "takes a lease"


# --------------------------------------------------------------------------
# refuse, never repair
# --------------------------------------------------------------------------


def test_a_draft_citing_no_references_admits_nothing_and_leaves_no_row(repo):
    """THE rule. The obvious repair -- attach the subject's own span, it is right there
    and known-good -- would put a pipeline-authored citation in the store under the
    generator's name, verifying forever, with nothing to distinguish it from a citation
    the model actually chose."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=()))

    assert report.admitted == 0
    assert report.refused_no_citation == 2
    assert _count(conn, "assertions") == 0
    assert _count(conn, "evidence_spans") == 0


def test_a_draft_citing_only_off_menu_references_is_refused_and_counted(repo):
    """Resolving to nothing is the same as citing nothing, and it must not become a
    row by a different route."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(97, 98)))

    assert report.admitted == 0
    assert report.refused_no_citation == 2
    assert report.invalid_refs == 4
    assert report.drafts_citing_off_menu == 2
    assert _count(conn, "assertions") == 0


def test_an_empty_claim_admits_nothing_even_with_a_perfect_citation(repo):
    """A blank claim with good evidence is still nothing: a judge would have to
    adjudicate it and a reader would have to read it, and both would find no
    statement."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(1,), claim="   \n\t "))

    assert report.admitted == 0
    assert report.refused_empty_claim == 2
    assert report.refused_no_citation == 0
    assert _count(conn, "assertions") == 0
    assert _count(conn, "evidence_spans") == 0


def test_off_menu_refs_are_dropped_and_counted_while_valid_ones_still_admit(repo):
    """Off-menu references are the measurement of whether numbered citation is
    holding, so they are counted even when the draft lands. Dropping them silently
    would make a generator that cites half at random look identical to one that does
    not."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(1, 999)))

    assert report.admitted == 2
    assert report.invalid_refs == 2  # one per draft
    assert report.drafts_citing_off_menu == 2
    assert len(report.results) == 2  # or the per-result checks below are vacuous
    for result in report.results:
        assert result.invalid_refs == (999,)
        assert len(result.citations) == 1
    assert _count(conn, "evidence_spans") == 2


def test_duplicate_references_collapse_to_one_span(repo):
    """Citing the same span three times is one citation. Counting it three times would
    make a thinly-evidenced claim look well supported."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(1, 1, 1)))

    assert report.admitted == 2
    assert _count(conn, "evidence_spans") == 2
    assert all(len(r.citations) == 1 for r in report.results)


# --------------------------------------------------------------------------
# an outage is not a result
# --------------------------------------------------------------------------


def test_generator_unavailable_propagates_and_does_not_become_an_error_count(repo):
    """A dead backend must stop the run. Absorbed into a per-symbol error count it
    would produce a report shaped exactly like a completed run against a bad
    generator, with a coverage hole biased toward whenever the outage started."""
    root, conn = repo
    with pytest.raises(GeneratorUnavailable):
        learn(conn, root, OutageGenerator(fail_on=2))


def test_an_outage_leaves_the_claims_it_reached_and_nothing_half_written(repo):
    """Writes are per-claim, so an interrupted run keeps what it actually admitted --
    and an assertion row without its spans (which would be vacuously servable) must
    never exist."""
    root, conn = repo
    with pytest.raises(GeneratorUnavailable):
        learn(conn, root, OutageGenerator(fail_on=2))

    assert _count(conn, "assertions") == 1
    orphans = conn.execute(
        "SELECT count(*) c FROM assertions a "
        "WHERE NOT EXISTS (SELECT 1 FROM evidence_spans e WHERE e.assertion_id = a.id)"
    ).fetchone()["c"]
    assert orphans == 0
    assert len(store.servable_assertions(conn, root)) == 1


def test_an_ordinary_generator_error_is_counted_and_the_walk_continues(repo):
    """The other half of the rule: one symbol that breaks the model is a fact about
    that symbol, and must not cost the rest of the run."""
    root, conn = repo
    report = learn(conn, root, BrokenGenerator("leases.acquire"))

    assert report.generator_errors == 1
    assert report.admitted == 1
    failed = [r for r in report.results if r.outcome == OUTCOME_ERROR]
    assert [r.qualname for r in failed] == ["leases.acquire"]
    assert "ValueError" in failed[0].error


# --------------------------------------------------------------------------
# re-runs
# --------------------------------------------------------------------------


def test_re_running_skips_symbols_that_already_hold_an_active_claim(repo):
    """The default, and the reason for it: the store never deletes, so a second run
    that drafted again would double the store permanently and silently weight every
    later rate by how often a symbol got re-drafted."""
    root, conn = repo
    first = learn(conn, root, FakeGenerator())
    second = learn(conn, root, FakeGenerator())

    assert first.admitted == 2
    assert second.admitted == 0
    assert second.drafts_requested == 0
    assert second.skipped_existing == 2
    assert [r.outcome for r in second.results] == [OUTCOME_SKIPPED_EXISTING] * 2
    assert _count(conn, "assertions") == 2


def test_skip_existing_false_drafts_again_and_says_so_in_the_counts(repo):
    """The deliberate second opinion. It duplicates, which is why it is not the
    default."""
    root, conn = repo
    learn(conn, root, FakeGenerator())
    again = learn(conn, root, FakeGenerator(), skip_existing=False)

    assert again.admitted == 2
    assert again.skipped_existing == 0
    assert _count(conn, "assertions") == 4


def test_a_claim_that_went_stale_is_re_derived_on_the_next_run(repo):
    """Only ACTIVE claims suppress a re-draft. A stale claim's evidence moved, so its
    symbol is exactly the one worth asking about again."""
    root, conn = repo
    first = learn(conn, root, FakeGenerator())
    store.mark_stale(conn, first.admitted_ids[0], store.REASON_HASH_MISMATCH)

    second = learn(conn, root, FakeGenerator())
    assert second.drafts_requested == 1
    assert second.admitted == 1
    assert second.skipped_existing == 1


def test_another_generators_claims_do_not_suppress_this_one(repo):
    """`assertions.generator` exists so two generators over one repo can be compared.
    Suppressing on subject alone would make the second run measure nothing."""
    root, conn = repo
    learn(conn, root, FakeGenerator(name="model-a/v1"))
    second = learn(conn, root, FakeGenerator(name="model-b/v1"))

    assert second.admitted == 2
    generators = {
        r["generator"] for r in conn.execute("SELECT DISTINCT generator FROM assertions")
    }
    assert generators == {"model-a/v1", "model-b/v1"}


# --------------------------------------------------------------------------
# the menu
# --------------------------------------------------------------------------


def test_offers_are_one_based_and_the_subject_is_always_first(repo):
    root, conn = repo
    offers = build_offers(conn, root, _symbol_id(conn, "leases.acquire"))

    assert [o.ref for o in offers] == list(range(1, len(offers) + 1))
    assert offers[0].label.startswith(ROLE_SUBJECT)
    assert "leases.acquire" in offers[0].label


def test_the_menu_holds_the_subject_its_callees_and_its_callers_in_a_named_order(repo):
    """The graph is what makes the menu worth anything -- `_record` is what `acquire`
    does and `renew` is what it is for, and neither is text-similar to a question about
    acquiring -- and the ORDER is what makes a reference number mean something twice.

    Asserted against a written-out expectation rather than against a re-derivation of
    the same rule, because a stored claim's reference `[3]` has to keep pointing at the
    span it pointed at when the claim was made."""
    root, conn = repo
    offers = build_offers(conn, root, _symbol_id(conn, "leases.acquire"))

    assert [
        (o.ref, o.label.split(": ")[0], o.label.rsplit(" ", 1)[-1]) for o in offers
    ] == ACQUIRE_MENU


def test_reference_numbering_is_stable_across_calls(repo):
    """Numbering that depended on row order, on scoring, or on which rows SQLite
    happened to return first would stop being reproducible without ever looking
    wrong."""
    root, conn = repo
    sid = _symbol_id(conn, "leases.acquire")
    first = build_offers(conn, root, sid)
    second = build_offers(conn, root, sid)

    assert [(o.ref, o.citation, o.label) for o in first] == [
        (o.ref, o.citation, o.label) for o in second
    ]


def test_a_recursive_symbol_is_not_offered_twice_as_its_own_neighbour(repo, tmp_path):
    """A self-call makes the subject its own callee. Two menu entries for one span
    waste a reference number and let a model 'corroborate' a claim by citing the same
    bytes twice."""
    root = tmp_path / "recursive"
    root.mkdir()
    (root / "walk.py").write_text(
        "def descend(node):\n"
        '    """Walk a tree."""\n'
        "    if node is None:\n"
        "        return 0\n"
        "    return 1 + descend(node)\n"
    )
    conn, _ = index_repo(root, index_path=tmp_path / "recursive.db")
    offers = build_offers(conn, root, _symbol_id(conn, "walk.descend"))

    assert len({o.citation for o in offers}) == len(offers)
    assert len(offers) == 1


def test_the_menu_is_bounded_and_the_bound_keeps_the_subject(repo):
    """An unbounded menu blows the context window of a local model and makes reference
    numbers move with a hub's fan-in."""
    root, conn = repo
    offers = build_offers(conn, root, _symbol_id(conn, "leases.acquire"), max_offers=2)

    assert len(offers) == 2
    assert offers[0].label.startswith(ROLE_SUBJECT)
    # The budget goes to callees first: what the subject DOES is what a `purpose`
    # claim is about, so a hub's callers must not crowd out its own body.
    assert offers[1].label == f"{ROLE_CALLEE}: function leases._audit"


def test_a_menu_with_no_room_for_the_subject_is_refused(repo):
    root, conn = repo
    with pytest.raises(ValueError, match="subject"):
        build_offers(conn, root, _symbol_id(conn, "leases.acquire"), max_offers=0)


def test_the_offered_text_is_exactly_the_bytes_the_citation_covers(repo):
    """`Offer` carries `span` and `text` together so that what the model read and what
    a reader will later verify cannot drift apart. A header, a docstring appended, or
    a widened window here would break that quietly.

    Floored, because the whole check is a loop body: a `build_offers` that returned
    nothing -- the exact failure that empties the model's menu -- would make this pass
    without comparing a single byte.
    """
    root, conn = repo
    offers = build_offers(conn, root, _symbol_id(conn, "leases.acquire"))
    assert len(offers) >= 2, "the subject and at least one neighbour, or nothing is compared"
    for offer in offers:
        source = (root / offer.span.path).read_bytes()
        expected = source[offer.span.byte_start : offer.span.byte_end]
        assert offer.text == expected.decode()


def test_a_symbol_the_index_disagrees_with_on_disk_is_not_offered(repo):
    """The index is behind the working tree. Offering the CURRENT text with the STALE
    hash attached would mean the model reads one thing and the citation records
    another -- and it would expire on the first serve, blaming the repo for an
    inconsistency the pipeline introduced."""
    root, conn = repo
    (root / "leases.py").write_text(LEASES.replace("return False", "return None"))

    assert build_offers(conn, root, _symbol_id(conn, "leases.acquire")) == []
    report = learn(conn, root, FakeGenerator())
    assert report.admitted == 0
    assert report.drafts_requested == 0
    assert report.symbols_without_offers == 2
    assert [r.outcome for r in report.results] == [OUTCOME_NO_OFFERS] * 2
    assert _count(conn, "assertions") == 0


# --------------------------------------------------------------------------
# the structural rule: no span can come from a generator
# --------------------------------------------------------------------------


def test_a_draft_has_no_way_to_express_a_span(repo):
    """Structural, and the reason the rest of this file can be short. `Draft` carries
    integers, so 'the model invented a citation' is not a failure mode that has to be
    caught -- it is unrepresentable. If a field is ever added here that can carry a
    path or an offset, this test is the alarm."""
    names = {f.name for f in dataclasses.fields(Draft)}
    assert names == {"claim", "cited_refs", "kind", "confidence"}
    hostile = Draft(claim="see leases.py[0:9999]", cited_refs=(1,))
    assert all(isinstance(ref, int) for ref in hostile.cited_refs)


def test_every_stored_span_is_one_the_index_published(repo):
    """The invariant, asserted against the store rather than against the code that
    writes it: every evidence span in the database matches a symbol row byte for byte
    and hash for hash. A fallback citation, a widened range, or a span built from
    model output would all fail this."""
    root, conn = repo
    hostile = FakeGenerator(
        refs=(1, 2, 4242),
        claim="this claim cites leases.py:1-3 and bytes [0:9999] on its own authority",
    )
    learn(conn, root, hostile)

    indexed = {
        (str(r["path"]), int(r["byte_start"]), int(r["byte_end"]), str(r["content_hash"]))
        for r in conn.execute(
            "SELECT f.path, s.byte_start, s.byte_end, s.content_hash "
            "FROM symbols s JOIN files f ON f.id = s.file_id"
        )
    }
    stored = [
        (str(r["path"]), int(r["byte_start"]), int(r["byte_end"]), str(r["content_hash"]))
        for r in conn.execute(
            "SELECT path, byte_start, byte_end, content_hash FROM evidence_spans"
        )
    ]
    assert stored
    assert all(span in indexed for span in stored)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_selection_skips_tests_private_and_trivial_symbols_by_default(repo):
    """A store full of "this test tests X" crowds out the claims about X, and an
    inferred restatement of a one-line signature is a tier-2 row carrying a tier-0
    fact."""
    _, conn = repo
    chosen = [c.qualname for c in candidate_symbols(conn)]

    assert chosen == ["leases.acquire", "leases.renew"]
    assert not any(q.startswith("tests.") for q in chosen)


def test_selection_is_stable_and_a_limit_takes_the_same_prefix_every_time(repo):
    """A run that picks a different set each time cannot be compared to the previous
    one, and a limit applied to an unordered scan is the quietest way to get one."""
    _, conn = repo
    assert candidate_symbols(conn) == candidate_symbols(conn)
    assert candidate_symbols(conn, limit=1) == candidate_symbols(conn)[:1]


def test_selection_walks_the_repo_in_file_order(tmp_path):
    """The documented order is `(path, line_start)`, and it is asserted rather than
    left to whatever SQLite returns. Any deterministic-but-content-derived order (row
    id, a hash) would look identical here and would quietly re-sample the repo after an
    unrelated edit -- so a limited run could never be compared to the previous one."""
    root = tmp_path / "ordered"
    root.mkdir()
    body = '\n\n\ndef {name}(x):\n    """Do a thing."""\n    return x + 1\n'
    (root / "zeta.py").write_text(
        (body.format(name="first") + body.format(name="second")).lstrip()
    )
    (root / "alpha.py").write_text(
        (body.format(name="third") + body.format(name="fourth")).lstrip()
    )
    conn, _ = index_repo(root, index_path=tmp_path / "ordered.db")

    assert [c.qualname for c in candidate_symbols(conn)] == [
        "alpha.third",
        "alpha.fourth",
        "zeta.first",
        "zeta.second",
    ]


def test_selection_can_be_widened_deliberately(repo):
    """The exclusions are a sampling policy, not a correctness rule, so they are all
    arguments."""
    _, conn = repo
    widened = [c.qualname for c in candidate_symbols(conn, include_private=True)]
    assert "leases._record" in widened

    with_tests = [c.qualname for c in candidate_symbols(conn, include_tests=True)]
    assert any("test_acquire_takes_a_lease" in q for q in with_tests)

    assert [c.qualname for c in candidate_symbols(conn, min_lines=1)] == [
        "leases.acquire",
        "leases.renew",
        "leases.tiny",
    ]


def test_learn_can_be_handed_the_exact_candidate_set_the_eval_will_score(repo):
    """Selection is a named function and an argument so the pipeline and the
    measurement cannot each decide for themselves what the denominator was."""
    root, conn = repo
    only = [c for c in candidate_symbols(conn) if c.qualname == "leases.renew"]
    report = learn(conn, root, FakeGenerator(), candidates=only)

    assert report.considered == 1
    assert [a.subject_qualname for a in store.servable_assertions(conn, root)] == [
        "leases.renew"
    ]


def test_a_candidate_is_carried_with_what_selection_judged_it_on(repo):
    _, conn = repo
    acquire = candidate_symbols(conn)[0]
    assert isinstance(acquire, Candidate)
    assert acquire.qualname == "leases.acquire"
    assert acquire.path == "leases.py"
    assert acquire.lines == acquire.line_end - acquire.line_start + 1


# --------------------------------------------------------------------------
# reporting and progress
# --------------------------------------------------------------------------


def test_the_counters_partition_every_symbol_and_every_draft(repo):
    """A run whose numbers do not add up has lost drafts somewhere, which is the shape
    of the bug a pipeline like this gets: a swallowed exception turning into a symbol
    nobody notices was never asked about."""
    root, conn = repo
    learn(conn, root, FakeGenerator())  # so the second run has something to skip
    report = learn(
        conn, root, BrokenGenerator("leases.acquire"), skip_existing=False
    )

    assert report.considered == (
        report.skipped_existing + report.symbols_without_offers + report.drafts_requested
    )
    assert report.drafts_requested == (
        report.admitted + report.refused_by_the_gate + report.generator_errors
    )
    # Spelled out as well as summed. `refused_by_the_gate` is a property, so a counter
    # dropped out of it would keep the identity above true while losing the draft from
    # every number a reader actually looks at.
    assert report.refused_by_the_gate == (
        report.refused_empty_claim
        + report.refused_no_citation
        + report.refused_invalid_span
        + report.refused_unverifiable
        + report.refused_unknown_subject
        + report.refused_stale_evidence
        + report.refused_escaping_span
    )
    assert len(report.results) == report.considered


def test_the_admission_rate_of_a_run_that_drafted_nothing_is_none_not_one(repo):
    """"Every draft was admitted" is trivially true of no drafts, and this repo has
    already been bitten once by a vacuous truth reading as success."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(), candidates=[])

    assert report.admission_rate is None
    assert "nothing drafted" in report.summary()


def test_the_report_names_the_failure_mode_of_every_refused_draft(repo):
    """A refusal count with no attached detail cannot be acted on: empty claims,
    off-menu references and a raising generator call for different repairs."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(refs=(31, 32)))
    text = report.format_report()

    assert "no_valid_citation=2" in text
    assert "leases.acquire" in text
    assert "31, 32" in text


def test_progress_is_reported_before_and_after_each_symbol(repo):
    """These runs take hours on a local model. The tick that matters is the one BEFORE
    the model call, because that is the silence a caller has to fill -- and none of it
    may be printed from library code."""
    root, conn = repo
    ticks = []
    learn(conn, root, FakeGenerator(), on_progress=ticks.append)

    assert [t.phase for t in ticks] == [PHASE_START, PHASE_DONE] * 2
    assert [t.index for t in ticks] == [1, 1, 2, 2]
    assert all(t.total == 2 for t in ticks)
    assert ticks[0].result is None
    assert ticks[1].result is not None
    assert ticks[1].result.outcome == OUTCOME_ADMITTED
    assert ticks[0].candidate.qualname == "leases.acquire"


def test_a_skipped_symbol_still_ticks_so_a_resumed_run_looks_like_it_is_moving(repo):
    root, conn = repo
    learn(conn, root, FakeGenerator())
    ticks = []
    learn(conn, root, FakeGenerator(), on_progress=ticks.append)

    assert [t.phase for t in ticks] == [PHASE_DONE, PHASE_DONE]
    assert all(t.result.outcome == OUTCOME_SKIPPED_EXISTING for t in ticks)


def test_the_generator_sees_the_subject_and_the_menu_it_will_cite_against(repo):
    """The seam itself: what a generator is handed is a qualname and numbered offers,
    and nothing else. Anything more (a path, a byte range, the whole file) would be
    something it could cite back."""
    root, conn = repo
    generator = FakeGenerator()
    learn(conn, root, generator)

    subjects = [subject for subject, _ in generator.seen]
    assert subjects == ["leases.acquire", "leases.renew"]
    for _, offered in generator.seen:
        assert offered
        assert all(isinstance(o, Offer) for o in offered)
        assert [o.ref for o in offered] == list(range(1, len(offered) + 1))
        assert {o.label.split(":")[0] for o in offered} <= {
            ROLE_SUBJECT, ROLE_CALLEE, ROLE_CALLER
        }


def test_an_unbound_index_refuses_rather_than_guessing_where_the_repo_is(tmp_path):
    """Same refusal as the store's. A pipeline that defaulted to the cwd would offer
    "evidence" read from whatever happened to be at those paths."""
    conn = db.init_db(tmp_path / "unbound.db")
    with pytest.raises(ValueError, match="repo root"):
        learn(conn, None, FakeGenerator())


# --------------------------------------------------------------------------
# the store's other five refusals -- WP4 moved the rules, and they arrived here
# --------------------------------------------------------------------------


def _refusals_write_assertion_can_raise() -> set[type]:
    """Every store exception reachable from `write_assertion`, read out of its source.

    Read rather than listed, because a list is exactly what went stale: `write_assertion`
    grew from one refusal to six and the pipeline's single `except` did not notice. The
    walk follows one level of module-private helper, which is what reaches
    `_verification_root` -- the rule that fires before any span is looked at and the only
    one not raised in the function's own body.

    One level, and the limit is honest rather than incidental: a rule delegated two calls
    deep would escape this and would have to be caught by the run-level tests below
    instead. Bare `ValueError` raises are excluded on purpose -- `_repo_root` raises one,
    it is not a refusal of a claim, and `learn` is meant to propagate it.
    """
    module = ast.parse(inspect.getsource(store))
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}

    def names_raised(func: str, depth: int) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(functions[func]):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
            ):
                found.add(node.exc.func.id)
            if (
                depth
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
                and node.func.id != func
            ):
                found |= names_raised(node.func.id, depth - 1)
        return found

    classes = set()
    for name in names_raised("write_assertion", 1):
        obj = getattr(store, name, None)
        if isinstance(obj, type) and issubclass(obj, ValueError) and obj.__module__ == store.__name__:
            classes.add(obj)
    return classes


def test_every_store_refusal_has_its_own_outcome_and_its_own_counter():
    """The structural guard, and the one that would have caught this whole class of bug
    the day WP4 landed.

    `write_assertion` grew from one refusal to six. This module caught one of them, so
    the other five came out of `_admit` as unhandled exceptions -- three of the four
    reproducible ones ended the run outright. Enumerated from the store's own source
    rather than from a list here, so that the NEXT rule added to the chokepoint fails
    this test until the pipeline decides what to call it. A list would have gone stale in
    exactly the way the code did.

    Exactly, not merely a superset. An entry mapped here that `write_assertion` cannot
    raise is a handler for a condition that never happens, which reads as coverage and is
    the same kind of lie in the other direction -- `NotReinstatable` belongs to
    `reinstate` and has no business in this pipeline's counters."""
    declared = _refusals_write_assertion_can_raise()
    assert len(declared) >= 6, "the enumeration has stopped finding the store's rules"
    assert set(REFUSAL_OUTCOMES) == declared, (
        "a store refusal with no pipeline outcome would leave `learn` and crash a run"
    )
    # Every outcome is distinct -- collapsing two would merge their counters -- and
    # every one of them is counted somewhere.
    assert len(set(REFUSAL_OUTCOMES.values())) == len(declared)
    for outcome in REFUSAL_OUTCOMES.values():
        assert outcome in _OUTCOME_COUNTERS, outcome
    assert len(set(_OUTCOME_COUNTERS.values())) == len(_OUTCOME_COUNTERS)


def test_no_gate_refusal_is_counted_as_a_generator_error():
    """`generator_errors` is the number a reader uses to judge a MODEL. A corrupt index,
    a moving repository and a caller-supplied subject are facts about the environment,
    and the obvious fix for the crash above -- widening the generator's `except
    Exception` -- would have put all three in that column while keeping every total
    correct. That is worse than the crash: the crash is visible."""
    assert "generator_errors" not in _OUTCOME_COUNTERS.values()
    assert OUTCOME_ERROR not in REFUSAL_OUTCOMES.values()


def test_a_repository_that_moves_mid_run_is_counted_rather_than_ending_the_run(repo):
    """The menu is built, the model is asked, and forty seconds later the bytes it was
    shown are not the bytes on disk. `write_assertion` refuses -- correctly, because the
    citation would have failed its first verification -- and before this was caught the
    refusal propagated out of `learn` and the run returned nothing at all.

    Counted under its own name, because the repair is specific: re-index and re-run. A
    generator error would send the reader to the model."""
    root, conn = repo

    class MovingRepo(FakeGenerator):
        def draft(self, *, subject, offered):
            path = root / "leases.py"
            path.write_text(path.read_text().replace("Take a lease", "Take a leash"))
            return super().draft(subject=subject, offered=offered)

    report = learn(conn, root, MovingRepo(), limit=1)

    assert report.drafts_requested == 1
    assert report.refused_stale_evidence == 1
    assert report.generator_errors == 0
    assert [r.outcome for r in report.results] == [OUTCOME_STALE_EVIDENCE]
    assert "EvidenceStale" in report.results[0].error
    assert _count(conn, "assertions") == 0


def test_a_subject_the_index_never_parsed_is_counted_rather_than_ending_the_run(repo):
    """`candidates` overrides selection entirely -- that is how the eval scores the exact
    set a run used -- so a caller can hand `learn` a qualname the index does not hold.
    The menu still builds, because it is keyed on the symbol id, and the refusal arrives
    only at the write. It is a fact about the candidate list, so it gets a counter of its
    own rather than being blamed on the generator."""
    root, conn = repo
    invented = Candidate(
        symbol_id=_symbol_id(conn, "leases.acquire"),
        qualname="leases.acquire_renamed_away",
        kind="function",
        path="leases.py",
        line_start=1,
        line_end=7,
    )
    report = learn(conn, root, FakeGenerator(), candidates=[invented])

    assert report.drafts_requested == 1
    assert report.refused_unknown_subject == 1
    assert report.generator_errors == 0
    assert [r.outcome for r in report.results] == [OUTCOME_UNKNOWN_SUBJECT]
    assert _count(conn, "assertions") == 0


def test_a_degenerate_span_in_the_index_is_counted_apart_from_everything_else(repo):
    """Every `Offer.span` comes from `store.span_for_symbol`, so the pipeline cannot
    build a bad byte range -- but it can be handed one. A symbol row whose byte range is
    empty passes the menu's hash check (sha256 of nothing is a perfectly stable hash) and
    is offered, and the store then refuses the citation as `InvalidSpan`.

    That is the zero-length span the gate controls reproduced as admitted and servable
    before WP4, arriving from the other direction: not a caller inventing offsets, but an
    index publishing them. Its counter is separate because its repair is separate --
    nothing about the model or the repository is wrong, the index needs rebuilding."""
    root, conn = repo
    conn.execute(
        "UPDATE symbols SET byte_start = byte_end, content_hash = ? WHERE qualname = ?",
        (content_hash(b""), "leases.acquire"),
    )
    conn.commit()

    report = learn(conn, root, FakeGenerator(), limit=1)

    assert report.drafts_requested == 1
    assert report.refused_invalid_span == 1
    assert report.generator_errors == 0
    assert [r.outcome for r in report.results] == [OUTCOME_INVALID_SPAN]
    assert "InvalidSpan" in report.results[0].error
    assert _count(conn, "evidence_spans") == 0


def test_an_indexed_path_that_leaves_the_repository_is_counted_rather_than_crashing(
    repo, tmp_path
):
    """The counter that shipped wired to nothing, and how it gets exercised at all.

    Every `Offer.span` comes from `store.span_for_symbol`, so the pipeline cannot invent
    a path -- but it takes the one the index recorded, and `files.path` is data. Point a
    row at `../leases.py`, put a byte-identical file there, and the menu builds happily:
    the file reads, the hash matches, the offer is made, and the store refuses the
    citation at the door because a reader of THIS repository cannot open it.

    `refused_escaping_span` existed as a field, and as a term in the documented
    partition identity, with no entry in `_OUTCOME_COUNTERS` -- so the first draft
    refused this way would have vanished out of `drafts_requested` and the identity
    would have stopped holding on that run and no run before it. A counter nothing
    increments is worse than a missing counter: it reads zero forever."""
    root, conn = repo
    escaped = tmp_path / "leases.py"
    escaped.write_text((root / "leases.py").read_text())
    conn.execute("UPDATE files SET path = ? WHERE path = ?", ("../leases.py", "leases.py"))
    conn.commit()

    report = learn(conn, root, FakeGenerator())

    assert report.drafts_requested == 2
    assert report.refused_escaping_span == 2
    assert report.generator_errors == 0
    assert [r.outcome for r in report.results] == [OUTCOME_ESCAPING_SPAN] * 2
    assert "SpanEscapesRepo" in report.results[0].error
    assert _count(conn, "assertions") == 0
    # The identity has to survive the refusal that was not being counted.
    assert report.drafts_requested == (
        report.admitted + report.refused_by_the_gate + report.generator_errors
    )
    assert "escaping_span=2" in report.summary()


def test_a_run_across_two_trees_is_refused_before_the_first_model_call(repo, tmp_path):
    """The one store refusal that is NOT per-symbol, and is therefore not counted.

    Menus are built from the root this function was handed; every citation is verified
    against the root the index is bound to. If those differ, the model is shown one tree
    and its claims are checked against another -- and when the two happen to hold
    identical bytes the run SUCCEEDS, storing citations nobody read. That is the outcome
    this refuses, and it is worse than the failing one.

    Settled up front by the same argument that leaves `GeneratorUnavailable` uncaught: it
    is a property of the configuration, not of a symbol, so discovering it once per
    symbol spends the whole run to learn one fact -- and reports it as four hundred
    refusals, which reads as a completed measurement of a bad generator."""
    root, conn = repo
    twin = tmp_path / "twin"
    shutil.copytree(root, twin)
    generator = FakeGenerator()

    with pytest.raises(store.EvidenceUnverifiable, match="bound"):
        learn(conn, twin, generator)

    assert generator.seen == [], "a doomed run must not spend a single model call"
    assert _count(conn, "assertions") == 0


def test_an_unbound_index_is_refused_before_the_first_model_call(repo):
    """The other run-wide shape of the same exception, and the reason the pre-flight asks
    the store the question `_admit` will ask rather than the one about its own argument.
    With a root supplied and no binding, menus build perfectly and every write raises --
    so before this check the failure cost one model call per symbol and returned no
    report."""
    root, conn = repo
    conn.execute("DELETE FROM meta WHERE key = 'repo_root'")
    conn.commit()
    generator = FakeGenerator()

    with pytest.raises(store.EvidenceUnverifiable, match="not bound"):
        learn(conn, root, generator)

    assert generator.seen == []


def test_an_empty_claim_is_still_an_empty_claim_when_the_references_also_miss(repo):
    """Why the local empty-claim check stays although `write_assertion` makes the same
    decision. The store's rules run cheapest first, so it checks the spans BEFORE the
    text: a draft with no claim and no resolved reference is `EvidenceRequired` there,
    and would be counted as `refused_no_citation` -- pointing its reader at the
    reference-numbering design when what happened is that the generator emitted nothing.

    Deleting the check in `_admit` does not change which drafts are stored. It changes
    which counter moves, which is the entire content of the report."""
    root, conn = repo
    report = learn(conn, root, FakeGenerator(claim="  \n ", refs=()))

    assert report.refused_empty_claim == 2
    assert report.refused_no_citation == 0
    assert [r.outcome for r in report.results] == [OUTCOME_EMPTY_CLAIM] * 2
    # And the same generator with a resolvable reference is still an empty claim, so the
    # precedence above is not an artefact of there being nothing to cite.
    again = learn(conn, root, FakeGenerator(claim="  \n ", refs=(1,)))
    assert (again.refused_empty_claim, again.refused_no_citation) == (2, 0)


def test_the_report_prints_the_gates_refusals_even_when_they_are_all_zero(repo):
    """A line that appears only when something is wrong is a line nobody learns to read,
    and these four are the counters a reader would not think to ask for. Printed on every
    run, so that the day one of them is non-zero it is a number in a familiar place
    rather than a new sentence."""
    root, conn = repo
    text = learn(conn, root, FakeGenerator()).summary()

    assert "invalid_span=0" in text
    assert "unverifiable=0" in text
    assert "unknown_subject=0" in text
    assert "stale_evidence=0" in text
    assert "escaping_span=0" in text
    assert OUTCOME_UNVERIFIABLE == "refused_unverifiable"


def test_the_repo_root_can_come_from_the_index_binding(repo):
    """`index_repo` binds the root, so the common call does not have to repeat it --
    and must not be able to pass a different one by accident."""
    root, conn = repo
    report = learn(conn, None, FakeGenerator())
    assert report.admitted == 2
    assert len(store.servable_assertions(conn, root)) == 2


def test_an_empty_claim_and_an_off_menu_ref_are_counted_as_different_failures(repo):
    """Collapsing them would make a generator that returns nothing and one that cites
    wildly the same number, with opposite repairs."""
    root, conn = repo
    empty = learn(conn, root, FakeGenerator(claim="", refs=(1,)))
    assert (empty.refused_empty_claim, empty.refused_no_citation) == (2, 0)
    assert [r.outcome for r in empty.results] == [OUTCOME_EMPTY_CLAIM] * 2

    missed = learn(conn, root, FakeGenerator(refs=(77,)))
    assert (missed.refused_empty_claim, missed.refused_no_citation) == (0, 2)
    assert [r.outcome for r in missed.results] == [OUTCOME_NO_CITATION] * 2


# ---------------------------------------------------------------------------
# why an offer never reached the menu
# ---------------------------------------------------------------------------


def test_an_oversize_neighbour_is_counted_apart_from_an_unreadable_one(repo):
    """REGRESSION, from a real run against swarm-sync.

    These were one counter, reported as `offers dropped as unreadable`. The first real
    run printed 96 of them for a repository in which every one of 1,345 symbols hashed
    clean against disk -- all 96 were simply neighbours longer than `max_offer_bytes`.

    The two mean opposite things. Oversize is the menu obeying its context budget and
    happens on any repo with long functions; unreadable means the index and the working
    tree disagree, which invalidates the citations the run is about to write and is a
    reason to stop and re-index. Summed under the alarming name, the number cries wolf
    on every healthy repository -- and would therefore be ignored on the one where it
    was real."""
    root, conn = repo
    subject = _symbol_id(conn, "leases.acquire")

    # A byte cap low enough that every neighbour is oversize, and the subject is exempt.
    offers = build_offers(conn, root, subject, max_offer_bytes=1)
    assert [o.ref for o in offers] == [1]
    assert offers[0].label.startswith(ROLE_SUBJECT)

    generator = FakeGenerator()
    report = learn(
        conn,
        root,
        generator,
        candidates=[c for c in candidate_symbols(conn) if c.symbol_id == subject],
        max_offer_bytes=1,
    )

    assert report.offers_dropped_oversize > 0
    assert report.offers_dropped_unreadable == 0

    rendered = report.summary()
    assert f"oversize={report.offers_dropped_oversize}" in rendered
    assert "unreadable=0" in rendered


def test_a_generous_byte_cap_drops_nothing_at_all(repo):
    """The control for the test above. Without it, a bug that counted every neighbour
    as oversize would satisfy the assertion that oversize is non-zero."""
    root, conn = repo
    subject = _symbol_id(conn, "leases.acquire")

    generator = FakeGenerator()
    report = learn(
        conn,
        root,
        generator,
        candidates=[c for c in candidate_symbols(conn) if c.symbol_id == subject],
        max_offer_bytes=100_000,
    )

    assert report.offers_dropped_oversize == 0
    assert report.offers_dropped_unreadable == 0
