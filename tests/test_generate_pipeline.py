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

import dataclasses
import urllib.error
import urllib.request

import pytest

from codelearner import db
from codelearner.assertions import store
from codelearner.generate.pipeline import (
    OUTCOME_ADMITTED,
    OUTCOME_EMPTY_CLAIM,
    OUTCOME_ERROR,
    OUTCOME_NO_CITATION,
    OUTCOME_NO_OFFERS,
    OUTCOME_SKIPPED_EXISTING,
    PHASE_DONE,
    PHASE_START,
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
    a widened window here would break that quietly."""
    root, conn = repo
    for offer in build_offers(conn, root, _symbol_id(conn, "leases.acquire")):
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
        report.admitted
        + report.refused_empty_claim
        + report.refused_no_citation
        + report.generator_errors
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
