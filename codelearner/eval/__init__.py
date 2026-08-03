"""Evaluation: retrieval ablation, tier-2 faithfulness, purpose accuracy, gate controls.

Four measurements of four different things, deliberately in one package. The
ablation asks whether the right code was retrieved -- scored against a hand-labelled
gold set, with no model in the loop beyond the embedder. Faithfulness asks whether a
stored claim follows from the spans it cites, which cannot be scored without a judge,
and so is scored by a model from a different family than the one that wrote the claim.
Purpose accuracy asks the third question -- is the claim about what this code is FOR
actually right -- scored against gold labels mined from git history, prose the
generator is structurally prevented from seeing. Hand-labelling purpose costs a
paragraph per symbol; mining it costs nothing, and `gold_from_history` reports exactly
how much of a repo that buys (13% of swarm-sync's symbols) rather than implying all.

`gate_controls` asks the fourth question, and it is the only one whose answer must be
perfect. The other three measure how good the system is; this one measures whether the
gate refuses what the README says it refuses. It generates an adversarial corpus from
what an index actually holds -- nine distinct attacks, seven per symbol and two per file,
from zero citations through a hash that was correct until the file changed under it --
and reports a rejection rate that has
to read 1.0, beside a positive pass rate that also has to read 1.0, because a gate which
refuses everything scores perfectly against attacks alone. Every control is verified by
deleting the rule it targets from a copy of the package and confirming the attack then
succeeds: a control that cannot see its own rule removed is decoration.

`SourceView`, `Generator`, `LeakDetected` and `assert_view_is_source_only` are
re-exported below but no longer defined here. They are the seam the shipped generator
in `generate/` also has to satisfy, and while they lived in this package that package
imported this one -- the direction `generate/llm.py` says in bold must stay empty, and
the edge that closed a four-package cycle. They now live in the leaf
`codelearner.sourceview`. Nothing about what this package measures changed; what
changed is that the thing being measured no longer has to reach into the measurer to
find out what shape it must be.
"""

from .ablation import (
    CALIBRATION_FLOOR,
    POOLED,
    QueryResult,
    Scorecard,
    ci_half_width,
    design_effect,
    format_delta_report,
    format_table,
    load_gold,
    paired_sd,
    power_curve,
    required_n,
    run_ablation,
    run_ablation_multi,
    stratified_cards,
    stratify,
)
from .faithfulness import (
    LABEL_NOT_SUPPORTED,
    LABEL_SUPPORTED,
    LABEL_UNCERTAIN,
    Adjudication,
    FaithfulnessReport,
    Judge,
    Judgement,
    JudgeUnavailable,
    OllamaJudge,
    adjudicate,
    faithfulness,
)
from .gate_controls import (
    FAMILIES,
    SURFACES,
    GateReport,
    MutationResult,
    SurfaceComparison,
    VacuousCorpus,
    build_harness,
    compare_surfaces,
    run_controls,
    run_mutation,
)
from .gold_from_history import (
    LabelValidity,
    LeakDetected,
    MinedLabel,
    MineReport,
    PurposeScorecard,
    SourceView,
    assert_no_leak,
    assert_view_is_source_only,
    audit_leak_boundary,
    find_leaks,
    format_report,
    label_retrieval_validity,
    mine_labels,
    run_purpose_eval,
    score_purposes,
    source_view,
    token_f1,
)
from .goldset import (
    UNSPECIFIED_SOURCE,
    GoldError,
    GoldIndexMismatch,
    GoldQuery,
    GoldSchemaError,
    GoldSet,
    index_qualnames,
    load_gold_set,
    parse_gold,
    validate_against_index,
)

__all__ = [
    "CALIBRATION_FLOOR",
    "FAMILIES",
    "LABEL_NOT_SUPPORTED",
    "LABEL_SUPPORTED",
    "LABEL_UNCERTAIN",
    "POOLED",
    "UNSPECIFIED_SOURCE",
    "Adjudication",
    "FaithfulnessReport",
    "GateReport",
    "GoldError",
    "GoldIndexMismatch",
    "GoldQuery",
    "GoldSchemaError",
    "GoldSet",
    "Judge",
    "JudgeUnavailable",
    "Judgement",
    "LabelValidity",
    "LeakDetected",
    "MineReport",
    "MinedLabel",
    "MutationResult",
    "OllamaJudge",
    "PurposeScorecard",
    "QueryResult",
    "Scorecard",
    "SURFACES",
    "SourceView",
    "SurfaceComparison",
    "VacuousCorpus",
    "adjudicate",
    "assert_no_leak",
    "assert_view_is_source_only",
    "audit_leak_boundary",
    "build_harness",
    "ci_half_width",
    "compare_surfaces",
    "design_effect",
    "faithfulness",
    "find_leaks",
    "format_delta_report",
    "format_report",
    "format_table",
    "index_qualnames",
    "label_retrieval_validity",
    "load_gold",
    "load_gold_set",
    "mine_labels",
    "paired_sd",
    "parse_gold",
    "power_curve",
    "required_n",
    "run_ablation",
    "run_ablation_multi",
    "run_controls",
    "run_mutation",
    "run_purpose_eval",
    "score_purposes",
    "source_view",
    "stratified_cards",
    "stratify",
    "token_f1",
    "validate_against_index",
]
