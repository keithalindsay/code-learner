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
"""

from .ablation import Scorecard, format_table, load_gold, run_ablation
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
    GateReport,
    MutationResult,
    VacuousCorpus,
    build_harness,
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

__all__ = [
    "FAMILIES",
    "LABEL_NOT_SUPPORTED",
    "LABEL_SUPPORTED",
    "LABEL_UNCERTAIN",
    "Adjudication",
    "FaithfulnessReport",
    "GateReport",
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
    "Scorecard",
    "SourceView",
    "VacuousCorpus",
    "adjudicate",
    "assert_no_leak",
    "assert_view_is_source_only",
    "audit_leak_boundary",
    "build_harness",
    "faithfulness",
    "find_leaks",
    "format_report",
    "format_table",
    "label_retrieval_validity",
    "load_gold",
    "mine_labels",
    "run_ablation",
    "run_controls",
    "run_mutation",
    "run_purpose_eval",
    "score_purposes",
    "source_view",
    "token_f1",
]
