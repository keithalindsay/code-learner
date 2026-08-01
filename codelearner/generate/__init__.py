"""Tier-2 claim generation: the half of the system that had been specified but not built.

`assertions/` has always said, in its own module docstring, "the pipeline that generates
claims is not here". This is that pipeline, and it arrives last on purpose. Every
guarantee it depends on -- citation-or-nothing at the door, re-hashing at serve time, a
staleness engine that expires a claim when its evidence moves, an adversarial gate with
mutation-verified negative controls, and a cross-family judge to score what comes out --
was built and measured against a store filled by hand. Adding the generator first would
have meant tuning those guarantees against the output of the thing they exist to
restrain.

Four pieces, in dependency order:

- `types` fixes the citation contract. The model is handed a numbered menu of spans the
  *index* built and answers with integers, so a citation it invents is not a bad
  citation, it is not a citation at all. Read that module first; the rest is downstream
  of it.
- `llm` is the ollama-backed backend -- `OllamaClaimGenerator` for cited claims,
  `OllamaPurposeModel` for the purpose eval's plainer seam. Its default model is
  deliberately not a Qwen model, because the judge is, and `collides_with_judge` makes
  that hazard checkable rather than merely documented.
- `pipeline` walks a repo, builds each menu from the call graph, and admits what
  survives through `write_assertion` -- never around it. Its report keeps the refusal
  modes apart, because "the model abstained", "the model cited nothing on the menu" and
  "the model was unreachable" call for three different responses and average into
  nonsense.
- `purpose` adapts a model to the `SourceView -> str` seam that `eval.gold_from_history`
  already scores, so a real generator can be measured against gold labels mined from
  commit prose without the scoring code learning that models exist.

**What this package does not do is decide whether its output is any good.** It writes
claims and counts what happened to them; `eval/` scores them, from a different family of
model, against evidence and against history. Keeping the two apart is what makes the
numbers worth reading -- a generator that scored itself would be reporting its own
opinion of its own work, and a high number would mean nothing at all.
"""

from .llm import (
    DEFAULT_GENERATOR_MODEL,
    JUDGE_FAMILY,
    OllamaClaimGenerator,
    OllamaPurposeModel,
    build_generation_prompt,
    collides_with_judge,
    model_family,
    parse_draft,
    render_menu,
)
from .pipeline import (
    DEFAULT_MAX_OFFER_BYTES,
    DEFAULT_MAX_OFFERS,
    DEFAULT_MIN_LINES,
    ROLE_CALLEE,
    ROLE_CALLER,
    ROLE_SUBJECT,
    Candidate,
    LearnProgress,
    LearnReport,
    LearnResult,
    build_offers,
    candidate_symbols,
    learn,
)
from .purpose import (
    MAX_PURPOSE_WORDS,
    NORMALISATION_RULE,
    LLMPurposeGenerator,
    PurposeModel,
    assert_source_only,
    llm_condition,
    llm_conditions,
    normalise_purpose,
)
from .types import ClaimGenerator, Draft, GeneratorUnavailable, Offer

__all__ = [
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_MAX_OFFERS",
    "DEFAULT_MAX_OFFER_BYTES",
    "DEFAULT_MIN_LINES",
    "JUDGE_FAMILY",
    "MAX_PURPOSE_WORDS",
    "NORMALISATION_RULE",
    "ROLE_CALLEE",
    "ROLE_CALLER",
    "ROLE_SUBJECT",
    "Candidate",
    "ClaimGenerator",
    "Draft",
    "GeneratorUnavailable",
    "LLMPurposeGenerator",
    "LearnProgress",
    "LearnReport",
    "LearnResult",
    "Offer",
    "OllamaClaimGenerator",
    "OllamaPurposeModel",
    "PurposeModel",
    "assert_source_only",
    "build_generation_prompt",
    "build_offers",
    "candidate_symbols",
    "collides_with_judge",
    "learn",
    "llm_condition",
    "llm_conditions",
    "model_family",
    "normalise_purpose",
    "parse_draft",
    "render_menu",
]
