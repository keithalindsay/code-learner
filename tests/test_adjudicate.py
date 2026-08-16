# tests/test_adjudicate.py
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import codelearner.adjudicate as adj


def test_leaf_exposes_the_judging_symbols():
    for name in (
        "Judge", "Judgement", "Adjudication", "OllamaJudge", "JudgeUnavailable",
        "adjudicate_assertion", "render_evidence", "parse_judgement",
        "DEFAULT_OLLAMA_HOST",
    ):
        assert hasattr(adj, name), name


def test_faithfulness_still_reexports_for_compatibility():
    # `import_module` rather than `from codelearner.eval import faithfulness`:
    # `codelearner/eval/__init__.py` re-exports the `faithfulness` FUNCTION under
    # that same name (see its `from .faithfulness import (..., faithfulness, ...)`),
    # which shadows the submodule attribute on the package -- a pre-existing quirk,
    # documented in `tests/test_faithfulness.py`, unrelated to this move.
    faithfulness = importlib.import_module("codelearner.eval.faithfulness")
    assert faithfulness.adjudicate_assertion is adj.adjudicate_assertion
    assert faithfulness.OllamaJudge is adj.OllamaJudge
    # Measurement stays put.
    assert hasattr(faithfulness, "FaithfulnessReport")
    assert hasattr(faithfulness, "JudgeMisbehaving")


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module)
        elif isinstance(node, ast.Import):
            seen.update(alias.name for alias in node.names)
    return seen


def test_adjudicate_leaf_does_not_import_eval_or_cli():
    src = Path(adj.__file__)
    for module in _module_imports(src):
        assert not module.startswith("codelearner.eval"), module
        assert not module.startswith("codelearner.cli"), module
