"""Retrieval evaluation: per-modality ablation against hand-labelled gold sets."""

from .ablation import Scorecard, format_table, load_gold, run_ablation

__all__ = ["Scorecard", "format_table", "load_gold", "run_ablation"]
