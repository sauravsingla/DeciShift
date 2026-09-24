from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from decishift.diff.compare import ComparisonResult


def _precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return precision, recall


def analyze_outcomes(
    source_records: pd.DataFrame,
    result: ComparisonResult,
    *,
    outcome_column: str,
) -> dict[str, Any]:
    """Describe observed correctness transitions for optional binary outcomes."""
    if outcome_column not in source_records.columns:
        raise KeyError(outcome_column)
    if len(source_records) != len(result.records):
        raise ValueError("source_records and comparison result must have equal row counts")
    y = source_records[outcome_column].to_numpy()
    if pd.isna(y).any() or not np.isin(y, [0, 1, False, True]).all():
        raise ValueError("outcome_column must contain non-null binary 0/1 values")
    y = y.astype(int)
    b = result.records["baseline_decision"].to_numpy(dtype=int)
    c = result.records["candidate_decision"].to_numpy(dtype=int)
    b_correct = b == y
    c_correct = c == y
    bp, br = _precision_recall(y, b)
    cp, cr = _precision_recall(y, c)

    transitions = {
        "baseline_correct_to_candidate_correct": int((b_correct & c_correct).sum()),
        "baseline_correct_to_candidate_wrong": int((b_correct & ~c_correct).sum()),
        "baseline_wrong_to_candidate_correct": int((~b_correct & c_correct).sum()),
        "baseline_wrong_to_candidate_wrong": int((~b_correct & ~c_correct).sum()),
    }
    up = result.records["direction"].to_numpy() == "0->1"
    down = result.records["direction"].to_numpy() == "1->0"

    def _correctness(mask: np.ndarray) -> float | None:
        count = int(mask.sum())
        return None if count == 0 else float(c_correct[mask].mean())

    corrected = transitions["baseline_wrong_to_candidate_correct"]
    newly_wrong = transitions["baseline_correct_to_candidate_wrong"]
    payload: dict[str, Any] = {
        "outcome_column": outcome_column,
        "baseline_accuracy": float(b_correct.mean()) if len(y) else 0.0,
        "candidate_accuracy": float(c_correct.mean()) if len(y) else 0.0,
        "accuracy_delta": float(c_correct.mean() - b_correct.mean()) if len(y) else 0.0,
        "baseline_precision": bp,
        "baseline_recall": br,
        "candidate_precision": cp,
        "candidate_recall": cr,
        "correctness_among_0_to_1_flips": _correctness(up),
        "correctness_among_1_to_0_flips": _correctness(down),
        "corrected_decisions": int(corrected),
        "newly_incorrect_decisions": int(newly_wrong),
        "net_corrected_decisions": int(corrected - newly_wrong),
        "net_newly_incorrect_decisions": int(newly_wrong - corrected),
        "transition_table": transitions,
        "interpretation_note": "Observed historical correctness is descriptive and does not establish real-world causal impact.",
    }
    result.outcome_analysis = payload
    return payload
