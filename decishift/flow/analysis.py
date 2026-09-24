from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from decishift.cohorts.analyzer import select_cohort_columns, wilson_interval
from decishift.flow.compare import FlowComparisonResult


def analyze_flow_cohorts(
    source_records: pd.DataFrame,
    result: FlowComparisonResult,
    *,
    columns: Iterable[str] | None = None,
    id_column: str | None = None,
    quantiles: int = 5,
    min_size: int = 30,
    max_categories: int = 30,
    confidence_level: float = 0.95,
    enable_temporal: bool = False,
    requested_transitions: Iterable[str] | None = None,
) -> pd.DataFrame:
    if len(source_records) != len(result.records):
        raise ValueError("source_records and flow comparison result must have equal row counts")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    requested = list(requested_transitions or [])
    if columns is None:
        columns, exclusions = select_cohort_columns(
            source_records,
            id_column=id_column,
            max_categories=max_categories,
            enable_temporal=enable_temporal,
        )
        result.metadata["cohort_auto_exclusions"] = exclusions
    else:
        columns = list(columns)

    joined = result.records.reset_index(drop=True).copy()
    total_n = len(joined)
    global_rate = float(joined["changed"].mean()) if total_n else 0.0
    rows: list[dict] = []

    for column in columns:
        if column not in source_records.columns:
            raise KeyError(column)
        values = source_records[column].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(values):
            if not enable_temporal:
                continue
            labels = values.dt.to_period("M").astype(str)
        elif pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > quantiles:
            try:
                labels = pd.qcut(values, q=quantiles, duplicates="drop").astype(str)
            except ValueError:
                continue
        else:
            if values.nunique(dropna=False) > max_categories:
                continue
            labels = values.fillna("<missing>").astype(str)

        temp = joined.copy()
        temp["__cohort__"] = labels
        for label, group in temp.groupby("__cohort__", dropna=False, observed=True, sort=True):
            n = len(group)
            if n < min_size:
                continue
            changed = int(group["changed"].sum())
            rate = changed / n if n else 0.0
            low, high = wilson_interval(changed, n, confidence_level)
            changed_transitions = group.loc[group["changed"], "transition"]
            if changed_transitions.empty:
                common = "unchanged"
            else:
                counts = changed_transitions.value_counts()
                max_count = int(counts.max())
                common = sorted(str(key) for key, value in counts.items() if int(value) == max_count)[0]
            row = {
                "dimension": column,
                "cohort": str(label),
                "size": n,
                "coverage": n / total_n if total_n else 0.0,
                "action_shift_rate": rate,
                "action_shift_rate_ci_low": low,
                "action_shift_rate_ci_high": high,
                "global_action_shift_rate": global_rate,
                "excess_action_shift_rate": rate - global_rate,
                "most_common_transition": common,
            }
            for transition in requested:
                row[f"transition_rate:{transition}"] = float((group["transition"] == transition).mean())
            rows.append(row)

    base_columns = [
        "dimension", "cohort", "size", "coverage", "action_shift_rate",
        "action_shift_rate_ci_low", "action_shift_rate_ci_high",
        "global_action_shift_rate", "excess_action_shift_rate", "most_common_transition",
    ]
    extra = [f"transition_rate:{transition}" for transition in requested]
    if not rows:
        return pd.DataFrame(columns=base_columns + extra)
    return (
        pd.DataFrame(rows)[base_columns + extra]
        .sort_values(["action_shift_rate", "size", "dimension", "cohort"], ascending=[False, False, True, True], kind="mergesort")
        .reset_index(drop=True)
    )


def analyze_multiclass_outcomes(
    source_records: pd.DataFrame,
    result: FlowComparisonResult,
    *,
    outcome_column: str,
    actions_are_predictions: bool,
) -> dict | None:
    if not actions_are_predictions:
        return None
    if outcome_column not in source_records.columns:
        raise KeyError(outcome_column)
    actual = source_records[outcome_column].reset_index(drop=True)
    if actual.isna().any():
        raise ValueError(f"Outcome column '{outcome_column}' contains null values")
    baseline = result.records["baseline_action"].reset_index(drop=True)
    candidate = result.records["candidate_action"].reset_index(drop=True)
    base_correct = baseline == actual
    cand_correct = candidate == actual
    n = len(actual)

    transitions = {
        "correct->correct": int((base_correct & cand_correct).sum()),
        "correct->incorrect": int((base_correct & ~cand_correct).sum()),
        "incorrect->correct": int((~base_correct & cand_correct).sum()),
        "incorrect->incorrect": int((~base_correct & ~cand_correct).sum()),
    }
    labels = sorted({str(v) for v in actual} | {str(v) for v in baseline} | {str(v) for v in candidate})

    def confusion(predicted: pd.Series) -> dict[str, dict[str, int]]:
        a = actual.map(str)
        p = predicted.map(str)
        table = pd.crosstab(a, p).reindex(index=labels, columns=labels, fill_value=0)
        return {row: {col: int(table.loc[row, col]) for col in labels} for row in labels}

    candidate_str = candidate.map(str)
    actual_str = actual.map(str)
    per_class = {}
    for label in labels:
        support = int((actual_str == label).sum())
        tp = int(((candidate_str == label) & (actual_str == label)).sum())
        fp = int(((candidate_str == label) & (actual_str != label)).sum())
        fn = int(((candidate_str != label) & (actual_str == label)).sum())
        per_class[label] = {
            "support": support,
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "recall": (tp / (tp + fn)) if tp + fn else None,
        }

    baseline_accuracy = float(base_correct.mean()) if n else 0.0
    candidate_accuracy = float(cand_correct.mean()) if n else 0.0
    return {
        "actions_are_predictions": True,
        "outcome_column": outcome_column,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "correctness_transitions": transitions,
        "baseline_confusion_matrix": confusion(baseline),
        "candidate_confusion_matrix": confusion(candidate),
        "per_class_candidate": per_class,
        "note": "Classification metrics are reported only because actions_are_predictions was explicitly configured true.",
    }
