from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from decishift.diff.compare import ComparisonResult


def _summarize(group: pd.DataFrame, dimension: str, cohort: str) -> dict[str, object]:
    n = len(group)
    up = int((group["direction"] == "0->1").sum())
    down = int((group["direction"] == "1->0").sum())
    direction = "0->1" if up > down else "1->0" if down > up else "mixed/neutral"
    return {
        "dimension": dimension,
        "cohort": cohort,
        "size": n,
        "baseline_decision_rate": float(group["baseline_decision"].mean()),
        "candidate_decision_rate": float(group["candidate_decision"].mean()),
        "flip_rate": float(group["changed"].mean()),
        "direction": direction,
        "mean_score_delta": float(group["score_delta"].mean()),
    }


def analyze_cohorts(
    source_records: pd.DataFrame,
    result: ComparisonResult,
    *,
    columns: Iterable[str] | None = None,
    quantiles: int = 5,
    min_size: int = 30,
    max_categories: int = 30,
) -> pd.DataFrame:
    """Summarize decision shifts for categorical values and numeric quantile bins."""
    if len(source_records) != len(result.records):
        raise ValueError("source_records and comparison result must have equal row counts")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    columns = list(columns) if columns is not None else list(source_records.columns)
    joined = result.records.reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []

    for column in columns:
        if column not in source_records.columns:
            raise KeyError(column)
        values = source_records[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > quantiles:
            try:
                buckets = pd.qcut(values, q=quantiles, duplicates="drop")
            except ValueError:
                continue
            labels = buckets.astype(str)
        else:
            if values.nunique(dropna=False) > max_categories:
                continue
            labels = values.fillna("<missing>").astype(str)

        temp = joined.copy()
        temp["__cohort__"] = labels
        for label, group in temp.groupby("__cohort__", dropna=False, observed=True):
            if len(group) >= min_size:
                rows.append(_summarize(group, column, str(label)))

    if not rows:
        return pd.DataFrame(columns=["dimension", "cohort", "size", "baseline_decision_rate", "candidate_decision_rate", "flip_rate", "direction", "mean_score_delta"])
    return pd.DataFrame(rows).sort_values(["flip_rate", "size"], ascending=[False, False]).reset_index(drop=True)
