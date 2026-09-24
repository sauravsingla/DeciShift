from __future__ import annotations

from collections.abc import Iterable
from statistics import NormalDist

import numpy as np
import pandas as pd

from decishift.diff.compare import ComparisonResult

_PREDICTION_HINTS = {
    "score", "prediction", "pred", "decision", "threshold", "probability", "prob", "margin", "label"
}
_ID_HINTS = {"id", "uuid", "guid", "key", "identifier"}


def wilson_interval(successes: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def _looks_like_id(column: str, values: pd.Series, n: int) -> bool:
    name = column.lower().replace("-", "_")
    tokens = set(name.split("_"))
    if tokens & _ID_HINTS:
        return True
    unique_ratio = values.nunique(dropna=False) / max(n, 1)
    return n >= 20 and unique_ratio >= 0.98


def _looks_like_prediction(column: str) -> bool:
    name = column.lower().replace("-", "_")
    tokens = set(name.split("_"))
    return bool(tokens & _PREDICTION_HINTS)


def _looks_like_free_text(values: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
        return False
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return False
    lengths = non_null.str.len()
    return float(lengths.median()) > 40 or float(lengths.quantile(0.9)) > 120


def select_cohort_columns(
    source_records: pd.DataFrame,
    *,
    id_column: str | None = None,
    excluded_columns: Iterable[str] | None = None,
    max_categories: int = 30,
    enable_temporal: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Select conservative automatic cohort dimensions and explain exclusions."""
    excluded = set(excluded_columns or [])
    if id_column:
        excluded.add(id_column)
    selected: list[str] = []
    reasons: dict[str, str] = {}
    n = len(source_records)
    for column in source_records.columns:
        values = source_records[column]
        if column in excluded:
            reasons[column] = "explicitly excluded"
            continue
        if _looks_like_prediction(column):
            reasons[column] = "prediction/decision-like column"
            continue
        if pd.api.types.is_datetime64_any_dtype(values):
            if enable_temporal:
                selected.append(column)
            else:
                reasons[column] = "datetime cohorting disabled"
            continue
        if _looks_like_id(column, values, n):
            reasons[column] = "identifier or nearly unique column"
            continue
        if _looks_like_free_text(values):
            reasons[column] = "free-text column"
            continue
        nunique = values.nunique(dropna=False)
        if pd.api.types.is_numeric_dtype(values):
            # Numeric columns with very high uniqueness are still useful if user
            # explicitly asks, but automatic discovery is deliberately conservative.
            if nunique < 2:
                reasons[column] = "constant column"
                continue
            if n >= 100 and nunique / max(n, 1) > 0.95:
                reasons[column] = "near-unique numeric column"
                continue
            selected.append(column)
        else:
            if nunique > max_categories:
                reasons[column] = "high-cardinality column"
                continue
            selected.append(column)
    return selected, reasons


def _summarize(
    group: pd.DataFrame,
    dimension: str,
    cohort: str,
    *,
    total_n: int,
    global_flip_rate: float,
    confidence_level: float,
) -> dict[str, object]:
    n = len(group)
    up = int((group["direction"] == "0->1").sum())
    down = int((group["direction"] == "1->0").sum())
    flips = int(group["changed"].sum())
    direction = "0->1" if up > down else "1->0" if down > up else "mixed/neutral"
    low, high = wilson_interval(flips, n, confidence_level)
    baseline_rate = float(group["baseline_decision"].mean())
    candidate_rate = float(group["candidate_decision"].mean())
    flip_rate = float(flips / n) if n else 0.0
    return {
        "dimension": dimension,
        "cohort": cohort,
        "size": n,
        "coverage": float(n / total_n) if total_n else 0.0,
        "baseline_decision_rate": baseline_rate,
        "candidate_decision_rate": candidate_rate,
        "decision_rate_delta": candidate_rate - baseline_rate,
        "flip_rate": flip_rate,
        "flip_rate_ci_low": low,
        "flip_rate_ci_high": high,
        "dominant_direction": direction,
        "direction": direction,  # v0.1 compatibility
        "mean_score_delta": float(group["score_delta"].mean()),
        "global_flip_rate": global_flip_rate,
        "excess_flip_rate": flip_rate - global_flip_rate,
    }


def analyze_cohorts(
    source_records: pd.DataFrame,
    result: ComparisonResult,
    *,
    columns: Iterable[str] | None = None,
    id_column: str | None = None,
    quantiles: int = 5,
    min_size: int = 30,
    max_categories: int = 30,
    confidence_level: float = 0.95,
    enable_temporal: bool = False,
) -> pd.DataFrame:
    """Descriptive cohort summaries with conservative discovery and Wilson CIs."""
    if len(source_records) != len(result.records):
        raise ValueError("source_records and comparison result must have equal row counts")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")

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
    rows: list[dict[str, object]] = []
    total_n = len(joined)
    global_flip_rate = float(joined["changed"].mean()) if total_n else 0.0

    for column in columns:
        if column not in source_records.columns:
            raise KeyError(column)
        values = source_records[column].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(values):
            if not enable_temporal:
                continue
            labels = values.dt.to_period("M").astype(str)
        elif pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > quantiles:
            # Numeric binning is used for explicitly selected numeric columns and
            # auto-selected columns that survived identifier/cardinality checks.
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
        for label, group in temp.groupby("__cohort__", dropna=False, observed=True, sort=True):
            if len(group) >= min_size:
                rows.append(_summarize(
                    group,
                    column,
                    str(label),
                    total_n=total_n,
                    global_flip_rate=global_flip_rate,
                    confidence_level=confidence_level,
                ))

    columns_out = [
        "dimension", "cohort", "size", "coverage", "baseline_decision_rate",
        "candidate_decision_rate", "decision_rate_delta", "flip_rate",
        "flip_rate_ci_low", "flip_rate_ci_high", "dominant_direction", "direction",
        "mean_score_delta", "global_flip_rate", "excess_flip_rate",
    ]
    if not rows:
        return pd.DataFrame(columns=columns_out)
    return (
        pd.DataFrame(rows)[columns_out]
        .sort_values(["flip_rate", "size", "dimension", "cohort"], ascending=[False, False, True, True], kind="mergesort")
        .reset_index(drop=True)
    )
