from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline, PipelineTrace
from decishift.evidence import fingerprint_dataframe


def _direction(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    values = np.full(len(base), "unchanged", dtype=object)
    values[(base == 0) & (cand == 1)] = "0->1"
    values[(base == 1) & (cand == 0)] = "1->0"
    return values


def _validate_ids(records: pd.DataFrame, id_column: str | None, *, allow_duplicate_ids: bool, allow_null_ids: bool) -> None:
    if id_column is None:
        return
    if id_column not in records.columns:
        raise KeyError(f"id_column does not exist: {id_column}")
    ids = records[id_column]
    if not allow_null_ids and ids.isna().any():
        raise ValueError(f"id_column '{id_column}' contains null values")
    if not allow_duplicate_ids and ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].head(5).tolist()
        raise ValueError(f"id_column '{id_column}' must be unique; duplicate examples: {duplicates}")


def _validate_numeric(name: str, values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must have one value per input record")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _validate_binary(name: str, values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must have one value per input record")
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError(f"{name} must contain only binary 0/1 decisions")
    return arr.astype(int)


@dataclass
class ComparisonResult:
    records: pd.DataFrame
    changed_components: list[str] = field(default_factory=list)
    baseline_trace: PipelineTrace | None = None
    candidate_trace: PipelineTrace | None = None
    attribution: pd.DataFrame | None = None
    interactions: pd.DataFrame | None = None
    cohorts: pd.DataFrame | None = None
    diagnostics: Any | None = None
    outcome_analysis: dict[str, Any] | None = None
    fragility: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> pd.DataFrame:
        return self.records[self.records["changed"]].copy()

    def summary(self) -> dict[str, Any]:
        r = self.records
        n = len(r)
        changed = int(r["changed"].sum())
        flips_up = int((r["direction"] == "0->1").sum())
        flips_down = int((r["direction"] == "1->0").sum())
        summary: dict[str, Any] = {
            "total_records": n,
            "baseline_decision_distribution": {"0": int((r["baseline_decision"] == 0).sum()), "1": int((r["baseline_decision"] == 1).sum())},
            "candidate_decision_distribution": {"0": int((r["candidate_decision"] == 0).sum()), "1": int((r["candidate_decision"] == 1).sum())},
            "baseline_positive_rate": float(r["baseline_decision"].mean()) if n else 0.0,
            "candidate_positive_rate": float(r["candidate_decision"].mean()) if n else 0.0,
            "agreement_rate": float(1 - changed / n) if n else 1.0,
            "decision_shift_rate": float(changed / n) if n else 0.0,
            "changed_decisions": changed,
            "unchanged_decisions": n - changed,
            "flips_0_to_1": flips_up,
            "flips_1_to_0": flips_down,
            "rate_0_to_1": float(flips_up / n) if n else 0.0,
            "rate_1_to_0": float(flips_down / n) if n else 0.0,
            "mean_score_delta": float(r["score_delta"].mean()) if n else 0.0,
            "mean_abs_score_delta": float(r["score_delta"].abs().mean()) if n else 0.0,
            "mean_margin_delta": float((r["candidate_margin"] - r["baseline_margin"]).mean()) if n else 0.0,
            "changed_components": list(self.changed_components),
            **self.metadata,
        }
        if self.diagnostics is not None:
            summary["attribution_diagnostics"] = self.diagnostics.as_dict() if hasattr(self.diagnostics, "as_dict") else self.diagnostics
        if self.outcome_analysis is not None:
            summary["outcome_analysis"] = self.outcome_analysis
        if self.fragility is not None:
            summary["fragility"] = self.fragility.get("summary", self.fragility)
        return summary


def _result_frame(
    records: pd.DataFrame,
    baseline_scores: Iterable[float],
    candidate_scores: Iterable[float],
    baseline_thresholds: Iterable[float],
    candidate_thresholds: Iterable[float],
    baseline_decisions: Iterable[int],
    candidate_decisions: Iterable[int],
    id_column: str | None = None,
) -> pd.DataFrame:
    n = len(records)
    bs = _validate_numeric("baseline_scores", np.asarray(baseline_scores), n)
    cs = _validate_numeric("candidate_scores", np.asarray(candidate_scores), n)
    bt = _validate_numeric("baseline_thresholds", np.asarray(baseline_thresholds), n)
    ct = _validate_numeric("candidate_thresholds", np.asarray(candidate_thresholds), n)
    bd = _validate_binary("baseline_decisions", np.asarray(baseline_decisions), n)
    cd = _validate_binary("candidate_decisions", np.asarray(candidate_decisions), n)
    ids = records[id_column].to_numpy() if id_column else records.index.to_numpy()
    out = pd.DataFrame({
        "record_id": ids,
        "baseline_score": bs,
        "candidate_score": cs,
        "score_delta": cs - bs,
        "baseline_threshold": bt,
        "candidate_threshold": ct,
        "baseline_margin": bs - bt,
        "candidate_margin": cs - ct,
        "baseline_decision": bd,
        "candidate_decision": cd,
    })
    out["changed"] = out["baseline_decision"] != out["candidate_decision"]
    out["direction"] = _direction(bd, cd)
    return out


def compare_pipelines(
    baseline: DecisionPipeline,
    candidate: DecisionPipeline,
    records: pd.DataFrame,
    *,
    id_column: str | None = None,
    allow_duplicate_ids: bool = False,
    allow_null_ids: bool = False,
    hash_input: bool = True,
) -> ComparisonResult:
    """Replay the same row-aligned records through baseline and candidate pipelines."""
    _validate_ids(records, id_column, allow_duplicate_ids=allow_duplicate_ids, allow_null_ids=allow_null_ids)
    b = baseline.evaluate(records)
    c = candidate.evaluate(records)
    frame = _result_frame(
        records,
        b.calibrated_scores,
        c.calibrated_scores,
        b.thresholds,
        c.thresholds,
        b.decisions,
        c.decisions,
        id_column,
    )
    baseline_identities = {k: v.as_dict() for k, v in baseline.component_identities().items()}
    candidate_identities = {k: v.as_dict() for k, v in candidate.component_identities().items()}
    statuses = {baseline.reproducibility_status, candidate.reproducibility_status}
    if statuses == {"reproducible"}:
        repro = "reproducible"
    elif "unstable" in statuses and len(statuses) == 1:
        repro = "unstable"
    else:
        repro = "partially_reproducible"
    return ComparisonResult(
        records=frame,
        changed_components=baseline.changed_components(candidate),
        baseline_trace=b,
        candidate_trace=c,
        metadata={
            "mode": "executable",
            "baseline_component_identities": baseline_identities,
            "candidate_component_identities": candidate_identities,
            "reproducibility_status": repro,
            "input_data_fingerprint": fingerprint_dataframe(records, include_content=hash_input),
            "input_content_hashing_enabled": bool(hash_input),
            "id_column": id_column,
        },
    )


def compare_predictions(
    records: pd.DataFrame,
    *,
    baseline_score: str,
    candidate_score: str,
    baseline_decision: str,
    candidate_decision: str,
    baseline_threshold: str | float = 0.5,
    candidate_threshold: str | float = 0.5,
    id_column: str | None = None,
    allow_duplicate_ids: bool = False,
    allow_null_ids: bool = False,
    hash_input: bool = True,
) -> ComparisonResult:
    """Analyze precomputed predictions without inventing component attribution."""
    _validate_ids(records, id_column, allow_duplicate_ids=allow_duplicate_ids, allow_null_ids=allow_null_ids)

    def values(value: str | float) -> np.ndarray:
        if isinstance(value, str):
            if value not in records.columns:
                raise KeyError(value)
            return records[value].to_numpy(dtype=float)
        return np.full(len(records), float(value))

    for column in (baseline_score, candidate_score, baseline_decision, candidate_decision):
        if column not in records.columns:
            raise KeyError(column)
    frame = _result_frame(
        records,
        records[baseline_score].to_numpy(dtype=float),
        records[candidate_score].to_numpy(dtype=float),
        values(baseline_threshold),
        values(candidate_threshold),
        records[baseline_decision].to_numpy(),
        records[candidate_decision].to_numpy(),
        id_column,
    )
    return ComparisonResult(records=frame, metadata={
        "mode": "predictions-only",
        "attribution_status": "insufficient evidence",
        "reproducibility_status": "partially_reproducible",
        "input_data_fingerprint": fingerprint_dataframe(records, include_content=hash_input),
        "input_content_hashing_enabled": bool(hash_input),
        "id_column": id_column,
    })
