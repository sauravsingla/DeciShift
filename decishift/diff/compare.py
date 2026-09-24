from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline, PipelineTrace


def _direction(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    values = np.full(len(base), "unchanged", dtype=object)
    values[(base == 0) & (cand == 1)] = "0->1"
    values[(base == 1) & (cand == 0)] = "1->0"
    return values


@dataclass
class ComparisonResult:
    records: pd.DataFrame
    changed_components: list[str] = field(default_factory=list)
    baseline_trace: PipelineTrace | None = None
    candidate_trace: PipelineTrace | None = None
    attribution: pd.DataFrame | None = None
    interactions: pd.DataFrame | None = None
    cohorts: pd.DataFrame | None = None
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
        return {
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
            "mean_score_delta": float(r["score_delta"].mean()) if n else 0.0,
            "mean_abs_score_delta": float(r["score_delta"].abs().mean()) if n else 0.0,
            "mean_margin_delta": float((r["candidate_margin"] - r["baseline_margin"]).mean()) if n else 0.0,
            "changed_components": list(self.changed_components),
            **self.metadata,
        }


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
    bs = np.asarray(baseline_scores, dtype=float)
    cs = np.asarray(candidate_scores, dtype=float)
    bt = np.asarray(baseline_thresholds, dtype=float)
    ct = np.asarray(candidate_thresholds, dtype=float)
    bd = np.asarray(baseline_decisions, dtype=int)
    cd = np.asarray(candidate_decisions, dtype=int)
    for name, arr in {"baseline_scores": bs, "candidate_scores": cs, "baseline_thresholds": bt, "candidate_thresholds": ct, "baseline_decisions": bd, "candidate_decisions": cd}.items():
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"{name} must have one value per input record")
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
) -> ComparisonResult:
    """Replay the same records through baseline and candidate decision pipelines."""
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
    return ComparisonResult(
        records=frame,
        changed_components=baseline.changed_components(candidate),
        baseline_trace=b,
        candidate_trace=c,
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
) -> ComparisonResult:
    """Analyze precomputed predictions when executable components are unavailable."""
    def values(value: str | float) -> np.ndarray:
        if isinstance(value, str):
            return records[value].to_numpy(dtype=float)
        return np.full(len(records), float(value))

    frame = _result_frame(
        records,
        records[baseline_score].to_numpy(dtype=float),
        records[candidate_score].to_numpy(dtype=float),
        values(baseline_threshold),
        values(candidate_threshold),
        records[baseline_decision].to_numpy(dtype=int),
        records[candidate_decision].to_numpy(dtype=int),
        id_column,
    )
    return ComparisonResult(records=frame, metadata={"mode": "predictions-only", "attribution_status": "insufficient evidence"})
