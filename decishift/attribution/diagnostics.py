from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AttributionDiagnostics:
    method: str
    changed_components: list[str]
    hybrid_evaluations: int | None
    permutations: int | None
    score_efficiency_mae: float
    score_efficiency_max: float
    score_efficiency_p95: float
    score_efficiency_fraction_exceeding_tolerance: float
    decision_efficiency_mae: float
    decision_efficiency_max: float
    decision_efficiency_p95: float
    decision_efficiency_fraction_exceeding_tolerance: float
    converged: bool
    tolerance: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _residual_stats(values: np.ndarray, tolerance: float) -> tuple[float, float, float, float]:
    abs_values = np.abs(values.astype(float))
    if len(abs_values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(abs_values.mean()),
        float(abs_values.max()),
        float(np.quantile(abs_values, 0.95)),
        float((abs_values > tolerance).mean()),
    )


def calculate_attribution_diagnostics(
    result,
    attribution: pd.DataFrame | None,
    *,
    method: str,
    hybrid_evaluations: int | None = None,
    permutations: int | None = None,
    tolerance: float = 1e-9,
) -> AttributionDiagnostics:
    changed = list(getattr(result, "changed_components", []))
    warnings: list[str] = []
    if attribution is None or attribution.empty:
        if changed:
            warnings.append("Attribution evidence is unavailable for changed components.")
        return AttributionDiagnostics(
            method=method,
            changed_components=changed,
            hybrid_evaluations=hybrid_evaluations,
            permutations=permutations,
            score_efficiency_mae=float("nan"),
            score_efficiency_max=float("nan"),
            score_efficiency_p95=float("nan"),
            score_efficiency_fraction_exceeding_tolerance=float("nan"),
            decision_efficiency_mae=float("nan"),
            decision_efficiency_max=float("nan"),
            decision_efficiency_p95=float("nan"),
            decision_efficiency_fraction_exceeding_tolerance=float("nan"),
            converged=False,
            tolerance=tolerance,
            warnings=warnings,
        )

    grouped = attribution.groupby("record_id", sort=False)[["score_contribution", "decision_contribution"]].sum()
    records = result.records.set_index("record_id")
    aligned = records.loc[grouped.index]
    score_residual = grouped["score_contribution"].to_numpy(dtype=float) - aligned["score_delta"].to_numpy(dtype=float)
    decision_delta = (
        aligned["candidate_decision"].to_numpy(dtype=float)
        - aligned["baseline_decision"].to_numpy(dtype=float)
    )
    decision_residual = grouped["decision_contribution"].to_numpy(dtype=float) - decision_delta

    smae, smax, sp95, sfrac = _residual_stats(score_residual, tolerance)
    dmae, dmax, dp95, dfrac = _residual_stats(decision_residual, tolerance)
    converged = bool(sfrac == 0.0 and dfrac == 0.0)

    if method == "approximate" and "uncertain" in attribution.columns and bool(attribution["uncertain"].any()):
        converged = False
        warnings.append("Approximate attribution contains wide confidence intervals.")
    if not (sfrac == 0.0 and dfrac == 0.0):
        warnings.append("Attribution efficiency residual exceeded configured tolerance.")

    return AttributionDiagnostics(
        method=method,
        changed_components=changed,
        hybrid_evaluations=hybrid_evaluations,
        permutations=permutations,
        score_efficiency_mae=smae,
        score_efficiency_max=smax,
        score_efficiency_p95=sp95,
        score_efficiency_fraction_exceeding_tolerance=sfrac,
        decision_efficiency_mae=dmae,
        decision_efficiency_max=dmax,
        decision_efficiency_p95=dp95,
        decision_efficiency_fraction_exceeding_tolerance=dfrac,
        converged=converged,
        tolerance=tolerance,
        warnings=warnings,
    )
