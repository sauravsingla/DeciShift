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
    # v0.2.1+ semantics. These are deliberately separate: efficiency checks
    # additivity, while sampling fields describe Monte Carlo precision/stability.
    efficiency_valid: bool | None = None
    sampling_precision_sufficient: bool | None = None
    sampling_converged: bool | None = None
    permutations_used: int | None = None
    stopped_early: bool | None = None
    max_ci_width: float | None = None
    median_ci_width: float | None = None
    batch_stability_max_change: float | None = None

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


def _attr_sampling_state(attribution: pd.DataFrame, method: str, permutations: int | None) -> dict[str, Any]:
    if method != "approximate":
        return {
            "sampling_precision_sufficient": True,
            "sampling_converged": True,
            "permutations_used": permutations,
            "stopped_early": False,
            "max_ci_width": 0.0,
            "median_ci_width": 0.0,
            "batch_stability_max_change": 0.0,
        }

    attrs = attribution.attrs
    precision = attrs.get("sampling_precision_sufficient")
    if precision is None and "uncertain" in attribution.columns:
        precision = not bool(attribution["uncertain"].fillna(True).any())
    convergence = attrs.get("sampling_converged")
    if convergence is None:
        # Backward-compatible fallback for v0.2 attribution frames: wide-CI
        # status is the only available sampling diagnostic. Do not infer
        # convergence from efficiency residuals.
        convergence = bool(precision) if precision is not None else False

    max_width = attrs.get("max_ci_width")
    median_width = attrs.get("median_ci_width")
    if (max_width is None or median_width is None) and {
        "decision_ci_low", "decision_ci_high"
    }.issubset(attribution.columns):
        widths = (attribution["decision_ci_high"] - attribution["decision_ci_low"]).to_numpy(dtype=float)
        finite = widths[np.isfinite(widths)]
        if finite.size:
            if max_width is None:
                max_width = float(finite.max())
            if median_width is None:
                median_width = float(np.median(finite))

    return {
        "sampling_precision_sufficient": bool(precision) if precision is not None else False,
        "sampling_converged": bool(convergence),
        "permutations_used": attrs.get("permutations_used", permutations),
        "stopped_early": bool(attrs.get("stopped_early", False)),
        "max_ci_width": max_width,
        "median_ci_width": median_width,
        "batch_stability_max_change": attrs.get("batch_stability_max_change"),
    }


def calculate_attribution_diagnostics(
    result,
    attribution: pd.DataFrame | None,
    *,
    method: str,
    hybrid_evaluations: int | None = None,
    permutations: int | None = None,
    tolerance: float = 1e-9,
) -> AttributionDiagnostics:
    """Calculate additive-efficiency and sampling diagnostics independently.

    ``efficiency_valid`` answers whether component contributions add back to the
    observed baseline-to-candidate change. ``sampling_precision_sufficient`` and
    ``sampling_converged`` describe permutation Monte Carlo uncertainty and
    batch stability. The legacy ``converged`` field is retained and now means
    that both applicable checks are satisfied; efficiency alone never establishes
    Monte Carlo convergence.
    """
    changed = list(getattr(result, "changed_components", []))
    warnings: list[str] = []
    if attribution is None or attribution.empty:
        if changed:
            warnings.append("Attribution evidence is unavailable for changed components.")
        exact_sampling = method != "approximate"
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
            efficiency_valid=False,
            sampling_precision_sufficient=exact_sampling,
            sampling_converged=exact_sampling,
            permutations_used=permutations,
            stopped_early=False,
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
    efficiency_valid = bool(sfrac == 0.0 and dfrac == 0.0)
    sampling = _attr_sampling_state(attribution, method, permutations)

    if not efficiency_valid:
        warnings.append("Attribution efficiency residual exceeded configured tolerance.")
    if method == "approximate" and not sampling["sampling_precision_sufficient"]:
        warnings.append("Approximate attribution contains confidence intervals wider than the configured precision target.")
    if method == "approximate" and not sampling["sampling_converged"]:
        warnings.append("Approximate attribution did not satisfy the configured sampling precision and batch-stability criteria.")

    # Backward-compatible field: for exact attribution convergence reduces to
    # valid efficiency; for approximate attribution both efficiency and genuine
    # sampling convergence are required.
    converged = bool(efficiency_valid and sampling["sampling_converged"])

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
        efficiency_valid=efficiency_valid,
        sampling_precision_sufficient=sampling["sampling_precision_sufficient"],
        sampling_converged=sampling["sampling_converged"],
        permutations_used=sampling["permutations_used"],
        stopped_early=sampling["stopped_early"],
        max_ci_width=sampling["max_ci_width"],
        median_ci_width=sampling["median_ci_width"],
        batch_stability_max_change=sampling["batch_stability_max_change"],
    )
