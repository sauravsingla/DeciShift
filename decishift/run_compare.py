from __future__ import annotations

from typing import Any

import pandas as pd

from decishift.store import RunStore


def _component_map(attr: pd.DataFrame | None) -> dict[str, float]:
    if attr is None or attr.empty or "component" not in attr.columns:
        return {}
    grouped = attr.groupby("component")["decision_contribution"].apply(lambda s: float(s.abs().mean()))
    return {str(k): float(v) for k, v in grouped.sort_index().items()}


def _top_cohorts(cohorts: pd.DataFrame | None, n: int = 5) -> list[dict[str, Any]]:
    if cohorts is None or cohorts.empty:
        return []
    cols = [c for c in ["dimension", "cohort", "size", "flip_rate", "excess_flip_rate"] if c in cohorts.columns]
    return cohorts.head(n)[cols].to_dict(orient="records")


def compare_saved_runs(run_a: str, run_b: str, *, store: RunStore | None = None) -> dict[str, Any]:
    store = store or RunStore()
    a = store.load(run_a)
    b = store.load(run_b)
    sa, sb = a.summary(), b.summary()
    components_a, components_b = _component_map(a.attribution), _component_map(b.attribution)
    all_components = sorted(set(components_a) | set(components_b))
    component_delta = {
        c: {
            "run_a": components_a.get(c),
            "run_b": components_b.get(c),
            "delta": (components_b.get(c, 0.0) - components_a.get(c, 0.0)),
        }
        for c in all_components
    }
    frag_a = (a.fragility or {}).get("summary", {})
    frag_b = (b.fragility or {}).get("summary", {})
    out_a, out_b = a.outcome_analysis or {}, b.outcome_analysis or {}
    return {
        "run_a": run_a,
        "run_b": run_b,
        "decision_shift_rate": {"run_a": sa.get("decision_shift_rate"), "run_b": sb.get("decision_shift_rate"), "delta": sb.get("decision_shift_rate", 0.0) - sa.get("decision_shift_rate", 0.0)},
        "flips_0_to_1": {"run_a": sa.get("flips_0_to_1"), "run_b": sb.get("flips_0_to_1"), "delta": sb.get("flips_0_to_1", 0) - sa.get("flips_0_to_1", 0)},
        "flips_1_to_0": {"run_a": sa.get("flips_1_to_0"), "run_b": sb.get("flips_1_to_0"), "delta": sb.get("flips_1_to_0", 0) - sa.get("flips_1_to_0", 0)},
        "component_attribution": component_delta,
        "attribution_uncertainty": {
            "run_a_max_decision_ci_width": _max_ci_width(a.attribution),
            "run_b_max_decision_ci_width": _max_ci_width(b.attribution),
        },
        "top_affected_cohorts": {"run_a": _top_cohorts(a.cohorts), "run_b": _top_cohorts(b.cohorts)},
        "fragility": {"run_a": frag_a, "run_b": frag_b},
        "outcome_metrics": {
            "run_a_accuracy": out_a.get("candidate_accuracy"),
            "run_b_accuracy": out_b.get("candidate_accuracy"),
        },
        "reproducibility_status": {
            "run_a": a.metadata.get("reproducibility_status"),
            "run_b": b.metadata.get("reproducibility_status"),
        },
        "note": "This compares saved DeciShift evidence; it does not replay the underlying models.",
    }


def _max_ci_width(attr: pd.DataFrame | None) -> float | None:
    if attr is None or attr.empty or not {"decision_ci_low", "decision_ci_high"}.issubset(attr.columns):
        return None
    width = attr["decision_ci_high"] - attr["decision_ci_low"]
    width = width.dropna()
    return None if width.empty else float(width.max())
