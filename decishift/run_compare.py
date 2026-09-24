from __future__ import annotations

from typing import Any

import pandas as pd

from decishift.store import RunStore


def _attribution_map(attr: pd.DataFrame | None) -> dict[str, float]:
    if attr is None or attr.empty:
        return {}
    if {"component", "decision_contribution"}.issubset(attr.columns):
        grouped = attr.groupby("component")["decision_contribution"].apply(lambda s: float(s.abs().mean()))
    elif {"player", "target_contribution"}.issubset(attr.columns):
        grouped = attr.groupby("player")["target_contribution"].apply(lambda s: float(s.abs().mean()))
    else:
        return {}
    return {str(k): float(v) for k, v in grouped.sort_index().items()}


def _top_cohorts(cohorts: pd.DataFrame | None, n: int = 5) -> list[dict[str, Any]]:
    if cohorts is None or cohorts.empty:
        return []
    candidates = [
        "dimension", "cohort", "size", "flip_rate", "excess_flip_rate",
        "action_shift_rate", "excess_action_shift_rate", "most_common_transition",
    ]
    cols = [c for c in candidates if c in cohorts.columns]
    return cohorts.head(n)[cols].to_dict(orient="records")


def _numeric_delta(a: Any, b: Any) -> float | int | None:
    if a is None or b is None:
        return None
    return b - a


def compare_saved_runs(run_a: str, run_b: str, *, store: RunStore | None = None) -> dict[str, Any]:
    store = store or RunStore()
    a = store.load(run_a)
    b = store.load(run_b)
    sa, sb = a.summary(), b.summary()
    attr_a, attr_b = _attribution_map(a.attribution), _attribution_map(b.attribution)
    all_players = sorted(set(attr_a) | set(attr_b))
    attribution_delta = {
        player: {
            "run_a": attr_a.get(player),
            "run_b": attr_b.get(player),
            "delta": attr_b.get(player, 0.0) - attr_a.get(player, 0.0),
        }
        for player in all_players
    }
    frag_a = (a.fragility or {}).get("summary", {})
    frag_b = (b.fragility or {}).get("summary", {})
    out_a, out_b = a.outcome_analysis or {}, b.outcome_analysis or {}
    mode_a = a.metadata.get("mode", "pipeline")
    mode_b = b.metadata.get("mode", "pipeline")
    value: dict[str, Any] = {
        "run_a": run_a,
        "run_b": run_b,
        "mode": {"run_a": mode_a, "run_b": mode_b},
        "decision_shift_rate": {
            "run_a": sa.get("decision_shift_rate"),
            "run_b": sb.get("decision_shift_rate"),
            "delta": _numeric_delta(sa.get("decision_shift_rate"), sb.get("decision_shift_rate")),
        },
        "attribution": attribution_delta,
        # Historical key retained for consumers that used v0.2 saved-run output.
        "component_attribution": attribution_delta,
        "attribution_uncertainty": {
            "run_a_max_ci_width": _max_ci_width(a.attribution),
            "run_b_max_ci_width": _max_ci_width(b.attribution),
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
        "note": "This compares saved DeciShift evidence; it does not replay the underlying decision systems.",
    }
    if mode_a != "flow" and mode_b != "flow":
        value["flips_0_to_1"] = {
            "run_a": sa.get("flips_0_to_1"), "run_b": sb.get("flips_0_to_1"),
            "delta": _numeric_delta(sa.get("flips_0_to_1"), sb.get("flips_0_to_1")),
        }
        value["flips_1_to_0"] = {
            "run_a": sa.get("flips_1_to_0"), "run_b": sb.get("flips_1_to_0"),
            "delta": _numeric_delta(sa.get("flips_1_to_0"), sb.get("flips_1_to_0")),
        }
    if mode_a == "flow" or mode_b == "flow":
        value["action_transitions"] = {
            "run_a": sa.get("transition_counts"),
            "run_b": sb.get("transition_counts"),
        }
        value["topology_compatible"] = {
            "run_a": a.metadata.get("topology_compatible"),
            "run_b": b.metadata.get("topology_compatible"),
        }
    return value


def _max_ci_width(attr: pd.DataFrame | None) -> float | None:
    if attr is None or attr.empty:
        return None
    if {"decision_ci_low", "decision_ci_high"}.issubset(attr.columns):
        width = attr["decision_ci_high"] - attr["decision_ci_low"]
    elif {"ci_low", "ci_high"}.issubset(attr.columns):
        width = attr["ci_high"] - attr["ci_low"]
    else:
        return None
    width = width.dropna()
    return None if width.empty else float(width.max())
