from __future__ import annotations

import json
from typing import Any

import pandas as pd

from decishift.diff.compare import ComparisonResult


def _component_summary(attribution: pd.DataFrame | None) -> list[dict[str, Any]]:
    if attribution is None or attribution.empty:
        return []
    grouped = (
        attribution.groupby("component", as_index=False)
        .agg(
            mean_abs_score_contribution=("score_contribution", lambda s: float(s.abs().mean())),
            mean_abs_decision_contribution=("decision_contribution", lambda s: float(s.abs().mean())),
        )
        .sort_values("mean_abs_decision_contribution", ascending=False)
    )
    decision_total = float(grouped["mean_abs_decision_contribution"].sum())
    score_total = float(grouped["mean_abs_score_contribution"].sum())
    grouped["decision_attribution_share"] = (
        grouped["mean_abs_decision_contribution"] / decision_total if decision_total else 0.0
    )
    grouped["score_attribution_share"] = (
        grouped["mean_abs_score_contribution"] / score_total if score_total else 0.0
    )
    return grouped.to_dict(orient="records")


def _interaction_summary(interactions: pd.DataFrame | None) -> list[dict[str, Any]]:
    if interactions is None or interactions.empty:
        return []
    grouped = (
        interactions.groupby(["component_a", "component_b"], as_index=False)
        .agg(
            mean_abs_decision_interaction=("decision_interaction", lambda s: float(s.abs().mean())),
            interaction_only_flips=("interaction_only_flip", "sum"),
        )
        .sort_values(["interaction_only_flips", "mean_abs_decision_interaction"], ascending=False)
    )
    grouped["interaction_only_flips"] = grouped["interaction_only_flips"].astype(int)
    return grouped.to_dict(orient="records")


def payload(result: ComparisonResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "summary": result.summary(),
        "component_attribution": _component_summary(result.attribution),
        "pairwise_interactions": _interaction_summary(result.interactions),
        "scientific_note": "Counterfactual component attribution describes this executable decision pipeline; it does not by itself establish real-world causal effects.",
        "statistical_uncertainty": "Not estimated in v0.1; cohort minimum-size filtering is descriptive, not an uncertainty interval.",
    }
    if result.cohorts is not None:
        data["cohorts"] = result.cohorts.head(50).to_dict(orient="records")
    return data


def render_json(result: ComparisonResult, *, indent: int = 2) -> str:
    return json.dumps(payload(result), indent=indent, default=str)


def render_markdown(result: ComparisonResult) -> str:
    s = result.summary()
    lines = [
        "# DeciShift comparison",
        "",
        "## Decision shifts",
        "",
        f"- Records evaluated: **{s['total_records']:,}**",
        f"- Changed decisions: **{s['changed_decisions']:,}** ({s['decision_shift_rate']:.2%})",
        f"- Unchanged decisions: **{s['unchanged_decisions']:,}**",
        f"- 0→1 flips: **{s['flips_0_to_1']:,}**",
        f"- 1→0 flips: **{s['flips_1_to_0']:,}**",
        f"- Mean score Δ: **{s['mean_score_delta']:+.6f}**",
        "",
    ]
    components = _component_summary(result.attribution)
    if components:
        lines += ["## Component attribution", "", "| Component | Decision attribution share | Mean abs decision contribution | Mean abs score contribution |", "|---|---:|---:|---:|"]
        for row in components:
            lines.append(f"| {row['component']} | {row['decision_attribution_share']:.2%} | {row['mean_abs_decision_contribution']:.6f} | {row['mean_abs_score_contribution']:.6f} |")
        lines.append("")
    if result.cohorts is not None and not result.cohorts.empty:
        lines += ["## Most affected cohorts", "", "| Dimension | Cohort | Size | Flip rate | Mean score Δ |", "|---|---|---:|---:|---:|"]
        for _, row in result.cohorts.head(10).iterrows():
            lines.append(f"| {row['dimension']} | {row['cohort']} | {int(row['size'])} | {row['flip_rate']:.2%} | {row['mean_score_delta']:+.6f} |")
        lines.append("")
    lines += [
        "## Interpretation note",
        "",
        "Counterfactual component attribution describes changes inside the executable software pipeline. It does not by itself establish real-world causal effects.",
    ]
    return "\n".join(lines)


def render_terminal(result: ComparisonResult) -> str:
    s = result.summary()
    lines = [
        "DeciShift comparison",
        "=" * 56,
        f"Records evaluated       {s['total_records']:>12,}",
        f"Unchanged               {s['unchanged_decisions']:>12,}",
        f"Changed                 {s['changed_decisions']:>12,}",
        f"Decision shift rate     {s['decision_shift_rate']:>12.2%}",
        f"0 -> 1 flips            {s['flips_0_to_1']:>12,}",
        f"1 -> 0 flips            {s['flips_1_to_0']:>12,}",
        f"Mean score delta        {s['mean_score_delta']:>+12.6f}",
    ]
    components = _component_summary(result.attribution)
    if components:
        lines += ["", "Component attribution (share of absolute decision attribution)", "-" * 56]
        for row in components:
            lines.append(f"{row['component']:<24}{row['decision_attribution_share']:>12.2%}")
    interactions = _interaction_summary(result.interactions)
    notable = [r for r in interactions if r["interaction_only_flips"]]
    if notable:
        lines += ["", "Interaction-only flips", "-" * 56]
        for row in notable[:8]:
            lines.append(f"{row['component_a']} + {row['component_b']:<18}{row['interaction_only_flips']:>8}")
    if result.cohorts is not None and not result.cohorts.empty:
        lines += ["", "Most affected cohorts", "-" * 56]
        for _, row in result.cohorts.head(5).iterrows():
            label = f"{row['dimension']}={row['cohort']}"
            lines.append(f"{label[:38]:<40}{row['flip_rate']:>10.2%}")
    return "\n".join(lines)
