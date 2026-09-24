from __future__ import annotations

import html
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from decishift.diff.compare import ComparisonResult


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    return value


def _component_summary(attribution: pd.DataFrame | None) -> list[dict[str, Any]]:
    if attribution is None or attribution.empty:
        return []
    agg: dict[str, Any] = {
        "mean_abs_score_contribution": ("score_contribution", lambda s: float(s.abs().mean())),
        "mean_abs_decision_contribution": ("decision_contribution", lambda s: float(s.abs().mean())),
    }
    if "decision_standard_error" in attribution.columns:
        agg["mean_decision_standard_error"] = ("decision_standard_error", lambda s: float(s.dropna().mean()) if not s.dropna().empty else float("nan"))
    if {"decision_ci_low", "decision_ci_high"}.issubset(attribution.columns):
        temp = attribution.copy()
        temp["__decision_ci_width"] = temp["decision_ci_high"] - temp["decision_ci_low"]
    else:
        temp = attribution
    grouped = temp.groupby("component", as_index=False).agg(**agg)
    if "__decision_ci_width" in temp.columns:
        widths = temp.groupby("component")["__decision_ci_width"].max().rename("max_decision_ci_width")
        grouped = grouped.merge(widths, on="component", how="left")
    grouped = grouped.sort_values(["mean_abs_decision_contribution", "component"], ascending=[False, True])
    decision_total = float(grouped["mean_abs_decision_contribution"].sum())
    score_total = float(grouped["mean_abs_score_contribution"].sum())
    grouped["decision_attribution_share"] = grouped["mean_abs_decision_contribution"] / decision_total if decision_total else 0.0
    grouped["score_attribution_share"] = grouped["mean_abs_score_contribution"] / score_total if score_total else 0.0
    return [_finite_or_none(row) for row in grouped.to_dict(orient="records")]


def _interaction_summary(interactions: pd.DataFrame | None) -> list[dict[str, Any]]:
    if interactions is None or interactions.empty:
        return []
    grouped = (
        interactions.groupby(["component_a", "component_b"], as_index=False)
        .agg(
            mean_abs_decision_interaction=("decision_interaction", lambda s: float(s.abs().mean())),
            interaction_only_flips=("interaction_only_flip", "sum"),
        )
        .sort_values(["interaction_only_flips", "mean_abs_decision_interaction", "component_a", "component_b"], ascending=[False, False, True, True])
    )
    grouped["interaction_only_flips"] = grouped["interaction_only_flips"].astype(int)
    return grouped.to_dict(orient="records")


def payload(result: ComparisonResult) -> dict[str, Any]:
    diagnostics = None
    if result.diagnostics is not None:
        diagnostics = result.diagnostics.as_dict() if hasattr(result.diagnostics, "as_dict") else result.diagnostics
    data: dict[str, Any] = {
        "summary": result.summary(),
        "component_attribution": _component_summary(result.attribution),
        "attribution_diagnostics": diagnostics,
        "pairwise_interactions": _interaction_summary(result.interactions),
        "outcome_analysis": result.outcome_analysis,
        "fragility": None if result.fragility is None else result.fragility.get("summary", result.fragility),
        "reproducibility_status": result.metadata.get("reproducibility_status", "not recorded"),
        "evidence_integrity": "Tamper-evident artifact hashes are recorded in manifest.json; verify with `decishift verify RUN_ID`.",
        "scientific_note": "Counterfactual component attribution describes this executable decision pipeline; it does not by itself establish real-world causal effects.",
        "cohort_note": "Cohort summaries and confidence intervals are descriptive; they are not tests of statistical significance.",
    }
    if result.cohorts is not None:
        data["cohorts"] = result.cohorts.head(50).to_dict(orient="records")
    return _finite_or_none(data)


def render_json(result: ComparisonResult, *, indent: int = 2) -> str:
    return json.dumps(payload(result), indent=indent, ensure_ascii=False, allow_nan=False, default=str)


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
        "## Reproducibility",
        "",
        f"- Status: **{result.metadata.get('reproducibility_status', 'not recorded')}**",
        "",
    ]
    components = _component_summary(result.attribution)
    if components:
        lines += ["## Component attribution", "", "| Component | Decision share | Mean abs decision contribution | Mean abs score contribution | Max decision CI width |", "|---|---:|---:|---:|---:|"]
        for row in components:
            width = row.get("max_decision_ci_width")
            width_text = "—" if width is None else f"{width:.6f}"
            lines.append(f"| {row['component']} | {row['decision_attribution_share']:.2%} | {row['mean_abs_decision_contribution']:.6f} | {row['mean_abs_score_contribution']:.6f} | {width_text} |")
        lines.append("")
    if result.diagnostics is not None:
        d = result.diagnostics
        lines += [
            "## Attribution diagnostics", "",
            f"- Method: **{d.method}**",
            f"- Score efficiency MAE: **{d.score_efficiency_mae:.6g}**",
            f"- Decision efficiency MAE: **{d.decision_efficiency_mae:.6g}**",
            f"- Converged: **{d.converged}**",
        ]
        for warning in d.warnings:
            lines.append(f"- Warning: {warning}")
        lines.append("")
    if result.cohorts is not None and not result.cohorts.empty:
        lines += ["## Most affected cohorts", "", "| Dimension | Cohort | Size | Coverage | Flip rate | 95% CI | Excess vs global |", "|---|---|---:|---:|---:|---:|---:|"]
        for _, row in result.cohorts.head(10).iterrows():
            lines.append(
                f"| {row['dimension']} | {row['cohort']} | {int(row['size'])} | {row['coverage']:.2%} | {row['flip_rate']:.2%} | "
                f"[{row['flip_rate_ci_low']:.2%}, {row['flip_rate_ci_high']:.2%}] | {row['excess_flip_rate']:+.2%} |"
            )
        lines.append("")
    if result.outcome_analysis is not None:
        o = result.outcome_analysis
        lines += ["## Optional outcome analysis", "", f"- Baseline accuracy: **{o['baseline_accuracy']:.2%}**", f"- Candidate accuracy: **{o['candidate_accuracy']:.2%}**", f"- Accuracy Δ: **{o['accuracy_delta']:+.2%}**", f"- Net corrected decisions: **{o['net_corrected_decisions']}**", ""]
    if result.fragility is not None:
        f = result.fragility["summary"]
        lines += ["## Decision fragility", "", f"- Median minimum absolute margin: **{f['median_absolute_margin']:.6f}**", f"- p10 minimum absolute margin: **{f['p10_absolute_margin']:.6f}**", f"- Boundary-crossing changes: **{f['boundary_crossing_changes']}**", f"- Rule-forced changes: **{f['rule_forced_changes']}**", ""]
    lines += [
        "## Evidence integrity", "",
        "This saved run includes a tamper-evident `manifest.json`. Run `decishift verify RUN_ID` to recompute artifact hashes and the integrity root.", "",
        "## Interpretation note", "",
        "Counterfactual component attribution describes changes inside the executable software pipeline. It does not by itself establish real-world causal effects. A Decision Contract pass is not proof of safety, fairness, compliance, or correctness.",
    ]
    return "\n".join(lines)


def render_terminal(result: ComparisonResult) -> str:
    s = result.summary()
    lines = [
        "DeciShift comparison",
        "=" * 60,
        f"Records evaluated       {s['total_records']:>16,}",
        f"Unchanged               {s['unchanged_decisions']:>16,}",
        f"Changed                 {s['changed_decisions']:>16,}",
        f"Decision shift rate     {s['decision_shift_rate']:>16.2%}",
        f"0 -> 1 flips            {s['flips_0_to_1']:>16,}",
        f"1 -> 0 flips            {s['flips_1_to_0']:>16,}",
        f"Mean score delta        {s['mean_score_delta']:>+16.6f}",
        f"Reproducibility         {str(result.metadata.get('reproducibility_status', 'not recorded')):>16}",
    ]
    components = _component_summary(result.attribution)
    if components:
        lines += ["", "Component attribution", "-" * 60]
        for row in components:
            suffix = ""
            width = row.get("max_decision_ci_width")
            if width is not None:
                suffix = f"  CI width≤{width:.4f}"
            lines.append(f"{row['component']:<24}{row['decision_attribution_share']:>12.2%}{suffix}")
    if result.diagnostics is not None:
        d = result.diagnostics
        lines += ["", "Attribution diagnostics", "-" * 60,
                  f"method={d.method} score_efficiency_mae={d.score_efficiency_mae:.3g} decision_efficiency_mae={d.decision_efficiency_mae:.3g}",
                  f"converged={d.converged}"]
        for warning in d.warnings:
            lines.append(f"WARNING {warning}")
    interactions = _interaction_summary(result.interactions)
    notable = [r for r in interactions if r["interaction_only_flips"]]
    if notable:
        lines += ["", "Interaction-only flips", "-" * 60]
        for row in notable[:8]:
            lines.append(f"{row['component_a']} + {row['component_b']:<18}{row['interaction_only_flips']:>8}")
    if result.cohorts is not None and not result.cohorts.empty:
        lines += ["", "Most affected cohorts", "-" * 60]
        for _, row in result.cohorts.head(5).iterrows():
            label = f"{row['dimension']}={row['cohort']}"
            lines.append(f"{label[:40]:<42}{row['flip_rate']:>9.2%}  [{row['flip_rate_ci_low']:.2%},{row['flip_rate_ci_high']:.2%}]")
    if result.outcome_analysis is not None:
        o = result.outcome_analysis
        lines += ["", "Observed outcome analysis", "-" * 60,
                  f"accuracy: {o['baseline_accuracy']:.2%} -> {o['candidate_accuracy']:.2%} ({o['accuracy_delta']:+.2%})",
                  f"net corrected decisions: {o['net_corrected_decisions']}"]
    if result.fragility is not None:
        f = result.fragility["summary"]
        lines += ["", "Decision fragility", "-" * 60,
                  f"median minimum abs margin: {f['median_absolute_margin']:.6f}",
                  f"boundary crossings: {f['boundary_crossing_changes']} | rule-forced: {f['rule_forced_changes']}"]
    return "\n".join(lines)


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(result: ComparisonResult) -> str:
    p = payload(result)
    s = p["summary"]
    components = p["component_attribution"]
    interactions = p["pairwise_interactions"]
    cohorts = p.get("cohorts", [])[:15]
    diagnostics = p.get("attribution_diagnostics") or {}
    outcome = p.get("outcome_analysis")
    fragility = p.get("fragility")
    style = """
    body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.45;color:#202124}
    h1,h2{color:#17324d} table{border-collapse:collapse;width:100%;margin:12px 0 28px} th,td{border:1px solid #ddd;padding:7px;text-align:left} th{background:#f5f7f9} code{background:#f4f4f4;padding:2px 4px} .note{background:#fff8e1;padding:12px;border-left:4px solid #d9a400}.ok{font-weight:600}
    """
    transition_rows = []
    if outcome:
        transition_rows = [{"transition": k, "count": v} for k, v in outcome.get("transition_table", {}).items()]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeciShift evidence report</title><style>{style}</style></head>
<body><h1>DeciShift evidence report</h1>
<p><strong>{s['changed_decisions']:,}</strong> of <strong>{s['total_records']:,}</strong> decisions changed ({s['decision_shift_rate']:.2%}). Reproducibility: <strong>{html.escape(str(p['reproducibility_status']))}</strong>.</p>
<h2>1. Summary</h2><ul><li>0→1 flips: {s['flips_0_to_1']:,}</li><li>1→0 flips: {s['flips_1_to_0']:,}</li><li>Mean score Δ: {s['mean_score_delta']:+.6f}</li></ul>
<h2>2. Decision-transition table</h2>{_html_table(transition_rows, [('transition','Transition'),('count','Count')])}
<h2>3. Component attribution</h2>{_html_table(components, [('component','Component'),('decision_attribution_share','Decision share'),('mean_abs_decision_contribution','Mean abs decision contribution')])}
<h2>4. Approximation uncertainty</h2><p>{html.escape('No Monte Carlo uncertainty recorded.' if not any('max_decision_ci_width' in r for r in components) else 'Monte Carlo confidence interval widths are shown in component evidence and JSON output.')}</p>
<h2>5. Attribution diagnostics</h2><pre>{html.escape(json.dumps(diagnostics, indent=2, default=str))}</pre>
<h2>6. Interactions</h2>{_html_table(interactions, [('component_a','A'),('component_b','B'),('interaction_only_flips','Interaction-only flips')])}
<h2>7. Most affected cohorts</h2>{_html_table(cohorts, [('dimension','Dimension'),('cohort','Cohort'),('size','Size'),('flip_rate','Flip rate'),('flip_rate_ci_low','CI low'),('flip_rate_ci_high','CI high')])}
<h2>8. Optional outcome analysis</h2><pre>{html.escape(json.dumps(outcome, indent=2, default=str) if outcome else 'Not supplied.')}</pre>
<h2>9. Decision fragility</h2><pre>{html.escape(json.dumps(fragility, indent=2, default=str) if fragility else 'Not recorded.')}</pre>
<h2>10. Reproducibility status</h2><p>{html.escape(str(p['reproducibility_status']))}</p>
<h2>11. Evidence integrity</h2><p>{html.escape(p['evidence_integrity'])}</p>
<h2>12. Scientific interpretation note</h2><div class="note">{html.escape(p['scientific_note'])} {html.escape(p['cohort_note'])} A Decision Contract pass is not proof of safety, fairness, compliance, or correctness.</div>
</body></html>"""
