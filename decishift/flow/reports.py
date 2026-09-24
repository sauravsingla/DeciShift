from __future__ import annotations

import html
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from decishift.flow.compare import FlowComparisonResult


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _attribution_summary(attribution: pd.DataFrame | None) -> list[dict[str, Any]]:
    if attribution is None or attribution.empty:
        return []
    grouped = attribution.groupby("player", as_index=False).agg(
        mean_abs_contribution=("target_contribution", lambda s: float(s.abs().mean())),
        mean_contribution=("target_contribution", "mean"),
    )
    if {"ci_low", "ci_high"}.issubset(attribution.columns):
        temp = attribution.assign(__width=attribution["ci_high"] - attribution["ci_low"])
        widths = temp.groupby("player")["__width"].max().rename("max_ci_width")
        grouped = grouped.merge(widths, on="player", how="left")
    grouped = grouped.sort_values(["mean_abs_contribution", "player"], ascending=[False, True])
    total = float(grouped["mean_abs_contribution"].sum())
    grouped["absolute_share"] = grouped["mean_abs_contribution"] / total if total else 0.0
    return [_clean(row) for row in grouped.to_dict(orient="records")]


def _interaction_summary(interactions: pd.DataFrame | None) -> list[dict[str, Any]]:
    if interactions is None or interactions.empty:
        return []
    grouped = interactions.groupby(["player_a", "player_b"], as_index=False).agg(
        mean_abs_target_interaction=("target_interaction", lambda s: float(s.abs().mean())),
        interaction_only_transitions=("interaction_only_transition", "sum"),
    )
    grouped["interaction_only_transitions"] = grouped["interaction_only_transitions"].astype(int)
    return grouped.sort_values(
        ["interaction_only_transitions", "mean_abs_target_interaction", "player_a", "player_b"],
        ascending=[False, False, True, True],
    ).to_dict(orient="records")


def flow_payload(result: FlowComparisonResult) -> dict[str, Any]:
    diagnostics = None
    if result.diagnostics is not None:
        diagnostics = result.diagnostics.as_dict() if hasattr(result.diagnostics, "as_dict") else result.diagnostics
    return _clean({
        "summary": result.summary(),
        "topology_compatible": result.metadata.get("topology_compatible"),
        "topology_diff": result.metadata.get("topology_diff"),
        "structural_impact": result.metadata.get("structural_impact"),
        "execution": result.metadata.get("execution"),
        "node_attribution": _attribution_summary(result.attribution),
        "attribution_diagnostics": diagnostics,
        "pairwise_interactions": _interaction_summary(result.interactions),
        "cohorts": [] if result.cohorts is None else result.cohorts.head(50).to_dict(orient="records"),
        "outcome_analysis": result.outcome_analysis,
        "fragility": None if result.fragility is None else result.fragility.get("summary", result.fragility),
        "reproducibility_status": result.metadata.get("reproducibility_status", "not recorded"),
        "evidence_integrity": "Saved flow evidence is protected by the same SHA-256 manifest verification used by `decishift verify RUN_ID`.",
        "scientific_notes": [
            "Structural reachability is not causal impact and does not imply observed historical behavior changed.",
            "Flow attribution is software counterfactual attribution and does not establish real-world causality.",
            "Categorical actions are never numerically subtracted.",
            "Sampling intervals quantify permutation-sampling uncertainty only.",
            "Results depend on the supplied historical records.",
        ],
    })


def render_flow_json(result: FlowComparisonResult, *, indent: int = 2) -> str:
    return json.dumps(flow_payload(result), indent=indent, ensure_ascii=False, allow_nan=False, default=str)


def render_flow_terminal(result: FlowComparisonResult) -> str:
    summary = result.summary()
    lines = [
        "DeciShift DecisionFlow comparison",
        "=" * 64,
        f"Historical records       {summary['total_records']:>18,}",
        f"Actions changed          {summary['changed_actions']:>18,}",
        f"Actions unchanged        {summary['unchanged_actions']:>18,}",
        f"Action-shift rate        {summary['action_shift_rate']:>18.2%}",
        f"Topology compatible      {str(result.metadata.get('topology_compatible')):>18}",
        f"Reproducibility          {str(result.metadata.get('reproducibility_status')):>18}",
    ]
    transitions = summary["most_frequent_changed_transitions"]
    if transitions:
        lines += ["", "Transitions", "-" * 64]
        for row in transitions[:8]:
            lines.append(f"{row['transition']:<40}{row['count']:>10,}  {row['rate']:>8.2%}")
    lines += ["", "Changed nodes", "-" * 64]
    if result.changed_nodes:
        lines.extend(f"- {name}" for name in result.changed_nodes)
    else:
        lines.append("- none")
    impact = result.metadata.get("structural_impact") or {}
    lines += ["", "Structural impact", "-" * 64]
    descendants = impact.get("structural_descendants") or []
    lines.append("Potential downstream nodes: " + (", ".join(descendants) if descendants else "none"))
    lines.append("Structural reachability is not causal impact.")
    if not result.metadata.get("topology_compatible", False):
        lines += ["", "Attribution", "-" * 64, "Unavailable — topology changed. Observed final-action comparison remains valid."]
    else:
        attribution = _attribution_summary(result.attribution)
        if attribution:
            lines += ["", "Node/group attribution", "-" * 64]
            target = result.metadata.get("attribution_target", "candidate_action_support")
            lines.append(f"target={target}")
            for row in attribution:
                width = row.get("max_ci_width")
                suffix = "" if width is None else f"  max CI width={width:.4f}"
                lines.append(f"{row['player']:<30}{row['absolute_share']:>10.2%}{suffix}")
    if result.diagnostics is not None:
        d = result.diagnostics
        lines += ["", "Attribution diagnostics", "-" * 64]
        lines.append(f"efficiency_valid={d.efficiency_valid}")
        lines.append(f"sampling_precision_sufficient={d.sampling_precision_sufficient}")
        lines.append(f"sampling_converged={d.sampling_converged}")
        if d.permutations_used is not None:
            lines.append(f"permutations_used={d.permutations_used}")
        for warning in d.warnings:
            lines.append(f"WARNING {warning}")
    if result.cohorts is not None and not result.cohorts.empty:
        lines += ["", "Most affected cohorts", "-" * 64]
        for _, row in result.cohorts.head(5).iterrows():
            label = f"{row['dimension']}={row['cohort']}"
            lines.append(f"{label[:38]:<40}{row['action_shift_rate']:>10.2%}  {row['most_common_transition']}")
    if result.outcome_analysis is not None:
        o = result.outcome_analysis
        lines += ["", "Explicit multiclass outcome analysis", "-" * 64]
        lines.append(f"accuracy: {o['baseline_accuracy']:.2%} -> {o['candidate_accuracy']:.2%} ({o['accuracy_delta']:+.2%})")
    lines += ["", "Scientific interpretation", "-" * 64]
    lines.append("Software counterfactual attribution does not establish real-world causality.")
    lines.append("Categorical actions are never numerically subtracted.")
    return "\n".join(lines)


def render_flow_markdown(result: FlowComparisonResult) -> str:
    s = result.summary()
    lines = [
        "# DeciShift DecisionFlow comparison", "",
        "## Action shifts", "",
        f"- Historical records: **{s['total_records']:,}**",
        f"- Actions changed: **{s['changed_actions']:,}** ({s['action_shift_rate']:.2%})",
        f"- Actions unchanged: **{s['unchanged_actions']:,}**", "",
        "## Transitions", "",
        "| Transition | Count | Rate |", "|---|---:|---:|",
    ]
    for row in s["most_frequent_changed_transitions"][:10]:
        lines.append(f"| {row['transition']} | {row['count']:,} | {row['rate']:.2%} |")
    lines += ["", "## Changed nodes", ""]
    lines.extend(f"- `{name}`" for name in result.changed_nodes)
    impact = result.metadata.get("structural_impact") or {}
    lines += ["", "## Structural impact", ""]
    descendants = impact.get("structural_descendants", [])
    lines.append("Potential downstream nodes: " + (", ".join(f"`{x}`" for x in descendants) if descendants else "none"))
    lines.append("")
    lines.append("> Structural reachability is not causal impact and does not imply historical behavior changed.")
    lines += ["", "## Topology", "", f"- Compatible for hybrid attribution: **{result.metadata.get('topology_compatible')}**"]
    if not result.metadata.get("topology_compatible", False):
        lines.append("- Node attribution: **unavailable — topology changed**")
    attr = _attribution_summary(result.attribution)
    if attr:
        lines += ["", "## Node/group attribution", "", "| Player | Absolute share | Mean contribution | Max CI width |", "|---|---:|---:|---:|"]
        for row in attr:
            width = row.get("max_ci_width")
            width_text = "—" if width is None else f"{width:.6f}"
            lines.append(f"| {row['player']} | {row['absolute_share']:.2%} | {row['mean_contribution']:+.6f} | {width_text} |")
    if result.diagnostics is not None:
        d = result.diagnostics
        lines += ["", "## Attribution diagnostics", "",
                  f"- Efficiency valid: **{d.efficiency_valid}**",
                  f"- Sampling precision sufficient: **{d.sampling_precision_sufficient}**",
                  f"- Sampling converged: **{d.sampling_converged}**"]
        if d.permutations_used is not None:
            lines.append(f"- Permutations used: **{d.permutations_used}**")
    lines += ["", "## Evidence and interpretation", "",
              "Saved evidence remains locally verifiable with `decishift verify RUN_ID`.", "",
              "Categorical actions are never numerically subtracted. Candidate-action-support and change-from-baseline are explicitly defined numeric software-counterfactual games. Structural impact and software attribution do not establish real-world causal effects. Results depend on the supplied historical records."]
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_flow_html(result: FlowComparisonResult) -> str:
    p = flow_payload(result)
    s = p["summary"]
    transitions = s["most_frequent_changed_transitions"][:10]
    attr = p["node_attribution"]
    topology_note = "available" if p["topology_compatible"] else "unavailable — topology changed"
    structural = p["structural_impact"] or {}
    descendants = structural.get("structural_descendants", [])
    style = "body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.45;color:#202124}h1,h2{color:#17324d}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#f5f7f9}.note{background:#fff8e1;padding:12px;border-left:4px solid #d9a400}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeciShift DecisionFlow report</title><style>{style}</style></head><body>
<h1>DeciShift DecisionFlow evidence report</h1>
<p><strong>{s['changed_actions']:,}</strong> of <strong>{s['total_records']:,}</strong> actions changed ({s['action_shift_rate']:.2%}).</p>
<h2>1. Action transitions</h2>{_table(transitions, ['transition','count','rate'])}
<h2>2. Changed nodes</h2><p>{html.escape(', '.join(s['changed_nodes']) or 'none')}</p>
<h2>3. Structural impact</h2><p>{html.escape(', '.join(descendants) or 'none')}</p><p class="note">Structural reachability is not causal impact.</p>
<h2>4. Attribution</h2><p>Node/group attribution: <strong>{html.escape(topology_note)}</strong></p>{_table(attr, ['player','absolute_share','mean_contribution','max_ci_width'])}
<h2>5. Reproducibility</h2><p>{html.escape(str(p['reproducibility_status']))}</p>
<h2>6. Evidence integrity</h2><p>{html.escape(p['evidence_integrity'])}</p>
<h2>7. Scientific interpretation</h2><p class="note">Categorical actions are never numerically subtracted. Software counterfactual attribution does not establish real-world causality. Sampling intervals quantify permutation-sampling uncertainty only. Results depend on supplied historical records.</p>
</body></html>"""
