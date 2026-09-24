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


def _matrix_rows(matrix: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    actions = sorted(set(matrix) | {column for row in matrix.values() for column in row}, key=str)
    rows = []
    for baseline in actions:
        row: dict[str, Any] = {"baseline_action": baseline}
        for candidate in actions:
            row[candidate] = int((matrix.get(baseline) or {}).get(candidate, 0))
        rows.append(row)
    return rows


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
        "hybrid_execution": result.metadata.get("hybrid_execution"),
        "node_attribution": _attribution_summary(result.attribution),
        "attribution_diagnostics": diagnostics,
        "pairwise_interactions": _interaction_summary(result.interactions),
        "cohorts": [] if result.cohorts is None else result.cohorts.head(50).to_dict(orient="records"),
        "decision_contract": result.metadata.get("contract_evaluation"),
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
    payload = flow_payload(result)
    summary = payload["summary"]
    lines = [
        "DeciShift DecisionFlow comparison",
        "=" * 72,
        f"Historical records       {summary['total_records']:>18,}",
        f"Actions changed          {summary['changed_actions']:>18,}",
        f"Actions unchanged        {summary['unchanged_actions']:>18,}",
        f"Action-shift rate        {summary['action_shift_rate']:>18.2%}",
        f"Topology compatible      {str(payload['topology_compatible']):>18}",
        f"Reproducibility          {str(payload['reproducibility_status']):>18}",
    ]
    transitions = summary["most_frequent_changed_transitions"]
    lines += ["", "Transitions", "-" * 72]
    if transitions:
        for row in transitions[:8]:
            lines.append(f"{row['transition']:<44}{row['count']:>10,}  {row['rate']:>8.2%}")
    else:
        lines.append("No changed transitions.")

    matrix = summary["transition_matrix"]
    actions = sorted(set(matrix) | {c for row in matrix.values() for c in row}, key=str)
    lines += ["", "Transition matrix (baseline rows -> candidate columns)", "-" * 72]
    if actions:
        lines.append("baseline\\candidate  " + "  ".join(f"{a:>12}" for a in actions))
        for baseline in actions:
            values = "  ".join(f"{int((matrix.get(baseline) or {}).get(a, 0)):>12,}" for a in actions)
            lines.append(f"{baseline:<18}{values}")

    lines += ["", "Changed nodes", "-" * 72]
    lines.extend(f"- {name}" for name in result.changed_nodes) if result.changed_nodes else lines.append("- none")

    impact = payload["structural_impact"] or {}
    descendants = impact.get("structural_descendants") or []
    lines += ["", "Structural impact", "-" * 72]
    lines.append("Potential downstream nodes: " + (", ".join(descendants) if descendants else "none"))
    lines.append("Structural reachability is not causal impact.")

    lines += ["", "Node/group attribution", "-" * 72]
    if not payload["topology_compatible"]:
        lines.append("Unavailable — topology changed. Observed final-action comparison remains valid.")
    elif payload["node_attribution"]:
        lines.append(f"target={result.metadata.get('attribution_target', 'candidate_action_support')}")
        for row in payload["node_attribution"]:
            width = row.get("max_ci_width")
            suffix = "" if width is None else f"  max CI width={width:.4f}"
            lines.append(f"{row['player']:<34}{row['absolute_share']:>10.2%}{suffix}")
    else:
        lines.append("Not requested or no changed nodes.")

    lines += ["", "Attribution uncertainty / efficiency", "-" * 72]
    if result.diagnostics is None:
        lines.append("Not available.")
    else:
        d = result.diagnostics
        lines.append(f"efficiency_valid={d.efficiency_valid}")
        lines.append(f"sampling_precision_sufficient={d.sampling_precision_sufficient}")
        lines.append(f"sampling_converged={d.sampling_converged}")
        if d.permutations_used is not None:
            lines.append(f"permutations_used={d.permutations_used}")
        for warning in d.warnings:
            lines.append(f"WARNING {warning}")

    lines += ["", "Pairwise interactions", "-" * 72]
    if payload["pairwise_interactions"]:
        for row in payload["pairwise_interactions"][:6]:
            lines.append(
                f"{row['player_a']} x {row['player_b']}: mean|interaction|={row['mean_abs_target_interaction']:.4f}; "
                f"interaction-only transitions={row['interaction_only_transitions']}"
            )
    else:
        lines.append("Not available.")

    lines += ["", "Cohorts", "-" * 72]
    if result.cohorts is not None and not result.cohorts.empty:
        for _, row in result.cohorts.head(5).iterrows():
            label = f"{row['dimension']}={row['cohort']}"
            lines.append(f"{label[:38]:<40}{row['action_shift_rate']:>10.2%}  {row['most_common_transition']}")
    else:
        lines.append("Not available.")

    lines += ["", "Decision Contract", "-" * 72]
    contract = payload["decision_contract"]
    if contract:
        lines.append(f"RESULT: {contract.get('result')} (integrity is verified after the bundle is saved)")
        for check in (contract.get("checks") or [])[:12]:
            lines.append(f"{check.get('status', ''):<12}{check.get('name', '')}")
    else:
        lines.append("Not evaluated in this comparison.")

    if result.outcome_analysis is not None:
        o = result.outcome_analysis
        lines += ["", "Explicit multiclass outcome analysis", "-" * 72]
        lines.append(f"accuracy: {o['baseline_accuracy']:.2%} -> {o['candidate_accuracy']:.2%} ({o['accuracy_delta']:+.2%})")

    lines += ["", "Reproducibility", "-" * 72]
    lines.append(f"reproducibility_status={payload['reproducibility_status']}")
    lines += ["", "Topology compatibility", "-" * 72]
    lines.append(f"topology_compatible={payload['topology_compatible']}")
    lines += ["", "Evidence integrity", "-" * 72]
    lines.append(payload["evidence_integrity"])

    lines += ["", "Scientific interpretation", "-" * 72]
    lines.extend(payload["scientific_notes"])
    return "\n".join(lines)


def render_flow_markdown(result: FlowComparisonResult) -> str:
    p = flow_payload(result)
    s = p["summary"]
    lines = [
        "# DeciShift DecisionFlow comparison", "",
        "## 1. Action-shift summary", "",
        f"- Historical records: **{s['total_records']:,}**",
        f"- Actions changed: **{s['changed_actions']:,}** ({s['action_shift_rate']:.2%})",
        f"- Actions unchanged: **{s['unchanged_actions']:,}**", "",
        "## 2. Transitions", "",
        "| Transition | Count | Rate |", "|---|---:|---:|",
    ]
    for row in s["most_frequent_changed_transitions"][:10]:
        lines.append(f"| {row['transition']} | {row['count']:,} | {row['rate']:.2%} |")
    matrix = s["transition_matrix"]
    actions = sorted(set(matrix) | {c for row in matrix.values() for c in row}, key=str)
    if actions:
        lines += ["", "### Transition matrix", "", "| Baseline \\ Candidate | " + " | ".join(actions) + " |", "|---|" + "---:|" * len(actions)]
        for baseline in actions:
            values = [str(int((matrix.get(baseline) or {}).get(action, 0))) for action in actions]
            lines.append(f"| {baseline} | " + " | ".join(values) + " |")

    lines += ["", "## 3. Changed nodes", ""]
    lines.extend(f"- `{name}`" for name in result.changed_nodes) if result.changed_nodes else lines.append("- none")
    impact = p["structural_impact"] or {}
    descendants = impact.get("structural_descendants", [])
    lines += ["", "## 4. Structural impact", "",
              "Potential downstream nodes: " + (", ".join(f"`{x}`" for x in descendants) if descendants else "none"), "",
              "> Structural reachability is not causal impact and does not imply historical behavior changed."]

    lines += ["", "## 5. Node/group attribution", ""]
    if not p["topology_compatible"]:
        lines.append("**Unavailable — topology changed.** Observed final-action comparison remains valid.")
    elif p["node_attribution"]:
        lines += ["| Player | Absolute share | Mean contribution | Max CI width |", "|---|---:|---:|---:|"]
        for row in p["node_attribution"]:
            width = row.get("max_ci_width")
            width_text = "—" if width is None else f"{width:.6f}"
            lines.append(f"| {row['player']} | {row['absolute_share']:.2%} | {row['mean_contribution']:+.6f} | {width_text} |")
    else:
        lines.append("Not requested or no changed nodes.")

    lines += ["", "## 6. Attribution uncertainty", ""]
    if result.diagnostics is not None:
        d = result.diagnostics
        lines.extend([
            f"- Efficiency valid: **{d.efficiency_valid}**",
            f"- Sampling precision sufficient: **{d.sampling_precision_sufficient}**",
            f"- Sampling converged: **{d.sampling_converged}**",
        ])
        if d.permutations_used is not None:
            lines.append(f"- Permutations used: **{d.permutations_used}**")
    else:
        lines.append("Not available.")

    lines += ["", "## 7. Pairwise interactions", ""]
    if p["pairwise_interactions"]:
        lines += ["| Pair | Mean absolute interaction | Interaction-only transitions |", "|---|---:|---:|"]
        for row in p["pairwise_interactions"][:10]:
            lines.append(f"| {row['player_a']} × {row['player_b']} | {row['mean_abs_target_interaction']:.6f} | {row['interaction_only_transitions']} |")
    else:
        lines.append("Not available.")

    lines += ["", "## 8. Cohorts", ""]
    if p["cohorts"]:
        lines += ["| Cohort | Size | Action-shift rate | Most common transition |", "|---|---:|---:|---|"]
        for row in p["cohorts"][:10]:
            lines.append(f"| {row['dimension']}={row['cohort']} | {row['size']} | {row['action_shift_rate']:.2%} | {row['most_common_transition']} |")
    else:
        lines.append("Not available.")

    lines += ["", "## 9. Decision Contract", ""]
    if p["decision_contract"]:
        lines.append(f"Result: **{p['decision_contract'].get('result')}** (saved-bundle integrity is verified after serialization).")
    else:
        lines.append("Not evaluated in this comparison.")

    lines += ["", "## 10. Reproducibility", "", f"`{p['reproducibility_status']}`",
              "", "## 11. Topology compatibility", "", f"`{p['topology_compatible']}`",
              "", "## 12. Evidence integrity", "", p["evidence_integrity"],
              "", "## 13. Scientific interpretation", ""]
    lines.extend(f"- {note}" for note in p["scientific_notes"])
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
    matrix = _matrix_rows(s["transition_matrix"])
    matrix_columns = ["baseline_action"] + sorted(s["transition_matrix"], key=str)
    structural = p["structural_impact"] or {}
    descendants = structural.get("structural_descendants", [])
    contract_text = "Not evaluated in this comparison."
    if p["decision_contract"]:
        contract_text = f"Result: {p['decision_contract'].get('result')} (saved-bundle integrity is verified after serialization)."
    diagnostics = []
    if p["attribution_diagnostics"]:
        d = p["attribution_diagnostics"]
        diagnostics = [{
            "efficiency_valid": d.get("efficiency_valid"),
            "sampling_precision_sufficient": d.get("sampling_precision_sufficient"),
            "sampling_converged": d.get("sampling_converged"),
            "permutations_used": d.get("permutations_used"),
        }]
    style = "body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.45;color:#202124}h1,h2{color:#17324d}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#f5f7f9}.note{background:#fff8e1;padding:12px;border-left:4px solid #d9a400}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeciShift DecisionFlow report</title><style>{style}</style></head><body>
<h1>DeciShift DecisionFlow evidence report</h1>
<h2>1. Action-shift summary</h2><p><strong>{s['changed_actions']:,}</strong> of <strong>{s['total_records']:,}</strong> actions changed ({s['action_shift_rate']:.2%}).</p>
<h2>2. Transitions</h2>{_table(transitions, ['transition','count','rate'])}<h3>Transition matrix</h3>{_table(matrix, matrix_columns)}
<h2>3. Changed nodes</h2><p>{html.escape(', '.join(s['changed_nodes']) or 'none')}</p>
<h2>4. Structural impact</h2><p>{html.escape(', '.join(descendants) or 'none')}</p><p class="note">Structural reachability is not causal impact.</p>
<h2>5. Node/group attribution</h2>{_table(p['node_attribution'], ['player','absolute_share','mean_contribution','max_ci_width'])}
<h2>6. Attribution uncertainty</h2>{_table(diagnostics, ['efficiency_valid','sampling_precision_sufficient','sampling_converged','permutations_used'])}
<h2>7. Pairwise interactions</h2>{_table(p['pairwise_interactions'], ['player_a','player_b','mean_abs_target_interaction','interaction_only_transitions'])}
<h2>8. Cohorts</h2>{_table(p['cohorts'][:10], ['dimension','cohort','size','action_shift_rate','most_common_transition'])}
<h2>9. Decision Contract</h2><p>{html.escape(contract_text)}</p>
<h2>10. Reproducibility</h2><p>{html.escape(str(p['reproducibility_status']))}</p>
<h2>11. Topology compatibility</h2><p>{html.escape(str(p['topology_compatible']))}</p>
<h2>12. Evidence integrity</h2><p>{html.escape(p['evidence_integrity'])}</p>
<h2>13. Scientific interpretation</h2><ul>{''.join(f'<li>{html.escape(note)}</li>' for note in p['scientific_notes'])}</ul>
</body></html>"""
