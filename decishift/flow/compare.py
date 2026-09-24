from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from decishift.evidence import fingerprint_dataframe
from decishift.flow.actions import action_display_key, action_equal_mask
from decishift.flow.executor import FlowExecutor
from decishift.flow.flow import DecisionFlow
from decishift.flow.impact import structural_impact
from decishift.flow.trace import FlowTrace
from decishift.flow.validation import topology_diff


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


def _action_key(value: Any) -> str:
    """Stable readable evidence key for categorical actions."""
    return action_display_key(value)


def _transition_frame(
    records: pd.DataFrame,
    baseline_actions: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    id_column: str | None,
) -> pd.DataFrame:
    ids = records[id_column].to_numpy() if id_column else records.index.to_numpy()
    changed = ~action_equal_mask(baseline_actions, candidate_actions)
    transitions = np.asarray([
        f"{_action_key(base)}->{_action_key(cand)}" if is_changed else "unchanged"
        for base, cand, is_changed in zip(baseline_actions, candidate_actions, changed)
    ], dtype=object)
    return pd.DataFrame({
        "record_id": ids,
        "baseline_action": baseline_actions,
        "candidate_action": candidate_actions,
        "changed": changed.astype(bool),
        "transition": transitions,
    })


def _distribution(values: pd.Series) -> dict[str, int]:
    counts = values.map(_action_key).value_counts(dropna=False, sort=False)
    return {key: int(counts[key]) for key in sorted(counts.index, key=str)}


def _transition_matrix(records: pd.DataFrame) -> dict[str, dict[str, int]]:
    baseline = records["baseline_action"].map(_action_key)
    candidate = records["candidate_action"].map(_action_key)
    actions = sorted(set(baseline) | set(candidate), key=str)
    table = pd.crosstab(baseline, candidate, dropna=False).reindex(index=actions, columns=actions, fill_value=0)
    return {
        row: {column: int(table.loc[row, column]) for column in actions}
        for row in actions
    }


@dataclass
class FlowComparisonResult:
    records: pd.DataFrame
    changed_nodes: list[str] = field(default_factory=list)
    baseline_trace: FlowTrace | None = None
    candidate_trace: FlowTrace | None = None
    attribution: pd.DataFrame | None = None
    interactions: pd.DataFrame | None = None
    cohorts: pd.DataFrame | None = None
    diagnostics: Any | None = None
    outcome_analysis: dict[str, Any] | None = None
    fragility: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_components(self) -> list[str]:
        return self.changed_nodes

    @property
    def changed(self) -> pd.DataFrame:
        return self.records[self.records["changed"]].copy()

    def summary(self) -> dict[str, Any]:
        n = len(self.records)
        changed = int(self.records["changed"].sum())
        transition_counts = self.records["transition"].value_counts(sort=False)
        transition_counts_dict = {
            key: int(transition_counts[key]) for key in sorted(transition_counts.index, key=str)
        }
        transition_rates = {
            key: (count / n if n else 0.0) for key, count in transition_counts_dict.items()
        }
        changed_transitions = [(key, count) for key, count in transition_counts_dict.items() if key != "unchanged"]
        changed_transitions.sort(key=lambda item: (-item[1], item[0]))
        return {
            "mode": "flow",
            "total_records": n,
            "changed_actions": changed,
            "unchanged_actions": n - changed,
            "action_shift_rate": changed / n if n else 0.0,
            # Decision Contracts retain the historical overall field name.
            "changed_decisions": changed,
            "unchanged_decisions": n - changed,
            "decision_shift_rate": changed / n if n else 0.0,
            "agreement_rate": 1.0 - (changed / n) if n else 1.0,
            "baseline_action_distribution": _distribution(self.records["baseline_action"]),
            "candidate_action_distribution": _distribution(self.records["candidate_action"]),
            "transition_counts": transition_counts_dict,
            "transition_rates": transition_rates,
            "transition_matrix": _transition_matrix(self.records),
            "most_frequent_changed_transitions": [
                {"transition": key, "count": count, "rate": count / n if n else 0.0}
                for key, count in changed_transitions[:10]
            ],
            "changed_nodes": list(self.changed_nodes),
            **self.metadata,
        }


def compare_flows(
    baseline: DecisionFlow,
    candidate: DecisionFlow,
    records: pd.DataFrame,
    *,
    id_column: str | None = None,
    allow_duplicate_ids: bool = False,
    allow_null_ids: bool = False,
    hash_input: bool = True,
) -> FlowComparisonResult:
    """Execute two structured decision flows and compare final categorical actions."""
    _validate_ids(records, id_column, allow_duplicate_ids=allow_duplicate_ids, allow_null_ids=allow_null_ids)
    diff = topology_diff(baseline, candidate)

    # One cache is safe across independent flow executions because each key
    # includes node identity and dependency lineage. It also proves unaffected
    # branches can be reused without constructing an invalid hybrid graph.
    executor = FlowExecutor(records)
    baseline_trace = executor.evaluate(baseline)
    candidate_trace = executor.evaluate(candidate)
    frame = _transition_frame(
        records,
        baseline_trace.actions,
        candidate_trace.actions,
        id_column=id_column,
    )
    if baseline_trace.scores is not None:
        frame["baseline_score"] = baseline_trace.scores
    if candidate_trace.scores is not None:
        frame["candidate_score"] = candidate_trace.scores
    if baseline_trace.margins is not None:
        frame["baseline_margin"] = baseline_trace.margins
    if candidate_trace.margins is not None:
        frame["candidate_margin"] = candidate_trace.margins

    statuses = {baseline.reproducibility_status, candidate.reproducibility_status}
    if statuses == {"reproducible"}:
        reproducibility = "reproducible"
    elif statuses == {"unstable"}:
        reproducibility = "unstable"
    else:
        reproducibility = "partially_reproducible"

    changed_nodes = list(diff.changed_nodes)
    impact_nodes = changed_nodes + list(diff.added_nodes)
    impact = structural_impact(candidate, impact_nodes)
    metadata = {
        "mode": "flow",
        "topology_compatible": diff.topology_compatible,
        "attribution_status": "available" if diff.topology_compatible else "unsupported topology change",
        "topology_diff": diff.as_dict(),
        "baseline_topology_digest": baseline.topology_digest,
        "candidate_topology_digest": candidate.topology_digest,
        "baseline_topology": baseline.topology_payload(),
        "candidate_topology": candidate.topology_payload(),
        "baseline_node_identities": baseline.node_identities(),
        "candidate_node_identities": candidate.node_identities(),
        "attribution_groups": {
            "baseline": {name: baseline.node(name).group for name in baseline.node_names if baseline.node(name).group},
            "candidate": {name: candidate.node(name).group for name in candidate.node_names if candidate.node(name).group},
        },
        "final_node": candidate.final_node,
        "reproducibility_status": reproducibility,
        "input_data_fingerprint": fingerprint_dataframe(records, include_content=hash_input),
        "input_content_hashing_enabled": bool(hash_input),
        "id_column": id_column,
        "structural_impact": impact,
        "execution": {
            "nodes_evaluated": baseline_trace.nodes_evaluated + candidate_trace.nodes_evaluated,
            "node_outputs_reused": baseline_trace.node_outputs_reused + candidate_trace.node_outputs_reused,
            "cache_hits": baseline_trace.cache_hits + candidate_trace.cache_hits,
            "cache_misses": baseline_trace.cache_misses + candidate_trace.cache_misses,
        },
    }
    result = FlowComparisonResult(
        records=frame,
        changed_nodes=changed_nodes,
        baseline_trace=baseline_trace,
        candidate_trace=candidate_trace,
        metadata=metadata,
    )

    # Fragility is only meaningful when a numeric margin was explicitly emitted.
    if baseline_trace.margins is not None and candidate_trace.margins is not None:
        b = np.abs(baseline_trace.margins)
        c = np.abs(candidate_trace.margins)
        minimum = np.minimum(b, c)
        result.fragility = {
            "summary": {
                "median_absolute_margin": float(np.median(minimum)) if len(minimum) else 0.0,
                "p10_absolute_margin": float(np.quantile(minimum, 0.1)) if len(minimum) else 0.0,
                "explicit_margin": True,
            },
            "records": pd.DataFrame({"record_id": frame["record_id"], "minimum_absolute_margin": minimum}),
        }
    return result
