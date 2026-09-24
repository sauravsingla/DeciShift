import json

import pandas as pd
import pytest

from decishift.core.exceptions import ComponentExecutionError, InsufficientEvidenceError
from decishift.flow import DecisionFlow, DecisionNode, compare_flows
from decishift.flow.attribution import exact_flow_attribution
from decishift.flow.executor import FlowExecutor
from decishift.flow.reports import render_flow_html, render_flow_json, render_flow_markdown, render_flow_terminal
from decishift.store import RunStore


class ColumnAction:
    def __init__(self, column, version):
        self.column = column
        self.version = version

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(), index=records.index)


def test_numeric_and_string_action_labels_do_not_collapse():
    records = pd.DataFrame({"baseline": [1, "1"], "candidate": ["1", 1]})
    baseline = DecisionFlow([DecisionNode("action", ColumnAction("baseline", "b1"), version="b1")], "action")
    candidate = DecisionFlow([DecisionNode("action", ColumnAction("candidate", "c1"), version="c1")], "action")
    result = compare_flows(baseline, candidate, records)
    summary = result.summary()
    assert summary["baseline_action_distribution"] == {"1": 1, "int:1": 1}
    assert summary["candidate_action_distribution"] == {"1": 1, "int:1": 1}
    assert set(summary["transition_counts"]) == {"1->int:1", "int:1->1"}


def test_changed_final_node_disables_hybrid_attribution_but_observed_comparison_works():
    records = pd.DataFrame({"base": ["monitor"], "cand": ["inspect"]})
    baseline = DecisionFlow([
        DecisionNode("a", ColumnAction("base", "a1"), version="a1"),
        DecisionNode("b", ColumnAction("base", "b1"), version="b1"),
    ], "a")
    candidate = DecisionFlow([
        DecisionNode("a", ColumnAction("base", "a1"), version="a1"),
        DecisionNode("b", ColumnAction("cand", "b2"), version="b2"),
    ], "b")
    result = compare_flows(baseline, candidate, records)
    assert result.records.transition.iloc[0] == "monitor->inspect"
    assert result.metadata["topology_diff"]["changed_final_node"] is True
    assert result.metadata["attribution_status"] == "unsupported topology change"
    with pytest.raises(InsufficientEvidenceError):
        exact_flow_attribution(baseline, candidate, records, result=result)


def test_user_internal_typeerror_is_not_retried_as_an_arity_mismatch():
    class Broken:
        version = "broken_v1"

        def run(self, records, inputs):
            raise TypeError("internal user failure")

    flow = DecisionFlow([DecisionNode("final", Broken(), version="broken_v1")], "final")
    with pytest.raises(ComponentExecutionError, match="internal user failure") as exc:
        FlowExecutor(pd.DataFrame({"x": [1]})).evaluate(flow)
    assert isinstance(exc.value.__cause__, TypeError)


def test_custom_attribution_target_must_be_numeric():
    class TextTarget:
        name = "text"

        def evaluate(self, trace, baseline_trace, candidate_trace):
            return ["not numeric"] * len(trace.actions)

    records = pd.DataFrame({"baseline": ["monitor"], "candidate": ["inspect"]})
    baseline = DecisionFlow([DecisionNode("action", ColumnAction("baseline", "b1"), version="b1")], "action")
    candidate = DecisionFlow([DecisionNode("action", ColumnAction("candidate", "c1"), version="c1")], "action")
    result = compare_flows(baseline, candidate, records)
    with pytest.raises(TypeError, match="numeric"):
        exact_flow_attribution(baseline, candidate, records, result=result, target=TextTarget())


def test_flow_reports_surface_required_sections():
    records = pd.DataFrame({"baseline": ["monitor", "inspect"], "candidate": ["inspect", "inspect"]})
    baseline = DecisionFlow([DecisionNode("action", ColumnAction("baseline", "b1"), version="b1")], "action")
    candidate = DecisionFlow([DecisionNode("action", ColumnAction("candidate", "c1"), version="c1")], "action")
    result = compare_flows(baseline, candidate, records)
    result.attribution, result.diagnostics = exact_flow_attribution(baseline, candidate, records, result=result)
    result.metadata["attribution_target"] = "candidate_action_support"
    result.metadata["attribution_method"] = "exact"
    result.metadata["contract_evaluation"] = {"result": "PASS", "checks": []}
    terminal = render_flow_terminal(result)
    markdown = render_flow_markdown(result)
    html = render_flow_html(result)
    payload = json.loads(render_flow_json(result))
    for label in ("Transition matrix", "Structural impact", "Pairwise interactions", "Decision Contract", "Evidence integrity"):
        assert label in terminal
    assert "## 13. Scientific interpretation" in markdown
    assert "<h2>13. Scientific interpretation</h2>" in html
    assert "decision_contract" in payload
    assert "pairwise_interactions" in payload
    assert "topology_compatible" in payload


def test_unstable_runtime_identity_is_not_serialized_as_reproducible_evidence(tmp_path):
    records = pd.DataFrame({"x": ["monitor", "inspect"]})

    def local_component(frame, inputs):
        return pd.Series(frame["x"].to_numpy(), index=frame.index)

    baseline = DecisionFlow([DecisionNode("action", local_component)], "action")
    candidate = DecisionFlow([DecisionNode("action", local_component)], "action")
    result = compare_flows(baseline, candidate, records)
    store = RunStore(root=tmp_path / "runs")
    run_id = store.save(result)
    manifest_text = (store.root / run_id / "manifest.json").read_text()
    assert "runtime-only:" not in manifest_text
    manifest = json.loads(manifest_text)
    identity = manifest["baseline_node_identities"]["action"]
    assert "0x" not in json.dumps(identity)
