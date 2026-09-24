import pandas as pd
import pytest

from decishift.core.exceptions import InsufficientEvidenceError
from decishift.flow import DecisionFlow, DecisionNode, compare_flows
from decishift.flow.attribution import exact_flow_attribution
from decishift.flow.validation import topology_diff


class Source:
    def __init__(self, column, version):
        self.column = column
        self.version = version

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(), index=records.index)


class Pass:
    def __init__(self, version):
        self.version = version

    def run(self, records, inputs):
        return pd.Series(next(iter(inputs.values())).to_numpy(), index=records.index)


def test_same_topology_changed_versions_supports_attribution():
    records = pd.DataFrame({"base": ["monitor"], "cand": ["inspect"]})
    baseline = DecisionFlow([
        DecisionNode("source", Source("base", "s1"), version="s1"),
        DecisionNode("final", Pass("f1"), ("source",), version="f1"),
    ], "final")
    candidate = DecisionFlow([
        DecisionNode("source", Source("cand", "s2"), version="s2"),
        DecisionNode("final", Pass("f1"), ("source",), version="f1"),
    ], "final")
    diff = topology_diff(baseline, candidate)
    assert diff.topology_compatible
    result = compare_flows(baseline, candidate, records)
    attr, diagnostics = exact_flow_attribution(baseline, candidate, records, result=result)
    assert not attr.empty
    assert diagnostics.efficiency_valid


def _base_flow(column="base"):
    return DecisionFlow([
        DecisionNode("source", Source(column, "s1"), version="s1"),
        DecisionNode("final", Pass("f1"), ("source",), version="f1"),
    ], "final")


def test_added_node_disables_node_attribution_but_comparison_runs():
    records = pd.DataFrame({"base": ["monitor"], "cand": ["inspect"]})
    baseline = _base_flow("base")
    candidate = DecisionFlow([
        DecisionNode("source", Source("cand", "s2"), version="s2"),
        DecisionNode("extra", Pass("e1"), ("source",), version="e1"),
        DecisionNode("final", Pass("f1"), ("extra",), version="f1"),
    ], "final")
    result = compare_flows(baseline, candidate, records)
    assert not result.metadata["topology_compatible"]
    assert result.metadata["attribution_status"] == "unsupported topology change"
    assert result.records.candidate_action.iloc[0] == "inspect"
    with pytest.raises(InsufficientEvidenceError, match="topology changed"):
        exact_flow_attribution(baseline, candidate, records, result=result)


def test_removed_node_disables_node_attribution():
    records = pd.DataFrame({"base": ["monitor"], "cand": ["inspect"]})
    baseline = DecisionFlow([
        DecisionNode("source", Source("base", "s1"), version="s1"),
        DecisionNode("middle", Pass("m1"), ("source",), version="m1"),
        DecisionNode("final", Pass("f1"), ("middle",), version="f1"),
    ], "final")
    candidate = DecisionFlow([
        DecisionNode("source", Source("cand", "s2"), version="s2"),
        DecisionNode("final", Pass("f1"), ("source",), version="f1"),
    ], "final")
    result = compare_flows(baseline, candidate, records)
    assert result.metadata["topology_diff"]["removed_nodes"] == ["middle"]
    assert not result.metadata["topology_compatible"]


def test_edge_change_disables_node_attribution():
    records = pd.DataFrame({"base": ["monitor"], "cand": ["inspect"]})
    baseline = DecisionFlow([
        DecisionNode("a", Source("base", "a1"), version="a1"),
        DecisionNode("b", Source("cand", "b1"), version="b1"),
        DecisionNode("final", Pass("f1"), ("a",), version="f1"),
    ], "final")
    candidate = DecisionFlow([
        DecisionNode("a", Source("base", "a1"), version="a1"),
        DecisionNode("b", Source("cand", "b1"), version="b1"),
        DecisionNode("final", Pass("f1"), ("b",), version="f1"),
    ], "final")
    result = compare_flows(baseline, candidate, records)
    assert result.metadata["topology_diff"]["changed_dependencies"] == ["final"]
    assert not result.metadata["topology_compatible"]
    assert result.records.transition.iloc[0] == "monitor->inspect"
