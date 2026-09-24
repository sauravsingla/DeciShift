import pandas as pd
import pytest

from decishift.flow import DecisionFlow, DecisionNode, DecisionOutput, compare_flows
from decishift.flow.analysis import analyze_multiclass_outcomes
from decishift.flow.executor import FlowExecutor


class ActionColumn:
    def __init__(self, column, version):
        self.column = column
        self.version = version

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(), index=records.index)


def _flows():
    return (
        DecisionFlow([DecisionNode("action", ActionColumn("baseline", "b1"), version="b1")], "action"),
        DecisionFlow([DecisionNode("action", ActionColumn("candidate", "c1"), version="c1")], "action"),
    )


def test_unchanged_two_action_and_three_action_transitions_and_matrix():
    records = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "baseline": ["monitor", "monitor", "inspect", "inspect", "service"],
        "candidate": ["monitor", "inspect", "inspect", "service", "inspect"],
    })
    baseline, candidate = _flows()
    result = compare_flows(baseline, candidate, records, id_column="id")
    assert result.records["transition"].tolist() == [
        "unchanged", "monitor->inspect", "unchanged", "inspect->service", "service->inspect"
    ]
    summary = result.summary()
    assert summary["changed_actions"] == 3
    assert summary["unchanged_actions"] == 2
    assert summary["transition_counts"]["monitor->inspect"] == 1
    assert summary["transition_matrix"]["monitor"]["inspect"] == 1
    assert summary["transition_matrix"]["service"]["inspect"] == 1
    assert set(summary["baseline_action_distribution"]) == {"monitor", "inspect", "service"}


def test_null_action_is_rejected():
    records = pd.DataFrame({"candidate": ["monitor", None]})
    flow = DecisionFlow([DecisionNode("action", ActionColumn("candidate", "c1"), version="c1")], "action")
    with pytest.raises(ValueError, match="null"):
        FlowExecutor(records).evaluate(flow)


def test_non_scalar_action_is_rejected():
    class Bad:
        version = "bad1"

        def run(self, records, inputs):
            return pd.Series([["monitor"], ["inspect"]], index=records.index)

    records = pd.DataFrame({"x": [1, 2]})
    flow = DecisionFlow([DecisionNode("action", Bad(), version="bad1")], "action")
    with pytest.raises(ValueError, match="scalar hashable"):
        FlowExecutor(records).evaluate(flow)


def test_decision_output_margin_is_used_only_when_explicit():
    class Structured:
        def __init__(self, version, shift):
            self.version = version
            self.shift = shift

        def run(self, records, inputs):
            score = records["x"].to_numpy(dtype=float) + self.shift
            return DecisionOutput(
                action=pd.Series(["inspect" if x >= 0.5 else "monitor" for x in score], index=records.index),
                score=score,
                margin=score - 0.5,
            )

    records = pd.DataFrame({"x": [0.45, 0.55]})
    baseline = DecisionFlow([DecisionNode("action", Structured("s1", 0.0), version="s1")], "action")
    candidate = DecisionFlow([DecisionNode("action", Structured("s2", 0.1), version="s2")], "action")
    result = compare_flows(baseline, candidate, records)
    assert result.fragility is not None
    assert result.fragility["summary"]["explicit_margin"] is True


def test_multiclass_outcomes_require_explicit_prediction_semantics():
    records = pd.DataFrame({
        "baseline": ["a", "b", "c"],
        "candidate": ["a", "c", "c"],
        "actual": ["a", "b", "c"],
    })
    baseline, candidate = _flows()
    result = compare_flows(baseline, candidate, records)
    assert analyze_multiclass_outcomes(records, result, outcome_column="actual", actions_are_predictions=False) is None
    metrics = analyze_multiclass_outcomes(records, result, outcome_column="actual", actions_are_predictions=True)
    assert metrics["baseline_accuracy"] == 1.0
    assert metrics["candidate_accuracy"] == pytest.approx(2 / 3)
    assert metrics["correctness_transitions"]["correct->incorrect"] == 1
