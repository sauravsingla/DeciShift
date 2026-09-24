import numpy as np
import pandas as pd
import pytest

from decishift.flow import DecisionFlow, DecisionNode, FlowValidationError
from decishift.flow.executor import FlowExecutor


class Column:
    def __init__(self, column, version, shift=0.0):
        self.column = column
        self.version = version
        self.shift = shift

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(dtype=float) + self.shift, index=records.index)


class Merge:
    version = "merge_v1"

    def run(self, records, inputs):
        return pd.Series(
            np.asarray(inputs["left"], dtype=float) + np.asarray(inputs["right"], dtype=float),
            index=records.index,
        )


class Final:
    version = "final_v1"

    def run(self, records, inputs):
        values = np.asarray(next(iter(inputs.values())), dtype=float)
        return pd.Series(np.where(values >= 1.0, "inspect", "monitor"), index=records.index)


def test_deterministic_topological_order_and_digest():
    a = DecisionNode("a", Column("x", "a1"), version="a1")
    b = DecisionNode("b", Column("y", "b1"), version="b1")
    merge = DecisionNode("merge", Merge(), ("b", "a"), version="m1")
    final = DecisionNode("final", Final(), ("merge",), version="f1")
    one = DecisionFlow([final, merge, b, a], "final")
    two = DecisionFlow([a, b, merge, final], "final")
    assert one.topological_order == ("a", "b", "merge", "final")
    assert one.topological_order == two.topological_order
    assert one.topology_digest == two.topology_digest


def test_cycle_missing_dependency_duplicate_and_missing_final_are_rejected():
    with pytest.raises(FlowValidationError, match="cycle"):
        DecisionFlow([
            DecisionNode("a", Column("x", "a"), ("b",), version="a"),
            DecisionNode("b", Column("x", "b"), ("a",), version="b"),
        ], "a")
    with pytest.raises(FlowValidationError, match="missing node"):
        DecisionFlow([DecisionNode("a", Column("x", "a"), ("missing",), version="a")], "a")
    with pytest.raises(FlowValidationError, match="Duplicate node"):
        DecisionFlow([
            DecisionNode("a", Column("x", "a1"), version="a1"),
            DecisionNode("a", Column("x", "a2"), version="a2"),
        ], "a")
    with pytest.raises(FlowValidationError, match="final node"):
        DecisionFlow([DecisionNode("a", Column("x", "a"), version="a")], "missing")


def test_branch_and_merge_execution():
    records = pd.DataFrame({"x": [0.1, 0.5], "y": [0.2, 0.7]})
    flow = DecisionFlow([
        DecisionNode("left", Column("x", "l1"), version="l1"),
        DecisionNode("right", Column("y", "r1"), version="r1"),
        DecisionNode("merge", Merge(), ("left", "right"), version="m1"),
        DecisionNode("final", Final(), ("merge",), version="f1"),
    ], "final")
    trace = FlowExecutor(records).evaluate(flow)
    assert trace.actions.tolist() == ["monitor", "inspect"]
    assert np.allclose(trace.output("merge"), [0.3, 1.2])


def test_graph_cache_reuses_unaffected_branch_and_invalidates_descendants():
    records = pd.DataFrame({"x": [0.2, 0.4], "y": [0.3, 0.4]})
    baseline = DecisionFlow([
        DecisionNode("left", Column("x", "l1"), version="l1"),
        DecisionNode("right", Column("y", "r1"), version="r1"),
        DecisionNode("merge", Merge(), ("left", "right"), version="m1"),
        DecisionNode("final", Final(), ("merge",), version="f1"),
    ], "final")
    candidate = DecisionFlow([
        DecisionNode("left", Column("x", "l2", shift=0.6), version="l2"),
        DecisionNode("right", Column("y", "r1"), version="r1"),
        DecisionNode("merge", Merge(), ("left", "right"), version="m1"),
        DecisionNode("final", Final(), ("merge",), version="f1"),
    ], "final")
    executor = FlowExecutor(records)
    first = executor.evaluate(baseline)
    second = executor.evaluate(candidate)
    assert first.cache_misses == 4
    assert second.cache_hits == 1  # unchanged right branch
    assert second.cache_misses == 3  # changed left + merge + final descendants
    assert second.actions.tolist() != first.actions.tolist()


def test_row_reordering_is_rejected():
    class Reorder:
        version = "bad1"

        def run(self, records, inputs):
            return records[["x"]].iloc[::-1]

    records = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 11, 12])
    flow = DecisionFlow([DecisionNode("final", Reorder(), version="bad1")], "final")
    with pytest.raises(ValueError, match="reordered or changed index"):
        FlowExecutor(records).evaluate(flow)
