import numpy as np
import pandas as pd
import pytest

from decishift.flow import DecisionFlow, DecisionNode, compare_flows
from decishift.flow.attribution import (
    approximate_flow_attribution,
    exact_flow_attribution,
    flow_pairwise_interactions,
)


class ConstantSignal:
    def __init__(self, value, version):
        self.value = value
        self.version = version

    def run(self, records, inputs):
        return pd.Series(np.full(len(records), self.value, dtype=float), index=records.index)


class PairPolicy:
    version = "policy_v1"

    def run(self, records, inputs):
        total = np.asarray(inputs["a"], dtype=float) + np.asarray(inputs["b"], dtype=float)
        return pd.Series(np.where(total >= 2.0, "inspect", "monitor"), index=records.index)


class PassFinal:
    version = "final_v1"

    def run(self, records, inputs):
        return pd.Series(np.asarray(inputs["policy"], dtype=object), index=records.index)


def _interaction_flows(*, group=False, dummy=False):
    group_name = "signals" if group else None
    baseline_nodes = [
        DecisionNode("a", ConstantSignal(0.0, "a1"), version="a1", group=group_name),
        DecisionNode("b", ConstantSignal(0.0, "b1"), version="b1", group=group_name),
        DecisionNode("policy", PairPolicy(), ("a", "b"), version="p1"),
        DecisionNode("final", PassFinal(), ("policy",), version="f1"),
    ]
    candidate_nodes = [
        DecisionNode("a", ConstantSignal(1.0, "a2"), version="a2", group=group_name),
        DecisionNode("b", ConstantSignal(1.0, "b2"), version="b2", group=group_name),
        DecisionNode("policy", PairPolicy(), ("a", "b"), version="p1"),
        DecisionNode("final", PassFinal(), ("policy",), version="f1"),
    ]
    if dummy:
        baseline_nodes.append(DecisionNode("dummy", ConstantSignal(0.0, "d1"), version="d1"))
        candidate_nodes.append(DecisionNode("dummy", ConstantSignal(0.0, "d2"), version="d2"))
    return DecisionFlow(baseline_nodes, "final"), DecisionFlow(candidate_nodes, "final")


def test_candidate_action_support_efficiency_symmetry_and_dummy():
    records = pd.DataFrame({"id": [1, 2]})
    baseline, candidate = _interaction_flows(dummy=True)
    result = compare_flows(baseline, candidate, records, id_column="id")
    attr, diagnostics = exact_flow_attribution(
        baseline, candidate, records, result=result, target="candidate_action_support"
    )
    means = attr.groupby("player")["target_contribution"].mean()
    assert means["a"] == pytest.approx(0.5)
    assert means["b"] == pytest.approx(0.5)
    assert means["dummy"] == pytest.approx(0.0)
    assert diagnostics.efficiency_valid
    assert diagnostics.efficiency_max < 1e-12


def test_change_from_baseline_efficiency():
    records = pd.DataFrame({"id": [1]})
    baseline, candidate = _interaction_flows()
    result = compare_flows(baseline, candidate, records, id_column="id")
    attr, diagnostics = exact_flow_attribution(
        baseline, candidate, records, result=result, target="change_from_baseline"
    )
    assert attr["target_contribution"].sum() == pytest.approx(1.0)
    assert diagnostics.efficiency_valid


def test_declared_group_attribution():
    records = pd.DataFrame({"id": [1]})
    baseline, candidate = _interaction_flows(group=True)
    result = compare_flows(baseline, candidate, records, id_column="id")
    attr, diagnostics = exact_flow_attribution(
        baseline, candidate, records, result=result, grouped=True
    )
    assert attr["player"].unique().tolist() == ["signals"]
    assert attr["target_contribution"].iloc[0] == pytest.approx(1.0)
    assert diagnostics.efficiency_valid


def test_interaction_only_transition_is_identified():
    records = pd.DataFrame({"id": [1]})
    baseline, candidate = _interaction_flows()
    result = compare_flows(baseline, candidate, records, id_column="id")
    interactions = flow_pairwise_interactions(baseline, candidate, records, result=result)
    row = interactions[(interactions.player_a == "a") & (interactions.player_b == "b")].iloc[0]
    assert bool(row.interaction_only_transition)
    assert row.resulting_transition == "monitor->inspect"
    assert row.target_interaction == pytest.approx(1.0)


def test_exact_and_approximate_flow_attribution_agree():
    records = pd.DataFrame({"id": range(20)})
    baseline, candidate = _interaction_flows()
    result = compare_flows(baseline, candidate, records, id_column="id")
    exact, _ = exact_flow_attribution(baseline, candidate, records, result=result)
    approx, diagnostics = approximate_flow_attribution(
        baseline, candidate, records, result=result, permutations=2000, seed=13
    )
    exact_mean = exact.groupby("player")["target_contribution"].mean()
    approx_mean = approx.groupby("player")["target_contribution"].mean()
    assert np.allclose(exact_mean.sort_index(), approx_mean.sort_index(), atol=0.04)
    assert diagnostics.efficiency_valid


def test_adaptive_flow_early_stopping_for_deterministic_single_player():
    class Action:
        def __init__(self, action, version):
            self.action = action
            self.version = version

        def run(self, records, inputs):
            return pd.Series([self.action] * len(records), index=records.index)

    records = pd.DataFrame({"id": [1, 2]})
    baseline = DecisionFlow([DecisionNode("action", Action("monitor", "a1"), version="a1")], "action")
    candidate = DecisionFlow([DecisionNode("action", Action("inspect", "a2"), version="a2")], "action")
    result = compare_flows(baseline, candidate, records, id_column="id")
    attr, diagnostics = approximate_flow_attribution(
        baseline,
        candidate,
        records,
        result=result,
        min_permutations=4,
        max_permutations=40,
        batch_size=2,
        target_ci_width=0.01,
        seed=4,
    )
    assert attr.attrs["permutations_used"] == 4
    assert attr.attrs["stopped_early"] is True
    assert diagnostics.sampling_precision_sufficient
    assert diagnostics.sampling_converged


def test_adaptive_flow_maximum_permutation_fallback():
    records = pd.DataFrame({"id": [1]})
    baseline, candidate = _interaction_flows()
    result = compare_flows(baseline, candidate, records, id_column="id")
    attr, diagnostics = approximate_flow_attribution(
        baseline,
        candidate,
        records,
        result=result,
        min_permutations=4,
        max_permutations=8,
        batch_size=4,
        target_ci_width=1e-12,
        seed=5,
    )
    assert attr.attrs["permutations_used"] == 8
    assert attr.attrs["stopped_early"] is False
    assert not diagnostics.sampling_precision_sufficient
