import json

import pandas as pd

from decishift.contracts import EXIT_CONTRACT_VIOLATED, EXIT_PASS, evaluate_contract
from decishift.flow import DecisionFlow, DecisionNode, compare_flows
from decishift.flow.attribution import exact_flow_attribution
from decishift.store import RunStore


class Actions:
    def __init__(self, column, version):
        self.column = column
        self.version = version

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(), index=records.index)


def _result():
    records = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "baseline": ["monitor", "monitor", "inspect", "service"],
        "candidate": ["monitor", "inspect", "service", "inspect"],
    })
    baseline = DecisionFlow([DecisionNode("action", Actions("baseline", "a1"), version="a1")], "action")
    candidate = DecisionFlow([DecisionNode("action", Actions("candidate", "a2"), version="a2")], "action")
    result = compare_flows(baseline, candidate, records, id_column="id")
    result.attribution, result.diagnostics = exact_flow_attribution(baseline, candidate, records, result=result)
    result.metadata.update({
        "attribution_status": "available",
        "attribution_method": "exact",
        "attribution_target": "candidate_action_support",
        "attribution_parameters": {"max_players": 10},
    })
    return result


def test_flow_evidence_schema_two_round_trip_and_verify(tmp_path):
    store = RunStore(root=tmp_path / "runs")
    run_id = store.save(_result())
    path = store.root / run_id
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["evidence_schema_version"] == "2.0"
    assert manifest["topology_compatible"] is True
    assert manifest["attribution_target"] == "candidate_action_support"
    assert "transition_matrix" in manifest
    verification = store.verify(run_id)
    assert verification.passed
    assert "authenticity" in verification.message.lower()
    assert "not established" in verification.message.lower()
    loaded = store.load(run_id)
    assert loaded.metadata["mode"] == "flow"
    assert loaded.summary()["action_shift_rate"] == 0.75
    assert loaded.diagnostics.efficiency_valid


def test_flow_evidence_tampering_fails_verification(tmp_path):
    store = RunStore(root=tmp_path / "runs")
    run_id = store.save(_result())
    path = store.root / run_id
    records = path / "records.csv"
    records.write_text(records.read_text() + "\n", encoding="utf-8")
    assert not store.verify(run_id).passed


def test_multi_action_contract_transition_and_action_rules():
    result = _result()
    passing = evaluate_contract(result, {
        "max_decision_shift_rate": 0.8,
        "transitions": {"monitor->inspect": {"max_count": 1, "max_rate": 0.3}},
        "candidate_actions": {"service": {"max_count": 1, "max_rate": 0.3}},
    })
    assert passing.exit_code == EXIT_PASS
    blocked = evaluate_contract(result, {
        "transitions": {"service->inspect": {"max_count": 0}},
    })
    assert blocked.exit_code == EXIT_CONTRACT_VIOLATED
