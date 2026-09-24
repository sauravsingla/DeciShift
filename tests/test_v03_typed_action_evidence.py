import pandas as pd

from decishift.contracts import evaluate_contract
from decishift.flow import DecisionFlow, DecisionNode, compare_flows
from decishift.store import RunStore


class ColumnAction:
    def __init__(self, column, version):
        self.column = column
        self.version = version

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(), index=records.index)


def test_saved_flow_preserves_mixed_typed_action_semantics(tmp_path):
    records = pd.DataFrame(
        {
            "baseline": [True, 1],
            "candidate": [1, True],
        },
        dtype=object,
    )
    baseline = DecisionFlow(
        [DecisionNode("action", ColumnAction("baseline", "b1"), version="b1")],
        "action",
    )
    candidate = DecisionFlow(
        [DecisionNode("action", ColumnAction("candidate", "c1"), version="c1")],
        "action",
    )
    result = compare_flows(baseline, candidate, records)
    assert result.records["candidate_action_key"].tolist() == ["int:1", "bool:true"]
    assert result.records["transition"].tolist() == ["bool:true->int:1", "int:1->bool:true"]

    store = RunStore(root=tmp_path / "runs")
    run_id = store.save(result)
    assert store.verify(run_id).passed is True
    loaded = store.load(run_id)

    summary = loaded.summary()
    assert summary["changed_actions"] == 2
    assert summary["candidate_action_distribution"] == {"bool:true": 1, "int:1": 1}
    assert summary["transition_matrix"]["bool:true"]["int:1"] == 1
    assert summary["transition_matrix"]["int:1"]["bool:true"] == 1

    integer_limit = evaluate_contract(loaded, {"candidate_actions": {1: {"max_count": 0}}})
    boolean_limit = evaluate_contract(loaded, {"candidate_actions": {True: {"max_count": 0}}})
    assert integer_limit.result == "BLOCK"
    assert boolean_limit.result == "BLOCK"
    assert integer_limit.checks[0].observed == 1
    assert boolean_limit.checks[0].observed == 1
