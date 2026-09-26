"""Minimal DecisionFlow comparison where only a downstream rule changes."""

from __future__ import annotations

import pandas as pd

from decishift import DecisionFlow, DecisionNode, FlowComparisonResult, compare_flows


class FixedScore:
    """Return the same precomputed score in both flow versions."""

    def run(self, records: pd.DataFrame, inputs: dict[str, object]) -> pd.Series:
        return records["score"].copy()


class ReviewRule:
    """Route rows at or above a deterministic threshold to review."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def run(self, records: pd.DataFrame, inputs: dict[str, object]) -> pd.Series:
        scores = inputs["score"]
        assert isinstance(scores, pd.Series)
        return scores.ge(self.threshold).map({True: "review", False: "approve"})


def build_comparison() -> FlowComparisonResult:
    records = pd.DataFrame(
        {
            "case_id": ["case-a", "case-b", "case-c"],
            "score": [0.40, 0.65, 0.85],
        }
    )
    shared_score = DecisionNode("score", FixedScore(), version="fixed-score-v1")
    baseline = DecisionFlow(
        [shared_score, DecisionNode("rule", ReviewRule(0.70), ("score",), version="threshold-0.70")],
        "rule",
        name="baseline-rule",
    )
    candidate = DecisionFlow(
        [shared_score, DecisionNode("rule", ReviewRule(0.60), ("score",), version="threshold-0.60")],
        "rule",
        name="candidate-rule",
    )
    return compare_flows(baseline, candidate, records, id_column="case_id")


def main() -> None:
    comparison = build_comparison()
    columns = ["record_id", "baseline_action", "candidate_action", "transition"]

    print("Changed nodes:", ", ".join(comparison.changed_nodes))
    print(comparison.records.loc[comparison.records["changed"], columns].to_string(index=False))
    print("This synthetic example demonstrates software behavior only; it makes no real-world causal claim.")


if __name__ == "__main__":
    main()
