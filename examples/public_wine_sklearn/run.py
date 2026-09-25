from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decishift.adapters.sklearn import adapt
from decishift.flow import DecisionFlow, DecisionNode, compare_flows, exact_flow_attribution
from decishift.flow.hybrid import FlowHybridCache


FEATURE_COLUMNS = [
    "alcohol",
    "flavanoids",
    "color_intensity",
    "hue",
    "proline",
    "od280/od315_of_diluted_wines",
]


def _split() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.copy()
    target = (frame["target"].to_numpy() == 2).astype(int)
    features = frame.drop(columns=["target"])
    train_idx, eval_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.40,
        random_state=17,
        stratify=target,
    )
    return (
        features.iloc[train_idx].reset_index(drop=True),
        target[train_idx],
        features.iloc[eval_idx].reset_index(drop=True),
        target[eval_idx],
    )


class WineFeaturesV1:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.DataFrame:
        return records[FEATURE_COLUMNS].copy()


class WineFeaturesV2:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.DataFrame:
        out = records[FEATURE_COLUMNS].copy()
        out["color_intensity"] = out["color_intensity"] / out["hue"].clip(lower=0.20)
        return out


class WineModel:
    def __init__(self, c: float, version: str):
        train, target, _, _ = _split()
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2_000, random_state=17),
        )
        estimator.fit(train[FEATURE_COLUMNS], target)
        self.adapter = adapt(estimator, version=version)

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        values = self.adapter.predict(inputs["features"])
        return pd.Series(values, index=records.index, name="class_2_probability")


class WinePolicy:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        score = np.asarray(inputs["model"], dtype=float)
        action = np.full(len(records), "manual_review", dtype=object)
        action[score < self.low] = "route_not_class2"
        action[score >= self.high] = "route_class2"
        return pd.Series(action, index=records.index, name="routing_action")


class WineRules:
    def __init__(self, color_intensity_review_threshold: float):
        self.threshold = color_intensity_review_threshold

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        action = np.asarray(inputs["policy"], dtype=object).copy()
        force_review = (
            (records["color_intensity"].to_numpy(dtype=float) >= self.threshold)
            & (action == "route_not_class2")
        )
        action[force_review] = "manual_review"
        return pd.Series(action, index=records.index, name="reviewed_action")


class FinalAction:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        return pd.Series(np.asarray(inputs["rules"], dtype=object), index=records.index, name="action")


def _records() -> tuple[pd.DataFrame, np.ndarray]:
    _, _, evaluation, target = _split()
    records = evaluation.copy()
    records.insert(0, "record_id", np.arange(len(records)))
    records["actual_is_class2"] = target
    records["alcohol_band"] = pd.cut(
        records["alcohol"],
        bins=[-np.inf, 12.5, 13.5, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return records, target


def _flow(*, features: str, model: str, policy: str, rules: str, name: str) -> DecisionFlow:
    feature_component = WineFeaturesV1() if features == "v1" else WineFeaturesV2()
    model_component = WineModel(0.40, "wine_logreg_v1") if model == "v1" else WineModel(1.50, "wine_logreg_v2")
    policy_component = WinePolicy(0.25, 0.75) if policy == "v1" else WinePolicy(0.30, 0.70)
    rule_component = WineRules(8.0) if rules == "v1" else WineRules(7.0)

    return DecisionFlow(
        [
            DecisionNode("features", feature_component, version=f"wine_features_{features}"),
            DecisionNode("model", model_component, ("features",), version=f"wine_model_{model}"),
            DecisionNode("policy", policy_component, ("model",), version=f"wine_policy_{policy}"),
            DecisionNode("rules", rule_component, ("policy",), version=f"wine_rules_{rules}"),
            DecisionNode("final_action", FinalAction(), ("rules",), version="wine_final_v1"),
        ],
        "final_action",
        name=name,
    )


def _model_metrics(records: pd.DataFrame, target: np.ndarray) -> dict[str, float]:
    features = WineFeaturesV1().run(records, {})
    base = WineModel(0.40, "wine_logreg_v1").run(records, {"features": features}).to_numpy()
    candidate = WineModel(1.50, "wine_logreg_v2").run(records, {"features": features}).to_numpy()
    return {
        "baseline_accuracy": float(accuracy_score(target, base >= 0.50)),
        "candidate_accuracy": float(accuracy_score(target, candidate >= 0.50)),
        "baseline_roc_auc": float(roc_auc_score(target, base)),
        "candidate_roc_auc": float(roc_auc_score(target, candidate)),
    }


def _scenario(records: pd.DataFrame, changed: str) -> dict:
    baseline = _flow(features="v1", model="v1", policy="v1", rules="v1", name="wine-baseline")
    versions = {"features": "v1", "model": "v1", "policy": "v1", "rules": "v1"}
    if changed == "all":
        versions = {key: "v2" for key in versions}
    else:
        versions[changed] = "v2"
    candidate = _flow(name=f"wine-{changed}", **versions)

    result = compare_flows(baseline, candidate, records, id_column="record_id")
    cache = FlowHybridCache(baseline, candidate, records)
    attribution, diagnostics = exact_flow_attribution(
        baseline,
        candidate,
        records,
        result=result,
        cache=cache,
    )
    result.attribution = attribution
    result.diagnostics = diagnostics

    transitions = (
        result.records.loc[result.records["changed"], "transition"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    shares: dict[str, float] = {}
    if not attribution.empty:
        grouped = attribution.groupby("player")["target_contribution"].apply(lambda s: float(s.abs().mean()))
        total = float(grouped.sum())
        shares = {str(k): (float(v) / total if total else 0.0) for k, v in grouped.items()}

    summary = result.summary()
    return {
        "changed_component": changed,
        "changed_nodes": list(result.changed_nodes),
        "records": int(summary["total_records"]),
        "changed_actions": int(summary["changed_actions"]),
        "action_shift_rate": float(summary["action_shift_rate"]),
        "transitions": {str(k): int(v) for k, v in transitions.items()},
        "attribution_absolute_share": shares,
        "hybrid_evaluations": int(cache.evaluations),
        "nodes_executed": int(cache.nodes_evaluated),
        "node_outputs_reused": int(cache.node_outputs_reused),
        "efficiency_valid": bool(diagnostics.efficiency_valid),
    }


def run() -> dict:
    records, target = _records()
    scenarios = [_scenario(records, name) for name in ("features", "model", "policy", "rules", "all")]
    return {
        "dataset": {
            "name": "scikit-learn Wine classification dataset",
            "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
            "total_public_samples": 178,
            "evaluation_records": len(records),
            "target_for_example": "class_2 versus all other classes",
            "split_random_state": 17,
        },
        "model_metrics_on_baseline_features": _model_metrics(records, target),
        "scenarios": scenarios,
    }


def _print(payload: dict) -> None:
    metrics = payload["model_metrics_on_baseline_features"]
    print("Public Wine / scikit-learn DecisionFlow")
    print("=" * 72)
    print(
        f"model accuracy: {metrics['baseline_accuracy']:.4f} -> {metrics['candidate_accuracy']:.4f}; "
        f"ROC AUC: {metrics['baseline_roc_auc']:.4f} -> {metrics['candidate_roc_auc']:.4f}"
    )
    print()
    print("change       changed nodes                     actions changed   shift")
    print("-" * 72)
    for row in payload["scenarios"]:
        nodes = ",".join(row["changed_nodes"])
        print(
            f"{row['changed_component']:<12} {nodes:<32} "
            f"{row['changed_actions']:>7}/{row['records']:<7} {row['action_shift_rate']:>7.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    payload = run()
    _print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
