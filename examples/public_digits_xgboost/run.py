from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from decishift.adapters.xgboost import adapt
from decishift.contracts import evaluate_contract
from decishift.flow import DecisionFlow, DecisionNode, compare_flows, exact_flow_attribution
from decishift.flow.analysis import analyze_flow_cohorts
from decishift.flow.hybrid import FlowHybridCache


PIXEL_COLUMNS = [f"pixel_{i:02d}" for i in range(64)]


def _split() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = load_digits()
    frame = pd.DataFrame(dataset.data, columns=PIXEL_COLUMNS)
    actual_digit = np.asarray(dataset.target, dtype=int)
    target = (actual_digit == 8).astype(int)
    train_idx, eval_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.40,
        random_state=23,
        stratify=target,
    )
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        target[train_idx],
        actual_digit[train_idx],
        frame.iloc[eval_idx].reset_index(drop=True),
        target[eval_idx],
        actual_digit[eval_idx],
    )


class DigitFeaturesV1:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.DataFrame:
        return records[PIXEL_COLUMNS].astype(float) / 16.0


class DigitFeaturesV2:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.DataFrame:
        values = records[PIXEL_COLUMNS].to_numpy(dtype=float) / 16.0
        values = np.where(values < 0.13, 0.0, values)
        return pd.DataFrame(values, columns=PIXEL_COLUMNS, index=records.index)


class DigitXGBoostModel:
    def __init__(self, rounds: int, version: str):
        train, target, _, _, _, _ = _split()
        train_features = DigitFeaturesV1().run(train, {})
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,
            "eta": 0.08,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "seed": 23,
            "nthread": 1,
        }
        booster = xgb.train(params, xgb.DMatrix(train_features, label=target), num_boost_round=rounds)
        self.adapter = adapt(booster, version=version)

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        values = self.adapter.predict(inputs["features"])
        return pd.Series(values, index=records.index, name="digit_8_probability")


class DigitPolicy:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        score = np.asarray(inputs["model"], dtype=float)
        action = np.full(len(records), "manual_review", dtype=object)
        action[score < self.low] = "auto_not_8"
        action[score >= self.high] = "auto_8"
        return pd.Series(action, index=records.index, name="ocr_action")


class DigitRules:
    def __init__(self, center_density_review_threshold: float):
        self.threshold = center_density_review_threshold

    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        action = np.asarray(inputs["policy"], dtype=object).copy()
        images = records[PIXEL_COLUMNS].to_numpy(dtype=float).reshape(-1, 8, 8) / 16.0
        center_density = images[:, 2:6, 2:6].mean(axis=(1, 2))
        force_review = (action == "auto_not_8") & (center_density >= self.threshold)
        action[force_review] = "manual_review"
        return pd.Series(action, index=records.index, name="reviewed_action")


class FinalAction:
    def run(self, records: pd.DataFrame, inputs: dict) -> pd.Series:
        return pd.Series(np.asarray(inputs["rules"], dtype=object), index=records.index, name="action")


def _records() -> tuple[pd.DataFrame, np.ndarray]:
    _, _, _, evaluation, target, actual_digit = _split()
    records = evaluation.copy()
    records.insert(0, "record_id", np.arange(len(records)))
    records["actual_digit"] = actual_digit.astype(str)
    records["actual_is_8"] = target
    return records, target


def _flow(*, features: str, model: str, policy: str, rules: str, name: str) -> DecisionFlow:
    feature_component = DigitFeaturesV1() if features == "v1" else DigitFeaturesV2()
    model_component = (
        DigitXGBoostModel(20, "digits_xgb_v1")
        if model == "v1"
        else DigitXGBoostModel(25, "digits_xgb_v2")
    )
    policy_component = DigitPolicy(0.10, 0.30) if policy == "v1" else DigitPolicy(0.0905, 0.2675)
    rule_component = DigitRules(0.74) if rules == "v1" else DigitRules(0.75)

    return DecisionFlow(
        [
            DecisionNode("features", feature_component, version=f"digits_features_{features}"),
            DecisionNode("model", model_component, ("features",), version=f"digits_xgb_model_{model}"),
            DecisionNode("policy", policy_component, ("model",), version=f"digits_policy_{policy}"),
            DecisionNode("rules", rule_component, ("policy",), version=f"digits_rules_{rules}"),
            DecisionNode("final_action", FinalAction(), ("rules",), version="digits_final_v1"),
        ],
        "final_action",
        name=name,
    )


def _model_metrics(records: pd.DataFrame, target: np.ndarray) -> dict[str, float]:
    features = DigitFeaturesV1().run(records, {})
    baseline = DigitXGBoostModel(20, "digits_xgb_v1").run(records, {"features": features}).to_numpy()
    candidate = DigitXGBoostModel(25, "digits_xgb_v2").run(records, {"features": features}).to_numpy()
    return {
        "baseline_accuracy": float(accuracy_score(target, baseline >= 0.50)),
        "candidate_accuracy": float(accuracy_score(target, candidate >= 0.50)),
        "baseline_roc_auc": float(roc_auc_score(target, baseline)),
        "candidate_roc_auc": float(roc_auc_score(target, candidate)),
    }


def _attributed_comparison(
    records: pd.DataFrame,
    baseline: DecisionFlow,
    candidate: DecisionFlow,
) -> tuple[object, FlowHybridCache]:
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
    result.metadata["attribution_method"] = "exact"
    result.metadata["attribution_status"] = "available"
    return result, cache


def _attribution_shares(attribution: pd.DataFrame | None) -> dict[str, float]:
    if attribution is None or attribution.empty:
        return {}
    grouped = attribution.groupby("player")["target_contribution"].apply(lambda s: float(s.abs().mean()))
    total = float(grouped.sum())
    return {str(k): (float(v) / total if total else 0.0) for k, v in grouped.items()}


def _scenario(records: pd.DataFrame, changed: str) -> dict:
    baseline = _flow(features="v1", model="v1", policy="v1", rules="v1", name="digits-baseline")
    versions = {"features": "v1", "model": "v1", "policy": "v1", "rules": "v1"}
    if changed == "all":
        versions = {key: "v2" for key in versions}
    else:
        versions[changed] = "v2"
    candidate = _flow(name=f"digits-{changed}", **versions)
    result, cache = _attributed_comparison(records, baseline, candidate)
    summary = result.summary()
    return {
        "changed_component": changed,
        "changed_nodes": list(result.changed_nodes),
        "records": int(summary["total_records"]),
        "changed_actions": int(summary["changed_actions"]),
        "action_shift_rate": float(summary["action_shift_rate"]),
        "transitions": {
            str(k): int(v)
            for k, v in result.records.loc[result.records["changed"], "transition"].value_counts().sort_index().items()
        },
        "attribution_absolute_share": _attribution_shares(result.attribution),
        "hybrid_evaluations": int(cache.evaluations),
        "nodes_executed": int(cache.nodes_evaluated),
        "node_outputs_reused": int(cache.node_outputs_reused),
        "efficiency_valid": bool(result.diagnostics.efficiency_valid),
    }


def _case_study(records: pd.DataFrame, model_metrics: dict[str, float]) -> dict:
    baseline = _flow(features="v1", model="v1", policy="v1", rules="v1", name="digits-case-baseline")
    candidate = _flow(features="v1", model="v2", policy="v2", rules="v2", name="digits-case-candidate")
    result, cache = _attributed_comparison(records, baseline, candidate)

    result.cohorts = analyze_flow_cohorts(
        records,
        result,
        columns=["actual_digit"],
        id_column="record_id",
        min_size=20,
        requested_transitions=["auto_not_8->manual_review", "manual_review->auto_8"],
    )
    contract = {
        "max_decision_shift_rate": 0.05,
        "transitions": {
            "auto_not_8->manual_review": {"max_rate": 0.03},
            "manual_review->auto_8": {"max_rate": 0.02},
        },
        "attribution": {
            "require_reproducible_identity": True,
            "max_efficiency_mae": 0.000001,
        },
        "cohorts": {
            "min_size": 20,
            "max_action_shift_rate": 0.10,
            "overrides": {
                "actual_digit=6": {"max_action_shift_rate": 0.08},
            },
        },
    }
    evaluation = evaluate_contract(result, contract, integrity_passed=True)

    baseline_counts = Counter(result.records["baseline_action"].astype(str))
    candidate_counts = Counter(result.records["candidate_action"].astype(str))
    transitions = Counter(result.records.loc[result.records["changed"], "transition"].astype(str))
    cohort_rows = result.cohorts.to_dict(orient="records")

    return {
        "model_metrics": model_metrics,
        "records": len(records),
        "changed_actions": int(result.records["changed"].sum()),
        "action_shift_rate": float(result.records["changed"].mean()),
        "baseline_action_distribution": {k: int(v) for k, v in sorted(baseline_counts.items())},
        "candidate_action_distribution": {k: int(v) for k, v in sorted(candidate_counts.items())},
        "transitions": {k: int(v) for k, v in sorted(transitions.items())},
        "changed_nodes": list(result.changed_nodes),
        "attribution_absolute_share": _attribution_shares(result.attribution),
        "cohorts": cohort_rows,
        "contract": contract,
        "contract_evaluation": evaluation.as_dict(),
        "hybrid_evaluations": int(cache.evaluations),
        "nodes_executed": int(cache.nodes_evaluated),
        "node_outputs_reused": int(cache.node_outputs_reused),
        "efficiency_valid": bool(result.diagnostics.efficiency_valid),
    }


def run() -> dict:
    records, target = _records()
    model_metrics = _model_metrics(records, target)
    return {
        "dataset": {
            "name": "scikit-learn Digits classification dataset",
            "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "total_public_samples": 1797,
            "evaluation_records": len(records),
            "target_for_example": "digit 8 versus all other digits",
            "split_random_state": 23,
        },
        "model_metrics_on_baseline_features": model_metrics,
        "scenarios": [_scenario(records, name) for name in ("features", "model", "policy", "rules", "all")],
        "decision_change_case_study": _case_study(records, model_metrics),
    }


def _case_markdown(payload: dict) -> str:
    case = payload["decision_change_case_study"]
    metrics = case["model_metrics"]
    lines = [
        "# Decision-change case study — public Digits data",
        "",
        "This file is machine-generated by `examples/public_digits_xgboost/run.py`.",
        "",
        f"- Evaluation records: **{case['records']}**",
        f"- Model accuracy: **{metrics['baseline_accuracy']:.2%} → {metrics['candidate_accuracy']:.2%}**",
        f"- Model ROC AUC: **{metrics['baseline_roc_auc']:.4f} → {metrics['candidate_roc_auc']:.4f}**",
        f"- Individual final actions changed: **{case['changed_actions']} ({case['action_shift_rate']:.2%})**",
        f"- Changed nodes: **{', '.join(case['changed_nodes'])}**",
        "",
        "## Final action distributions",
        "",
        "| action | baseline | candidate |",
        "|---|---:|---:|",
    ]
    actions = sorted(set(case["baseline_action_distribution"]) | set(case["candidate_action_distribution"]))
    for action in actions:
        lines.append(
            f"| {action} | {case['baseline_action_distribution'].get(action, 0)} | "
            f"{case['candidate_action_distribution'].get(action, 0)} |"
        )
    lines += [
        "",
        "## Changed transitions",
        "",
        "| transition | count |",
        "|---|---:|",
    ]
    for transition, count in case["transitions"].items():
        lines.append(f"| {transition} | {count} |")
    lines += [
        "",
        "## Software-counterfactual attribution",
        "",
        "| node | absolute share |",
        "|---|---:|",
    ]
    for node, share in sorted(case["attribution_absolute_share"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {node} | {share:.2%} |")
    lines += [
        "",
        "## Governance contract",
        "",
        f"Result: **{case['contract_evaluation']['result']}**",
        "",
        "| check | status | observed | limit |",
        "|---|---|---:|---:|",
    ]
    for check in case["contract_evaluation"]["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {check.get('observed')} | {check.get('limit')} |"
        )
    lines += [
        "",
        "The example is descriptive evidence over this fixed public evaluation split. "
        "Software-counterfactual attribution does not establish real-world causality.",
        "",
    ]
    return "\n".join(lines)


def _print(payload: dict) -> None:
    metrics = payload["model_metrics_on_baseline_features"]
    case = payload["decision_change_case_study"]
    print("Public Digits / XGBoost DecisionFlow")
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
    print()
    print(
        f"case study: {case['changed_actions']}/{case['records']} actions changed "
        f"({case['action_shift_rate']:.2%}); contract={case['contract_evaluation']['result']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    payload = run()
    _print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_case_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
