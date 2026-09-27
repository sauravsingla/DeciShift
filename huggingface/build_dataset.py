from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from decishift.contracts import evaluate_contract
from decishift.flow.analysis import analyze_flow_cohorts


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf-dataset"
EXAMPLE = ROOT / "examples" / "public_digits_xgboost" / "run.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("decishift_digits_example", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract() -> dict:
    return {
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


def _dataset_card(summary: dict, source_commit: str) -> str:
    case = summary["decision_change_case_study"]
    metrics = case["model_metrics"]
    contract_result = case["contract_evaluation"]["result"]
    return f"""---
license: apache-2.0
pretty_name: DeciShift Decision-Change Benchmark
size_categories:
- n<1K
tags:
- tabular
- machine-learning
- ml-evaluation
- behavioral-regression
- model-regression
- decision-systems
- decision-shift
- software-testing
- mlops
- responsible-ai
configs:
- config_name: default
  data_files:
  - split: train
    path: data/digits_xgboost.parquet
---

# DeciShift Decision-Change Benchmark

**Record-level evidence for analyzing how final ML-system decisions change between baseline and candidate software versions.**

This dataset is generated reproducibly from the public DeciShift repository. The first configuration uses scikit-learn's bundled Digits dataset with trained XGBoost models and versioned policy/rules components. It is designed to make **decision-level behavioral regression** inspectable rather than to provide another copy of the underlying image dataset.

## Why this dataset exists

Model-level metrics can improve while operational decisions still move. In the committed Digits/XGBoost case study:

- evaluation records: **{case['records']}**
- baseline accuracy: **{metrics['baseline_accuracy']:.2%}**
- candidate accuracy: **{metrics['candidate_accuracy']:.2%}**
- baseline ROC AUC: **{metrics['baseline_roc_auc']:.4f}**
- candidate ROC AUC: **{metrics['candidate_roc_auc']:.4f}**
- changed final actions: **{case['changed_actions']} ({case['action_shift_rate']:.2%})**
- Decision Contract result: **{contract_result}**

The global shift can stay within its declared limit while a governed cohort exceeds its own limit. That is the kind of regression DeciShift is built to surface.

## Files

- `data/digits_xgboost.parquet` — canonical 719-record dataset split with baseline/candidate actions, transition labels, model scores, cohort context, contract context, and attribution-share context.
- `analysis/digits_xgboost_changed.parquet` — convenience artifact containing changed-decision records only; it is intentionally excluded from the canonical dataset split to avoid duplicate rows in the Dataset Viewer.
- `summaries/digits_xgboost.json` — machine-readable aggregate case-study evidence.
- `summaries/scenarios.jsonl` — feature/model/policy/rules/all-change scenario summaries.

## Core columns

| Column | Meaning |
|---|---|
| `record_id` | Stable evaluation-record identifier for this deterministic split |
| `actual_digit` | Ground-truth digit label from the public Digits dataset |
| `baseline_score` / `candidate_score` | Baseline/candidate model score when emitted by the flow |
| `baseline_action` / `candidate_action` | Final operational action from each executable system version |
| `changed` | Whether the final action changed |
| `transition` | Explicit action transition, or `unchanged` |
| `cohort` | Governed cohort label used for slice analysis |
| `cohort_shift_rate` | Observed action-shift rate for that cohort |
| `cohort_contract_limit` | Declared cohort limit used by this example |
| `cohort_contract_violation` | Whether the observed cohort rate exceeds its declared limit |
| `global_contract_result` | PASS/BLOCK-style result produced by the Decision Contract evaluation |
| `changed_nodes` | Versioned nodes that differ between baseline and candidate |
| `attribution_share_*` | Global absolute software-counterfactual attribution share for changed nodes |
| `source_commit` | Git commit that generated the artifact |

## Reproducibility

Generated from DeciShift source commit `{source_commit}` by:

```bash
python huggingface/build_dataset.py
```

Source repository: https://github.com/sauravsingla/DeciShift

PyPI: https://pypi.org/project/decishift/

## Important limits

This is descriptive evaluation evidence over a fixed public split. Software-counterfactual attribution does **not** establish real-world causality. A Decision Contract pass or block only reflects user-declared limits over the supplied evidence; it is not proof of safety, fairness, compliance, correctness, or production fitness.

The repository does not republish the raw 8×8 pixel arrays. Source dataset provenance is retained through the upstream scikit-learn Digits reference; upstream dataset terms continue to apply to source data.

## License

DeciShift-generated code and dataset artifacts in this repository are published under Apache-2.0. Upstream source-data terms remain applicable to the original source dataset.
"""


def main() -> None:
    module = _load_example_module()
    records, target = module._records()
    metrics = module._model_metrics(records, target)

    baseline = module._flow(
        features="v1", model="v1", policy="v1", rules="v1", name="digits-case-baseline"
    )
    candidate = module._flow(
        features="v1", model="v2", policy="v2", rules="v2", name="digits-case-candidate"
    )
    result, _ = module._attributed_comparison(records, baseline, candidate)
    result.cohorts = analyze_flow_cohorts(
        records,
        result,
        columns=["actual_digit"],
        id_column="record_id",
        min_size=20,
        requested_transitions=["auto_not_8->manual_review", "manual_review->auto_8"],
    )
    evaluation = evaluate_contract(result, _contract(), integrity_passed=True)
    evaluation_dict = evaluation.as_dict()
    attribution_shares = module._attribution_shares(result.attribution)

    rows = result.records.copy()
    context = records[["record_id", "actual_digit", "actual_is_8"]].copy()
    context["actual_digit"] = context["actual_digit"].astype(str)
    rows = rows.merge(context, on="record_id", how="left", validate="one_to_one")

    rows.insert(0, "scenario", "digits_xgboost")
    rows["cohort"] = "actual_digit=" + rows["actual_digit"].astype(str)

    cohort_rates = {
        str(row["cohort"]): float(row["action_shift_rate"])
        for row in result.cohorts.to_dict(orient="records")
        if row.get("dimension") == "actual_digit"
    }
    rows["cohort_shift_rate"] = rows["actual_digit"].map(cohort_rates)
    rows["cohort_contract_limit"] = rows["actual_digit"].map(
        lambda value: 0.08 if str(value) == "6" else 0.10
    )
    rows["cohort_contract_violation"] = (
        rows["cohort_shift_rate"].notna()
        & (rows["cohort_shift_rate"] > rows["cohort_contract_limit"])
    )
    rows["global_contract_result"] = evaluation_dict["result"]
    rows["changed_nodes"] = "|".join(result.changed_nodes)
    for player, share in sorted(attribution_shares.items()):
        rows[f"attribution_share_{player}"] = float(share)

    source_commit = os.environ.get("GITHUB_SHA", "local")
    rows["source_commit"] = source_commit

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "analysis").mkdir(exist_ok=True)
    (OUT / "summaries").mkdir(exist_ok=True)

    rows.to_parquet(OUT / "data" / "digits_xgboost.parquet", index=False)
    rows.loc[rows["changed"]].to_parquet(
        OUT / "analysis" / "digits_xgboost_changed.parquet", index=False
    )

    full_summary = module.run()
    full_summary["decision_change_case_study"]["contract_evaluation"] = evaluation_dict
    (OUT / "summaries" / "digits_xgboost.json").write_text(
        json.dumps(full_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "summaries" / "scenarios.jsonl").open("w", encoding="utf-8") as handle:
        for scenario in full_summary["scenarios"]:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")

    (OUT / "README.md").write_text(
        _dataset_card(full_summary, source_commit), encoding="utf-8"
    )

    print(f"Wrote {len(rows)} canonical records to {OUT}")
    print(f"Changed decisions: {int(rows['changed'].sum())}")
    print(f"Contract result: {evaluation_dict['result']}")


if __name__ == "__main__":
    main()
