from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf-model"
EXAMPLE = ROOT / "examples" / "public_digits_xgboost" / "run.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("decishift_digits_example", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_card(metrics: dict[str, float], source_commit: str) -> str:
    return f"""---
license: apache-2.0
library_name: xgboost
pipeline_tag: tabular-classification
tags:
- xgboost
- tabular-classification
- behavioral-regression
- model-regression
- decision-systems
- ml-evaluation
- mlops
- decishift
---

# DeciShift Digits XGBoost Baseline & Candidate Models

Deterministic XGBoost checkpoints used by the public **DeciShift** decision-change case study.

This model repository contains the two trained model artifacts that DeciShift compares inside a larger executable decision system. DeciShift itself is a behavioral-regression testing toolkit, not a single pretrained model; these checkpoints are published separately so the benchmark model layer is inspectable and reproducible.

## Files

- `baseline.json` — XGBoost booster trained for **20 boosting rounds** (`digits_xgb_v1`).
- `candidate.json` — XGBoost booster trained for **25 boosting rounds** (`digits_xgb_v2`).
- `metadata.json` — reproducibility metadata, evaluation metrics, source commit, and training configuration summary.

## Task

Binary classification over the public scikit-learn Digits dataset: predict whether a handwritten digit is **8** (`1`) or **not 8** (`0`).

The models operate on 64 pixel-intensity features. DeciShift's case study then combines model scores with versioned policy and rules components to produce final operational actions.

## Evaluation snapshot

Using the deterministic held-out split from the DeciShift public example:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Accuracy | {metrics['baseline_accuracy']:.2%} | {metrics['candidate_accuracy']:.2%} |
| ROC AUC | {metrics['baseline_roc_auc']:.4f} | {metrics['candidate_roc_auc']:.4f} |

These predictive metrics describe only the model layer. The associated DeciShift benchmark demonstrates that improved model metrics can still coincide with changed downstream decisions once policy and rule versions are included.

## Reproduce

Generated from DeciShift source commit `{source_commit}` with:

```bash
python -m pip install -e '.[dev,xgboost]'
python huggingface/build_model.py
```

## Load with XGBoost

```python
import xgboost as xgb

baseline = xgb.Booster()
baseline.load_model("baseline.json")

candidate = xgb.Booster()
candidate.load_model("candidate.json")
```

Inputs must use the same 64-feature ordering as the DeciShift example (`pixel_00` through `pixel_63`) after the example's V1 feature scaling (`value / 16.0`).

## Related artifacts

- Source: https://github.com/sauravsingla/DeciShift
- Hugging Face Space: https://huggingface.co/spaces/sauravsingla08/DeciShift
- Benchmark dataset: https://huggingface.co/datasets/sauravsingla08/DeciShift-Decision-Change-Benchmark
- PyPI: https://pypi.org/project/decishift/

## Intended use

These checkpoints are intended for reproducibility, testing, education, and inspection of DeciShift's public decision-change benchmark. They are not presented as general-purpose digit-recognition models or as production recommendations.

## Limitations

The benchmark uses a fixed public dataset split and a deliberately compact binary task. Results do not establish production safety, fairness, compliance, causal attribution, or fitness for deployment. Downstream decision behavior depends on the full executable system, including policy and rules components, not only these model files.

## License

The generated DeciShift model artifacts are published under Apache-2.0. Upstream source-data terms for the original scikit-learn Digits dataset remain applicable to the source data.
"""


def main() -> None:
    module = _load_example_module()
    records, target = module._records()

    baseline = module.DigitXGBoostModel(20, "digits_xgb_v1")
    candidate = module.DigitXGBoostModel(25, "digits_xgb_v2")
    metrics = module._model_metrics(records, target)

    source_commit = os.environ.get("GITHUB_SHA", "local")

    OUT.mkdir(parents=True, exist_ok=True)
    baseline.adapter.booster.save_model(OUT / "baseline.json")
    candidate.adapter.booster.save_model(OUT / "candidate.json")

    metadata = {
        "source_repository": "https://github.com/sauravsingla/DeciShift",
        "source_commit": source_commit,
        "task": "binary classification: handwritten digit 8 vs not 8",
        "dataset": "scikit-learn Digits",
        "feature_count": 64,
        "feature_order": [f"pixel_{i:02d}" for i in range(64)],
        "feature_scaling": "value / 16.0",
        "split": {
            "test_size": 0.40,
            "random_state": 23,
            "stratify": "digit_is_8",
        },
        "common_xgboost_params": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,
            "eta": 0.08,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "seed": 23,
            "nthread": 1,
        },
        "models": {
            "baseline": {"file": "baseline.json", "version": "digits_xgb_v1", "boosting_rounds": 20},
            "candidate": {"file": "candidate.json", "version": "digits_xgb_v2", "boosting_rounds": 25},
        },
        "evaluation_records": int(len(records)),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        _model_card(metrics, source_commit), encoding="utf-8"
    )

    print(f"Wrote Hugging Face model package to {OUT}")
    print(f"Evaluation records: {len(records)}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
