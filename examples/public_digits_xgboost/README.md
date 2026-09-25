# Public Digits / XGBoost example

This example uses scikit-learn's bundled **Digits classification dataset**: 1,797 public 8×8 digit images with 64 integer-valued pixel features. The model target is `digit 8` versus all other digits. The model score is converted into three actions: `auto_not_8`, `manual_review`, and `auto_8`.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

The XGBoost boosters are genuinely trained at runtime on a deterministic split. No network dataset download is required.

Run:

```bash
python -m pip install -e '.[dev,xgboost]'
python examples/public_digits_xgboost/run.py \
  --json-out evaluation-artifacts/digits-xgboost.json \
  --markdown-out evaluation-artifacts/decision-change-case-study.md
```

The script separately changes features, model, policy and rules, then runs an all-change scenario. It also emits a focused case study where model metrics improve while aggregate action counts remain close, yet record-level actions move. DeciShift traces the transitions, attributes them across versioned nodes, and evaluates a Decision Contract with a cohort override.

This is an OCR-routing demonstration, not a safety-critical deployment recommendation.
