# Public Wine / scikit-learn example

This example uses scikit-learn's bundled **Wine classification dataset**: 178 public samples, 13 numeric features and 3 classes. The example turns it into a binary model target (`class_2` versus the other classes) and then maps the score into three operational routing actions: `route_not_class2`, `manual_review`, and `route_class2`.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

The model is genuinely fitted at runtime with scikit-learn on a deterministic training split. No network download is required.

Run:

```bash
python -m pip install -e '.[dev]'
python examples/public_wine_sklearn/run.py --json-out evaluation-artifacts/wine-sklearn.json
```

The script compares the same baseline against five candidate scenarios:

- feature transform only;
- fitted model only;
- policy thresholds only;
- review rule only;
- all four together.

Each comparison uses the same held-out public records and reports final-action changes plus exact software-counterfactual attribution. The example is for behavior/evaluation mechanics, not wine-quality advice.
