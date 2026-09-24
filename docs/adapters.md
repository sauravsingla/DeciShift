# Model adapters

DeciShift's core accepts any object exposing `predict(X)` or `predict_proba(X)`, so many fitted estimators can be passed directly. Lightweight adapters make common cases explicit.

## scikit-learn

```python
from decishift.adapters.sklearn import adapt
model = adapt(fitted_estimator, version="model_v12")
```

## XGBoost

The optional adapter converts array-like input to `xgboost.DMatrix` for a native `Booster`.

```python
from decishift.adapters.xgboost import adapt
model = adapt(fitted_booster, version="model_v12")
```

Install the optional dependency with `pip install -e '.[xgboost]'`.

## LightGBM

```python
from decishift.adapters.lightgbm import adapt
model = adapt(fitted_booster, version="model_v12")
```

Install with `pip install -e '.[lightgbm]'`.
