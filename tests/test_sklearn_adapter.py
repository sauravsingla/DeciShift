from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from decishift.adapters.sklearn import adapt


def test_sklearn_classifier_adapter_uses_positive_class_probability() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    estimator = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)

    adapter = adapt(estimator, version="logreg-v1")
    expected = estimator.predict_proba(X)[:, 1]

    np.testing.assert_allclose(adapter.predict(X), expected)
    assert adapter.version == "logreg-v1"


def test_sklearn_regressor_adapter_falls_back_to_predict() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    estimator = LinearRegression().fit(X, y)

    adapter = adapt(estimator, version="linear-v2")

    np.testing.assert_allclose(adapter.predict(X), estimator.predict(X))
    assert adapter.predict(X).shape == (len(X),)
    assert adapter.version == "linear-v2"
