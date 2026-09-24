from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SklearnModelAdapter:
    """Normalize a fitted scikit-learn estimator to one score per record."""

    estimator: Any
    version: str = "sklearn"

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            values = np.asarray(self.estimator.predict_proba(X))
            return values[:, 1] if values.ndim == 2 and values.shape[1] > 1 else values.reshape(-1)
        return np.asarray(self.estimator.predict(X)).reshape(-1)


def adapt(estimator: Any, *, version: str = "sklearn") -> SklearnModelAdapter:
    return SklearnModelAdapter(estimator, version=version)
