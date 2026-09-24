from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class XGBoostBoosterAdapter:
    """Adapter for an xgboost.Booster without making xgboost a base dependency."""

    booster: Any
    version: str = "xgboost"

    def predict(self, X: Any) -> np.ndarray:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("Install DeciShift with the 'xgboost' extra") from exc
        matrix = X if isinstance(X, xgb.DMatrix) else xgb.DMatrix(X)
        return np.asarray(self.booster.predict(matrix)).reshape(-1)


def adapt(booster: Any, *, version: str = "xgboost") -> XGBoostBoosterAdapter:
    return XGBoostBoosterAdapter(booster, version=version)
