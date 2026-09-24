from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class FeatureTransformer(Protocol):
    def transform(self, X: pd.DataFrame) -> Any: ...


@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: Any) -> Any: ...


@runtime_checkable
class ProbabilityPredictor(Protocol):
    def predict_proba(self, X: Any) -> Any: ...


@runtime_checkable
class Calibrator(Protocol):
    def transform(self, scores: np.ndarray) -> Any: ...


@runtime_checkable
class RuleSet(Protocol):
    def apply(
        self,
        records: pd.DataFrame,
        scores: np.ndarray,
        decisions: np.ndarray,
    ) -> Any: ...
