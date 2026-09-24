from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LightGBMBoosterAdapter:
    """Adapter for a lightgbm.Booster; LightGBM remains optional."""

    booster: Any
    version: str = "lightgbm"

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.booster.predict(X)).reshape(-1)


def adapt(booster: Any, *, version: str = "lightgbm") -> LightGBMBoosterAdapter:
    return LightGBMBoosterAdapter(booster, version=version)
