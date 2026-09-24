from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline


@dataclass
class ShiftFeatures:
    shift: float
    version: str

    def transform(self, frame: pd.DataFrame):
        return frame[["x"]].to_numpy() + self.shift


@dataclass
class SimpleModel:
    shift: float
    version: str

    def predict(self, X):
        return np.asarray(X)[:, 0] + self.shift


@dataclass
class ClampCalibrator:
    scale: float
    version: str

    def transform(self, scores):
        return np.clip(np.asarray(scores) * self.scale, 0, 1)


@dataclass
class FlagRules:
    enabled: bool
    version: str

    def apply(self, records, scores, decisions):
        out = decisions.copy()
        if self.enabled:
            out[records["force"].to_numpy(dtype=bool)] = 1
        return out


def frame(values=(0.4, 0.6, 0.2, 0.8)):
    n = len(values)
    force = [False] * n
    if n >= 3:
        force[2] = True
    group = (["a", "a", "b", "b"] * ((n + 3) // 4))[:n]
    return pd.DataFrame({"id": range(n), "x": list(values), "force": force, "group": group})


def pipeline(*, feature_shift=0.0, model_shift=0.0, scale=1.0, threshold=0.5, rules=False, labels=None):
    labels = labels or {}
    return DecisionPipeline(
        features=ShiftFeatures(feature_shift, labels.get("features", f"f{feature_shift}")),
        model=SimpleModel(model_shift, labels.get("model", f"m{model_shift}")),
        calibrator=ClampCalibrator(scale, labels.get("calibrator", f"c{scale}")),
        threshold=threshold,
        rules=FlagRules(rules, labels.get("rules", f"r{rules}")),
        versions={
            "features": labels.get("features", f"f{feature_shift}"),
            "model": labels.get("model", f"m{model_shift}"),
            "calibrator": labels.get("calibrator", f"c{scale}"),
            "threshold": labels.get("threshold", f"t{threshold}"),
            "rules": labels.get("rules", f"r{rules}"),
        },
    )
