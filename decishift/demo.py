from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decishift.attribution import calculate_attribution_diagnostics, exact_attribution, pairwise_interactions
from decishift.cohorts import analyze_cohorts
from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import compare_pipelines
from decishift.fragility import analyze_fragility
from decishift.replay.engine import HybridReplayCache


@dataclass
class MaintenanceFeatures:
    version: str
    temperature_weight: float

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        age = frame["equipment_age_years"].to_numpy() / 12.0
        vibration = frame["vibration"].to_numpy()
        temperature = frame["temperature_c"].to_numpy() / 100.0
        utilization = frame["utilization"].to_numpy()
        return np.column_stack([age, vibration, self.temperature_weight * temperature, utilization])


@dataclass
class LinearRiskModel:
    version: str
    weights: np.ndarray
    bias: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class MaintenanceRules:
    version: str
    temperature_cutoff: float

    def apply(self, records: pd.DataFrame, scores: np.ndarray, decisions: np.ndarray) -> np.ndarray:
        out = decisions.copy()
        out[records["temperature_c"].to_numpy() >= self.temperature_cutoff] = 1
        return out


def dataset(n: int = 10_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "equipment_age_years": rng.uniform(0.5, 15, n),
        "vibration": np.clip(rng.normal(0.48, 0.16, n), 0, 1),
        "temperature_c": np.clip(rng.normal(66, 13, n), 20, 120),
        "utilization": rng.uniform(0.1, 1.0, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "equipment_type": rng.choice(["pump", "compressor", "fan"], n, p=[0.45, 0.30, 0.25]),
    })


def pipelines() -> tuple[DecisionPipeline, DecisionPipeline]:
    baseline = DecisionPipeline(
        features=MaintenanceFeatures("features_v1", 0.85),
        model=LinearRiskModel("model_v1", np.array([1.05, 2.15, 1.20, 0.65]), -2.15),
        calibrator=None,
        threshold=0.72,
        rules=MaintenanceRules("rules_v1", 110.0),
        versions={"features": "features_v1", "model": "model_v1", "calibrator": "identity", "threshold": "0.72", "rules": "rules_v1"},
        name="baseline",
    )
    candidate = DecisionPipeline(
        features=MaintenanceFeatures("features_v2", 0.90),
        model=LinearRiskModel("model_v2", np.array([1.04, 2.17, 1.22, 0.66]), -2.16),
        calibrator=None,
        threshold=0.715,
        rules=MaintenanceRules("rules_v2", 100.0),
        versions={"features": "features_v2", "model": "model_v2", "calibrator": "identity", "threshold": "0.715", "rules": "rules_v2"},
        name="candidate",
    )
    return baseline, candidate


def run_demo(n: int = 10_000):
    frame = dataset(n=n)
    baseline, candidate = pipelines()
    result = compare_pipelines(baseline, candidate, frame, id_column="record_id")
    replay = HybridReplayCache(baseline, candidate, frame)
    result.attribution = exact_attribution(baseline, candidate, frame, result=result, cache=replay)
    result.interactions = pairwise_interactions(baseline, candidate, frame, result=result, cache=replay)
    result.metadata.update({
        "hybrid_pipeline_evaluations": replay.evaluations,
        "attribution_method": "exact",
        "attribution_parameters": {"max_components": 10},
        "random_seed": None,
    })
    result.diagnostics = calculate_attribution_diagnostics(
        result, result.attribution, method="exact", hybrid_evaluations=replay.evaluations, tolerance=1e-9
    )
    result.cohorts = analyze_cohorts(
        frame,
        result,
        columns=["equipment_age_years", "vibration", "temperature_c", "utilization", "region", "equipment_type"],
        id_column="record_id",
        min_size=max(30, n // 200),
    )
    analyze_fragility(result)
    return frame, result
