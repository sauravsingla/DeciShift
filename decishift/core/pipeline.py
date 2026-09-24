from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

COMPONENTS = ("features", "model", "calibrator", "threshold", "rules")


def _as_1d(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must return one value per record; got shape {arr.shape}")
    return arr


def _transform_features(component: Any, records: pd.DataFrame) -> Any:
    if component is None:
        return records
    if hasattr(component, "transform"):
        return component.transform(records)
    if callable(component):
        return component(records)
    raise TypeError("features component must be None, callable, or expose transform(X)")


def _predict_scores(model: Any, features: Any, n: int) -> np.ndarray:
    if model is None:
        raise ValueError("A model component is required for executable replay")
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features))
        if raw.ndim == 2:
            if raw.shape[1] < 2:
                raw = raw[:, 0]
            else:
                raw = raw[:, 1]
        return _as_1d(raw, n, "model.predict_proba").astype(float)
    if hasattr(model, "predict"):
        return _as_1d(model.predict(features), n, "model.predict").astype(float)
    if callable(model):
        return _as_1d(model(features), n, "model callable").astype(float)
    raise TypeError("model must be callable or expose predict/predict_proba")


def _calibrate(component: Any, raw_scores: np.ndarray, n: int) -> np.ndarray:
    if component is None:
        return raw_scores.astype(float, copy=True)
    if hasattr(component, "transform"):
        values = component.transform(raw_scores)
    elif hasattr(component, "predict"):
        values = component.predict(raw_scores)
    elif callable(component):
        values = component(raw_scores)
    else:
        raise TypeError("calibrator must be None, callable, or expose transform/predict")
    return _as_1d(values, n, "calibrator").astype(float)


def _threshold_values(threshold: float | Callable[..., Any], records: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    n = len(records)
    if callable(threshold):
        try:
            values = threshold(records, scores)
        except TypeError:
            values = threshold(records)
        return _as_1d(values, n, "threshold callable").astype(float)
    return np.full(n, float(threshold), dtype=float)


def _apply_rules(component: Any, records: pd.DataFrame, scores: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    n = len(records)
    if component is None:
        return decisions.astype(int, copy=True)
    if hasattr(component, "apply"):
        values = component.apply(records, scores, decisions.copy())
    elif callable(component):
        values = component(records, scores, decisions.copy())
    else:
        raise TypeError("rules must be None, callable, or expose apply(records, scores, decisions)")
    return _as_1d(values, n, "rules").astype(int)


@dataclass(frozen=True)
class PipelineTrace:
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray
    thresholds: np.ndarray
    pre_rule_decisions: np.ndarray
    decisions: np.ndarray

    @property
    def margins(self) -> np.ndarray:
        return self.calibrated_scores - self.thresholds


@dataclass(frozen=True)
class DecisionPipeline:
    """An ordered, versioned decision pipeline.

    Components are intentionally lightweight Python objects so the core remains
    framework-agnostic and fully local/offline.
    """

    features: Any = None
    model: Any = None
    calibrator: Any = None
    threshold: float | Callable[..., Any] = 0.5
    rules: Any = None
    versions: Mapping[str, str] = field(default_factory=dict)
    name: str = "pipeline"

    def evaluate(self, records: pd.DataFrame, *, cache: dict[tuple[str, ...], Any] | None = None) -> PipelineTrace:
        """Evaluate the ordered pipeline, optionally reusing version-keyed intermediates.

        A cache is safe only for evaluations over the same immutable input records.
        `HybridReplayCache` owns such a cache for one comparison and uses dependency
        keys so changing a component invalidates that component and its descendants.
        """
        n = len(records)
        local_cache = cache if cache is not None else {}
        fv = self.version_of("features")
        mv = self.version_of("model")
        cv = self.version_of("calibrator")
        tv = self.version_of("threshold")
        rv = self.version_of("rules")

        def cached(key: tuple[str, ...], compute):
            if key not in local_cache:
                local_cache[key] = compute()
            return local_cache[key]

        transformed = cached(("features", fv), lambda: _transform_features(self.features, records))
        raw_scores = cached(("model", fv, mv), lambda: _predict_scores(self.model, transformed, n))
        calibrated = cached(("calibrator", fv, mv, cv), lambda: _calibrate(self.calibrator, raw_scores, n))
        thresholds = cached(("threshold", fv, mv, cv, tv), lambda: _threshold_values(self.threshold, records, calibrated))
        pre_rule = cached(("pre_rule", fv, mv, cv, tv), lambda: (calibrated >= thresholds).astype(int))
        final = cached(("rules", fv, mv, cv, tv, rv), lambda: _apply_rules(self.rules, records, calibrated, pre_rule))
        if not np.isin(final, [0, 1]).all():
            raise ValueError("rules must produce binary 0/1 decisions")
        return PipelineTrace(raw_scores, calibrated, thresholds, pre_rule, final)

    def version_of(self, component: str) -> str:
        if component not in COMPONENTS:
            raise KeyError(component)
        if component in self.versions:
            return str(self.versions[component])
        value = getattr(self, component)
        if component == "threshold" and not callable(value):
            return f"threshold:{float(value):.12g}"
        if value is None:
            return "identity"
        explicit = getattr(value, "version", None)
        if explicit is not None:
            return str(explicit)
        return f"object:{type(value).__module__}.{type(value).__qualname__}:{id(value)}"

    def changed_components(self, other: "DecisionPipeline") -> list[str]:
        return [name for name in COMPONENTS if self.version_of(name) != other.version_of(name)]

    def hybrid(self, candidate: "DecisionPipeline", candidate_components: set[str] | frozenset[str]) -> "DecisionPipeline":
        changes: dict[str, Any] = {}
        merged_versions = dict(self.versions)
        for component in COMPONENTS:
            if component in candidate_components:
                changes[component] = getattr(candidate, component)
                merged_versions[component] = candidate.version_of(component)
            else:
                merged_versions[component] = self.version_of(component)
        changes["versions"] = merged_versions
        changes["name"] = f"hybrid[{','.join(sorted(candidate_components))}]"
        return replace(self, **changes)
