from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from decishift.core.exceptions import ComponentExecutionError, ReproducibilityError
from decishift.core.identity import ComponentIdentity, component_identity, reproducibility_status

COMPONENTS = ("features", "model", "calibrator", "threshold", "rules")


def _as_1d(values: Any, n: int, name: str, *, finite: bool = False) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must return one value per record; got shape {arr.shape}")
    if finite:
        numeric = arr.astype(float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} produced NaN or infinite values")
    return arr


def _invoke_with_signature(fn: Callable[..., Any], label: str, variants: list[tuple[Any, ...]]) -> Any:
    """Choose a supported calling convention without swallowing internal TypeError."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # If introspection is unavailable, use the primary documented convention.
        args = variants[0]
    else:
        args = None
        for candidate in variants:
            try:
                signature.bind(*candidate)
            except TypeError:
                continue
            args = candidate
            break
        if args is None:
            expected = " or ".join(str(len(v)) for v in variants)
            raise TypeError(f"{label} does not accept a supported positional arity ({expected})")
    try:
        return fn(*args)
    except Exception as exc:
        raise ComponentExecutionError(f"{label} failed: {type(exc).__name__}: {exc}") from exc


def _validate_feature_alignment(output: Any, records: pd.DataFrame) -> Any:
    try:
        n = len(output)
    except TypeError as exc:
        raise ValueError("features must return a row-aligned object with one row per input record") from exc
    if n != len(records):
        raise ValueError(f"features changed row count from {len(records)} to {n}")
    if isinstance(output, (pd.DataFrame, pd.Series)) and not output.index.equals(records.index):
        raise ValueError("features returned a pandas object with reordered or changed index; record identity is ambiguous")
    return output


def _transform_features(component: Any, records: pd.DataFrame) -> Any:
    if component is None:
        return records
    if hasattr(component, "transform"):
        values = _invoke_with_signature(component.transform, f"Feature component {getattr(component, 'version', type(component).__name__)}", [(records,)])
    elif callable(component):
        values = _invoke_with_signature(component, f"Feature component {getattr(component, 'version', type(component).__name__)}", [(records,)])
    else:
        raise TypeError("features component must be None, callable, or expose transform(X)")
    return _validate_feature_alignment(values, records)


def _predict_scores(model: Any, features: Any, n: int) -> np.ndarray:
    if model is None:
        raise ValueError("A model component is required for executable replay")
    label = f"Model component {getattr(model, 'version', type(model).__name__)}"
    if hasattr(model, "predict_proba"):
        raw = np.asarray(_invoke_with_signature(model.predict_proba, label, [(features,)]))
        if raw.ndim == 2:
            if raw.shape[1] < 2:
                raw = raw[:, 0]
            else:
                raw = raw[:, 1]
        return _as_1d(raw, n, "model.predict_proba", finite=True).astype(float)
    if hasattr(model, "predict"):
        return _as_1d(_invoke_with_signature(model.predict, label, [(features,)]), n, "model.predict", finite=True).astype(float)
    if callable(model):
        return _as_1d(_invoke_with_signature(model, label, [(features,)]), n, "model callable", finite=True).astype(float)
    raise TypeError("model must be callable or expose predict/predict_proba")


def _calibrate(component: Any, raw_scores: np.ndarray, n: int) -> np.ndarray:
    if component is None:
        return raw_scores.astype(float, copy=True)
    label = f"Calibration component {getattr(component, 'version', type(component).__name__)}"
    if hasattr(component, "transform"):
        values = _invoke_with_signature(component.transform, label, [(raw_scores,)])
    elif hasattr(component, "predict"):
        values = _invoke_with_signature(component.predict, label, [(raw_scores,)])
    elif callable(component):
        values = _invoke_with_signature(component, label, [(raw_scores,)])
    else:
        raise TypeError("calibrator must be None, callable, or expose transform/predict")
    return _as_1d(values, n, "calibrator", finite=True).astype(float)


def _threshold_values(threshold: float | Callable[..., Any], records: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    n = len(records)
    if callable(threshold):
        label = f"Threshold component {getattr(threshold, 'version', getattr(threshold, '__name__', type(threshold).__name__))}"
        values = _invoke_with_signature(threshold, label, [(records, scores), (records,)])
        return _as_1d(values, n, "threshold callable", finite=True).astype(float)
    values = np.full(n, float(threshold), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("threshold produced NaN or infinite values")
    return values


def _apply_rules(component: Any, records: pd.DataFrame, scores: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    n = len(records)
    if component is None:
        return decisions.astype(int, copy=True)
    label = f"Rules component {getattr(component, 'version', type(component).__name__)}"
    if hasattr(component, "apply"):
        values = _invoke_with_signature(component.apply, label, [(records, scores, decisions.copy())])
    elif callable(component):
        values = _invoke_with_signature(component, label, [(records, scores, decisions.copy())])
    else:
        raise TypeError("rules must be None, callable, or expose apply(records, scores, decisions)")
    arr = _as_1d(values, n, "rules")
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError("rules must produce binary 0/1 decisions")
    return arr.astype(int)


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

    Components stay lightweight Python objects so the core remains framework-
    agnostic and local. Explicit provenance is strongly preferred; strict mode
    rejects components whose identity cannot be reproduced across processes.
    """

    features: Any = None
    model: Any = None
    calibrator: Any = None
    threshold: float | Callable[..., Any] = 0.5
    rules: Any = None
    versions: Mapping[str, Any] = field(default_factory=dict)
    identities: Mapping[str, ComponentIdentity | Mapping[str, Any]] = field(default_factory=dict)
    artifact_paths: Mapping[str, str | Path] = field(default_factory=dict)
    name: str = "pipeline"
    strict_reproducibility: bool = False

    def component_identity(self, component: str) -> ComponentIdentity:
        if component not in COMPONENTS:
            raise KeyError(component)
        explicit_identity = self.identities.get(component)
        explicit_version = self.versions.get(component)
        explicit = explicit_identity if explicit_identity is not None else explicit_version
        return component_identity(
            component,
            getattr(self, component),
            explicit=explicit,
            artifact_path=self.artifact_paths.get(component),
        )

    def component_identities(self) -> dict[str, ComponentIdentity]:
        return {name: self.component_identity(name) for name in COMPONENTS}

    @property
    def reproducibility_status(self) -> str:
        return reproducibility_status(self.component_identities())

    def validate_reproducibility(self) -> None:
        if self.strict_reproducibility:
            unstable = [name for name, ident in self.component_identities().items() if not ident.reproducible]
            if unstable:
                raise ReproducibilityError(
                    f"{self.name} has unstable component identity: {', '.join(unstable)}. "
                    "Configure an explicit version/digest or artifact path."
                )

    def evaluate(self, records: pd.DataFrame, *, cache: dict[tuple[str, ...], Any] | None = None) -> PipelineTrace:
        self.validate_reproducibility()
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
        return PipelineTrace(raw_scores, calibrated, thresholds, pre_rule, final)

    def version_of(self, component: str) -> str:
        ident = self.component_identity(component)
        if ident.digest and ident.version:
            return f"{ident.version}|{ident.digest}"
        if ident.digest:
            return ident.digest
        if ident.version:
            return ident.version
        return f"unstable:{component}"

    def changed_components(self, other: "DecisionPipeline") -> list[str]:
        return [name for name in COMPONENTS if self.version_of(name) != other.version_of(name)]

    def hybrid(self, candidate: "DecisionPipeline", candidate_components: set[str] | frozenset[str]) -> "DecisionPipeline":
        changes: dict[str, Any] = {}
        merged_versions: dict[str, Any] = {}
        merged_identities: dict[str, Any] = {}
        merged_artifacts: dict[str, Any] = {}
        for component in COMPONENTS:
            source = candidate if component in candidate_components else self
            changes[component] = getattr(source, component)
            ident = source.component_identity(component)
            merged_identities[component] = ident
            if component in source.versions:
                merged_versions[component] = source.versions[component]
            if component in source.artifact_paths:
                merged_artifacts[component] = source.artifact_paths[component]
        changes["versions"] = merged_versions
        changes["identities"] = merged_identities
        changes["artifact_paths"] = merged_artifacts
        changes["name"] = f"hybrid[{','.join(sorted(candidate_components))}]"
        changes["strict_reproducibility"] = self.strict_reproducibility or candidate.strict_reproducibility
        return replace(self, **changes)
