from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from decishift.core.exceptions import ConfigurationError
from decishift.core.pipeline import DecisionPipeline


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("YAML root must be a mapping")
    return data


def load_data(spec: dict[str, Any], *, base_dir: Path) -> pd.DataFrame:
    raw_path = spec.get("path")
    if not raw_path:
        raise ConfigurationError("data.path is required")
    path = (base_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
    if not path.exists():
        raise ConfigurationError(f"Local data file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ConfigurationError("Only local CSV and Parquet data are supported in v0.1")


def _import_object(path: str) -> Any:
    if ":" not in path:
        raise ConfigurationError(f"Import path must use module:object syntax: {path}")
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ConfigurationError(f"Cannot import {path}") from exc


def _component(spec: Any) -> tuple[Any, str | None]:
    if spec is None or spec == "identity":
        return None, "identity"
    if not isinstance(spec, dict):
        return spec, str(spec)
    version = spec.get("version")
    if "object" in spec:
        return _import_object(spec["object"]), version
    if "factory" in spec:
        factory = _import_object(spec["factory"])
        return factory(**(spec.get("kwargs") or {})), version
    raise ConfigurationError("Component mappings require 'object' or 'factory'")


def build_pipeline(spec: dict[str, Any], name: str) -> DecisionPipeline:
    versions: dict[str, str] = {}
    features, version = _component(spec.get("features"))
    if version is not None:
        versions["features"] = version
    model, version = _component(spec.get("model"))
    if version is not None:
        versions["model"] = version
    calibrator, version = _component(spec.get("calibrator"))
    if version is not None:
        versions["calibrator"] = version
    rules, version = _component(spec.get("rules"))
    if version is not None:
        versions["rules"] = version

    threshold_spec = spec.get("threshold", 0.5)
    if isinstance(threshold_spec, dict):
        if "value" not in threshold_spec:
            raise ConfigurationError("threshold mapping requires value")
        threshold = float(threshold_spec["value"])
        versions["threshold"] = str(threshold_spec.get("version", threshold))
    else:
        threshold = float(threshold_spec)
        versions["threshold"] = str(threshold)

    return DecisionPipeline(
        features=features,
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        rules=rules,
        versions=versions,
        name=name,
    )
