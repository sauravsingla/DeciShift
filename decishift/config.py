from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from decishift.core.exceptions import ConfigurationError
from decishift.core.identity import ComponentIdentity, sha256_file
from decishift.core.pipeline import DecisionPipeline
from decishift.flow.flow import DecisionFlow
from decishift.flow.node import DecisionNode


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("YAML root must be a mapping")
    return data


def configuration_sha256(path: str | Path) -> str:
    return sha256_file(path)


def resolve_data_path(spec: dict[str, Any], *, base_dir: Path) -> Path:
    raw_path = spec.get("path")
    if not raw_path:
        raise ConfigurationError("data.path is required")
    path = (base_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    if not path.exists() or not path.is_file():
        raise ConfigurationError(f"Local data file does not exist: {path}")
    return path


def load_data(spec: dict[str, Any], *, base_dir: Path) -> pd.DataFrame:
    path = resolve_data_path(spec, base_dir=base_dir)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ConfigurationError("Only local CSV and Parquet data are supported")


def _import_object(path: str) -> Any:
    if ":" not in path:
        raise ConfigurationError(f"Import path must use module:object syntax: {path}")
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ConfigurationError(f"Cannot import {path}") from exc


def _identity_from_spec(name: str, spec: dict[str, Any], *, base_dir: Path | None = None) -> tuple[ComponentIdentity | None, str | None]:
    version = spec.get("version")
    digest = spec.get("digest")
    artifact = spec.get("artifact") or spec.get("artifact_path")
    artifact_path: str | None = None
    if artifact:
        path = Path(str(artifact))
        if base_dir is not None and not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists() or not path.is_file():
            raise ConfigurationError(f"Artifact path for {name} does not exist: {path}")
        artifact_path = str(path)
        if digest is None:
            digest = sha256_file(path)
    if digest is not None and not str(digest).startswith("sha256:"):
        digest = "sha256:" + str(digest)
    if version is not None or digest is not None:
        source = "artifact" if artifact_path else "explicit"
        return ComponentIdentity(name, None if version is None else str(version), None if digest is None else str(digest), source, True), artifact_path
    return None, artifact_path


def _component(name: str, spec: Any, *, base_dir: Path | None = None) -> tuple[Any, str | None, ComponentIdentity | None, str | None]:
    if spec is None or spec == "identity":
        return None, "identity", ComponentIdentity(name, "identity", None, "explicit", True), None
    if not isinstance(spec, dict):
        return spec, str(spec), ComponentIdentity(name, str(spec), None, "explicit", True), None
    version = spec.get("version")
    identity, artifact_path = _identity_from_spec(name, spec, base_dir=base_dir)
    if "object" in spec:
        return _import_object(spec["object"]), version, identity, artifact_path
    if "factory" in spec:
        factory = _import_object(spec["factory"])
        try:
            value = factory(**(spec.get("kwargs") or {}))
        except Exception as exc:
            raise ConfigurationError(f"Factory {spec['factory']} failed: {type(exc).__name__}: {exc}") from exc
        return value, version, identity, artifact_path
    if "value" in spec and name == "threshold":
        value = float(spec["value"])
        return value, version or str(value), identity, artifact_path
    raise ConfigurationError("Component mappings require 'object' or 'factory' (threshold also supports 'value')")


def build_pipeline(spec: dict[str, Any], name: str, *, base_dir: Path | None = None) -> DecisionPipeline:
    versions: dict[str, str] = {}
    identities: dict[str, ComponentIdentity] = {}
    artifact_paths: dict[str, str] = {}
    values: dict[str, Any] = {}

    for component_name in ("features", "model", "calibrator", "rules"):
        value, version, identity, artifact_path = _component(component_name, spec.get(component_name), base_dir=base_dir)
        values[component_name] = value
        if version is not None:
            versions[component_name] = str(version)
        if identity is not None:
            identities[component_name] = identity
        if artifact_path:
            artifact_paths[component_name] = artifact_path

    threshold_spec = spec.get("threshold", 0.5)
    threshold, version, identity, artifact_path = _component("threshold", threshold_spec, base_dir=base_dir)
    values["threshold"] = threshold
    if version is not None:
        versions["threshold"] = str(version)
    if identity is not None:
        identities["threshold"] = identity
    if artifact_path:
        artifact_paths["threshold"] = artifact_path

    strict = bool(spec.get("strict_reproducibility", False))
    return DecisionPipeline(
        features=values["features"],
        model=values["model"],
        calibrator=values["calibrator"],
        threshold=values["threshold"],
        rules=values["rules"],
        versions=versions,
        identities=identities,
        artifact_paths=artifact_paths,
        name=name,
        strict_reproducibility=strict,
    )


def build_flow(spec: dict[str, Any], name: str, *, base_dir: Path | None = None) -> DecisionFlow:
    """Build a local DecisionFlow from a YAML-compatible mapping."""
    nodes_spec = spec.get("nodes")
    final_node = spec.get("final_node")
    if not isinstance(nodes_spec, dict) or not nodes_spec:
        raise ConfigurationError(f"{name}.nodes must be a non-empty mapping")
    if not isinstance(final_node, str) or not final_node:
        raise ConfigurationError(f"{name}.final_node is required")

    nodes: list[DecisionNode] = []
    for node_name, node_spec in nodes_spec.items():
        if not isinstance(node_spec, dict):
            raise ConfigurationError(f"Flow node '{node_name}' must be a mapping")
        component, version, identity, artifact_path = _component(node_name, node_spec, base_dir=base_dir)
        depends_on = node_spec.get("depends_on") or []
        if not isinstance(depends_on, (list, tuple)):
            raise ConfigurationError(f"Flow node '{node_name}'.depends_on must be a list")
        nodes.append(DecisionNode(
            name=str(node_name),
            component=component,
            depends_on=tuple(str(dep) for dep in depends_on),
            version=None if version is None else str(version),
            group=None if node_spec.get("group") is None else str(node_spec.get("group")),
            identity=identity,
            artifact_path=artifact_path,
        ))

    return DecisionFlow(
        nodes=tuple(nodes),
        final_node=final_node,
        name=name,
        strict_reproducibility=bool(spec.get("strict_reproducibility", False)),
    )
