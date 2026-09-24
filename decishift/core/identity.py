from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentIdentity:
    name: str
    version: str | None = None
    digest: str | None = None
    source: str = "unstable"
    reproducible: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source": self.source,
            "reproducible": self.reproducible,
        }

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "ComponentIdentity":
        return cls(
            name=name,
            version=None if value.get("version") is None else str(value.get("version")),
            digest=value.get("digest"),
            source=str(value.get("source", "explicit")),
            reproducible=bool(value.get("reproducible", True)),
        )


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _source_digest(value: Any) -> str | None:
    target = value
    if not isinstance(value, (FunctionType, type)) and callable(value) and inspect.isfunction(getattr(value, "__call__", None)):
        target = value.__call__
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    qualname = f"{getattr(target, '__module__', '')}:{getattr(target, '__qualname__', type(target).__qualname__)}"
    return sha256_bytes((qualname + "\n" + source).encode("utf-8"))


def component_identity(
    name: str,
    value: Any,
    *,
    explicit: str | Mapping[str, Any] | ComponentIdentity | None = None,
    artifact_path: str | Path | None = None,
) -> ComponentIdentity:
    """Resolve component provenance without serializing arbitrary user objects.

    Priority: explicit identity/version, artifact SHA-256, component `.version`,
    deterministic source identity for stateless callables, otherwise unstable.
    """
    if isinstance(explicit, ComponentIdentity):
        return explicit
    if isinstance(explicit, Mapping):
        ident = ComponentIdentity.from_mapping(name, explicit)
        if ident.digest and not ident.digest.startswith("sha256:"):
            ident = ComponentIdentity(name, ident.version, "sha256:" + ident.digest, ident.source, ident.reproducible)
        return ident
    if explicit is not None:
        return ComponentIdentity(name=name, version=str(explicit), source="explicit", reproducible=True)

    if artifact_path is not None:
        path = Path(artifact_path)
        if path.exists() and path.is_file():
            version = getattr(value, "version", None)
            return ComponentIdentity(
                name=name,
                version=None if version is None else str(version),
                digest=sha256_file(path),
                source="artifact",
                reproducible=True,
            )

    if name == "threshold" and value is not None and not callable(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            pass
        else:
            return ComponentIdentity(name=name, version=f"threshold:{numeric:.12g}", source="explicit", reproducible=True)

    if value is None:
        return ComponentIdentity(name=name, version="identity", source="explicit", reproducible=True)

    explicit_version = getattr(value, "version", None)
    if explicit_version is not None:
        return ComponentIdentity(name=name, version=str(explicit_version), source="explicit", reproducible=True)

    # Stateless functions/classes can be identified by source. Stateful instances
    # without explicit provenance are deliberately *not* fingerprinted from memory
    # addresses or pickled state.
    if inspect.isfunction(value) or inspect.isclass(value):
        # A closure can capture mutable/runtime state that is not represented by
        # source text. Do not label such a callable reproducible merely because
        # its source is available. Plain module-level functions/classes are safe
        # candidates for deterministic source identity.
        if inspect.isfunction(value) and value.__closure__:
            return ComponentIdentity(
                name=name,
                version=f"unstable:{value.__module__}.{value.__qualname__}",
                source="unstable",
                reproducible=False,
            )
        digest = _source_digest(value)
        if digest:
            return ComponentIdentity(
                name=name,
                version=f"source:{getattr(value, '__qualname__', name)}",
                digest=digest,
                source="source",
                reproducible=True,
            )

    return ComponentIdentity(
        name=name,
        version=f"unstable:{type(value).__module__}.{type(value).__qualname__}",
        digest=None,
        source="unstable",
        reproducible=False,
    )


def reproducibility_status(identities: Mapping[str, ComponentIdentity]) -> str:
    values = list(identities.values())
    if not values or all(item.reproducible for item in values):
        return "reproducible"
    if any(item.reproducible for item in values):
        return "partially_reproducible"
    return "unstable"
