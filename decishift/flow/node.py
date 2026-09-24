from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from decishift.core.identity import ComponentIdentity, component_identity


@dataclass(frozen=True)
class DecisionNode:
    """One executable, versioned node in a :class:`DecisionFlow`.

    ``group`` is optional and is used only when a caller explicitly requests
    grouped attribution. DeciShift never invents attribution groups.
    """

    name: str
    component: Any
    depends_on: tuple[str, ...] = ()
    version: str | None = None
    group: str | None = None
    identity: ComponentIdentity | Mapping[str, Any] | None = None
    artifact_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DecisionNode.name must be a non-empty string")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if any(not isinstance(dep, str) or not dep.strip() for dep in self.depends_on):
            raise ValueError(f"Node '{self.name}' dependencies must be non-empty strings")
        if self.group is not None and (not isinstance(self.group, str) or not self.group.strip()):
            raise ValueError(f"Node '{self.name}' group must be a non-empty string when provided")

    def component_identity(self) -> ComponentIdentity:
        explicit: str | Mapping[str, Any] | ComponentIdentity | None
        explicit = self.identity if self.identity is not None else self.version
        return component_identity(
            self.name,
            self.component,
            explicit=explicit,
            artifact_path=self.artifact_path,
        )

    def cache_token(self) -> str:
        """Execution-local token; runtime identity is never serialized as evidence."""
        ident = self.component_identity()
        if ident.reproducible:
            if ident.digest and ident.version:
                return f"{ident.version}|{ident.digest}"
            return str(ident.digest or ident.version or "identity")
        return f"runtime-only:{self.name}:{id(self.component)}"

    def evidence_identity(self) -> dict[str, Any]:
        """Serializable provenance that never contains runtime memory addresses."""
        return self.component_identity().as_dict()
