from __future__ import annotations

from dataclasses import dataclass

from decishift.core.exceptions import ReproducibilityError
from decishift.core.identity import reproducibility_status
from decishift.flow.node import DecisionNode
from decishift.flow.validation import (
    FlowValidationError,
    topology_diff,
    topology_digest,
    validate_flow_definition,
)


@dataclass(frozen=True)
class DecisionFlow:
    """A small, local DAG of versioned decision nodes.

    DecisionFlow is intentionally not a generic workflow engine: it executes a
    deterministic row-aligned decision graph over one historical DataFrame.
    """

    nodes: tuple[DecisionNode, ...] | list[DecisionNode]
    final_node: str
    name: str = "flow"
    strict_reproducibility: bool = False

    def __post_init__(self) -> None:
        ordered = validate_flow_definition(tuple(self.nodes), self.final_node)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "_topological_nodes", ordered)
        object.__setattr__(self, "_nodes_by_name", {node.name: node for node in self.nodes})

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self._topological_nodes)

    @property
    def topological_order(self) -> tuple[str, ...]:
        return self.node_names

    def node(self, name: str) -> DecisionNode:
        try:
            return self._nodes_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown flow node: {name}") from exc

    def node_identities(self) -> dict[str, dict]:
        return {name: self.node(name).evidence_identity() for name in self.node_names}

    @property
    def reproducibility_status(self) -> str:
        identities = {name: self.node(name).component_identity() for name in self.node_names}
        return reproducibility_status(identities)

    def validate_reproducibility(self) -> None:
        if not self.strict_reproducibility:
            return
        unstable = [name for name in self.node_names if not self.node(name).component_identity().reproducible]
        if unstable:
            raise ReproducibilityError(
                f"{self.name} has unstable node identity: {', '.join(unstable)}. "
                "Configure an explicit version/digest or artifact path."
            )

    @property
    def topology_digest(self) -> str:
        return topology_digest(self.nodes, self.final_node)

    def topology_payload(self) -> dict:
        return {
            "final_node": self.final_node,
            "topological_order": list(self.topological_order),
            "nodes": [
                {
                    "name": node.name,
                    "depends_on": list(node.depends_on),
                    "group": node.group,
                }
                for node in sorted(self.nodes, key=lambda item: item.name)
            ],
        }

    def node_changed(self, other: "DecisionFlow", name: str) -> bool:
        left = self.node(name)
        right = other.node(name)
        left_id = left.component_identity()
        right_id = right.component_identity()
        if left_id.reproducible and right_id.reproducible:
            return (left_id.version, left_id.digest) != (right_id.version, right_id.digest)
        return left.component is not right.component

    def changed_nodes(self, other: "DecisionFlow") -> list[str]:
        diff = topology_diff(self, other)
        if not diff.topology_compatible:
            # Stable version changes can still be reported for shared nodes, but
            # hybrid attribution must consult topology_compatible separately.
            return list(diff.changed_nodes)
        return [name for name in self.topological_order if self.node_changed(other, name)]

    def hybrid(self, candidate: "DecisionFlow", candidate_nodes: set[str] | frozenset[str]) -> "DecisionFlow":
        diff = topology_diff(self, candidate)
        if not diff.topology_compatible:
            raise FlowValidationError("Cannot construct a hybrid flow across a topology change")
        selected = set(candidate_nodes)
        unknown = selected - set(self.node_names)
        if unknown:
            raise FlowValidationError(f"Unknown hybrid nodes: {', '.join(sorted(unknown))}")
        nodes = tuple(candidate.node(name) if name in selected else self.node(name) for name in self.topological_order)
        return DecisionFlow(
            nodes=nodes,
            final_node=self.final_node,
            name=f"hybrid[{','.join(sorted(selected))}]",
            strict_reproducibility=self.strict_reproducibility or candidate.strict_reproducibility,
        )

    def attribution_groups(self, changed_nodes: list[str]) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for name in changed_nodes:
            node = self.node(name)
            player = node.group or name
            groups.setdefault(player, []).append(name)
        return {key: tuple(sorted(value)) for key, value in sorted(groups.items())}

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for name in self.topological_order:
            node = self.node(name)
            if not node.depends_on:
                lines.append(f"    {name}[{name}]")
            for dep in sorted(node.depends_on):
                lines.append(f"    {dep} --> {name}")
        return "\n".join(lines)

    def to_text(self) -> str:
        lines = [f"DecisionFlow: {self.name}", f"Final node: {self.final_node}", "Nodes:"]
        for name in self.topological_order:
            node = self.node(name)
            deps = ", ".join(node.depends_on) if node.depends_on else "records"
            group = f" group={node.group}" if node.group else ""
            lines.append(f"  {name} <- {deps}{group}")
        return "\n".join(lines)
