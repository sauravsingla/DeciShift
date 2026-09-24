from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Iterable

from decishift.flow.node import DecisionNode


class FlowValidationError(ValueError):
    """Raised when a DecisionFlow graph is structurally invalid."""


def deterministic_topological_order(nodes: Iterable[DecisionNode]) -> tuple[str, ...]:
    node_list = list(nodes)
    by_name: dict[str, DecisionNode] = {}
    for node in node_list:
        if node.name in by_name:
            raise FlowValidationError(f"Duplicate node name: {node.name}")
        by_name[node.name] = node

    indegree = {name: 0 for name in by_name}
    children: dict[str, list[str]] = {name: [] for name in by_name}
    for node in node_list:
        seen: set[str] = set()
        for dep in node.depends_on:
            if dep == node.name:
                raise FlowValidationError(f"Node '{node.name}' cannot depend on itself")
            if dep not in by_name:
                raise FlowValidationError(f"Node '{node.name}' depends on missing node '{dep}'")
            if dep in seen:
                raise FlowValidationError(f"Node '{node.name}' declares duplicate dependency '{dep}'")
            seen.add(dep)
            indegree[node.name] += 1
            children[dep].append(node.name)

    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for child in sorted(children[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(order) != len(by_name):
        remaining = sorted(name for name, degree in indegree.items() if degree > 0)
        raise FlowValidationError(f"DecisionFlow contains a cycle involving: {', '.join(remaining)}")
    return tuple(order)


def validate_flow_definition(nodes: Iterable[DecisionNode], final_node: str) -> tuple[DecisionNode, ...]:
    node_list = tuple(nodes)
    if not node_list:
        raise FlowValidationError("DecisionFlow requires at least one node")
    order = deterministic_topological_order(node_list)
    by_name = {node.name: node for node in node_list}
    if not final_node or final_node not in by_name:
        raise FlowValidationError(f"DecisionFlow final node does not exist: {final_node!r}")

    # A designated final decision node must be terminal. Other terminal helper
    # nodes are permitted, but they are not interpreted as final decisions.
    downstream = [node.name for node in node_list if final_node in node.depends_on]
    if downstream:
        raise FlowValidationError(
            f"Final node '{final_node}' must be terminal; downstream nodes: {', '.join(sorted(downstream))}"
        )
    return tuple(by_name[name] for name in order)


def canonical_topology_payload(nodes: Iterable[DecisionNode], final_node: str) -> dict:
    """Deterministic serialization independent of insertion order."""
    node_list = list(nodes)
    # Validate before serializing so invalid graphs cannot acquire a trusted
    # topology digest.
    deterministic_topological_order(node_list)
    return {
        "final_node": final_node,
        "nodes": [
            {
                "name": node.name,
                "depends_on": sorted(node.depends_on),
                "group": node.group,
            }
            for node in sorted(node_list, key=lambda item: item.name)
        ],
    }


def topology_digest(nodes: Iterable[DecisionNode], final_node: str) -> str:
    payload = canonical_topology_payload(nodes, final_node)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FlowTopologyDiff:
    unchanged_nodes: tuple[str, ...]
    changed_nodes: tuple[str, ...]
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    changed_dependencies: tuple[str, ...]
    changed_final_node: bool
    topology_compatible: bool

    def as_dict(self) -> dict:
        return {
            "unchanged_nodes": list(self.unchanged_nodes),
            "changed_nodes": list(self.changed_nodes),
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "changed_dependencies": list(self.changed_dependencies),
            "changed_final_node": self.changed_final_node,
            "topology_compatible": self.topology_compatible,
        }


def topology_diff(baseline, candidate) -> FlowTopologyDiff:
    b_names = set(baseline.node_names)
    c_names = set(candidate.node_names)
    shared = b_names & c_names
    added = sorted(c_names - b_names)
    removed = sorted(b_names - c_names)
    changed_dependencies = sorted(
        name
        for name in shared
        if tuple(sorted(baseline.node(name).depends_on)) != tuple(sorted(candidate.node(name).depends_on))
    )
    changed_nodes = sorted(
        name for name in shared if baseline.node_changed(candidate, name)
    )
    unchanged = sorted(shared - set(changed_nodes))
    changed_final = baseline.final_node != candidate.final_node
    compatible = not added and not removed and not changed_dependencies and not changed_final
    return FlowTopologyDiff(
        tuple(unchanged),
        tuple(changed_nodes),
        tuple(added),
        tuple(removed),
        tuple(changed_dependencies),
        changed_final,
        compatible,
    )
