from __future__ import annotations

from decishift.flow.flow import DecisionFlow


def descendants(flow: DecisionFlow, node_name: str) -> tuple[str, ...]:
    children: dict[str, set[str]] = {name: set() for name in flow.node_names}
    for name in flow.node_names:
        for dep in flow.node(name).depends_on:
            children[dep].add(name)
    seen: set[str] = set()
    stack = list(sorted(children[node_name], reverse=True))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(children[current], reverse=True))
    order_index = {name: i for i, name in enumerate(flow.topological_order)}
    return tuple(sorted(seen, key=lambda name: (order_index[name], name)))


def structural_impact(flow: DecisionFlow, changed_nodes: list[str]) -> dict:
    per_node = {name: list(descendants(flow, name)) for name in changed_nodes if name in flow.node_names}
    union = set()
    for values in per_node.values():
        union.update(values)
    ordered_union = [name for name in flow.topological_order if name in union]
    return {
        "changed_nodes": list(changed_nodes),
        "per_node_descendants": per_node,
        "structural_descendants": ordered_union,
        "structural_descendant_count": len(ordered_union),
        "note": "Structural reachability identifies potentially affected downstream software nodes; it is not causal impact and does not imply observed behavior changed.",
    }
