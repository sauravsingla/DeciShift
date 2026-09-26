from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from decishift.core.exceptions import ComponentExecutionError
from decishift.evidence import fingerprint_dataframe
from decishift.flow.flow import DecisionFlow
from decishift.flow.trace import FlowTrace, normalize_final_output, validate_row_aligned_output


def _invoke_node(component: Any, node_name: str, records: pd.DataFrame, inputs: dict[str, Any]) -> Any:
    if component is None:
        if inputs:
            raise TypeError(f"Node '{node_name}' with dependencies cannot use component=None")
        return records
    fn: Callable[..., Any]
    if hasattr(component, "run"):
        fn = component.run
    elif callable(component):
        fn = component
    else:
        raise TypeError(f"Node '{node_name}' component must be callable or expose run(records, inputs)")

    variants = [(records, inputs), (records,)]
    args: tuple[Any, ...] | None
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
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
            raise TypeError(
                f"Node '{node_name}' component must accept run(records, inputs) or a records-only callable"
            )
    try:
        return fn(*args)
    except Exception as exc:
        # Do not catch a TypeError and retry a different calling convention: a
        # TypeError raised *inside* user code is an execution failure, not arity.
        raise ComponentExecutionError(
            f"DecisionFlow node '{node_name}' failed: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class FlowExecutor:
    """Execute one row-aligned DecisionFlow with dependency-aware local caching."""

    records: pd.DataFrame
    node_cache: dict[tuple, Any] = field(default_factory=dict)

    def evaluate(self, flow: DecisionFlow) -> FlowTrace:
        flow.validate_reproducibility()
        # Snapshot caller-owned records once, then give every user node its own
        # copy. A node can mutate its local frame without affecting siblings,
        # descendants that read records directly, a later candidate evaluation,
        # or the caller's original DataFrame.
        execution_records = self.records.copy(deep=True)
        outputs: dict[str, Any] = {}
        keys: dict[str, tuple] = {}
        records_fingerprint = fingerprint_dataframe(execution_records)
        hits = 0
        misses = 0

        for name in flow.topological_order:
            node = flow.node(name)
            dependency_key = tuple((dep, keys[dep]) for dep in sorted(node.depends_on))
            key = ("flow-node", records_fingerprint, name, node.cache_token(), dependency_key)
            # Give user code isolated dependency values so an in-place mutation cannot
            # corrupt upstream outputs or the cache entry used by later evaluations.
            inputs = {dep: copy.deepcopy(outputs[dep]) for dep in node.depends_on}
            if key in self.node_cache:
                output = copy.deepcopy(self.node_cache[key])
                hits += 1
            else:
                output = _invoke_node(
                    node.component,
                    name,
                    execution_records.copy(deep=True),
                    inputs,
                )
                output = validate_row_aligned_output(name, output, execution_records)
                self.node_cache[key] = copy.deepcopy(output)
                misses += 1
            outputs[name] = output
            keys[name] = key

        actions, scores, margins = normalize_final_output(
            outputs[flow.final_node],
            execution_records,
            flow.final_node,
        )
        return FlowTrace(
            outputs=outputs,
            actions=actions,
            final_node=flow.final_node,
            record_index=execution_records.index.copy(),
            scores=scores,
            margins=margins,
            node_keys=keys,
            nodes_evaluated=misses,
            node_outputs_reused=hits,
            cache_hits=hits,
            cache_misses=misses,
        )
