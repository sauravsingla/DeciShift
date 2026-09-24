from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionOutput:
    """Optional structured final-node output.

    Fields may be row-aligned vectors/Series or, when returned as a sequence of
    DecisionOutput objects, per-record scalars. A numeric margin is never
    invented for categorical actions.
    """

    action: Any
    score: Any | None = None
    margin: Any | None = None


def validate_row_aligned_output(name: str, output: Any, records: pd.DataFrame) -> Any:
    if isinstance(output, DecisionOutput):
        return output
    if isinstance(output, (str, bytes)) or np.isscalar(output):
        raise ValueError(f"Node '{name}' must return one row-aligned value/object per input record")
    try:
        n = len(output)
    except TypeError as exc:
        raise ValueError(f"Node '{name}' must return a row-aligned output") from exc
    if n != len(records):
        raise ValueError(f"Node '{name}' changed row count from {len(records)} to {n}")
    if isinstance(output, (pd.DataFrame, pd.Series)) and not output.index.equals(records.index):
        raise ValueError(
            f"Node '{name}' returned a pandas object with reordered or changed index; row identity is ambiguous"
        )
    if isinstance(output, np.ndarray) and output.ndim == 0:
        raise ValueError(f"Node '{name}' returned a scalar array instead of one row per record")
    return output


def _vector(values: Any, records: pd.DataFrame, label: str, *, numeric: bool = False) -> np.ndarray:
    if isinstance(values, pd.Series):
        if not values.index.equals(records.index):
            raise ValueError(f"{label} pandas index does not match input record order")
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or len(arr) != len(records):
        raise ValueError(f"{label} must contain exactly one value per record")
    if numeric:
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} contains NaN or infinite values")
    return arr


def _normalize_action(value: Any, position: int) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        raise ValueError(f"Final action at row {position} is null")
    # Final actions are scalar labels. Sequences/mappings/arrays are not silently
    # coerced into labels even if a particular container happens to be hashable.
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series, pd.DataFrame)):
        raise ValueError(f"Final action at row {position} must be a scalar hashable value")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"Final action at row {position} is not hashable") from exc
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Final action at row {position} is not JSON-serializable") from exc
    return value


def normalize_final_output(output: Any, records: pd.DataFrame, final_node: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    score = None
    margin = None

    if isinstance(output, DecisionOutput):
        actions = _vector(output.action, records, f"Final node '{final_node}' action")
        if output.score is not None:
            score = _vector(output.score, records, f"Final node '{final_node}' score", numeric=True)
        if output.margin is not None:
            margin = _vector(output.margin, records, f"Final node '{final_node}' margin", numeric=True)
    elif isinstance(output, pd.DataFrame):
        if not output.index.equals(records.index):
            raise ValueError(f"Final node '{final_node}' reordered the pandas index")
        if "action" not in output.columns:
            raise ValueError(f"Final node '{final_node}' DataFrame output requires an 'action' column")
        actions = output["action"].to_numpy()
        if "score" in output.columns:
            score = _vector(output["score"], records, f"Final node '{final_node}' score", numeric=True)
        if "margin" in output.columns:
            margin = _vector(output["margin"], records, f"Final node '{final_node}' margin", numeric=True)
    elif isinstance(output, (list, tuple)) and output and all(isinstance(item, DecisionOutput) for item in output):
        if len(output) != len(records):
            raise ValueError(f"Final node '{final_node}' returned {len(output)} outputs for {len(records)} records")
        actions = np.asarray([item.action for item in output], dtype=object)
        if any(item.score is not None for item in output):
            if not all(item.score is not None for item in output):
                raise ValueError("DecisionOutput.score must be present for all rows or none")
            score = _vector([item.score for item in output], records, f"Final node '{final_node}' score", numeric=True)
        if any(item.margin is not None for item in output):
            if not all(item.margin is not None for item in output):
                raise ValueError("DecisionOutput.margin must be present for all rows or none")
            margin = _vector([item.margin for item in output], records, f"Final node '{final_node}' margin", numeric=True)
    else:
        validate_row_aligned_output(final_node, output, records)
        actions = _vector(output, records, f"Final node '{final_node}' action")

    normalized = np.asarray([_normalize_action(value, i) for i, value in enumerate(actions)], dtype=object)
    return normalized, score, margin


@dataclass(frozen=True)
class FlowTrace:
    outputs: Mapping[str, Any]
    actions: np.ndarray
    final_node: str
    record_index: pd.Index
    scores: np.ndarray | None = None
    margins: np.ndarray | None = None
    node_keys: Mapping[str, tuple] = field(default_factory=dict, repr=False)
    nodes_evaluated: int = 0
    node_outputs_reused: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def output(self, node_name: str) -> Any:
        return self.outputs[node_name]
