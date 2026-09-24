from __future__ import annotations

import json
from typing import Any

import numpy as np


_RESERVED_STRING_PREFIXES = ("bool:", "int:", "float:", "str:")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def action_identity(value: Any) -> str:
    """Return a deterministic type-aware identity for a scalar action label.

    Python considers values such as ``True`` and ``1`` equal. DecisionFlow
    categorical actions must not inherit that numeric equality because they can
    represent distinct operational actions. The identity therefore includes the
    Python scalar type and canonical JSON representation.
    """
    value = _python_scalar(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"{type(value).__name__}:{encoded}"


def action_display_key(value: Any) -> str:
    """Readable evidence key that remains unambiguous for supported scalars."""
    value = _python_scalar(value)
    if isinstance(value, str):
        # Keep ordinary categorical strings concise while escaping strings that
        # could otherwise look identical to a type-tagged scalar such as int:1.
        if value.startswith(_RESERVED_STRING_PREFIXES):
            encoded = json.dumps(value, ensure_ascii=False)
            return f"str:{encoded}"
        return value
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"{type(value).__name__}:{encoded}"


def action_equal_mask(left: Any, right: Any) -> np.ndarray:
    """Type-aware elementwise equality for row-aligned categorical actions."""
    left_arr = np.asarray(left, dtype=object)
    right_arr = np.asarray(right, dtype=object)
    if left_arr.ndim != 1 or right_arr.ndim != 1 or len(left_arr) != len(right_arr):
        raise ValueError("Action comparisons require equally sized one-dimensional arrays")
    return np.asarray(
        [action_identity(a) == action_identity(b) for a, b in zip(left_arr, right_arr)],
        dtype=bool,
    )
