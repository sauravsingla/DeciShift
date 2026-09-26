from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from decishift.evidence import canonical_json_bytes
from decishift.flow.actions import action_display_key, action_equal_mask, action_identity

JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=40),
)

NUMPY_SCALAR = st.one_of(
    st.booleans().map(np.bool_),
    st.integers(min_value=-(2**7), max_value=2**7 - 1).map(np.int8),
    st.integers(min_value=-(2**31), max_value=2**31 - 1).map(np.int64),
    st.floats(allow_nan=False, allow_infinity=False, width=32).map(np.float32),
    st.floats(allow_nan=False, allow_infinity=False, width=64).map(np.float64),
)


@given(st.lists(JSON_SCALAR, max_size=50))
def test_action_equal_mask_is_reflexive(values: list[object]) -> None:
    mask = action_equal_mask(values, values)
    assert mask.tolist() == [True] * len(values)


@given(JSON_SCALAR)
def test_action_identity_is_deterministic(value: object) -> None:
    assert action_identity(value) == action_identity(value)
    assert action_display_key(value) == action_display_key(value)


@given(st.dictionaries(st.text(max_size=20), JSON_SCALAR, max_size=20))
def test_canonical_json_dict_order_is_irrelevant(mapping: dict[str, object]) -> None:
    reversed_mapping = dict(reversed(list(mapping.items())))
    assert canonical_json_bytes(mapping) == canonical_json_bytes(reversed_mapping)


def test_python_equal_values_keep_distinct_action_identity() -> None:
    assert True == 1
    assert 1 == 1.0
    assert action_identity(True) != action_identity(1)
    assert action_identity(1) != action_identity(1.0)
    assert action_display_key("int:1") != action_display_key(1)


@given(NUMPY_SCALAR)
def test_numpy_scalars_have_their_equivalent_python_identity(value: np.generic) -> None:
    python_value = value.item()

    assert action_identity(value) == action_identity(python_value)
    assert action_display_key(value) == action_display_key(python_value)


def test_numpy_equal_values_keep_distinct_action_identity() -> None:
    values = [np.bool_(True), np.int64(1), np.float64(1.0)]

    assert action_equal_mask(values, list(reversed(values))).tolist() == [False, True, False]
    assert len({action_identity(value) for value in values}) == 3


@pytest.mark.parametrize(
    ("dtype", "value", "expected_identity"),
    [
        ("Int64", 7, "int:7"),
        ("Float64", 2.5, "float:2.5"),
        ("boolean", True, "bool:true"),
        ("string", "review", 'str:"review"'),
    ],
)
def test_valid_pandas_nullable_scalars_are_normalized(dtype: str, value: object, expected_identity: str) -> None:
    scalar = pd.Series([value], dtype=dtype).iloc[0]

    assert action_identity(scalar) == expected_identity


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean", "string"])
def test_pandas_null_scalar_is_not_accepted_as_an_action(dtype: str) -> None:
    scalar = pd.Series([pd.NA], dtype=dtype).iloc[0]

    with pytest.raises(TypeError):
        action_identity(scalar)
