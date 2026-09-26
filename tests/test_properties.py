from __future__ import annotations

from hypothesis import given, strategies as st

from decishift.evidence import canonical_json_bytes
from decishift.flow.actions import action_display_key, action_equal_mask, action_identity

JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=40),
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
