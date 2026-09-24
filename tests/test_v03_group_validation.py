import pandas as pd
import pytest

from decishift.flow import DecisionFlow, DecisionNode
from decishift.flow.validation import FlowValidationError


class Constant:
    def __init__(self, value, version):
        self.value = value
        self.version = version

    def run(self, records, inputs):
        return pd.Series([self.value] * len(records), index=records.index)


def test_group_name_cannot_collide_with_ungrouped_player_name():
    flow = DecisionFlow(
        [
            DecisionNode("x", Constant(1, "x2"), version="x2", group="y"),
            DecisionNode("y", Constant(1, "y2"), version="y2"),
            DecisionNode("final", Constant("inspect", "f1"), ("x", "y"), version="f1"),
        ],
        "final",
    )
    with pytest.raises(FlowValidationError, match="collides"):
        flow.attribution_groups(["x", "y"])
