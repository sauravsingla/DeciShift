from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_example():
    path = Path("examples/rule_only_regression/run.py")
    spec = importlib.util.spec_from_file_location("rule_only_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rule_only_example_isolates_one_changed_node(capsys) -> None:
    example = _load_example()
    comparison = example.build_comparison()

    assert comparison.changed_nodes == ["rule"]
    assert np.array_equal(comparison.baseline_trace.output("score"), comparison.candidate_trace.output("score"))
    assert comparison.records["changed"].tolist() == [False, True, False]
    assert comparison.records.loc[1, "transition"] == "approve->review"

    example.main()
    output = capsys.readouterr().out
    assert "Changed nodes: rule" in output
    assert "software behavior only" in output
