#!/usr/bin/env python3
"""Generate a deterministic inventory of the public top-level API."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import decishift

TARGET = Path("docs/api-reference.md")


def _kind(value: object) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value):
        return "function"
    return "value"


def render() -> str:
    lines = [
        "# Public API reference",
        "",
        "This page is generated from `decishift.__all__` by `python scripts/generate_api_reference.py`. ",
        "It is an inventory of the stable top-level import surface, not a replacement for the conceptual guides.",
        "",
        "| Symbol | Kind | Defined in |",
        "|---|---|---|",
    ]
    for name in decishift.__all__:
        value = getattr(decishift, name)
        module = "decishift.version" if name == "__version__" else getattr(value, "__module__", type(value).__module__)
        lines.append(f"| `{name}` | {_kind(value)} | `{module}` |")
    lines.extend(
        [
            "",
            "## Entry points",
            "",
            "- Use `DecisionFlow`, `DecisionNode`, and `compare_flows` for structured row-aligned decision DAGs and categorical actions.",
            "- Use `DecisionPipeline` and `compare_pipelines` for the backwards-compatible fixed binary pipeline path.",
            "- See [Decision flows](decision-flows.md), [Flow attribution](flow-attribution.md), and [Failure modes](failure-modes.md) for semantics and trust boundaries.",
            "",
            "## Minimal examples",
            "",
            "These examples use local, deterministic components and the CPU-only `pandas` and `numpy` dependencies.",
            "",
            "### Compare fixed binary decisions",
            "",
            "```python",
            "import pandas as pd",
            "",
            "from decishift import DecisionPipeline, compare_pipelines",
            "",
            'records = pd.DataFrame({"record_id": ["a", "b", "c"], "score": [0.2, 0.6, 0.9]})',
            "",
            "def score(frame):",
            '    return frame["score"].to_numpy()',
            "",
            'baseline = DecisionPipeline(model=score, threshold=0.5, versions={"model": "score-v1"})',
            'candidate = DecisionPipeline(model=score, threshold=0.7, versions={"model": "score-v1"})',
            'comparison = compare_pipelines(baseline, candidate, records, id_column="record_id")',
            "",
            'print(comparison.summary()["changed_decisions"])  # 1',
            "```",
            "",
            "### Compare categorical decision flows",
            "",
            "```python",
            "import numpy as np",
            "import pandas as pd",
            "",
            "from decishift import DecisionFlow, DecisionNode, compare_flows",
            "",
            'records = pd.DataFrame({"record_id": ["a", "b", "c"], "score": [0.2, 0.6, 0.9]})',
            "",
            "def score(frame):",
            '    return frame["score"].to_numpy()',
            "",
            "def route_at(threshold):",
            "    def route(_frame, inputs):",
            '        return np.where(inputs["score"] >= threshold, "approve", "review")',
            "    return route",
            "",
            "baseline = DecisionFlow(",
            "    nodes=(",
            '        DecisionNode("score", score, version="score-v1"),',
            '        DecisionNode("route", route_at(0.5), depends_on=("score",), version="route-v1"),',
            "    ),",
            '    final_node="route",',
            ")",
            "candidate = DecisionFlow(",
            "    nodes=(",
            '        DecisionNode("score", score, version="score-v1"),',
            '        DecisionNode("route", route_at(0.7), depends_on=("score",), version="route-v2"),',
            "    ),",
            '    final_node="route",',
            ")",
            'comparison = compare_flows(baseline, candidate, records, id_column="record_id")',
            "",
            'print(comparison.summary()["changed_actions"])  # 1',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if the committed reference is stale.")
    args = parser.parse_args()
    expected = render()

    if args.check:
        actual = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if actual != expected:
            print(f"{TARGET} is stale; run python scripts/generate_api_reference.py")
            return 1
        print(f"{TARGET} is current.")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected, encoding="utf-8")
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
