from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_flow_cpu import branched_8_node_flow, dataset, linear_5_node_flow
from decishift.flow import compare_flows
from decishift.flow.attribution import approximate_flow_attribution, exact_flow_attribution
from decishift.flow.hybrid import FlowHybridCache


def _measure(fn):
    tracemalloc.start()
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak / 1024 / 1024


def _source_commit() -> str:
    explicit = os.environ.get("DECISHIFT_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _reuse_ratio(executed: int, reused: int) -> float:
    denominator = executed + reused
    return reused / denominator if denominator else 0.0


def benchmark_case(name: str, baseline, candidate, records: pd.DataFrame) -> list[dict]:
    result, compare_seconds, compare_peak_mb = _measure(
        lambda: compare_flows(baseline, candidate, records, id_column="record_id")
    )
    common = {
        "flow": name,
        "rows": len(records),
        "changed_nodes": len(result.changed_nodes),
        "changed_actions": int(result.summary()["changed_actions"]),
        "compare_wall_seconds": compare_seconds,
        "compare_peak_python_mb": compare_peak_mb,
        "comparison_cache_hits": int(result.metadata["execution"]["cache_hits"]),
        "comparison_cache_misses": int(result.metadata["execution"]["cache_misses"]),
    }

    exact_cache = FlowHybridCache(baseline, candidate, records)
    (_, exact_diag), exact_seconds, exact_peak_mb = _measure(
        lambda: exact_flow_attribution(
            baseline,
            candidate,
            records,
            result=result,
            cache=exact_cache,
        )
    )
    exact_row = {
        **common,
        "method": "exact",
        "attribution_wall_seconds": exact_seconds,
        "attribution_peak_python_mb": exact_peak_mb,
        "hybrid_evaluations": exact_cache.evaluations,
        "nodes_executed": exact_cache.nodes_evaluated,
        "node_outputs_reused": exact_cache.node_outputs_reused,
        "cache_reuse_ratio": _reuse_ratio(exact_cache.nodes_evaluated, exact_cache.node_outputs_reused),
        "efficiency_valid": bool(exact_diag.efficiency_valid),
        "efficiency_mae": float(exact_diag.efficiency_mae),
        "sampling_precision_sufficient": bool(exact_diag.sampling_precision_sufficient),
        "sampling_converged": bool(exact_diag.sampling_converged),
        "permutations_used": exact_diag.permutations_used,
        "stopped_early": bool(exact_diag.stopped_early),
        "max_ci_width": exact_diag.max_ci_width,
    }

    sampled_cache = FlowHybridCache(baseline, candidate, records)
    (sampled, sampled_diag), sampled_seconds, sampled_peak_mb = _measure(
        lambda: approximate_flow_attribution(
            baseline,
            candidate,
            records,
            result=result,
            min_permutations=32,
            max_permutations=256,
            batch_size=16,
            target_ci_width=0.05,
            confidence_level=0.95,
            seed=0,
            cache=sampled_cache,
        )
    )
    sampled_row = {
        **common,
        "method": "sampled",
        "attribution_wall_seconds": sampled_seconds,
        "attribution_peak_python_mb": sampled_peak_mb,
        "hybrid_evaluations": sampled_cache.evaluations,
        "nodes_executed": sampled_cache.nodes_evaluated,
        "node_outputs_reused": sampled_cache.node_outputs_reused,
        "cache_reuse_ratio": _reuse_ratio(sampled_cache.nodes_evaluated, sampled_cache.node_outputs_reused),
        "efficiency_valid": bool(sampled_diag.efficiency_valid),
        "efficiency_mae": float(sampled_diag.efficiency_mae),
        "sampling_precision_sufficient": bool(sampled_diag.sampling_precision_sufficient),
        "sampling_converged": bool(sampled_diag.sampling_converged),
        "permutations_used": int(sampled.attrs["permutations_used"]),
        "stopped_early": bool(sampled.attrs["stopped_early"]),
        "max_ci_width": None if sampled_diag.max_ci_width is None else float(sampled_diag.max_ci_width),
    }
    return [exact_row, sampled_row]


def run(sizes: list[int]) -> dict:
    rows: list[dict] = []
    for size in sizes:
        records = dataset(size)
        cases = [
            ("linear-5", linear_5_node_flow(False), linear_5_node_flow(True)),
            ("branched-8", branched_8_node_flow(False), branched_8_node_flow(True)),
        ]
        for name, baseline, candidate in cases:
            rows.extend(benchmark_case(name, baseline, candidate, records))
    return {
        "source_commit": _source_commit(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "benchmark": {
            "attribution_target": "candidate_action_support",
            "sampled_min_permutations": 32,
            "sampled_max_permutations": 256,
            "sampled_batch_size": 16,
            "sampled_target_ci_width": 0.05,
            "sampled_confidence_level": 0.95,
            "sampled_seed": 0,
            "memory_metric": "tracemalloc peak Python allocations; not process RSS",
        },
        "rows": rows,
    }


def _fmt(value, kind: str = "default") -> str:
    if value is None:
        return "—"
    if kind == "seconds":
        return f"{float(value):.4f}"
    if kind == "memory":
        return f"{float(value):.2f}"
    if kind == "ratio":
        return f"{float(value):.1%}"
    if kind == "ci":
        return f"{float(value):.4f}"
    return str(value)


def markdown(payload: dict) -> str:
    lines = [
        "# Reproducible DecisionFlow benchmark",
        "",
        f"Source commit: `{payload['source_commit']}`",
        "",
        "These numbers are machine-generated. They are not hand-entered into the README.",
        "Wall time and memory depend on the runner; the source commit and runtime metadata are recorded below.",
        "",
        "## Runtime",
        "",
    ]
    for key, value in payload["runtime"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Results",
        "",
        "| flow | rows | changed nodes | method | compare s | compare MB | attribution s | attribution MB | "
        "hybrid evals | nodes executed | outputs reused | reuse ratio | efficiency | sampling converged | perms | max CI width |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['flow']} | {row['rows']} | {row['changed_nodes']} | {row['method']} | "
            f"{_fmt(row['compare_wall_seconds'], 'seconds')} | {_fmt(row['compare_peak_python_mb'], 'memory')} | "
            f"{_fmt(row['attribution_wall_seconds'], 'seconds')} | {_fmt(row['attribution_peak_python_mb'], 'memory')} | "
            f"{row['hybrid_evaluations']} | {row['nodes_executed']} | {row['node_outputs_reused']} | "
            f"{_fmt(row['cache_reuse_ratio'], 'ratio')} | {row['efficiency_valid']} | "
            f"{row['sampling_converged']} | {_fmt(row['permutations_used'])} | {_fmt(row['max_ci_width'], 'ci')} |"
        )
    lines += [
        "",
        "Memory is `tracemalloc` peak Python allocation, not total process RSS.",
        "Exact attribution has no permutation-sampling interval, so its sampling fields are not a Monte Carlo convergence claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, 100_000])
    parser.add_argument("--include-million", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    sizes = list(args.sizes)
    if args.include_million and 1_000_000 not in sizes:
        sizes.append(1_000_000)
    payload = run(sizes)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))


if __name__ == "__main__":
    main()
