from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

from decishift.flow import DecisionFlow, DecisionNode, compare_flows
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


class Source:
    def __init__(self, column: str, version: str, shift: float = 0.0):
        self.column = column
        self.version = version
        self.shift = shift

    def run(self, records, inputs):
        return pd.Series(records[self.column].to_numpy(dtype=float) + self.shift, index=records.index)


class Transform:
    def __init__(self, version: str, scale: float = 1.0):
        self.version = version
        self.scale = scale

    def run(self, records, inputs):
        value = np.asarray(next(iter(inputs.values())), dtype=float)
        return pd.Series(value * self.scale, index=records.index)


class MergeMean:
    def __init__(self, version="merge_v1"):
        self.version = version

    def run(self, records, inputs):
        arrays = [np.asarray(inputs[name], dtype=float) for name in sorted(inputs)]
        return pd.Series(np.mean(arrays, axis=0), index=records.index)


class Policy:
    def __init__(self, version: str, inspect: float, service: float):
        self.version = version
        self.inspect = inspect
        self.service = service

    def run(self, records, inputs):
        score = np.asarray(next(iter(inputs.values())), dtype=float)
        action = np.full(len(records), "monitor", dtype=object)
        action[score >= self.inspect] = "inspect"
        action[score >= self.service] = "service"
        return pd.Series(action, index=records.index)


class Pass:
    def __init__(self, version: str):
        self.version = version

    def run(self, records, inputs):
        return pd.Series(np.asarray(next(iter(inputs.values())), dtype=object), index=records.index)


def dataset(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "record_id": np.arange(n),
        "x": rng.random(n),
        "y": rng.random(n),
        "z": rng.random(n),
        "w": rng.random(n),
    })


def linear_5_node_flow(candidate: bool) -> DecisionFlow:
    suffix = "2" if candidate else "1"
    shift = 0.03 if candidate else 0.0
    scale = 1.04 if candidate else 1.0
    return DecisionFlow([
        DecisionNode("source", Source("x", f"source_v{suffix}", shift), version=f"source_v{suffix}"),
        DecisionNode("transform", Transform(f"transform_v{suffix}", scale), ("source",), version=f"transform_v{suffix}"),
        DecisionNode("model", Transform(f"model_v{suffix}", 0.92), ("transform",), version=f"model_v{suffix}"),
        DecisionNode("policy", Policy(f"policy_v{suffix}", 0.42, 0.72), ("model",), version=f"policy_v{suffix}"),
        DecisionNode("final", Pass("final_v1"), ("policy",), version="final_v1"),
    ], "final", name="linear-5")


def branched_8_node_flow(candidate: bool) -> DecisionFlow:
    suffix = "2" if candidate else "1"
    x_shift = 0.025 if candidate else 0.0
    model_scale = 1.05 if candidate else 1.0
    policy_inspect = 0.47 if candidate else 0.50
    return DecisionFlow([
        DecisionNode("features", Source("x", f"features_v{suffix}", x_shift), version=f"features_v{suffix}"),
        DecisionNode("model_a", Transform(f"model_a_v{suffix}", model_scale), ("features",), version=f"model_a_v{suffix}", group="predictive_models"),
        DecisionNode("model_b", Source("y", "model_b_v1"), version="model_b_v1", group="predictive_models"),
        DecisionNode("model_c", Source("z", "model_c_v1"), version="model_c_v1", group="predictive_models"),
        DecisionNode("ensemble", MergeMean(), ("model_a", "model_b", "model_c"), version="ensemble_v1"),
        DecisionNode("policy", Policy(f"policy_v{suffix}", policy_inspect, 0.72), ("ensemble",), version=f"policy_v{suffix}"),
        DecisionNode("rules", Pass("rules_v1"), ("policy",), version="rules_v1"),
        DecisionNode("final", Pass("final_v1"), ("rules",), version="final_v1"),
    ], "final", name="branched-8")


def benchmark_flow(name: str, baseline: DecisionFlow, candidate: DecisionFlow, records: pd.DataFrame) -> dict:
    result, compare_seconds, compare_peak_mb = _measure(
        lambda: compare_flows(baseline, candidate, records, id_column="record_id")
    )

    exact_cache = FlowHybridCache(baseline, candidate, records)
    (_, exact_diag), exact_seconds, exact_peak_mb = _measure(
        lambda: exact_flow_attribution(baseline, candidate, records, result=result, cache=exact_cache)
    )

    approx_cache = FlowHybridCache(baseline, candidate, records)
    (approx, approx_diag), approx_seconds, approx_peak_mb = _measure(
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
            cache=approx_cache,
        )
    )
    return {
        "flow": name,
        "rows": len(records),
        "changed_actions": result.summary()["changed_actions"],
        "compare_wall_seconds": compare_seconds,
        "compare_peak_python_mb": compare_peak_mb,
        "comparison_cache_hits": result.metadata["execution"]["cache_hits"],
        "comparison_cache_misses": result.metadata["execution"]["cache_misses"],
        "exact_wall_seconds": exact_seconds,
        "exact_peak_python_mb": exact_peak_mb,
        "exact_hybrid_evaluations": exact_cache.evaluations,
        "exact_nodes_executed": exact_cache.nodes_evaluated,
        "exact_nodes_reused": exact_cache.node_outputs_reused,
        "exact_efficiency_valid": exact_diag.efficiency_valid,
        "adaptive_wall_seconds": approx_seconds,
        "adaptive_peak_python_mb": approx_peak_mb,
        "adaptive_hybrid_evaluations": approx_cache.evaluations,
        "adaptive_nodes_executed": approx_cache.nodes_evaluated,
        "adaptive_nodes_reused": approx_cache.node_outputs_reused,
        "adaptive_cache_hits": approx_cache.cache_hits,
        "adaptive_cache_misses": approx_cache.cache_misses,
        "adaptive_permutations_used": approx.attrs["permutations_used"],
        "adaptive_stopped_early": approx.attrs["stopped_early"],
        "adaptive_sampling_precision_sufficient": approx_diag.sampling_precision_sufficient,
    }


if __name__ == "__main__":
    for size in (10_000, 100_000):
        records = dataset(size)
        cases = [
            ("linear-5", linear_5_node_flow(False), linear_5_node_flow(True)),
            ("branched-8", branched_8_node_flow(False), branched_8_node_flow(True)),
        ]
        for name, baseline, candidate in cases:
            metrics = benchmark_flow(name, baseline, candidate, records)
            print(" | ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
