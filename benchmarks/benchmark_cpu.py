from __future__ import annotations

import tempfile
import time
import tracemalloc
from pathlib import Path

from decishift.attribution import approximate_attribution, exact_attribution
from decishift.demo import dataset, pipelines
from decishift.diff.compare import compare_pipelines
from decishift.replay.engine import HybridReplayCache
from decishift.store import RunStore


def _measure(fn):
    tracemalloc.start()
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak / 1024 / 1024


def benchmark(n: int, *, approximate_permutations: int = 64) -> dict:
    records = dataset(n=n)
    baseline, candidate = pipelines()
    result, compare_seconds, compare_peak_mb = _measure(
        lambda: compare_pipelines(baseline, candidate, records, id_column="record_id")
    )
    exact_cache = HybridReplayCache(baseline, candidate, records)
    _, exact_seconds, exact_peak_mb = _measure(
        lambda: exact_attribution(baseline, candidate, records, result=result, cache=exact_cache)
    )
    approx_cache = HybridReplayCache(baseline, candidate, records)
    approximate, approx_seconds, approx_peak_mb = _measure(
        lambda: approximate_attribution(
            baseline,
            candidate,
            records,
            result=result,
            permutations=approximate_permutations,
            seed=0,
            cache=approx_cache,
        )
    )
    result.attribution = approximate
    result.metadata.update({
        "attribution_method": "approximate",
        "attribution_parameters": {"permutations": approximate_permutations, "confidence_level": 0.95},
        "random_seed": 0,
    })
    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(Path(tmp) / "runs")
        run_id, serialize_seconds, serialize_peak_mb = _measure(lambda: store.save(result))
        verification, verify_seconds, verify_peak_mb = _measure(lambda: store.verify(run_id))
        assert verification.passed
    return {
        "rows": n,
        "changed_decisions": result.summary()["changed_decisions"],
        "compare_wall_seconds": compare_seconds,
        "compare_peak_python_mb": compare_peak_mb,
        "exact_wall_seconds": exact_seconds,
        "exact_peak_python_mb": exact_peak_mb,
        "exact_hybrid_evaluations": exact_cache.evaluations,
        "approximate_permutations": approximate_permutations,
        "approximate_wall_seconds": approx_seconds,
        "approximate_peak_python_mb": approx_peak_mb,
        "approximate_hybrid_evaluations": approx_cache.evaluations,
        "evidence_serialization_seconds": serialize_seconds,
        "evidence_serialization_peak_python_mb": serialize_peak_mb,
        "verification_seconds": verify_seconds,
        "verification_peak_python_mb": verify_peak_mb,
    }


if __name__ == "__main__":
    for size in (10_000, 100_000):
        metrics = benchmark(size)
        print(" | ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
