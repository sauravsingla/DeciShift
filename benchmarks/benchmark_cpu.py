from __future__ import annotations

import time
import tracemalloc

from decishift.demo import dataset, pipelines
from decishift.attribution import exact_attribution, pairwise_interactions
from decishift.diff.compare import compare_pipelines
from decishift.replay.engine import HybridReplayCache


def benchmark(n: int) -> dict:
    records = dataset(n=n)
    baseline, candidate = pipelines()
    tracemalloc.start()
    start = time.perf_counter()
    result = compare_pipelines(baseline, candidate, records, id_column="record_id")
    replay = HybridReplayCache(baseline, candidate, records)
    result.attribution = exact_attribution(baseline, candidate, records, result=result, cache=replay)
    result.interactions = pairwise_interactions(baseline, candidate, records, result=result, cache=replay)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "rows": n,
        "wall_clock_seconds": elapsed,
        "python_peak_memory_mb": peak / 1024 / 1024,
        "changed_decisions": result.summary()["changed_decisions"],
        "pipeline_evaluations": 2 + replay.evaluations,
    }


if __name__ == "__main__":
    for size in (10_000, 100_000):
        metrics = benchmark(size)
        print(" | ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
