from __future__ import annotations

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import ComparisonResult
from decishift.replay.engine import HybridReplayCache


def approximate_attribution(
    baseline: DecisionPipeline,
    candidate: DecisionPipeline,
    records: pd.DataFrame,
    *,
    result: ComparisonResult | None = None,
    permutations: int = 256,
    seed: int = 0,
    cache: HybridReplayCache | None = None,
) -> pd.DataFrame:
    """Permutation Monte Carlo approximation of component Shapley values."""
    changed = baseline.changed_components(candidate)
    if not changed:
        return pd.DataFrame(columns=["record_id", "component", "score_contribution", "decision_contribution"])
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    n = len(records)
    ids = result.records["record_id"].to_numpy() if result is not None else records.index.to_numpy()
    rng = np.random.default_rng(seed)
    cache = cache or HybridReplayCache(baseline, candidate, records)
    score = {c: np.zeros(n, dtype=float) for c in changed}
    decision = {c: np.zeros(n, dtype=float) for c in changed}

    for _ in range(permutations):
        order = list(rng.permutation(changed))
        subset: set[str] = set()
        before = cache.evaluate(subset)
        for component in order:
            subset = subset | {component}
            after = cache.evaluate(subset)
            score[component] += after.calibrated_scores - before.calibrated_scores
            decision[component] += after.decisions - before.decisions
            before = after

    rows = []
    for component in changed:
        rows.append(pd.DataFrame({
            "record_id": ids,
            "component": component,
            "score_contribution": score[component] / permutations,
            "decision_contribution": decision[component] / permutations,
        }))
    return pd.concat(rows, ignore_index=True)
