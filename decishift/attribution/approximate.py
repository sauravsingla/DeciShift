from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import ComparisonResult
from decishift.replay.engine import HybridReplayCache


class _OnlineMoments:
    def __init__(self, n: int):
        self.count = 0
        self.mean = np.zeros(n, dtype=float)
        self.m2 = np.zeros(n, dtype=float)

    def update(self, values: np.ndarray) -> None:
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def standard_error(self) -> np.ndarray:
        if self.count < 2:
            return np.full_like(self.mean, np.nan, dtype=float)
        variance = self.m2 / (self.count - 1)
        variance = np.maximum(variance, 0.0)
        return np.sqrt(variance / self.count)


def approximate_attribution(
    baseline: DecisionPipeline,
    candidate: DecisionPipeline,
    records: pd.DataFrame,
    *,
    result: ComparisonResult | None = None,
    permutations: int = 256,
    seed: int = 0,
    cache: HybridReplayCache | None = None,
    confidence_level: float = 0.95,
    uncertainty_width_threshold: float = 0.25,
) -> pd.DataFrame:
    """Permutation Monte Carlo component attribution with streaming uncertainty.

    Confidence intervals describe Monte Carlo uncertainty across sampled
    permutations. They do not quantify external or causal uncertainty.
    """
    changed = baseline.changed_components(candidate)
    columns = [
        "record_id", "component", "score_contribution", "decision_contribution",
        "score_standard_error", "decision_standard_error",
        "score_ci_low", "score_ci_high", "decision_ci_low", "decision_ci_high",
        "permutations_used", "confidence_level", "uncertain",
    ]
    if not changed:
        return pd.DataFrame(columns=columns)
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if uncertainty_width_threshold <= 0:
        raise ValueError("uncertainty_width_threshold must be > 0")

    n = len(records)
    ids = result.records["record_id"].to_numpy() if result is not None else records.index.to_numpy()
    rng = np.random.default_rng(seed)
    cache = cache or HybridReplayCache(baseline, candidate, records)
    score = {c: _OnlineMoments(n) for c in changed}
    decision = {c: _OnlineMoments(n) for c in changed}

    for _ in range(permutations):
        order = list(rng.permutation(changed))
        subset: set[str] = set()
        before = cache.evaluate(subset)
        for component in order:
            subset = subset | {component}
            after = cache.evaluate(subset)
            score[component].update(after.calibrated_scores - before.calibrated_scores)
            decision[component].update(after.decisions.astype(float) - before.decisions.astype(float))
            before = after

    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rows: list[pd.DataFrame] = []
    for component in changed:
        score_se = score[component].standard_error()
        decision_se = decision[component].standard_error()
        score_low = score[component].mean - z * score_se
        score_high = score[component].mean + z * score_se
        decision_low = decision[component].mean - z * decision_se
        decision_high = decision[component].mean + z * decision_se
        decision_width = decision_high - decision_low
        uncertain = np.isnan(decision_width) | (decision_width > uncertainty_width_threshold)
        rows.append(pd.DataFrame({
            "record_id": ids,
            "component": component,
            "score_contribution": score[component].mean,
            "decision_contribution": decision[component].mean,
            "score_standard_error": score_se,
            "decision_standard_error": decision_se,
            "score_ci_low": score_low,
            "score_ci_high": score_high,
            "decision_ci_low": decision_low,
            "decision_ci_high": decision_high,
            "permutations_used": permutations,
            "confidence_level": confidence_level,
            "uncertain": uncertain,
        }))
    out = pd.concat(rows, ignore_index=True)
    out.attrs.update({
        "method": "approximate",
        "permutations": permutations,
        "seed": seed,
        "confidence_level": confidence_level,
        "uncertainty_width_threshold": uncertainty_width_threshold,
    })
    return out
