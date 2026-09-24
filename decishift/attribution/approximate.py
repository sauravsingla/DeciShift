from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import ComparisonResult
from decishift.replay.engine import HybridReplayCache


class _OnlineMoments:
    """Streaming per-record moments using Welford's algorithm."""

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


def _interval_widths(moments: dict[str, _OnlineMoments], z: float) -> np.ndarray:
    widths: list[np.ndarray] = []
    for value in moments.values():
        se = value.standard_error()
        widths.append(2.0 * z * se)
    return np.concatenate(widths) if widths else np.array([], dtype=float)


def _finite_width_stats(widths: np.ndarray) -> tuple[float | None, float | None]:
    finite = widths[np.isfinite(widths)]
    if finite.size == 0:
        return None, None
    return float(finite.max()), float(np.median(finite))


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
    min_permutations: int | None = None,
    max_permutations: int | None = None,
    batch_size: int = 32,
    target_ci_width: float | None = None,
) -> pd.DataFrame:
    """Permutation Monte Carlo component attribution with streaming uncertainty.

    The historical ``permutations`` argument remains fixed-permutation mode when
    no adaptive options are supplied. Adaptive mode is enabled when any of
    ``min_permutations``, ``max_permutations`` or ``target_ci_width`` is set.

    Confidence intervals quantify permutation-sampling uncertainty only. They do
    not quantify external, model, data-generating, or causal uncertainty.
    Samples are summarized online; individual permutation samples are never kept
    in memory.
    """
    changed = baseline.changed_components(candidate)
    columns = [
        "record_id", "component", "score_contribution", "decision_contribution",
        "score_standard_error", "decision_standard_error",
        "score_ci_low", "score_ci_high", "decision_ci_low", "decision_ci_high",
        "permutations_used", "confidence_level", "uncertain",
    ]
    if not changed:
        out = pd.DataFrame(columns=columns)
        out.attrs.update({
            "method": "approximate",
            "permutations": 0,
            "permutations_used": 0,
            "stopped_early": False,
            "sampling_precision_sufficient": True,
            "sampling_converged": True,
            "max_ci_width": 0.0,
            "median_ci_width": 0.0,
            "batch_stability_max_change": 0.0,
            "adaptive": False,
            "confidence_level": confidence_level,
        })
        return out
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if uncertainty_width_threshold <= 0:
        raise ValueError("uncertainty_width_threshold must be > 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if min_permutations is not None and min_permutations < 1:
        raise ValueError("min_permutations must be >= 1")
    if max_permutations is not None and max_permutations < 1:
        raise ValueError("max_permutations must be >= 1")
    if target_ci_width is not None and target_ci_width <= 0:
        raise ValueError("target_ci_width must be > 0")

    adaptive = any(value is not None for value in (min_permutations, max_permutations, target_ci_width))
    max_goal = int(max_permutations if max_permutations is not None else permutations)
    if adaptive:
        min_goal = int(min_permutations if min_permutations is not None else min(batch_size, max_goal))
        if max_goal < min_goal:
            raise ValueError("max_permutations must be >= min_permutations")
    else:
        min_goal = max_goal
    width_target = float(target_ci_width if target_ci_width is not None else uncertainty_width_threshold)

    n = len(records)
    ids = result.records["record_id"].to_numpy() if result is not None else records.index.to_numpy()
    rng = np.random.default_rng(seed)
    cache = cache or HybridReplayCache(baseline, candidate, records)
    score = {c: _OnlineMoments(n) for c in changed}
    decision = {c: _OnlineMoments(n) for c in changed}
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    used = 0
    previous_checkpoint: np.ndarray | None = None
    stability_max_change: float | None = None
    precision_sufficient = False
    sampling_converged = False

    while used < max_goal:
        current_batch = min(batch_size, max_goal - used)
        for _ in range(current_batch):
            order = list(rng.permutation(changed))
            subset: set[str] = set()
            before = cache.evaluate(subset)
            for component in order:
                subset = subset | {component}
                after = cache.evaluate(subset)
                score[component].update(after.calibrated_scores - before.calibrated_scores)
                decision[component].update(after.decisions.astype(float) - before.decisions.astype(float))
                before = after
            used += 1

        decision_widths = _interval_widths(decision, z)
        max_width, _ = _finite_width_stats(decision_widths)
        precision_sufficient = max_width is not None and max_width <= width_target

        checkpoint = np.concatenate([decision[c].mean for c in changed])
        if previous_checkpoint is not None:
            stability_max_change = float(np.max(np.abs(checkpoint - previous_checkpoint))) if checkpoint.size else 0.0
        previous_checkpoint = checkpoint.copy()

        # Precision alone is not called convergence. Require the estimated
        # contributions to be stable across successive sampling batches too.
        stability_limit = max(width_target / 4.0, 1e-12)
        stable = stability_max_change is not None and stability_max_change <= stability_limit
        sampling_converged = bool(precision_sufficient and stable)

        if adaptive and used >= min_goal and sampling_converged:
            break

    decision_widths = _interval_widths(decision, z)
    max_ci_width, median_ci_width = _finite_width_stats(decision_widths)
    precision_sufficient = max_ci_width is not None and max_ci_width <= width_target
    stability_limit = max(width_target / 4.0, 1e-12)
    stable = stability_max_change is not None and stability_max_change <= stability_limit
    sampling_converged = bool(precision_sufficient and stable)
    stopped_early = bool(adaptive and used < max_goal)

    rows: list[pd.DataFrame] = []
    for component in changed:
        score_se = score[component].standard_error()
        decision_se = decision[component].standard_error()
        score_low = score[component].mean - z * score_se
        score_high = score[component].mean + z * score_se
        decision_low = decision[component].mean - z * decision_se
        decision_high = decision[component].mean + z * decision_se
        decision_width = decision_high - decision_low
        uncertain = np.isnan(decision_width) | (decision_width > width_target)
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
            "permutations_used": used,
            "confidence_level": confidence_level,
            "uncertain": uncertain,
        }))
    out = pd.concat(rows, ignore_index=True)
    out.attrs.update({
        "method": "approximate",
        "permutations": used,
        "permutations_used": used,
        "requested_permutations": permutations,
        "min_permutations": min_goal if adaptive else None,
        "max_permutations": max_goal,
        "batch_size": batch_size,
        "target_ci_width": width_target,
        "seed": seed,
        "confidence_level": confidence_level,
        "uncertainty_width_threshold": uncertainty_width_threshold,
        "adaptive": adaptive,
        "stopped_early": stopped_early,
        "sampling_precision_sufficient": bool(precision_sufficient),
        "sampling_converged": bool(sampling_converged),
        "max_ci_width": max_ci_width,
        "median_ci_width": median_ci_width,
        "batch_stability_max_change": stability_max_change,
    })
    return out
