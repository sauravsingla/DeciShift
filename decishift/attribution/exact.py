from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
import pandas as pd

from decishift.core.exceptions import InsufficientEvidenceError
from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import ComparisonResult
from decishift.replay.engine import HybridReplayCache


def _changed_components(baseline: DecisionPipeline, candidate: DecisionPipeline) -> list[str]:
    changed = baseline.changed_components(candidate)
    if not changed:
        return []
    return changed


def _all_subsets(items: list[str]) -> Iterable[frozenset[str]]:
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


def exact_attribution(
    baseline: DecisionPipeline,
    candidate: DecisionPipeline,
    records: pd.DataFrame,
    *,
    result: ComparisonResult | None = None,
    max_components: int = 10,
    cache: HybridReplayCache | None = None,
) -> pd.DataFrame:
    """Exact Shapley attribution over changed software components.

    Returns per-record contribution to calibrated score and final binary decision.
    These are counterfactual software attributions, not real-world causal effects.
    """
    changed = _changed_components(baseline, candidate)
    k = len(changed)
    if k == 0:
        return pd.DataFrame(columns=["record_id", "component", "score_contribution", "decision_contribution"])
    if k > max_components:
        raise InsufficientEvidenceError(
            f"Exact attribution requires 2^{k} hybrid evaluations; use approximate_attribution instead"
        )
    cache = cache or HybridReplayCache(baseline, candidate, records)
    for subset in _all_subsets(changed):
        cache.evaluate(subset)

    n = len(records)
    ids = result.records["record_id"].to_numpy() if result is not None else records.index.to_numpy()
    score_phi = {c: np.zeros(n, dtype=float) for c in changed}
    decision_phi = {c: np.zeros(n, dtype=float) for c in changed}

    denom = math.factorial(k)
    for component in changed:
        others = [c for c in changed if c != component]
        for subset in _all_subsets(others):
            s = len(subset)
            weight = math.factorial(s) * math.factorial(k - s - 1) / denom
            before = cache.evaluate(subset)
            after = cache.evaluate(set(subset) | {component})
            score_phi[component] += weight * (after.calibrated_scores - before.calibrated_scores)
            decision_phi[component] += weight * (after.decisions - before.decisions)

    rows: list[pd.DataFrame] = []
    for component in changed:
        rows.append(pd.DataFrame({
            "record_id": ids,
            "component": component,
            "score_contribution": score_phi[component],
            "decision_contribution": decision_phi[component],
        }))
    return pd.concat(rows, ignore_index=True)


def pairwise_interactions(
    baseline: DecisionPipeline,
    candidate: DecisionPipeline,
    records: pd.DataFrame,
    *,
    result: ComparisonResult | None = None,
    cache: HybridReplayCache | None = None,
) -> pd.DataFrame:
    """Baseline-anchored pairwise second-order interactions.

    For pair (i,j): f({i,j}) - f({i}) - f({j}) + f({}). The method also flags
    an interaction-only flip when neither component alone flips the baseline
    decision but their pair does. This is not a claim of real-world causality.
    """
    changed = _changed_components(baseline, candidate)
    if len(changed) < 2:
        return pd.DataFrame(columns=["record_id", "component_a", "component_b", "score_interaction", "decision_interaction", "interaction_only_flip"])
    cache = cache or HybridReplayCache(baseline, candidate, records)
    empty = cache.evaluate(set())
    ids = result.records["record_id"].to_numpy() if result is not None else records.index.to_numpy()
    rows = []
    for a, b in itertools.combinations(changed, 2):
        ta = cache.evaluate({a})
        tb = cache.evaluate({b})
        tab = cache.evaluate({a, b})
        score_interaction = tab.calibrated_scores - ta.calibrated_scores - tb.calibrated_scores + empty.calibrated_scores
        decision_interaction = tab.decisions - ta.decisions - tb.decisions + empty.decisions
        only_flip = (
            (ta.decisions == empty.decisions)
            & (tb.decisions == empty.decisions)
            & (tab.decisions != empty.decisions)
        )
        rows.append(pd.DataFrame({
            "record_id": ids,
            "component_a": a,
            "component_b": b,
            "score_interaction": score_interaction,
            "decision_interaction": decision_interaction,
            "interaction_only_flip": only_flip,
        }))
    return pd.concat(rows, ignore_index=True)
