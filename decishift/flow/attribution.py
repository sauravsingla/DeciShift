from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

from decishift.core.exceptions import InsufficientEvidenceError
from decishift.flow.actions import action_display_key, action_equal_mask
from decishift.flow.flow import DecisionFlow
from decishift.flow.hybrid import FlowHybridCache
from decishift.flow.targets import AttributionTarget, evaluate_numeric_target, resolve_attribution_target
from decishift.flow.validation import topology_diff


def _all_subsets(items: list[str]) -> Iterable[frozenset[str]]:
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


def _player_mapping(
    baseline: DecisionFlow,
    candidate: DecisionFlow,
    *,
    grouped: bool,
) -> dict[str, tuple[str, ...]]:
    diff = topology_diff(baseline, candidate)
    if not diff.topology_compatible:
        raise InsufficientEvidenceError("Node attribution is unsupported because the baseline/candidate topology changed")
    changed = baseline.changed_nodes(candidate)
    if not grouped:
        return {name: (name,) for name in changed}
    for name in changed:
        if baseline.node(name).group != candidate.node(name).group:
            raise InsufficientEvidenceError(
                f"Grouped attribution requires stable group declarations; node '{name}' changed group"
            )
    return candidate.attribution_groups(changed)


def _nodes_for_players(mapping: dict[str, tuple[str, ...]], players: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for player in players:
        selected.update(mapping[player])
    return selected


def _record_ids(records: pd.DataFrame, result=None) -> np.ndarray:
    if result is not None and "record_id" in result.records.columns:
        return result.records["record_id"].to_numpy()
    return records.index.to_numpy()


@dataclass
class FlowAttributionDiagnostics:
    method: str
    target: str
    players: list[str]
    hybrid_evaluations: int
    efficiency_mae: float
    efficiency_max: float
    efficiency_p95: float
    efficiency_fraction_exceeding_tolerance: float
    efficiency_valid: bool
    sampling_precision_sufficient: bool
    sampling_converged: bool
    permutations_used: int | None
    stopped_early: bool
    max_ci_width: float | None
    median_ci_width: float | None
    tolerance: float = 1e-9
    warnings: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """Compatibility-style aggregate: efficiency plus applicable sampling checks."""
        return bool(self.efficiency_valid and self.sampling_converged)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["converged"] = self.converged
        return value


def _flow_diagnostics(
    attribution: pd.DataFrame,
    cache: FlowHybridCache,
    mapping: dict[str, tuple[str, ...]],
    target: AttributionTarget,
    *,
    method: str,
    tolerance: float,
) -> FlowAttributionDiagnostics:
    players = list(mapping)
    baseline_trace = cache.evaluate(set())
    all_nodes = _nodes_for_players(mapping, players)
    candidate_trace = cache.evaluate(all_nodes)
    expected = (
        evaluate_numeric_target(target, candidate_trace, baseline_trace, candidate_trace)
        - evaluate_numeric_target(target, baseline_trace, baseline_trace, candidate_trace)
    )
    if attribution.empty:
        residual = expected
    else:
        grouped = attribution.groupby("record_id", sort=False)["target_contribution"].sum().to_numpy(dtype=float)
        residual = grouped - expected
    absolute = np.abs(residual)
    mae = float(absolute.mean()) if len(absolute) else 0.0
    maximum = float(absolute.max()) if len(absolute) else 0.0
    p95 = float(np.quantile(absolute, 0.95)) if len(absolute) else 0.0
    fraction = float((absolute > tolerance).mean()) if len(absolute) else 0.0
    efficiency_valid = fraction == 0.0
    attrs = attribution.attrs
    if method == "approximate":
        precision = bool(attrs.get("sampling_precision_sufficient", False))
        sampling_converged = bool(attrs.get("sampling_converged", False))
        permutations_used = attrs.get("permutations_used")
        stopped_early = bool(attrs.get("stopped_early", False))
        max_width = attrs.get("max_ci_width")
        median_width = attrs.get("median_ci_width")
    else:
        precision = True
        sampling_converged = True
        permutations_used = None
        stopped_early = False
        max_width = 0.0
        median_width = 0.0
    warnings: list[str] = []
    if not efficiency_valid:
        warnings.append("Flow attribution efficiency residual exceeded configured tolerance.")
    if method == "approximate" and not precision:
        warnings.append("Flow attribution confidence intervals exceed the configured sampling-precision target.")
    if method == "approximate" and not sampling_converged:
        warnings.append("Flow attribution did not satisfy both sampling precision and batch-stability criteria.")
    return FlowAttributionDiagnostics(
        method=method,
        target=target.name,
        players=players,
        hybrid_evaluations=cache.evaluations,
        efficiency_mae=mae,
        efficiency_max=maximum,
        efficiency_p95=p95,
        efficiency_fraction_exceeding_tolerance=fraction,
        efficiency_valid=efficiency_valid,
        sampling_precision_sufficient=precision,
        sampling_converged=sampling_converged,
        permutations_used=permutations_used,
        stopped_early=stopped_early,
        max_ci_width=max_width,
        median_ci_width=median_width,
        tolerance=tolerance,
        warnings=warnings,
    )


def exact_flow_attribution(
    baseline: DecisionFlow,
    candidate: DecisionFlow,
    records: pd.DataFrame,
    *,
    result=None,
    target: str | AttributionTarget = "candidate_action_support",
    grouped: bool = False,
    max_players: int = 10,
    cache: FlowHybridCache | None = None,
) -> tuple[pd.DataFrame, FlowAttributionDiagnostics]:
    mapping = _player_mapping(baseline, candidate, grouped=grouped)
    players = list(mapping)
    if len(players) > max_players:
        raise InsufficientEvidenceError(
            f"Exact flow attribution requires 2^{len(players)} hybrids; use approximate flow attribution instead"
        )
    target_obj = resolve_attribution_target(target)
    cache = cache or FlowHybridCache(baseline, candidate, records)
    baseline_trace = cache.evaluate(set())
    all_nodes = _nodes_for_players(mapping, players)
    candidate_trace = cache.evaluate(all_nodes)
    values: dict[frozenset[str], np.ndarray] = {}
    for subset in _all_subsets(players):
        trace = cache.evaluate(_nodes_for_players(mapping, subset))
        values[subset] = evaluate_numeric_target(target_obj, trace, baseline_trace, candidate_trace)

    n = len(records)
    ids = _record_ids(records, result)
    phi = {player: np.zeros(n, dtype=float) for player in players}
    if players:
        denom = math.factorial(len(players))
        for player in players:
            others = [value for value in players if value != player]
            for subset in _all_subsets(others):
                s = len(subset)
                weight = math.factorial(s) * math.factorial(len(players) - s - 1) / denom
                phi[player] += weight * (values[subset | {player}] - values[subset])

    rows = []
    specific_column = (
        "candidate_action_contribution"
        if target_obj.name == "candidate_action_support"
        else "change_from_baseline_contribution"
        if target_obj.name == "change_from_baseline"
        else None
    )
    for player in players:
        data = {
            "record_id": ids,
            "player": player,
            "attribution_target": target_obj.name,
            "target_contribution": phi[player],
        }
        if specific_column:
            data[specific_column] = phi[player]
        rows.append(pd.DataFrame(data))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["record_id", "player", "attribution_target", "target_contribution"]
    )
    out.attrs.update({
        "method": "exact",
        "players": players,
        "player_nodes": {key: list(value) for key, value in mapping.items()},
        "attribution_target": target_obj.name,
        "hybrid_evaluations": cache.evaluations,
        "sampling_precision_sufficient": True,
        "sampling_converged": True,
        "permutations_used": None,
        "stopped_early": False,
        "max_ci_width": 0.0,
        "median_ci_width": 0.0,
    })
    diagnostics = _flow_diagnostics(out, cache, mapping, target_obj, method="exact", tolerance=1e-9)
    return out, diagnostics


class _OnlineMoments:
    def __init__(self, n: int):
        self.count = 0
        self.mean = np.zeros(n, dtype=float)
        self.m2 = np.zeros(n, dtype=float)

    def update(self, values: np.ndarray) -> None:
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    def standard_error(self) -> np.ndarray:
        if self.count < 2:
            return np.full_like(self.mean, np.nan, dtype=float)
        variance = np.maximum(self.m2 / (self.count - 1), 0.0)
        return np.sqrt(variance / self.count)


def approximate_flow_attribution(
    baseline: DecisionFlow,
    candidate: DecisionFlow,
    records: pd.DataFrame,
    *,
    result=None,
    target: str | AttributionTarget = "candidate_action_support",
    grouped: bool = False,
    permutations: int = 256,
    min_permutations: int | None = None,
    max_permutations: int | None = None,
    batch_size: int = 32,
    target_ci_width: float | None = None,
    confidence_level: float = 0.95,
    seed: int = 0,
    cache: FlowHybridCache | None = None,
) -> tuple[pd.DataFrame, FlowAttributionDiagnostics]:
    mapping = _player_mapping(baseline, candidate, grouped=grouped)
    players = list(mapping)
    if permutations < 1 or batch_size < 1:
        raise ValueError("permutations and batch_size must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    adaptive = any(value is not None for value in (min_permutations, max_permutations, target_ci_width))
    max_goal = int(max_permutations if max_permutations is not None else permutations)
    min_goal = int(min_permutations if min_permutations is not None else min(batch_size, max_goal)) if adaptive else max_goal
    if min_goal < 1 or max_goal < min_goal:
        raise ValueError("Require 1 <= min_permutations <= max_permutations")
    width_target = float(target_ci_width if target_ci_width is not None else 0.25)
    if width_target <= 0:
        raise ValueError("target_ci_width must be > 0")

    target_obj = resolve_attribution_target(target)
    cache = cache or FlowHybridCache(baseline, candidate, records)
    baseline_trace = cache.evaluate(set())
    all_nodes = _nodes_for_players(mapping, players)
    candidate_trace = cache.evaluate(all_nodes)
    n = len(records)
    ids = _record_ids(records, result)
    moments = {player: _OnlineMoments(n) for player in players}
    rng = np.random.default_rng(seed)
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    used = 0
    previous_checkpoint: np.ndarray | None = None
    stability_max_change: float | None = None
    precision = not players
    sampling_converged = not players

    while players and used < max_goal:
        for _ in range(min(batch_size, max_goal - used)):
            order = list(rng.permutation(players))
            subset: set[str] = set()
            before_trace = cache.evaluate(set())
            before_value = evaluate_numeric_target(target_obj, before_trace, baseline_trace, candidate_trace)
            for player in order:
                subset.add(player)
                after_trace = cache.evaluate(_nodes_for_players(mapping, subset))
                after_value = evaluate_numeric_target(target_obj, after_trace, baseline_trace, candidate_trace)
                moments[player].update(after_value - before_value)
                before_value = after_value
            used += 1

        widths = np.concatenate([2.0 * z * moments[player].standard_error() for player in players])
        finite = widths[np.isfinite(widths)]
        max_width = float(finite.max()) if finite.size else None
        precision = max_width is not None and max_width <= width_target
        checkpoint = np.concatenate([moments[player].mean for player in players])
        if previous_checkpoint is not None:
            stability_max_change = float(np.max(np.abs(checkpoint - previous_checkpoint))) if checkpoint.size else 0.0
        previous_checkpoint = checkpoint.copy()
        stable = stability_max_change is not None and stability_max_change <= max(width_target / 4.0, 1e-12)
        sampling_converged = bool(precision and stable)
        if adaptive and used >= min_goal and sampling_converged:
            break

    rows = []
    all_widths: list[np.ndarray] = []
    specific_column = (
        "candidate_action_contribution"
        if target_obj.name == "candidate_action_support"
        else "change_from_baseline_contribution"
        if target_obj.name == "change_from_baseline"
        else None
    )
    for player in players:
        se = moments[player].standard_error()
        low = moments[player].mean - z * se
        high = moments[player].mean + z * se
        widths = high - low
        all_widths.append(widths)
        data = {
            "record_id": ids,
            "player": player,
            "attribution_target": target_obj.name,
            "target_contribution": moments[player].mean,
            "standard_error": se,
            "ci_low": low,
            "ci_high": high,
            "permutations_used": used,
            "confidence_level": confidence_level,
            "uncertain": np.isnan(widths) | (widths > width_target),
        }
        if specific_column:
            data[specific_column] = moments[player].mean
        rows.append(pd.DataFrame(data))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["record_id", "player", "attribution_target", "target_contribution", "standard_error", "ci_low", "ci_high"]
    )
    combined_widths = np.concatenate(all_widths) if all_widths else np.array([0.0])
    finite = combined_widths[np.isfinite(combined_widths)]
    max_width = float(finite.max()) if finite.size else 0.0
    median_width = float(np.median(finite)) if finite.size else 0.0
    precision = max_width <= width_target
    stable = (not players) or (stability_max_change is not None and stability_max_change <= max(width_target / 4.0, 1e-12))
    sampling_converged = bool(precision and stable)
    out.attrs.update({
        "method": "approximate",
        "players": players,
        "player_nodes": {key: list(value) for key, value in mapping.items()},
        "attribution_target": target_obj.name,
        "hybrid_evaluations": cache.evaluations,
        "permutations_used": used,
        "requested_permutations": permutations,
        "min_permutations": min_goal if adaptive else None,
        "max_permutations": max_goal,
        "batch_size": batch_size,
        "target_ci_width": width_target,
        "confidence_level": confidence_level,
        "seed": seed,
        "adaptive": adaptive,
        "stopped_early": bool(adaptive and used < max_goal),
        "sampling_precision_sufficient": bool(precision),
        "sampling_converged": bool(sampling_converged),
        "max_ci_width": max_width,
        "median_ci_width": median_width,
        "batch_stability_max_change": stability_max_change,
    })
    diagnostics = _flow_diagnostics(out, cache, mapping, target_obj, method="approximate", tolerance=1e-8)
    return out, diagnostics


def flow_pairwise_interactions(
    baseline: DecisionFlow,
    candidate: DecisionFlow,
    records: pd.DataFrame,
    *,
    result=None,
    target: str | AttributionTarget = "candidate_action_support",
    grouped: bool = False,
    cache: FlowHybridCache | None = None,
) -> pd.DataFrame:
    mapping = _player_mapping(baseline, candidate, grouped=grouped)
    players = list(mapping)
    if len(players) < 2:
        return pd.DataFrame(columns=[
            "record_id", "player_a", "player_b", "attribution_target", "target_interaction",
            "interaction_only_transition", "resulting_transition",
        ])
    target_obj = resolve_attribution_target(target)
    cache = cache or FlowHybridCache(baseline, candidate, records)
    empty = cache.evaluate(set())
    all_nodes = _nodes_for_players(mapping, players)
    candidate_trace = cache.evaluate(all_nodes)
    v0 = evaluate_numeric_target(target_obj, empty, empty, candidate_trace)
    ids = _record_ids(records, result)
    rows = []
    for a, b in itertools.combinations(players, 2):
        ta = cache.evaluate(_nodes_for_players(mapping, {a}))
        tb = cache.evaluate(_nodes_for_players(mapping, {b}))
        tab = cache.evaluate(_nodes_for_players(mapping, {a, b}))
        va = evaluate_numeric_target(target_obj, ta, empty, candidate_trace)
        vb = evaluate_numeric_target(target_obj, tb, empty, candidate_trace)
        vab = evaluate_numeric_target(target_obj, tab, empty, candidate_trace)
        interaction = vab - va - vb + v0
        interaction_only = (
            action_equal_mask(ta.actions, empty.actions)
            & action_equal_mask(tb.actions, empty.actions)
            & ~action_equal_mask(tab.actions, empty.actions)
        )
        same_as_baseline = action_equal_mask(empty.actions, tab.actions)
        transition = np.asarray([
            "unchanged" if same else f"{action_display_key(base)}->{action_display_key(after)}"
            for base, after, same in zip(empty.actions, tab.actions, same_as_baseline)
        ], dtype=object)
        rows.append(pd.DataFrame({
            "record_id": ids,
            "player_a": a,
            "player_b": b,
            "attribution_target": target_obj.name,
            "target_interaction": interaction,
            "interaction_only_transition": interaction_only,
            "resulting_transition": transition,
        }))
    return pd.concat(rows, ignore_index=True)
