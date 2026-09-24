from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from decishift.flow.actions import action_equal_mask
from decishift.flow.trace import FlowTrace


@runtime_checkable
class AttributionTarget(Protocol):
    """Numeric row-level value function for DecisionFlow attribution."""

    name: str

    def evaluate(
        self,
        trace: FlowTrace,
        baseline_trace: FlowTrace,
        candidate_trace: FlowTrace,
    ) -> np.ndarray:
        ...


class CandidateActionSupport:
    name = "candidate_action_support"

    def evaluate(self, trace: FlowTrace, baseline_trace: FlowTrace, candidate_trace: FlowTrace) -> np.ndarray:
        return action_equal_mask(trace.actions, candidate_trace.actions).astype(float)


class ChangeFromBaseline:
    name = "change_from_baseline"

    def evaluate(self, trace: FlowTrace, baseline_trace: FlowTrace, candidate_trace: FlowTrace) -> np.ndarray:
        return (~action_equal_mask(trace.actions, baseline_trace.actions)).astype(float)


BUILTIN_TARGETS = {
    "candidate_action_support": CandidateActionSupport,
    "change_from_baseline": ChangeFromBaseline,
}


def resolve_attribution_target(value: str | AttributionTarget) -> AttributionTarget:
    if isinstance(value, str):
        try:
            return BUILTIN_TARGETS[value]()
        except KeyError as exc:
            raise ValueError(
                "Unknown flow attribution target. Use candidate_action_support, change_from_baseline, or a numeric AttributionTarget."
            ) from exc
    if not hasattr(value, "evaluate"):
        raise TypeError("Custom attribution target must implement evaluate(trace, baseline_trace, candidate_trace)")
    return value


def evaluate_numeric_target(
    target: AttributionTarget,
    trace: FlowTrace,
    baseline_trace: FlowTrace,
    candidate_trace: FlowTrace,
) -> np.ndarray:
    raw = target.evaluate(trace, baseline_trace, candidate_trace)
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("AttributionTarget.evaluate must return numeric row-level values") from exc
    if values.ndim != 1 or len(values) != len(trace.actions):
        raise ValueError("AttributionTarget.evaluate must return one numeric value per record")
    if not np.isfinite(values).all():
        raise ValueError("AttributionTarget.evaluate returned NaN or infinite values")
    return values
