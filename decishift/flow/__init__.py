"""Composable, row-aligned decision-flow analysis for DeciShift v0.3."""

from decishift.flow.attribution import (
    FlowAttributionDiagnostics,
    approximate_flow_attribution,
    exact_flow_attribution,
    flow_pairwise_interactions,
)
from decishift.flow.compare import FlowComparisonResult, compare_flows
from decishift.flow.flow import DecisionFlow
from decishift.flow.node import DecisionNode
from decishift.flow.targets import AttributionTarget, CandidateActionSupport, ChangeFromBaseline
from decishift.flow.trace import DecisionOutput, FlowTrace
from decishift.flow.validation import FlowTopologyDiff, FlowValidationError, topology_diff

__all__ = [
    "AttributionTarget",
    "CandidateActionSupport",
    "ChangeFromBaseline",
    "DecisionFlow",
    "DecisionNode",
    "DecisionOutput",
    "FlowAttributionDiagnostics",
    "FlowComparisonResult",
    "FlowTopologyDiff",
    "FlowTrace",
    "FlowValidationError",
    "approximate_flow_attribution",
    "compare_flows",
    "exact_flow_attribution",
    "flow_pairwise_interactions",
    "topology_diff",
]
