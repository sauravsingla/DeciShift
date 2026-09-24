"""DeciShift: trustworthy version-aware decision-system diffing."""

from decishift.core.identity import ComponentIdentity
from decishift.core.pipeline import DecisionPipeline, PipelineTrace
from decishift.diff.compare import ComparisonResult, compare_pipelines, compare_predictions
from decishift.flow import DecisionFlow, DecisionNode, DecisionOutput, FlowComparisonResult, FlowTrace, compare_flows
from decishift.version import __version__

__all__ = [
    "ComponentIdentity",
    "DecisionPipeline",
    "PipelineTrace",
    "ComparisonResult",
    "compare_pipelines",
    "compare_predictions",
    "DecisionFlow",
    "DecisionNode",
    "DecisionOutput",
    "FlowComparisonResult",
    "FlowTrace",
    "compare_flows",
    "__version__",
]
