from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from decishift.flow.executor import FlowExecutor
from decishift.flow.flow import DecisionFlow
from decishift.flow.trace import FlowTrace
from decishift.flow.validation import topology_diff


@dataclass
class FlowHybridCache:
    """Memoize hybrid DecisionFlow traces and reusable node outputs."""

    baseline: DecisionFlow
    candidate: DecisionFlow
    records: pd.DataFrame
    _trace_cache: dict[frozenset[str], FlowTrace] = field(default_factory=dict)
    _executor: FlowExecutor = field(init=False)

    def __post_init__(self) -> None:
        if not topology_diff(self.baseline, self.candidate).topology_compatible:
            raise ValueError("Flow hybrid attribution is unavailable because topology changed")
        self._executor = FlowExecutor(self.records)

    def evaluate(self, candidate_nodes: set[str] | frozenset[str]) -> FlowTrace:
        key = frozenset(candidate_nodes)
        if key not in self._trace_cache:
            hybrid = self.baseline.hybrid(self.candidate, key)
            self._trace_cache[key] = self._executor.evaluate(hybrid)
        return self._trace_cache[key]

    @property
    def evaluations(self) -> int:
        return len(self._trace_cache)

    @property
    def nodes_evaluated(self) -> int:
        return sum(trace.nodes_evaluated for trace in self._trace_cache.values())

    @property
    def node_outputs_reused(self) -> int:
        return sum(trace.node_outputs_reused for trace in self._trace_cache.values())

    @property
    def cache_hits(self) -> int:
        return self.node_outputs_reused

    @property
    def cache_misses(self) -> int:
        return self.nodes_evaluated
