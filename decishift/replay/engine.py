from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from decishift.core.pipeline import DecisionPipeline, PipelineTrace


@dataclass
class HybridReplayCache:
    """Memoize hybrid traces and version-keyed intermediate component outputs.

    The cache is scoped to one immutable input frame. Dependency-aware keys let
    hybrids reuse unchanged upstream work while recomputing changed components
    and their descendants.
    """

    baseline: DecisionPipeline
    candidate: DecisionPipeline
    records: pd.DataFrame
    _cache: dict[frozenset[str], PipelineTrace] = field(default_factory=dict)
    _intermediate_cache: dict[tuple[str, ...], object] = field(default_factory=dict)

    def evaluate(self, candidate_components: set[str] | frozenset[str]) -> PipelineTrace:
        key = frozenset(candidate_components)
        if key not in self._cache:
            self._cache[key] = self.baseline.hybrid(self.candidate, key).evaluate(
                self.records, cache=self._intermediate_cache
            )
        return self._cache[key]

    @property
    def evaluations(self) -> int:
        return len(self._cache)

    @property
    def intermediate_entries(self) -> int:
        return len(self._intermediate_cache)
