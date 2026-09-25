from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from decishift.core.pipeline import DecisionPipeline, PipelineTrace
from decishift.evidence import fingerprint_dataframe


@dataclass
class HybridReplayCache:
    """Memoize hybrid traces and version-keyed intermediate component outputs.

    Cache entries are bound to the current input-frame content fingerprint. If a
    caller mutates ``records`` between evaluations, the changed content produces
    a different cache namespace rather than reusing stale traces or intermediates.
    ``DecisionPipeline.evaluate`` also isolates the caller frame and defensively
    copies cached intermediate values before handing them to user components.
    """

    baseline: DecisionPipeline
    candidate: DecisionPipeline
    records: pd.DataFrame
    _cache: dict[tuple[str, frozenset[str]], PipelineTrace] = field(default_factory=dict)
    _intermediate_cache: dict[tuple[str, ...], object] = field(default_factory=dict)

    def evaluate(self, candidate_components: set[str] | frozenset[str]) -> PipelineTrace:
        records_fingerprint = fingerprint_dataframe(self.records)
        key = (records_fingerprint, frozenset(candidate_components))
        if key not in self._cache:
            self._cache[key] = self.baseline.hybrid(self.candidate, key[1]).evaluate(
                self.records,
                cache=self._intermediate_cache,
            )
        return self._cache[key]

    @property
    def evaluations(self) -> int:
        return len(self._cache)

    @property
    def intermediate_entries(self) -> int:
        return len(self._intermediate_cache)
