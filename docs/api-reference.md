# Public API reference

This page is generated from `decishift.__all__` by `python scripts/generate_api_reference.py`. 
It is an inventory of the stable top-level import surface, not a replacement for the conceptual guides.

| Symbol | Kind | Defined in |
|---|---|---|
| `ComponentIdentity` | class | `decishift.core.identity` |
| `DecisionPipeline` | class | `decishift.core.pipeline` |
| `PipelineTrace` | class | `decishift.core.pipeline` |
| `ComparisonResult` | class | `decishift.diff.compare` |
| `compare_pipelines` | function | `decishift.diff.compare` |
| `compare_predictions` | function | `decishift.diff.compare` |
| `DecisionFlow` | class | `decishift.flow.flow` |
| `DecisionNode` | class | `decishift.flow.node` |
| `DecisionOutput` | class | `decishift.flow.trace` |
| `FlowComparisonResult` | class | `decishift.flow.compare` |
| `FlowTrace` | class | `decishift.flow.trace` |
| `compare_flows` | function | `decishift.flow.compare` |
| `__version__` | value | `decishift.version` |

## Entry points

- Use `DecisionFlow`, `DecisionNode`, and `compare_flows` for structured row-aligned decision DAGs and categorical actions.
- Use `DecisionPipeline` and `compare_pipelines` for the backwards-compatible fixed binary pipeline path.
- See [Decision flows](decision-flows.md), [Flow attribution](flow-attribution.md), and [Failure modes](failure-modes.md) for semantics and trust boundaries.
