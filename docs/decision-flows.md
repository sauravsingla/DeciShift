# Decision flows

`DecisionFlow` is the v0.3 advanced abstraction for structured decision systems. `DecisionPipeline` remains the backwards-compatible linear binary API.

A flow is a deterministic DAG evaluated over one immutable historical `pandas.DataFrame`:

```text
sensor_features
      │
      ├── failure_risk_model ──┐
      │                        ├── triage_policy → safety_rules → final_action
      └── downtime_model ──────┘
```

DeciShift remains a decision-change analysis library, not a generic workflow engine. Nodes cannot fetch remote state through DeciShift, there is no scheduler/server/database, and execution stays local and row aligned.

## Nodes

```python
from decishift import DecisionFlow, DecisionNode

flow = DecisionFlow(
    nodes=[
        DecisionNode(
            name="risk_model",
            component=risk_model,
            depends_on=("features",),
            version="risk_model_v13",
            group="predictive_models",
        ),
    ],
    final_node="final_action",
)
```

A component normally exposes:

```python
def run(records, inputs):
    ...
```

`records` is the original source DataFrame. `inputs` maps declared dependency names to their row-aligned outputs. Lightweight records-only callables are also accepted. User exceptions are wrapped with node context; an exception raised *inside* a component is never reinterpreted as a calling-convention mismatch.

## Validation

Before execution DeciShift checks:

- unique node names;
- all dependencies exist;
- no self-dependency;
- no cycles;
- the configured final node exists and is terminal;
- deterministic topological ordering;
- row count and pandas index identity/order at every node.

DAG handling uses Python standard-library data structures; NetworkX is not required.

## Caching

Node cache keys contain the node execution identity plus dependency lineage. If one node changes, that node and descendants are recomputed while unaffected upstream/parallel branches can be reused. Reports expose node evaluations, reused outputs, cache hits and misses.

Runtime object identity may be used only as an ephemeral in-process discriminator for an unstable component. It is never serialized as reproducible evidence.

## Topology compatibility

Baseline and candidate flows are attribution-compatible only when they have:

- the same node names;
- the same dependency edges;
- the same final node.

Node implementations/versions may differ.

When topology changes, DeciShift can still execute each complete flow independently and compare observed final actions. It does **not** construct invalid hybrid graphs. Evidence reports `topology_compatible: false` and `attribution_status: unsupported topology change`, together with added/removed nodes, changed dependencies and changed final-node information.

## Canonical topology identity

Topology serialization sorts node names, dependencies and stable metadata before SHA-256 hashing. The digest therefore does not depend on Python dictionary insertion order.

## YAML

```yaml
mode: flow
data:
  path: history.csv
id_column: record_id
baseline:
  final_node: final_action
  nodes:
    features:
      factory: myproject.components:FeaturesV1
      version: features_v1
      depends_on: []
    risk_model:
      factory: myproject.models:RiskModelV1
      version: risk_model_v1
      depends_on: [features]
      group: predictive_models
    final_action:
      factory: myproject.policy:FinalActionV1
      version: final_action_v1
      depends_on: [risk_model]
candidate:
  # same topology for node attribution; implementations/versions may differ
```

Factories and objects are imported from the local Python environment. DeciShift does not perform remote artifact resolution.

Inspect a flow without executing components:

```bash
decishift graph examples/triage/flow.yaml
decishift graph examples/triage/flow.yaml --format mermaid
```
