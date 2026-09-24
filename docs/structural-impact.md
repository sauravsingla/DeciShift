# Structural impact

DecisionFlow reports deterministic graph reachability separately from observed historical behavior.

If `risk_model` changes in:

```text
risk_model -> ensemble -> policy -> rules -> final_action
```

DeciShift can state:

```text
Changed node: risk_model
Potential downstream nodes: ensemble, policy, rules, final_action
Structural descendants: 4
```

This is **structural impact**: nodes that are reachable downstream in the declared software graph.

It is not the same as observed action impact. Replaying historical records may show, for example:

```text
Historical records: 100,000
Observed action changes: 2,841
Observed action-shift rate: 2.84%
```

The first statement follows from graph structure. The second follows from executable historical replay. Neither statement by itself establishes a real-world causal effect.

## Why keep them separate?

A changed node may have many structural descendants while producing no changed action for the supplied records. Conversely, a small structurally localized change can alter many historical actions.

Reports therefore keep these ideas separate:

- **changed nodes** — stable provenance differs;
- **structural descendants** — downstream reachability;
- **observed action transitions** — final actions actually differed on supplied records;
- **software counterfactual attribution** — contribution under explicitly defined hybrid substitutions/targets.

Structural reachability never means “caused” and never implies that historical behavior changed.
