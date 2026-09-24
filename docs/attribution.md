# Attribution

DeciShift preserves the v0.1/v0.2 `DecisionPipeline` APIs:

```python
exact_attribution(...)
approximate_attribution(...)
pairwise_interactions(...)
```

and adds separate DecisionFlow APIs under `decishift.flow`.

## Linear DecisionPipeline attribution

Exact attribution enumerates subsets of changed components (subject to `max_components`) and computes Shapley contributions to calibrated score and binary final decision.

Approximate attribution samples component permutations and keeps streaming moments instead of all samples.

### Efficiency is not Monte Carlo convergence

`AttributionDiagnostics` now separates:

- `efficiency_valid` — contributions add back to the observed baseline-to-candidate output change within tolerance;
- `sampling_precision_sufficient` — permutation-sampling confidence intervals satisfy the configured width target;
- `sampling_converged` — precision is sufficient and running estimates are stable across sampling batches.

The legacy `converged` field remains for backwards compatibility. For approximate attribution it requires both valid efficiency and genuine sampling convergence. Shapley efficiency alone is never treated as Monte Carlo convergence.

See [adaptive-attribution.md](adaptive-attribution.md) for fixed and adaptive permutation modes.

## DecisionFlow attribution

Categorical actions are never numerically subtracted. Instead DecisionFlow uses explicit numeric row-level `AttributionTarget` games.

Built-ins:

- `candidate_action_support`: 1 when a hybrid produces exactly the candidate action;
- `change_from_baseline`: 1 when a hybrid leaves the baseline action.

These are intentionally different games. A hybrid can leave the baseline action without matching the final candidate action.

Changed nodes are players by default. Explicitly declared groups may be used instead; DeciShift does not infer groups automatically.

Exact flow attribution enumerates all `2^K` player subsets up to a configured safety limit. Approximate flow attribution uses the same streaming/adaptive permutation machinery.

## Interactions

For numeric targets pairwise interaction remains baseline anchored:

```text
f({A,B}) - f({A}) - f({B}) + f({})
```

For categorical flows DeciShift additionally marks `interaction_only_transition` when neither player alone changes the baseline action but their pair does, and records the resulting action transition directly.

## Topology constraint

Flow node/group hybrid attribution requires the same node names, dependency edges and final node in baseline and candidate. If topology changes, complete flows may still be executed and observed final actions compared, but hybrid node attribution is reported as unsupported rather than fabricated.

All attribution in DeciShift is **software counterfactual attribution** over the supplied executable system and historical records. It does not establish external or real-world causal effects.
