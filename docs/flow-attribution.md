# Flow attribution

Categorical final actions cannot be subtracted meaningfully. DecisionFlow attribution therefore operates on an explicitly defined **numeric row-level attribution target**.

## AttributionTarget protocol

Advanced callers may implement:

```python
class AttributionTarget(Protocol):
    name: str

    def evaluate(self, trace, baseline_trace, candidate_trace) -> np.ndarray:
        ...
```

The return value must be one finite numeric value per record. Non-numeric Shapley games are rejected.

## Candidate-action support

Built-in target:

```text
candidate_action_support
```

For hybrid system `S` and each historical record:

```text
v(S) = 1  if hybrid_action(S) == candidate_action
       0  otherwise
```

For a record whose baseline action differs from its candidate action, `v(empty)=0` and `v(all changed players)=1`. Shapley contributions therefore allocate movement toward **exactly the final candidate action**.

The output column is `candidate_action_contribution` in addition to the generic numeric `target_contribution`.

## Change from baseline

Built-in target:

```text
change_from_baseline
```

```text
v(S) = 1  if hybrid_action(S) != baseline_action
       0  otherwise
```

This allocates movement **away from the baseline action**. It is intentionally different from candidate-action support: a hybrid can leave the baseline action while producing a third action that is not the final candidate action.

## Players and groups

By default each changed node is one player. Nodes can be explicitly assigned a `group`, and callers may request grouped attribution. DeciShift never invents groups automatically.

## Exact attribution

Exact mode evaluates all `2^K` changed-node/group combinations up to a safety limit. It reports additive efficiency against the explicitly defined numeric target.

## Approximate attribution

Approximate mode samples player permutations and supports the adaptive stopping implementation described in [adaptive-attribution.md](adaptive-attribution.md). It persists:

- players and player-to-node mapping;
- attribution target;
- method;
- hybrid evaluations;
- permutations used;
- confidence level;
- sampling precision/convergence diagnostics.

Efficiency validity and Monte Carlo convergence are separate concepts.

## Pairwise interactions

For numeric targets DeciShift reports the baseline-anchored interaction:

```text
f({A,B}) - f({A}) - f({B}) + f({})
```

For categorical actions it additionally marks an `interaction_only_transition` when A alone retains the baseline action, B alone retains the baseline action, but A+B changes the action. The resulting categorical transition (for example `monitor->inspect`) is stored directly rather than numerically encoded.

## Topology changes

Node/group hybrid attribution in v0.3 requires compatible baseline/candidate topology: same node names, dependency edges and final node. If topology changes, DeciShift independently compares observed final actions but reports node attribution as unsupported. It never fabricates a hybrid graph across incompatible topology.

All of these quantities are **software counterfactual attribution** over the supplied executable system and historical records. They do not establish real-world causal effects.
