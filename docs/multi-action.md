# Multi-action decisions

`DecisionFlow` supports categorical final actions such as:

```text
monitor
inspect
service
```

or:

```text
approve
manual_review
decline
```

An action is a label, not an ordered numeric quantity. DeciShift therefore **never numerically subtracts categorical actions** and never invents an ordinal encoding for attribution.

## Final action requirements

The final node must produce exactly one action per historical record. Actions are non-null by default and must be scalar, hashable and JSON-serializable. Row count and pandas index order must remain aligned with the source records.

Categorical identity is type-aware. Python values that compare equal numerically but have different scalar types, such as boolean `True` and integer `1`, remain distinct DecisionFlow actions. Ordinary string actions stay readable in reports, while evidence keys disambiguate typed scalar labels when necessary.

A comparison records:

```text
record_id  baseline_action  candidate_action  changed  transition
101        monitor          inspect           true     monitor->inspect
102        inspect          inspect           false    unchanged
103        inspect          service           true     inspect->service
```

Reports include:

- total, changed and unchanged records;
- action-shift rate;
- baseline and candidate action distributions;
- transition counts and rates;
- a complete transition matrix;
- the most frequent changed transitions.

Binary `DecisionPipeline` output and terminology remain unchanged for backwards compatibility.

## DecisionOutput

Advanced final nodes may explicitly return numeric diagnostic evidence alongside a categorical action:

```python
from decishift import DecisionOutput

return DecisionOutput(
    action=actions,
    score=scores,
    margin=margins,
)
```

A margin is used for fragility only when the final node explicitly provides a meaningful numeric margin. DeciShift does not invent a margin for categorical actions, and it does not automatically interpret a multiclass top-1/top-2 score gap.

## Cohorts

Flow cohorts stay compact. They report:

- dimension and cohort;
- size and coverage;
- action-shift rate;
- global and excess action-shift rate;
- Wilson interval for the shift rate;
- most common changed transition.

Specific transition rates are added only when explicitly requested, avoiding hundreds of columns for high-cardinality action spaces.

## Outcomes

Operational actions are not assumed to be class predictions. Multiclass accuracy/confusion analysis runs only with explicit configuration:

```yaml
outcome:
  column: actual_class
  actions_are_predictions: true
```

When enabled, DeciShift reports accuracy, correctness transitions, confusion matrices, per-class support, precision and recall where defined. If the actions represent interventions rather than predicted labels, leave `actions_are_predictions` false or omit the outcome block.

Historical correctness does not establish causal production impact.
