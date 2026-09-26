# Rule-only DecisionFlow regression

This small offline example keeps a precomputed score unchanged and lowers one deterministic review threshold from `0.70` to `0.60`. The comparison therefore isolates a downstream rule change: `case-b` moves from `approve` to `review`, while the score node and its outputs remain identical.

Run it from the repository root:

```bash
python examples/rule_only_regression/run.py
```

The records and thresholds are synthetic. The output demonstrates a software regression-testing workflow and does not support real-world causal or policy-effect claims.
