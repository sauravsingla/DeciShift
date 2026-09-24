# Decision Contracts

A Decision Contract declares limits chosen by the user. DeciShift never invents acceptable thresholds.

## Existing binary contracts

All v0.2 fields remain supported:

```yaml
contract:
  max_decision_shift_rate: 0.05
  max_0_to_1_rate: 0.04
  max_1_to_0_rate: 0.03
  attribution:
    require_reproducible_identity: true
    max_score_efficiency_mae: 0.001
  approximate_attribution:
    max_decision_ci_width: 0.10
  cohorts:
    min_size: 100
    max_flip_rate: 0.12
```

## Multi-action flow contracts

DecisionFlow adds transition/action rules without assigning numeric order to categorical actions:

```yaml
contract:
  max_decision_shift_rate: 0.05
  transitions:
    "monitor->inspect":
      max_rate: 0.04
      max_count: 500
    "inspect->service":
      max_rate: 0.01
  candidate_actions:
    service:
      max_rate: 0.10
  attribution:
    require_reproducible_identity: true
    max_efficiency_mae: 0.001
  cohorts:
    min_size: 100
    max_action_shift_rate: 0.12
```

Transition/action rules support `max_rate` and/or `max_count`. Flow cohorts use `max_action_shift_rate`; the historical `max_flip_rate` remains available for binary pipelines and is accepted as a backwards-friendly shift limit when a reused contract is applied to a flow.

Evaluate saved evidence:

```bash
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

Exit codes are unchanged:

| Code | Meaning |
|---:|---|
| 0 | contract passed |
| 2 | configuration/usage error |
| 10 | contract violated |
| 11 | insufficient evidence |
| 12 | evidence integrity failure |

A Decision Contract PASS means only that the user-declared checks passed against the saved DeciShift evidence. It is not proof that a system is safe, fair, compliant, correct, causal, or suitable for deployment.
