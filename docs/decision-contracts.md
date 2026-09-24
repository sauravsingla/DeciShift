# Decision Contracts

A Decision Contract declares limits chosen by the user.

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
    overrides:
      region=North:
        max_flip_rate: 0.08
```

Evaluate a saved run:

```bash
decishift gate RUN_ID --contract examples/decision-contract.yaml
```

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | contract passed |
| 2 | configuration/usage error |
| 10 | contract violated |
| 11 | insufficient evidence |
| 12 | evidence integrity failure |

A passing contract is not proof that a system is safe, fair, compliant, correct, or suitable for deployment.
