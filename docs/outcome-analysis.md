# Optional outcome analysis

When a combined config supplies:

```yaml
outcome_column: actual_outcome
```

DeciShift describes observed binary correctness for baseline and candidate, including accuracy, precision/recall where defined, correctness among directional flips, net corrected/newly incorrect decisions, and all four correctness transitions.

The primary transition table is:

- baseline correct -> candidate correct;
- baseline correct -> candidate wrong;
- baseline wrong -> candidate correct;
- baseline wrong -> candidate wrong.

No outcome column is required for normal operation. Historical correctness is observational evidence and must not be interpreted as proof of causal impact in production.
