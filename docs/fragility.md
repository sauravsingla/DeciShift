# Decision Fragility

Decision Fragility summarizes proximity to the threshold boundary:

```text
fragility_margin = abs(calibrated_score - threshold)
```

The default summary includes fractions within 0.01 and 0.05 of a boundary, median minimum absolute margin, p10 margin, and changed-decision classifications:

- boundary-crossing change;
- far-from-boundary change;
- rule-forced change.

Boundary bands are configurable. Fragility is a threshold-proximity diagnostic, not a causal robustness guarantee.
