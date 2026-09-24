# Attribution

DeciShift preserves the v0.1 public APIs:

```python
exact_attribution(...)
approximate_attribution(...)
pairwise_interactions(...)
```

Exact attribution enumerates all subsets of changed components (subject to `max_components`) and computes Shapley contributions to calibrated score and final decision.

Approximate attribution samples component permutations. Each sampled permutation telescopes from the baseline hybrid to the candidate hybrid; contributions are averaged by component and record.

`AttributionDiagnostics` reports efficiency residuals:

```text
sum(component contributions) - (candidate output - baseline output)
```

for calibrated score and final decision. Exact attribution should be numerically near zero. Approximate runs also expose their Monte Carlo uncertainty.

Pairwise interactions remain baseline anchored:

```text
f({A,B}) - f({A}) - f({B}) + f({})
```

An interaction-only flip means neither component alone flips the baseline decision while the pair does.

These are executable-software counterfactual quantities, not external causal effects.
