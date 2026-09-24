# Adaptive approximate attribution

DeciShift approximate attribution uses permutation Monte Carlo sampling and streaming moments. It does not retain every sampled permutation in memory.

## Separate diagnostics

Three ideas are deliberately distinct:

- `efficiency_valid`: component contributions add back to the observed baseline-to-candidate software-output change within the configured tolerance.
- `sampling_precision_sufficient`: the final permutation-sampling confidence intervals meet the configured width target.
- `sampling_converged`: the precision target is met **and** estimated contributions are stable across successive sampling batches.

Shapley efficiency is an additivity check. It is not evidence that Monte Carlo sampling converged.

The legacy `converged` field is retained for compatibility. For approximate attribution it is true only when efficiency is valid and the sampling convergence criteria are satisfied.

## Fixed-permutation mode

Existing code remains valid:

```python
approximate_attribution(
    baseline,
    candidate,
    records,
    permutations=256,
    seed=0,
)
```

Exactly 256 permutations are evaluated.

## Adaptive mode

Adaptive mode is enabled by supplying one or more adaptive options:

```python
approximate_attribution(
    baseline,
    candidate,
    records,
    min_permutations=64,
    max_permutations=2048,
    batch_size=32,
    target_ci_width=0.05,
    confidence_level=0.95,
    seed=0,
)
```

Sampling proceeds in batches. After the minimum number of permutations, DeciShift checks both confidence-interval precision and stability of the running contribution estimates across batches. It stops early only when both checks pass; otherwise it continues to `max_permutations`.

Reported sampling metadata includes:

- `permutations_used`
- `stopped_early`
- `sampling_precision_sufficient`
- `sampling_converged`
- `max_ci_width`
- `median_ci_width`
- `batch_stability_max_change`

Confidence intervals quantify **permutation-sampling uncertainty only**. They do not quantify causal uncertainty, deployment uncertainty, future-data uncertainty, or model correctness.
