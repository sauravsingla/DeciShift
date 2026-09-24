# Approximation uncertainty

`approximate_attribution` tracks Monte Carlo variability online with Welford's numerically stable algorithm. It does not retain every permutation sample.

For each record/component it can report:

- score and decision contribution;
- standard error;
- configurable normal confidence interval (95% by default);
- permutations used;
- an `uncertain` flag when the configured interval-width warning threshold is exceeded.

`statistics.NormalDist` is used for the normal critical value, avoiding a SciPy dependency solely for confidence intervals.

The interval describes uncertainty from **finite permutation sampling**. It does not represent uncertainty about the historical data-generating process, model estimation, deployment effects, or real-world causality.
