# Limitations

- Component attribution is **software counterfactual attribution** over the executable pipeline supplied to DeciShift.
- It does not establish external or real-world causal effects.
- Shapley attribution itself is not claimed as novel.
- Results depend on the supplied historical records and their representativeness.
- Replay does not prove production safety or future behavior.
- Cohort summaries and Wilson intervals are descriptive; DeciShift does not claim statistical significance from them.
- A Decision Contract pass is not proof of compliance, fairness, safety, correctness, or deployment suitability.
- Tamper-evident hashes do not prove signer identity.
- Outcome analysis, when used, reflects observed historical labels and not causal deployment impact.
- Unstable component identity weakens cross-process reproducibility and is rejected only when strict mode is enabled.

DeciShift intentionally remains CPU-first, local-first and framework-agnostic; it is not a hosted monitoring service, model registry, feature store, training framework or general MLOps platform.
