# Limitations

- Component/node attribution is **software counterfactual attribution** over the executable system supplied to DeciShift.
- It does not establish external or real-world causal effects.
- Structural impact is downstream graph reachability, not causal impact, and does not imply that historical behavior changed.
- Shapley attribution itself is not claimed as novel.
- Categorical actions are never numerically subtracted or silently assigned an ordinal scale.
- `candidate_action_support` and `change_from_baseline` are explicitly defined numeric attribution games; they answer different software-counterfactual questions.
- DecisionFlow node/group attribution in v0.3 requires compatible topology (same node names, dependency edges and final node). Topology-changing attribution is unsupported rather than approximated with invalid hybrids.
- Approximate-attribution confidence intervals describe permutation-sampling uncertainty only; they do not capture data-generating, deployment, causal or model uncertainty.
- Efficiency validity is an additivity check and is not, by itself, evidence of Monte Carlo convergence.
- Results depend on the supplied historical records and their representativeness.
- Replay does not prove production safety, future behavior or deployment suitability.
- Cohort summaries and Wilson intervals are descriptive; DeciShift does not claim statistical significance from them.
- A Decision Contract pass is not proof of compliance, fairness, safety, correctness, causality or deployment suitability.
- Tamper-evident hashes do not prove signer identity.
- Outcome/classification analysis for flows is run only when `actions_are_predictions: true` is explicitly configured. Historical correctness does not establish causal production impact.
- Decision fragility for categorical flows is reported only when a meaningful numeric margin is explicitly supplied by the final node.
- Unstable component identity weakens cross-process reproducibility and is rejected only when strict mode is enabled.
- Runtime memory identity may be used as an ephemeral in-process cache discriminator for unstable objects but is never serialized as reproducible evidence.

DeciShift intentionally remains CPU-first, local-first and framework-agnostic. It is not a hosted monitoring service, generic workflow engine, model registry, feature store, training framework, database service, Kubernetes/Kafka platform, LLM-agent evaluator or general MLOps platform.
