---
layout: default
title: DeciShift — ML Behavioral Regression Testing
description: Open-source ML behavioral regression testing and decision-change analysis for comparing machine-learning system versions before deployment.
---

# DeciShift

**Behavioral regression testing for ML decision systems.**

DeciShift is an open-source, CPU-first framework for comparing two executable versions of a machine-learning decision system before deployment. It shows **which final operational decisions changed**, **which action transitions occurred**, **which versioned components contributed**, and whether observed changes stay inside explicit **Decision Contracts**.

[GitHub repository](https://github.com/sauravsingla/DeciShift) · [PyPI](https://pypi.org/project/decishift/) · [Discussions](https://github.com/sauravsingla/DeciShift/discussions) · [Help test v0.3.1](https://github.com/sauravsingla/DeciShift/issues/13)

## Why it exists

A candidate model can improve aggregate accuracy or AUC while downstream operational actions still change because models interact with features, policies, thresholds, and rules.

```text
baseline + candidate
        ↓
execute both on the same records
        ↓
compare final actions and transitions
        ↓
software-counterfactual attribution
        ↓
verify saved evidence
        ↓
Decision Contract → PASS / BLOCK
```

DeciShift complements experiment tracking, model registries, evaluation/monitoring, orchestration, and CI/CD rather than replacing them.

## Public reproducible example

The Digits/XGBoost case study uses a fixed held-out split of **719 public records**:

- accuracy: **90.26% → 91.79%**
- ROC AUC: **0.9429 → 0.9472**
- final actions changed: **30 / 719 (4.17%)**
- global 5% shift contract: **PASS**
- `actual_digit=6` cohort: **9.41%** shift against an 8% limit → **BLOCK**

This is descriptive evaluation evidence, not a claim of production safety, fairness, compliance, or real-world causality.

[Read the evaluation study](evaluation-study.md)

## Start here

```bash
python -m pip install --upgrade decishift==0.3.1
decishift demo --rows 1000 --no-save
```

From a source checkout:

```bash
decishift graph examples/triage/flow.yaml
decishift compare examples/triage/flow.yaml
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

[Getting started](getting-started.md)

## Documentation map

- [Architecture](architecture.md) — execution abstractions, evidence layers, trust boundaries
- [Decision flows](decision-flows.md) — structured row-aligned DAGs
- [Flow attribution](flow-attribution.md) — exact/sampled software-counterfactual attribution
- [Decision Contracts](decision-contracts.md) — deterministic release limits and exit codes
- [Evidence integrity](evidence-integrity.md) — manifests and SHA-256 verification
- [Public API reference](api-reference.md) — generated top-level import inventory
- [Compatibility](compatibility.md) — Python/OS/dependency verification matrix
- [Failure modes](failure-modes.md) — explicit negative-path behavior
- [Related categories](comparisons.md) — complementary tooling and information matrix
- [Community](community.md) — independent validation and contribution paths
- [Roadmap](../ROADMAP.md) — direction and non-goals

## Current integration directions

- **MLflow** — proposed decision-level release-gating example: [issue #15](https://github.com/sauravsingla/DeciShift/issues/15)
- **ZenML** — proposed pipeline integration example: [issue #16](https://github.com/sauravsingla/DeciShift/issues/16)

These are active integration directions, not claims of built-in production integrations today.

## Join the project

The most valuable current contribution is an **independent reproducibility run** on a different environment. Successful reproductions, failures, installation friction, and counterexamples are all useful.

[Community and contribution paths](community.md) · [Independent testing issue](https://github.com/sauravsingla/DeciShift/issues/13) · [GitHub Discussions](https://github.com/sauravsingla/DeciShift/discussions)

---

**Keywords:** ML behavioral regression testing, machine learning regression testing, decision-level model evaluation, MLOps release gating, model change impact analysis, ML decision comparison, model governance, behavioral testing, decision drift, machine learning testing.
