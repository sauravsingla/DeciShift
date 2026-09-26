---
layout: default
title: DeciShift — ML Behavioral Regression Testing
description: Open-source ML behavioral regression testing and decision-change analysis for comparing machine-learning system versions before deployment.
---

# DeciShift

**Behavioral regression testing for ML decision systems.**

DeciShift is an open-source, CPU-first framework for comparing two executable versions of a machine-learning decision system before deployment. It shows **which final operational decisions changed**, **which action transitions occurred**, **which versioned components contributed**, and whether the observed changes stay inside explicit **Decision Contracts**.

[View the GitHub repository](https://github.com/sauravsingla/DeciShift) · [Install from PyPI](https://pypi.org/project/decishift/) · [Join Discussions](https://github.com/sauravsingla/DeciShift/discussions) · [Help test v0.3.1](https://github.com/sauravsingla/DeciShift/issues/13)

## Why DeciShift exists

A candidate model can improve aggregate accuracy or AUC while the downstream system still changes individual actions because models interact with features, thresholds, policies, and rules.

DeciShift focuses on the decision-level regression question that sits between model evaluation and release:

```text
baseline/candidate evaluation
        ↓
execute both decision-system versions
        ↓
DeciShift behavioral comparison + attribution
        ↓
Decision Contract
        ↓
release PASS / BLOCK
```

It is designed to complement experiment tracking, model registries, monitoring, orchestration, and CI/CD rather than replace them.

## What it answers

1. **Which records changed action?**
2. **Which action transitions occurred?**
3. **Which features, models, policies, thresholds, or rules contributed to those changes?**
4. **Do global and cohort-level shifts remain inside declared release limits?**

## Public reproducible example

The bundled Digits/XGBoost case study compares a baseline and candidate decision system over a fixed held-out split of **719 public records**.

- Accuracy: **90.26% → 91.79%**
- ROC AUC: **0.9429 → 0.9472**
- Final actions changed: **30 / 719 (4.17%)**
- Global 5% shift contract: **PASS**
- `actual_digit=6` cohort: **9.41%** shift against an 8% limit → **BLOCK**

This is descriptive evidence from a public evaluation split, not a claim of production safety, fairness, compliance, or real-world causality.

[Read the evaluation study](evaluation-study.md)

## Install and run in about a minute

```bash
python -m pip install --upgrade decishift==0.3.1
decishift demo --rows 1000 --no-save
```

From a source checkout, the core trust path is:

```bash
decishift graph examples/triage/flow.yaml
decishift compare examples/triage/flow.yaml
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

[Getting started](getting-started.md)

## Where DeciShift fits

| Layer | Typical responsibility | DeciShift focus |
|---|---|---|
| Experiment tracking / model registry | Runs, metrics, artifacts, versions | Compare downstream decisions produced by two executable versions |
| Evaluation / monitoring | Data and model quality over time | Record-level decision/action transitions between versions |
| CI/CD / orchestration | Execute validation and release workflows | Deterministic Decision Contract PASS/BLOCK signal |
| Feature stores | Serve/version features | Test whether feature changes alter operational decisions unexpectedly |

[Explore use cases](use-cases.md)

## Current integration directions

- **MLflow** — proposed decision-level release-gating example: [issue #15](https://github.com/sauravsingla/DeciShift/issues/15)
- **ZenML** — proposed pipeline integration example: [issue #16](https://github.com/sauravsingla/DeciShift/issues/16)

These are active integration directions rather than claims of built-in production integrations today.

## Built for reproducibility

- CPU-first and local-first
- No hosted service required
- Public scikit-learn/XGBoost examples
- Exact and sampled software-counterfactual attribution
- Cohort-level decision shift checks
- Tamper-evident SHA-256 evidence verification
- Deterministic Decision Contracts
- Apache-2.0 license
- Python 3.11, 3.12, and 3.13 CI

## Join the project

The most useful contribution right now is an **independent reproducibility run** on a different environment. Successful reproductions, failures, installation friction, and counterexamples are all useful.

[Community and contribution paths](community.md) · [Independent testing issue](https://github.com/sauravsingla/DeciShift/issues/13) · [GitHub Discussions](https://github.com/sauravsingla/DeciShift/discussions)

---

**Keywords:** ML behavioral regression testing, machine learning regression testing, decision-level model evaluation, MLOps release gating, model change impact analysis, ML decision comparison, model governance, behavioral testing, decision drift, machine learning testing.
