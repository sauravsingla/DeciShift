---
layout: default
title: DeciShift Use Cases — ML Decision Regression Testing and Release Gating
description: Practical use cases for ML behavioral regression testing, decision-level model comparison, cohort shift detection, model governance, and MLOps release gates.
---

# DeciShift Use Cases

DeciShift is most useful when a machine-learning system produces a downstream **operational action**, not just a score or aggregate metric.

Typical actions include `approve / review / decline`, `route / escalate`, `monitor / inspect / service`, or any other categorical decision produced by a versioned decision flow.

## 1. Model metrics improve but decisions regress

A candidate model can improve aggregate accuracy, AUC, calibration, or another evaluation metric while changing important individual actions after thresholds, policies, and rules are applied.

DeciShift compares final actions record by record so the release decision is not based only on aggregate model metrics.

## 2. Small global change, large cohort change

A system can show a low overall action-shift rate while one cohort changes substantially more than the population average.

Decision Contracts can apply both global and cohort-level limits so a globally small change does not automatically hide a concentrated shift.

## 3. Threshold, policy, and rule regression testing

Many production decisions are influenced by more than the predictive model. DeciShift can compare versioned features, models, thresholds, policies, and rules as parts of one executable decision system.

This supports regression testing when:

- a threshold moves;
- a manual-review policy changes;
- a business rule is added or removed;
- a feature transformation changes;
- multiple components change together.

## 4. MLOps release gating

A typical pattern is:

```text
train / track / evaluate
        ↓
baseline + candidate executable versions
        ↓
DeciShift comparison
        ↓
Decision Contract
        ↓
PASS / BLOCK consumed by CI/CD or orchestration
```

This makes the decision-level comparison usable as a deterministic release signal rather than only a notebook report.

## 5. Model governance evidence

Saved DeciShift evidence records decision transitions, structural information, attribution metadata, and Decision Contract results. Evidence integrity checks are tamper-evident SHA-256 consistency checks.

Integrity verification does **not** establish signer identity, authenticity of the original inputs, regulatory compliance, or real-world causality.

## 6. Feature-store change validation

A feature definition can remain schema-compatible while changing downstream decisions. A feature-store integration can compare the same evaluation entities under two feature-service/version configurations and then inspect action changes.

## 7. Experiment-tracker / model-registry integration

Experiment tracking and model registries identify runs, artifacts, metrics, and model versions. DeciShift can sit downstream of that metadata and test whether two selected versions produce materially different operational actions.

Proposed examples:

- [MLflow integration issue #15](https://github.com/sauravsingla/DeciShift/issues/15)
- [ZenML integration issue #16](https://github.com/sauravsingla/DeciShift/issues/16)

## 8. Independent reproducibility testing

A behavioral regression framework should itself be reproducible across environments. The project is actively collecting independent runs across operating systems and Python versions.

[Join the reproducibility effort](https://github.com/sauravsingla/DeciShift/issues/13)

## What DeciShift is not

DeciShift is not intended to replace:

- experiment tracking;
- model registries;
- feature stores;
- production monitoring platforms;
- generic workflow orchestrators;
- causal inference systems.

Its scope is deliberately narrower: **compare the final operational behavior of two executable ML decision-system versions and explain the observed software-output changes.**
