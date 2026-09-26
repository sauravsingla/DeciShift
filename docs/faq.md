---
layout: default
title: DeciShift FAQ — ML Behavioral Regression Testing
description: Answers to common questions about ML behavioral regression testing, decision-level model comparison, MLOps release gating, Decision Contracts, attribution, monitoring, MLflow, and ZenML.
---

# DeciShift FAQ

## What is ML behavioral regression testing?

ML behavioral regression testing compares the operational behavior of two machine-learning system versions over the same evaluation records. Instead of looking only at aggregate model metrics, it asks which individual final actions changed and how those transitions are distributed.

DeciShift focuses on executable decision systems where features, models, thresholds, policies, and rules can all influence the final action.

## Why are model metrics alone not enough?

A candidate model can improve accuracy, AUC, calibration, or another aggregate metric while still changing individual downstream actions after thresholds, policies, and rules are applied.

That does not automatically mean the candidate is bad. It means the release decision may need both **model-level evaluation** and **decision-level behavioral comparison**.

## How is DeciShift different from model monitoring?

Monitoring usually observes data or system behavior over time. DeciShift compares two executable versions directly over aligned evaluation records before or during a release workflow.

The tools are complementary: monitoring can tell you that something changed; DeciShift is designed to show **which final decisions changed between versions** and map those changes to versioned software components.

## How is DeciShift different from MLflow?

MLflow can track runs, metrics, artifacts, and model/version metadata. DeciShift can sit downstream of that information: select a baseline and candidate, execute both through the decision layer, compare final actions, and apply a Decision Contract.

A public integration example is proposed in [issue #15](https://github.com/sauravsingla/DeciShift/issues/15).

## Can DeciShift work with ZenML?

DeciShift is framework-agnostic and can be inserted as a decision-comparison and release-gating step in an orchestrated ML workflow. A public ZenML example is proposed in [issue #16](https://github.com/sauravsingla/DeciShift/issues/16).

## What is a Decision Contract?

A Decision Contract is a deterministic set of user-declared limits over observed decision-change evidence. Contracts can constrain overall action-shift rates and more specific transition, action, or cohort behavior.

A contract `PASS` means the supplied evidence stayed within those declared limits. It does not prove safety, fairness, compliance, correctness, or production fitness.

## Does DeciShift perform causal inference?

No. DeciShift uses **software-counterfactual attribution** inside the executable decision system. The attribution describes how versioned software components contribute to observed output changes under hybrid executions. It does not establish real-world causal effects.

## Can DeciShift detect cohort-specific changes?

Yes. Cohort analysis can reveal cases where the global action-shift rate is small but a particular configured cohort changes more substantially.

## Does DeciShift require a GPU or hosted service?

No. The project is CPU-first, local-first, and designed to run without a hosted service, model registry, database, Docker runtime, LLM API, or telemetry platform.

## What data can I use to try it?

The repository includes synthetic and public-data examples, including scikit-learn Wine and Digits/XGBoost workflows. No proprietary data is required to evaluate the basic behavior.

## How can I help validate DeciShift?

Run one of the public examples independently and report your operating system, Python version, runtime, decision-shift result, attribution result, and any discrepancy or usability problem.

[Independent reproducibility testing — issue #13](https://github.com/sauravsingla/DeciShift/issues/13)
