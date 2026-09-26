---
layout: default
title: Getting Started with DeciShift
description: Install DeciShift and run ML behavioral regression testing, decision comparison, evidence verification, and Decision Contract release gating locally on CPU.
---

# Getting Started with DeciShift

DeciShift is designed to make **ML behavioral regression testing** runnable locally without a hosted platform, GPU, database, or model registry.

## Install

```bash
python -m pip install --upgrade decishift==0.3.1
```

Run the bundled demo:

```bash
decishift demo --rows 1000 --no-save
```

## Run the DecisionFlow trust path

From a source checkout:

```bash
python -m pip install -e ".[dev]"

decishift graph examples/triage/flow.yaml
decishift compare examples/triage/flow.yaml
```

After a saved run is produced:

```bash
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

The workflow separates four questions:

1. **Execution:** can both versions run over the same aligned records?
2. **Behavioral comparison:** which final actions changed?
3. **Attribution:** which versioned components contributed to those software-output changes?
4. **Release gating:** do observed global, transition, action, or cohort shifts remain within the configured Decision Contract?

## Public examples

### Wine + scikit-learn

The public Wine example uses a trained logistic-regression model and a three-action downstream policy. It is useful for testing feature-only, model-only, policy-only, rule-only, and combined changes.

Repository path: [`examples/public_wine_sklearn/`](https://github.com/sauravsingla/DeciShift/tree/main/examples/public_wine_sklearn)

### Digits + XGBoost

The Digits example uses a trained XGBoost model and exposes a case where candidate metrics improve while individual operational actions still change.

Repository path: [`examples/public_digits_xgboost/`](https://github.com/sauravsingla/DeciShift/tree/main/examples/public_digits_xgboost)

[Read the public evaluation study](evaluation-study.md)

## What a PASS means

A Decision Contract `PASS` means only that the observed evidence remained inside limits declared by the user. It does **not** prove production safety, fairness, compliance, correctness, or real-world causal validity.

Likewise, software-counterfactual attribution explains changes inside the executable software system. It does not establish real-world causality.

## Try it and report your environment

Independent reproductions are particularly valuable. If you run DeciShift, please report your OS, Python version, example, runtime, observed shift, and any discrepancy in the public testing issue:

[Independent reproducibility testing for v0.3.1](https://github.com/sauravsingla/DeciShift/issues/13)
