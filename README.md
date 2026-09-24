# DeciShift

[![PyPI version](https://img.shields.io/pypi/v/decishift.svg)](https://pypi.org/project/decishift/)
[![Python versions](https://img.shields.io/pypi/pyversions/decishift.svg)](https://pypi.org/project/decishift/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22932948.svg)](https://doi.org/10.5281/zenodo.22932948)

**Explain why decisions changed between ML system versions.**

> Your candidate model improved AUC. But historical decisions changed. Why?

Git explains **which code changed**. Model monitoring explains **whether metrics changed**. DeciShift explains **which decisions changed and which components of the decision system contributed to those changes**.

DeciShift is a CPU-first, local-first Python framework for comparing two versions of a machine-learning decision pipeline. Its fundamental unit is the **decision change** — not the model, metric, code diff, or explanation artifact.

## Five-minute demo

Install from PyPI:

```bash
python -m pip install decishift
decishift demo
```

To install the latest development version directly from GitHub:

```bash
python -m pip install git+https://github.com/sauravsingla/DeciShift.git
```

Or clone the repository for development:

```bash
git clone https://github.com/sauravsingla/DeciShift.git
cd DeciShift
python -m pip install -e .
decishift demo
```

The demo uses a synthetic **equipment-maintenance intervention** problem. It makes no network calls, uses no LLM API, requires no GPU, and uploads no data.

Typical output shape:

```text
DeciShift comparison
========================================================
Records evaluated             10,000
Unchanged                       ...
Changed                         ...
Decision shift rate             ...
0 -> 1 flips                    ...
1 -> 0 flips                    ...
Mean score delta                ...

Component attribution (share of absolute decision attribution)
--------------------------------------------------------
features                        ...
model                           ...
threshold                       ...
rules                           ...
```

Numbers are computed at runtime; this README intentionally does not publish fabricated benchmark or demo measurements.

## What is a decision system?

DeciShift models the operational path explicitly:

```text
Input data
  -> Feature transformation
  -> Predictive model
  -> Calibration
  -> Threshold / decision policy
  -> Deterministic rules
  -> Final decision
```

A candidate release can change any subset of those components. A model-only comparison can therefore miss decision changes caused by feature logic, calibration, thresholds, deterministic policy, or component interactions.

## Core API

```python
from decishift import DecisionPipeline, compare_pipelines
from decishift.attribution import exact_attribution, pairwise_interactions

baseline = DecisionPipeline(
    features=features_v1,
    model=model_v1,
    calibrator=calibrator_v3,
    threshold=0.72,
    rules=rules_v7,
    versions={
        "features": "features_v1",
        "model": "model_v1",
        "calibrator": "calibration_v3",
        "threshold": "0.72",
        "rules": "rules_v7",
    },
)

candidate = DecisionPipeline(
    features=features_v2,
    model=model_v2,
    calibrator=calibrator_v3,
    threshold=0.67,
    rules=rules_v8,
    versions={
        "features": "features_v2",
        "model": "model_v2",
        "calibrator": "calibration_v3",
        "threshold": "0.67",
        "rules": "rules_v8",
    },
)

result = compare_pipelines(baseline, candidate, historical_records, id_column="record_id")
result.attribution = exact_attribution(baseline, candidate, historical_records, result=result)
result.interactions = pairwise_interactions(baseline, candidate, historical_records, result=result)
```

For each record DeciShift retains:

- baseline and candidate score
- baseline and candidate threshold
- margin-to-threshold values
- baseline and candidate decision
- flip direction (`0->1`, `1->0`, or unchanged)
- per-component score and decision attribution when executable evidence is available

## Counterfactual component attribution

If `K` components changed, DeciShift can construct hybrid pipelines in which each changed component takes either its baseline or candidate version.

For small `K`, `exact_attribution(...)` enumerates all `2^K` subsets and computes Shapley contributions to:

1. calibrated score, and
2. final binary decision.

For larger `K`, `approximate_attribution(...)` uses deterministic permutation Monte Carlo.

Pairwise interaction analysis reports the baseline-anchored second-order effect:

```text
f({A,B}) - f({A}) - f({B}) + f({})
```

and explicitly flags an **interaction-only flip** when neither component alone changes the baseline decision but the pair does.

### Scientific honesty

These quantities are **counterfactual component attributions inside the executable software system**. They do not automatically establish real-world causal effects. DeciShift distinguishes:

- observed decision differences,
- counterfactual software-component attribution,
- statistical uncertainty,
- real-world causal effects.

When executable components or sufficient counterfactual evidence are unavailable, DeciShift reports **insufficient evidence** rather than inventing attribution.

## Predictions-only mode

Executable historical models are often unavailable. DeciShift can still analyze precomputed evidence:

```yaml
mode: predictions-only

data:
  path: predictions.csv

id_column: record_id
cohorts: [region, customer_type, age]
min_cohort_size: 50

columns:
  baseline_score: baseline_score
  candidate_score: candidate_score
  baseline_threshold: baseline_threshold
  candidate_threshold: candidate_threshold
  baseline_decision: baseline_decision
  candidate_decision: candidate_decision
```

```bash
decishift compare config.yaml
```

Predictions-only mode provides decision-shift rate, flip direction, score changes, threshold margins, cohorts, and reports. Component attribution remains unavailable unless the required executable or counterfactual evidence exists.

## Executable YAML configuration

```yaml
data:
  path: history.csv

id_column: record_id
cohorts: [region, segment]

baseline:
  features:
    factory: myproject.components:FeaturesV1
    version: features_v1
  model:
    factory: myproject.models:load_model_v12
    version: model_v12
  calibrator: identity
  threshold:
    value: 0.72
    version: threshold_072
  rules:
    factory: myproject.rules:RulesV7
    version: rules_v7

candidate:
  features:
    factory: myproject.components:FeaturesV2
    version: features_v2
  model:
    factory: myproject.models:load_model_v13
    version: model_v13
  calibrator: identity
  threshold:
    value: 0.67
    version: threshold_067
  rules:
    factory: myproject.rules:RulesV8
    version: rules_v8
```

All paths and imports are local. DeciShift performs no network resolution.

## CLI

```bash
decishift demo
decishift compare config.yaml
decishift compare baseline.yaml candidate.yaml
decishift explain --run RUN_ID --id RECORD_ID
decishift report RUN_ID --format markdown
decishift report RUN_ID --format json
```

Saved evidence is written locally under `.decishift/runs/` and includes record-level comparison CSV plus terminal, JSON, and Markdown reports.

## Cohort analysis

DeciShift automatically considers eligible columns for:

- categorical cohorts,
- numeric quantile bins,
- user-selected cohort columns,
- minimum cohort-size controls.

Set `cohorts: [...]` to constrain the analysis, or `auto_cohorts: false` to disable automatic cohort summaries.

Each cohort reports size, baseline decision rate, candidate decision rate, flip rate, dominant flip direction, and average score delta.

## Model interfaces

The core package is framework-agnostic. Models may expose:

```python
predict(X)
```

or:

```python
predict_proba(X)
```

or be wrapped as a callable. scikit-learn estimators work directly. Dedicated lightweight adapters are provided for scikit-learn, native XGBoost Boosters, and LightGBM Boosters; XGBoost and LightGBM remain optional extras. See `docs/adapters.md`. PyTorch and TensorFlow are not base dependencies.

## CPU-first and local-first

The core is designed to:

- require no GPU,
- require no LLM API,
- require no cloud service,
- require no external database,
- require no Docker,
- operate offline,
- make no telemetry or network calls,
- run on an ordinary Python 3.11+ laptop.

The v0.1 replay cache memoizes hybrid evaluations and version-keyed intermediate outputs. If an upstream component is unchanged, its cached output is reused; changing a component invalidates that component and its descendants for the hybrid being evaluated.

## Benchmark

Run reproducible CPU benchmarks locally:

```bash
python benchmarks/benchmark_cpu.py
```

The script measures 10,000 and 100,000 rows, reporting wall-clock time, Python-traced peak memory, changed decisions, and the configured number of hybrid evaluations. No benchmark numbers are hard-coded in the repository.

## Tests

```bash
python -m pip install -e '.[dev]'
pytest
```

The suite covers threshold-only, model-only, feature-only, rules-only, no-change, opposing changes, an interaction-only flip, exact attribution additivity, approximate-attribution convergence, deterministic reproducibility, predictions-only mode, missing evidence, and numeric/categorical cohorts.

## Research positioning

DeciShift is related to model regression testing, behavioral model diffing, slice regression analysis, ML observability, explainability, Shapley attribution, and unit-change attribution methods.

It does **not** claim Shapley attribution is novel. Its differentiating proposition is:

> **Version-aware attribution across the complete structured decision pipeline, with the final decision as the object of analysis.**

## Scope of v0.1

Deliberately included:

- replay of two local decision pipelines,
- decision diffs and margins,
- exact and approximate component attribution,
- pairwise interaction analysis,
- predictions-only mode,
- cohort analysis,
- individual saved-run explanation,
- terminal, JSON, and Markdown reports,
- scikit-learn-compatible generic model interface,
- optional XGBoost/LightGBM compatibility,
- reproducible CPU benchmark harness.

Deliberately not included:

- web dashboard,
- cloud service,
- telemetry,
- model training in the core,
- fabricated causal claims,
- mandatory heavyweight infrastructure.

## Privacy

DeciShift is local by default. The package contains no telemetry, data-upload path, remote model API, or network client in its core dependencies.

## Citation

For the archived **v0.1.0** release, use the version-specific DOI: **10.5281/zenodo.22932949**.

For references intended to resolve across all DeciShift versions, use the Zenodo concept DOI: **10.5281/zenodo.22932948**.

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0. See [LICENSE](LICENSE).
