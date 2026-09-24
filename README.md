# DeciShift

[![CI](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml/badge.svg)](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/decishift.svg)](https://pypi.org/project/decishift/)
[![Python versions](https://img.shields.io/pypi/pyversions/decishift.svg)](https://pypi.org/project/decishift/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22932948.svg)](https://doi.org/10.5281/zenodo.22932948)

**Explain why decisions changed between ML system versions.**

> Your candidate model improved AUC. But historical decisions changed. Why?

Git explains **which code changed**. Model monitoring explains **whether metrics changed**. DeciShift explains **which decisions changed and which components of the decision system contributed to those changes**.

DeciShift is a CPU-first, local-first Python framework for comparing two versions of a machine-learning decision pipeline. Its fundamental unit is the **decision change** — not the model, metric, code diff, or explanation artifact.

## Trustworthy decision-change evidence

v0.2.0 extends the original decision-diff engine with attribution uncertainty, efficiency diagnostics, stable component provenance, tamper-evident evidence bundles, Decision Contracts, optional outcome analysis, decision fragility, static HTML reports and CI-friendly exit codes.

An illustrative evidence shape looks like:

```text
Candidate AUC improved.

3,158 historical decisions changed.

Component attribution:
features       38.4%
model          31.7%
threshold      21.8%
rules           8.1%

Interaction-only flips: 26
Approximation uncertainty: acceptable
Evidence integrity: verified

Decision Contract:
BLOCK — priority cohort flip rate exceeded configured limit.
```

Those values are illustrative, not benchmark claims. Component attribution and pairwise interaction diagnostics are distinct outputs; pairwise interactions are not an additional additive share of the component-attribution total. Runtime results are computed from the supplied pipelines and records.

Model metrics answer:

> **Did predictive performance change?**

DeciShift answers:

> **Which decisions changed, why, how confidently, and whether the observed change satisfies your declared Decision Contract?**

## Five-minute demo

Install the current published release:

```bash
python -m pip install --upgrade decishift==0.2.0
decishift demo
```

Or install from this repository when developing against `main`.

The demo uses a synthetic **equipment-maintenance intervention** problem. It makes no network calls, uses no LLM API, requires no GPU, uploads no data, and runs on ordinary CPU hardware.

## Decision pipeline

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

The v0.1 public imports remain supported in v0.2.0:

```python
from decishift import DecisionPipeline, compare_pipelines, compare_predictions
from decishift.attribution import exact_attribution, approximate_attribution, pairwise_interactions
```

For executable pipelines:

```python
result = compare_pipelines(
    baseline,
    candidate,
    historical_records,
    id_column="record_id",
)
```

Each row records baseline/candidate score, threshold, margin, final decision, flip direction and score delta. Executable evidence can then support exact or approximate component attribution.

## Approximate attribution uncertainty

Approximate permutation attribution reports streaming Monte Carlo uncertainty without storing all samples:

```text
component       contribution   95% CI
features          +0.018       [+0.015,+0.021]
model             +0.011       [+0.008,+0.014]
threshold          0.000       [ 0.000, 0.000]
```

Per record/component evidence includes standard errors, confidence bounds and permutations used. Wide intervals are marked as uncertain. The interval describes permutation-sampling uncertainty, not external causal uncertainty.

## Attribution diagnostics

`AttributionDiagnostics` checks the Shapley efficiency residual for calibrated score and final decision:

```text
sum(component contributions)
-
(candidate output - baseline output)
```

Exact attribution should be numerically near zero. Approximate runs report MAE, maximum/p95 absolute residual, fraction above tolerance, convergence state and warnings.

## Reproducible component identity

DeciShift does not use runtime memory addresses as reproducible evidence. Identity resolution prefers:

1. explicit configured version/digest;
2. artifact SHA-256;
3. component `.version`;
4. deterministic source identity where safe;
5. otherwise an explicit `unstable` status.

Strict reproducibility mode can reject unstable identity. Reports surface `reproducible`, `partially_reproducible` or `unstable` rather than silently overstating provenance.

## Evidence bundles and verification

Every saved v0.2.0 run produces a local evidence bundle such as:

```text
.decishift/runs/<run_id>/
    manifest.json
    summary.json
    records.csv
    attribution.csv
    interactions.csv
    cohorts.csv
    report.md
    report.txt
    report.html
    fragility.csv   # when fragility evidence is available
    outcome.json    # when outcome analysis is configured
```

The core CSV/report files are written for every saved run; optional analysis artifacts are included only when that analysis is available.

Verify a saved bundle offline:

```bash
decishift verify RUN_ID
```

Verification recomputes declared SHA-256 artifact hashes and the manifest integrity root, reports missing/modified files and exits nonzero on failure. This is **tamper-evident integrity verification**, not signer authentication.

## Decision Contracts

A Decision Contract is a deterministic, user-authored policy for acceptable observed behavioral change:

```yaml
contract:
  max_decision_shift_rate: 0.05
  attribution:
    require_reproducible_identity: true
    max_score_efficiency_mae: 0.001
  approximate_attribution:
    max_decision_ci_width: 0.10
  cohorts:
    min_size: 100
    max_flip_rate: 0.12
```

Gate saved evidence:

```bash
decishift gate RUN_ID --contract examples/decision-contract.yaml
```

Exit codes distinguish pass (`0`), contract violation (`10`), insufficient evidence (`11`), integrity failure (`12`) and usage/configuration errors (`2`). DeciShift does **not** invent normative limits, and a passing contract does not prove safety, fairness, compliance or correctness.

## Safer cohorts

Automatic cohort discovery excludes or warns on obvious identifiers, nearly unique columns, free text, high-cardinality values, prediction/decision columns and datetimes unless temporal cohorting is enabled. Cohort output includes coverage, decision-rate delta, Wilson interval for flip rate, global flip rate and excess flip rate.

Cohort analysis is descriptive; DeciShift does not claim statistical significance from it.

## Optional outcome analysis

With:

```yaml
outcome_column: actual_outcome
```

binary historical outcomes add baseline/candidate accuracy, precision/recall where defined, directional-flip correctness, net corrected/newly incorrect decisions and the four-way correctness transition table. Historical correctness does not establish causal production impact.

## Decision Fragility

For threshold-based pipelines, DeciShift summarizes:

```text
fragility_margin = abs(calibrated_score - threshold)
```

including fractions near the boundary, median/p10 margin and boundary-crossing, far-from-boundary or rule-forced changed decisions. Fragility is a threshold-proximity diagnostic, not a causal robustness guarantee.

## CLI

```bash
decishift demo

decishift compare config.yaml

decishift explain --run RUN_ID --id RECORD_ID

decishift report RUN_ID --format terminal
decishift report RUN_ID --format json
decishift report RUN_ID --format markdown
decishift report RUN_ID --format html

decishift verify RUN_ID

decishift gate RUN_ID --contract examples/decision-contract.yaml

decishift compare-runs RUN_A RUN_B
```

All commands work offline. The HTML report is self-contained, uses no CDN, requires no JavaScript and makes no network request.

## Predictions-only mode

If historical executable components are unavailable, DeciShift can analyze precomputed baseline/candidate scores, thresholds and decisions. It reports decision shifts, flip directions, margins and cohorts, but deliberately reports **insufficient evidence** for component attribution rather than inventing it.

## CPU-first and local-first

The core requires no GPU, LLM API, cloud service, external database, Docker, telemetry or network call. Model interfaces remain framework agnostic; optional adapters support scikit-learn, XGBoost and LightGBM without making them core dependencies.

## Benchmark

Run reproducible benchmarks locally:

```bash
python benchmarks/benchmark_cpu.py
```

The benchmark script measures 10,000 and 100,000 rows and records comparison, exact attribution, approximate attribution with uncertainty, evidence serialization and verification. No benchmark numbers are hard-coded into this README.

## Tests and quality checks

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest --cov=decishift
python -m build
python -m twine check dist/*
decishift demo --rows 1000 --no-save
```

CI runs these checks on Python 3.11, 3.12 and 3.13.

## Research positioning

DeciShift is related to model regression testing, behavioral diffing, slice analysis, ML monitoring, Shapley attribution, unit-change attribution and CI release gates.

It does **not** claim that Shapley attribution is novel. Its specific object of analysis is:

> **The discrete decision transition produced by a versioned structured decision pipeline.**

See `docs/comparisons.md` and `docs/limitations.md` for positioning and limitations.

## Privacy

DeciShift is local by default. The package contains no telemetry, data-upload path, remote model API, authentication service or network client in its core dependencies.

## License and citation

Apache-2.0. See [LICENSE](LICENSE). Citation metadata is provided in `CITATION.cff`.

The current **v0.2.0** archive is available at Zenodo DOI [`10.5281/zenodo.22934681`](https://doi.org/10.5281/zenodo.22934681). The persistent all-versions concept DOI is [`10.5281/zenodo.22932948`](https://doi.org/10.5281/zenodo.22932948).
