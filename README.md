<h1 align="center">DeciShift</h1>

<p align="center">
  <b>Catch ML decision changes before they reach production.</b>
</p>

<p align="center">
  Behavioral regression testing for versioned ML decision systems: compare final actions, attribute software changes, preserve evidence, and gate releases on declared limits.
</p>

<p align="center">
  <a href="https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml"><img src="https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/decishift/"><img src="https://img.shields.io/pypi/v/decishift" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.11 | 3.12 | 3.13">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="https://huggingface.co/spaces/sauravsingla08/DeciShift"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-FFD21E" alt="Hugging Face Space"></a>
  <a href="https://huggingface.co/datasets/sauravsingla08/DeciShift-Decision-Change-Benchmark"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark%20Dataset-FFD21E" alt="Hugging Face Dataset"></a>
</p>

## What DeciShift does

A candidate model can improve accuracy or ROC AUC while the **actual operational decisions still change for individual records**. Thresholds, policies, rules, calibration, or model versions can all move the final action even when aggregate metrics look acceptable.

DeciShift compares a baseline and candidate decision system on the same records and answers four questions:

1. **Which final actions changed?**
2. **Which transitions occurred?**
3. **Which versioned software components account for the observed changes?**
4. **Do those changes violate a declared release contract?**

It is **CPU-first, local-first, offline-capable, and framework-agnostic**. Core operation requires no GPU, cloud service, database, hosted model registry, LLM/API, or telemetry backend.

### 30-second mental model

```text
Baseline system ─┐
                 ├─ execute on the same records
Candidate system ┘
                         │
                         ▼
              compare final actions
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        action transitions    changed components
              │                     │
              └──────────┬──────────┘
                         ▼
              verify saved evidence
                         │
                         ▼
                Decision Contract
                   PASS / BLOCK
```

> **Evidence boundary:** DeciShift performs software-behavior comparison and tamper-evident evidence verification. It does not by itself establish real-world causality, safety, fairness, compliance, source-data truth, or signer authenticity.

## Try it in under a minute

```bash
python -m pip install --upgrade decishift==0.3.1
decishift demo --rows 1000 --no-save
```

For the complete trust path from a source checkout:

```bash
git clone https://github.com/sauravsingla/DeciShift.git
cd DeciShift
python -m pip install -e ".[dev]"
decishift graph examples/triage/flow.yaml
decishift compare examples/triage/flow.yaml
# copy the emitted Run ID
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

The bundled real run demonstrates the same `graph → compare → verify → gate` flow:

![DeciShift real DecisionFlow run](https://github.com/sauravsingla/DeciShift/releases/download/v0.3.0/decishift-real-run-inline.gif)

## Explore the live decision-change explorer

The **[DeciShift Decision Change Explorer](https://huggingface.co/spaces/sauravsingla08/DeciShift)** is a browser-only visualization of generated public benchmark evidence. It surfaces changed actions, transition counts, software-counterfactual attribution, cohort action-shift rates, and Decision Contract interpretation for the public Digits/XGBoost case study.

The Space does not fabricate a separate demo dataset: it visualizes evidence generated from DeciShift's public evaluation path and links back to the benchmark dataset and PyPI package.

<p align="center">
  <a href="https://huggingface.co/spaces/sauravsingla08/DeciShift"><b>Open the live Hugging Face Space →</b></a>
</p>

## Measured public-data snapshot

The repository includes a machine-generated case study using the public scikit-learn **Digits** dataset with a trained XGBoost model and downstream actions `auto_not_8`, `manual_review`, and `auto_8`.

On a fixed held-out split of **719 records**:

| Measure | Baseline | Candidate / result |
|---|---:|---:|
| Model accuracy | 90.26% | **91.79%** |
| ROC AUC | 0.9429 | **0.9472** |
| Final actions changed | — | **30 / 719 (4.17%)** |
| Global action-shift limit | — | **PASS at 5%** |
| `actual_digit=6` cohort shift | — | **9.41% vs 8% limit** |
| Final Decision Contract | — | **BLOCK** |

The model improved on the standard predictive metrics, yet 30 individual operational actions still changed and one governed cohort exceeded its declared limit. That is the gap DeciShift is designed to expose.

Changed transitions in the same committed evaluation:

| Transition | Records |
|---|---:|
| `auto_not_8 → manual_review` | 13 |
| `manual_review → auto_8` | 8 |
| `manual_review → auto_not_8` | 9 |

Exact software-counterfactual attribution over the changed nodes assigned **48.86%** absolute attribution share to the model, **46.59%** to policy, and **4.55%** to rules.

The figures are tied to the committed machine-generated snapshot for source commit [`76bbf63abb7fdbba26739f3c8f96f94e0110bfc8`](docs/case-studies/digits-xgboost-76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md). They are descriptive evidence over this public evaluation split, not a production-safety, fairness, compliance, or causal claim.

## Why DeciShift?

- **Decision-level regression testing** — compare final operational actions, not only model metrics.
- **Transition analysis** — inspect changes such as `review → auto_approve` record by record.
- **Software-counterfactual attribution** — estimate how versioned executable components account for observed action changes.
- **Cohort-aware release gates** — enforce declared global and subgroup limits through Decision Contracts.
- **Tamper-evident run evidence** — preserve manifests and artifacts with integrity verification.
- **CI-friendly outcomes** — deterministic exit codes distinguish pass, violation, insufficient evidence, and integrity failure.

DeciShift complements experiment trackers, model registries, evaluation/monitoring systems, orchestration, and CI/CD rather than replacing them. See [`docs/comparisons.md`](docs/comparisons.md).

## Pick your path

| If you want to... | Start here |
|---|---|
| Understand the core workflow | [`docs/getting-started.md`](docs/getting-started.md) |
| Run the basic demo | `decishift demo --rows 1000 --no-save` |
| Compare a structured DecisionFlow | `decishift compare examples/triage/flow.yaml` |
| Verify saved evidence | `decishift verify RUN_ID` |
| Gate a release | `decishift gate RUN_ID --contract examples/triage/decision-contract.yaml` |
| Inspect the public case study | [`docs/case-studies/digits-xgboost-76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md`](docs/case-studies/digits-xgboost-76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md) |
| Understand attribution | [`docs/flow-attribution.md`](docs/flow-attribution.md) |
| Inspect failure behavior | [`docs/failure-modes.md`](docs/failure-modes.md) |

## Core scope

| Property | Current scope |
|---|---|
| Primary task | Behavioral regression testing for ML decision systems |
| Execution model | Baseline/candidate evaluation on the same row-aligned records |
| Structured path | `DecisionFlow` DAG with models, policies, rules, and categorical actions |
| Legacy path | Backwards-compatible `DecisionPipeline` binary flow |
| Attribution | Exact or sampled software-counterfactual attribution |
| Release governance | Global/cohort Decision Contracts with deterministic exit codes |
| Evidence | Saved manifests/artifacts with SHA-256 integrity root |
| Compute | CPU-first, local/offline core |
| Python | 3.11, 3.12, 3.13 |

## Technical depth

The README summarizes the main trust path; detailed definitions, algorithms, compatibility and failure behavior live in the documentation.

### DecisionFlow and comparison

`DecisionFlow` is a row-aligned DAG for structured executable decision systems:

```text
features
├── risk_model ──┐
└── value_model ─┼── policy -> rules -> action
```

It supports branching, merging, multiple models/policies/rules, and categorical actions without becoming a generic workflow engine. Every node output remains aligned to the same records; pandas index reordering is rejected.

A comparison can include record-level baseline/candidate actions, transition matrices, changed nodes, structural downstream reachability, exact or sampled software-counterfactual attribution, pairwise interactions, cohort summaries, optional outcomes, fragility information, cache diagnostics, and saved evidence.

Categorical actions are never numerically subtracted or silently ordinal-encoded. If baseline/candidate topology is incompatible for hybrid substitution, attribution is reported as unsupported rather than fabricating a substitution graph.

Details: [`docs/decision-flows.md`](docs/decision-flows.md) · [`docs/flow-attribution.md`](docs/flow-attribution.md) · [`docs/architecture.md`](docs/architecture.md)

### Evidence and Decision Contracts

Saved DecisionFlow runs use evidence schema `2.0`; supported older DecisionPipeline evidence remains readable.

```bash
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

Verification checks manifest/artifact integrity. It is **tamper-evident**, not signer authentication or proof that source data are true.

Decision Contract exit codes are automation-friendly:

| Outcome | Exit code |
|---|---:|
| pass | `0` |
| usage/config error | `2` |
| contract violation | `10` |
| insufficient evidence | `11` |
| integrity failure | `12` |

Details: [`docs/decision-contracts.md`](docs/decision-contracts.md) · [`docs/evidence-integrity.md`](docs/evidence-integrity.md) · [`docs/failure-modes.md`](docs/failure-modes.md)

### Public examples and reproducible benchmarks

The repository includes three complementary examples:

- [`examples/triage/`](examples/triage/) — bundled synthetic equipment-maintenance DecisionFlow;
- [`examples/public_wine_sklearn/`](examples/public_wine_sklearn/) — public Wine data with a trained scikit-learn model;
- [`examples/public_digits_xgboost/`](examples/public_digits_xgboost/) — public Digits data with a trained XGBoost model and the case study above.

Benchmarks are machine-generated and tied to source commits:

```bash
python benchmarks/benchmark_cpu.py
python benchmarks/benchmark_flow_cpu.py
python benchmarks/evaluate_flow_attribution.py \
  --json-out evaluation-artifacts/flow-benchmark.json \
  --markdown-out evaluation-artifacts/flow-benchmark.md
```

The first committed flow-attribution snapshot is [`benchmarks/results/76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md`](benchmarks/results/76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md). Its bounded sampled runs did **not** meet the configured CI-width convergence target; that negative result is retained as measured rather than tuned away.

### Compatibility and quality gates

| Area | Verification |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| Linux | full unit/integration/coverage/build matrix |
| macOS | core smoke path on Python 3.11 |
| Windows | core smoke path on Python 3.11 |
| Core dependencies | NumPy `>=1.24`, pandas `>=2.0`, PyYAML `>=6.0`, Typer `>=0.26.0` |
| Optional adapters | scikit-learn `>=1.3`, XGBoost `>=2.0`, LightGBM `>=4.0` |

Normal CI covers the declared Python matrix, minimum runtime dependencies, property/golden tests, package validation, the DecisionFlow trust path, and dedicated scikit-learn/XGBoost/LightGBM examples. The repository also maintains targeted typing checks, trust-critical coverage checks, API-reference drift detection, mutation testing, release SBOM generation, and build-provenance attestations where supported.

Details: [`docs/compatibility.md`](docs/compatibility.md)

## What is new in v0.3.1

- public trained DecisionFlow examples using scikit-learn Wine and XGBoost Digits;
- machine-generated decision-change case study showing improved model metrics alongside a 4.17% action shift and a cohort-aware BLOCK result;
- commit-tied exact-vs-sampled attribution benchmark snapshots with timing, allocation and convergence diagnostics;
- stronger cache/input-content isolation and hybrid replay invalidation;
- branch-aware coverage floor raised to 80% with targeted trust-path regression tests;
- GitHub Actions refreshed and pinned to immutable commit SHAs; and
- expanded security/reproducibility documentation around integrity versus authenticity.

Latest release: **[DeciShift v0.3.1 — Trust and evaluation hardening](https://github.com/sauravsingla/DeciShift/releases/tag/v0.3.1)**.

## Installation and API

Development install:

```bash
python -m pip install -e ".[dev]"
```

Core Python usage is available through `DecisionFlow`, `DecisionNode`, and `compare_flows`. For a compact end-to-end API example, see [`docs/getting-started.md`](docs/getting-started.md) and the generated [`docs/api-reference.md`](docs/api-reference.md).

## Verification

Useful repository checks and workflows are defined under `.github/workflows/`, including normal tests, evaluation, mutation testing, dataset/Space publication, README demo publication, and release publication.

The v0.3.1 release tag points to exact source commit `082250abc1c0a82a41480949594168bb56f323ad`, which the release notes identify as having passed the repository's post-merge test matrix. The same notes explicitly record that branch-level enforcement was not enabled for that release and is tracked separately.

## Documentation

- [`docs/getting-started.md`](docs/getting-started.md) — getting started
- [`docs/architecture.md`](docs/architecture.md) — architecture
- [`docs/decision-flows.md`](docs/decision-flows.md) — DecisionFlow semantics
- [`docs/flow-attribution.md`](docs/flow-attribution.md) — attribution definitions
- [`docs/decision-contracts.md`](docs/decision-contracts.md) — release contracts
- [`docs/evidence-integrity.md`](docs/evidence-integrity.md) — evidence verification
- [`docs/evaluation-study.md`](docs/evaluation-study.md) — evaluation study
- [`docs/api-reference.md`](docs/api-reference.md) — public API reference
- [`docs/compatibility.md`](docs/compatibility.md) — compatibility
- [`docs/failure-modes.md`](docs/failure-modes.md) — negative-path behavior
- [`docs/comparisons.md`](docs/comparisons.md) — related categories
- [`ROADMAP.md`](ROADMAP.md) — roadmap

## Research positioning and limits

DeciShift is related to behavioral/model regression testing, slice analysis, ML monitoring, Shapley attribution, unit-change attribution, and CI release gates. It does **not** claim that Shapley attribution is novel. Its specific object of analysis is the **decision/action transition produced by a versioned structured executable decision system**.

Important limits:

- results depend on the supplied records;
- structural reachability is not causal impact;
- software-counterfactual attribution does not establish real-world causality;
- sampling intervals quantify permutation-sampling uncertainty only;
- integrity verification does not establish evidence authenticity or signer identity; and
- public benchmark datasets are evaluation evidence, not proof of production deployment fitness.

## Reproduce or contribute

A useful external signal for DeciShift is an **independent run on a real decision pipeline**.

1. Install the release and reproduce one bundled example.
2. Apply the baseline/candidate comparison to a versioned decision system.
3. Keep the generated evidence and environment details.
4. Open an issue with a minimal reproducible case or improvement proposal.

Contribution guidance is in [`CONTRIBUTING.md`](CONTRIBUTING.md); support guidance is in [`SUPPORT.md`](SUPPORT.md).

## Citation and license

If you use DeciShift in research or development, cite the software using [`CITATION.cff`](CITATION.cff).

Current release: **v0.3.1**  
License: **Apache-2.0**

The v0.3.1 Zenodo DOI is intentionally not predeclared. Citation metadata should be updated only to a DOI actually assigned to an archived version.
