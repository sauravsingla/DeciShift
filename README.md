# DeciShift

[![CI](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml/badge.svg)](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/badge/PyPI-v0.3.1-blue.svg)](https://pypi.org/project/decishift/0.3.1/)
[![Python versions](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://pypi.org/project/decishift/0.3.1/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Explain why final decisions changed between versions of an ML decision system.**

A candidate model can improve an aggregate metric while individual operational decisions still change. DeciShift compares two executable versions of a decision system and answers four questions:

1. **Which records changed action?**
2. **Which action transitions occurred?**
3. **Which versioned features, models, policies or rules contributed to those changes?**
4. **Do the observed changes remain inside declared governance limits?**

Git explains **which code changed**. Monitoring explains **whether aggregate metrics changed**. DeciShift focuses on **which final decisions changed and how those changes map back to versioned parts of the executable decision system**.

DeciShift is CPU-first, local-first, offline-capable and framework-agnostic. It requires no GPU, cloud service, database, Docker runtime, LLM/API, telemetry, model registry or hosted dashboard.

## Public-data result: the gap DeciShift is built to expose

The repository includes a machine-generated case study using the public scikit-learn **Digits** dataset with a genuinely trained XGBoost model driving a three-action policy: `auto_not_8`, `manual_review`, and `auto_8`.

On a fixed held-out split of **719 records**:

| Measure | Baseline | Candidate |
|---|---:|---:|
| Model accuracy | 90.26% | **91.79%** |
| ROC AUC | 0.9429 | **0.9472** |
| `auto_8` actions | 66 | 74 |
| `auto_not_8` actions | 530 | 526 |
| `manual_review` actions | 123 | 119 |

The aggregate action mix remains close, but **30 of 719 individual actions changed (4.17%)**.

Changed transitions:

| Transition | Records |
|---|---:|
| `auto_not_8 → manual_review` | 13 |
| `manual_review → auto_8` | 8 |
| `manual_review → auto_not_8` | 9 |

Exact software-counterfactual attribution over the changed nodes assigns the following absolute attribution share:

| Node | Share |
|---|---:|
| Model | **48.86%** |
| Policy | **46.59%** |
| Rules | **4.55%** |

The global Decision Contract passes its **5%** action-shift limit because the observed shift is **4.17%**. The same contract still returns **BLOCK** because the `actual_digit=6` cohort shifts **9.41%**, above its declared **8%** limit.

That is the central use case:

> **model metric improves → aggregate action mix looks similar → individual actions still move → DeciShift identifies the transitions, attributes them across versioned components, and can block a governed cohort**

These figures come from the committed machine-generated snapshot for source commit [`76bbf63abb7fdbba26739f3c8f96f94e0110bfc8`](docs/case-studies/digits-xgboost-76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md). They are descriptive evidence over this public evaluation split, not a claim of production safety, fairness, compliance or real-world causality.

## Quick start

Current release: **v0.3.1 — Trust and evaluation hardening**.

```bash
python -m pip install --upgrade decishift==0.3.1
```

Run the bundled demo:

```bash
decishift demo --rows 1000 --no-save
```

Run the full DecisionFlow trust path from a source checkout:

```bash
python -m pip install -e ".[dev]"

decishift graph examples/triage/flow.yaml

decishift compare examples/triage/flow.yaml
```

A saved run can then be checked and gated with:

```bash
decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

## 60-second real run

The bundled workflow demonstrates `graph → compare → verify → gate` on the synthetic equipment-maintenance DecisionFlow.

![DeciShift 60-second real DecisionFlow run](https://github.com/sauravsingla/DeciShift/releases/download/v0.3.0/decishift-real-run-inline.gif)

The recorded run compares 18 historical records. Eleven final actions change, for a 61.11% action-shift rate. Software-counterfactual attribution assigns 34.51% to `failure_risk_model`, 28.87% to `triage_policy`, 23.24% to `safety_rules`, and 13.38% to `sensor_features`. Evidence verification passes and the configured Decision Contract returns `PASS`.

A contract pass means only that the observed evidence stayed inside the user-declared limits. It is not proof that a candidate is automatically safer, better, fairer, compliant or correct.

## How DeciShift models a decision system

`DecisionPipeline` remains the backwards-compatible API for the original simple binary path:

```text
features -> model -> calibrator -> threshold -> rules -> binary decision
```

`DecisionFlow` represents a deterministic row-aligned DAG whose final node emits the operational action:

```text
features
├── risk_model ──┐
└── value_model ─┼── policy -> rules -> action
```

This supports branching, merging, multiple models or policy stages, and categorical actions such as `monitor`, `inspect`, `service` or `approve`, `manual_review`, `decline` without turning DeciShift into a generic workflow engine.

A node is one executable versioned unit with declared dependencies:

```python
from decishift import DecisionFlow, DecisionNode

flow = DecisionFlow(
    nodes=[
        DecisionNode(
            name="features",
            component=features,
            version="features_v2",
        ),
        DecisionNode(
            name="risk_model",
            component=risk_model,
            depends_on=("features",),
            version="risk_model_v13",
            group="predictive_models",
        ),
        DecisionNode(
            name="policy",
            component=policy,
            depends_on=("risk_model",),
            version="policy_v4",
        ),
        DecisionNode(
            name="final_action",
            component=final_action,
            depends_on=("policy",),
            version="rules_v7",
        ),
    ],
    final_node="final_action",
)
```

Components may expose `run(records, inputs)` or be compatible local callables. Every node output must remain aligned to the same historical records. NumPy vectors/matrices and pandas Series/DataFrames are supported; pandas index reordering is rejected.

DAG validation checks duplicate or missing nodes, self-dependencies, cycles, final-node validity and deterministic topological ordering using standard Python data structures—no NetworkX dependency.

## Multi-action comparison

Categorical actions are compared as explicit transitions rather than numerically encoded labels:

```text
record_id  baseline_action  candidate_action  changed  transition
101        monitor          inspect           true     monitor->inspect
102        inspect          inspect           false    unchanged
103        inspect          service           true     inspect->service
```

Reports include action distributions, transition counts/rates, a transition matrix and the most frequent changed transitions.

A final action must be one non-null, hashable, JSON-serializable scalar per record. Numeric ordering is not required or inferred.

**Categorical actions are never numerically subtracted or silently ordinal-encoded.**

## Software-counterfactual attribution

Categorical actions require an explicitly numeric attribution game. Built-in targets are:

- `candidate_action_support` — movement toward exactly the final candidate action;
- `change_from_baseline` — movement away from the baseline action.

For candidate-action support:

```text
v(S) = 1 if hybrid_action(S) == candidate_action else 0
```

Exact attribution evaluates `2^K` changed nodes or groups up to a safety limit. Approximate attribution samples permutations with streaming uncertainty and optional adaptive stopping. Nodes may belong to explicit attribution groups; DeciShift never invents groups automatically.

Pairwise flow interactions use the numeric target and can also identify `interaction_only_transition` when neither node alone changes the baseline action but their pair does.

All attribution is **software counterfactual attribution**. It does not establish real-world causal effects.

## Efficiency and sampling diagnostics

DeciShift keeps three concepts separate:

- `efficiency_valid` — contributions add back to the observed software-output change;
- `sampling_precision_sufficient` — confidence intervals satisfy the configured width target;
- `sampling_converged` — precision is sufficient and contribution estimates are stable across sampling batches.

Shapley efficiency is **not** treated as Monte Carlo convergence.

Adaptive approximate attribution supports:

```text
min_permutations
max_permutations
batch_size
target_ci_width
confidence_level
seed
```

and reports permutations used, early stopping, max/median CI width and batch-stability information without retaining every permutation sample in memory.

## Topology changes

Node/group hybrid attribution requires compatible topology: the same node names, dependency edges and final node.

If a node is added or removed, an edge changes, or the final node changes, DeciShift may still execute the two complete flows and compare observed actions, but it will not fabricate an invalid hybrid attribution graph:

```text
topology_compatible: false
attribution_status: unsupported topology change
```

## Graph-aware caching

A node cache key contains node execution identity plus dependency lineage. A changed node invalidates its descendants while unaffected upstream or parallel branches can be reused.

Flow evidence reports node evaluations, reused outputs, cache hits and cache misses. Runtime memory identity may be used only as an ephemeral in-process discriminator for unstable objects; it is never serialized as reproducible provenance.

## Evidence integrity and verification

Saved DecisionFlow runs use explicit evidence schema `2.0`, while existing v0.1/v0.2 evidence remains readable.

Flow evidence persists topology/digests, node identities, changed nodes, groups, final node, actions/transitions, structural impact, attribution target/method and sampling diagnostics.

```bash
decishift verify RUN_ID
```

Verification is a **tamper-evident SHA-256 integrity check**. It verifies internal consistency of the saved evidence bundle. It does **not** establish authenticity, signer identity, or trust in the original input data.

## Decision Contracts

Decision Contracts turn observed change evidence into deterministic release gates.

```yaml
contract:
  max_decision_shift_rate: 0.05
  transitions:
    "monitor->inspect":
      max_rate: 0.04
    "inspect->service":
      max_rate: 0.01
  candidate_actions:
    service:
      max_rate: 0.10
```

Gate saved evidence:

```bash
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
```

Exit codes are deterministic: pass `0`, usage/config `2`, contract violation `10`, insufficient evidence `11`, integrity failure `12`.

A `PASS` means only that the user-declared contract passed. It is not proof of safety, fairness, compliance or correctness.

## Cohorts, outcomes and fragility

Flow cohorts report size, coverage, action-shift rate, global/excess shift rate and the most common transition. Requested transition-specific rates are optional.

Classification metrics are calculated only when explicitly configured:

```yaml
outcome:
  column: actual_class
  actions_are_predictions: true
```

Operational actions are not automatically treated as predicted labels.

An advanced final node may return a row-aligned `DecisionOutput(action=..., score=..., margin=...)`, or a per-record sequence of `DecisionOutput` objects. Fragility is computed only when a meaningful numeric margin is explicitly supplied. DeciShift never invents a margin for categorical actions.

## Public examples

The repository contains three complementary examples.

### 1. Equipment maintenance — synthetic, fully bundled

[`examples/triage/`](examples/triage/) is a general-purpose equipment-maintenance triage flow:

```text
sensor_features
      │
      ├── failure_risk_model ──┐
      │                        ├── triage_policy -> safety_rules -> final_action
      └── downtime_model ──────┘
```

Actions are `monitor`, `inspect`, and `service`. Baseline/candidate versions change a sensor transform, risk model, policy and safety rule. No fraud, payments, mule-detection or employer-specific data is used.

### 2. Wine — public data, trained scikit-learn model

[`examples/public_wine_sklearn/`](examples/public_wine_sklearn/) uses scikit-learn's bundled Wine classification dataset: 178 public samples, 13 numeric features and 3 classes.

A genuinely trained logistic-regression score drives three actions:

```text
route_not_class2 | manual_review | route_class2
```

The example separately evaluates:

```text
feature-only
model-only
policy-only
rule-only
all changes combined
```

### 3. Digits — public data, trained XGBoost model

[`examples/public_digits_xgboost/`](examples/public_digits_xgboost/) uses scikit-learn's bundled Digits dataset: 1,797 public 8×8 digit images with 64 pixel features.

A genuinely trained XGBoost score for `digit 8` versus other digits drives:

```text
auto_not_8 | manual_review | auto_8
```

It evaluates the same feature/model/policy/rule isolation matrix and produces the decision-change case study summarized at the top of this README.

Both public datasets are bundled with scikit-learn, so no network dataset download is required at runtime.

## Reproducible benchmarks

Existing binary benchmark:

```bash
python benchmarks/benchmark_cpu.py
```

DecisionFlow benchmark:

```bash
python benchmarks/benchmark_flow_cpu.py
```

Reproducible exact-vs-sampled evaluation:

```bash
python benchmarks/evaluate_flow_attribution.py \
  --json-out evaluation-artifacts/flow-benchmark.json \
  --markdown-out evaluation-artifacts/flow-benchmark.md
```

The evaluation benchmark runs linear 5-node and branched 8-node cases at **10,000** and **100,000** rows by default, with an explicit optional **1,000,000-record** mode.

For each case it records:

- changed nodes and changed final actions;
- comparison wall time;
- attribution wall time;
- `tracemalloc` peak Python allocation;
- hybrid evaluations;
- nodes executed and outputs reused;
- cache-reuse ratio;
- attribution efficiency and MAE;
- permutation count;
- sampling precision/convergence;
- maximum confidence-interval width.

Benchmark results are machine-generated and tied to the source commit. Timing and memory values are not hand-entered into this README.

The first committed snapshot is [`benchmarks/results/76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md`](benchmarks/results/76bbf63abb7fdbba26739f3c8f96f94e0110bfc8.md). For those 3–4 changed-node cases, exact attribution was faster than the bounded sampled protocol. The sampled runs used all 64 permutations and did **not** meet the configured CI-width convergence target. That result is retained as measured rather than tuned away.

The heavier benchmark publication workflow is explicit and read-only. It records the exact source commit it executes and does not write to `main` or overwrite versioned release assets.

## CLI

```bash
decishift demo

decishift compare config.yaml
decishift compare examples/triage/flow.yaml

decishift graph examples/triage/flow.yaml
decishift graph examples/triage/flow.yaml --format mermaid

decishift explain --run RUN_ID --id RECORD_ID

decishift report RUN_ID --format terminal
decishift report RUN_ID --format json
decishift report RUN_ID --format markdown
decishift report RUN_ID --format html

decishift verify RUN_ID
decishift gate RUN_ID --contract examples/triage/decision-contract.yaml
decishift compare-runs RUN_A RUN_B
```

All core commands work locally/offline. HTML reports are self-contained and require no CDN, JavaScript or web server.

## Tests and quality gate

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest --cov=decishift
python -m build
python -m twine check dist/*
decishift demo --rows 1000 --no-save
```

CI runs on Python **3.11, 3.12 and 3.13**, tests the declared minimum-supported dependency floor, and exercises the full `graph → compare → verify → gate` trust path. Dedicated optional-dependency jobs cover XGBoost and LightGBM, and the trained public scikit-learn/XGBoost examples are executed in CI.

## Research positioning and limits

DeciShift is related to model regression testing, behavioral diffing, slice analysis, ML monitoring, Shapley attribution, unit-change attribution and CI release gates. It does **not** claim that Shapley attribution is novel.

Its specific object of analysis is the **decision/action transition produced by a versioned structured executable decision system**.

Important limits:

- results depend on the supplied historical records;
- structural reachability is not causal impact;
- software-counterfactual attribution does not establish real-world causality;
- sampling intervals quantify permutation-sampling uncertainty only;
- integrity verification does not establish evidence authenticity or signer identity;
- public benchmark datasets are evaluation evidence, not proof of production deployment fitness.

## Documentation

- [Decision flows](docs/decision-flows.md)
- [Multi-action decisions](docs/multi-action.md)
- [Flow attribution](docs/flow-attribution.md)
- [Structural impact](docs/structural-impact.md)
- [Adaptive attribution](docs/adaptive-attribution.md)
- [Decision Contracts](docs/decision-contracts.md)
- [Evidence integrity](docs/evidence-integrity.md)
- [Evaluation study](docs/evaluation-study.md)
- [Limitations](docs/limitations.md)

## Release, license and citation

Current release: **v0.3.1 — Trust and evaluation hardening**.

- GitHub release: https://github.com/sauravsingla/DeciShift/releases/tag/v0.3.1
- PyPI: https://pypi.org/project/decishift/0.3.1/
- License: [Apache-2.0](LICENSE)
- Citation metadata: [`CITATION.cff`](CITATION.cff)

The v0.3.1 Zenodo DOI is intentionally not predeclared. If a new Zenodo archive is minted for this release, the repository citation metadata should be updated to the DOI actually assigned to that archived version.
