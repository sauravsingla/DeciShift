# DeciShift

[![CI](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml/badge.svg)](https://github.com/sauravsingla/DeciShift/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/decishift/0.3.0/)
[![Python versions](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://pypi.org/project/decishift/0.3.0/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sauravsingla/DeciShift/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22942359.svg)](https://doi.org/10.5281/zenodo.22942359)

**Explain why decisions changed between versions of an ML decision system.**

> Your candidate model or policy improved a metric. But historical decisions changed. Which ones changed, which software nodes contributed, and what evidence supports that conclusion?

Git explains **which code changed**. Model monitoring explains **whether metrics changed**. DeciShift explains **which final decisions/actions changed and which versioned parts of the executable decision system contributed to those changes**.

DeciShift is CPU-first, local-first, offline-capable and framework-agnostic. It requires no GPU, cloud service, database, Docker runtime, LLM/API, telemetry, model registry or hosted dashboard.

## 60-second real run

The workflow runs `graph → compare → verify → gate`, then renders the actual results as a 60-second animation.

![DeciShift 60-second real DecisionFlow run](https://github.com/sauravsingla/DeciShift/releases/download/v0.3.0/decishift-real-run-inline.gif)

The recorded run compares 18 historical records: 11 actions change, giving a 61.11% action-shift rate. The largest software-counterfactual attribution is `failure_risk_model` at 34.51%, followed by `triage_policy` at 28.87%, `safety_rules` at 23.24%, and `sensor_features` at 13.38%. Evidence verification passes and the configured Decision Contract returns `PASS`. A contract pass means only that the observed changes remain inside the user-declared governance limits; it is not proof that the candidate is automatically safer, better, fairer, compliant or correct.

## Install

The current release is **v0.3.0**:

```bash
python -m pip install --upgrade decishift==0.3.0
decishift demo --rows 1000 --no-save
```

For development from a source checkout:

```bash
python -m pip install -e ".[dev]"
decishift graph examples/triage/flow.yaml
decishift compare examples/triage/flow.yaml
```

## v0.3.0 — Composable Decision Flows

`DecisionPipeline` remains the backwards-compatible API for the original simple binary path:

```text
features -> model -> calibrator -> threshold -> rules -> binary decision
```

v0.3.0 adds `DecisionFlow` for deterministic row-aligned DAGs:

```text
features
├── risk_model ──┐
└── value_model ─┼── policy -> rules -> action
```

This supports branching, merging, multiple models/policy stages and categorical actions such as `monitor`, `inspect`, `service` or `approve`, `manual_review`, `decline` without turning DeciShift into a generic workflow engine.

An **illustrative report shape** (not benchmark output) is:

```text
Historical records: 100,000
Actions changed:     2,841
Action-shift rate:   2.84%

Transitions
monitor -> inspect
inspect -> service
service -> inspect

Changed nodes
risk_model
policy

Structural descendants
policy
rules
final_action
```

Structural descendants are graph reachability only. They are **not causal impact** and do not imply that every downstream node or historical action actually changed.

## DecisionFlow

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

Components may expose `run(records, inputs)` or be compatible local callables. Exceptions raised inside user components are not silently reinterpreted as calling-convention errors.

Every node output must remain aligned to the same historical records. NumPy vectors/matrices and pandas Series/DataFrames are supported; pandas index reordering is rejected. DAG validation checks duplicate/missing nodes, self-dependencies, cycles, final-node validity and deterministic topological ordering using standard Python data structures—no NetworkX dependency.

## Multi-action comparison

For categorical actions DeciShift records explicit transitions instead of subtracting labels:

```text
record_id  baseline_action  candidate_action  changed  transition
101        monitor          inspect           true     monitor->inspect
102        inspect          inspect           false    unchanged
103        inspect          service           true     inspect->service
```

A final action must be one non-null, hashable, JSON-serializable scalar per record. Numeric ordering is not required or inferred.

Reports include action distributions, transition counts/rates, a transition matrix and the most frequent changed transitions.

**Categorical actions are never numerically subtracted or silently ordinal-encoded.**

## Flow attribution

Categorical actions require an explicitly numeric attribution game. Built-in targets are:

- `candidate_action_support` — movement toward exactly the final candidate action;
- `change_from_baseline` — movement away from the baseline action.

For candidate-action support:

```text
v(S) = 1 if hybrid_action(S) == candidate_action else 0
```

Exact attribution evaluates `2^K` changed nodes/groups up to a safety limit. Approximate attribution samples permutations with streaming uncertainty and optional adaptive stopping. Nodes may belong to explicit attribution groups; DeciShift never invents groups automatically.

Pairwise flow interactions use the numeric target and can also identify `interaction_only_transition` when neither node alone changes the baseline action but their pair does.

All attribution is **software counterfactual attribution**. It does not establish real-world causal effects.

## Efficiency and sampling convergence

v0.3.0 separates three ideas that should not be conflated:

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

If a node is added/removed, an edge changes, or the final node changes, DeciShift may still execute the two complete flows and compare observed actions. It reports:

```text
topology_compatible: false
attribution_status: unsupported topology change
```

It does not fabricate invalid hybrid graphs.

## Graph-aware caching

A node cache key contains node execution identity plus dependency lineage. A changed node invalidates its descendants while unaffected upstream/parallel branches can be reused. Flow evidence reports node evaluations, reused outputs, cache hits and cache misses.

Runtime memory identity may be used only as an ephemeral in-process discriminator for unstable objects; it is never serialized as reproducible provenance.

## Evidence and verification

Saved DecisionFlow runs use explicit evidence schema `2.0`, while existing v0.1/v0.2 evidence remains readable.

Flow evidence persists topology/digests, node identities, changed nodes, groups, final node, actions/transitions, structural impact, attribution target/method and sampling diagnostics. SHA-256 evidence verification remains the same command:

```bash
decishift verify RUN_ID
```

Verification is tamper-evident integrity checking, not evidence authenticity or signer authentication. A passing verification shows internal SHA-256 consistency of the saved bundle; it does not prove who created it or that the original inputs were trusted.

## Decision Contracts

Existing binary contract fields remain supported. Flows add action/transition limits:

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

Exit codes remain deterministic: pass `0`, usage/config `2`, contract violation `10`, insufficient evidence `11`, integrity failure `12`.

A PASS means only that the user-declared contract passed. It is not proof of safety, fairness, compliance or correctness.

## Cohorts, outcomes and fragility

Flow cohorts report size, coverage, action-shift rate, global/excess shift rate and the most common transition; requested transition-specific rates are optional.

Classification metrics are calculated only when explicitly configured:

```yaml
outcome:
  column: actual_class
  actions_are_predictions: true
```

Operational actions are not automatically treated as predicted labels.

An advanced final node may return a row-aligned `DecisionOutput(action=..., score=..., margin=...)`, or a per-record sequence of `DecisionOutput` objects. Fragility is computed only when a meaningful numeric margin is explicitly supplied. DeciShift never invents a margin for categorical actions.

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

## Synthetic triage example

The source repository includes `examples/triage/`, a general-purpose equipment-maintenance triage flow:

```text
sensor_features
      │
      ├── failure_risk_model ──┐
      │                        ├── triage_policy -> safety_rules -> final_action
      └── downtime_model ──────┘
```

Actions are `monitor`, `inspect`, and `service`. Baseline/candidate versions change a sensor transform, risk model, policy and safety rule and generate categorical transitions on the bundled synthetic records. No fraud, payments, mule-detection or employer-specific data is used.

## Benchmarks

Existing binary benchmark:

```bash
python benchmarks/benchmark_cpu.py
```

DecisionFlow benchmark:

```bash
python benchmarks/benchmark_flow_cpu.py
```

The flow benchmark executes linear 5-node and branched 8-node cases at 10,000 and 100,000 rows, measuring wall-clock time, `tracemalloc` peak Python memory, cache hits/misses, nodes executed/reused, exact attribution where feasible and adaptive approximate attribution. No benchmark numbers are fabricated or hard-coded into this README.

## Tests and quality gate

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest --cov=decishift
python -m build
python -m twine check dist/*
decishift demo --rows 1000 --no-save
```

CI runs the quality gate on Python 3.11, 3.12 and 3.13 and additionally exercises the flow graph/compare/verify/gate path.

## Documentation

- [Decision flows](https://github.com/sauravsingla/DeciShift/blob/main/docs/decision-flows.md)
- [Multi-action decisions](https://github.com/sauravsingla/DeciShift/blob/main/docs/multi-action.md)
- [Flow attribution](https://github.com/sauravsingla/DeciShift/blob/main/docs/flow-attribution.md)
- [Structural impact](https://github.com/sauravsingla/DeciShift/blob/main/docs/structural-impact.md)
- [Adaptive attribution](https://github.com/sauravsingla/DeciShift/blob/main/docs/adaptive-attribution.md)
- [Decision Contracts](https://github.com/sauravsingla/DeciShift/blob/main/docs/decision-contracts.md)
- [Evidence integrity](https://github.com/sauravsingla/DeciShift/blob/main/docs/evidence-integrity.md)
- [Limitations](https://github.com/sauravsingla/DeciShift/blob/main/docs/limitations.md)

## Research positioning

DeciShift is related to model regression testing, behavioral diffing, slice analysis, ML monitoring, Shapley attribution, unit-change attribution and CI release gates. It does **not** claim that Shapley attribution is novel.

Its specific object of analysis is the **decision/action transition produced by a versioned structured executable decision system**.

Results depend on supplied historical records. Structural reachability is not causal impact. Software counterfactual attribution does not establish real-world causality. Sampling intervals quantify permutation-sampling uncertainty only.

## Release, license and citation

Current release: **v0.3.0 — Composable Decision Flows**.

- GitHub release: https://github.com/sauravsingla/DeciShift/releases/tag/v0.3.0
- PyPI: https://pypi.org/project/decishift/0.3.0/
- License: [Apache-2.0](https://github.com/sauravsingla/DeciShift/blob/main/LICENSE)
- Zenodo v0.3.0: https://zenodo.org/records/22942359
- DOI: https://doi.org/10.5281/zenodo.22942359
- Citation metadata: [`CITATION.cff`](https://github.com/sauravsingla/DeciShift/blob/main/CITATION.cff)

For DeciShift v0.3.0, cite the version-specific DOI **10.5281/zenodo.22942359**.
