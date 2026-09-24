# Architecture

DeciShift compares the **decision transition produced by a versioned structured decision system** on the same historical records.

Two execution abstractions coexist.

## DecisionPipeline — simple backwards-compatible path

```text
historical records
  -> features
  -> model
  -> calibration
  -> threshold/policy
  -> deterministic rules
  -> binary final decision
```

`DecisionPipeline` remains the v0.1/v0.2 public API. Its binary comparison schema, imports, exact/approximate attribution and interaction behavior remain supported.

## DecisionFlow — composable advanced path

```text
features
├── risk_model ─┐
└── value_model ├── policy -> rules -> final_action
```

`DecisionFlow` is a deterministic row-aligned DAG. It supports branching, merging, multiple models/policies/rules and categorical final actions. It is intentionally not a generic workflow orchestrator.

Nodes execute locally in deterministic topological order. Dependency-aware cache keys include node execution identity and upstream lineage, allowing unchanged branches to be reused while a changed node and its descendants are recomputed.

Baseline and candidate flows may have different topology for **independent observed-action comparison**. Hybrid node attribution is available only when node names, dependency edges and final node are compatible.

## Evidence layers

```text
comparison
  -> attribution + sampling diagnostics
  -> structural impact / transitions
  -> cohorts / optional explicit outcomes / optional fragility
  -> saved evidence bundle
  -> manifest + SHA-256 integrity root
  -> optional Decision Contract gate
```

DecisionPipeline evidence remains schema `1.0`. DecisionFlow introduces explicit schema `2.0` because topology, node identities, action transitions and flow attribution targets materially extend the evidence model. Existing v0.1/v0.2 evidence remains readable and verifiable.

## Identity and reproducibility

Both abstractions reuse the same component identity framework. Stable explicit versions/digests and artifact hashes are preferred. Runtime memory addresses may only distinguish unstable objects inside the current process; they are never serialized as reproducible evidence.

## Local execution boundary

The package remains CPU-first and file based. There is no required GPU, cloud service, database, Docker runtime, model registry, feature store, telemetry path, LLM/API, scheduler or hosted dashboard.

## Trust boundaries

DeciShift can verify its own declared artifact hashes and report supplied/safely derived software provenance. Hash verification does not authenticate a person or organization and does not prove that an external model/data artifact is trustworthy.

Structural impact is graph reachability, not causal impact. Software counterfactual attribution does not establish real-world causality.
