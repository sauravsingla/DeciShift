# Reproducibility and component identity

DeciShift separates three ideas that are easy to conflate:

1. **component identity** — what executable software/artifact is being compared;
2. **input identity** — what historical records the comparison ran against;
3. **evidence integrity** — whether a saved evidence bundle still matches its manifest.

None of these, by itself, establishes real-world causality or signer authenticity.

## Component identity

DeciShift does not use runtime memory addresses as reproducible component evidence.

Component identity priority is:

1. explicit configured version/digest;
2. artifact SHA-256 when an artifact path is supplied;
3. explicit component `.version`;
4. deterministic source identity for stateless source-inspectable callables;
5. otherwise `unstable`.

Reports expose one of:

- `reproducible`;
- `partially_reproducible`;
- `unstable`.

Strict mode (`strict_reproducibility: true`) rejects unstable component identity. Within-process comparisons can still run without strict mode, but the limitation remains visible.

DeciShift deliberately does not pickle arbitrary user objects just to derive a fingerprint.

## Input identity and mutation isolation

Executable comparisons fingerprint the supplied DataFrame content. Reusable `DecisionPipeline`, `HybridReplayCache`, and `DecisionFlow` cache entries are bound to that content fingerprint, so a cache created for one record set is not silently reused for different record content.

Caller-owned DataFrames are not handed directly to user components. Pipeline evaluations use an execution snapshot, and flow nodes receive isolated record copies. Reusable dependency/intermediate outputs are also defensively copied before user code receives them. As a result, ordinary in-place mutation inside a feature, model, threshold, rule, or flow node cannot:

- modify the caller's original DataFrame;
- poison an intermediate cache entry used by another hybrid replay;
- leak record mutations into an independent sibling node; or
- make a later evaluation reuse a trace for different record content.

This is correctness isolation, **not a security sandbox**. Trusted Python components can still use any process capability available to Python.

## Unstable components within one process

When provenance is unavailable, DeciShift can still distinguish different runtime component objects for execution and cache correctness. That runtime identity is deliberately ephemeral: it is never serialized, hashed into evidence, or presented as reproducible provenance. Reports continue to mark the component as unstable.

## Evidence integrity versus authenticity

`decishift verify RUN_ID` recomputes the manifest integrity root and SHA-256 hashes of declared artifacts. A PASS means the local evidence matches its manifest. It does not identify the creator or signer. If authenticated provenance is required, combine the evidence bundle with an external signing/attestation mechanism.

## Practical reproducibility checklist

For a comparison intended to be independently rerun:

- use explicit versions or artifact digests for stateful components/models;
- enable strict reproducibility where feasible;
- keep input content hashing enabled;
- preserve the exact configuration and evidence bundle;
- record the source commit/tag and dependency environment;
- verify the bundle before applying a Decision Contract; and
- do not execute untrusted configuration or imported Python components.
