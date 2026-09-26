# DeciShift roadmap

This roadmap describes direction, not delivery dates or commitments. Scope discipline is intentional: DeciShift remains focused on decision-level behavioral regression testing rather than becoming a general MLOps platform.

## Current baseline — v0.3.1

The current release provides DecisionPipeline and DecisionFlow comparison, categorical action transitions, exact/sampled software-counterfactual attribution, cohort analysis, evidence integrity verification, Decision Contracts, public-data examples, and CPU-first local execution.

## Near-term themes

### 1. Trust-critical quality

- Expand static typing over evidence, contracts, attribution, and flow execution.
- Keep trust-critical modules above a high coverage floor while broadening overall coverage toward 90%.
- Add property-based, compatibility/golden, and mutation tests for invariants that ordinary line coverage can miss.
- Preserve backward-readable evidence formats when schemas evolve.

### 2. Cross-platform confidence

- Maintain full Linux CI across supported Python versions.
- Exercise core install/CLI/compare/verify/gate behavior on macOS and Windows.
- Keep optional ML dependencies isolated so the core package remains lightweight.

### 3. Documentation and API stability

- Keep a generated public-API inventory tied to `decishift.__all__`.
- Document failure modes and trust boundaries alongside successful examples.
- Move deep technical detail to docs so the README remains a fast entry point.

### 4. Independent validation and integrations

- Collect independent reproducibility reports across environments ([#13](https://github.com/sauravsingla/DeciShift/issues/13)).
- Demonstrate complementary integration patterns with MLflow ([#15](https://github.com/sauravsingla/DeciShift/issues/15)) and ZenML ([#16](https://github.com/sauravsingla/DeciShift/issues/16)) without requiring hosted services.
- Prefer public datasets and reproducible CPU-only examples.

### 5. Release provenance

- Publish a machine-readable declared-dependency SBOM with releases.
- Produce GitHub build provenance attestations for release artifacts where supported.
- Keep PyPI Trusted Publishing/OIDC and tag/version consistency checks.
- Add enforced `main` protection when repository administration permits it ([#7](https://github.com/sauravsingla/DeciShift/issues/7)).

## Non-goals

The roadmap does not include turning DeciShift into a model registry, feature store, hosted dashboard, generic DAG engine, training platform, or telemetry service. New capabilities should make decision changes easier to detect, explain, verify, or gate.
