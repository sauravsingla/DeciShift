# Changelog

All notable changes to DeciShift are documented here.

## 0.2.1 - Unreleased

### Changed

- Separate attribution additivity (`efficiency_valid`) from Monte Carlo sampling precision and convergence semantics.
- Preserve the legacy `converged` field while requiring genuine sampling convergence for approximate attribution rather than inferring convergence from Shapley efficiency alone.
- Add optional adaptive permutation sampling with `min_permutations`, `max_permutations`, `batch_size`, `target_ci_width`, `confidence_level`, and deterministic `seed` handling.
- Report `permutations_used`, `stopped_early`, `sampling_precision_sufficient`, `sampling_converged`, maximum/median CI width, and batch-stability diagnostics without retaining permutation samples in memory.
- Keep existing fixed-permutation approximate attribution behavior and public APIs working.

## 0.2.0 - 2026-09-24

Theme: **Trustworthy Decision-Change Evidence**.

### Added

- Streaming Monte Carlo uncertainty for approximate component attribution, including standard errors and configurable confidence intervals.
- `AttributionDiagnostics` with score/decision efficiency residual statistics and warnings.
- Stable `ComponentIdentity` provenance with explicit, artifact, source and unstable identity states plus strict reproducibility mode.
- Evidence schema `1.0`, deterministic input fingerprints, evidence manifests and tamper-evident SHA-256 integrity verification.
- `decishift verify RUN_ID` with deterministic integrity-failure exit behavior.
- User-defined Decision Contracts and `decishift gate RUN_ID --contract ...` with documented CI exit codes.
- Stronger row identity, finite-output and pandas index-order validation.
- Signature-aware callable invocation that does not reinterpret internal user `TypeError` failures as arity mismatches.
- Dynamic YAML threshold objects/factories while preserving numeric thresholds.
- Conservative automatic cohort discovery, Wilson flip-rate intervals, coverage/global context and excess flip rate.
- Optional binary outcome correctness-transition analysis.
- Decision Fragility boundary-proximity analysis.
- Fully local, self-contained static HTML reports.
- `decishift compare-runs RUN_A RUN_B` for comparing saved evidence bundles.
- Expanded benchmarks for exact/approximate attribution, uncertainty, serialization and verification.
- Ruff/build/twine checks in CI and expanded regression coverage.

### Compatibility

- Preserves `DecisionPipeline`, `compare_pipelines`, `compare_predictions`, `exact_attribution`, `approximate_attribution` and `pairwise_interactions`.
- Existing attribution columns remain; v0.2 adds uncertainty columns rather than renaming the v0.1 schema.
- Best-effort loading of v0.1 saved runs does not invent missing provenance.

## 0.1.0 - 2026-09-24

Initial public release.

### Added

- CPU-first, local-first comparison of baseline and candidate ML decision pipelines.
- Version-aware pipeline components for features, model, calibration, threshold/policy, and deterministic rules.
- Record-level decision diffs, score deltas, threshold margins, and flip direction.
- Exact Shapley counterfactual component attribution for small changed-component sets.
- Deterministic permutation approximation for component attribution.
- Baseline-anchored pairwise interaction analysis, including interaction-only flips.
- Predictions-only analysis with explicit insufficient-evidence handling for component attribution.
- Numeric and categorical cohort analysis with minimum cohort-size controls.
- Saved local runs, individual record explanation, terminal/JSON/Markdown reporting.
- Framework-agnostic model interface with optional scikit-learn, XGBoost, and LightGBM support.
- Synthetic equipment-maintenance demonstration and reproducible CPU benchmark harness.
- Regression tests and GitHub Actions CI.
- Apache-2.0 license and `CITATION.cff` metadata for archival/citation workflows.
