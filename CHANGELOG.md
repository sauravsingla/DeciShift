# Changelog

All notable changes to DeciShift are documented here.

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
