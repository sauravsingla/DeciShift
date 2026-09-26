# Compatibility and support matrix

DeciShift is designed to be CPU-first, local-first, and OS-independent at the Python package level. The matrix below distinguishes full CI coverage from lighter smoke coverage.

| Area | Current support / verification |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| Linux | Full unit/integration/coverage/build matrix |
| macOS | Core installation and CLI/DecisionFlow smoke tests on Python 3.11 |
| Windows | Core installation and CLI/DecisionFlow smoke tests on Python 3.11 |
| NumPy | `>=1.24` |
| pandas | `>=2.0` |
| PyYAML | `>=6.0` |
| Typer | `>=0.26.0` |
| scikit-learn adapter/example | Optional, `>=1.3` |
| XGBoost adapter/example | Optional, `>=2.0` |
| LightGBM adapter | Optional, `>=4.0` |

The optional XGBoost and LightGBM jobs are exercised separately from the core platform-smoke matrix. A core platform pass therefore does not imply every optional ML library/version is validated on that operating system.

## Runtime assumptions

Core DeciShift does not require a GPU, cloud account, Docker runtime, database, model registry, hosted dashboard, LLM/API, or telemetry service. The bundled core and public scikit-learn examples can run locally; the scikit-learn datasets used by the public examples are bundled with scikit-learn and do not require a dataset download at runtime.

## Dependency floors

CI separately installs the declared minimum runtime dependency versions on Python 3.11. This catches accidental reliance on newer APIs while the main matrix also tests against current compatible packages.

## Compatibility promises

The project is still classified as alpha. Backward-readable evidence is treated more conservatively than ordinary internal implementation details: DecisionPipeline schema `1.0` and older supported evidence remain readable while DecisionFlow uses schema `2.0`. Any future incompatible public-API or evidence change should be documented explicitly.
