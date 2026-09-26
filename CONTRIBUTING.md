# Contributing

Contributions are welcome. DeciShift is intentionally local-first, CPU-friendly, deterministic where practical, minimal in required infrastructure, and explicit about scientific limits.

## Development setup

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m mypy decishift/evidence.py decishift/flow/actions.py decishift/flow/node.py
pytest --cov=decishift
python scripts/generate_api_reference.py --check
```

The CI suite also builds the package, checks distribution metadata, runs public examples, exercises `graph → compare → verify → gate`, validates minimum-supported runtime dependencies, and runs core smoke tests on Linux, macOS, and Windows.

## Contribution principles

1. Add or update tests for behavior changes.
2. Do not introduce telemetry or implicit network calls.
3. Avoid infrastructure dependencies unless the feature genuinely requires them.
4. Preserve row alignment and deterministic behavior where the API promises them.
5. For attribution changes, document exactly what quantity is decomposed and test efficiency/approximation behavior where applicable.
6. For evidence changes, preserve backward readability where documented and add compatibility/golden tests.
7. Distinguish structural reachability, software-counterfactual attribution, and real-world causality.
8. Update user documentation and `CHANGELOG.md` for release-facing behavior changes.

## Property and compatibility testing

Property-based tests use Hypothesis for invariants such as deterministic canonicalization and typed categorical-action identity. Golden evidence fixtures protect verifier compatibility and tamper detection.

## Mutation testing

Mutation testing is deliberately separate from normal PR CI because it is substantially heavier and requires a POSIX/fork-capable environment.

```bash
python -m pip install -e ".[dev,mutation]"
mutmut run
```

The scheduled/manual mutation workflow targets trust-critical evidence and categorical-action logic rather than mutating the entire package indiscriminately.

## Pull requests

Keep pull requests focused. The PR template calls out testing, evidence/schema compatibility, attribution semantics, scientific claims, and network/telemetry boundaries. Small documentation, test, and compatibility contributions are encouraged; look for issues labeled `good first issue`.

For usage questions, use [GitHub Discussions](https://github.com/sauravsingla/DeciShift/discussions). For vulnerabilities, follow [SECURITY.md](SECURITY.md).
