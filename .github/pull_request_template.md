## Summary

Describe the behavioral change and why it belongs in DeciShift.

## Validation

- [ ] `ruff check .`
- [ ] `pytest --cov=decishift`
- [ ] Relevant CLI/example path exercised when applicable
- [ ] Tests added or updated for behavior changes

## Trust and compatibility

- [ ] Evidence/schema compatibility considered for evidence changes
- [ ] Attribution target/quantity is explicit for attribution changes
- [ ] No real-world causal claim is implied by software-counterfactual attribution
- [ ] No implicit telemetry or required network call was introduced
- [ ] New infrastructure/runtime dependencies are justified

## Documentation

- [ ] User-visible behavior is documented
- [ ] `CHANGELOG.md` updated when release-facing behavior changes
- [ ] Public API inventory remains current (`python scripts/generate_api_reference.py --check`)
