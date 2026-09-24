# Contributing

Contributions are welcome. Keep the v0.1 principles intact: local-first, CPU-friendly, deterministic where practical, minimal dependencies, independently testable modules, and explicit scientific limits.

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run `pytest`.
4. Avoid adding infrastructure dependencies unless the feature genuinely requires them.
5. Do not introduce telemetry or implicit network calls.
6. For attribution changes, document what quantity is decomposed and verify additivity or approximation behavior where applicable.
