# CI usage

DeciShift can be used as a normal command-line release gate; no hosted GitHub App is required.

Typical flow:

```bash
decishift compare decishift.yaml --save
decishift verify RUN_ID
decishift gate RUN_ID --contract decision-contract.yaml
```

The gate's deterministic exit code can be consumed by GitHub Actions or another CI system. See `examples/github-actions/decishift-gate.yml`.

CI should preserve the evidence bundle as an artifact when the comparison itself is part of a review or release process.
