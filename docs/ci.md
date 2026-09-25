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

## Supply-chain guidance

The repository workflows and example workflow pin external GitHub Actions to full immutable commit SHAs and keep the corresponding release version in a comment. When upgrading an action:

1. identify the intended upstream release;
2. resolve that release/tag to the exact commit SHA;
3. review the upstream release notes and compatibility requirements;
4. replace the SHA and version comment together; and
5. run the full DeciShift test/release matrix before merging.

Read-only jobs use `contents: read`, checkout credentials are not persisted when no later `git push` is required, and the PyPI publishing job grants `id-token: write` only where Trusted Publishing needs it.

For repositories using DeciShift as a required gate, also protect the target branch/ruleset so required checks cannot be bypassed by an ordinary direct push. CI that exists but is not enforced is evidence, not an access-control boundary.
