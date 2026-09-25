# Evidence integrity

Every v0.2 saved run writes an evidence bundle under `.decishift/runs/<run_id>/` with `manifest.json`, structured CSV/JSON evidence and terminal/Markdown/HTML reports.

The manifest records SHA-256 digests for declared evidence artifacts and an integrity root derived from canonical manifest content. Verify it with:

```bash
decishift verify RUN_ID
```

Verification recomputes artifact hashes and the manifest integrity root, reports missing or modified files, and exits nonzero on failure.

This is **tamper-evident integrity verification, not evidence authenticity or signer authentication**. A successful verification means the current artifact bytes match the SHA-256 digests declared by the current manifest and that the manifest integrity root is internally consistent. Because the manifest and artifacts can be replaced together by someone with write access, this mechanism alone does **not** prove who created the evidence, when it was created, that it came from a trusted production system, or that the original inputs were truthful. Establishing those properties requires an external trust mechanism such as signed attestations, trusted timestamps, or independently controlled provenance.

Input content hashing can be disabled for environments where even deterministic content fingerprints are disallowed. Doing so reduces reproducibility evidence and is recorded in the manifest.


DecisionFlow execution also fingerprints the input DataFrame content for cache identity, validates row alignment (including structured intermediate `DecisionOutput` fields), and isolates cached/dependency outputs from in-place component mutation. These controls prevent stale cache reuse after record mutation and reduce accidental cross-node/cache corruption; they are execution-integrity controls, not cryptographic authentication.
