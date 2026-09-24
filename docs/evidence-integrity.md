# Evidence integrity

Every v0.2 saved run writes an evidence bundle under `.decishift/runs/<run_id>/` with `manifest.json`, structured CSV/JSON evidence and terminal/Markdown/HTML reports.

The manifest records SHA-256 digests for declared evidence artifacts and an integrity root derived from canonical manifest content. Verify it with:

```bash
decishift verify RUN_ID
```

Verification recomputes artifact hashes and the manifest integrity root, reports missing or modified files, and exits nonzero on failure.

This is **tamper-evident integrity verification**. A hash can show that bytes differ from those declared in the manifest; it does not prove who created the evidence, provide signer identity, or establish trust in the original inputs.

Input content hashing can be disabled for environments where even deterministic content fingerprints are disallowed. Doing so reduces reproducibility evidence and is recorded in the manifest.
