# Security

DeciShift is intended for local analysis of potentially sensitive model-decision evidence. Its local-first design reduces data movement, but it does not make untrusted code or untrusted inputs safe to execute.

## Treat executable configuration as code

DeciShift YAML can import local Python objects and invoke configured factories. A configuration file can therefore cause Python code to execute with the permissions of the current process. Only run configurations, component modules, model artifacts, and factories from sources you trust. `yaml.safe_load` prevents YAML object construction, but it does not sandbox the Python objects that the DeciShift configuration explicitly imports.

## Protect data and generated evidence

- Do not commit production data, credentials, proprietary models, API keys, or generated `.decishift/runs` directories unless you deliberately intend to publish them.
- Saved evidence may contain record identifiers, transitions, cohort values, scores, and other derived information. Treat the evidence bundle according to the sensitivity of the source records.
- DeciShift hashes input content for reproducibility/integrity metadata by default; a hash is not encryption and does not make sensitive source data public-safe.

## Integrity is not authenticity

`decishift verify RUN_ID` validates the SHA-256 manifest and declared evidence artifacts. A successful verification detects tampering relative to that manifest. It does **not** establish who created the evidence, who approved it, whether the original execution environment was trustworthy, or whether the manifest itself came from a trusted signer.

For workflows that require signer identity or stronger provenance, store DeciShift evidence inside an independently authenticated release, attestation, or signing system.

## Execution isolation and caches

DeciShift isolates caller-owned DataFrames before handing records to executable pipeline/flow components, binds reusable caches to input-content fingerprints, and defensively copies reusable intermediate values. This protects comparison correctness from ordinary in-place mutation by custom components. It is **not** a sandbox: imported Python code can still access the filesystem, environment, network, subprocesses, or other process capabilities available to it.

## CI and release supply chain

Repository workflows use least-privilege permissions and pin third-party GitHub Actions to immutable commit SHAs. When copying the example workflow, preserve SHA pinning and review any action update before changing the pinned commit. PyPI publication should use Trusted Publishing/OIDC rather than a long-lived package token.

## Reporting vulnerabilities

Please report security concerns privately to the repository owner rather than opening a public issue with exploit details. Include the affected version/commit, a minimal reproduction, impact, and any suggested mitigation when possible. Do not include real sensitive data in a vulnerability report.
