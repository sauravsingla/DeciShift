# Architecture

DeciShift compares the **discrete decision transition produced by a versioned structured decision pipeline**.

```text
historical records
  -> features
  -> model
  -> calibration
  -> threshold/policy
  -> deterministic rules
  -> final decision
```

A comparison evaluates the baseline and candidate on the same row-aligned records. Hybrid replay then substitutes changed components to estimate software-counterfactual contributions. A `HybridReplayCache` memoizes complete hybrid traces and dependency-keyed intermediate outputs so unchanged upstream work can be reused.

v0.2 adds an evidence layer around that execution:

```text
comparison
  -> attribution + uncertainty
  -> diagnostics
  -> cohorts / optional outcomes / fragility
  -> saved evidence bundle
  -> manifest + SHA-256 integrity root
  -> optional Decision Contract gate
```

The evidence layer remains local and file based. There is no server, database, telemetry path, hosted service, or mandatory model framework.

## Trust boundaries

DeciShift can verify its own saved artifact hashes and report component provenance supplied or safely derived by the user. It does not authenticate a human or organization, and it does not prove that a supplied model artifact is trustworthy.
