# Failure modes and expected behavior

DeciShift is intended to fail explicitly when evidence or execution assumptions are not satisfied. These negative paths are part of the trust model.

## Incompatible flow topology

Baseline and candidate flows can still be executed independently and their final actions compared when nodes/edges differ. Hybrid node/group attribution is not fabricated across incompatible topology; evidence reports `topology_compatible: false` and an unsupported-attribution status.

## Row or identifier misalignment

DecisionFlow outputs must remain aligned to the supplied historical records. Reordered pandas indices are rejected. When an `id_column` is configured, duplicate or null identifiers are rejected by default unless the caller explicitly opts into them.

## User component failure

A node must be callable or expose `run(records, inputs)`. Exceptions raised inside user code are wrapped as component-execution failures; DeciShift does not reinterpret an internal `TypeError` as a signal to retry a different calling convention.

## Insufficient sampled-attribution precision

Approximate attribution reports sampling precision and convergence separately from Shapley efficiency. Contributions can add up correctly while sampling precision is still insufficient. A non-converged sampled run should not be described as converged merely because efficiency holds.

## Evidence tampering or missing artifacts

`decishift verify RUN_ID` recomputes the manifest integrity root and each declared artifact SHA-256. Changed, missing, path-escaping, or non-canonicalizable evidence fails verification. Verification establishes internal tamper evidence, not signer identity, input truth, or organizational authenticity.

## Decision Contract outcomes

The CLI uses deterministic exit codes for automation:

| Outcome | Exit code |
|---|---:|
| Contract passed | `0` |
| Usage/configuration error | `2` |
| Contract violation | `10` |
| Insufficient evidence | `11` |
| Evidence integrity failure | `12` |

A contract `PASS` means only that the observed evidence stayed within the user-declared limits. It is not proof of safety, fairness, compliance, correctness, or deployment fitness.

## Outcomes and operational actions

Categorical operational actions are not automatically treated as predicted class labels. Classification metrics are computed only when the caller explicitly configures an outcome column and declares that actions are predictions.

## Trust-boundary checklist

When a result looks surprising, verify: the same intended historical records were replayed; component identities represent the intended versions; topology is compatible for hybrid attribution; action labels remain row aligned; approximate-attribution diagnostics meet the configured precision target; and the contract expresses the operational limits actually intended.
