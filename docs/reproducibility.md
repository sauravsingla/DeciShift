# Reproducibility and component identity

DeciShift does not use runtime memory addresses as reproducible component evidence.

Component identity priority is:

1. explicit configured version/digest;
2. artifact SHA-256 when an artifact path is supplied;
3. explicit component `.version`;
4. deterministic source identity for stateless source-inspectable callables;
5. otherwise `unstable`.

Reports expose one of:

- `reproducible`;
- `partially_reproducible`;
- `unstable`.

Strict mode (`strict_reproducibility: true`) rejects unstable component identity. Within-process comparisons can still run without strict mode, but the limitation remains visible.

DeciShift deliberately does not pickle arbitrary user objects just to derive a fingerprint.
