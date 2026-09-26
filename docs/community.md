---
layout: default
title: DeciShift Community — Test, Discuss, Contribute
description: Join the DeciShift community, reproduce public examples, discuss edge cases, contribute integrations, and help validate ML behavioral regression testing across environments.
---

# DeciShift Community

DeciShift is early-stage open source, so the most valuable community signal is **independent use**: people installing it, running the examples, finding edge cases, and reporting where the abstraction works or fails.

## Start here

- [GitHub Discussions](https://github.com/sauravsingla/DeciShift/discussions)
- [Independent reproducibility testing — issue #13](https://github.com/sauravsingla/DeciShift/issues/13)
- [Windows and macOS smoke testing — issue #14](https://github.com/sauravsingla/DeciShift/issues/14)
- [MLflow integration example — issue #15](https://github.com/sauravsingla/DeciShift/issues/15)
- [ZenML integration example — issue #16](https://github.com/sauravsingla/DeciShift/issues/16)

## Independent tester checklist

A useful reproduction report includes:

1. operating system;
2. Python version;
3. installation method;
4. example or dataset used;
5. runtime;
6. number and percentage of changed final actions;
7. whether attribution and Decision Contract results matched the documented output;
8. any installation friction, unclear terminology, unexpected behavior, or discrepancy.

Negative results are welcome. A reproducibility failure or confusing API is useful evidence, not noise.

## Discussion topics we especially want

- model metrics improve but operational decisions regress;
- low global shift with high cohort shift;
- threshold-only changes;
- feature transformation changes;
- policy/rule interactions;
- multiple models feeding one policy;
- categorical actions with no natural ordering;
- topology changes between versions;
- very large tabular datasets;
- attribution that conflicts with practitioner intuition.

## Integration feedback

DeciShift is designed to sit beside existing MLOps tools. Practical constraints from real workflows are especially useful:

- Should integrations consume run IDs, model versions, files, or artifacts?
- Where should a behavioral release gate sit in the pipeline?
- Which metadata should be retained for reproducibility?
- What would make an integration production-useful rather than another example notebook?

## Contributing

Please read the repository contribution guide before opening a pull request:

[CONTRIBUTING.md](https://github.com/sauravsingla/DeciShift/blob/main/CONTRIBUTING.md)

The project values reproducible examples, clear failure cases, focused integrations, tests, documentation, and corrections to over-broad claims.
