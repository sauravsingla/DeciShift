# Architecture

- `core/`: pipeline abstraction and protocols
- `replay/`: hybrid replay cache
- `diff/`: record-level decision comparison
- `attribution/`: exact and permutation component attribution plus interactions
- `cohorts/`: categorical and numeric slice analysis
- `adapters/`: lightweight model-framework compatibility helpers
- `reports/`: terminal, JSON, and Markdown rendering
- `config.py`: strictly local YAML/data/component loading
- `store.py`: local machine-readable run evidence
- `cli.py`: `demo`, `compare`, `explain`, and `report`

No module requires a network service or background daemon.
