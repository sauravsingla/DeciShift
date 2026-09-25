# Evaluation study: public flows, reproducible benchmarks, and decision-change evidence

## Scope

This study evaluates the current DeciShift object of analysis: changes in the final action produced by a versioned executable decision system. It does not introduce a new attribution algorithm.

The evaluation has three parts:

1. **Public-data flows** — one genuinely trained scikit-learn flow and one genuinely trained XGBoost flow.
2. **Reproducible DecisionFlow benchmarks** — 10K and 100K records by default, with an optional 1M run.
3. **Decision-change case study** — a model metric improves while aggregate action counts stay close, but individual actions still move.

The existing synthetic equipment-maintenance example remains unchanged.

## Public-data examples

### Wine / scikit-learn

`examples/public_wine_sklearn/run.py` uses scikit-learn's bundled Wine dataset and trains deterministic logistic-regression models. A score for `class_2` versus other classes drives the three-action routing policy `route_not_class2`, `manual_review`, `route_class2`.

The script runs five comparisons against the same baseline: feature-only, model-only, policy-only, rule-only, and all changes together.

### Digits / XGBoost

`examples/public_digits_xgboost/run.py` uses scikit-learn's bundled Digits dataset and trains deterministic XGBoost boosters for `digit 8` versus other digits. The score drives `auto_not_8`, `manual_review`, and `auto_8`.

The same five change-isolation scenarios are executed. The feature-only scenario changes pixel preprocessing; the model-only scenario changes the fitted booster; the policy-only scenario changes action thresholds; the rule-only scenario changes the manual-review override.

## Decision-change case study

The Digits script also creates a deliberately narrow case study:

- the candidate model changes;
- the candidate policy thresholds change;
- the candidate review rule changes;
- the feature transform stays fixed for this case study so the changed-node set is explicit.

The output records:

- baseline and candidate model accuracy and ROC AUC;
- baseline and candidate final-action distributions;
- every changed transition;
- exact software-counterfactual attribution across the changed nodes;
- cohort action-shift rates by actual digit;
- a Decision Contract with an `actual_digit=6` cohort override.

The generated case-study Markdown is published from CI as an artifact and can be committed as a snapshot tied to the source commit. No percentage or benchmark timing should be copied into documentation before a machine run produces it.

## Benchmark protocol

`benchmarks/evaluate_flow_attribution.py` runs the existing 5-node linear and 8-node branched DecisionFlow benchmark cases at:

- 10,000 records;
- 100,000 records;
- optionally 1,000,000 records with `--include-million`.

For each flow/size it records both **exact** and **sampled** attribution with:

- number of changed nodes and changed final actions;
- comparison wall time;
- attribution wall time;
- `tracemalloc` peak Python allocation;
- hybrid evaluations;
- nodes executed;
- node outputs reused;
- cache-reuse ratio;
- efficiency validity and MAE;
- permutation count;
- sampling precision/convergence;
- maximum confidence-interval width.

The sampled protocol uses a deterministic seed and bounded adaptive sampling of 16–64 permutations, in batches of 8, with a target CI width of 0.10. The table records whether that bounded run actually reaches the requested convergence target rather than assuming that it did.

Memory is explicitly `tracemalloc` peak Python allocation, **not process RSS**.

## Reproducibility and publication discipline

Every benchmark payload records the source commit SHA, Python version, NumPy version, pandas version and runner platform. The workflow never writes to `main` and never overwrites versioned release assets. It uploads commit-scoped artifacts.

Committed benchmark snapshots live under `benchmarks/results/` and must retain the source SHA that produced them. README tables should link to those generated snapshots rather than hand-entering timing or memory numbers.

## Interpretation limits

These examples show software behavior over fixed public evaluation splits. They do not establish real-world causality, fairness, safety, compliance or deployment fitness. Model metrics and aggregate action distributions are not substitutes for record-level decision-change evidence.
