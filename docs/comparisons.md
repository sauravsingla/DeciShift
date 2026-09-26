# Related categories

DeciShift overlaps with established categories but keeps a narrower object of analysis: **what changed in the final decisions/actions of a versioned executable decision system, and what software evidence supports that change?**

| Category | Typical question | DeciShift distinction |
|---|---|---|
| Model regression testing | Did behavior/metrics regress? | Replays the declared complete decision system and records discrete action transitions. |
| ML monitoring | Did production distributions or metrics move? | Works offline on supplied historical records and versioned local components. |
| Slice analysis | Which groups show different metrics? | Cohorts summarize where decision/action transitions concentrate. |
| Behavioral diffing | What behavior changed? | Adds version-aware structured execution, transition evidence and component/node substitution. |
| Shapley attribution | How is a numeric value allocated to players/features? | Applies established Shapley machinery to explicit versioned software substitutions/targets; it does not claim Shapley novelty. |
| Workflow/DAG engines | How do I orchestrate arbitrary tasks? | `DecisionFlow` is deliberately limited to deterministic row-aligned decision graphs; DeciShift is not a general orchestrator. |
| MLOps platforms | How do I host/register/deploy/monitor models? | No hosting, registry, feature store, training system, database, or cloud account is required. |
| CI release gates | Does a check satisfy declared limits? | Decision Contracts gate observed decision-change evidence with deterministic exit codes. |

## Information available from the same evaluation run

This matrix is about **information focus**, not a product ranking. Experiment trackers and evaluation/monitoring systems are extensible; the table describes what is typically available when they are used for their primary role versus what DeciShift computes explicitly.

| Information | Aggregate model metrics alone | Experiment tracking / registry | Evaluation / monitoring | DeciShift |
|---|---|---|---|---|
| Accuracy/AUC or other model metrics | Yes | Usually stored | Common | Optional/contextual |
| Parameters/artifact/version metadata | No | Primary focus | Sometimes | Component/node identities used for comparison |
| Per-record final action transition (`A→B`) | No | Only if custom-logged | Possible with custom evaluation | Explicit core output |
| Transition matrix across categorical actions | No | Only if custom-logged | Possible with custom evaluation | Explicit core output |
| Attribution to versioned software substitutions | No | No default interpretation | No default interpretation | Explicit software-counterfactual target |
| Cohort-specific action-shift limits | No | Custom | Possible | Decision Contracts |
| Deterministic release exit code from decision-change evidence | No | Custom | Custom | Built in |
| Tamper-evident saved comparison bundle | No | Artifact-dependent | Tool-dependent | SHA-256 manifest verification |

The intended integration pattern is complementary: track/evaluate the baseline and candidate using existing tooling, replay both through the declared decision system, use DeciShift for decision-level behavioral comparison and attribution, then consume the Decision Contract result in CI/CD.

`DecisionPipeline` remains the simple fixed binary path. `DecisionFlow` adds branching, merging, multiple models/policies/rules and categorical actions without turning DeciShift into a generic workflow platform.

Structural reachability, observed action changes, and software-counterfactual attribution are reported as different quantities. None is described as real-world causal impact.
