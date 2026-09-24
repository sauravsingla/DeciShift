# Related categories

DeciShift overlaps with several established categories but has a narrower object of analysis.

| Category | Typical question | DeciShift distinction |
|---|---|---|
| Model regression testing | Did behavior/metrics regress? | Replays the complete structured decision pipeline and records discrete transitions. |
| ML monitoring | Did production distributions or metrics move? | Works offline on declared historical records and versioned components. |
| Slice analysis | Which groups show different metrics? | Cohorts summarize where **decision transitions** concentrate. |
| Behavioral diffing | What behavior changed? | Attributes changes across features/model/calibration/threshold/rules. |
| Shapley attribution | How is an output allocated to players/features? | Uses Shapley ideas for versioned software-component substitutions; does not claim Shapley novelty. |
| Unit-change attribution | Which changed unit contributed? | Treats pipeline components as structured versioned units with replay evidence. |
| CI release gates | Does a check satisfy declared limits? | Decision Contracts gate observed decision-change evidence with deterministic exit codes. |

DeciShift's specific object of analysis is **the discrete decision transition produced by a versioned structured decision pipeline**.
