import numpy as np
import pandas as pd

from decishift.cohorts import analyze_cohorts, select_cohort_columns, wilson_interval
from decishift.diff.compare import ComparisonResult, compare_pipelines
from decishift.fragility import analyze_fragility
from decishift.outcome import analyze_outcomes
from tests.helpers import pipeline


def test_auto_cohorts_exclude_ids_high_cardinality_text_and_datetime():
    n=100
    data=pd.DataFrame({
        "record_id":range(n),
        "segment":["a","b"]*50,
        "unique_name":[f"u-{i}" for i in range(n)],
        "notes":["this is a very long free text description "*4]*n,
        "when":pd.date_range("2026-01-01",periods=n),
        "small_numeric":np.tile([1,2,3,4],25),
    })
    selected,reasons=select_cohort_columns(data,id_column="record_id")
    assert "segment" in selected and "small_numeric" in selected
    assert "record_id" not in selected and "unique_name" not in selected and "notes" not in selected and "when" not in selected
    assert reasons["when"]=="datetime cohorting disabled"


def test_wilson_interval_and_minimum_size():
    low,high=wilson_interval(5,10)
    assert 0 < low < 0.5 < high < 1
    data=pd.DataFrame({"x":[0.55]*10+[0.2]*10,"segment":["a"]*10+["b"]*10,"force":False})
    result=compare_pipelines(pipeline(threshold=0.6),pipeline(threshold=0.5,labels={"threshold":"t2"}),data)
    cohorts=analyze_cohorts(data,result,columns=["segment"],min_size=11)
    assert cohorts.empty


def test_outcome_transition_table_has_all_four_cells_and_undefined_precision():
    source=pd.DataFrame({"y":[0,0,1,1]})
    records=pd.DataFrame({
        "record_id":[0,1,2,3],"baseline_score":[0]*4,"candidate_score":[0]*4,"score_delta":[0]*4,
        "baseline_threshold":[.5]*4,"candidate_threshold":[.5]*4,"baseline_margin":[0]*4,"candidate_margin":[0]*4,
        "baseline_decision":[0,0,0,1],"candidate_decision":[0,1,1,0],"changed":[False,True,True,True],"direction":["unchanged","0->1","0->1","1->0"],
    })
    result=ComparisonResult(records=records)
    out=analyze_outcomes(source,result,outcome_column="y")
    t=out["transition_table"]
    assert set(t)=={"baseline_correct_to_candidate_correct","baseline_correct_to_candidate_wrong","baseline_wrong_to_candidate_correct","baseline_wrong_to_candidate_wrong"}
    zero_source=pd.DataFrame({"y":[0,0]})
    zero_records=records.iloc[:2].copy(); zero_records["candidate_decision"]=[0,0]; zero_records["baseline_decision"]=[0,0]
    zero=ComparisonResult(records=zero_records)
    o2=analyze_outcomes(zero_source,zero,outcome_column="y")
    assert o2["baseline_precision"] is None and o2["baseline_recall"] is None


def test_fragility_boundary_and_rule_forced_classification():
    data=pd.DataFrame({"x":[0.49,0.1,0.2],"force":[False,False,True],"id":[1,2,3]})
    b=pipeline(threshold=0.5,rules=False)
    c=pipeline(threshold=0.48,rules=True,labels={"threshold":"t2","rules":"r2"})
    result=compare_pipelines(b,c,data,id_column="id")
    frag=analyze_fragility(result,boundary_bands=[0.01,0.05])
    classes=dict(zip(frag["records"].record_id,frag["records"].change_classification))
    assert classes[1]=="boundary-crossing change"
    assert classes[3]=="rule-forced change"
    assert "fraction_within_0.01_of_boundary" in frag["summary"]


def test_dataframe_fingerprint_handles_unhashable_object_values():
    from decishift.evidence import fingerprint_dataframe
    data = pd.DataFrame({"id": [1, 2], "payload": [[1, 2], {"b": 2, "a": 1}]})
    one = fingerprint_dataframe(data)
    two = fingerprint_dataframe(data.copy())
    assert one == two
    assert one.startswith("sha256:")
