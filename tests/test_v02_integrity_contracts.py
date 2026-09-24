import json

import pandas as pd

from decishift.attribution import calculate_attribution_diagnostics, exact_attribution
from decishift.contracts import (
    EXIT_CONTRACT_VIOLATED, EXIT_INSUFFICIENT_EVIDENCE, EXIT_INTEGRITY_FAILURE, EXIT_PASS,
    evaluate_contract,
)
from decishift.diff.compare import compare_pipelines, compare_predictions
from decishift.fragility import analyze_fragility
from decishift.replay.engine import HybridReplayCache
from decishift.run_compare import compare_saved_runs
from decishift.store import RunStore
from tests.helpers import frame, pipeline


def make_saved(store: RunStore):
    data=frame((0.4,0.49,0.6,0.2))
    b=pipeline(); c=pipeline(threshold=0.45,labels={"threshold":"t2"})
    result=compare_pipelines(b,c,data,id_column="id")
    replay=HybridReplayCache(b,c,data)
    result.attribution=exact_attribution(b,c,data,result=result,cache=replay)
    result.metadata.update({"attribution_method":"exact","attribution_parameters":{},"random_seed":None})
    result.diagnostics=calculate_attribution_diagnostics(result,result.attribution,method="exact",hybrid_evaluations=replay.evaluations)
    analyze_fragility(result)
    return result,store.save(result)


def test_clean_run_verifies_and_manifest_has_required_fields(tmp_path):
    store=RunStore(tmp_path/"runs")
    _,run=make_saved(store)
    verification=store.verify(run)
    assert verification.passed
    manifest=json.loads((store.root/run/"manifest.json").read_text())
    assert manifest["evidence_schema_version"]=="1.0"
    assert manifest["integrity_root"].startswith("sha256:")
    for name in ["manifest.json","summary.json","records.csv","attribution.csv","interactions.csv","cohorts.csv","report.md","report.txt","report.html"]:
        assert (store.root/run/name).exists()


def test_modified_records_attribution_missing_artifact_and_manifest_fail(tmp_path):
    for target,mode in [("records.csv","modify"),("attribution.csv","modify"),("cohorts.csv","delete"),("manifest.json","modify")]:
        store=RunStore(tmp_path/target.replace(".","_")/"runs")
        _,run=make_saved(store); path=store.root/run/target
        if mode=="delete": path.unlink()
        elif target=="manifest.json":
            data=json.loads(path.read_text()); data["number_of_records"]+=1; path.write_text(json.dumps(data))
        else: path.write_text(path.read_text()+"\n#tampered")
        assert not store.verify(run).passed


def test_contract_pass_violation_insufficient_and_integrity_codes(tmp_path):
    store=RunStore(tmp_path/"runs")
    result,_=make_saved(store)
    good={"max_decision_shift_rate":1.0,"attribution":{"require_reproducible_identity":True,"max_score_efficiency_mae":1e-9}}
    assert evaluate_contract(result,good,integrity_passed=True).exit_code==EXIT_PASS
    bad={"max_decision_shift_rate":0.0}
    assert evaluate_contract(result,bad,integrity_passed=True).exit_code==EXIT_CONTRACT_VIOLATED
    pred_data=pd.DataFrame({"id":[1,2],"b":[.2,.7],"c":[.3,.8],"bd":[0,1],"cd":[0,1]})
    pred=compare_predictions(pred_data,baseline_score="b",candidate_score="c",baseline_decision="bd",candidate_decision="cd",id_column="id")
    insufficient={"attribution":{"max_score_efficiency_mae":0.1}}
    assert evaluate_contract(pred,insufficient,integrity_passed=True).exit_code==EXIT_INSUFFICIENT_EVIDENCE
    assert evaluate_contract(result,good,integrity_passed=False).exit_code==EXIT_INTEGRITY_FAILURE


def test_compare_saved_runs(tmp_path):
    store=RunStore(tmp_path/"runs")
    _,a=make_saved(store); _,b=make_saved(store)
    out=compare_saved_runs(a,b,store=store)
    assert out["decision_shift_rate"]["delta"]==0
    assert "component_attribution" in out


def test_manifest_path_traversal_is_rejected(tmp_path):
    store=RunStore(tmp_path/"runs")
    _,run=make_saved(store)
    manifest_path=store.root/run/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["evidence_artifact_sha256"]["../outside.txt"]="sha256:" + "0"*64
    from decishift.evidence import make_integrity_root
    manifest.pop("integrity_root",None)
    manifest["integrity_root"]=make_integrity_root(manifest)
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    verification=store.verify(run)
    assert not verification.passed
    assert any(item.artifact=="../outside.txt" and item.status=="FAIL" for item in verification.items)


def test_future_evidence_schema_is_rejected(tmp_path):
    store=RunStore(tmp_path/"runs")
    _,run=make_saved(store)
    manifest_path=store.root/run/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["evidence_schema_version"]="99.0"
    manifest_path.write_text(json.dumps(manifest))
    import pytest
    with pytest.raises(ValueError,match="Unsupported future evidence schema"):
        store.load(run)
