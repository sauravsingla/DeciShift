import numpy as np

from decishift.diff.compare import compare_pipelines
from tests.helpers import frame, pipeline


def test_threshold_only_change():
    data = frame((0.49, 0.51))
    b = pipeline(threshold=0.5)
    c = pipeline(threshold=0.48, labels={"threshold": "candidate"})
    result = compare_pipelines(b, c, data, id_column="id")
    assert result.summary()["changed_decisions"] == 1
    assert result.changed_components == ["threshold"]


def test_model_only_change():
    data = frame((0.4, 0.6))
    b = pipeline()
    c = pipeline(model_shift=0.2, labels={"model": "model_v2"})
    result = compare_pipelines(b, c, data)
    assert result.records["changed"].sum() == 1
    assert result.changed_components == ["model"]


def test_feature_only_change():
    data = frame((0.4, 0.6))
    b = pipeline()
    c = pipeline(feature_shift=0.15, labels={"features": "features_v2"})
    result = compare_pipelines(b, c, data)
    assert result.records["changed"].sum() == 1
    assert result.changed_components == ["features"]


def test_rules_only_change():
    data = frame((0.1, 0.9, 0.2))
    b = pipeline(rules=False)
    c = pipeline(rules=True, labels={"rules": "rules_v2"})
    result = compare_pipelines(b, c, data)
    assert result.records.loc[2, "candidate_decision"] == 1
    assert result.records.loc[2, "baseline_decision"] == 0
    assert result.changed_components == ["rules"]


def test_no_change_pipeline():
    data = frame()
    b = pipeline()
    c = pipeline()
    result = compare_pipelines(b, c, data)
    assert result.summary()["changed_decisions"] == 0
    assert result.changed_components == []


def test_opposing_component_changes_can_cancel_decision_effect():
    data = frame((0.45,))
    b = pipeline(threshold=0.5)
    c = pipeline(model_shift=0.10, threshold=0.60, labels={"model": "m2", "threshold": "t2"})
    result = compare_pipelines(b, c, data)
    assert result.records.loc[0, "baseline_decision"] == 0
    assert result.records.loc[0, "candidate_decision"] == 0
    assert set(result.changed_components) == {"model", "threshold"}
    assert np.isclose(result.records.loc[0, "score_delta"], 0.10)


def test_calibration_only_change():
    data = frame((0.49, 0.7))
    b = pipeline(scale=1.0)
    c = pipeline(scale=1.1, labels={"calibrator": "calibration_v2"})
    result = compare_pipelines(b, c, data)
    assert result.records["changed"].sum() == 1
    assert result.changed_components == ["calibrator"]
