import numpy as np
import pandas as pd

from decishift.cohorts import analyze_cohorts
from decishift.diff.compare import compare_pipelines, compare_predictions
from tests.helpers import pipeline


def test_predictions_only_mode_and_missing_component_evidence():
    data = pd.DataFrame({
        "id": [1, 2, 3],
        "b_score": [0.2, 0.6, 0.49],
        "c_score": [0.3, 0.55, 0.52],
        "b_dec": [0, 1, 0],
        "c_dec": [0, 1, 1],
        "b_thr": [0.5, 0.5, 0.5],
        "c_thr": [0.5, 0.5, 0.5],
    })
    result = compare_predictions(
        data,
        baseline_score="b_score",
        candidate_score="c_score",
        baseline_decision="b_dec",
        candidate_decision="c_dec",
        baseline_threshold="b_thr",
        candidate_threshold="c_thr",
        id_column="id",
    )
    assert result.summary()["changed_decisions"] == 1
    assert result.attribution is None
    assert result.summary()["attribution_status"] == "insufficient evidence"


def test_numeric_cohort_generation():
    n = 100
    data = pd.DataFrame({"x": np.linspace(0, 1, n), "force": False})
    baseline = pipeline(threshold=0.6)
    candidate = pipeline(threshold=0.5, labels={"threshold": "t2"})
    result = compare_pipelines(baseline, candidate, data)
    cohorts = analyze_cohorts(data, result, columns=["x"], quantiles=4, min_size=10)
    assert len(cohorts) == 4
    assert (cohorts["dimension"] == "x").all()


def test_categorical_cohort_generation():
    n = 100
    groups = ["a"] * 50 + ["b"] * 50
    x = [0.55] * 50 + [0.30] * 50
    data = pd.DataFrame({"x": x, "force": False, "segment": groups})
    baseline = pipeline(threshold=0.6)
    candidate = pipeline(threshold=0.5, labels={"threshold": "t2"})
    result = compare_pipelines(baseline, candidate, data)
    cohorts = analyze_cohorts(data, result, columns=["segment"], min_size=10)
    a = cohorts[cohorts["cohort"] == "a"].iloc[0]
    b = cohorts[cohorts["cohort"] == "b"].iloc[0]
    assert a.flip_rate == 1.0
    assert b.flip_rate == 0.0
