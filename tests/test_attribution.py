import numpy as np

from decishift.attribution import approximate_attribution, exact_attribution, pairwise_interactions
from decishift.diff.compare import compare_pipelines
from tests.helpers import frame, pipeline


def interaction_case():
    data = frame((0.40,))
    baseline = pipeline(feature_shift=0.0, threshold=0.50)
    candidate = pipeline(
        feature_shift=0.05,
        threshold=0.45,
        labels={"features": "features_v2", "threshold": "threshold_v2"},
    )
    return data, baseline, candidate


def test_interaction_only_flip_is_surfaced():
    data, baseline, candidate = interaction_case()
    result = compare_pipelines(baseline, candidate, data, id_column="id")
    assert result.records.loc[0, "changed"]
    interactions = pairwise_interactions(baseline, candidate, data, result=result)
    row = interactions[(interactions.component_a == "features") & (interactions.component_b == "threshold")].iloc[0]
    assert bool(row.interaction_only_flip)
    assert row.decision_interaction == 1


def test_exact_attribution_adds_to_score_and_decision_difference():
    data = frame((0.35, 0.48, 0.7))
    baseline = pipeline()
    candidate = pipeline(
        feature_shift=0.05,
        model_shift=0.08,
        scale=0.95,
        threshold=0.48,
        rules=True,
        labels={"features": "f2", "model": "m2", "calibrator": "c2", "threshold": "t2", "rules": "r2"},
    )
    result = compare_pipelines(baseline, candidate, data, id_column="id")
    attr = exact_attribution(baseline, candidate, data, result=result)
    score_sum = attr.groupby("record_id")["score_contribution"].sum().sort_index().to_numpy()
    decision_sum = attr.groupby("record_id")["decision_contribution"].sum().sort_index().to_numpy()
    expected_score = result.records.sort_values("record_id")["score_delta"].to_numpy()
    expected_decision = (
        result.records.sort_values("record_id")["candidate_decision"].to_numpy()
        - result.records.sort_values("record_id")["baseline_decision"].to_numpy()
    )
    assert np.allclose(score_sum, expected_score)
    assert np.allclose(decision_sum, expected_decision)


def test_approximate_attribution_converges_to_exact():
    data = frame((0.35, 0.48, 0.7, 0.55))
    baseline = pipeline()
    candidate = pipeline(feature_shift=0.05, model_shift=0.08, threshold=0.48, labels={"features": "f2", "model": "m2", "threshold": "t2"})
    result = compare_pipelines(baseline, candidate, data, id_column="id")
    exact = exact_attribution(baseline, candidate, data, result=result)
    approx = approximate_attribution(baseline, candidate, data, result=result, permutations=2000, seed=42)
    e = exact.groupby("component")["score_contribution"].mean().sort_index()
    a = approx.groupby("component")["score_contribution"].mean().sort_index()
    assert np.allclose(a.to_numpy(), e.to_numpy(), atol=0.01)


def test_approximate_attribution_is_deterministic_with_seed():
    data = frame((0.35, 0.48, 0.7))
    baseline = pipeline()
    candidate = pipeline(feature_shift=0.05, model_shift=0.08, labels={"features": "f2", "model": "m2"})
    one = approximate_attribution(baseline, candidate, data, permutations=100, seed=123)
    two = approximate_attribution(baseline, candidate, data, permutations=100, seed=123)
    assert one.equals(two)


def test_hybrid_replay_reuses_unchanged_upstream_components():
    import pandas as pd
    from decishift.core.pipeline import DecisionPipeline
    from decishift.replay.engine import HybridReplayCache

    calls = {"features": 0, "model": 0}

    class Features:
        version = "f1"
        def transform(self, frame):
            calls["features"] += 1
            return frame[["x"]].to_numpy()

    class Model:
        version = "m1"
        def predict(self, X):
            calls["model"] += 1
            return X[:, 0]

    data = pd.DataFrame({"x": [0.4, 0.6]})
    baseline = DecisionPipeline(features=Features(), model=Model(), threshold=0.5, versions={"features": "f1", "model": "m1", "threshold": "t1", "calibrator": "identity", "rules": "identity"})
    candidate = DecisionPipeline(features=baseline.features, model=baseline.model, threshold=0.45, versions={"features": "f1", "model": "m1", "threshold": "t2", "calibrator": "identity", "rules": "identity"})
    replay = HybridReplayCache(baseline, candidate, data)
    replay.evaluate(set())
    replay.evaluate({"threshold"})
    assert calls == {"features": 1, "model": 1}
