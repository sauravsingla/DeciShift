import numpy as np

from decishift.attribution import approximate_attribution, calculate_attribution_diagnostics, exact_attribution
from decishift.diff.compare import compare_pipelines
from decishift.replay.engine import HybridReplayCache
from tests.helpers import frame, pipeline


def test_approximate_uncertainty_columns_and_fixed_seed():
    data = frame((0.35, 0.48, 0.7, 0.55))
    b = pipeline()
    c = pipeline(feature_shift=0.05, model_shift=0.08, threshold=0.48, labels={"features":"f2","model":"m2","threshold":"t2"})
    result = compare_pipelines(b, c, data, id_column="id")
    one = approximate_attribution(b, c, data, result=result, permutations=100, seed=9, confidence_level=0.95)
    two = approximate_attribution(b, c, data, result=result, permutations=100, seed=9, confidence_level=0.95)
    required = {"score_standard_error","decision_standard_error","score_ci_low","score_ci_high","decision_ci_low","decision_ci_high","permutations_used"}
    assert required.issubset(one.columns)
    assert one.equals(two)
    assert (one["permutations_used"] == 100).all()


def test_ci_width_decreases_with_more_permutations_aggregately():
    data = frame((0.41, 0.46, 0.49, 0.52, 0.58, 0.62))
    b = pipeline()
    c = pipeline(feature_shift=0.06, model_shift=0.06, threshold=0.48, labels={"features":"f2","model":"m2","threshold":"t2"})
    small = approximate_attribution(b, c, data, permutations=30, seed=5)
    large = approximate_attribution(b, c, data, permutations=1000, seed=5)
    small_width = (small.score_ci_high - small.score_ci_low).mean()
    large_width = (large.score_ci_high - large.score_ci_low).mean()
    assert large_width < small_width


def test_exact_diagnostics_efficiency_and_dummy_component():
    data = frame((0.2, 0.3, 0.4))
    b = pipeline()
    c = pipeline(feature_shift=0.0, model_shift=0.1, labels={"features":"f2","model":"m2"})
    result = compare_pipelines(b, c, data, id_column="id")
    replay = HybridReplayCache(b, c, data)
    attr = exact_attribution(b, c, data, result=result, cache=replay)
    dummy = attr[attr.component == "features"]
    assert np.allclose(dummy.score_contribution, 0)
    assert np.allclose(dummy.decision_contribution, 0)
    diagnostics = calculate_attribution_diagnostics(result, attr, method="exact", hybrid_evaluations=replay.evaluations)
    assert diagnostics.score_efficiency_max < 1e-12
    assert diagnostics.decision_efficiency_max < 1e-12
    assert diagnostics.converged


def test_shapley_symmetry_for_equal_additive_changes():
    data = frame((0.10, 0.20, 0.30))
    b = pipeline()
    c = pipeline(feature_shift=0.05, model_shift=0.05, labels={"features":"f2","model":"m2"})
    result = compare_pipelines(b, c, data, id_column="id")
    attr = exact_attribution(b, c, data, result=result)
    means = attr.groupby("component")["score_contribution"].mean()
    assert np.isclose(means["features"], means["model"], atol=1e-12)
