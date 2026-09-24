import numpy as np

from decishift.attribution import approximate_attribution, calculate_attribution_diagnostics
from decishift.diff.compare import compare_pipelines
from tests.helpers import frame, pipeline


def test_efficiency_is_not_monte_carlo_convergence():
    data = frame((0.41, 0.46, 0.49, 0.52, 0.58, 0.62))
    baseline = pipeline()
    candidate = pipeline(
        feature_shift=0.06,
        model_shift=0.06,
        threshold=0.48,
        labels={"features": "f2", "model": "m2", "threshold": "t2"},
    )
    result = compare_pipelines(baseline, candidate, data, id_column="id")
    attr = approximate_attribution(
        baseline,
        candidate,
        data,
        result=result,
        permutations=2,
        seed=7,
        uncertainty_width_threshold=1e-12,
        batch_size=1,
    )
    diagnostics = calculate_attribution_diagnostics(
        result,
        attr,
        method="approximate",
        permutations=2,
        tolerance=1e-8,
    )
    assert diagnostics.efficiency_valid
    assert not diagnostics.sampling_precision_sufficient
    assert not diagnostics.sampling_converged
    assert not diagnostics.converged


def test_adaptive_attribution_stops_early_when_precision_and_stability_are_met():
    data = frame((0.2, 0.3, 0.4, 0.5))
    baseline = pipeline()
    candidate = pipeline(model_shift=0.1, labels={"model": "m2"})
    attr = approximate_attribution(
        baseline,
        candidate,
        data,
        min_permutations=8,
        max_permutations=64,
        batch_size=4,
        target_ci_width=0.01,
        seed=3,
    )
    assert attr.attrs["adaptive"] is True
    assert attr.attrs["permutations_used"] == 8
    assert attr.attrs["stopped_early"] is True
    assert attr.attrs["sampling_precision_sufficient"] is True
    assert attr.attrs["sampling_converged"] is True
    assert (attr["permutations_used"] == 8).all()


def test_adaptive_attribution_uses_maximum_when_target_not_reached():
    data = frame((0.41, 0.46, 0.49, 0.52, 0.58, 0.62))
    baseline = pipeline()
    candidate = pipeline(
        feature_shift=0.06,
        model_shift=0.06,
        threshold=0.48,
        labels={"features": "f2", "model": "m2", "threshold": "t2"},
    )
    attr = approximate_attribution(
        baseline,
        candidate,
        data,
        min_permutations=4,
        max_permutations=12,
        batch_size=4,
        target_ci_width=1e-12,
        seed=5,
    )
    assert attr.attrs["permutations_used"] == 12
    assert attr.attrs["stopped_early"] is False
    assert attr.attrs["sampling_precision_sufficient"] is False
    assert np.isfinite(attr.attrs["max_ci_width"])


def test_fixed_permutation_mode_remains_exactly_fixed():
    data = frame((0.35, 0.48, 0.7, 0.55))
    baseline = pipeline()
    candidate = pipeline(feature_shift=0.05, model_shift=0.08, labels={"features": "f2", "model": "m2"})
    attr = approximate_attribution(baseline, candidate, data, permutations=37, seed=11)
    assert attr.attrs["adaptive"] is False
    assert attr.attrs["permutations_used"] == 37
    assert attr.attrs["stopped_early"] is False
    assert (attr["permutations_used"] == 37).all()
