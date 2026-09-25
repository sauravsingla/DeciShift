from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def test_xgboost_adapter_matches_real_booster_predictions() -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost optional dependency is not installed")

    import xgboost as xgb

    from decishift.adapters.xgboost import adapt

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    matrix = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "max_depth": 1,
            "eta": 1.0,
            "verbosity": 0,
            "nthread": 1,
            "seed": 0,
        },
        matrix,
        num_boost_round=2,
    )

    adapter = adapt(booster, version="xgb-real-v1")

    np.testing.assert_allclose(adapter.predict(X), booster.predict(matrix))
    assert adapter.version == "xgb-real-v1"


def test_lightgbm_adapter_matches_real_booster_predictions() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm optional dependency is not installed")

    import lightgbm as lgb

    from decishift.adapters.lightgbm import adapt

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    dataset = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "num_leaves": 3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "feature_pre_filter": False,
            "num_threads": 1,
            "seed": 0,
        },
        dataset,
        num_boost_round=2,
    )

    adapter = adapt(booster, version="lgb-real-v1")

    np.testing.assert_allclose(adapter.predict(X), booster.predict(X))
    assert adapter.version == "lgb-real-v1"
