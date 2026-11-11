from __future__ import annotations

import numpy as np

from fconcord.metrics import evaluate_predictions, point_metrics
from fconcord.squared_error_report import squared_error_metrics


def test_point_metrics_include_mse_with_rmse_identity() -> None:
    target = np.asarray([[[0.01], [-0.02]]], dtype=np.float64)
    prediction = np.asarray([[[0.00], [-0.01]]], dtype=np.float64)
    result = point_metrics(target, prediction)
    assert np.isclose(result["mse"], 0.0001)
    assert np.isclose(result["rmse"] ** 2, result["mse"])


def test_probabilistic_metrics_include_horizon_mse() -> None:
    target = np.asarray([[[0.01], [-0.02]]], dtype=np.float64)
    mean = np.asarray([[[0.00], [-0.01]]], dtype=np.float64)
    samples = np.stack([mean - 0.001, mean + 0.001], axis=0)
    result = evaluate_predictions(target, mean, samples, [0.9])
    assert np.isclose(result["mse"], result["rmse"] ** 2)
    assert all(np.isclose(item["mse"], item["rmse"] ** 2) for item in result["horizon"])


def test_squared_error_report_metric_uses_zero_return_reference() -> None:
    target = np.asarray([0.01, -0.02], dtype=np.float64)
    prediction = np.asarray([0.00, -0.01], dtype=np.float64)
    result = squared_error_metrics(target, prediction)
    assert np.isclose(result["mse"], 0.0001)
    assert np.isclose(result["zero_return_mse"], 0.00025)
    assert np.isclose(result["oos_r2_zero"], 0.6)
