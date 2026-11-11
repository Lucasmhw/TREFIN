import numpy as np
import torch

from fconcord.losses import energy_score
from fconcord.metrics import energy_score_numpy, multi_step_information_metrics


def _manual_energy(samples: np.ndarray, target: np.ndarray) -> float:
    sample_count = samples.shape[0]
    first = np.linalg.norm(samples - target[None], axis=-1).mean(axis=0)
    ordered = np.zeros_like(first, dtype=np.float64)
    for first_index in range(sample_count):
        for second_index in range(sample_count):
            if first_index != second_index:
                ordered += np.linalg.norm(
                    samples[first_index] - samples[second_index], axis=-1
                )
    second = ordered / (2 * sample_count * (sample_count - 1))
    return float(np.mean(first - second))


def test_energy_score_uses_every_distinct_sample_pair() -> None:
    rng = np.random.default_rng(73)
    samples = rng.normal(size=(5, 3, 2, 4)).astype(np.float32)
    target = rng.normal(size=(3, 2, 4)).astype(np.float32)
    expected = _manual_energy(samples, target)
    actual_numpy = energy_score_numpy(samples, target)
    actual_torch = float(energy_score(torch.from_numpy(samples), torch.from_numpy(target)))
    np.testing.assert_allclose(actual_numpy, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual_torch, expected, rtol=1e-6, atol=1e-6)


def test_fintsbridge_information_metrics_match_released_code_definition() -> None:
    target = np.asarray(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[1.0], [3.0], [2.0], [5.0]],
            [[4.0], [1.0], [3.0], [2.0]],
        ]
    )
    prediction = np.asarray(
        [
            [[1.0], [2.5], [2.0], [5.0]],
            [[2.0], [2.0], [4.0], [4.5]],
            [[3.0], [2.0], [4.0], [1.0]],
        ]
    )
    target_centered = target - target.mean(axis=1, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    correlations = np.mean(target_centered * prediction_centered, axis=1) / (
        target.std(axis=1) * prediction.std(axis=1)
    )
    expected_ic = float(correlations.mean())
    expected_ir = float(correlations.mean(axis=0)[0] / correlations.std(axis=0)[0])
    result = multi_step_information_metrics(target, prediction)
    assert np.isclose(result["msic"], expected_ic)
    assert np.isclose(result["msir"], expected_ir)
    assert result["valid_origin_asset_fraction"] == 1.0
