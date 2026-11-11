import numpy as np

from fconcord.benchmark_training import residual_bootstrap_scenarios


def test_residual_bootstrap_is_reproducible_centered_and_joint() -> None:
    mean = np.zeros((4, 2, 3), dtype=np.float32)
    residual = np.arange(30, dtype=np.float32).reshape(5, 2, 3)
    first, first_indices = residual_bootstrap_scenarios(mean, residual, 12, seed=17)
    second, second_indices = residual_bootstrap_scenarios(mean, residual, 12, seed=17)
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_allclose(first, second)
    assert first.shape == (12, 4, 2, 3)
    centered = residual - residual.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(first[3, 2], centered[first_indices[3, 2]])
