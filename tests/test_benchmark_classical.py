import numpy as np

from fconcord.benchmark_classical import _long_tree_features, _var_features


def _benchmark():
    rng = np.random.default_rng(1)
    samples, length, assets, concepts, horizon = 10, 8, 3, 4, 2
    arrays = {
        "return_history": rng.normal(size=(samples, length, assets)).astype(np.float32),
        "concept_state": rng.normal(size=(samples, assets, concepts)).astype(np.float32),
        "graph_state": rng.normal(size=(samples, assets, assets)).astype(np.float32),
        "market_state": rng.normal(size=(samples, 3)).astype(np.float32),
        "target_scaled": rng.normal(size=(samples, horizon, assets)).astype(np.float32),
    }
    return {
        "arrays": arrays,
        "metadata": {
            "split_indices": {"train": list(range(6)), "val": [6, 7], "test": [8, 9]},
            "dimensions": {"num_assets": assets, "horizon": horizon},
        },
    }


def test_var_and_long_tree_designs_preserve_registered_output_axes() -> None:
    benchmark = _benchmark()
    var_x, var_y, var_indices = _var_features(benchmark, "train", lags=5)
    tree_x, tree_y, tree_indices = _long_tree_features(
        benchmark, "train", lags=5
    )
    assert var_x.shape == (6, 15)
    assert var_y.shape == (6, 6)
    assert tree_x.shape[0] == tree_y.shape[0] == 6 * 3 * 2
    np.testing.assert_array_equal(var_indices, tree_indices)
