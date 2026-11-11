import numpy as np

from fconcord.benchmark_data import BenchmarkTensorDataset, build_benchmark_arrays


def _fixture():
    steps, assets, factors, samples, horizon = 40, 3, 2, 12, 2
    origins = np.arange(10, 10 + samples)
    rng = np.random.default_rng(5)
    panel = {
        "returns": rng.normal(scale=0.01, size=(steps, assets)),
        "dollar_volume": rng.lognormal(size=(steps, assets)),
        "order_flow": rng.normal(size=(steps, assets)),
        "factors": rng.normal(size=(steps, factors)),
    }
    c0 = rng.normal(size=(samples, assets, 4)).astype(np.float32)
    a0 = rng.normal(size=(samples, assets, assets)).astype(np.float32)
    s0 = rng.normal(size=(samples, 3)).astype(np.float32)
    y = rng.normal(scale=0.01, size=(samples, horizon, assets)).astype(np.float32)
    prepared = {
        "arrays": {"origins": origins, "c0": c0, "a0": a0, "s0": s0, "y": y},
        "metadata": {
            "num_assets": assets,
            "horizon": horizon,
            "return_scale": [0.01, 0.02, 0.03],
            "split_indices": {
                "train": list(range(6)),
                "val": list(range(6, 9)),
                "test": list(range(9, 12)),
            },
        },
    }
    return prepared, panel


def test_benchmark_histories_are_causal_and_have_registered_dimensions() -> None:
    prepared, panel = _fixture()
    benchmark = build_benchmark_arrays(prepared, panel, history_length=5)
    arrays = benchmark["arrays"]
    assert arrays["return_history"].shape == (12, 5, 3)
    assert arrays["asset_features"].shape == (12, 5, 3, 3)
    assert arrays["market_history"].shape == (12, 5, 4)
    first_origin = int(prepared["arrays"]["origins"][0])
    expected = panel["returns"][first_origin - 4 : first_origin + 1, 0] / 0.01
    np.testing.assert_allclose(arrays["return_history"][0, :, 0], expected)
    assert benchmark["metadata"]["normalization_fit_scope"]["last_raw_row"] == 15


def test_benchmark_tensor_dataset_returns_targets_without_test_special_cases() -> None:
    prepared, panel = _fixture()
    benchmark = build_benchmark_arrays(prepared, panel, history_length=5)
    dataset = BenchmarkTensorDataset(benchmark, "val")
    row = dataset[0]
    assert len(dataset) == 3
    assert row["return_history"].shape == (5, 3)
    assert row["target"].shape == (2, 3)
    np.testing.assert_allclose(
        row["target_scaled"].numpy(),
        row["target"].numpy() / np.asarray([0.01, 0.02, 0.03]),
    )
