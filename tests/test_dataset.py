import copy

import numpy as np
import pandas as pd
import pytest

from fconcord.data import build_research_dataset, generate_synthetic_market, load_panel_csv


def debug_data_config() -> dict:
    return {
        "horizon": 2,
        "embargo": 2,
        "graph_window": 24,
        "concept_scales": [5, 10],
        "market_scale": 10,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "ridge_alpha": 0.1,
        "graph_alpha": 0.08,
        "graph_gate_center": 0.08,
        "graph_gate_temperature": 0.04,
        "projection_quantiles": [0.001, 0.999],
        "projection_margin_fraction": 0.05,
        "graph_bound_quantile": 0.999,
        "graph_bound_margin_fraction": 0.05,
        "order_flow_mode": "provided_interval",
    }


def test_future_perturbation_does_not_change_early_origin_state() -> None:
    panel = generate_synthetic_market(110, 5, 2, seed=7)
    perturbed = copy.deepcopy(panel)
    perturbed["returns"] = perturbed["returns"].copy()
    perturbed["returns"][-10:] = np.random.default_rng(11).normal(size=(10, 5))
    first = build_research_dataset(panel, debug_data_config())
    second = build_research_dataset(perturbed, debug_data_config())
    np.testing.assert_allclose(first["arrays"]["c0"][0], second["arrays"]["c0"][0])
    np.testing.assert_allclose(first["arrays"]["a0"][0], second["arrays"]["a0"][0])


def test_multi_horizon_splits_are_purged_and_labels_are_exact() -> None:
    panel = generate_synthetic_market(120, 5, 2, seed=19)
    prepared = build_research_dataset(panel, debug_data_config())
    arrays = prepared["arrays"]
    metadata = prepared["metadata"]
    origins = arrays["origins"]
    horizon = metadata["horizon"]
    train = np.asarray(metadata["split_indices"]["train"])
    validation = np.asarray(metadata["split_indices"]["val"])
    test = np.asarray(metadata["split_indices"]["test"])
    assert origins[train[-1]] + horizon < origins[validation[0]]
    assert origins[validation[-1]] + horizon < origins[test[0]]
    ranges = metadata["split_date_ranges"]
    assert ranges["train"]["samples"] == len(train)
    assert ranges["val"]["samples"] == len(validation)
    assert ranges["test"]["samples"] == len(test)
    for position in [0, len(origins) // 2, len(origins) - 1]:
        origin = origins[position]
        expected = np.stack(
            [panel["returns"][origin + lead] for lead in range(1, horizon + 1)]
        )
        np.testing.assert_allclose(arrays["y"][position], expected)
        np.testing.assert_allclose(arrays["c_target"][position, 0], arrays["c0"][position])
        np.testing.assert_allclose(arrays["a_target"][position, 0], arrays["a0"][position])
    assert arrays["c_target"].shape[1] == horizon + 1
    assert arrays["a_target"].shape[1] == horizon + 1
    assert np.linalg.norm(arrays["a0"], axis=(-2, -1)).max() <= metadata["graph_bound"] + 1e-6


def test_graph_estimation_diagnostics_are_finite() -> None:
    prepared = build_research_dataset(
        generate_synthetic_market(100, 4, 2, seed=23), debug_data_config()
    )
    arrays = prepared["arrays"]
    assert np.isfinite(arrays["graph_alpha_used"]).all()
    assert np.all(arrays["graph_alpha_used"] > 0)
    assert 0.0 <= prepared["metadata"]["graph_estimation"]["fallback_rate"] <= 1.0


def test_slow_graph_is_refit_on_registered_causal_cadence() -> None:
    config = debug_data_config()
    config["graph_refit_stride"] = 5
    prepared = build_research_dataset(
        generate_synthetic_market(110, 4, 2, seed=31), config
    )
    arrays = prepared["arrays"]
    origins = arrays["origins"]
    expected = (origins - origins[0]) % 5 == 0
    np.testing.assert_array_equal(arrays["graph_refit"], expected)
    metadata = prepared["metadata"]["graph_estimation"]
    assert metadata["refit_stride"] == 5
    assert metadata["refit_anchor_origin"] == 23
    assert metadata["refit_count"] == int(expected.sum())
    assert metadata["carry_forward_is_causal"] is True
    for position in np.flatnonzero(~expected):
        np.testing.assert_allclose(
            arrays["partial_corr"][position], arrays["partial_corr"][position - 1]
        )


def test_csv_loader_validates_financial_panel(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = [
        {
            "date": date,
            "asset": asset,
            "price": 100.0 + day,
            "dollar_volume": 1_000_000.0,
            "order_flow": 0.1,
        }
        for day, date in enumerate(dates)
        for asset in ["A", "B"]
    ]
    panel_path = tmp_path / "panel.csv"
    factor_path = tmp_path / "factors.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)
    pd.DataFrame({"date": dates, "MKT": [0.01, -0.02, 0.005]}).to_csv(
        factor_path, index=False
    )
    loaded = load_panel_csv(
        panel_path, factor_path, {"risk_free_policy": "explicit_zero"}
    )
    np.testing.assert_allclose(loaded["returns"], loaded["total_log_returns"])
    assert loaded["returns"].shape == (3, 2)
    duplicated = pd.DataFrame(rows + [rows[0]])
    duplicated.to_csv(panel_path, index=False)
    with pytest.raises(ValueError, match="duplicate date-asset"):
        load_panel_csv(panel_path, factor_path, {"risk_free_policy": "explicit_zero"})


def test_invalid_split_fractions_are_rejected() -> None:
    config = debug_data_config()
    config["train_fraction"] = 0.8
    config["val_fraction"] = 0.2
    with pytest.raises(ValueError, match="sum to less than one"):
        build_research_dataset(generate_synthetic_market(100, 4, 2, seed=29), config)


def test_explicit_date_fold_is_purged_without_using_fraction_boundaries() -> None:
    panel = generate_synthetic_market(160, 4, 2, seed=43)
    config = debug_data_config()
    config["split_dates"] = {
        "train_start": "2000-01-24",
        "train_end": "2000-03-15",
        "val_start": "2000-03-16",
        "val_end": "2000-04-15",
        "test_start": "2000-04-16",
        "test_end": "2000-05-25",
    }
    prepared = build_research_dataset(panel, config)
    metadata = prepared["metadata"]
    origins = prepared["arrays"]["origins"]
    train = np.asarray(metadata["split_indices"]["train"])
    validation = np.asarray(metadata["split_indices"]["val"])
    test = np.asarray(metadata["split_indices"]["test"])
    assert metadata["split_mode"] == "explicit_date_rolling_fold"
    assert metadata["split_fractions"] is None
    assert metadata["split_label_end_enforced"] is True
    assert origins[train[-1]] + metadata["embargo"] < origins[validation[0]]
    assert origins[validation[-1]] + metadata["embargo"] < origins[test[0]]
    for split, end_key in {
        "train": "train_end",
        "val": "val_end",
        "test": "test_end",
    }.items():
        assert np.datetime64(metadata["split_date_ranges"][split]["last_label_date"]) <= (
            np.datetime64(metadata["split_boundaries"][end_key])
        )


def test_explicit_hourly_boundaries_on_same_day_remain_distinct() -> None:
    panel = generate_synthetic_market(180, 4, 2, seed=47)
    start = np.datetime64("2000-01-01T00:00:00", "ns")
    panel["dates"] = start + np.arange(180).astype("timedelta64[h]")
    config = debug_data_config()
    config["split_dates"] = {
        "train_start": "2000-01-01 23:00:00",
        "train_end": "2000-01-04 12:00:00",
        "val_start": "2000-01-04 13:00:00",
        "val_end": "2000-01-05 18:00:00",
        "test_start": "2000-01-05 19:00:00",
        "test_end": "2000-01-07 23:00:00",
    }
    prepared = build_research_dataset(panel, config)
    boundaries = prepared["metadata"]["split_boundaries"]
    assert boundaries["train_end"].startswith("2000-01-04T12:00")
    assert boundaries["val_start"].startswith("2000-01-04T13:00")
