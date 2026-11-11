import numpy as np

from fconcord.statistical_tests import (
    block_bootstrap_spa,
    circular_block_bootstrap_mean_difference,
    diebold_mariano_hac,
    holm_adjust,
    model_confidence_set,
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)


def test_per_origin_losses_have_expected_shape_and_energy_mean() -> None:
    target = np.zeros((4, 2, 3), dtype=np.float64)
    prediction = np.ones_like(target)
    samples = np.stack([prediction, 2.0 * prediction, -prediction], axis=0)
    squared = per_origin_squared_error(target, prediction)
    absolute = per_origin_absolute_error(target, prediction)
    energy = per_origin_energy_score(samples, target)
    assert squared.shape == absolute.shape == energy.shape == (4,)
    np.testing.assert_allclose(squared, 1.0)
    np.testing.assert_allclose(absolute, 1.0)
    assert np.all(np.isfinite(energy))


def test_dm_and_block_bootstrap_detect_uniform_candidate_improvement() -> None:
    reference = np.linspace(1.0, 2.0, 200)
    candidate = reference - 0.2 + 0.01 * np.sin(np.arange(200))
    dm = diebold_mariano_hac(candidate, reference, horizon=5)
    bootstrap = circular_block_bootstrap_mean_difference(
        candidate, reference, block_length=5, repetitions=500, seed=9
    )
    assert dm["mean_loss_difference_candidate_minus_reference"] < 0.0
    assert dm["two_sided_p_value"] < 0.01
    assert bootstrap["confidence_interval"][1] < 0.0


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert set(adjusted) == {"a", "b", "c"}
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_spa_and_mcs_detect_a_uniformly_inferior_model() -> None:
    rng = np.random.default_rng(101)
    reference = 1.0 + rng.normal(scale=0.02, size=240)
    good = reference - 0.10 + rng.normal(scale=0.005, size=240)
    bad = reference + 0.20 + rng.normal(scale=0.005, size=240)
    spa = block_bootstrap_spa(
        {"good": good, "bad": bad},
        reference,
        block_length=5,
        repetitions=300,
        seed=7,
    )
    mcs = model_confidence_set(
        {"reference": reference, "good": good, "bad": bad},
        block_length=5,
        repetitions=300,
        seed=7,
    )
    assert spa["best_candidate"] == "good"
    assert spa["p_value"] < 0.05
    assert "bad" in mcs["excluded_models"]
    assert "good" in mcs["included_models"]
