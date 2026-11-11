import numpy as np

from fconcord.calibration import calibrate_predictable_signal


def test_predictable_signal_calibration_recovers_shrunken_validation_signal() -> None:
    rng = np.random.default_rng(123)
    mean = rng.normal(scale=0.02, size=(80, 3, 5))
    innovations = rng.normal(scale=0.01, size=(48, 80, 3, 5))
    samples = mean[None] + innovations
    target = 0.3 * mean
    report = calibrate_predictable_signal(mean, samples, target, lower=0.05)
    assert 0.20 < report["selected_signal_shrinkage"] < 0.40
    assert (
        report["calibrated_validation_metrics"]["energy_score"]
        <= report["uncalibrated_validation_metrics"]["energy_score"]
    )
    assert report["pairwise_invariance_max_error"] < 1e-12


def test_predictable_signal_calibration_validates_bounds_and_shapes() -> None:
    mean = np.zeros((2, 3, 4))
    samples = np.zeros((4, 2, 3, 4))
    target = np.zeros_like(mean)
    report = calibrate_predictable_signal(
        mean, samples, target, lower=0.25, upper=0.25
    )
    assert report["selected_signal_shrinkage"] == 0.25

    try:
        calibrate_predictable_signal(mean, samples[:, :, :, :3], target)
    except ValueError as error:
        assert "samples must have shape" in str(error)
    else:
        raise AssertionError("shape mismatch was not rejected")
