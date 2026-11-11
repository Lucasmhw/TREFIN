import numpy as np

from fconcord.concepts import exponential_lag_weights, fit_weighted_ridge_factor_model


def test_exponential_weights_are_causal_and_normalized() -> None:
    weights = exponential_lag_weights(21)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    assert np.all(np.diff(weights) < 0)


def test_weighted_ridge_recovers_factor_loading() -> None:
    rng = np.random.default_rng(3)
    factors = rng.normal(size=(200, 2))
    true_beta = np.array([[1.2, -0.4], [0.5, 0.8]])
    intercept = np.array([0.01, -0.02])
    returns = intercept + factors @ true_beta.T + rng.normal(0, 0.01, size=(200, 2))
    estimated_intercept, estimated_beta = fit_weighted_ridge_factor_model(
        returns, factors, exponential_lag_weights(200, tau=1000), ridge_alpha=1e-5
    )
    np.testing.assert_allclose(estimated_intercept, intercept, atol=5e-3)
    np.testing.assert_allclose(estimated_beta, true_beta, atol=3e-2)
