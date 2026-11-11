from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from fconcord.metrics import point_metrics


def evaluate_lightweight_baselines(
    prepared: dict[str, Any], ridge_alpha: float = 1.0
) -> dict[str, Any]:
    arrays = prepared["arrays"]
    splits = prepared["metadata"]["split_indices"]
    train = np.asarray(splits["train"], dtype=np.int64)
    test = np.asarray(splits["test"], dtype=np.int64)
    target = arrays["y"][test]
    horizon = target.shape[1]
    zero = np.zeros_like(target)
    last = np.repeat(arrays["last_return"][test, None, :], horizon, axis=1)
    historical = np.repeat(arrays["historical_mean"][test, None, :], horizon, axis=1)
    train_features = np.concatenate(
        [arrays["c0"][train].reshape(len(train), -1), arrays["s0"][train]], axis=1
    )
    test_features = np.concatenate(
        [arrays["c0"][test].reshape(len(test), -1), arrays["s0"][test]], axis=1
    )
    # LSQR is stable for the wide, correlated concept design and avoids the
    # ill-conditioned normal-equation solve used by the default auto solver.
    ridge = Ridge(alpha=ridge_alpha, solver="lsqr")
    ridge.fit(train_features, arrays["y"][train].reshape(len(train), -1))
    ridge_prediction = ridge.predict(test_features).reshape(target.shape)
    return {
        "ZeroReturn": point_metrics(target, zero),
        "LastReturn": point_metrics(target, last),
        "HistoricalMean": point_metrics(target, historical),
        "ConceptRidge": point_metrics(target, ridge_prediction),
    }
