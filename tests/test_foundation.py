from datetime import timedelta

import numpy as np
import pandas as pd

from fconcord.foundation import _chronos_predict, _timesfm_predict


class _FakeTimesFM:
    def forecast(self, horizon, inputs):
        point = np.stack(
            [np.full(horizon, float(values[-1]), dtype=np.float32) for values in inputs]
        )
        return point, np.zeros((*point.shape, 10), dtype=np.float32)


class _FakeChronos:
    def predict_df(self, frame, target, prediction_length, **kwargs):
        del kwargs
        rows = []
        for item_id, context in frame.groupby("item_id"):
            first_future = pd.Timestamp(context["timestamp"].max()) + timedelta(days=1)
            for target_name in target:
                value = float(context[target_name].iloc[-1])
                for lead in range(prediction_length):
                    rows.append(
                        {
                            "item_id": item_id,
                            "target_name": target_name,
                            "timestamp": first_future + timedelta(days=lead),
                            "predictions": value,
                        }
                    )
        return pd.DataFrame(rows)


def test_timesfm_adapter_preserves_origin_asset_lead_order() -> None:
    histories = np.zeros((3, 5, 2), dtype=np.float32)
    for origin in range(3):
        for asset in range(2):
            histories[origin, -1, asset] = 10 * origin + asset
    result = _timesfm_predict(_FakeTimesFM(), histories, horizon=4, origin_batch_size=2)
    assert result.shape == (3, 4, 2)
    for origin in range(3):
        for asset in range(2):
            np.testing.assert_allclose(result[origin, :, asset], 10 * origin + asset)


def test_chronos_adapter_preserves_multivariate_origin_asset_order() -> None:
    histories = np.zeros((3, 5, 2), dtype=np.float32)
    for origin in range(3):
        for asset in range(2):
            histories[origin, -1, asset] = 10 * origin + asset
    result = _chronos_predict(_FakeChronos(), histories, horizon=4, origin_batch_size=2)
    assert result.shape == (3, 4, 2)
    for origin in range(3):
        for asset in range(2):
            np.testing.assert_allclose(result[origin, :, asset], 10 * origin + asset)
