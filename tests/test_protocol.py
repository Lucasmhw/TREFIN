from __future__ import annotations

from fconcord.protocol import _aggregate


def test_protocol_aggregate_includes_mse() -> None:
    rows = [
        {
            "fold": "fold_1",
            "mse": 0.0004,
            "rmse": 0.02,
            "mae": 0.01,
            "directional_accuracy": 0.51,
            "energy_score": 0.03,
            "msic": 0.01,
            "msir": 0.02,
        }
    ]
    result = _aggregate(rows)
    assert result["overall"]["mse"]["mean"] == 0.0004
    assert result["by_fold"]["fold_1"]["mse"]["mean"] == 0.0004
