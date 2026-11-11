import pytest
import torch

from fconcord.benchmark_models import make_benchmark_model


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("StateMLP", {"hidden_dim": 16}),
        ("RawHistoryMLP", {"hidden_dim": 16}),
        ("DLinear", {"moving_average": 5, "individual": False}),
        ("GRU", {"hidden_dim": 16, "layers": 1, "dropout": 0.0}),
        (
            "TimeMixer",
            {"downsampling_factors": [1, 2, 4], "moving_average": 5, "dropout": 0.0},
        ),
        (
            "PatchTST",
            {"patch_size": 4, "stride": 2, "d_model": 8, "heads": 2, "layers": 1},
        ),
        ("iTransformer", {"d_model": 8, "heads": 2, "layers": 1}),
        ("MASTER", {"d_model": 8, "heads": 2, "dropout": 0.0}),
    ],
)
def test_benchmark_model_output_shape(name: str, parameters: dict) -> None:
    dimensions = {
        "num_assets": 4,
        "horizon": 3,
        "history_length": 12,
        "state_dim": 25,
        "asset_feature_dim": 3,
        "market_feature_dim": 5,
    }
    model = make_benchmark_model(name, dimensions, parameters)
    output = model(
        torch.randn(2, 12, 4),
        state=torch.randn(2, 25),
        asset_features=torch.randn(2, 12, 4, 3),
        market_history=torch.randn(2, 12, 5),
    )
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()


def test_dlinear_rejects_even_moving_average() -> None:
    dimensions = {
        "num_assets": 4,
        "horizon": 3,
        "history_length": 12,
        "state_dim": 25,
        "asset_feature_dim": 3,
        "market_feature_dim": 5,
    }
    with pytest.raises(ValueError, match="odd"):
        make_benchmark_model(
            "DLinear", dimensions, {"moving_average": 4, "individual": False}
        )
