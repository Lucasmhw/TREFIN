from fconcord.benchmark_protocol import _aggregate


def test_benchmark_protocol_aggregate_keeps_model_boundaries() -> None:
    rows = []
    for model, offset in (("a", 0.0), ("b", 1.0)):
        for run in range(2):
            rows.append(
                {
                    "model_name": model,
                    **{
                        metric: offset + run
                        for metric in (
                            "rmse",
                            "mae",
                            "directional_accuracy",
                            "energy_score",
                            "msic",
                            "msir",
                        )
                    },
                }
            )
    result = _aggregate(rows)
    assert result["a"]["rmse"]["mean"] == 0.5
    assert result["b"]["rmse"]["mean"] == 1.5
