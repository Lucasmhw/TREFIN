import pytest

from fconcord.benchmark import _apply_holm_across_models, _validate_benchmark_config


def _test_result(p_value: float):
    return {
        "paired_tests_vs_fconcord": {
            loss: {
                "diebold_mariano_hac": {"two_sided_p_value": p_value},
                "circular_block_bootstrap": {
                    "two_sided_bootstrap_p_value": p_value
                },
            }
            for loss in ("squared_error", "absolute_error", "energy_score")
        }
    }


def test_equal_candidate_budget_is_enforced() -> None:
    valid = {
        "benchmark": {"candidate_budget": 2},
        "models": {
            "ShrunkVAR": {
                "kind": "classical",
                "candidates": [{"lags": 2}, {"lags": 3}],
            },
            "GRU": {
                "kind": "neural",
                "candidates": [{"hidden_dim": 2}, {"hidden_dim": 3}],
            },
        },
    }
    _validate_benchmark_config(valid)
    valid["models"]["GRU"]["candidates"].pop()
    with pytest.raises(ValueError, match="equal budget"):
        _validate_benchmark_config(valid)


def test_holm_adjustment_is_written_to_every_model_and_loss() -> None:
    results = {"first": _test_result(0.01), "second": _test_result(0.04)}
    _apply_holm_across_models(results)
    for result in results.values():
        for loss in result["paired_tests_vs_fconcord"].values():
            assert "holm_adjusted_p_value_across_models" in loss["diebold_mariano_hac"]
            assert (
                "holm_adjusted_p_value_across_models"
                in loss["circular_block_bootstrap"]
            )
