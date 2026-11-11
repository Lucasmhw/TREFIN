from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from fconcord.statistical_tests import (
    per_origin_absolute_error,
    per_origin_squared_error,
)
from fconcord.utils import json_dump


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_benchmark_protocol(
    protocol_directory: str | Path, reference_protocol: str | Path
) -> dict[str, Any]:
    """Independently reopen a completed equal-budget benchmark protocol."""
    protocol = Path(protocol_directory).resolve()
    reference = Path(reference_protocol).resolve()
    manifest = _load_json(protocol / "protocol_manifest.json")
    folds = [str(value) for value in manifest["folds"]]
    seeds = [int(value) for value in manifest["seeds"]]
    expected_runs = len(folds) * len(seeds)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    checkpoint_count = 0
    prediction_count = 0
    for fold in folds:
        for seed in seeds:
            run_name = f"{fold}_seed_{seed}"
            run = protocol / "runs" / run_name
            reference_run = reference / "runs" / run_name
            try:
                completed = _load_json(run / "benchmark_manifest.json")
                budget = int(completed["candidate_budget_per_model"])
                with np.load(
                    reference_run / "predictions" / "test_predictions.npz",
                    allow_pickle=False,
                ) as handle:
                    reference_target = handle["y"]
                    reference_indices = handle["indices"]
                for model_name, result in completed["results"].items():
                    if len(result["candidates"]) != budget:
                        raise AssertionError(f"{model_name}: unequal candidate budget")
                    if int(result["test_access_count"]) != 1:
                        raise AssertionError(f"{model_name}: test access count is not one")
                    if not bool(result["checkpoint_reload_before_test"]):
                        raise AssertionError(f"{model_name}: no pre-test checkpoint reload")
                    for candidate in result["candidates"]:
                        checkpoint = Path(candidate["checkpoint"])
                        if not checkpoint.is_file():
                            raise AssertionError(f"missing checkpoint: {checkpoint}")
                        if checkpoint.suffix == ".pt":
                            payload = torch.load(
                                checkpoint, map_location="cpu", weights_only=False
                            )
                            if "model_state" not in payload:
                                raise AssertionError("neural checkpoint has no model_state")
                            history_path = (
                                checkpoint.parent.parent / "logs" / "training_history.csv"
                            )
                            history = pd.read_csv(history_path)
                            if any("test" in str(column).lower() for column in history.columns):
                                raise AssertionError("training history contains a test column")
                        else:
                            joblib.load(checkpoint)
                        if float(candidate["checkpoint_reload_max_abs_difference"]) >= 1e-7:
                            raise AssertionError("candidate checkpoint reload mismatch")
                        checkpoint_count += 1
                    prediction_path = (
                        run / "selected" / model_name / "reproducible_predictions.npz"
                    )
                    with np.load(prediction_path, allow_pickle=False) as handle:
                        mean = handle["mean"]
                        target = handle["target"]
                        indices = handle["test_indices"]
                        squared = handle["squared_error"]
                        absolute = handle["absolute_error"]
                        residual_pool = handle["train_residual_pool"]
                        residual_indices = handle["scenario_residual_indices"]
                    np.testing.assert_array_equal(indices, reference_indices)
                    np.testing.assert_allclose(target, reference_target, rtol=0.0, atol=1e-10)
                    np.testing.assert_allclose(
                        squared,
                        per_origin_squared_error(target, mean),
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        absolute,
                        per_origin_absolute_error(target, mean),
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    if residual_indices.shape[1] != len(target):
                        raise AssertionError("scenario draw indices omit test origins")
                    if residual_indices.min() < 0 or residual_indices.max() >= len(residual_pool):
                        raise AssertionError("scenario residual index is out of range")
                    prediction_count += 1
                records.append(
                    {
                        "run_name": run_name,
                        "models": len(completed["results"]),
                        "candidate_budget": budget,
                        "passed": True,
                    }
                )
            except Exception as error:  # noqa: BLE001 - audit must retain every failure
                failures.append(f"{run_name}: {type(error).__name__}: {error}")
                records.append({"run_name": run_name, "passed": False})
    report = {
        "schema": "fconcord-benchmark-protocol-audit-v1",
        "protocol_directory": str(protocol),
        "reference_protocol": str(reference),
        "expected_runs": expected_runs,
        "audited_runs": len(records),
        "passed_runs": sum(bool(record["passed"]) for record in records),
        "checkpoints_reopened": checkpoint_count,
        "selected_prediction_files_recomputed": prediction_count,
        "failures": failures,
        "records": records,
        "passed": len(records) == expected_runs and not failures,
    }
    json_dump(report, protocol / "benchmark_protocol_audit.json")
    return report
