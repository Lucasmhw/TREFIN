from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fconcord.utils import json_dump


def squared_error_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    """Return raw-scale squared-error metrics against the zero-return forecast."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("target and prediction must be nonempty arrays with equal shape")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("target and prediction must be finite")
    squared_error = (prediction - target) ** 2
    mse = float(np.mean(squared_error))
    zero_return_mse = float(np.mean(target**2))
    oos_r2_zero = (
        float(1.0 - mse / zero_return_mse) if zero_return_mse > 0.0 else float("nan")
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "zero_return_mse": zero_return_mse,
        "zero_return_rmse": float(np.sqrt(zero_return_mse)),
        "oos_r2_zero": oos_r2_zero,
        "forecast_cells": int(target.size),
        "sum_squared_error": float(np.sum(squared_error)),
        "zero_return_sum_squared_error": float(np.sum(target**2)),
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ("mse", "zero_return_mse", "oos_r2_zero")

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            result[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
        total_cells = sum(int(row["forecast_cells"]) for row in subset)
        total_sse = sum(float(row["sum_squared_error"]) for row in subset)
        total_zero_sse = sum(
            float(row["zero_return_sum_squared_error"]) for row in subset
        )
        result["pooled"] = {
            "forecast_cells": total_cells,
            "mse": total_sse / total_cells,
            "zero_return_mse": total_zero_sse / total_cells,
            "oos_r2_zero": 1.0 - total_sse / total_zero_sse,
        }
        return result

    by_fold = {
        fold: summarize([row for row in rows if row["fold"] == fold])
        for fold in sorted({str(row["fold"]) for row in rows})
    }
    return {"overall": summarize(rows), "by_fold": by_fold}


def build_squared_error_report(
    protocol_directory: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Derive MSE and OOS R2 from immutable final-test prediction archives."""
    protocol = Path(protocol_directory).resolve()
    summary_path = protocol / "protocol_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing completed protocol summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary["runs_completed"]) != int(summary["expected_runs"]):
        raise ValueError("Squared-error report requires a complete protocol")

    rows: list[dict[str, Any]] = []
    target_hashes: dict[str, set[str]] = {}
    maximum_rmse_identity_error = 0.0
    for source_row in summary["rows"]:
        fold = str(source_row["fold"])
        seed = int(source_row["seed"])
        run_path = Path(source_row["run_directory"])
        if not run_path.is_dir():
            run_path = protocol / "runs" / f"{fold}_seed_{seed}"
        prediction_path = run_path / "predictions" / "test_predictions.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing frozen test prediction archive: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            target = np.asarray(archive["y"])
            prediction = np.asarray(archive["mu"])
        metrics = squared_error_metrics(target, prediction)
        identity_error = abs(float(metrics["rmse"]) - float(source_row["rmse"]))
        maximum_rmse_identity_error = max(maximum_rmse_identity_error, identity_error)
        target_hash = _array_sha256(target)
        target_hashes.setdefault(fold, set()).add(target_hash)
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "run_directory": str(run_path.resolve()),
                "prediction_archive": str(prediction_path.resolve()),
                "target_sha256": target_hash,
                **metrics,
                "registered_rmse": float(source_row["rmse"]),
                "rmse_identity_abs_error": identity_error,
            }
        )

    report = {
        "schema": "fconcord-squared-error-report-v1",
        "protocol_directory": str(protocol),
        "derivation": "deterministic post-hoc computation from frozen final-test y and mu",
        "target_unit": "decimal one-period excess log return",
        "mse_unit": "squared decimal return",
        "percentage_conversion": (
            "multiply returns, RMSE, and MAE by 100 for percentage points; "
            "multiply MSE by 10000 only when expressing squared percentage points"
        ),
        "mape_included": False,
        "mape_exclusion_reason": "returns can be zero or arbitrarily close to zero",
        "oos_r2_definition": "1 - MSE_model / MSE_ZeroReturn",
        "rows": rows,
        "aggregate": _aggregate(rows),
        "audit": {
            "runs_expected": int(summary["expected_runs"]),
            "runs_derived": len(rows),
            "all_target_arrays_identical_within_fold": all(
                len(hashes) == 1 for hashes in target_hashes.values()
            ),
            "target_hashes_by_fold": {
                fold: sorted(hashes) for fold, hashes in sorted(target_hashes.items())
            },
            "maximum_registered_rmse_identity_abs_error": maximum_rmse_identity_error,
            "rmse_identity_tolerance": 1e-7,
            "rmse_identity_passed": maximum_rmse_identity_error <= 1e-7,
            "all_values_finite": bool(
                np.isfinite(
                    [
                        value
                        for row in rows
                        for key, value in row.items()
                        if key
                        in {
                            "mse",
                            "rmse",
                            "zero_return_mse",
                            "zero_return_rmse",
                            "oos_r2_zero",
                        }
                    ]
                ).all()
            ),
        },
    }
    report["audit"]["passed"] = bool(
        report["audit"]["runs_derived"] == report["audit"]["runs_expected"]
        and report["audit"]["all_target_arrays_identical_within_fold"]
        and report["audit"]["rmse_identity_passed"]
        and report["audit"]["all_values_finite"]
    )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else protocol / "protocol_squared_error_metrics.json"
    )
    json_dump(report, destination)
    csv_path = destination.with_suffix(".csv")
    csv_fields = [
        "fold",
        "seed",
        "mse",
        "rmse",
        "zero_return_mse",
        "zero_return_rmse",
        "oos_r2_zero",
        "forecast_cells",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination
