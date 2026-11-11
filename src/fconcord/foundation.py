from __future__ import annotations

import gc
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fconcord.benchmark import (
    _main_model_losses,
    _paired_tests,
    _reference_data_paths,
)
from fconcord.benchmark_data import build_benchmark_arrays
from fconcord.benchmark_training import residual_bootstrap_scenarios
from fconcord.config import load_config, save_config
from fconcord.data import load_panel_csv
from fconcord.experiment import load_prepared
from fconcord.metrics import evaluate_predictions
from fconcord.statistical_tests import (
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)
from fconcord.utils import environment_snapshot, json_dump, select_device, setup_logger

OFFICIAL_FOUNDATIONS = ("Chronos2", "TimesFM2p5")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _chronos_predict(
    pipeline: Any,
    histories: np.ndarray,
    horizon: int,
    origin_batch_size: int,
) -> np.ndarray:
    """Official Chronos-2 multivariate dataframe inference without cross-origin learning."""
    origins, length, assets = histories.shape
    target_columns = [f"asset_{asset:03d}" for asset in range(assets)]
    timestamp = pd.date_range("2000-01-01", periods=length, freq="D")
    output = np.empty((origins, horizon, assets), dtype=np.float32)
    for start in range(0, origins, origin_batch_size):
        stop = min(start + origin_batch_size, origins)
        identifiers = [f"origin_{origin}" for origin in range(start, stop)]
        frame = pd.DataFrame(
            histories[start:stop].reshape((stop - start) * length, assets),
            columns=target_columns,
        )
        frame.insert(0, "timestamp", np.tile(timestamp.to_numpy(), stop - start))
        frame.insert(0, "item_id", np.repeat(identifiers, length))
        predicted = pipeline.predict_df(
            frame,
            target=target_columns,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            cross_learning=False,
            validate_inputs=False,
            freq="D",
        )
        required = {"item_id", "target_name", "timestamp", "predictions"}
        if required - set(predicted.columns):
            raise ValueError("Chronos-2 output does not match its registered dataframe API")
        for origin in range(start, stop):
            origin_rows = predicted[predicted["item_id"] == f"origin_{origin}"]
            for asset, target_name in enumerate(target_columns):
                values = (
                    origin_rows[origin_rows["target_name"] == target_name]
                    .sort_values("timestamp")["predictions"]
                    .to_numpy(dtype=np.float32)
                )
                if len(values) != horizon:
                    raise ValueError("Chronos-2 returned an incomplete forecast path")
                output[origin, :, asset] = values
    return output


def _timesfm_predict(
    model: Any,
    histories: np.ndarray,
    horizon: int,
    origin_batch_size: int,
) -> np.ndarray:
    """Official TimesFM 2.5 univariate inference in deterministic origin-major order."""
    origins, _, assets = histories.shape
    output = np.empty((origins, horizon, assets), dtype=np.float32)
    for start in range(0, origins, origin_batch_size):
        stop = min(start + origin_batch_size, origins)
        inputs = [
            histories[origin, :, asset].astype(np.float32, copy=False)
            for origin in range(start, stop)
            for asset in range(assets)
        ]
        point, _ = model.forecast(horizon=horizon, inputs=inputs)
        point = np.asarray(point, dtype=np.float32)
        expected = ((stop - start) * assets, horizon)
        if point.shape != expected:
            raise ValueError(f"TimesFM returned {point.shape}; expected {expected}")
        output[start:stop] = point.reshape(stop - start, assets, horizon).transpose(0, 2, 1)
    return output


def _load_official_model(name: str, specification: dict[str, Any], device: torch.device) -> Any:
    if name == "Chronos2":
        try:
            from chronos import Chronos2Pipeline
        except ImportError as error:
            raise RuntimeError(
                "Install the official package with `pip install chronos-forecasting`"
            ) from error
        return Chronos2Pipeline.from_pretrained(
            str(specification["model_id"]),
            device_map=str(device),
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
    if name == "TimesFM2p5":
        try:
            import timesfm
        except ImportError as error:
            raise RuntimeError(
                "Install the official package with `pip install 'timesfm[torch]'`"
            ) from error
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            str(specification["model_id"])
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=int(specification["max_context"]),
                max_horizon=int(specification["max_horizon"]),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        return model
    raise ValueError(f"Unknown official foundation model: {name}")


def _predict_scaled(
    name: str,
    model: Any,
    histories: np.ndarray,
    horizon: int,
    specification: dict[str, Any],
) -> np.ndarray:
    batch_size = int(specification["origin_batch_size"])
    if name == "Chronos2":
        return _chronos_predict(model, histories, horizon, batch_size)
    if name == "TimesFM2p5":
        return _timesfm_predict(model, histories, horizon, batch_size)
    raise ValueError(f"Unknown official foundation model: {name}")


def run_foundation_benchmark(
    config_path: str | Path,
    reference_run: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run official zero-shot foundation models with train-residual joint calibration."""
    config_path = Path(config_path).resolve()
    reference = Path(reference_run).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Foundation output already exists: {output}")
    config = load_config(config_path)
    unknown = set(config["models"]) - set(OFFICIAL_FOUNDATIONS)
    if unknown:
        raise ValueError(f"Unknown foundation models: {sorted(unknown)}")
    output.mkdir(parents=True)
    for child in ("logs", "predictions", "tables", "audit"):
        (output / child).mkdir()
    logger = setup_logger(output / "logs" / "foundation.log")
    save_config(config, output / "config_resolved.yaml")
    json_dump(environment_snapshot(), output / "environment.json")
    started = time.perf_counter()

    reference_config = load_config(reference / "config_resolved.yaml")
    prepared = load_prepared(reference / "audit")
    panel_path, factor_path = _reference_data_paths(reference)
    panel = load_panel_csv(panel_path, factor_path, reference_config["data"])
    benchmark = build_benchmark_arrays(
        prepared, panel, int(config["foundation"]["history_length"])
    )
    json_dump(benchmark["metadata"], output / "audit" / "benchmark_metadata.json")
    arrays = benchmark["arrays"]
    splits = benchmark["metadata"]["split_indices"]
    train_positions = np.asarray(splits["train"], dtype=np.int64)
    test_positions = np.asarray(splits["test"], dtype=np.int64)
    horizon = int(benchmark["metadata"]["dimensions"]["horizon"])
    scale = arrays["return_scale"]
    device = select_device(str(config["foundation"].get("device", "auto")))
    reference_losses = _main_model_losses(reference)
    results: dict[str, Any] = {}
    for name, specification in config["models"].items():
        logger.info("official_foundation_started name=%s model_id=%s", name, specification["model_id"])
        model_start = time.perf_counter()
        model = _load_official_model(name, specification, device)
        train_scaled = _predict_scaled(
            name, model, arrays["return_history"][train_positions], horizon, specification
        )
        # The zero-shot model has no local selection step. This is its sole test access.
        test_scaled = _predict_scaled(
            name, model, arrays["return_history"][test_positions], horizon, specification
        )
        train_mean = train_scaled * scale[None, None]
        test_mean = test_scaled * scale[None, None]
        train_target = arrays["target"][train_positions]
        test_target = arrays["target"][test_positions]
        residual_pool = train_target - train_mean
        scenarios, residual_indices = residual_bootstrap_scenarios(
            test_mean,
            residual_pool,
            int(config["evaluation"]["test_samples"]),
            int(config["evaluation"]["evaluation_seed"]),
        )
        metrics = evaluate_predictions(
            test_target,
            test_mean,
            scenarios,
            [float(value) for value in config["evaluation"]["alpha_levels"]],
        )
        per_origin = {
            "squared_error": per_origin_squared_error(test_target, test_mean),
            "absolute_error": per_origin_absolute_error(test_target, test_mean),
            "energy_score": per_origin_energy_score(scenarios, test_target),
        }
        prediction_path = output / "predictions" / f"{name}.npz"
        np.savez_compressed(
            prediction_path,
            mean=test_mean,
            target=test_target,
            test_indices=test_positions,
            train_residual_pool=residual_pool,
            scenario_residual_indices=residual_indices,
            **per_origin,
        )
        package_name = "chronos-forecasting" if name == "Chronos2" else "timesfm"
        result = {
            "model_name": name,
            "official_model_id": specification["model_id"],
            "official_package": package_name,
            "official_package_version": _package_version(package_name),
            "zero_shot": True,
            "local_gradient_updates": 0,
            "test_access_count": 1,
            "test_used_for_selection": False,
            "test_metrics": metrics,
            "paired_tests_vs_fconcord": _paired_tests(
                prediction_path,
                reference_losses,
                horizon,
                config["statistical_tests"],
            ),
            "scenario_semantics": (
                "official zero-shot point forecast plus centered empirical joint [H,N] "
                "residual paths fitted only on the reference training split"
            ),
            "point_semantics": (
                "Chronos-2 official predictions column (the official pipeline maps median "
                "to mean)" if name == "Chronos2" else "TimesFM 2.5 official point forecast"
            ),
            "wall_seconds": float(time.perf_counter() - model_start),
        }
        results[name] = result
        json_dump(result, output / "tables" / f"{name}.json")
        logger.info(
            "official_foundation_complete name=%s rmse=%.8f es=%.8f seconds=%.2f",
            name,
            metrics["rmse"],
            metrics["energy_score"],
            result["wall_seconds"],
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "schema": "fconcord-official-foundation-benchmark-v1",
        "reference_run": str(reference),
        "config_path": str(config_path),
        "models": results,
        "test_policy": "zero-shot inference; each model accesses test once",
        "wall_seconds": float(time.perf_counter() - started),
    }
    json_dump(manifest, output / "foundation_manifest.json")
    return output


def run_foundation_protocol(
    config_path: str | Path,
    reference_protocol: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run deterministic official foundations once per fold, without pseudo-seed repeats."""
    config_path = Path(config_path).resolve()
    reference_root = Path(reference_protocol).resolve()
    output = Path(output_directory).resolve()
    with (reference_root / "protocol_manifest.json").open("r", encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    folds = [str(fold["id"]) for fold in reference_manifest["folds"]]
    reference_seed = int(reference_manifest["seeds"][0])
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    manifest = {
        "schema": "fconcord-official-foundation-protocol-v1",
        "config_path": str(config_path),
        "reference_protocol": str(reference_root),
        "folds": folds,
        "reference_seed_for_identical_fold_states": reference_seed,
        "runs_per_model": len(folds),
        "seed_repeats": 0,
        "seed_repeat_reason": (
            "zero-shot official models have no local initialization or gradient updates; "
            "the prepared states and raw histories are identical across seeds within a fold"
        ),
    }
    json_dump(manifest, output / "protocol_manifest.json")
    rows: list[dict[str, Any]] = []
    for fold in folds:
        reference_run = reference_root / "runs" / f"{fold}_seed_{reference_seed}"
        run_output = output / "runs" / fold
        result_path = run_output / "foundation_manifest.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            if run_output.exists():
                raise RuntimeError(
                    f"Incomplete foundation run exists; audit before retry: {run_output}"
                )
            run_foundation_benchmark(config_path, reference_run, run_output)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        for model_name, model_result in result["models"].items():
            rows.append(
                {
                    "fold": fold,
                    "model_name": model_name,
                    "run_directory": str(run_output),
                    **{
                        metric: float(model_result["test_metrics"][metric])
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
        json_dump(rows, output / "protocol_progress.json")
    aggregate: dict[str, Any] = {}
    for model_name in sorted({row["model_name"] for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name]
        aggregate[model_name] = {}
        for metric in (
            "rmse",
            "mae",
            "directional_accuracy",
            "energy_score",
            "msic",
            "msir",
        ):
            values = np.asarray([row[metric] for row in model_rows])
            aggregate[model_name][metric] = {
                "fold_mean": float(values.mean()),
                "fold_sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    expected_rows = len(folds) * len(load_config(config_path)["models"])
    summary = {
        **manifest,
        "runs_completed": len(folds),
        "all_runs_complete": len(rows) == expected_rows,
        "rows": rows,
        "aggregate": aggregate,
    }
    json_dump(summary, output / "protocol_summary.json")
    return output
