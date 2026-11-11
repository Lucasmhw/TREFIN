from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from fconcord.benchmark_training import residual_bootstrap_scenarios
from fconcord.experiment import load_prepared
from fconcord.statistical_tests import (
    block_bootstrap_spa,
    circular_block_bootstrap_mean_difference,
    diebold_mariano_hac,
    holm_adjust,
    model_confidence_set,
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)
from fconcord.utils import json_dump

_LOSSES = ("squared_error", "absolute_error", "energy_score")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_losses(reference: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[int]]:
    manifest = _load_json(reference / "protocol_manifest.json")
    folds = [str(fold["id"]) for fold in manifest["folds"]]
    seeds = [int(seed) for seed in manifest["seeds"]]
    output: dict[str, dict[str, np.ndarray]] = {}
    for fold in folds:
        by_seed: dict[str, list[np.ndarray]] = {loss: [] for loss in _LOSSES}
        reference_indices: np.ndarray | None = None
        for seed in seeds:
            path = reference / "runs" / f"{fold}_seed_{seed}" / "predictions" / "test_predictions.npz"
            with np.load(path, allow_pickle=False) as handle:
                if reference_indices is None:
                    reference_indices = handle["indices"]
                else:
                    np.testing.assert_array_equal(reference_indices, handle["indices"])
                by_seed["squared_error"].append(
                    per_origin_squared_error(handle["y"], handle["mu"])
                )
                by_seed["absolute_error"].append(
                    per_origin_absolute_error(handle["y"], handle["mu"])
                )
                by_seed["energy_score"].append(
                    per_origin_energy_score(handle["samples"], handle["y"])
                )
        output[fold] = {
            loss: np.mean(np.stack(values), axis=0) for loss, values in by_seed.items()
        }
        output[fold]["indices"] = np.asarray(reference_indices)
    return output, seeds


def _benchmark_losses(
    protocol: Path, folds: list[str], seeds: list[int]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for fold in folds:
        by_model: dict[str, dict[str, list[np.ndarray]]] = {}
        indices_by_model: dict[str, np.ndarray] = {}
        for seed in seeds:
            run = protocol / "runs" / f"{fold}_seed_{seed}"
            manifest = _load_json(run / "benchmark_manifest.json")
            for model_name in manifest["results"]:
                with np.load(
                    run / "selected" / model_name / "reproducible_predictions.npz",
                    allow_pickle=False,
                ) as handle:
                    losses = by_model.setdefault(
                        model_name, {loss: [] for loss in _LOSSES}
                    )
                    for loss in _LOSSES:
                        losses[loss].append(handle[loss])
                    if model_name in indices_by_model:
                        np.testing.assert_array_equal(
                            indices_by_model[model_name], handle["test_indices"]
                        )
                    else:
                        indices_by_model[model_name] = handle["test_indices"]
        output[fold] = {
            model_name: {
                **{
                    loss: np.mean(np.stack(values), axis=0)
                    for loss, values in losses.items()
                },
                "indices": indices_by_model[model_name],
            }
            for model_name, losses in by_model.items()
        }
    return output


def _ablation_losses(
    protocol: Path, folds: list[str], seeds: list[int]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    manifest = _load_json(protocol / "protocol_manifest.json")
    variants = [str(value) for value in manifest["variants"]]
    for fold in folds:
        fold_output: dict[str, dict[str, np.ndarray]] = {}
        for variant in variants:
            by_seed: dict[str, list[np.ndarray]] = {loss: [] for loss in _LOSSES}
            indices: np.ndarray | None = None
            for seed in seeds:
                path = (
                    protocol
                    / "runs"
                    / f"{fold}_seed_{seed}_{variant}"
                    / "predictions"
                    / "reproducible_predictions.npz"
                )
                with np.load(path, allow_pickle=False) as handle:
                    for loss in _LOSSES:
                        by_seed[loss].append(handle[loss])
                    if indices is None:
                        indices = handle["test_indices"]
                    else:
                        np.testing.assert_array_equal(indices, handle["test_indices"])
            fold_output[variant] = {
                **{
                    loss: np.mean(np.stack(values), axis=0)
                    for loss, values in by_seed.items()
                },
                "indices": np.asarray(indices),
            }
        output[fold] = fold_output
    return output


def _foundation_losses(
    protocol: Path, folds: list[str]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for fold in folds:
        run = protocol / "runs" / fold
        manifest = _load_json(run / "foundation_manifest.json")
        output[fold] = {}
        for model_name in manifest["models"]:
            with np.load(
                run / "predictions" / f"{model_name}.npz", allow_pickle=False
            ) as handle:
                output[fold][model_name] = {
                    **{loss: handle[loss] for loss in _LOSSES},
                    "indices": handle["test_indices"],
                }
    return output


def _lightweight_control_losses(
    reference: Path, folds: list[str], first_seed: int
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Reconstruct preregistered simple controls at the origin level."""
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for fold in folds:
        prepared = load_prepared(reference / "runs" / f"{fold}_seed_{first_seed}" / "audit")
        arrays = prepared["arrays"]
        splits = prepared["metadata"]["split_indices"]
        train = np.asarray(splits["train"], dtype=np.int64)
        test = np.asarray(splits["test"], dtype=np.int64)
        horizon = int(prepared["metadata"]["horizon"])
        train_target = arrays["y"][train]
        test_target = arrays["y"][test]
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "ZeroReturn": (
                np.zeros_like(train_target),
                np.zeros_like(test_target),
            ),
            "LastReturn": (
                np.repeat(arrays["last_return"][train, None], horizon, axis=1),
                np.repeat(arrays["last_return"][test, None], horizon, axis=1),
            ),
            "HistoricalMean": (
                np.repeat(arrays["historical_mean"][train, None], horizon, axis=1),
                np.repeat(arrays["historical_mean"][test, None], horizon, axis=1),
            ),
        }
        train_features = np.concatenate(
            [arrays["c0"][train].reshape(len(train), -1), arrays["s0"][train]], axis=1
        )
        test_features = np.concatenate(
            [arrays["c0"][test].reshape(len(test), -1), arrays["s0"][test]], axis=1
        )
        ridge = Ridge(alpha=1.0, solver="lsqr")
        ridge.fit(train_features, train_target.reshape(len(train), -1))
        predictions["ConceptRidge"] = (
            ridge.predict(train_features).reshape(train_target.shape),
            ridge.predict(test_features).reshape(test_target.shape),
        )
        fold_output: dict[str, dict[str, np.ndarray]] = {}
        for model_name, (train_mean, test_mean) in predictions.items():
            scenarios, _ = residual_bootstrap_scenarios(
                test_mean,
                train_target - train_mean,
                sample_count=256,
                seed=271829,
            )
            fold_output[model_name] = {
                "squared_error": per_origin_squared_error(test_target, test_mean),
                "absolute_error": per_origin_absolute_error(test_target, test_mean),
                "energy_score": per_origin_energy_score(scenarios, test_target),
                "indices": test,
            }
        output[fold] = fold_output
    return output


def _merge_benchmark_sources(
    destination: dict[str, dict[str, dict[str, np.ndarray]]],
    source: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    for fold, models in source.items():
        overlap = set(destination.setdefault(fold, {})) & set(models)
        if overlap:
            raise ValueError(f"Duplicate comparison model names: {sorted(overlap)}")
        destination[fold].update(models)


def _pool_verified(
    model_by_fold: dict[str, dict[str, np.ndarray]],
    reference_by_fold: dict[str, dict[str, np.ndarray]],
    loss: str,
) -> np.ndarray:
    values = []
    for fold in reference_by_fold:
        np.testing.assert_array_equal(
            model_by_fold[fold]["indices"], reference_by_fold[fold]["indices"]
        )
        values.append(model_by_fold[fold][loss])
    return np.concatenate(values)


def build_comparison_report(
    reference_protocol: str | Path,
    native_protocol: str | Path,
    output_directory: str | Path,
    timemixer_protocol: str | Path | None = None,
    foundation_protocol: str | Path | None = None,
    ablation_protocol: str | Path | None = None,
    block_length: int = 5,
    repetitions: int = 5000,
    seed: int = 424242,
) -> Path:
    """Seed-average repeated origins, then compare over non-overlapping test folds."""
    reference = Path(reference_protocol).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference_by_fold, seeds = _reference_losses(reference)
    folds = list(reference_by_fold)
    baselines = _benchmark_losses(Path(native_protocol).resolve(), folds, seeds)
    _merge_benchmark_sources(
        baselines, _lightweight_control_losses(reference, folds, seeds[0])
    )
    if timemixer_protocol is not None:
        _merge_benchmark_sources(
            baselines,
            _benchmark_losses(Path(timemixer_protocol).resolve(), folds, seeds),
        )
    if foundation_protocol is not None:
        _merge_benchmark_sources(
            baselines,
            _foundation_losses(Path(foundation_protocol).resolve(), folds),
        )
    ablations = (
        _ablation_losses(Path(ablation_protocol).resolve(), folds, seeds)
        if ablation_protocol is not None
        else {}
    )
    reference_pooled = {
        loss: np.concatenate([reference_by_fold[fold][loss] for fold in folds])
        for loss in _LOSSES
    }
    baseline_by_model = {
        model_name: {fold: baselines[fold][model_name] for fold in folds}
        for model_name in sorted(baselines[folds[0]])
    }
    ablation_by_model = {
        model_name: {fold: ablations[fold][model_name] for fold in folds}
        for model_name in sorted(ablations.get(folds[0], {}))
    }
    loss_reports: dict[str, Any] = {}
    for loss_offset, loss in enumerate(_LOSSES):
        pooled_baselines = {
            model_name: _pool_verified(model, reference_by_fold, loss)
            for model_name, model in baseline_by_model.items()
        }
        paired: dict[str, Any] = {}
        for model_name, values in pooled_baselines.items():
            paired[model_name] = {
                "diebold_mariano_hac": diebold_mariano_hac(
                    values, reference_pooled[loss], horizon=block_length
                ),
                "circular_block_bootstrap": circular_block_bootstrap_mean_difference(
                    values,
                    reference_pooled[loss],
                    block_length=block_length,
                    repetitions=repetitions,
                    seed=seed + loss_offset,
                ),
            }
        adjusted = holm_adjust(
            {
                model_name: float(report["diebold_mariano_hac"]["two_sided_p_value"])
                for model_name, report in paired.items()
            }
        )
        for model_name in paired:
            paired[model_name]["diebold_mariano_hac"][
                "holm_adjusted_p_value"
            ] = adjusted[model_name]
        mcs_losses = {"FCONCORD": reference_pooled[loss], **pooled_baselines}
        loss_reports[loss] = {
            "seed_aggregation": (
                "per-origin loss averaged across five seeds inside each fold before "
                "concatenating the three non-overlapping test folds"
            ),
            "origins_after_seed_aggregation": len(reference_pooled[loss]),
            "paired_baselines_vs_fconcord": paired,
            "spa": block_bootstrap_spa(
                pooled_baselines,
                reference_pooled[loss],
                block_length=block_length,
                repetitions=repetitions,
                seed=seed + 10 + loss_offset,
            ),
            "model_confidence_set": model_confidence_set(
                mcs_losses,
                block_length=block_length,
                repetitions=repetitions,
                seed=seed + 20 + loss_offset,
            ),
        }
        if ablation_by_model:
            loss_reports[loss]["paired_retrained_ablations_vs_full"] = {
                model_name: {
                    "diebold_mariano_hac": diebold_mariano_hac(
                        _pool_verified(model, reference_by_fold, loss),
                        reference_pooled[loss],
                        horizon=block_length,
                    ),
                    "circular_block_bootstrap": circular_block_bootstrap_mean_difference(
                        _pool_verified(model, reference_by_fold, loss),
                        reference_pooled[loss],
                        block_length=block_length,
                        repetitions=repetitions,
                        seed=seed + 30 + loss_offset,
                    ),
                }
                for model_name, model in ablation_by_model.items()
            }
    report = {
        "schema": "fconcord-seed-averaged-origin-level-comparison-v1",
        "reference_protocol": str(reference),
        "native_protocol": str(Path(native_protocol).resolve()),
        "timemixer_protocol": None
        if timemixer_protocol is None
        else str(Path(timemixer_protocol).resolve()),
        "foundation_protocol": None
        if foundation_protocol is None
        else str(Path(foundation_protocol).resolve()),
        "ablation_protocol": None
        if ablation_protocol is None
        else str(Path(ablation_protocol).resolve()),
        "folds": folds,
        "seeds": seeds,
        "baseline_models": sorted(baseline_by_model),
        "retrained_ablations": sorted(ablation_by_model),
        "block_length": block_length,
        "bootstrap_repetitions": repetitions,
        "loss_reports": loss_reports,
    }
    json_dump(report, output / "comparison_report.json")
    return output
