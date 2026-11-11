from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fconcord.benchmark_classical import (
    evaluate_selected_classical_candidate,
    fit_classical_candidate,
)
from fconcord.benchmark_data import build_benchmark_arrays
from fconcord.benchmark_training import (
    evaluate_selected_neural_candidate,
    train_neural_candidate,
)
from fconcord.config import load_config, save_config
from fconcord.data import load_panel_csv
from fconcord.experiment import load_prepared
from fconcord.statistical_tests import (
    circular_block_bootstrap_mean_difference,
    diebold_mariano_hac,
    holm_adjust,
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)
from fconcord.utils import environment_snapshot, json_dump, select_device, setup_logger

_CLASSICAL_MODELS = {"ShrunkVAR", "HistGradientBoosting"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_benchmark(benchmark: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "benchmark_arrays.npz", **benchmark["arrays"])
    json_dump(benchmark["metadata"], directory / "benchmark_metadata.json")


def _reference_data_paths(reference_run: Path) -> tuple[Path, Path]:
    provenance = _load_json(reference_run / "audit" / "data_provenance.json")
    files = provenance.get("files", {})
    try:
        panel = Path(files["panel"]["path"])
        factors = Path(files["factors"]["path"])
    except KeyError as error:
        raise ValueError("Reference run lacks real-data panel/factor provenance") from error
    if not panel.is_file() or not factors.is_file():
        raise FileNotFoundError(
            "The immutable panel/factor paths recorded by the reference run are unavailable"
        )
    for name, path in (("panel", panel), ("factors", factors)):
        recorded = str(files[name].get("sha256", ""))
        observed = _sha256(path)
        if recorded and recorded != observed:
            raise ValueError(f"Reference {name} SHA-256 differs from its recorded provenance")
    return panel, factors


def _validate_benchmark_config(config: dict[str, Any]) -> None:
    section = config.get("benchmark", {})
    budget = int(section.get("candidate_budget", 0))
    if budget < 1:
        raise ValueError("benchmark.candidate_budget must be positive")
    models = config.get("models", {})
    if not models:
        raise ValueError("At least one benchmark model must be registered")
    for name, specification in models.items():
        candidates = specification.get("candidates", [])
        if len(candidates) != budget:
            raise ValueError(
                f"{name} has {len(candidates)} candidates, but the equal budget is {budget}"
            )
        signatures = [json.dumps(row, sort_keys=True) for row in candidates]
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"{name} repeats an identical hyperparameter candidate")
        kind = str(specification.get("kind", ""))
        expected = "classical" if name in _CLASSICAL_MODELS else "neural"
        if kind != expected:
            raise ValueError(f"{name} must be registered as kind={expected!r}")


def _main_model_losses(reference_run: Path) -> dict[str, np.ndarray]:
    path = reference_run / "predictions" / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as handle:
        target = handle["y"]
        mean = handle["mu"]
        samples = handle["samples"]
        indices = handle["indices"]
    return {
        "target": target,
        "indices": indices,
        "squared_error": per_origin_squared_error(target, mean),
        "absolute_error": per_origin_absolute_error(target, mean),
        "energy_score": per_origin_energy_score(samples, target),
    }


def _paired_tests(
    candidate_path: Path,
    reference: dict[str, np.ndarray],
    horizon: int,
    statistical: dict[str, Any],
) -> dict[str, Any]:
    with np.load(candidate_path, allow_pickle=False) as handle:
        target = handle["target"]
        indices = handle["test_indices"]
        candidate_losses = {
            name: handle[name]
            for name in ("squared_error", "absolute_error", "energy_score")
        }
    np.testing.assert_array_equal(indices, reference["indices"])
    np.testing.assert_allclose(target, reference["target"], rtol=0.0, atol=1e-10)
    report: dict[str, Any] = {}
    for offset, (loss_name, candidate_loss) in enumerate(candidate_losses.items()):
        report[loss_name] = {
            "diebold_mariano_hac": diebold_mariano_hac(
                candidate_loss, reference[loss_name], horizon=horizon
            ),
            "circular_block_bootstrap": circular_block_bootstrap_mean_difference(
                candidate_loss,
                reference[loss_name],
                block_length=int(statistical["block_length"]),
                repetitions=int(statistical["bootstrap_repetitions"]),
                confidence=float(statistical["confidence"]),
                seed=int(statistical["seed"]) + offset,
            ),
        }
    return report


def _apply_holm_across_models(results: dict[str, Any]) -> None:
    for loss_name in ("squared_error", "absolute_error", "energy_score"):
        raw_dm = {
            name: float(result["paired_tests_vs_fconcord"][loss_name]
                        ["diebold_mariano_hac"]["two_sided_p_value"])
            for name, result in results.items()
        }
        raw_bootstrap = {
            name: float(result["paired_tests_vs_fconcord"][loss_name]
                        ["circular_block_bootstrap"]["two_sided_bootstrap_p_value"])
            for name, result in results.items()
        }
        adjusted_dm = holm_adjust(raw_dm)
        adjusted_bootstrap = holm_adjust(raw_bootstrap)
        for name, result in results.items():
            loss = result["paired_tests_vs_fconcord"][loss_name]
            loss["diebold_mariano_hac"]["holm_adjusted_p_value_across_models"] = (
                adjusted_dm[name]
            )
            loss["circular_block_bootstrap"][
                "holm_adjusted_p_value_across_models"
            ] = adjusted_bootstrap[name]


def run_benchmark(
    config_path: str | Path,
    reference_run: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run one equal-budget benchmark matrix against one completed F-CONCORD run."""
    config_path = Path(config_path).resolve()
    reference = Path(reference_run).resolve()
    output = Path(output_directory).resolve()
    config = load_config(config_path)
    _validate_benchmark_config(config)
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if not (reference / "run_summary.json").is_file():
        raise FileNotFoundError("Reference F-CONCORD run is incomplete")
    output.mkdir(parents=True)
    for child in ("audit", "candidates", "selected", "logs", "tables"):
        (output / child).mkdir()
    logger = setup_logger(output / "logs" / "benchmark.log")
    started = time.perf_counter()
    save_config(config, output / "config_resolved.yaml")
    json_dump(environment_snapshot(), output / "environment.json")

    reference_config = load_config(reference / "config_resolved.yaml")
    prepared = load_prepared(reference / "audit")
    panel_path, factor_path = _reference_data_paths(reference)
    panel = load_panel_csv(panel_path, factor_path, reference_config["data"])
    benchmark = build_benchmark_arrays(
        prepared, panel, int(config["benchmark"]["history_length"])
    )
    _save_benchmark(benchmark, output / "audit" / "benchmark_data")
    device = select_device(str(config["benchmark"].get("device", "auto")))
    seed = int(config["benchmark"]["seed"])
    evaluation = {
        **config["training"],
        **config["evaluation"],
    }
    logger.info(
        "benchmark_started reference=%s device=%s models=%d equal_candidate_budget=%d",
        reference,
        device,
        len(config["models"]),
        int(config["benchmark"]["candidate_budget"]),
    )

    main_losses = _main_model_losses(reference)
    all_results: dict[str, Any] = {}
    for model_name, specification in config["models"].items():
        logger.info("model_started name=%s kind=%s", model_name, specification["kind"])
        summaries: list[dict[str, Any]] = []
        for candidate_index, parameters in enumerate(specification["candidates"], start=1):
            candidate_directory = (
                output / "candidates" / model_name / f"candidate_{candidate_index:02d}"
            )
            # Hold the run seed fixed across candidates so hyperparameters are not
            # confounded with a different initialization or minibatch permutation.
            candidate_seed = seed
            if specification["kind"] == "classical":
                summary = fit_classical_candidate(
                    model_name,
                    dict(parameters),
                    benchmark,
                    evaluation,
                    candidate_directory,
                    candidate_seed,
                )
            else:
                summary = train_neural_candidate(
                    model_name,
                    dict(parameters),
                    benchmark,
                    evaluation,
                    candidate_directory,
                    device,
                    candidate_seed,
                )
            summary["candidate_index"] = candidate_index
            summaries.append(summary)
            json_dump(summary, candidate_directory / "candidate_summary.json")
            logger.info(
                "candidate_complete model=%s candidate=%d val_es=%.8f",
                model_name,
                candidate_index,
                summary["best_val_energy_score"],
            )
        selected = min(
            summaries,
            key=lambda row: (float(row["best_val_energy_score"]), int(row["candidate_index"])),
        )
        selected_directory = output / "selected" / model_name
        if specification["kind"] == "classical":
            result = evaluate_selected_classical_candidate(
                selected, benchmark, evaluation, selected_directory
            )
        else:
            result = evaluate_selected_neural_candidate(
                selected, benchmark, evaluation, selected_directory, device
            )
        result["candidate_budget"] = len(summaries)
        result["selected_candidate_index"] = int(selected["candidate_index"])
        result["candidates"] = summaries
        result["paired_tests_vs_fconcord"] = _paired_tests(
            selected_directory / "reproducible_predictions.npz",
            main_losses,
            int(benchmark["metadata"]["dimensions"]["horizon"]),
            config["statistical_tests"],
        )
        all_results[model_name] = result
        json_dump(result, selected_directory / "result.json")
        logger.info(
            "model_complete name=%s selected_candidate=%d test_rmse=%.8f test_es=%.8f",
            model_name,
            result["selected_candidate_index"],
            result["test_metrics"]["rmse"],
            result["test_metrics"]["energy_score"],
        )

    _apply_holm_across_models(all_results)
    for model_name, result in all_results.items():
        json_dump(result, output / "selected" / model_name / "result.json")
    manifest = {
        "schema": "fconcord-equal-budget-benchmark-v1",
        "reference_run": str(reference),
        "reference_run_summary_sha256": _sha256(reference / "run_summary.json"),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "candidate_budget_per_model": int(config["benchmark"]["candidate_budget"]),
        "selection_data": "validation only",
        "selection_metric": "validation energy score",
        "test_policy": "fresh checkpoint reload followed by exactly one selected-model access",
        "shared_split_indices": benchmark["metadata"]["split_indices"],
        "shared_target_and_test_indices_verified": True,
        "models_completed": len(all_results),
        "results": all_results,
        "wall_seconds": float(time.perf_counter() - started),
    }
    json_dump(manifest, output / "benchmark_manifest.json")
    logger.info("benchmark_complete seconds=%.2f", manifest["wall_seconds"])
    return output
