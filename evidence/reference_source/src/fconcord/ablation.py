from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fconcord.benchmark import _main_model_losses, _paired_tests
from fconcord.config import load_config, save_config
from fconcord.experiment import _model_factory, load_prepared
from fconcord.metrics import evaluate_predictions
from fconcord.statistical_tests import (
    per_origin_absolute_error,
    per_origin_energy_score,
    per_origin_squared_error,
)
from fconcord.trainer import checkpoint_reload_check, collect_predictions, train_model
from fconcord.utils import (
    count_parameters,
    environment_snapshot,
    json_dump,
    seed_everything,
    select_device,
    setup_logger,
)

RETRAINED_ABLATIONS = (
    "NoGraph",
    "NoMarketState",
    "NoRiskConcepts",
    "NoForwardConceptSupervision",
    "GaussianSymmetricDecoder",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_retrained_ablation(
    reference_run: str | Path,
    variant: str,
    output_directory: str | Path,
    statistical_config: dict[str, Any] | None = None,
) -> Path:
    """Retrain one structural ablation on the exact reference split and seed."""
    if variant not in RETRAINED_ABLATIONS:
        raise ValueError(f"Unknown retrained ablation: {variant}")
    reference = Path(reference_run).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Ablation output already exists: {output}")
    if not (reference / "run_summary.json").is_file():
        raise FileNotFoundError("Reference F-CONCORD run is incomplete")
    output.mkdir(parents=True)
    for child in ("checkpoints", "logs", "predictions", "tables", "audit"):
        (output / child).mkdir()

    config = copy.deepcopy(load_config(reference / "config_resolved.yaml"))
    config["experiment"]["name"] = f"{config['experiment']['name']}_{variant}"
    config["model"]["ablation_variant"] = variant
    if variant == "NoGraph":
        # A model with graph states identically zero cannot optimize a graph-target
        # loss; retaining it would add only a large parameter-independent constant.
        config["training"]["lambda_graph"] = 0.0
    if variant == "NoForwardConceptSupervision":
        config["training"]["lambda_concept"] = 0.0
    save_config(config, output / "config_resolved.yaml")
    json_dump(environment_snapshot(), output / "environment.json")
    logger = setup_logger(output / "logs" / "run.log")
    started = time.perf_counter()
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    device = select_device(str(config["experiment"]["device"]))
    prepared = load_prepared(reference / "audit")
    factory = lambda: _model_factory(prepared, config)
    model = factory()
    logger.info(
        "retrained_ablation_started variant=%s reference=%s seed=%d parameters=%d",
        variant,
        reference,
        seed,
        count_parameters(model),
    )
    model, _, training_summary = train_model(
        model, prepared, config["training"], output, device, logger
    )
    reloaded, reload_check = checkpoint_reload_check(
        factory,
        output / "checkpoints" / "best.pt",
        model,
        prepared,
        config["training"],
        device,
    )
    if not bool(reload_check["passed"]):
        raise RuntimeError(f"Ablation checkpoint reload failed: {reload_check}")
    json_dump(reload_check, output / "audit" / "checkpoint_reload_check.json")

    # No test split has been touched before this fresh-checkpoint evaluation.
    seed_everything(seed + 1, bool(config["experiment"]["deterministic"]))
    prediction = collect_predictions(
        reloaded, prepared, "test", config["training"], device
    )
    np.savez_compressed(output / "predictions" / "test_predictions.npz", **prediction)
    metrics = evaluate_predictions(
        prediction["y"],
        prediction["mu"],
        prediction["samples"],
        [float(value) for value in config["evaluation"]["alpha_levels"]],
    )
    per_origin = {
        "squared_error": per_origin_squared_error(prediction["y"], prediction["mu"]),
        "absolute_error": per_origin_absolute_error(prediction["y"], prediction["mu"]),
        "energy_score": per_origin_energy_score(prediction["samples"], prediction["y"]),
    }
    reproducible = output / "predictions" / "reproducible_predictions.npz"
    np.savez_compressed(
        reproducible,
        mean=prediction["mu"],
        target=prediction["y"],
        samples=prediction["samples"],
        test_indices=prediction["indices"],
        **per_origin,
    )
    if statistical_config is None:
        statistical_config = {
            "block_length": int(prepared["metadata"]["horizon"]),
            "bootstrap_repetitions": 2000,
            "confidence": 0.95,
            "seed": 314159,
        }
    paired = _paired_tests(
        reproducible,
        _main_model_losses(reference),
        int(prepared["metadata"]["horizon"]),
        statistical_config,
    )
    result = {
        "schema": "fconcord-retrained-ablation-v1",
        "variant": variant,
        "reference_run": str(reference),
        "seed": seed,
        "training_summary": training_summary,
        "test_metrics": metrics,
        "paired_tests_vs_full_fconcord": paired,
        "test_access_count": 1,
        "test_used_for_training_or_selection": False,
        "checkpoint_reload_before_test": True,
        "checkpoint_reload_max_abs_difference": reload_check["max_abs_difference"],
        "parameter_count_registered": count_parameters(reloaded),
        "ablation_semantics": {
            "NoGraph": (
                "zero graph at input, every transition, and decoder; graph-target "
                "loss weight is zero because no graph output is optimizable"
            ),
            "NoMarketState": "zero market state throughout training and inference",
            "NoRiskConcepts": (
                "zero risk concepts throughout the trajectory and mask their labels"
            ),
            "NoForwardConceptSupervision": "lambda_concept is exactly zero",
            "GaussianSymmetricDecoder": (
                "equal upper/lower scales and correlated Gaussian innovations"
            ),
        }[variant],
        "wall_seconds": float(time.perf_counter() - started),
    }
    json_dump(metrics, output / "tables" / "test_metrics.json")
    json_dump(result, output / "ablation_result.json")
    logger.info(
        "retrained_ablation_complete variant=%s rmse=%.8f es=%.8f seconds=%.2f",
        variant,
        metrics["rmse"],
        metrics["energy_score"],
        result["wall_seconds"],
    )
    return output


def run_ablation_protocol(
    reference_protocol: str | Path,
    output_directory: str | Path,
    variants: tuple[str, ...] = RETRAINED_ABLATIONS,
) -> Path:
    """Run every structural ablation over the reference folds and seeds."""
    reference_root = Path(reference_protocol).resolve()
    output = Path(output_directory).resolve()
    reference_manifest = _load_json(reference_root / "protocol_manifest.json")
    folds = [str(fold["id"]) for fold in reference_manifest["folds"]]
    seeds = [int(seed) for seed in reference_manifest["seeds"]]
    invalid = set(variants) - set(RETRAINED_ABLATIONS)
    if invalid:
        raise ValueError(f"Unknown variants: {sorted(invalid)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    manifest = {
        "schema": "fconcord-retrained-ablation-protocol-v1",
        "reference_protocol": str(reference_root),
        "folds": folds,
        "seeds": seeds,
        "variants": list(variants),
        "expected_runs": len(folds) * len(seeds) * len(variants),
        "same_data_splits_seeds_and_training_budget_as_full_model": True,
        "test_access_per_retrained_ablation": 1,
    }
    json_dump(manifest, output / "protocol_manifest.json")
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            reference_run = reference_root / "runs" / f"{fold}_seed_{seed}"
            for variant in variants:
                run_name = f"{fold}_seed_{seed}_{variant}"
                run_output = output / "runs" / run_name
                result_path = run_output / "ablation_result.json"
                if result_path.is_file():
                    result = _load_json(result_path)
                else:
                    if run_output.exists():
                        raise RuntimeError(
                            f"Incomplete ablation exists; audit before retry: {run_output}"
                        )
                    run_retrained_ablation(reference_run, variant, run_output)
                    result = _load_json(result_path)
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "run_directory": str(run_output),
                        **{
                            metric: float(result["test_metrics"][metric])
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
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {}
        for metric in ("rmse", "mae", "directional_accuracy", "energy_score", "msic", "msir"):
            values = np.asarray([row[metric] for row in variant_rows])
            aggregate[variant][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    summary = {
        **manifest,
        "runs_completed": len(rows),
        "all_runs_complete": len(rows) == manifest["expected_runs"],
        "rows": rows,
        "aggregate": aggregate,
    }
    json_dump(summary, output / "protocol_summary.json")
    return output
