from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from fconcord.benchmark import run_benchmark
from fconcord.config import load_config, save_config
from fconcord.utils import json_dump

_METRICS = ("rmse", "mae", "directional_accuracy", "energy_score", "msic", "msir")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({str(row["model_name"]) for row in rows})
    output: dict[str, Any] = {}
    for model_name in model_names:
        model_rows = [row for row in rows if row["model_name"] == model_name]
        output[model_name] = {}
        for metric in _METRICS:
            values = np.asarray([row[metric] for row in model_rows], dtype=np.float64)
            output[model_name][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return output


def _archive_incomplete(path: Path, root: Path) -> Path:
    archive = root / "incomplete_runs"
    archive.mkdir(exist_ok=True)
    attempt = 1
    while True:
        destination = archive / f"{path.name}_attempt_{attempt:03d}"
        if not destination.exists():
            path.rename(destination)
            return destination
        attempt += 1


def run_benchmark_protocol(
    config_path: str | Path,
    reference_protocol: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run the benchmark matrix on every preregistered reference fold and seed."""
    config_path = Path(config_path).resolve()
    reference_root = Path(reference_protocol).resolve()
    output = Path(output_directory).resolve()
    config = load_config(config_path)
    reference_manifest = _load_json(reference_root / "protocol_manifest.json")
    seeds = [int(seed) for seed in reference_manifest["seeds"]]
    folds = [str(fold["id"]) for fold in reference_manifest["folds"]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    protocol_manifest = {
        "schema": "fconcord-benchmark-protocol-v1",
        "base_benchmark_config": str(config_path),
        "reference_protocol": str(reference_root),
        "folds": folds,
        "seeds": seeds,
        "expected_runs": len(folds) * len(seeds),
        "same_fold_seed_pairs_as_fconcord": True,
        "same_seed_across_hyperparameter_candidates_within_each_run": True,
        "selection_data": "validation only",
        "test_access_per_selected_model": 1,
    }
    json_dump(protocol_manifest, output / "protocol_manifest.json")

    rows: list[dict[str, Any]] = []
    recovery: list[dict[str, str]] = []
    for fold in folds:
        for seed in seeds:
            run_name = f"{fold}_seed_{seed}"
            reference_run = reference_root / "runs" / run_name
            run_output = output / "runs" / run_name
            benchmark_manifest = run_output / "benchmark_manifest.json"
            if benchmark_manifest.is_file():
                completed = _load_json(benchmark_manifest)
            else:
                if run_output.exists():
                    archived = _archive_incomplete(run_output, output)
                    recovery.append(
                        {
                            "run_name": run_name,
                            "archived_path": str(archived),
                            "reason": "missing benchmark_manifest.json",
                        }
                    )
                    json_dump(recovery, output / "protocol_recovery.json")
                run_config = copy.deepcopy(config)
                run_config["benchmark"]["name"] = run_name
                run_config["benchmark"]["seed"] = seed
                run_config_path = output / "configs" / f"{run_name}.yaml"
                save_config(run_config, run_config_path)
                run_benchmark(run_config_path, reference_run, run_output)
                completed = _load_json(benchmark_manifest)
            for model_name, result in completed["results"].items():
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "model_name": model_name,
                        "run_directory": str(run_output),
                        "selected_candidate_index": int(
                            result["selected_candidate_index"]
                        ),
                        **{
                            metric: float(result["test_metrics"][metric])
                            for metric in _METRICS
                        },
                    }
                )
            json_dump(rows, output / "protocol_progress.json")
    summary = {
        **protocol_manifest,
        "runs_completed": len(rows) // len(config["models"]),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "all_runs_complete": len(rows)
        == len(folds) * len(seeds) * len(config["models"]),
    }
    json_dump(summary, output / "protocol_summary.json")
    return output
