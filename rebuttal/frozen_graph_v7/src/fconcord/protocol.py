from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fconcord.config import load_config
from fconcord.experiment import run_experiment
from fconcord.utils import json_dump


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol: dict[str, Any]) -> None:
    section = protocol.get("protocol", {})
    required = {"name", "base_config", "seeds", "folds"}
    missing = required - set(section)
    if missing:
        raise ValueError(f"Protocol is missing keys: {sorted(missing)}")
    seeds = [int(seed) for seed in section["seeds"]]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Protocol seeds must be nonempty and unique")
    minimum_seeds = int(section.get("minimum_seeds", 5))
    if len(seeds) < minimum_seeds:
        raise ValueError(
            f"Protocol requires at least {minimum_seeds} seeds, received {len(seeds)}"
        )
    folds = section["folds"]
    if not folds:
        raise ValueError("Protocol must define at least one fold")
    identifiers = [str(fold["id"]) for fold in folds]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Fold identifiers must be unique")
    date_keys = {
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    }
    previous_test_end: np.datetime64 | None = None
    for fold in folds:
        split = fold.get("split_dates", {})
        if date_keys - set(split):
            raise ValueError(f"Fold {fold['id']} has incomplete split_dates")
        values = [
            np.datetime64(str(split[key]), "D")
            for key in (
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
            )
        ]
        if not values[0] <= values[1] < values[2] <= values[3] < values[4] <= values[5]:
            raise ValueError(f"Fold {fold['id']} dates are not chronological")
        if previous_test_end is not None and values[4] <= previous_test_end:
            raise ValueError("Protocol test windows must be ordered and non-overlapping")
        previous_test_end = values[5]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "mse",
        "rmse",
        "mae",
        "directional_accuracy",
        "energy_score",
        "msic",
        "msir",
    ]
    overall: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        overall[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    by_fold: dict[str, Any] = {}
    for fold in sorted({str(row["fold"]) for row in rows}):
        fold_rows = [row for row in rows if str(row["fold"]) == fold]
        by_fold[fold] = {}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in fold_rows], dtype=np.float64)
            by_fold[fold][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
    return {"overall": overall, "by_fold": by_fold}


def _normalize_fold_dates(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-stable copy with every YAML date encoded as ISO text."""
    normalized = copy.deepcopy(folds)
    for fold in normalized:
        fold["id"] = str(fold["id"])
        fold["split_dates"] = {
            str(key): str(value) for key, value in fold["split_dates"].items()
        }
    return normalized


def _archive_incomplete_run(run_directory: Path, output: Path) -> Path:
    """Move an incomplete run to an audit-preserving recovery directory."""
    archive_root = output / "incomplete_runs"
    archive_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        destination = archive_root / f"{run_directory.name}_attempt_{attempt:03d}"
        if not destination.exists():
            run_directory.rename(destination)
            return destination
        attempt += 1


def run_protocol(
    protocol_path: str | Path, output_directory: str | Path | None = None
) -> Path:
    protocol_path = Path(protocol_path).resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    validate_protocol(protocol)
    section = protocol["protocol"]
    folds = _normalize_fold_dates(section["folds"])
    base_path = (protocol_path.parent / section["base_config"]).resolve()
    base = load_config(base_path)
    if output_directory is None:
        output = Path(base["experiment"]["run_root"]) / "protocols" / section["name"]
    else:
        output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "seeds": [int(seed) for seed in section["seeds"]],
        "folds": folds,
        "test_windows_preregistered_before_runs": True,
        "test_metrics_used_for_selection": False,
    }
    json_dump(manifest, output / "protocol_manifest.json")
    rows: list[dict[str, Any]] = []
    recovery_path = output / "protocol_recovery.json"
    if recovery_path.is_file():
        with recovery_path.open("r", encoding="utf-8") as handle:
            recovery_events = json.load(handle)
    else:
        recovery_events = []
    for fold in folds:
        for seed in section["seeds"]:
            run_name = f"{fold['id']}_seed_{int(seed)}"
            run_directory = output / "runs" / run_name
            summary_path = run_directory / "run_summary.json"
            reuse_completed = False
            if summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                reuse_completed = bool(
                    summary.get("artifact_integrity", {}).get("passed", False)
                )
            if not reuse_completed:
                if run_directory.exists():
                    archived = _archive_incomplete_run(run_directory, output)
                    recovery_events.append(
                        {
                            "run_name": run_name,
                            "archived_path": str(archived.resolve()),
                            "reason": "missing or failed artifact-integrity summary",
                        }
                    )
                    json_dump(recovery_events, recovery_path)
                config = copy.deepcopy(base)
                config["experiment"]["name"] = run_name
                config["experiment"]["seed"] = int(seed)
                config["data"]["split_dates"] = dict(fold["split_dates"])
                run_experiment(config, run_directory)
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            test_metrics = summary["test_metrics"]
            row = {
                "fold": str(fold["id"]),
                "seed": int(seed),
                "run_directory": str(run_directory.resolve()),
                "mse": float(
                    test_metrics.get("mse", float(test_metrics["rmse"]) ** 2)
                ),
                **{
                    key: float(test_metrics[key])
                    for key in (
                        "rmse",
                        "mae",
                        "directional_accuracy",
                        "energy_score",
                        "msic",
                        "msir",
                    )
                },
                "structural_certificate_passed": bool(
                    summary.get("gain_proxy_passed", False)
                ),
                "artifact_integrity_passed": bool(
                    summary["artifact_integrity"]["passed"]
                ),
            }
            rows.append(row)
            json_dump(rows, output / "protocol_progress.json")
    report = {
        "runs_completed": len(rows),
        "expected_runs": len(section["folds"]) * len(section["seeds"]),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "all_artifact_integrity_passed": all(
            row["artifact_integrity_passed"] for row in rows
        ),
        "structural_certificate_coverage": float(
            np.mean([row["structural_certificate_passed"] for row in rows])
        ),
    }
    json_dump(report, output / "protocol_summary.json")
    return output
