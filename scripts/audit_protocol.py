"""Independently audit every run in a completed F-CONCORD protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date(value: str) -> np.datetime64:
    return np.datetime64(value.split("T", maxsplit=1)[0], "D")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _audit_run(
    run_directory: Path, fold: str, seed: int, require_large_arrays: bool
) -> dict[str, Any]:
    required = [
        "run_summary.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "logs/run.log",
        "logs/training_history.csv",
        "audit/checkpoint_reload_check.json",
        "audit/data_provenance.json",
        "audit/empirical_jacobian_certificates.json",
        "audit/methodology_implementation_manifest.json",
        "audit/prepared_metadata.json",
        "audit/run_integrity_report.json",
        "audit/theory_audit.json",
        "figures/manifest.json",
        "tables/test_metrics.json",
    ]
    if require_large_arrays:
        required.append("predictions/test_predictions.npz")
    for name in required:
        _require((run_directory / name).is_file(), f"{run_directory}: missing {name}")

    summary = _load_json(run_directory / "run_summary.json")
    integrity = _load_json(run_directory / "audit/run_integrity_report.json")
    reload_check = _load_json(run_directory / "audit/checkpoint_reload_check.json")
    metadata = _load_json(run_directory / "audit/prepared_metadata.json")
    theory = _load_json(run_directory / "audit/theory_audit.json")
    local = _load_json(run_directory / "audit/empirical_jacobian_certificates.json")
    provenance = _load_json(run_directory / "audit/data_provenance.json")
    implementation = _load_json(
        run_directory / "audit/methodology_implementation_manifest.json"
    )

    _require(bool(summary["artifact_integrity"]["passed"]), f"{run_directory}: summary")
    _require(bool(integrity["passed"]), f"{run_directory}: integrity report")
    _require(bool(theory["hard_invariants_passed"]), f"{run_directory}: invariants")
    _require(bool(reload_check["passed"]), f"{run_directory}: checkpoint reload")
    _require(
        float(reload_check["max_abs_difference"]) == 0.0,
        f"{run_directory}: nonzero checkpoint-reload difference",
    )

    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint = torch.load(
            run_directory / "checkpoints" / checkpoint_name,
            map_location="cpu",
            weights_only=False,
        )
        _require("model_state" in checkpoint, f"{run_directory}: invalid {checkpoint_name}")
        _require(len(checkpoint["model_state"]) > 0, f"{run_directory}: empty {checkpoint_name}")

    with (run_directory / "logs/training_history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        history = list(csv.DictReader(handle))
    _require(len(history) == int(integrity["epochs_logged"]), f"{run_directory}: epochs")
    columns = set(history[0]) if history else set()
    _require(not any(name.startswith("test_") for name in columns), f"{run_directory}: leak")
    _require(
        bool(integrity["per_epoch_train_validation_metrics_verified"]),
        f"{run_directory}: per-epoch metrics",
    )
    _require(
        bool(integrity["test_metrics_absent_from_training_history"]),
        f"{run_directory}: test metrics in history",
    )

    log_text = (run_directory / "logs/run.log").read_text(encoding="utf-8")
    _require(log_text.count("final_test_started") == 1, f"{run_directory}: test starts")
    _require(log_text.count("final_test_complete") == 1, f"{run_directory}: test completes")
    reload_position = log_text.find("checkpoint_reload")
    test_position = log_text.find("final_test_started")
    _require(0 <= reload_position < test_position, f"{run_directory}: reload/test order")

    _require(bool(metadata["split_label_end_enforced"]), f"{run_directory}: label end")
    _require(int(metadata["embargo"]) >= int(metadata["horizon"]), f"{run_directory}: purge")
    ranges = metadata["split_date_ranges"]
    boundaries = metadata["split_boundaries"]
    for split in ("train", "val", "test"):
        _require(
            _date(ranges[split]["forecast_origin_start"])
            >= _date(boundaries[f"{split}_start"]),
            f"{run_directory}: {split} origin starts before boundary",
        )
        _require(
            _date(ranges[split]["last_label_date"])
            <= _date(boundaries[f"{split}_end"]),
            f"{run_directory}: {split} label exceeds boundary",
        )
    _require(
        _date(ranges["train"]["last_label_date"])
        < _date(ranges["val"]["forecast_origin_start"]),
        f"{run_directory}: train/validation overlap",
    )
    _require(
        _date(ranges["val"]["last_label_date"])
        < _date(ranges["test"]["forecast_origin_start"]),
        f"{run_directory}: validation/test overlap",
    )

    figure_manifest = _load_json(run_directory / "figures/manifest.json")
    figure_paths = [run_directory / "figures" / row["file"] for row in figure_manifest]
    _require(len(figure_paths) >= 50, f"{run_directory}: fewer than 50 figures")
    _require(all(path.is_file() for path in figure_paths), f"{run_directory}: missing figure")
    figure_hashes = [_sha256(path) for path in figure_paths]
    _require(len(figure_hashes) == len(set(figure_hashes)), f"{run_directory}: duplicate figure")
    _require(
        int(integrity["unique_figure_hashes"]) == len(figure_hashes),
        f"{run_directory}: figure hash report",
    )

    test_metrics = summary["test_metrics"]
    for metric in ("rmse", "mae", "directional_accuracy", "energy_score", "msic", "msir"):
        _require(math.isfinite(float(test_metrics[metric])), f"{run_directory}: {metric}")

    local_max = max(float(section["radius_max"]) for section in local.values())
    graph = metadata["graph_estimation"]
    source_hashes = {
        name: str(record["sha256"]) for name, record in provenance["files"].items()
    }
    return {
        "fold": fold,
        "seed": seed,
        "run_directory": str(run_directory.resolve()),
        "checkpoint_reload_max_abs_difference": float(reload_check["max_abs_difference"]),
        "epochs_logged": len(history),
        "single_final_test_verified": True,
        "large_prediction_array_present": (
            run_directory / "predictions/test_predictions.npz"
        ).is_file(),
        "split_date_ranges": ranges,
        "figure_count": len(figure_paths),
        "unique_figure_hashes": len(set(figure_hashes)),
        "graph_convergence_rate": float(graph["convergence_rate"]),
        "graph_fallback_rate": float(graph["fallback_rate"]),
        "graph_carry_forward_max_error": float(integrity["graph_carry_forward_max_error"]),
        "global_structural_radius": float(theory["gain_radius"]),
        "global_structural_certificate_passed": bool(theory["structural_certificate_passed"]),
        "local_sampled_radius_max": local_max,
        "local_sampled_contraction_coverage": min(
            float(section["contraction_coverage_rho_lt_1"]) for section in local.values()
        ),
        "source_hashes": source_hashes,
        "implementation_combined_sha256": implementation["implementation_combined_sha256"],
    }


def _aggregate_comparisons(run_directories: list[Path]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group, filename in (
        ("lightweight_baselines", "baseline_metrics.json"),
        ("inference_ablations", "ablation_metrics.json"),
    ):
        tables = [_load_json(path / "tables" / filename) for path in run_directories]
        _require(all(set(table) == set(tables[0]) for table in tables), f"{group}: names")
        output[group] = {}
        for model_name in sorted(tables[0]):
            scalar_metrics = {
                metric
                for metric, value in tables[0][model_name].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            output[group][model_name] = {}
            for metric in sorted(scalar_metrics):
                values = np.asarray(
                    [float(table[model_name][metric]) for table in tables], dtype=np.float64
                )
                output[group][model_name][metric] = {
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
    return output


def audit_protocol(
    protocol_directory: Path, require_large_arrays: bool = True
) -> dict[str, Any]:
    protocol_directory = protocol_directory.resolve()
    manifest = _load_json(protocol_directory / "protocol_manifest.json")
    summary = _load_json(protocol_directory / "protocol_summary.json")
    expected_pairs = {
        (str(fold["id"]), int(seed))
        for fold in manifest["folds"]
        for seed in manifest["seeds"]
    }
    actual_pairs = {(str(row["fold"]), int(row["seed"])) for row in summary["rows"]}
    _require(actual_pairs == expected_pairs, "protocol rows do not match registered fold/seed pairs")
    _require(int(summary["runs_completed"]) == len(expected_pairs), "incomplete protocol")
    _require(int(summary["expected_runs"]) == len(expected_pairs), "expected-run mismatch")
    _require(bool(summary["all_artifact_integrity_passed"]), "protocol integrity failed")
    _require(bool(manifest["test_windows_preregistered_before_runs"]), "test preregistration")
    _require(not bool(manifest["test_metrics_used_for_selection"]), "test used for selection")

    rows = []
    run_directories: set[Path] = set()
    for row in summary["rows"]:
        run_directory = Path(row["run_directory"])
        if not run_directory.is_dir():
            run_directory = (
                protocol_directory
                / "runs"
                / f"{row['fold']!s}_seed_{int(row['seed'])}"
            )
        _require(
            run_directory.is_dir(),
            f"missing run directory for {row['fold']}/{row['seed']}",
        )
        run_directory = run_directory.resolve()
        _require(run_directory not in run_directories, "duplicate run directory")
        run_directories.add(run_directory)
        rows.append(
            _audit_run(
                run_directory,
                str(row["fold"]),
                int(row["seed"]),
                require_large_arrays=require_large_arrays,
            )
        )

    source_sets = {json.dumps(row["source_hashes"], sort_keys=True) for row in rows}
    implementation_hashes = {row["implementation_combined_sha256"] for row in rows}
    _require(len(source_sets) == 1, "source-data hashes differ between protocol runs")
    _require(len(implementation_hashes) == 1, "implementation hashes differ between runs")
    total_figures = sum(int(row["figure_count"]) for row in rows)
    report = {
        "passed": True,
        "protocol_directory": str(protocol_directory),
        "runs_audited": len(rows),
        "registered_fold_seed_pairs": len(expected_pairs),
        "all_checkpoint_files_loadable": True,
        "all_reload_differences_zero": all(
            row["checkpoint_reload_max_abs_difference"] == 0.0 for row in rows
        ),
        "all_single_final_test_sequences_verified": True,
        "compact_evidence_mode": not require_large_arrays,
        "large_prediction_arrays_present": all(
            row["large_prediction_array_present"] for row in rows
        ),
        "all_label_paths_within_registered_split_ends": True,
        "all_temporal_purges_verified": True,
        "total_figures": total_figures,
        "all_figures_unique_within_run": True,
        "minimum_graph_convergence_rate": min(
            row["graph_convergence_rate"] for row in rows
        ),
        "maximum_graph_fallback_rate": max(row["graph_fallback_rate"] for row in rows),
        "maximum_graph_carry_forward_error": max(
            row["graph_carry_forward_max_error"] for row in rows
        ),
        "global_structural_certificate_coverage": float(
            np.mean([row["global_structural_certificate_passed"] for row in rows])
        ),
        "global_structural_radius_range": [
            min(row["global_structural_radius"] for row in rows),
            max(row["global_structural_radius"] for row in rows),
        ],
        "maximum_local_sampled_radius": max(row["local_sampled_radius_max"] for row in rows),
        "minimum_local_sampled_contraction_coverage": min(
            row["local_sampled_contraction_coverage"] for row in rows
        ),
        "source_hashes_consistent": True,
        "implementation_hash_consistent": True,
        "implementation_combined_sha256": next(iter(implementation_hashes)),
        "comparison_aggregate": _aggregate_comparisons(sorted(run_directories)),
        "rows": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol_directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: <protocol_directory>/protocol_audit.json)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="audit a compact evidence copy where large prediction arrays were intentionally omitted",
    )
    args = parser.parse_args()
    report = audit_protocol(
        args.protocol_directory, require_large_arrays=not args.compact
    )
    output = args.output or args.protocol_directory / "protocol_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
