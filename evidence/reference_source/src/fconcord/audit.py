from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from fconcord.utils import json_dump


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Run-integrity audit failed: {message}")


def audit_completed_run(run_dir: str | Path, minimum_figures: int = 28) -> dict[str, Any]:
    """Validate the saved experiment as an independently reloadable artifact."""
    run_path = Path(run_dir)
    summary = _load_json(run_path / "run_summary.json")
    metadata = _load_json(run_path / "audit" / "prepared_metadata.json")
    theory = _load_json(run_path / "audit" / "theory_audit.json")
    reload_check = _load_json(run_path / "audit" / "checkpoint_reload_check.json")
    manifest = _load_json(run_path / "figures" / "manifest.json")
    history = pd.read_csv(run_path / "logs" / "training_history.csv")

    _require((run_path / "checkpoints" / "best.pt").stat().st_size > 0, "best.pt missing")
    _require((run_path / "checkpoints" / "last.pt").stat().st_size > 0, "last.pt missing")
    _require(bool(reload_check["passed"]), "fresh-instance checkpoint reload mismatch")
    _require(bool(theory["hard_invariants_passed"]), "hard mathematical invariant failed")
    _require(len(manifest) >= minimum_figures, "figure count below preregistered minimum")
    _require(len({item["id"] for item in manifest}) == len(manifest), "duplicate figure IDs")
    _require(len({item["file"] for item in manifest}) == len(manifest), "duplicate figure files")
    _require(len(history) == int(summary["epochs_completed"]), "history/epoch count mismatch")
    _require(np.isfinite(history.select_dtypes(include=[np.number])).all().all(), "nonfinite log")
    if "epoch_metric_protocol" in summary:
        declared_metrics = summary["epoch_metric_protocol"].get(
            "metrics",
            [
                "total",
                "energy_score",
                "rmse",
                "mae",
                "directional_accuracy",
                "msic",
                "msir",
            ],
        )
        required_history_columns = {
            f"{split}_{metric}"
            for split in ("train", "val")
            for metric in declared_metrics
        }
        missing_columns = required_history_columns - set(history.columns)
        _require(not missing_columns, f"missing epoch metrics: {sorted(missing_columns)}")
        _require(
            not any(column.startswith("test_") for column in history.columns),
            "test metrics must not appear in per-epoch training history",
        )
        _require(
            not bool(summary["epoch_metric_protocol"]["test_used_for_model_selection"]),
            "test set was marked as used for model selection",
        )
        if not bool(summary.get("debug_run", False)) and "test_scenarios" in summary[
            "epoch_metric_protocol"
        ]:
            _require(
                int(summary["epoch_metric_protocol"]["test_scenarios"]) >= 256,
                "real-data final test used fewer than 256 joint scenarios",
            )
    if "manual_compliance" in summary:
        _require(
            (run_path / "audit" / "manual_contract.yaml").is_file(),
            "manual contract snapshot missing",
        )
        _require(
            (run_path / "audit" / "manual_compliance_summary.json").is_file(),
            "manual compliance summary missing",
        )
        _require(
            (run_path / "audit" / "METHODOLOGY_CODE_SPEC.md").is_file(),
            "code-normative methodology snapshot missing",
        )
        _require(
            (
                run_path
                / "audit"
                / "methodology_implementation_manifest.json"
            ).is_file(),
            "methodology implementation manifest missing",
        )
        _require(
            (run_path / "audit" / "empirical_jacobian_certificates.json").is_file(),
            "empirical Jacobian certificate report missing",
        )

    image_hashes: list[str] = []
    minimum_pixel_standard_deviation = float("inf")
    minimum_width = 10**9
    minimum_height = 10**9
    for item in manifest:
        path = run_path / "figures" / item["file"]
        _require(path.is_file() and path.stat().st_size > 0, f"missing figure {path.name}")
        pixels = mpimg.imread(path)
        _require(pixels.ndim in (2, 3), f"unreadable image shape for {path.name}")
        height, width = pixels.shape[:2]
        standard_deviation = float(np.std(pixels))
        _require(width >= 300 and height >= 250, f"undersized figure {path.name}")
        _require(standard_deviation > 1e-4, f"visually blank figure {path.name}")
        image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        minimum_pixel_standard_deviation = min(
            minimum_pixel_standard_deviation, standard_deviation
        )
        minimum_width = min(minimum_width, width)
        minimum_height = min(minimum_height, height)
    _require(len(set(image_hashes)) == len(image_hashes), "byte-identical figures detected")

    with np.load(run_path / "predictions" / "test_predictions.npz", allow_pickle=False) as data:
        predictions = {key: data[key] for key in data.files}
    _require(predictions["mu"].shape == predictions["y"].shape, "mu/label shape mismatch")
    _require(
        predictions["samples"].shape[1:] == predictions["y"].shape,
        "scenario/label shape mismatch",
    )
    for key, value in predictions.items():
        if np.issubdtype(value.dtype, np.number):
            _require(np.isfinite(value).all(), f"nonfinite prediction array: {key}")

    with np.load(run_path / "audit" / "prepared_arrays.npz", allow_pickle=False) as data:
        origins = data["origins"]
        graph_refit = data["graph_refit"] if "graph_refit" in data.files else None
        partial_corr = data["partial_corr"] if "partial_corr" in data.files else None
    splits = metadata["split_indices"]
    train = np.asarray(splits["train"], dtype=np.int64)
    validation = np.asarray(splits["val"], dtype=np.int64)
    test = np.asarray(splits["test"], dtype=np.int64)
    horizon = int(metadata["horizon"])
    _require(
        predictions["concepts"].shape[1] == horizon + 1,
        "concept path must contain h=0,...,H",
    )
    _require(
        predictions["graphs"].shape[1] == horizon + 1,
        "graph path must contain h=0,...,H",
    )
    _require(origins[train[-1]] + horizon < origins[validation[0]], "train/val overlap")
    _require(origins[validation[-1]] + horizon < origins[test[0]], "val/test overlap")
    if bool(metadata.get("split_label_end_enforced", False)):
        split_end_keys = {
            "train": "train_end",
            "val": "val_end",
            "test": "test_end",
        }
        for split, end_key in split_end_keys.items():
            last_label = np.datetime64(
                metadata["split_date_ranges"][split]["last_label_date"], "ns"
            )
            registered_end = np.datetime64(
                metadata["split_boundaries"][end_key], "ns"
            )
            _require(
                last_label <= registered_end,
                f"{split} label extends beyond its registered end date",
            )
    graph_refit_count: int | None = None
    graph_carry_forward_max_error: float | None = None
    if graph_refit is not None:
        graph_refit = np.asarray(graph_refit, dtype=np.bool_)
        graph_metadata = metadata["graph_estimation"]
        stride = int(graph_metadata["refit_stride"])
        anchor = int(graph_metadata["refit_anchor_origin"])
        expected_refit = (origins - anchor) % stride == 0
        expected_refit[0] = True
        _require(stride >= 1, "graph refit stride must be positive")
        _require(
            np.array_equal(graph_refit, expected_refit),
            "residual graph refit flags violate the registered cadence",
        )
        graph_refit_count = int(np.sum(graph_refit))
        _require(
            graph_refit_count == int(graph_metadata["refit_count"]),
            "graph refit count disagrees with metadata",
        )
        _require(
            bool(graph_metadata["carry_forward_is_causal"]),
            "graph carry-forward is not registered as causal",
        )
        carried = np.flatnonzero(~graph_refit)
        if len(carried) > 0:
            _require(partial_corr is not None, "partial correlations missing")
            graph_carry_forward_max_error = float(
                np.max(np.abs(partial_corr[carried] - partial_corr[carried - 1]))
            )
            _require(
                graph_carry_forward_max_error == 0.0,
                "a non-refit graph differs from the latest past estimate",
            )
        else:
            graph_carry_forward_max_error = 0.0

    report = {
        "passed": True,
        "checkpoint_reload_max_abs_difference": float(reload_check["max_abs_difference"]),
        "hard_invariants_passed": True,
        "figure_count": len(manifest),
        "unique_figure_hashes": len(set(image_hashes)),
        "minimum_figure_width": minimum_width,
        "minimum_figure_height": minimum_height,
        "minimum_pixel_standard_deviation": minimum_pixel_standard_deviation,
        "prediction_shape": list(predictions["mu"].shape),
        "scenario_shape": list(predictions["samples"].shape),
        "epochs_logged": len(history),
        "per_epoch_train_validation_metrics_verified": "epoch_metric_protocol" in summary,
        "test_metrics_absent_from_training_history": not any(
            column.startswith("test_") for column in history.columns
        ),
        "embargo": int(metadata["embargo"]),
        "graph_convergence_rate": float(metadata["graph_estimation"]["convergence_rate"]),
        "graph_fallback_rate": float(metadata["graph_estimation"]["fallback_rate"]),
        "graph_refit_count": graph_refit_count,
        "graph_carry_forward_max_error": graph_carry_forward_max_error,
    }
    json_dump(report, run_path / "audit" / "run_integrity_report.json")
    return report
