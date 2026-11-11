from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fconcord.calibration import calibrate_predictable_signal
from fconcord.data import FinancialTensorDataset
from fconcord.losses import total_loss
from fconcord.metrics import energy_score_numpy, point_metrics
from fconcord.model import FCONCORD
from fconcord.utils import json_dump


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(
    prepared: dict[str, Any], split: str, config: dict[str, Any], shuffle: bool
) -> DataLoader:
    dataset = FinancialTensorDataset(prepared, split)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def evaluate_epoch_split(
    model: FCONCORD,
    prepared: dict[str, Any],
    split: str,
    config: dict[str, Any],
    device: torch.device,
    evaluation_seed: int,
) -> dict[str, float]:
    """Evaluate one non-test split without changing the caller's RNG state.

    These are post-update, fixed-random-number metrics.  Using the same Monte Carlo
    draws at every epoch makes changes in the curves attributable to the model rather
    than to fresh scenario-sampling noise.  The test split is deliberately rejected:
    final-test evaluation belongs to ``run_experiment`` after checkpoint selection.
    """
    if split not in {"train", "val"}:
        raise ValueError("Per-epoch evaluation is restricted to train and val splits")
    loader = make_loader(prepared, split, config, shuffle=False)
    model.eval()
    component_sums: dict[str, float] = {}
    targets: list[np.ndarray] = []
    means: list[np.ndarray] = []
    sample_batches: list[np.ndarray] = []
    observations = 0
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(evaluation_seed))
        validation_samples = int(config.get("validation_samples", config["eval_samples"]))
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(
                batch["c0"], batch["a0"], batch["s0"], validation_samples
            )
            _, components = total_loss(outputs, batch, config, model.risk_indices)
            batch_size = int(batch["y"].shape[0])
            observations += batch_size
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + batch_size * value
            targets.append(batch["y"].detach().cpu().numpy())
            means.append(outputs["mu"].detach().cpu().numpy())
            sample_batches.append(outputs["samples"].detach().cpu().numpy())
    if observations == 0:
        raise RuntimeError(f"Empty {split} loader")
    target = np.concatenate(targets, axis=0)
    mean = np.concatenate(means, axis=0)
    samples = np.concatenate(sample_batches, axis=1)
    result = {key: value / observations for key, value in component_sums.items()}
    result.update(point_metrics(target, mean))
    result["energy_score"] = energy_score_numpy(samples, target)
    return result


def train_model(
    model: FCONCORD,
    prepared: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    logger: Any,
) -> tuple[FCONCORD, list[dict[str, float]], dict[str, Any]]:
    train_loader = make_loader(prepared, "train", config, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["epochs"]))
    )
    model.to(device)
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    selection_metric = str(config.get("selection_metric", "energy_score"))
    if selection_metric not in {"energy_score", "total", "mse", "rmse", "mae"}:
        raise ValueError(f"Unsupported validation selection metric: {selection_metric}")
    evaluation_seed = int(config.get("evaluation_seed", 271828))

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_records: list[dict[str, float]] = []
        grad_norms: list[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["c0"], batch["a0"], batch["s0"], int(config["train_samples"])
            )
            loss, components = total_loss(
                outputs, batch, config, model.risk_indices
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            train_records.append(components)
            grad_norms.append(float(grad_norm))

        train_metrics = evaluate_epoch_split(
            model, prepared, "train", config, device, evaluation_seed
        )
        val_metrics = evaluate_epoch_split(
            model, prepared, "val", config, device, evaluation_seed + 1
        )
        scheduler.step()

        record: dict[str, float] = {"epoch": float(epoch)}
        for key, value in train_metrics.items():
            record[f"train_{key}"] = float(value)
        for key, value in val_metrics.items():
            record[f"val_{key}"] = float(value)
        for key in train_records[0]:
            record[f"optimization_train_{key}"] = float(
                np.mean([row[key] for row in train_records])
            )
        record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        record["gradient_norm"] = float(np.mean(grad_norms))
        record["epoch_seconds"] = float(time.perf_counter() - epoch_start)
        history.append(record)
        pd.DataFrame(history).to_csv(
            run_dir / "logs" / "training_history.csv", index=False
        )
        logger.info(
            "epoch=%d "
            "train_total=%.6f train_es=%.6f train_mse=%.8f "
            "train_rmse=%.6f train_mae=%.6f "
            "train_da=%.4f train_msic=%.4f train_msir=%.4f "
            "val_total=%.6f val_es=%.6f val_mse=%.8f val_rmse=%.6f "
            "val_mae=%.6f val_da=%.4f val_msic=%.4f val_msir=%.4f "
            "rho=%.4f grad=%.4f seconds=%.2f",
            epoch,
            record["train_total"],
            record["train_energy_score"],
            record["train_mse"],
            record["train_rmse"],
            record["train_mae"],
            record["train_directional_accuracy"],
            record["train_msic"],
            record["train_msir"],
            record["val_total"],
            record["val_energy_score"],
            record["val_mse"],
            record["val_rmse"],
            record["val_mae"],
            record["val_directional_accuracy"],
            record["val_msic"],
            record["val_msir"],
            record["val_gain_radius"],
            record["gradient_norm"],
            record["epoch_seconds"],
        )

        selection_value = record[f"val_{selection_metric}"]
        if selection_value < best_val - float(config.get("selection_min_delta", 1e-8)):
            best_val = selection_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val": best_val,
                    "selection_metric": selection_metric,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metadata": prepared["metadata"],
                },
                run_dir / "checkpoints" / "best.pt",
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "epoch": epoch,
                "best_val": best_val,
                "selection_metric": selection_metric,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metadata": prepared["metadata"],
            },
            run_dir / "checkpoints" / "last.pt",
        )
        if stale_epochs >= int(config["patience"]):
            logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    calibration: dict[str, Any] | None = None
    if bool(config.get("signal_shrinkage_calibration", False)):
        if abs(model.signal_shrinkage - 1.0) > 1e-12:
            raise ValueError("signal calibration must start from unshrunk model output")
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(evaluation_seed + 17)
            validation_predictions = collect_predictions(
                model, prepared, "val", config, device
            )
        calibration = calibrate_predictable_signal(
            validation_predictions["mu"],
            validation_predictions["samples"],
            validation_predictions["y"],
            lower=float(config.get("signal_shrinkage_lower", 0.05)),
            upper=float(config.get("signal_shrinkage_upper", 1.0)),
            tolerance=float(config.get("signal_shrinkage_tolerance", 1e-5)),
        )
        model.signal_shrinkage = float(calibration["selected_signal_shrinkage"])
        checkpoint["signal_shrinkage"] = model.signal_shrinkage
        checkpoint["signal_shrinkage_validation"] = calibration
        torch.save(checkpoint, run_dir / "checkpoints" / "best.pt")
        json_dump(calibration, run_dir / "logs" / "signal_shrinkage_validation.json")
        logger.info(
            "signal_shrinkage_calibrated split=val rho=%.6f "
            "uncalibrated_es=%.8f calibrated_es=%.8f",
            model.signal_shrinkage,
            calibration["uncalibrated_validation_metrics"]["energy_score"],
            calibration["calibrated_validation_metrics"]["energy_score"],
        )
    summary = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "selection_metric": selection_metric,
        "train_seconds": time.perf_counter() - start_time,
        "epochs_completed": len(history),
        "signal_shrinkage_calibration": calibration,
    }
    return model, history, summary


@torch.no_grad()
def collect_predictions(
    model: FCONCORD,
    prepared: dict[str, Any],
    split: str,
    config: dict[str, Any],
    device: torch.device,
    input_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> dict[str, np.ndarray]:
    loader = make_loader(prepared, split, config, shuffle=False)
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    sample_batches: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        if input_transform is not None:
            batch = input_transform(batch)
        if split == "test":
            sample_count = int(config.get("test_samples", config["eval_samples"]))
        else:
            sample_count = int(config.get("validation_samples", config["eval_samples"]))
        outputs = model(batch["c0"], batch["a0"], batch["s0"], sample_count)
        for key in [
            "mu",
            "sigma_plus",
            "sigma_minus",
            "nu",
            "correlation",
            "precision",
        ]:
            collected.setdefault(key, []).append(outputs[key].detach().cpu().numpy())
        collected.setdefault("concepts", []).append(
            outputs["concept_path"].detach().cpu().numpy()
        )
        collected.setdefault("graphs", []).append(
            outputs["graph_path"].detach().cpu().numpy()
        )
        sample_batches.append(outputs["samples"].detach().cpu().numpy())
        collected.setdefault("y", []).append(batch["y"].detach().cpu().numpy())
        collected.setdefault("c0", []).append(batch["c0"].detach().cpu().numpy())
        collected.setdefault("a0", []).append(batch["a0"].detach().cpu().numpy())
        collected.setdefault("s0", []).append(batch["s0"].detach().cpu().numpy())
        collected.setdefault("c_target", []).append(batch["c_target"].detach().cpu().numpy())
        collected.setdefault("a_target", []).append(batch["a_target"].detach().cpu().numpy())
        indices.append(batch["index"].detach().cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    result["samples"] = np.concatenate(sample_batches, axis=1)
    result["indices"] = np.concatenate(indices)
    return result


def checkpoint_reload_check(
    model_factory: Callable[[], FCONCORD],
    checkpoint_path: Path,
    reference_model: FCONCORD,
    prepared: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[FCONCORD, dict[str, float | bool]]:
    # Reload verification must not inspect the final test set.  Validation data are
    # sufficient to prove that serialization preserves the deterministic model mean.
    loader = make_loader(prepared, "val", config, shuffle=False)
    batch = _move_batch(next(iter(loader)), device)
    reference_model.eval()
    torch.manual_seed(12345)
    validation_samples = int(config.get("validation_samples", config["eval_samples"]))
    with torch.no_grad():
        reference = reference_model(
            batch["c0"], batch["a0"], batch["s0"], validation_samples
        )["mu"]
    reloaded = model_factory().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state"])
    if "signal_shrinkage" in checkpoint:
        reloaded.signal_shrinkage = float(checkpoint["signal_shrinkage"])
    reloaded.eval()
    torch.manual_seed(12345)
    with torch.no_grad():
        restored = reloaded(
            batch["c0"], batch["a0"], batch["s0"], validation_samples
        )["mu"]
    max_difference = float((reference - restored).abs().max().cpu())
    return reloaded, {"max_abs_difference": max_difference, "passed": max_difference < 1e-7}
