#!/usr/bin/env python3
"""Direct decoder sensitivities at archived, evolved TREFIN states.

Example (run with the restored source on PYTHONPATH):
  python mean_sensitivity.py --protocol-dir /path/to/calibrated_protocol \
    --prepared-protocol-dir /path/to/original_protocol --output-dir /path/to/out

Expected archive layout: PROTOCOL/runs/fold_1_seed_1/config_resolved.yaml,
checkpoints/best.pt, and seed-1 audit/prepared_{arrays.npz,metadata.json}.
Only current measured inputs and test-origin indices are used; labels are never
read by the diagnostic computation. No fitting or checkpoint writing occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fconcord.config import load_config
from fconcord.experiment import _model_factory, load_prepared


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_path(protocol: Path, fold: int, seed: int) -> Path:
    name = f"fold_{fold}_seed_{seed}"
    candidates = [protocol / "runs" / name, protocol / name]
    found = [candidate for candidate in candidates if candidate.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one run directory for {name}: {candidates}")
    return found[0]


def summarize(values: np.ndarray) -> dict[str, Any]:
    """Summarize over seeds, sampled origins, and assets; keep lead, concept."""
    if values.ndim != 5:
        raise ValueError("Expected derivative layout [seed, origin, lead, asset, concept]")
    flattened = values.transpose(2, 4, 0, 1, 3).reshape(values.shape[2], values.shape[4], -1)
    quantiles = np.quantile(flattened, [0.05, 0.25, 0.5, 0.75, 0.95], axis=-1)
    return {
        "array_axes": ["lead", "concept"],
        "signed_median": quantiles[2].tolist(),
        "q05": quantiles[0].tolist(),
        "q25": quantiles[1].tolist(),
        "q75": quantiles[3].tolist(),
        "q95": quantiles[4].tolist(),
        "mean_absolute": np.abs(flattened).mean(axis=-1).tolist(),
        "fraction_positive": (flattened > 0).mean(axis=-1).tolist(),
        "fraction_negative": (flattened < 0).mean(axis=-1).tolist(),
        "fraction_exact_zero": (flattened == 0).mean(axis=-1).tolist(),
        "sample_count_per_lead_concept": int(flattened.shape[-1]),
        "uncertainty_interpretation": "Empirical distribution of fitted partial derivatives, not a sampling confidence interval or causal effect.",
    }


def _diagonal_derivative(output: torch.Tensor, concepts: torch.Tensor, asset: int) -> torch.Tensor:
    # Batch items are independent: summing output for one asset across origins
    # yields one own-asset derivative row per origin without cross-origin terms.
    gradient = torch.autograd.grad(output[:, asset].sum(), concepts, retain_graph=True)[0]
    return gradient[:, asset, :]


def decode_derivatives(model, concepts: torch.Tensor, graph: torch.Tensor,
                       market: torch.Tensor, lead_index: int, positive_scale: float):
    """No reroll: C is a fresh leaf; graph, market, and lead are held fixed."""
    rho = float(model.signal_shrinkage)
    c = concepts.detach().clone().requires_grad_(True)
    a = graph.detach()
    s = market.detach()
    lead = model._lead(c.shape[0], lead_index, c.device).detach()
    calibrated = model._decode_state(c, a, s, lead)
    try:
        model.signal_shrinkage = 1.0
        structural = model._decode_state(c, a, s, lead)
    finally:
        model.signal_shrinkage = rho
    cal_rows, raw_rows, risk_rows, scaled_risk_rows = [], [], [], []
    for asset in range(model.num_assets):
        cal_rows.append(_diagonal_derivative(calibrated["mu"], c, asset).detach().cpu().numpy())
        raw_rows.append(_diagonal_derivative(structural["mu"], c, asset).detach().cpu().numpy())
        risk = _diagonal_derivative(calibrated["sigma_minus"], c, asset)
        scaled = _diagonal_derivative(positive_scale * calibrated["sigma_minus"], c, asset)
        risk_rows.append(risk.index_select(-1, model.risk_indices).detach().cpu().numpy())
        scaled_risk_rows.append(scaled.index_select(-1, model.risk_indices).detach().cpu().numpy())
    # Rows are [asset, origin, concept]; output is [origin, asset, concept].
    return tuple(np.stack(rows, axis=1) for rows in (cal_rows, raw_rows, risk_rows, scaled_risk_rows))


def audit_one_run(run: Path, prepared: dict[str, Any], selected_indices: np.ndarray,
                  device: torch.device, positive_scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config_path = run / "config_resolved.yaml"
    checkpoint_path = run / "checkpoints" / "best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    config_hash = sha256(config_path)
    config = load_config(config_path)
    if config["model"].get("ablation_variant", "Full") != "Full":
        raise ValueError(f"Expected original Full checkpoint, got {run}")
    model = _model_factory(prepared, config).to(device).eval()
    # The checkpoint is supplied by the author; load optimizer/metadata format
    # faithfully but use only the fixed model weights and recorded calibration.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("num_assets", "num_concepts", "horizon", "asset_names", "concept_names", "split_indices"):
        recorded = checkpoint.get("metadata", {}).get(key)
        if recorded is not None and recorded != prepared["metadata"].get(key):
            raise ValueError(f"Prepared/checkpoint metadata mismatch for {key}: {run}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    rho_source = "checkpoint"
    if "signal_shrinkage" in checkpoint:
        rho = float(checkpoint["signal_shrinkage"])
    else:
        rho_source = "resolved_config"
        rho = float(config["model"].get("signal_shrinkage", 1.0))
        if bool(config["training"].get("signal_shrinkage_calibration", False)) and rho == 1.0:
            raise ValueError(f"Calibrated run has no archived fitted scalar: {run}")
    if not 0 < rho <= 1:
        raise ValueError(f"Invalid mean calibration {rho}")
    model.signal_shrinkage = rho
    model.requires_grad_(False)
    inputs = {
        key: torch.as_tensor(prepared["arrays"][key][selected_indices], device=device)
        for key in ("c0", "a0", "s0")
    }
    # Match forward's initial sanitization; Full keeps all measured coordinates.
    concepts = inputs["c0"] * model.active_concept_mask.view(1, 1, -1)
    graph = inputs["a0"]
    market = inputs["s0"]
    states = []
    with torch.no_grad():
        for h in range(model.horizon):
            update_lead = model._lead(len(selected_indices), h, device)
            concepts, graph = model._step(concepts, graph, market, update_lead)
            states.append((concepts.detach().clone(), graph.detach().clone()))
    cal_values, raw_values, risk_values, scaled_values = [], [], [], []
    for h, (c, a) in enumerate(states, start=1):
        cal, raw, risk, scaled = decode_derivatives(model, c, a, market, h, positive_scale)
        cal_values.append(cal)
        raw_values.append(raw)
        risk_values.append(risk)
        scaled_values.append(scaled)
    arrays = {
        "mean_derivative_calibrated": np.stack(cal_values, axis=1),
        "mean_derivative_structural": np.stack(raw_values, axis=1),
        "downside_scale_derivative": np.stack(risk_values, axis=1),
        "scaled_downside_scale_derivative": np.stack(scaled_values, axis=1),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Nonfinite direct derivative: {run}")
    calibrated = arrays["mean_derivative_calibrated"]
    structural = arrays["mean_derivative_structural"]
    risk = arrays["downside_scale_derivative"]
    scaled = arrays["scaled_downside_scale_derivative"]
    rho_error = float(np.abs(calibrated - rho * structural).max())
    scale_error = float(np.abs(scaled - positive_scale * risk).max())
    np.testing.assert_allclose(calibrated, rho * structural, rtol=1e-4, atol=2e-9)
    np.testing.assert_allclose(scaled, positive_scale * risk, rtol=1e-4, atol=2e-9)
    if not np.all(risk > 0) or not np.all(scaled > 0):
        raise AssertionError(f"Nonpositive designated direct risk derivative: {run}")
    if sha256(checkpoint_path) != checkpoint_hash or sha256(config_path) != config_hash:
        raise RuntimeError("Source checkpoint/config changed during read-only sensitivity audit")
    audit = {
        "run": str(run.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "signal_shrinkage": rho,
        "signal_shrinkage_source": rho_source,
        "mean_derivative_scaling_max_abs_error": rho_error,
        "mean_derivative_sign_mismatches": int(np.count_nonzero(np.sign(calibrated) != np.sign(structural))),
        "positive_innovation_scale": positive_scale,
        "risk_derivative_scaling_max_abs_error": scale_error,
        "minimum_direct_downside_scale_derivative": float(risk.min()),
        "minimum_scaled_direct_downside_scale_derivative": float(scaled.min()),
        "risk_minimum_by_lead_and_risk_coordinate": risk.min(axis=(0, 2)).tolist(),
        "scaled_risk_minimum_by_lead_and_risk_coordinate": scaled.min(axis=(0, 2)).tolist(),
        "all_risk_derivatives_strictly_positive": True,
        "checkpoint_and_config_unchanged": True,
    }
    del checkpoint, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--prepared-protocol-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 19, 37, 73])
    parser.add_argument("--origins", type=int, default=32)
    parser.add_argument("--positive-scale", type=float, default=1.25,
                        help="Fixed positive algebraic audit factor; not a fitted coverage calibration")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.origins < 1 or args.positive_scale <= 0:
        parser.error("origins and positive-scale must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    started = time.perf_counter()
    prepared_protocol = args.prepared_protocol_dir or args.protocol_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "mean_sensitivity.json"
    output_npz = args.output_dir / "mean_sensitivity_arrays.npz"
    if output_json.exists() or output_npz.exists():
        raise FileExistsError("Sensitivity deliverables already exist; choose a new output directory")
    result: dict[str, Any] = {
        "schema": "trefin-direct-decoder-sensitivity-v1",
        "derivative": "own-asset partial derivative d mu[t,h,i] / d C_hat[t,h,i,k]",
        "units": "decimal one-period excess return per one robust-standardized concept unit",
        "held_fixed": ["evolved graph", "all other evolved concept coordinates", "observed market state", "lead embedding", "fitted parameters", "validation mean scalar"],
        "state_source": "original model rollout from measured input at each sampled held-out origin",
        "reroll_during_intervention": False,
        "selection": "32 (or requested number of) evenly spaced chronological test-origin positions per fold; same positions for every seed",
        "test_labels_used": False,
        "fitting_or_model_selection": False,
        "positive_scale_audit_is_fitted_calibration": False,
        "positive_scale_audit": args.positive_scale,
        "quantiles_are": "descriptive quantiles of derivatives, not confidence intervals",
        "seeds": args.seeds,
        "device": str(device),
        "torch_version": torch.__version__,
        "script_sha256": sha256(Path(__file__)),
        "folds": {},
    }
    saved_arrays: dict[str, np.ndarray] = {}
    for fold in args.folds:
        prepared_dir = run_path(prepared_protocol, fold, 1) / "audit"
        prepared = load_prepared(prepared_dir)
        # load_prepared reads its bundled archive; discard unused future labels
        # immediately. They are never passed to a model, derivative, or summary.
        needed = {"c0", "a0", "s0", "origins", "directional_mask", "risk_indices", "dates"}
        prepared["arrays"] = {key: value for key, value in prepared["arrays"].items() if key in needed}
        metadata = prepared["metadata"]
        test_indices = np.asarray(metadata["split_indices"]["test"], dtype=np.int64)
        if len(test_indices) < args.origins:
            raise ValueError(f"Fold {fold} has fewer test origins than requested")
        selected_positions = np.linspace(0, len(test_indices) - 1, args.origins, dtype=np.int64)
        selected = test_indices[selected_positions]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Origin sampling produced duplicate indices")
        per_seed, audits = [], []
        for seed in args.seeds:
            run = run_path(args.protocol_dir, fold, seed)
            audit, arrays = audit_one_run(run, prepared, selected, device, args.positive_scale)
            audit["seed"] = seed
            audits.append(audit)
            per_seed.append(arrays)
            print(json.dumps({"event": "run_complete", "fold": fold, "seed": seed,
                              "rho": audit["signal_shrinkage"],
                              "minimum_risk_derivative": audit["minimum_direct_downside_scale_derivative"]}), flush=True)
        stacked = {key: np.stack([entry[key] for entry in per_seed], axis=0) for key in per_seed[0]}
        for key, value in stacked.items():
            saved_arrays[f"fold_{fold}__{key}"] = value
        saved_arrays[f"fold_{fold}__selected_prepared_indices"] = selected
        saved_arrays[f"fold_{fold}__selected_test_positions"] = selected_positions
        origin_rows = np.asarray(prepared["arrays"]["origins"])[selected]
        saved_arrays[f"fold_{fold}__selected_original_origin_rows"] = origin_rows
        fold_result = {
            "concept_names": metadata["concept_names"],
            "asset_names": metadata["asset_names"],
            "risk_indices": np.asarray(prepared["arrays"]["risk_indices"]).tolist(),
            "leads": list(range(1, int(metadata["horizon"]) + 1)),
            "selected_test_positions": selected_positions.tolist(),
            "selected_prepared_indices": selected.tolist(),
            "selected_original_origin_rows": origin_rows.tolist(),
            "prepared_metadata_sha256": sha256(prepared_dir / "prepared_metadata.json"),
            "prepared_arrays_sha256": sha256(prepared_dir / "prepared_arrays.npz"),
            "mean_calibrated": summarize(stacked["mean_derivative_calibrated"]),
            "mean_structural": summarize(stacked["mean_derivative_structural"]),
            "run_audits": audits,
        }
        if "dates" in prepared["arrays"]:
            dates = np.asarray(prepared["arrays"]["dates"])
            if int(origin_rows.max()) < len(dates):
                fold_result["selected_origin_dates"] = dates[origin_rows].astype(str).tolist()
        result["folds"][f"fold_{fold}"] = fold_result
    saved_arrays["seeds"] = np.asarray(args.seeds, dtype=np.int64)
    result["npz_layout"] = {
        "mean_derivatives": ["seed", "sampled_origin", "lead", "asset", "concept"],
        "risk_derivatives": ["seed", "sampled_origin", "lead", "asset", "risk_coordinate"],
    }
    result["wall_seconds"] = time.perf_counter() - started
    np.savez_compressed(output_npz, **saved_arrays)
    result["arrays_file"] = output_npz.name
    result["arrays_sha256"] = sha256(output_npz)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "json": str(output_json), "npz": str(output_npz),
                      "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
