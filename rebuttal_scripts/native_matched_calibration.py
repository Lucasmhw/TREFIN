#!/usr/bin/env python3
"""Validation-only mean shrinkage of recovered native baseline checkpoints.

This supplements, and does not reconstruct, the manuscript's separate-host table.
No training or checkpoint reselection occurs. The historical residual distribution
is frozen while a scalar in [0.05, 1] is selected by validation energy score.

Run from the restored repository with its environment available:
  python /path/native_matched_calibration.py --repo . --out runs/rebuttal_native_rho
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", default="1")
    ap.add_argument("--seeds", default="1,7,19,37,73")
    ap.add_argument("--models", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "src"))
    import joblib
    import torch
    from fconcord.benchmark_classical import _predict_classical
    from fconcord.benchmark_models import make_benchmark_model
    from fconcord.benchmark_training import predict_neural_split, residual_bootstrap_scenarios
    from fconcord.calibration import calibrate_predictable_signal
    from fconcord.config import load_config
    from fconcord.metrics import point_metrics
    from fconcord.utils import seed_everything

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly")
    native = repo / "runs/benchmarks/finmultitime_us_native_3fold_5seed_v1/runs"
    folds = [int(s) for s in args.folds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    requested = set(args.models.split(",")) if args.models else None
    all_results = []
    missing = []
    started = time.time()
    protocol = {
        "schema": "trefin-native-matched-calibration-v1",
        "scope": "Additional native baseline comparison; NOT the manuscript separate-host Table 1",
        "checkpoint_selection": "Frozen historically selected checkpoint, no retraining/reselection",
        "calibration": {"selection_split": "validation", "metric": "energy_score", "bounds": [0.05, 1.0], "tolerance": 1e-5},
        "scenario_adapter": "point forecast plus fixed centered empirical joint training residual paths",
        "test_reuse": "Post-submission diagnostic replay of archived held-out predictions; no test-driven selection",
        "device_replay_tolerance_return_units": 1e-5,
        "device_replay_note": "Recovered checkpoints evaluated on requested device; original archived train residuals and test means retained exactly. Validation predictions may have bounded cross-device floating-point drift.",
        "folds": folds, "seeds": seeds, "models": sorted(requested) if requested else "all archived native models",
        "script_sha256": sha256(__file__), "device": str(device),
    }
    write_json(out / "protocol.json", protocol)

    for fold in folds:
        for seed in seeds:
            run = native / f"fold_{fold}_seed_{seed}"
            if not (run / "config_resolved.yaml").exists():
                missing.append({"run": run.name, "reason": "no archived run config"})
                continue
            config = load_config(run / "config_resolved.yaml")
            selection_config = {**config["training"], **config["evaluation"]}
            data_dir = run / "audit/benchmark_data"
            benchmark = {"metadata": read_json(data_dir / "benchmark_metadata.json")}
            with np.load(data_dir / "benchmark_arrays.npz", allow_pickle=False) as f:
                benchmark["arrays"] = {key: f[key] for key in f.files}
            train_pos = np.asarray(benchmark["metadata"]["split_indices"]["train"])
            val_pos = np.asarray(benchmark["metadata"]["split_indices"]["val"])
            test_pos = np.asarray(benchmark["metadata"]["split_indices"]["test"])
            assert not set(train_pos) & set(val_pos)
            assert not set(val_pos) & set(test_pos)
            assert max(train_pos) < min(val_pos) < min(test_pos)
            for name, spec in config["models"].items():
                if requested and name not in requested:
                    continue
                selected = run / "selected" / name
                result_path = selected / "result.json"
                if not result_path.exists():
                    missing.append({"run": run.name, "model": name, "reason": "no selected result"})
                    continue
                dest = out / run.name / name
                result_file = dest / "result.json"
                if result_file.exists():
                    previous = read_json(result_file)
                    if previous.get("script_sha256") != protocol["script_sha256"]:
                        raise RuntimeError(f"Resume script hash differs: {result_file}")
                    all_results.append(previous)
                    print(json.dumps({"event": "resume", "run": run.name, "model": name}), flush=True)
                    continue
                original = read_json(result_path)
                candidate = run / "candidates" / name / f"candidate_{int(original['selected_candidate_index']):02d}"
                classical = spec["kind"] == "classical"
                checkpoint = candidate / ("model.joblib" if classical else "checkpoints/best.pt")
                prediction_file = selected / "reproducible_predictions.npz"
                if not checkpoint.exists() or not prediction_file.exists():
                    missing.append({"run": run.name, "model": name, "reason": "checkpoint or archived predictions missing"})
                    continue
                checkpoint_hash = sha256(checkpoint)
                dest.mkdir(parents=True, exist_ok=True)
                seed_everything(seed, True)
                if classical:
                    model = joblib.load(checkpoint)
                    predict = lambda split: _predict_classical(name, model, benchmark, split, original["selected_parameters"])
                else:
                    ck = torch.load(checkpoint, map_location=device, weights_only=False)
                    model = make_benchmark_model(name, benchmark["metadata"]["dimensions"], original["selected_parameters"]).to(device)
                    model.load_state_dict(ck["model_state"])
                    predict = lambda split: predict_neural_split(model, benchmark, split, int(selection_config["batch_size"]), device)
                print(json.dumps({"event": "validation_started", "run": run.name, "model": name}), flush=True)
                train_prediction = predict("train")
                val_prediction = predict("val")
                assert np.array_equal(train_prediction["indices"], train_pos)
                assert np.array_equal(val_prediction["indices"], val_pos)
                regenerated_residual_pool = train_prediction["target"] - train_prediction["mean"]
                # NPZ members are loaded lazily: access only the TRAINING residual
                # member here. No test prediction/target member enters selection.
                with np.load(prediction_file, allow_pickle=False) as f:
                    residual_pool = f["train_residual_pool"]
                residual_error = float(np.max(np.abs(regenerated_residual_pool - residual_pool)))
                if residual_error > 1e-5:
                    raise RuntimeError(f"Training residual replay mismatch: {run.name}/{name}: {residual_error}")
                val_samples, val_draw_indices = residual_bootstrap_scenarios(
                    val_prediction["mean"], residual_pool,
                    int(selection_config["validation_samples"]), int(selection_config["evaluation_seed"]),
                )
                calibration = calibrate_predictable_signal(val_prediction["mean"], val_samples, val_prediction["target"], 0.05, 1.0, 1e-5)
                if not calibration["optimizer_success"]:
                    raise RuntimeError(f"Failed scalar optimization: {run.name}/{name}")
                validation_difference = abs(calibration["uncalibrated_validation_metrics"]["energy_score"] - original["validation_energy_score"])
                if validation_difference > 1e-5:
                    raise RuntimeError(f"Validation replay differs from historical score: {run.name}/{name}: {validation_difference}")
                rho = calibration["selected_signal_shrinkage"]
                # Persist the decision before reading/replaying any test predictions.
                write_json(dest / "validation_selection.json", calibration)
                np.savez_compressed(dest / "validation_predictions.npz", mean=val_prediction["mean"], target=val_prediction["target"], indices=val_pos, residual_indices=val_draw_indices)
                print(json.dumps({"event": "test_replay_started", "run": run.name, "model": name, "rho": rho}), flush=True)
                replay = predict("test")
                with np.load(prediction_file, allow_pickle=False) as f:
                    archived = {key: f[key] for key in f.files}
                assert np.array_equal(replay["indices"], archived["test_indices"])
                assert np.array_equal(replay["target"], archived["target"])
                reload_error = float(np.max(np.abs(replay["mean"] - archived["mean"])))
                assert np.array_equal(residual_pool, archived["train_residual_pool"])
                if max(reload_error, residual_error) > 1e-5:
                    raise RuntimeError(f"Historical replay mismatch {run.name}/{name}: {reload_error}, {residual_error}")
                mean = archived["mean"].astype(np.float64)
                target = archived["target"].astype(np.float64)
                pool = archived["train_residual_pool"]
                centered = pool - pool.mean(axis=0, keepdims=True)
                samples = (archived["mean"][None] + centered[archived["scenario_residual_indices"]]).astype(np.float32).astype(np.float64)
                translated = samples + (rho - 1.0) * mean[None]
                before = point_metrics(target, mean)
                after = point_metrics(target, rho * mean)
                before_es = archived["energy_score"].astype(np.float64)
                # Translation cancels the pairwise scenario term exactly. Reuse the
                # archived all-pairs ES and add only the changed observation term.
                first_before = np.linalg.norm(samples - target[None], axis=-1).mean(axis=(0, 2))
                first_after = np.linalg.norm(translated - target[None], axis=-1).mean(axis=(0, 2))
                after_es = before_es + first_after - first_before
                before["energy_score"] = float(before_es.mean())
                after["energy_score"] = float(after_es.mean())
                for metrics, scenario_set in [(before, samples), (after, translated)]:
                    lo, hi = np.quantile(scenario_set, [0.05, 0.95], axis=0)
                    metrics["coverage_90"] = float(((target >= lo) & (target <= hi)).mean())
                    metrics["width_90"] = float((hi - lo).mean())
                invariance_error = float(np.max(np.abs((translated[1]-translated[0])-(samples[1]-samples[0]))))
                assert invariance_error < 1e-12
                assert abs(before["energy_score"] - original["test_metrics"]["energy_score"]) < 1e-7
                assert checkpoint_hash == sha256(checkpoint)
                np.savez_compressed(dest / "paired_test_losses.npz", test_indices=archived["test_indices"], mse_before=np.mean((mean-target)**2,axis=(1,2)), mse_after=np.mean((rho*mean-target)**2,axis=(1,2)), mae_before=np.mean(np.abs(mean-target),axis=(1,2)), mae_after=np.mean(np.abs(rho*mean-target),axis=(1,2)), es_before=before_es, es_after=after_es)
                row = {
                    "fold": fold, "seed": seed, "model": name, "rho": rho,
                    "before": before, "after": after,
                    "scope": protocol["scope"], "script_sha256": protocol["script_sha256"],
                    "source_checkpoint": str(checkpoint.relative_to(repo)), "checkpoint_sha256": checkpoint_hash,
                    "source_predictions": str(prediction_file.relative_to(repo)), "prediction_sha256": sha256(prediction_file),
                    "selected_candidate": original["selected_candidate_index"],
                    "validation_score_replay_abs_difference": validation_difference,
                    "test_mean_replay_max_error": reload_error, "training_residual_replay_max_error": residual_error,
                    "pairwise_translation_invariance_max_error": invariance_error,
                    "train_origins": len(train_pos), "validation_origins": len(val_pos), "test_origins": len(test_pos),
                    "scenario_adapter": protocol["scenario_adapter"],
                }
                write_json(result_file, row)
                all_results.append(row)
                print(json.dumps({"event": "complete", "run": run.name, "model": name, "rho": rho, "mse_before": before["mse"], "mse_after": after["mse"], "es_before": before["energy_score"], "es_after": after["energy_score"]}), flush=True)
                del model, samples, translated, archived, train_prediction, val_prediction, replay
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del benchmark
    summary = []
    for fold in folds:
        for name in sorted({r["model"] for r in all_results if r["fold"] == fold}):
            rows = [r for r in all_results if r["fold"] == fold and r["model"] == name]
            group = {"fold": fold, "model": name, "n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
            group["rho_mean"] = float(np.mean([r["rho"] for r in rows]))
            for stage in ["before", "after"]:
                for metric in ["mse", "rmse", "mae", "energy_score", "msic", "msir", "coverage_90", "width_90"]:
                    values = [r[stage][metric] for r in rows]
                    group[f"{stage}_{metric}_mean"] = float(np.mean(values))
                    group[f"{stage}_{metric}_sample_std"] = float(np.std(values, ddof=1)) if len(rows)>1 else None
            summary.append(group)
    write_json(out / "summary.json", {"protocol": protocol, "groups": summary, "runs": all_results, "missing": missing, "wall_seconds": time.time()-started})
    if summary:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({"event": "all_complete", "completed": len(all_results), "missing": len(missing), "seconds": time.time()-started}), flush=True)


if __name__ == "__main__":
    main()
