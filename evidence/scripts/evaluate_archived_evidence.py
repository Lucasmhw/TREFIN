"""Rebuttal diagnostics from immutable, stored TREFIN prediction files.

Usage: python evaluate_archived_evidence.py --manifest inputs.json --output out
Manifest: {"runs": [{"fold":"fold_1", "seed":1,
 "predictions":"/path/predictions/test_predictions.npz",
 "prepared_dir":"/path/audit"}]}
prepared_dir is optional, but required for the training-mean baseline. It contains
prepared_arrays.npz + prepared_metadata.json. Relative paths resolve to manifest.

No fitting, selection, or test-based tuning occurs here. Seeds are averaged at
each origin BEFORE temporal block bootstrap; they never become extra observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "seed_sample_std": float(v.std(ddof=1)) if len(v) > 1 else None,
            "values": v.tolist(), "seed_count": len(v)}


def paired_bootstrap(difference: np.ndarray, block: int, repetitions: int, seed: int) -> dict[str, Any]:
    d = np.asarray(difference, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Paired differences must be a finite origin-level vector")
    block = min(block, len(d))
    rng = np.random.default_rng(seed)
    boots = np.empty(repetitions)
    offsets = np.arange(block)
    for i in range(repetitions):
        starts = rng.integers(len(d), size=int(np.ceil(len(d) / block)))
        positions = ((starts[:, None] + offsets[None]) % len(d)).ravel()[:len(d)]
        boots[i] = d[positions].mean()
    effect = float(d.mean())
    ci = np.quantile(boots, [0.025, 0.975])
    # Centered null distribution, with a finite-resample correction.
    p = (1 + np.count_nonzero(np.abs(boots - effect) >= abs(effect))) / (repetitions + 1)
    return {"mean_difference": effect, "confidence_interval_95": ci.tolist(),
            "centered_two_sided_p": float(p), "block_length": block,
            "repetitions": repetitions, "random_seed": seed, "origins": len(d)}


def future_path(path: np.ndarray, initial: np.ndarray, horizon: int, label: str) -> np.ndarray:
    if path.shape[1] == horizon + 1:
        if not np.allclose(path[:, 0], initial, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{label}: h=0 is not the recorded initial state")
        return np.asarray(path[:, 1:], dtype=np.float64)
    if path.shape[1] == horizon:
        return np.asarray(path, dtype=np.float64)
    raise ValueError(f"{label}: path does not have H or H+1 leads")


def graph_iou(predicted: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    # Manuscript E.5: strict |A_ij| > 0.05; empty union -> zero.
    n = target.shape[-1]
    offdiag = ~np.eye(n, dtype=bool)
    p = np.abs(predicted[..., offdiag]) > threshold
    t = np.abs(target[..., offdiag]) > threshold
    intersection = np.logical_and(p, t).sum(-1)
    union = np.logical_or(p, t).sum(-1)
    return intersection / np.maximum(union, 1)


def train_mean(prepared_dir: Path, target: np.ndarray, prediction_indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ap = prepared_dir / "prepared_arrays.npz"
    mp = prepared_dir / "prepared_metadata.json"
    metadata = json.loads(mp.read_text(encoding="utf-8"))
    with np.load(ap, allow_pickle=False) as arrays:
        train_idx = np.asarray(metadata["split_indices"]["train"], dtype=int)
        test_idx = np.asarray(metadata["split_indices"]["test"], dtype=int)
        if not np.array_equal(test_idx, prediction_indices):
            raise ValueError("Prediction indices differ from prepared test split")
        y = arrays["y"]
        if not np.allclose(y[test_idx], target, rtol=1e-6, atol=1e-8):
            raise ValueError("Prediction targets differ from prepared test labels")
        origins = arrays["origins"][train_idx]
        labels = y[train_idx]
        # Deduplicate overlapping training label dates, preventing H-fold weighting.
        times = (origins[:, None] + np.arange(1, labels.shape[1] + 1)).ravel()
        flat = labels.reshape(-1, labels.shape[-1]).astype(np.float64)
        unique, first, inverse = np.unique(times, return_index=True, return_inverse=True)
        if not np.allclose(flat, flat[first][inverse], rtol=1e-6, atol=1e-8):
            raise ValueError("Training labels disagree at a common realized time")
        fitted = flat[first].mean(axis=0)
        meta = {"estimator": "assetwise arithmetic mean of unique realized label dates in purged training origins",
                "unique_training_dates": len(unique), "first_training_label_row": int(unique.min()),
                "last_training_label_row": int(unique.max()), "vector": fitted.tolist(),
                "prepared_arrays_sha256": sha256(ap), "prepared_metadata_sha256": sha256(mp)}
        return np.broadcast_to(fitted[None, None], target.shape), meta


def evaluate_run(entry: dict[str, Any], base: Path, threshold: float) -> dict[str, Any]:
    p = Path(entry["predictions"])
    if not p.is_absolute():
        p = base / p
    prepared = Path(entry["prepared_dir"]) if entry.get("prepared_dir") else None
    if prepared is not None and not prepared.is_absolute():
        prepared = base / prepared
    with np.load(p, allow_pickle=False) as f:
        target = np.asarray(f["y"], dtype=np.float64)
        mean = np.asarray(f["mu"], dtype=np.float64)
        indices = np.asarray(f["indices"], dtype=np.int64)
        if target.ndim != 3 or mean.shape != target.shape:
            raise ValueError("Expected mean and target shape [origin, lead, asset]")
        if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
            raise ValueError("Origins must be unique and chronological")
        if not np.isfinite(target).all() or not np.isfinite(mean).all():
            raise ValueError("Nonfinite predictions")
        horizon = target.shape[1]
        c0, a0 = np.asarray(f["c0"]), np.asarray(f["a0"])
        c = future_path(f["concepts"], c0, horizon, "predicted concepts")
        ct = future_path(f["c_target"], c0, horizon, "target concepts")
        a = future_path(f["graphs"], a0, horizon, "predicted graph")
        at = future_path(f["a_target"], a0, horizon, "target graph")
    cp = np.broadcast_to(c0[:, None], ct.shape)
    ap = np.broadcast_to(a0[:, None], at.shape)
    offdiag = ~np.eye(target.shape[-1], dtype=bool)
    states = {
        "concept_mae_learned": np.abs(c - ct).mean(axis=2),
        "concept_mae_persistence": np.abs(cp - ct).mean(axis=2),
        "graph_mae_learned": np.abs(a - at)[..., offdiag].mean(-1),
        "graph_mae_persistence": np.abs(ap - at)[..., offdiag].mean(-1),
        "graph_iou_learned": graph_iou(a, at, threshold),
        "graph_iou_persistence": graph_iou(ap, at, threshold),
    }
    if not all(np.isfinite(x).all() for x in states.values()):
        raise ValueError("Nonfinite state diagnostic")
    forecasts = {"TREFIN": mean, "ZeroReturn": np.zeros_like(target)}
    extra = {}
    if prepared is not None:
        forecasts["TrainMean"], extra["train_mean_fit"] = train_mean(prepared, target, indices)
        metadata = json.loads((prepared / "prepared_metadata.json").read_text(encoding="utf-8"))
        names = metadata.get("concept_names")
    else:
        names = None
    losses = {}
    point = {}
    for name, forecast in forecasts.items():
        diff = forecast - target
        mse = np.square(diff).mean(axis=(1, 2))
        mae = np.abs(diff).mean(axis=(1, 2))
        losses[name] = {"mse": mse, "mae": mae}
        point[name] = {"mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())), "mae": float(mae.mean())}
    return {"fold": str(entry["fold"]), "seed": int(entry["seed"]), "source": str(p.resolve()),
            "sha256": sha256(p), "indices": indices, "target": target, "states": states,
            "point": point, "losses": losses, "extra": extra,
            "concept_names": names or [f"concept_{i}" for i in range(ct.shape[-1])]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(manifest_path: Path, output: Path, blocks: list[int], repetitions: int, rng_seed: int, threshold: float = .05) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = [evaluate_run(e, manifest_path.resolve().parent, threshold) for e in manifest["runs"]]
    if not runs:
        raise ValueError("Manifest has no runs")
    output.mkdir(parents=True, exist_ok=True)
    point_rows, concept_rows, graph_rows, test_rows = [], [], [], []
    report = {"schema": "trefin-stored-evidence-v1", "manifest_sha256": sha256(manifest_path),
              "protocol": {"seed_aggregation": "average losses across seeds at each shared chronological origin before temporal bootstrap",
                           "bootstrap": "paired circular blocks within each fold; 95% percentile CI; centered-null p value",
                           "concept_error": "MAE in stored robust-standardized projected coordinates, averaged over assets; h=0 excluded",
                           "graph_support": f"strict abs(edge)>{threshold}; off diagonal; per-origin IoU; empty union assigned zero",
                           "point_reduction": "average squared/absolute error over five leads and assets at each origin, then average origins; RMSE computed separately for each seed",
                           "display_multipliers": {"mse": 1e4, "rmse": 1e2, "mae": 1e2},
                           "no_test_based_model_selection": True}, "runs": [], "folds": {}}
    saved = {}
    for run in runs:
        report["runs"].append({k:run[k] for k in ["fold", "seed", "source", "sha256", "extra"]})
    for fold in sorted({r["fold"] for r in runs}):
        group = sorted([r for r in runs if r["fold"] == fold], key=lambda x:x["seed"])
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError(f"Duplicate seed in {fold}")
        first = group[0]
        for r in group[1:]:
            if not np.array_equal(r["indices"], first["indices"]) or not np.array_equal(r["target"], first["target"]):
                raise ValueError(f"Seed alignment failed for {fold}")
            if r["concept_names"] != first["concept_names"] or set(r["point"]) != set(first["point"]):
                raise ValueError(f"Inconsistent diagnostics/baselines for {fold}")
        folded = {"seeds": [r["seed"] for r in group], "origins": len(first["indices"]), "point": {}, "paired_tests": {}, "concepts": [], "graphs": [], "concept_aggregate_paired_tests": []}
        for model in first["point"]:
            folded["point"][model] = {}
            for metric, scale in [("mse", 1e4), ("rmse", 1e2), ("mae", 1e2)]:
                stats = summary([r["point"][model][metric] for r in group])
                folded["point"][model][metric] = stats
                point_rows.append({"fold": fold, "model": model, "metric": metric, "mean_raw": stats["mean"],
                                   "seed_sample_std_raw": stats["seed_sample_std"], "display_multiplier": scale,
                                   "mean_display": stats["mean"]*scale,
                                   "seed_sample_std_display": stats["seed_sample_std"]*scale if stats["seed_sample_std"] is not None else None,
                                   "seeds": len(group), "origins": len(first["indices"])})
        for baseline in sorted(set(first["point"]) - {"TREFIN"}):
            for metric in ["mse", "mae"]:
                d = np.mean([r["losses"]["TREFIN"][metric] - r["losses"][baseline][metric] for r in group], axis=0)
                label = f"TREFIN_minus_{baseline}_{metric}"
                saved[f"{fold}_{label}"] = d
                folded["paired_tests"][label] = [paired_bootstrap(d,b,repetitions,rng_seed+i) for i,b in enumerate(blocks)]
                for test in folded["paired_tests"][label]:
                    test_rows.append({"fold":fold,"comparison":label, "mean_difference_raw":test["mean_difference"],
                                      "ci_lower_raw":test["confidence_interval_95"][0],"ci_upper_raw":test["confidence_interval_95"][1],
                                      "centered_two_sided_p":test["centered_two_sided_p"],"block_length":test["block_length"],
                                      "origins":test["origins"],"seeds_averaged":len(group),"repetitions":repetitions})
        state = {key: np.stack([r["states"][key] for r in group]) for key in first["states"]}
        horizon = first["target"].shape[1]
        for h in range(horizon):
            wins = 0
            for k, name in enumerate(first["concept_names"]):
                learned = summary(state["concept_mae_learned"][:, :, h, k].mean(axis=1))
                persistent = summary(state["concept_mae_persistence"][:, :, h, k].mean(axis=1))
                difference = learned["mean"] - persistent["mean"]
                wins += difference < 0
                skill = 1-learned["mean"]/persistent["mean"] if persistent["mean"] > 0 else None
                row = {"fold":fold,"lead":h+1,"coordinate":name,"learned_mae":learned["mean"],
                       "learned_seed_std":learned["seed_sample_std"],"persistence_mae":persistent["mean"],
                       "difference":difference,"relative_mae_improvement":skill,"learned_better":bool(difference<0)}
                concept_rows.append(row); folded["concepts"].append(row)
            for metric in ["mae", "iou"]:
                learned = summary(state[f"graph_{metric}_learned"][:, :, h].mean(axis=1))
                persistent = summary(state[f"graph_{metric}_persistence"][:, :, h].mean(axis=1))
                d = (state[f"graph_{metric}_learned"][:, :, h] - state[f"graph_{metric}_persistence"][:, :, h]).mean(axis=0)
                test = paired_bootstrap(d,blocks[0],repetitions,rng_seed+h)
                row = {"fold":fold,"lead":h+1,"metric":metric,"learned":learned["mean"],
                       "learned_seed_std":learned["seed_sample_std"],"persistence":persistent["mean"],
                       "difference_learned_minus_persistence":float(d.mean()),"ci_lower":test["confidence_interval_95"][0],
                       "ci_upper":test["confidence_interval_95"][1],"block_length":blocks[0]}
                graph_rows.append(row); folded["graphs"].append(row)
            saved[f"{fold}_concept_mae_difference_h{h+1}"] = (state["concept_mae_learned"][:, :, h] - state["concept_mae_persistence"][:, :, h]).mean(axis=0)
            aggregate_difference = saved[f"{fold}_concept_mae_difference_h{h+1}"].mean(axis=1)
            folded["concept_aggregate_paired_tests"].append({"lead":h+1,"tests":[paired_bootstrap(aggregate_difference,b,repetitions,rng_seed+h+i) for i,b in enumerate(blocks)]})
        folded["concept_coordinates_improved_per_lead"] = [sum(r["learned_better"] for r in folded["concepts"] if r["lead"]==h+1) for h in range(horizon)]
        folded["concept_mean_mae_per_lead"] = {method:state[f"concept_mae_{method}"].mean(axis=(0,1,3)).tolist() for method in ["learned","persistence"]}
        overall_learned = state["concept_mae_learned"].mean(axis=(0,1,2))
        overall_persistence = state["concept_mae_persistence"].mean(axis=(0,1,2))
        folded["concept_coordinates_improved_all_leads"] = int(np.sum(overall_learned < overall_persistence))
        folded["concept_improved_coordinate_names_all_leads"] = [name for k,name in enumerate(first["concept_names"]) if overall_learned[k] < overall_persistence[k]]
        folded["concept_overall_relative_mae_reduction"] = float(1-overall_learned.mean()/overall_persistence.mean())
        report["folds"][fold] = folded
        saved[f"{fold}_origin_indices"] = first["indices"]
    (output/"evidence.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    for name, rows in [("point_metrics",point_rows),("concept_persistence",concept_rows),("graph_persistence",graph_rows),("paired_tests",test_rows)]:
        write_csv(output/f"{name}.csv",rows)
    np.savez_compressed(output/"origin_level_differences.npz",**saved)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--block-lengths",type=int,nargs="+",default=[5,10,20])
    parser.add_argument("--repetitions",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=20261108)
    parser.add_argument("--graph-threshold",type=float,default=.05)
    args=parser.parse_args()
    if min(args.block_lengths)<1 or args.repetitions<100:
        parser.error("Positive block lengths and at least 100 replicates required")
    report=evaluate(args.manifest,args.output,args.block_lengths,args.repetitions,args.seed,args.graph_threshold)
    print(json.dumps({"output":str(args.output.resolve()),"runs":len(report["runs"]),"folds":list(report["folds"])}))


if __name__ == "__main__":
    main()
