#!/usr/bin/env python3
"""Analyze the registered 30-evaluation Full/FrozenGraph study without training.

python analyze_frozen_graph.py --study-root /path/to/study --output-dir /path/to/analysis
python analyze_frozen_graph.py --self-test

Requires numpy, scipy, PyYAML and matplotlib. All 30 runs and all registered
artifacts are required. Main tables retain the original evaluator's metrics;
paired losses and state metrics are independently recomputed in float64.
Default summaries are descriptive. --block-intervals adds percentile temporal
intervals (blocks 20, 5, 10) without p-values or significance annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist

FOLDS = (1, 2, 3)
SEEDS = (1, 7, 19, 37, 73)
VARIANTS = ("Full", "FrozenGraph")
EXPECTED_ORIGINS = {1: 500, 2: 496, 3: 306}
RETURN_METRICS = ("mse", "rmse", "mae", "energy_score", "msic", "msir",
                  "coverage90", "width90")
STATE_METRICS = ("concept_mae", "graph_mae", "graph_rmse", "graph_iou")
SCALES = {"mse": 1e4, "rmse": 1e2, "mae": 1e2, "energy_score": 1e2,
          "msic": 1e2, "msir": 1e2, "coverage90": 100, "width90": 100,
          **{k: 1 for k in STATE_METRICS}}
IDENTITY_KEYS = ("indices", "y", "c0", "a0", "s0", "c_target", "a_target")


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(rows: list[dict], path: Path) -> None:
    require(bool(rows), f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stat(values: Any) -> dict:
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), "Invalid statistics")
    return {"mean": float(x.mean()), "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "n": len(x), "values": x.tolist()}


def energy_by_lead(samples: np.ndarray, target: np.ndarray,
                   translation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Cross-sectional ES[S,B,H,N], unbiased pair term; never flatten H*N."""
    require(samples.ndim == 4 and samples.shape[1:] == target.shape, "ES shape mismatch")
    require(len(samples) >= 2, "ES needs at least two scenarios")
    if translation is not None:
        require(translation.shape == target.shape, "Translation shape mismatch")
    count = len(samples)
    es = np.empty(target.shape[:2], dtype=np.float64)
    moved_es = np.empty_like(es) if translation is not None else None
    for b in range(len(target)):
        for h in range(target.shape[1]):
            x = samples[:, b, h].astype(np.float64)
            pair = pdist(x, metric="euclidean").sum() / (count * (count - 1))
            es[b, h] = np.linalg.norm(x - target[b, h], axis=-1).mean() - pair
            if moved_es is not None:
                moved_es[b, h] = np.linalg.norm(
                    x + translation[b, h] - target[b, h], axis=-1).mean() - pair
    return es, moved_es


def information_metrics(y: np.ndarray, mu: np.ndarray) -> dict:
    yc = y - y.mean(axis=1, keepdims=True)
    mc = mu - mu.mean(axis=1, keepdims=True)
    covariance = (yc * mc).mean(axis=1)
    denominator = y.std(axis=1) * mu.std(axis=1)
    valid = denominator > 1e-12
    corr = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=valid)
    ic = corr.mean(axis=0)
    ir = ic / np.maximum(corr.std(axis=0), 1e-12)
    return {"msic": float(ic.mean()), "msir": float(ir.mean()),
            "valid_origin_asset_fraction": float(valid.mean())}


def graph_losses(a: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    off = ~np.eye(a.shape[-1], dtype=bool)
    pred_support = np.abs(a[..., off]) > .05
    true_support = np.abs(target[..., off]) > .05
    union = (pred_support | true_support).sum(axis=-1)
    intersection = (pred_support & true_support).sum(axis=-1)
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                    where=union > 0)
    errors = a[..., off] - target[..., off]
    return {"graph_iou": iou, "graph_mae": np.abs(errors).mean(axis=-1),
            "graph_mse": np.square(errors).mean(axis=-1)}


def state_analysis(p: dict[str, np.ndarray], variant: str) -> tuple[dict, dict, dict]:
    c, ct = p["concepts"][:, 1:].astype(float), p["c_target"][:, 1:].astype(float)
    a, at = p["graphs"][:, 1:].astype(float), p["a_target"][:, 1:].astype(float)
    require(c.shape == ct.shape and c.shape[1:] == (5, 32, 26), "Concept path shape")
    require(a.shape == at.shape and a.shape[1:] == (5, 32, 32), "Graph path shape")
    ce = np.abs(c - ct)
    pe = np.abs(p["c0"][:, None].astype(float) - ct)
    g = graph_losses(a, at)
    gp = graph_losses(np.broadcast_to(p["a0"][:, None].astype(float), at.shape), at)
    departure = float(np.max(np.abs(p["graphs"] - p["a0"][:, None])))
    audit = {"graph_max_abs_departure_from_initial": departure,
             "graph_symmetry_max_abs": float(np.max(np.abs(p["graphs"] - p["graphs"].swapaxes(-1, -2)))),
             "graph_diagonal_max_abs": float(np.max(np.abs(np.diagonal(p["graphs"], axis1=-2, axis2=-1)))),
             "initial_concept_max_abs_departure": float(np.max(np.abs(p["concepts"][:, 0] - p["c0"])))}
    require(audit["graph_symmetry_max_abs"] <= 1e-7 and audit["graph_diagonal_max_abs"] == 0,
            "Observed graph domain invariant failed")
    require(audit["initial_concept_max_abs_departure"] <= 1e-7, "Concept anchor mismatch")
    if variant == "FrozenGraph":
        require(departure == 0, "FrozenGraph departs from A0")
        require(all(np.array_equal(g[k], gp[k]) for k in g), "Frozen graph diagnostics != persistence")
        audit["frozen_graph_equals_persistence_diagnostics"] = True
    states = {
        "concept_mae_lead_coordinate": ce.mean(axis=(0, 2)),
        "concept_persistence_mae_lead_coordinate": pe.mean(axis=(0, 2)),
        "concept_mae_lead": ce.mean(axis=(0, 2, 3)),
        "concept_persistence_mae_lead": pe.mean(axis=(0, 2, 3)),
        "graph_iou_lead": g["graph_iou"].mean(axis=0),
        "graph_mae_lead": g["graph_mae"].mean(axis=0),
        "graph_rmse_lead": np.sqrt(g["graph_mse"].mean(axis=0)),
        "graph_persistence_iou_lead": gp["graph_iou"].mean(axis=0),
        "graph_persistence_mae_lead": gp["graph_mae"].mean(axis=0),
        "graph_persistence_rmse_lead": np.sqrt(gp["graph_mse"].mean(axis=0))}
    overall = {"concept_mae": float(ce.mean()), "graph_mae": float(g["graph_mae"].mean()),
               "graph_rmse": float(np.sqrt(g["graph_mse"].mean())), "graph_iou": float(g["graph_iou"].mean())}
    origin_losses = {"concept_mae": ce.mean(axis=(1, 2, 3)),
                     "graph_mae": g["graph_mae"].mean(axis=1),
                     "graph_iou": g["graph_iou"].mean(axis=1)}
    persistence = {"concept_mae": float(pe.mean()),
                   "graph_iou": float(gp["graph_iou"].mean()),
                   "graph_mae": float(gp["graph_mae"].mean()),
                   "graph_rmse": float(np.sqrt(gp["graph_mse"].mean()))}
    return {"overall": overall, "persistence": persistence, "arrays": states,
            "origin": origin_losses}, audit, gp


def close(actual: Any, expected: Any, description: str,
          rtol: float = 2e-6, atol: float = 1e-7) -> float:
    x, y = np.asarray(actual), np.asarray(expected)
    require(x.shape == y.shape and np.allclose(x, y, rtol=rtol, atol=atol), description)
    return float(np.max(np.abs(x - y)))


def metric_values(record: dict) -> dict:
    return {**{k: float(record[k]) for k in RETURN_METRICS[:6]},
            "coverage90": float(record["coverage"]["0.9"]),
            "width90": float(record["interval_width"]["0.9"])}


def verify_freeze(root: Path) -> tuple[dict, str]:
    path = root / "frozen_selections.json"
    frozen = read_json(path)
    digest = sha256(path)
    require(frozen["schema"] == "trefin-frozen-graph-v7-selections", "Freeze schema")
    require(frozen["status"] == "frozen_before_test" and frozen["test_used_for_selection"] is False,
            "Selection was not frozen before test")
    require(set(frozen["selections"]) == {f"fold_{f}/{v}" for f in FOLDS for v in VARIANTS},
            "Need exactly six frozen choices")
    for key, entry in frozen["selections"].items():
        for prefix in ("selection", "registration"):
            require(sha256(root / entry[prefix + "_path"]) == entry[prefix + "_sha256"],
                    f"Changed frozen {prefix}: {key}")
        selected = read_json(root / entry["selection_path"])
        require(selected["test_access_count"] == 0, "Selection used test data")
        best = min(selected["trials"], key=lambda t: (t["calibrated_validation_energy_score"], t["order"]))
        require(best["candidate"] == selected["selected_candidate"] == entry["selected_candidate"],
                "Frozen candidate is not validation choice")
        require(sha256(root / selected["selected_checkpoint"]) == selected["selected_checkpoint_sha256"],
                "Selected checkpoint changed")
    return frozen, digest


def load_run(root: Path, fold: int, seed: int, variant: str, frozen: dict, frozen_sha: str) -> dict:
    import yaml
    directory = root / "runs" / f"fold_{fold}_seed_{seed}" / variant
    complete_path = directory / "complete.json"
    complete = read_json(complete_path)
    require(complete["schema"] == "trefin-frozen-graph-v7-complete", "Run schema")
    require((complete["fold"], complete["seed"], complete["variant"]) == (fold, seed, variant),
            "Run identity")
    require(complete["global_selection_sha256"] == frozen_sha, "Run selection changed")
    entry = frozen["selections"][f"fold_{fold}/{variant}"]
    require(complete["selected_candidate"] == entry["selected_candidate"], "Candidate changed across seeds")
    for relative, expected in complete["artifact_sha256"].items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()), "Artifact outside run")
        require(sha256(path) == expected, f"Artifact hash mismatch: {path}")
    provenance = read_json(directory / "audit/provenance.json")
    require(provenance["test_used_for_selection"] is False and provenance["all_six_selections_frozen"],
            "Run provenance selection failure")
    require(provenance["global_selection_sha256"] == frozen_sha and
            provenance["source_sha256"] == frozen["source_sha256"], "Source identity changed")
    guard = read_json(directory / "audit/train_validation_access_guard.json")
    require(all(guard[k] for k in ("test_split_key_removed", "test_dataset_construction_raises_KeyError",
                                   "origin_arrays_physically_subset")), "Train/validation access guard")
    require(read_json(directory / "audit/checkpoint_reload_check.json")["passed"], "Reload failed")
    test_start = read_json(directory / "audit/test_evaluation_started.json")
    require(test_start["test_scenarios"] == 256 and test_start["global_selection_sha256"] == frozen_sha,
            "Test registration differs")
    require(test_start["checkpoint_sha256"] == sha256(directory / "checkpoints/best.pt"), "Test checkpoint")
    cfg = yaml.safe_load((directory / "config_resolved.yaml").read_text(encoding="utf-8"))
    require(cfg["model"]["ablation_variant"] == variant, "Resolved variant")
    require(cfg["experiment"]["seed"] == seed, "Resolved training seed")
    require(cfg["training"]["lambda_gain"] == .1 and cfg["training"]["gain_target"] == .995,
            "Gain protocol differs")
    for name, value in (("epochs", 20), ("patience", 6), ("batch_size", 64),
                        ("train_samples", 16), ("validation_samples", 64), ("test_samples", 256)):
        require(cfg["training"][name] == value, f"Changed fitting protocol: {name}")
    if variant == "FrozenGraph":
        require(cfg["training"]["lambda_graph"] == 0, "Frozen graph loss not zero")
    rho = float(complete["rho"])
    require(.05 <= rho <= 1 and cfg["model"]["signal_shrinkage"] == rho, "Rho mismatch")
    effective = {"model": dict(cfg["model"]), "training": dict(cfg["training"])}
    effective["model"]["signal_shrinkage"] = 1.0
    if variant == "FrozenGraph":
        effective["model"].pop("graph_refresh", None)
    fingerprint = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    require(fingerprint == provenance["effective_fingerprint"], "Effective selected configuration differs")
    with np.load(directory / "predictions/test_predictions.npz", allow_pickle=False) as archive:
        p = {k: archive[k] for k in archive.files}
    require(all(np.isfinite(a).all() for a in p.values()), "Nonfinite prediction")
    y, mu = p["y"].astype(float), p["mu"].astype(float)
    require(y.shape == mu.shape == (EXPECTED_ORIGINS[fold], 5, 32), "Target dimensions")
    require(p["samples"].shape == (256, *y.shape), "Scenario dimensions")
    require(np.all(np.diff(p["indices"]) > 0), "Origins must be unique and chronological")
    with np.load(directory / "tables/origin_losses.npz", allow_pickle=False) as archive:
        stored = {k: archive[k] for k in archive.files}
    for k in ("indices", "mu", "y"):
        require(np.array_equal(stored[k], p[k]), f"Stored origin {k} mismatch")
    raw_mu = mu / rho
    es_lead, raw_es_lead = energy_by_lead(p["samples"], y, raw_mu - mu)
    losses = {"mse": np.square(mu - y).mean(axis=(1, 2)),
              "mae": np.abs(mu - y).mean(axis=(1, 2)), "energy_score": es_lead.mean(axis=1)}
    checks = {k: close(losses[k], stored[v], f"Stored origin {v}", rtol=1e-12, atol=1e-12)
              for k, v in (("mse", "mse"), ("mae", "mae"), ("energy_score", "es"))}
    checks["es_lead"] = close(es_lead, stored["es_lead"], "Stored lead ES", rtol=1e-12, atol=1e-12)
    values = {cal: metric_values(complete[cal]) for cal in ("calibrated", "uncalibrated")}
    for cal, m, es in (("calibrated", mu, es_lead), ("uncalibrated", raw_mu, raw_es_lead)):
        mse = float(np.square(m - y).mean())
        recomputed = {"mse": mse, "rmse": float(np.sqrt(mse)),
                      "mae": float(np.abs(m - y).mean()), "energy_score": float(es.mean()),
                      **information_metrics(y, m)}
        for metric in RETURN_METRICS[:6]:
            checks[f"{cal}_{metric}"] = close(recomputed[metric], values[cal][metric],
                                               f"Registered {cal} {metric}")
    q = np.quantile(p["samples"], [.05, .95], axis=0)
    checks["coverage90"] = close(np.mean((y >= q[0]) & (y <= q[1])),
                                 values["calibrated"]["coverage90"], "Coverage")
    checks["width90"] = close(np.mean(q[1] - q[0]), values["calibrated"]["width90"], "Width")
    # Reproduce the original evaluator's float32 translation before checking its
    # empirical quantiles; translating float64 quantiles could move a boundary.
    raw_samples_registered = p["samples"] + (p["mu"] / rho - p["mu"])[None]
    raw_q = np.quantile(raw_samples_registered, [.05, .95], axis=0)
    checks["uncalibrated_coverage90"] = close(
        np.mean((y >= raw_q[0]) & (y <= raw_q[1])),
        values["uncalibrated"]["coverage90"], "Uncalibrated coverage", rtol=0, atol=1e-12)
    checks["uncalibrated_width90"] = close(
        np.mean(raw_q[1] - raw_q[0]), values["uncalibrated"]["width90"], "Uncalibrated width")
    del raw_samples_registered
    checks["translation_width90"] = close(values["calibrated"]["width90"],
                                           values["uncalibrated"]["width90"], "Translation width")
    for metric in ("msic", "msir"):
        checks["translation_" + metric] = close(values["calibrated"][metric],
                                                 values["uncalibrated"][metric], "Positive-rho correlation")
    state, audit, _ = state_analysis(p, variant)
    states_record = read_json(directory / "tables/states.json")
    for k in ("concept_mae_lead_coordinate", "concept_persistence_mae_lead_coordinate",
              "graph_iou_lead", "graph_mae_lead", "graph_rmse_lead"):
        checks[k] = close(state["arrays"][k], states_record[k], f"State metric {k}",
                          rtol=1e-12, atol=1e-12)
    require(states_record["graph_threshold"] == .05, "Graph threshold")
    require(audit["graph_max_abs_departure_from_initial"] ==
            states_record["graph_max_abs_departure_from_initial"] ==
            complete["graph_max_abs_departure_from_initial"], "Graph departure record")
    for k in ("sigma_plus", "sigma_minus"):
        audit["minimum_" + k] = float(p[k].min())
        require(audit["minimum_" + k] > 0, f"Nonpositive {k}")
    audit["minimum_nu"] = float(p["nu"].min())
    audit["minimum_precision_eigenvalue"] = float(np.linalg.eigvalsh(p["precision"].astype(float)).min())
    require(audit["minimum_nu"] > 2 and audit["minimum_precision_eigenvalue"] > 0,
            "Observed return-law validity failure")
    losses.update(state["origin"])
    identity = {k: p[k].copy() for k in IDENTITY_KEYS}
    identity["origins"] = stored["origins"].copy()
    return {"fold": fold, "seed": seed, "variant": variant, "rho": rho,
            "selected_candidate": complete["selected_candidate"], "metrics": values,
            "selected_settings": {
                "residual_scale": cfg["model"]["residual_scale"],
                "graph_refresh": cfg["model"]["graph_refresh"],
                "lambda_concept": cfg["training"]["lambda_concept"],
                "lambda_graph": cfg["training"]["lambda_graph"],
                "lambda_gain": cfg["training"]["lambda_gain"],
                "gain_target": cfg["training"]["gain_target"]},
            "state": state, "origin_losses": losses, "identity": identity,
            "training": complete["training"], "audit": audit, "checks": checks,
            "provenance": {"complete_sha256": sha256(complete_path),
                           "prediction_sha256": complete["artifact_sha256"]["predictions/test_predictions.npz"],
                           "artifacts_verified": len(complete["artifact_sha256"]),
                           "identity_hashes": {k: array_hash(a) for k, a in identity.items()}}}


def block_interval(d: np.ndarray, block: int, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(d)
    values = np.empty(reps)
    for start in range(0, reps, 200):
        end = min(start + 200, reps)
        starts = rng.integers(n, size=(end - start, int(np.ceil(n / block))))
        indices = ((starts[..., None] + np.arange(block)) % n).reshape(end - start, -1)[:, :n]
        values[start:end] = d[indices].mean(axis=1)
    return {"block_length": block, "replicates": reps, "random_seed": seed,
            "percentile95": np.quantile(values, [.025, .975]).tolist()}


def make_figure(paired_seed_rows: list[dict], output: Path, *, synthetic: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    specs = [("mse", "MSE difference (×10⁴)", 1e4),
             ("energy_score", "ES difference (×10²)", 1e2),
             ("concept_mae", "Concept MAE difference", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.55))
    plot_rows = []
    for ax, (metric, label, scale) in zip(axes, specs):
        ax.axhline(0, color=".55", lw=.7, zorder=0)
        for fold in FOLDS:
            selected = [r for r in paired_seed_rows if r["fold"] == fold and r["metric"] == metric
                        and r["calibration"] == "calibrated"]
            selected.sort(key=lambda r: SEEDS.index(r["seed"]))
            require(len(selected) == 5, "Figure requires all five seed pairs")
            y = np.array([r["difference_frozen_minus_full"] * scale for r in selected])
            ax.scatter(fold + np.linspace(-.13, .13, 5), y, s=12, c=".55", alpha=.8,
                       edgecolors="none", zorder=2)
            ax.errorbar(fold, y.mean(), yerr=y.std(ddof=1), fmt="D", markersize=3.7,
                        capsize=3, color="#0072B2", lw=1, zorder=3)
            for row, value in zip(selected, y):
                plot_rows.append({"fold": fold, "metric": metric, "seed": row["seed"],
                                  "display_scale": scale, "difference_display": float(value),
                                  "paired_mean_display": float(y.mean()),
                                  "paired_sample_sd_display": float(y.std(ddof=1))})
        ax.set_xticks(FOLDS, ["1", "2", "3"])
        ax.set_xlabel("Fold", labelpad=2)
        ax.set_xlim(.6, 3.4)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.margins(y=.18)
    fig.legend(handles=[Line2D([], [], marker="o", color=".55", linestyle="none",
                               markersize=4, label="Matched seed"),
                        Line2D([], [], marker="D", color="#0072B2", linestyle="none",
                               markersize=4, label="Mean ± SD")],
               ncol=2, loc="lower center", frameon=False, bbox_to_anchor=(.5, .0))
    if synthetic:
        fig.suptitle("SYNTHETIC SELF-TEST — not study results", fontsize=9)
    fig.subplots_adjust(left=.14, right=.985, top=.94 if not synthetic else .85,
                        bottom=.28, wspace=.80)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / ("paired_graph_updates." + suffix), dpi=250, facecolor="white")
    plt.close(fig)
    save_csv(plot_rows, output / "paired_graph_updates_plot_data.csv")
    (output / "paired_graph_updates_caption.txt").write_text(
        "Paired Full/FrozenGraph differences across three chronological periods. "
        "Each gray point is one matched training-seed difference (FrozenGraph minus Full); "
        "blue diamonds and error bars show the mean and sample SD across five paired seeds. "
        "Positive values indicate lower loss for Full. MSE and ES use the indicated display multipliers; "
        "concept MAE uses robust-standardized coordinates and averages future leads 1–5.\n",
        encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    root = args.study_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "analysis_independent"
    frozen, frozen_sha = verify_freeze(root)
    required = [(f, s, v) for f in FOLDS for s in SEEDS for v in VARIANTS]
    missing = [str(root / "runs" / f"fold_{f}_seed_{s}" / v / "complete.json")
               for f, s, v in required if not (root / "runs" / f"fold_{f}_seed_{s}" / v / "complete.json").is_file()]
    require(not missing, "Incomplete study; no report emitted. Missing: " + json.dumps(missing))
    runs, identities = {}, {}
    for f, s, v in required:
        run = load_run(root, f, s, v, frozen, frozen_sha)
        if f not in identities:
            identities[f] = run["identity"]
        else:
            for key, values in identities[f].items():
                require(np.array_equal(values, run["identity"][key]),
                        f"Exact paired identity differs: fold{f} seed{s} {v} {key}")
        run.pop("identity")
        runs[(f, s, v)] = run
        print(f"Verified fold={f} seed={s} variant={v}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "trefin-frozen-graph-independent-analysis-v1", "status": "complete",
              "source_root": str(root), "expected_runs": 30, "verified_runs": 30,
              "folds": {}, "definitions": {
                  "table": "Original evaluator metrics; mean and sample SD across five seeds. RMSE is computed per run before averaging.",
                  "energy_score": "Float64 per-origin verification of cross-sectional N-dimensional ES, averaged over H=5 leads; pdist sum divided by S(S-1), S=256.",
                  "paired_direction": "FrozenGraph minus Full. Positive losses favor Full; negative graph-IoU difference favors Full.",
                  "paired_unit": "Match training seeds and exact within-fold origins/targets; seed-average origin differences keep all assets/leads grouped.",
                  "selection": "Independent fold/variant selection on seed 1 validation, frozen globally before final test; equal nominal four slots, three unique FrozenGraph candidates.",
                  "study_identity": "Selected Full models belong to this separate control study; original benchmark/theory checkpoints are unchanged.",
                  "state": "Future leads 1..5 only; graph support strict abs(A)>0.05 off diagonal, empty union zero.",
                  "inference": "Descriptive by default; optional temporal percentile intervals, no p-values or significance labels."},
              "provenance": {"script_sha256": sha256(Path(__file__)), "freeze_sha256": frozen_sha,
                             "source_sha256": frozen["source_sha256"], "runs": []}}
    metric_rows, run_rows, main_rows, paired_seed, paired_origin, lead_rows, coordinate_rows = [], [], [], [], [], [], []
    pair_arrays = {}
    for fold in FOLDS:
        fr = {"methods": {}, "paired": {}, "identity": {k: array_hash(a) for k, a in identities[fold].items()}}
        for variant in VARIANTS:
            group = [runs[(fold, seed, variant)] for seed in SEEDS]
            method = {"selected_candidate": group[0]["selected_candidate"],
                      "rho": stat([r["rho"] for r in group]), "calibrated": {}, "uncalibrated": {},
                      "state": {}, "persistence": {}, "lead_metrics": {}}
            for cal in ("calibrated", "uncalibrated"):
                for metric in RETURN_METRICS:
                    values = [r["metrics"][cal][metric] for r in group]
                    summary = stat(values)
                    method[cal][metric] = summary
                    metric_rows.append({"fold": fold, "variant": variant, "calibration": cal,
                                        "metric": metric, "mean_raw": summary["mean"],
                                        "sample_sd_raw": summary["sample_sd"], "display_scale": SCALES[metric],
                                        "mean_display": summary["mean"] * SCALES[metric],
                                        "sample_sd_display": summary["sample_sd"] * SCALES[metric]})
            for metric in STATE_METRICS:
                summary = stat([r["state"]["overall"][metric] for r in group])
                method["state"][metric] = summary
                method["persistence"][metric] = stat([r["state"]["persistence"][metric] for r in group])
                metric_rows.append({"fold": fold, "variant": variant, "calibration": "state",
                                    "metric": metric, "mean_raw": summary["mean"], "sample_sd_raw": summary["sample_sd"],
                                    "display_scale": 1, "mean_display": summary["mean"], "sample_sd_display": summary["sample_sd"]})
            for r in group:
                for cal in ("calibrated", "uncalibrated"):
                    run_rows.append({"fold": fold, "seed": r["seed"], "variant": variant,
                                     "candidate": r["selected_candidate"], "rho": r["rho"], "calibration": cal,
                                     "selected_epoch": r["training"]["best_epoch"],
                                     "completed_epochs": r["training"]["epochs_completed"],
                                     "training_seconds": r["training"]["train_seconds"],
                                     **r["selected_settings"],
                                     **r["metrics"][cal], **r["state"]["overall"]})
                report["provenance"]["runs"].append({k: r[k] for k in
                    ("fold", "seed", "variant", "selected_candidate", "rho", "selected_settings",
                     "training", "provenance", "audit", "checks")})
            for name in group[0]["state"]["arrays"]:
                array = np.stack([r["state"]["arrays"][name] for r in group])
                if array.ndim == 2:
                    method["lead_metrics"][name] = {"seed_mean": array.mean(0).tolist(),
                                                   "seed_sample_sd": array.std(0, ddof=1).tolist()}
                    for lead in range(5):
                        lead_rows.append({"fold": fold, "variant": variant, "lead": lead + 1, "metric": name,
                                          "mean": float(array[:, lead].mean()),
                                          "sample_sd": float(array[:, lead].std(ddof=1))})
                else:
                    for lead in range(5):
                        for coordinate in range(26):
                            coordinate_rows.append({"fold": fold, "variant": variant, "lead": lead + 1,
                                                    "coordinate_index": coordinate, "metric": name,
                                                    "mean": float(array[:, lead, coordinate].mean()),
                                                    "sample_sd": float(array[:, lead, coordinate].std(ddof=1))})
            c = np.stack([r["state"]["arrays"]["concept_mae_lead_coordinate"] for r in group])
            cp = np.stack([r["state"]["arrays"]["concept_persistence_mae_lead_coordinate"] for r in group])
            method["concept_coordinates_better_than_persistence_per_lead"] = (c.mean(0) < cp.mean(0)).sum(1).tolist()
            method["concept_coordinates_better_than_persistence_overall"] = int((c.mean((0, 1)) < cp.mean((0, 1))).sum())
            main = {"fold": fold, "variant": variant, "seeds": 5}
            for metric in ("mse", "energy_score", "concept_mae"):
                st = method["state" if metric == "concept_mae" else "calibrated"][metric]
                main[metric + "_mean_display"] = st["mean"] * SCALES[metric]
                main[metric + "_sample_sd_display"] = st["sample_sd"] * SCALES[metric]
            main_rows.append(main)
            fr["methods"][variant] = method
        for cal, metrics in (("calibrated", RETURN_METRICS), ("uncalibrated", RETURN_METRICS),
                             ("state", STATE_METRICS)):
            for metric in metrics:
                differences = []
                for seed in SEEDS:
                    pair = [runs[(fold, seed, v)] for v in VARIANTS]
                    a, b = [r["state"]["overall"][metric] if cal == "state" else r["metrics"][cal][metric] for r in pair]
                    differences.append(b - a)
                    paired_seed.append({"fold": fold, "seed": seed, "calibration": cal if cal != "state" else "calibrated",
                                        "metric": metric, "full": a, "frozen_graph": b,
                                        "difference_frozen_minus_full": b - a, "display_scale": SCALES[metric]})
                fr["paired"][cal + "_" + metric] = {"seed_difference": stat(differences)}
        indices, origins = identities[fold]["indices"], identities[fold]["origins"]
        pair_arrays[f"fold{fold}_indices"] = indices
        pair_arrays[f"fold{fold}_origins"] = origins
        for metric in ("mse", "mae", "energy_score", "concept_mae", "graph_mae", "graph_iou"):
            difference = np.stack([runs[(fold, s, "FrozenGraph")]["origin_losses"][metric] -
                                   runs[(fold, s, "Full")]["origin_losses"][metric] for s in SEEDS])
            mean = difference.mean(axis=0)
            pair_arrays[f"fold{fold}_{metric}_seed_differences"] = difference
            pair_arrays[f"fold{fold}_{metric}_seed_ids"] = np.array(SEEDS)
            paired = {"mean": float(mean.mean()), "quantile_levels": [.05, .25, .5, .75, .95],
                      "quantiles": np.quantile(mean, [.05, .25, .5, .75, .95]).tolist(),
                      "positive_origin_fraction": float(np.mean(mean > 0)),
                      "zero_origin_fraction": float(np.mean(mean == 0)), "origins": len(mean)}
            if args.block_intervals:
                paired["temporal_intervals"] = [block_interval(mean, b, args.replicates, 712001 + fold * 100 + b)
                                                for b in (20, 5, 10)]
            fr["paired"]["origin_" + metric] = paired
            for j in range(len(indices)):
                paired_origin.append({"fold": fold, "index": int(indices[j]), "origin": str(origins[j]),
                                      "metric": metric, "seed_mean_difference": float(mean[j]),
                                      **{f"seed_{s}_difference": float(difference[i, j]) for i, s in enumerate(SEEDS)}})
        report["folds"][str(fold)] = fr
    for name, rows in (("main_six_rows", main_rows), ("all_metrics", metric_rows), ("per_run_metrics", run_rows),
                       ("paired_seed_differences", paired_seed), ("paired_origin_differences", paired_origin),
                       ("state_leadwise", lead_rows), ("state_coordinate_leadwise", coordinate_rows)):
        save_csv(rows, output / (name + ".csv"))
    np.savez_compressed(output / "paired_origin_arrays.npz", **pair_arrays)
    make_figure(paired_seed, output)
    save_json(report, output / "results.json")
    lines = ["# Full/FrozenGraph selected comparison", "",
             "Entries are five-seed mean ± sample SD. MSE ×10⁴; ES ×10²; concept MAE in standardized units.",
             "", "| Fold | Variant | MSE | ES | Concept MAE |", "|---:|---|---:|---:|---:|"]
    for row in main_rows:
        cells = [f"{row[m + '_mean_display']:.4f} ± {row[m + '_sample_sd_display']:.4f}"
                 for m in ("mse", "energy_score", "concept_mae")]
        lines.append(f"| {row['fold']} | {row['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "All 30 final evaluations passed exact within-fold input/target alignment and artifact checks.",
              "FrozenGraph retained A0 at every lead and exactly reproduced its graph-persistence diagnostics.",
              "Paired differences and figure error bars describe matched training seeds. No result direction is assumed.",
              "This selected control study is separate from the original forecasting and theory-checkpoint cohorts.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    save_json({"status": "complete", "script_sha256": sha256(Path(__file__)), "global_selection_sha256": frozen_sha,
               "files": hashes, "figure_derivation": "Only paired seed values from verified runs; no smoothing, fitting or omitted seeds."},
              output / "manifest.json")
    print(json.dumps({"status": "complete", "runs": 30, "output": str(output)}), flush=True)


def self_test(output: Path | None) -> None:
    x = np.array([0., 1., 3.])[:, None, None, None]
    y = np.full((1, 1, 1), 2.)
    es, moved = energy_by_lead(x, y, np.full_like(y, .5))
    require(np.allclose(es, 1/3), "Manual three-sample U-statistic")
    manual = np.abs(np.array([.5, 1.5, 3.5]) - 2).mean() - 1.
    require(np.allclose(moved, manual), "ES translation preserves pair term")
    target = np.array([[[0.], [1.], [2.]], [[2.], [1.], [0.]]])
    require(information_metrics(target, np.zeros_like(target))["msic"] == 0, "Constant-path IC")
    require(np.isclose(information_metrics(target, target)["msic"], 1), "Perfect path IC")
    a = np.zeros((2, 5, 3, 3)); t = a.copy()
    require((graph_losses(a, t)["graph_iou"] == 0).all(), "Empty support IoU")
    a[0, :, 0, 1] = a[0, :, 1, 0] = .05
    require((graph_losses(a, t)["graph_iou"] == 0).all(), "Strict threshold")
    require(stat([1, 2, 3])["sample_sd"] == 1, "Sample SD")
    require(np.allclose(block_interval(np.zeros(25), 20, 100, 1)["percentile95"], [0, 0]),
            "Zero paired block interval")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        rows = [{"fold": f, "seed": s, "calibration": "calibrated", "metric": m,
                 "difference_frozen_minus_full": ((f - 2) * .3 + (i - 2) * .1) / scale}
                for f in FOLDS for i, s in enumerate(SEEDS)
                for m, scale in (("mse", 1e4), ("energy_score", 1e2), ("concept_mae", 1))]
        make_figure(rows, output, synthetic=True)
    print(json.dumps({"status": "passed", "scope": "Synthetic mathematical/unit tests only; not study evidence.",
                      "checks": ["exact ES U-statistic", "translation", "constant/Pearson IC",
                                 "empty/strict-threshold IoU", "sample SD", "block interval",
                                 "optional synthetic plot"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--block-intervals", action="store_true")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(args.self_test_output)
    else:
        if args.study_root is None:
            parser.error("--study-root is required")
        analyze(args)
