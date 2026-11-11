#!/usr/bin/env python3
"""Registered four-slot validation search for matched TREFIN variants.

This script never evaluates a test split. Its in-memory prepared object contains
only train/validation origin rows and rejects test dataset construction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from fconcord.data import FinancialTensorDataset
from fconcord.experiment import _model_factory, load_prepared
from fconcord.trainer import checkpoint_reload_check, train_model
from fconcord.utils import environment_snapshot, seed_everything, setup_logger


VARIANTS = ["Full", "NoGraph", "SingleSigned", "SingleUnsigned", "NoStateSupervision"]
CANDIDATES = [
    {"id": "base", "model": {}, "training": {}},
    {"id": "slow_graph", "model": {"graph_refresh": 0.02}, "training": {}},
    {"id": "stronger_dynamics", "model": {"residual_scale": 0.05, "graph_refresh": 0.10}, "training": {}},
    {"id": "weaker_state_supervision", "model": {}, "training": {"lambda_concept": 0.10, "lambda_graph": 0.02}},
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(value, path):
    """Atomic file replacement avoids a partially written selection record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def train_validation_only(prepared):
    metadata = copy.deepcopy(prepared["metadata"])
    train = np.asarray(metadata["split_indices"]["train"], dtype=np.int64)
    val = np.asarray(metadata["split_indices"]["val"], dtype=np.int64)
    test = np.asarray(metadata["split_indices"].get("test", []), dtype=np.int64)
    keep = np.sort(np.concatenate([train, val]))
    if len(np.unique(keep)) != len(keep) or np.intersect1d(keep, test).size:
        raise AssertionError("Train/validation rows overlap each other or test")
    arrays = prepared["arrays"]
    sample_arrays = {"c0", "a0", "s0", "c_target", "a_target", "y", "origins"}
    static_arrays = {"directional_mask", "risk_indices"}
    restricted = {key: arrays[key][keep].copy() for key in sample_arrays}
    restricted.update({key: arrays[key].copy() for key in static_arrays})
    metadata["split_indices"] = {
        "train": np.searchsorted(keep, train).tolist(),
        "val": np.searchsorted(keep, val).tolist(),
    }
    metadata["num_samples"] = len(keep)
    for key in ("split_date_ranges", "split_boundaries", "purged_indices"):
        if isinstance(metadata.get(key), dict):
            metadata[key] = {k: v for k, v in metadata[key].items() if k != "test"}
    metadata["search_access_scope"] = "train_and_validation_only"
    limited = {"arrays": restricted, "metadata": metadata}
    try:
        FinancialTensorDataset(limited, "test")
    except KeyError:
        pass
    else:
        raise AssertionError("Test dataset construction must be impossible")
    assert set(metadata["split_indices"]) == {"train", "val"}
    assert all(len(restricted[key]) == len(keep) for key in sample_arrays)
    return limited, {
        "retained_original_indices": keep.tolist(),
        "train_count": len(train), "validation_count": len(val),
        "excluded_test_origin_count": len(test),
        "test_split_key_removed": True,
        "test_dataset_construction_raises_KeyError": True,
        "all_origin_arrays_physically_subset_to_train_and_validation": True,
    }


def configure(base, variant, candidate):
    config = copy.deepcopy(base)
    config["model"].update(candidate["model"])
    config["training"].update(candidate["training"])
    config["model"].update(ablation_variant=variant, signal_shrinkage=1.0)
    config["training"].update(
        epochs=20, patience=6, selection_metric="energy_score",
        signal_shrinkage_calibration=True, signal_shrinkage_lower=0.05,
        signal_shrinkage_upper=1.0, signal_shrinkage_tolerance=1e-5,
    )
    if variant in {"NoGraph", "NoStateSupervision"}:
        config["training"]["lambda_graph"] = 0.0
    if variant == "NoStateSupervision":
        config["training"]["lambda_concept"] = 0.0
    config["experiment"].update(name=f"rebuttal_search_fold_1_seed_1_{variant}_{candidate['id']}",
                                seed=1, device="cuda", deterministic=True)
    return config


def effective_fingerprint(config):
    effective = {"model": copy.deepcopy(config["model"]), "training": config["training"]}
    if effective["model"]["ablation_variant"] == "NoGraph":
        # With graph identically zero, refresh never enters forward or gain loss.
        effective["model"].pop("graph_refresh", None)
    return hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()


def run(args):
    started = time.perf_counter()
    root = args.root.resolve()
    protocol = args.protocol_dir or root / "runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1"
    reference = protocol / "runs/fold_1_seed_1"
    prepared_dir = args.prepared_dir or reference / "audit"
    output = args.output_dir or root / "rebuttal/validation_search"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("This registered search requires CUDA")
    device = torch.device("cuda")
    base_path = reference / "config_resolved.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    prepared, guard = train_validation_only(load_prepared(prepared_dir))
    source_paths = {name: root / "src/fconcord" / name for name in ("model.py", "trainer.py", "losses.py", "calibration.py")}
    registration = {
        "schema": "trefin-four-slot-validation-search-v1",
        "fold": 1, "training_seed": 1, "variants": args.variants,
        "candidates_in_tie_break_order": CANDIDATES,
        "base_configuration_sha256": sha256(base_path),
        "prepared_arrays_sha256": sha256(prepared_dir / "prepared_arrays.npz"),
        "prepared_metadata_sha256": sha256(prepared_dir / "prepared_metadata.json"),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "script_sha256": sha256(Path(__file__)),
        "training_budget_per_effective_candidate": {"epochs_ceiling": 20, "patience": 6, "seed": 1},
        "candidate_selection": "minimum calibrated validation energy score from train_model summary; exact tie chooses earlier registered candidate",
        "post_checkpoint_calibration": "identical validation-only positive mean scalar rho in [0.05,1] for every candidate and variant",
        "nominal_candidate_slots_per_variant": 4,
        "duplicate_policy": "skip mathematically identical effective candidates and record aliases; do not reallocate the unused slot",
        "known_effective_duplicates": {"NoGraph/slow_graph": "NoGraph/base", "NoStateSupervision/weaker_state_supervision": "NoStateSupervision/base"},
        "test_evaluation_permitted": False,
        "test_used_for_selection": False,
        "guard": {k: v for k, v in guard.items() if k != "retained_original_indices"},
    }
    registered_path = output / "search_registration.json"
    if registered_path.exists():
        if json.loads(registered_path.read_text()) != registration:
            raise ValueError("Search registration differs from existing output; choose a new output directory")
    else:
        dump(registration, registered_path)
        dump(guard, output / "train_validation_access_guard.json")
        dump(environment_snapshot(), output / "environment.json")
    selections = {}
    for variant in args.variants:
        variant_dir = output / variant
        variant_dir.mkdir(exist_ok=True)
        trials, effective_seen = [], {}
        for order, candidate in enumerate(CANDIDATES):
            config = configure(base, variant, candidate)
            fingerprint = effective_fingerprint(config)
            if fingerprint in effective_seen:
                canonical = effective_seen[fingerprint]
                trials.append({"candidate": candidate["id"], "order": order,
                               "effective_duplicate_of": canonical["candidate"],
                               "calibrated_validation_energy_score": canonical["calibrated_validation_energy_score"],
                               "effective_fingerprint": fingerprint, "executed": False})
                continue
            trial_dir = variant_dir / candidate["id"]
            completion_path = trial_dir / "complete.json"
            if completion_path.exists():
                record = json.loads(completion_path.read_text())
                if record["effective_fingerprint"] != fingerprint:
                    raise ValueError(f"Existing trial config mismatch: {trial_dir}")
            else:
                if trial_dir.exists() and any(trial_dir.iterdir()):
                    raise FileExistsError(f"Incomplete trial directory exists; preserve it and choose a new output directory: {trial_dir}")
                for sub in ("checkpoints", "logs", "audit"):
                    (trial_dir / sub).mkdir(parents=True, exist_ok=True)
                (trial_dir / "config_before_training.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                seed_everything(1, True)
                logger = setup_logger(trial_dir / "logs/run.log")
                logger.info("validation_search_started variant=%s candidate=%s test_access=forbidden", variant, candidate["id"])
                factory = lambda: _model_factory(prepared, config)
                model = factory()
                model, _, summary = train_model(model, prepared, config["training"], trial_dir, device, logger)
                calibration = summary["signal_shrinkage_calibration"]
                if calibration is None or calibration["selection_split"] != "validation":
                    raise AssertionError("Every candidate must have validation-only mean calibration")
                score = float(calibration["calibrated_validation_metrics"]["energy_score"])
                if not np.isfinite(score):
                    raise ValueError("Nonfinite validation selection score")
                config["model"]["signal_shrinkage"] = float(model.signal_shrinkage)
                reloaded, check = checkpoint_reload_check(factory, trial_dir / "checkpoints/best.pt", model,
                                                          prepared, config["training"], device)
                if not check["passed"]:
                    raise AssertionError("Validation checkpoint reload check failed")
                dump(check, trial_dir / "audit/checkpoint_reload_check.json")
                (trial_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                record = {
                    "variant": variant, "candidate": candidate["id"], "order": order,
                    "effective_fingerprint": fingerprint, "executed": True,
                    "calibrated_validation_energy_score": score,
                    "selected_signal_shrinkage": float(model.signal_shrinkage),
                    "training_summary": summary,
                    "checkpoint_sha256": sha256(trial_dir / "checkpoints/best.pt"),
                    "checkpoint_reload": check,
                    "test_access_count": 0,
                }
                dump(record, completion_path)
                del reloaded, model
                torch.cuda.empty_cache()
                print(json.dumps({"event": "candidate_complete", "variant": variant,
                                  "candidate": candidate["id"], "validation_es": score}), flush=True)
            trials.append(record)
            effective_seen[fingerprint] = record
        selected = min(trials, key=lambda item: (item["calibrated_validation_energy_score"], item["order"]))
        if not selected["executed"]:
            selected = next(item for item in trials if item["candidate"] == selected["effective_duplicate_of"])
        candidate = next(item for item in CANDIDATES if item["id"] == selected["candidate"])
        selected_config = configure(base, variant, candidate)
        selected_record = {
            "variant": variant, "selection_fold": 1, "selection_training_seed": 1,
            "selected_candidate": selected["candidate"],
            "selected_calibrated_validation_energy_score": selected["calibrated_validation_energy_score"],
            "best_overrides": {"model": candidate["model"], "training": candidate["training"]},
            "required_variant_overrides": {
                "model": {"ablation_variant": variant, "signal_shrinkage": 1.0},
                "training": {key: selected_config["training"][key] for key in ("lambda_concept", "lambda_graph")},
            },
            "selected_checkpoint": str((variant_dir / selected["candidate"] / "checkpoints/best.pt").resolve()),
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "nominal_slots": 4, "executed_unique_candidates": len(effective_seen),
            "trials": trials, "test_access_count": 0,
            "selection_frozen_before_any_test_evaluation": True,
            "follow_up": "Freeze best_overrides; retrain/evaluate all requested seeds using the same training/calibration protocol. Apply variant-specific zero loss coefficients after candidate overrides.",
        }
        # The selected config and choice are durably recorded before any other
        # program may run the separate final held-out evaluation.
        dump(selected_record, variant_dir / "selection.json")
        selections[variant] = selected_record
        print(json.dumps({"event": "variant_selected", "variant": variant,
                          "candidate": selected["candidate"],
                          "validation_es": selected["calibrated_validation_energy_score"]}), flush=True)
    dump({"registration_sha256": sha256(registered_path), "selections": selections,
          "wall_seconds": time.perf_counter() - started, "test_access_count": 0}, output / "selections.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
