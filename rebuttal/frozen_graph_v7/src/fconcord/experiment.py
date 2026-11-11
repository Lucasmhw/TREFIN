from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fconcord.audit import audit_completed_run
from fconcord.baselines import evaluate_lightweight_baselines
from fconcord.certificates import empirical_certificate_report
from fconcord.config import save_config
from fconcord.data import build_research_dataset, generate_synthetic_market, load_panel_csv
from fconcord.metrics import evaluate_predictions
from fconcord.model import FCONCORD
from fconcord.trainer import checkpoint_reload_check, collect_predictions, train_model
from fconcord.utils import (
    count_parameters,
    environment_snapshot,
    json_dump,
    make_run_dir,
    seed_everything,
    select_device,
    setup_logger,
)
from fconcord.visualization import generate_all_figures, generate_runtime_figure


def _save_prepared(prepared: dict[str, Any], directory: Path) -> None:
    np.savez_compressed(directory / "prepared_arrays.npz", **prepared["arrays"])
    json_dump(prepared["metadata"], directory / "prepared_metadata.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_cache_key(
    config: dict[str, Any], data_provenance: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Hash every deterministic input to causal state precomputation."""
    repository_root = Path(__file__).resolve().parents[2]
    implementation_files = [
        repository_root / "src" / "fconcord" / "concepts.py",
        repository_root / "src" / "fconcord" / "graph.py",
        repository_root / "src" / "fconcord" / "data" / "dataset.py",
    ]
    data_config = {
        key: value
        for key, value in config["data"].items()
        if key not in {"prepared_cache", "prepared_cache_dir"}
    }
    payload = {
        "schema": "fconcord-prepared-state-v2",
        "data_config": data_config,
        "input_sha256": {
            role: record["sha256"]
            for role, record in data_provenance["files"].items()
            if role in {"panel", "factors"}
        },
        "implementation_sha256": {
            str(path.relative_to(repository_root)).replace("\\", "/"): _sha256(path)
            for path in implementation_files
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _prepare_with_cache(
    panel: dict[str, Any],
    config: dict[str, Any],
    data_provenance: dict[str, Any],
    run_path: Path,
    logger: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build once per data/fold/formula contract and reuse across random seeds."""
    enabled = config["data"]["source"] == "csv" and bool(
        config["data"].get("prepared_cache", True)
    )
    if not enabled:
        prepared = build_research_dataset(
            panel, config["data"], progress_callback=logger.info
        )
        _save_prepared(prepared, run_path / "audit")
        record = {"enabled": False, "hit": False, "cache_key": None}
        json_dump(record, run_path / "audit" / "prepared_cache.json")
        return prepared, record

    cache_key, cache_inputs = _prepared_cache_key(config, data_provenance)
    cache_root = Path(
        config["data"].get("prepared_cache_dir", "data/prepared_cache")
    ).resolve()
    cache_path = cache_root / cache_key
    arrays_path = cache_path / "prepared_arrays.npz"
    metadata_path = cache_path / "prepared_metadata.json"
    hit = arrays_path.is_file() and metadata_path.is_file()
    if hit:
        logger.info("prepared_cache_hit key=%s path=%s", cache_key, cache_path)
    else:
        logger.info("prepared_cache_miss key=%s path=%s", cache_key, cache_path)
        prepared = build_research_dataset(
            panel, config["data"], progress_callback=logger.info
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        _save_prepared(prepared, cache_path)
        json_dump(cache_inputs, cache_path / "cache_inputs.json")
    shutil.copy2(arrays_path, run_path / "audit" / "prepared_arrays.npz")
    shutil.copy2(metadata_path, run_path / "audit" / "prepared_metadata.json")
    prepared = load_prepared(run_path / "audit")
    record = {
        "enabled": True,
        "hit": hit,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "cache_inputs": cache_inputs,
    }
    json_dump(record, run_path / "audit" / "prepared_cache.json")
    return prepared, record


def _record_csv_provenance(config: dict[str, Any], run_path: Path) -> dict[str, Any]:
    """Snapshot immutable identifiers for every real-data input used by this run."""
    records: dict[str, Any] = {"source": config["data"]["source"], "files": {}}
    if config["data"]["source"] != "csv":
        json_dump(records, run_path / "audit" / "data_provenance.json")
        return records
    for role, key in (("panel", "input_path"), ("factors", "factor_path")):
        path = Path(config["data"][key]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} data file: {path}")
        records["files"][role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_value = config["data"].get("manifest_path")
    if manifest_value is not None:
        manifest = Path(manifest_value).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing configured data manifest: {manifest}")
        copied_manifest = run_path / "audit" / "source_dataset_manifest.json"
        shutil.copy2(manifest, copied_manifest)
        records["files"]["manifest"] = {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": _sha256(manifest),
            "snapshot": str(copied_manifest.resolve()),
        }
    json_dump(records, run_path / "audit" / "data_provenance.json")
    return records


def _snapshot_manual_contract(run_path: Path) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    contract_path = repository_root / "docs" / "manual_contract.yaml"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing manual compliance contract: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    snapshot = run_path / "audit" / "manual_contract.yaml"
    shutil.copy2(contract_path, snapshot)
    methodology_path = repository_root / "docs" / "METHODOLOGY_CODE_SPEC.md"
    if not methodology_path.is_file():
        raise FileNotFoundError(f"Missing code-normative methodology: {methodology_path}")
    methodology_snapshot = run_path / "audit" / "METHODOLOGY_CODE_SPEC.md"
    shutil.copy2(methodology_path, methodology_snapshot)
    implementation_relative_paths = [
        "src/fconcord/concepts.py",
        "src/fconcord/graph.py",
        "src/fconcord/data/dataset.py",
        "src/fconcord/model.py",
        "src/fconcord/losses.py",
        "src/fconcord/metrics.py",
        "src/fconcord/certificates.py",
        "src/fconcord/trainer.py",
        "src/fconcord/experiment.py",
    ]
    implementation_hashes = {
        relative_path: _sha256(repository_root / relative_path)
        for relative_path in implementation_relative_paths
    }
    fingerprint_payload = json.dumps(
        implementation_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    methodology_manifest = {
        "normative_policy": "executed code + resolved config + run audit",
        "implementation_files": implementation_hashes,
        "implementation_combined_sha256": hashlib.sha256(
            fingerprint_payload
        ).hexdigest(),
        "resolved_config_sha256": _sha256(run_path / "config_resolved.yaml"),
        "methodology_document_sha256": _sha256(methodology_path),
        "methodology_snapshot": str(methodology_snapshot.resolve()),
    }
    json_dump(
        methodology_manifest,
        run_path / "audit" / "methodology_implementation_manifest.json",
    )
    counts: dict[str, int] = {}
    status_by_id: dict[str, str] = {}
    for requirement in contract["requirements"]:
        status = str(requirement["status"])
        counts[status] = counts.get(status, 0) + 1
        status_by_id[str(requirement["id"])] = status
    required = contract["claim_rule"]["publishable_manual_replication_requires"]
    open_required = [
        identifier
        for identifier in required
        if status_by_id[identifier] != "implemented"
    ]
    summary = {
        "manual_sha256": contract["manual"]["sha256"],
        "methodology_implementation_sha256": methodology_manifest[
            "implementation_combined_sha256"
        ],
        "resolved_config_sha256": methodology_manifest["resolved_config_sha256"],
        "status_counts": counts,
        "publishable_manual_replication_complete": not open_required,
        "open_publishable_requirements": open_required,
        "allowed_claim": contract["claim_rule"]["current_claim_allowed"],
        "contract_snapshot": str(snapshot.resolve()),
        "methodology_snapshot": str(methodology_snapshot.resolve()),
    }
    json_dump(summary, run_path / "audit" / "manual_compliance_summary.json")
    return summary


def load_prepared(directory: Path) -> dict[str, Any]:
    with np.load(directory / "prepared_arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    import json

    with (directory / "prepared_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {"arrays": arrays, "metadata": metadata}


def _model_factory(
    prepared: dict[str, Any], config: dict[str, Any]
) -> FCONCORD:
    metadata = prepared["metadata"]
    return FCONCORD(
        num_assets=int(metadata["num_assets"]),
        num_concepts=int(metadata["num_concepts"]),
        horizon=int(metadata["horizon"]),
        directional_mask=torch.from_numpy(prepared["arrays"]["directional_mask"]),
        risk_indices=[int(value) for value in prepared["arrays"]["risk_indices"]],
        config=config["model"],
        concept_bound=float(
            max(
                np.max(np.abs(metadata["concept_lower_bounds"])),
                np.max(np.abs(metadata["concept_upper_bounds"])),
            )
        ),
        graph_bound=float(metadata["graph_bound"]),
        return_scale=torch.tensor(metadata["return_scale"], dtype=torch.float32),
        concept_lower_bound=torch.tensor(
            metadata["concept_lower_bounds"], dtype=torch.float32
        ),
        concept_upper_bound=torch.tensor(
            metadata["concept_upper_bounds"], dtype=torch.float32
        ),
        market_lower_bound=torch.tensor(
            metadata["market_lower_bounds"], dtype=torch.float32
        ),
        market_upper_bound=torch.tensor(
            metadata["market_upper_bounds"], dtype=torch.float32
        ),
        concept_loss_scale=torch.tensor(
            metadata["concept_loss_scale"], dtype=torch.float32
        ),
        graph_loss_scale=float(metadata["graph_loss_scale"]),
    )


def _evaluate_ablations(
    model: FCONCORD,
    prepared: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    levels = [float(value) for value in config["evaluation"]["alpha_levels"]]

    def no_graph(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {**batch, "a0": torch.zeros_like(batch["a0"])}

    def no_market(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {**batch, "s0": torch.zeros_like(batch["s0"])}

    risk_indices = model.risk_indices

    def no_risk(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concepts = batch["c0"].clone()
        concepts.index_fill_(-1, risk_indices, 0.0)
        return {**batch, "c0": concepts}

    transforms = {"NoGraph": no_graph, "NoMarketState": no_market, "NoRiskConcepts": no_risk}
    results: dict[str, Any] = {}
    for name, transform in transforms.items():
        # Common random numbers isolate the ablated input from Monte Carlo
        # scenario noise. No test result is used to tune these transforms.
        seed_everything(
            int(config["training"].get("evaluation_seed", 271828)) + 101,
            bool(config["experiment"]["deterministic"]),
        )
        prediction = collect_predictions(
            model, prepared, "test", config["training"], device, input_transform=transform
        )
        results[name] = evaluate_predictions(
            prediction["y"], prediction["mu"], prediction["samples"], levels
        )
    return results


@torch.no_grad()
def _run_theory_audits(
    model: FCONCORD,
    prepared: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    test_index = int(prepared["metadata"]["split_indices"]["test"][0])
    arrays = prepared["arrays"]
    c0 = torch.from_numpy(arrays["c0"][test_index : test_index + 1]).to(device)
    a0 = torch.from_numpy(arrays["a0"][test_index : test_index + 1]).to(device)
    s0 = torch.from_numpy(arrays["s0"][test_index : test_index + 1]).to(device)
    risk_index = int(arrays["risk_indices"][0])
    grid = np.linspace(-2.0, 2.0, 17)
    response_mean, response_down = [], []
    closed_loop_down = []
    base_output = model(c0, a0, s0, num_samples=16)
    fixed_concepts = base_output["concepts"][:, 0].clone()
    fixed_graph = base_output["graphs"][:, 0].clone()
    decode_lead = model._lead(c0.shape[0], 1, c0.device)
    for value in grid:
        conditional_concepts = fixed_concepts.clone()
        conditional_concepts[:, :, risk_index] = value
        conditional = model._decode_state(
            conditional_concepts, fixed_graph, s0, decode_lead
        )
        response_mean.append(float(conditional["mu"].mean().cpu()))
        response_down.append(float(conditional["sigma_minus"].mean().cpu()))
        initial_intervention = c0.clone()
        initial_intervention[:, :, risk_index] = value
        closed_loop = model(initial_intervention, a0, s0, num_samples=16)
        closed_loop_down.append(float(closed_loop["sigma_minus"][:, 0].mean().cpu()))
    slopes = np.diff(response_down) / np.diff(grid)

    reference = base_output["mu"]
    perturbation_size = np.logspace(-4, -1, 7)
    amplification = []
    generator = torch.Generator(device=device).manual_seed(90210)
    direction = torch.randn(c0.shape, generator=generator, device=device)
    direction = direction / torch.linalg.vector_norm(direction)
    for epsilon in perturbation_size:
        perturbed = model(c0 + float(epsilon) * direction, a0, s0, num_samples=16)["mu"]
        amplification.append(float(torch.linalg.vector_norm(perturbed - reference).cpu() / epsilon))
    gain_matrix, gain_radius = model.gain_proxy()
    minimum_precision_eigenvalue = float(
        torch.linalg.eigvalsh(base_output["precision"]).min().cpu()
    )
    graph_symmetry_error = float(
        (base_output["graphs"] - base_output["graphs"].transpose(-1, -2))
        .abs()
        .max()
        .cpu()
    )
    graph_diagonal_error = float(
        torch.diagonal(base_output["graphs"], dim1=-2, dim2=-1).abs().max().cpu()
    )
    lower_violation = torch.relu(
        model.concept_lower_bound.view(1, 1, 1, -1) - base_output["concept_path"]
    )
    upper_violation = torch.relu(
        base_output["concept_path"] - model.concept_upper_bound.view(1, 1, 1, -1)
    )
    concept_bound_violation = float(
        torch.maximum(lower_violation, upper_violation).max().cpu()
    )
    gain_target = float(config["training"]["gain_target"])
    return {
        "intervention_grid": grid.tolist(),
        "intervention_mean": response_mean,
        "intervention_down_scale": response_down,
        "closed_loop_intervention_down_scale": closed_loop_down,
        "minimum_intervention_slope": float(slopes.min()),
        "conditional_monotonicity_passed": bool(slopes.min() >= -1e-10),
        "perturbation_size": perturbation_size.tolist(),
        "perturbation_amplification": amplification,
        "gain_matrix": gain_matrix.detach().cpu().tolist(),
        "gain_radius": float(gain_radius.detach().cpu()),
        "gain_target": gain_target,
        "structural_certificate_passed": bool(
            float(gain_radius.detach().cpu()) < gain_target
        ),
        "gain_proxy_passed": bool(float(gain_radius.detach().cpu()) < gain_target),
        "minimum_precision_eigenvalue": minimum_precision_eigenvalue,
        "precision_spd_passed": bool(minimum_precision_eigenvalue > 0.0),
        "graph_symmetry_max_error": graph_symmetry_error,
        "graph_diagonal_max_error": graph_diagonal_error,
        "concept_bound_violation": concept_bound_violation,
        "risk_concept_index": risk_index,
        "strategy_top_fraction": float(config["evaluation"]["top_fraction"]),
        "transaction_cost_bps": float(config["evaluation"]["transaction_cost_bps"]),
    }


def run_experiment(config: dict[str, Any], run_dir: str | Path | None = None) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    device = select_device(str(config["experiment"]["device"]))
    if run_dir is None:
        run_path = make_run_dir(config["experiment"]["run_root"], config["experiment"]["name"])
    else:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=False)
        for child in ["checkpoints", "logs", "predictions", "figures", "tables", "audit"]:
            (run_path / child).mkdir()
    logger = setup_logger(run_path / "logs" / "run.log")
    save_config(config, run_path / "config_resolved.yaml")
    json_dump(environment_snapshot(), run_path / "environment.json")
    manual_compliance = _snapshot_manual_contract(run_path)
    logger.info("run_dir=%s device=%s", run_path.resolve(), device)

    data_start = time.perf_counter()
    data_provenance = _record_csv_provenance(config, run_path)
    if config["data"]["source"] == "synthetic":
        panel = generate_synthetic_market(
            int(config["data"]["num_steps"]),
            int(config["data"]["num_assets"]),
            int(config["data"]["num_factors"]),
            seed,
        )
    elif config["data"]["source"] == "csv":
        panel = load_panel_csv(
            config["data"]["input_path"],
            config["data"]["factor_path"],
            config["data"],
        )
    else:
        raise ValueError(f"Unsupported data source: {config['data']['source']}")
    prepared, prepared_cache = _prepare_with_cache(
        panel, config, data_provenance, run_path, logger
    )
    data_seconds = time.perf_counter() - data_start
    logger.info(
        "prepared samples=%d assets=%d concepts=%d horizon=%d",
        prepared["metadata"]["num_samples"],
        prepared["metadata"]["num_assets"],
        prepared["metadata"]["num_concepts"],
        prepared["metadata"]["horizon"],
    )
    logger.info("split_date_ranges=%s", prepared["metadata"]["split_date_ranges"])
    if data_provenance["source"] == "csv":
        logger.info(
            "real_data_sha256 panel=%s factors=%s",
            data_provenance["files"]["panel"]["sha256"],
            data_provenance["files"]["factors"]["sha256"],
        )

    factory = lambda: _model_factory(prepared, config)
    model = factory()
    logger.info("trainable_parameters=%d", count_parameters(model))
    model, history, training_summary = train_model(
        model, prepared, config["training"], run_path, device, logger
    )
    calibration = training_summary.get("signal_shrinkage_calibration")
    if calibration is not None:
        config["model"]["signal_shrinkage"] = float(
            calibration["selected_signal_shrinkage"]
        )
        # The resolved configuration must reconstruct the calibrated decoder
        # before the fresh checkpoint reload and every later prediction run.
        save_config(config, run_path / "config_resolved.yaml")
    logger.info(
        "training_complete best_epoch=%d selection_metric=val_%s best_value=%.8f",
        training_summary["best_epoch"],
        training_summary["selection_metric"],
        training_summary["best_val"],
    )

    reloaded, reload_check = checkpoint_reload_check(
        factory,
        run_path / "checkpoints" / "best.pt",
        model,
        prepared,
        config["training"],
        device,
    )
    json_dump(reload_check, run_path / "audit" / "checkpoint_reload_check.json")
    if not bool(reload_check["passed"]):
        raise RuntimeError(f"Checkpoint reload mismatch: {reload_check}")
    logger.info("checkpoint_reload max_abs_difference=%.3e", reload_check["max_abs_difference"])

    # This is the first primary evaluation of the final holdout.  No test metric has
    # participated in gradient updates, early stopping, or checkpoint selection.
    logger.info("final_test_started checkpoint=best.pt")
    prediction_start = time.perf_counter()
    seed_everything(seed + 1, bool(config["experiment"]["deterministic"]))
    predictions = collect_predictions(
        reloaded, prepared, "test", config["training"], device
    )
    np.savez_compressed(run_path / "predictions" / "test_predictions.npz", **predictions)
    levels = [float(value) for value in config["evaluation"]["alpha_levels"]]
    metrics = evaluate_predictions(
        predictions["y"], predictions["mu"], predictions["samples"], levels
    )
    baselines = evaluate_lightweight_baselines(prepared)
    ablations = _evaluate_ablations(reloaded, prepared, config, device)
    prediction_seconds = time.perf_counter() - prediction_start
    json_dump(metrics, run_path / "tables" / "test_metrics.json")
    json_dump(baselines, run_path / "tables" / "baseline_metrics.json")
    json_dump(ablations, run_path / "tables" / "ablation_metrics.json")
    logger.info(
        "final_test_complete mse=%.10f rmse=%.8f mae=%.8f "
        "directional_accuracy=%.6f energy_score=%.8f",
        metrics["mse"],
        metrics["rmse"],
        metrics["mae"],
        metrics["directional_accuracy"],
        metrics["energy_score"],
    )

    audit = _run_theory_audits(reloaded, prepared, config, device)
    certificate_iterations = int(
        config["evaluation"].get("empirical_certificate_iterations", 20)
    )
    configured_max_states = config["evaluation"].get(
        "empirical_certificate_max_states"
    )
    certificate_max_states = (
        None if configured_max_states is None else int(configured_max_states)
    )
    empirical_certificates = {
        split: empirical_certificate_report(
            reloaded,
            prepared,
            split,
            device,
            iterations=certificate_iterations,
            max_states=certificate_max_states,
            batch_size=int(
                config["evaluation"].get("empirical_certificate_batch_size", 16)
            ),
        )
        for split in ("train", "val", "test")
    }
    json_dump(
        empirical_certificates,
        run_path / "audit" / "empirical_jacobian_certificates.json",
    )
    audit["empirical_jacobian_summary"] = {
        split: {
            key: report[key]
            for key in (
                "iterations",
                "audited_origins",
                "total_split_origins",
                "origin_coverage",
                "radius_mean",
                "radius_max",
                "contraction_coverage_rho_lt_1",
                "block_gain_mean",
                "block_gain_max",
                "scope",
            )
        }
        for split, report in empirical_certificates.items()
    }
    audit["runtime_seconds"] = {
        "data": data_seconds,
        "training": training_summary["train_seconds"],
        "prediction": prediction_seconds,
        "figures": 0.0,
    }
    hard_checks = {
        "conditional_monotonicity": bool(audit["conditional_monotonicity_passed"]),
        "precision_spd": bool(audit["precision_spd_passed"]),
        "graph_symmetry": float(audit["graph_symmetry_max_error"]) < 1e-6,
        "graph_zero_diagonal": float(audit["graph_diagonal_max_error"]) < 1e-6,
        "concept_projection": float(audit["concept_bound_violation"]) < 1e-6,
    }
    audit["hard_invariant_checks"] = hard_checks
    audit["hard_invariants_passed"] = all(hard_checks.values())
    audit["certificate_scope"] = (
        "conditional decoder monotonicity and algebraic constraints are hard checks; "
        "gain_matrix is a conservative post-parameter structural bound over the registered "
        "state sets. Empirical Jacobian diagnostics are reported separately when enabled."
    )
    json_dump(audit, run_path / "audit" / "theory_audit.json")
    if not audit["hard_invariants_passed"]:
        raise RuntimeError(f"Hard theory invariant failed: {hard_checks}")
    figure_start = time.perf_counter()
    manifest = generate_all_figures(
        run_path,
        predictions,
        prepared,
        history,
        metrics,
        baselines,
        ablations,
        audit,
        config["figures"],
    )
    audit["runtime_seconds"]["figures"] = time.perf_counter() - figure_start
    json_dump(audit, run_path / "audit" / "theory_audit.json")
    generate_runtime_figure(run_path, audit, config["figures"])
    summary = {
        **training_summary,
        "device": str(device),
        "parameter_count": count_parameters(reloaded),
        "test_metrics": metrics,
        "reload_check": reload_check,
        "figure_count": len(manifest),
        "hard_invariant_checks": hard_checks,
        "gain_proxy_passed": bool(audit["gain_proxy_passed"]),
        "graph_estimation": prepared["metadata"]["graph_estimation"],
        "embargo": prepared["metadata"]["embargo"],
        "runtime_seconds": audit["runtime_seconds"],
        "debug_run": config["data"]["source"] == "synthetic",
        "data_provenance": data_provenance,
        "prepared_cache": prepared_cache,
        "manual_compliance": manual_compliance,
        "epoch_metric_protocol": {
            "logged_splits": ["train", "validation"],
            "metrics": [
                "total",
                "energy_score",
                "mse",
                "rmse",
                "mae",
                "directional_accuracy",
                "msic",
                "msir",
            ],
            "test_used_for_model_selection": False,
            "final_test_after_checkpoint_reload": True,
            "training_scenarios": int(config["training"]["train_samples"]),
            "validation_scenarios": int(
                config["training"].get(
                    "validation_samples", config["training"]["eval_samples"]
                )
            ),
            "test_scenarios": int(
                config["training"].get(
                    "test_samples", config["training"]["eval_samples"]
                )
            ),
        },
    }
    json_dump(summary, run_path / "run_summary.json")
    integrity_report = audit_completed_run(run_path)
    summary["artifact_integrity"] = integrity_report
    json_dump(summary, run_path / "run_summary.json")
    latest = Path(config["experiment"]["run_root"]) / "LATEST_RUN.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(run_path.resolve()), encoding="utf-8")
    logger.info(
        "complete rmse=%.6f energy=%.6f figures=%d integrity=%s",
        metrics["rmse"],
        metrics["energy_score"],
        len(manifest),
        integrity_report["passed"],
    )
    return run_path


def regenerate_figures(run_dir: str | Path) -> list[dict[str, str]]:
    import json

    import yaml

    run_path = Path(run_dir)
    with (run_path / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prepared = load_prepared(run_path / "audit")
    with np.load(run_path / "predictions" / "test_predictions.npz", allow_pickle=False) as handle:
        predictions = {key: handle[key] for key in handle.files}
    history = pd_read_records(run_path / "logs" / "training_history.csv")
    def load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return generate_all_figures(
        run_path,
        predictions,
        prepared,
        history,
        load_json(run_path / "tables" / "test_metrics.json"),
        load_json(run_path / "tables" / "baseline_metrics.json"),
        load_json(run_path / "tables" / "ablation_metrics.json"),
        load_json(run_path / "audit" / "theory_audit.json"),
        config["figures"],
    )


def pd_read_records(path: Path) -> list[dict[str, float]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")
