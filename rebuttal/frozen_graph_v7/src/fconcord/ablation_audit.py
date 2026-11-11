from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fconcord.utils import json_dump


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_ablation_protocol(
    protocol_directory: str | Path, reference_protocol: str | Path
) -> dict[str, Any]:
    """Independently verify retraining, serialization and structural removals."""
    protocol = Path(protocol_directory).resolve()
    reference = Path(reference_protocol).resolve()
    manifest = _load_json(protocol / "protocol_manifest.json")
    folds = [str(value) for value in manifest["folds"]]
    seeds = [int(value) for value in manifest["seeds"]]
    variants = [str(value) for value in manifest["variants"]]
    expected = len(folds) * len(seeds) * len(variants)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    checkpoints_reopened = 0
    for fold in folds:
        for seed in seeds:
            reference_run = reference / "runs" / f"{fold}_seed_{seed}"
            with np.load(
                reference_run / "predictions" / "test_predictions.npz",
                allow_pickle=False,
            ) as handle:
                reference_target = handle["y"]
                reference_indices = handle["indices"]
            for variant in variants:
                name = f"{fold}_seed_{seed}_{variant}"
                run = protocol / "runs" / name
                try:
                    result = _load_json(run / "ablation_result.json")
                    if result["variant"] != variant:
                        raise AssertionError("variant identity mismatch")
                    if int(result["test_access_count"]) != 1:
                        raise AssertionError("test access count is not one")
                    if bool(result["test_used_for_training_or_selection"]):
                        raise AssertionError("test data participated in selection")
                    if not bool(result["checkpoint_reload_before_test"]):
                        raise AssertionError("best checkpoint was not reloaded before test")
                    if float(result["checkpoint_reload_max_abs_difference"]) >= 1e-7:
                        raise AssertionError("checkpoint reload equality failed")
                    for checkpoint_name in ("best.pt", "last.pt"):
                        payload = torch.load(
                            run / "checkpoints" / checkpoint_name,
                            map_location="cpu",
                            weights_only=False,
                        )
                        if "model_state" not in payload:
                            raise AssertionError("checkpoint lacks model_state")
                        checkpoints_reopened += 1
                    history = pd.read_csv(run / "logs" / "training_history.csv")
                    if any("test" in str(column).lower() for column in history.columns):
                        raise AssertionError("training history contains test metrics")
                    with (run / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
                        config = yaml.safe_load(handle)
                    if config["model"]["ablation_variant"] != variant:
                        raise AssertionError("resolved config variant mismatch")
                    if variant == "NoGraph" and float(config["training"]["lambda_graph"]) != 0.0:
                        raise AssertionError("NoGraph retained graph-target supervision")
                    if (
                        variant == "NoForwardConceptSupervision"
                        and float(config["training"]["lambda_concept"]) != 0.0
                    ):
                        raise AssertionError("concept supervision was not removed")
                    with np.load(
                        run / "predictions" / "test_predictions.npz",
                        allow_pickle=False,
                    ) as handle:
                        target = handle["y"]
                        indices = handle["indices"]
                        if variant == "NoGraph" and np.count_nonzero(handle["graphs"]):
                            raise AssertionError("NoGraph produced a nonzero graph state")
                        if variant == "NoRiskConcepts":
                            prepared = np.load(
                                reference_run / "audit" / "prepared_arrays.npz",
                                allow_pickle=False,
                            )
                            risk_indices = prepared["risk_indices"]
                            prepared.close()
                            if np.count_nonzero(handle["concepts"][..., risk_indices]):
                                raise AssertionError("NoRiskConcepts regenerated risk state")
                        if variant == "GaussianSymmetricDecoder":
                            np.testing.assert_allclose(
                                handle["sigma_plus"],
                                handle["sigma_minus"],
                                rtol=0.0,
                                atol=0.0,
                            )
                    np.testing.assert_array_equal(indices, reference_indices)
                    np.testing.assert_allclose(target, reference_target, rtol=0.0, atol=1e-10)
                    records.append({"run_name": name, "variant": variant, "passed": True})
                except Exception as error:  # noqa: BLE001 - retain all independent failures
                    failures.append(f"{name}: {type(error).__name__}: {error}")
                    records.append({"run_name": name, "variant": variant, "passed": False})
    report = {
        "schema": "fconcord-retrained-ablation-protocol-audit-v1",
        "protocol_directory": str(protocol),
        "reference_protocol": str(reference),
        "expected_runs": expected,
        "audited_runs": len(records),
        "passed_runs": sum(bool(record["passed"]) for record in records),
        "checkpoints_reopened": checkpoints_reopened,
        "failures": failures,
        "records": records,
        "passed": len(records) == expected and not failures,
    }
    json_dump(report, protocol / "ablation_protocol_audit.json")
    return report
