from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact finite-sample multivariate energy score from Manual Eq. (ES).

    ``samples`` is [S,B,H,N].  The second term uses every ordered pair with
    unequal sample indices.  Chunking retains the exact double sum while
    bounding peak memory for large evaluation sample counts.
    """
    sample_count = samples.shape[0]
    if sample_count < 2:
        raise ValueError("Energy score requires at least two Monte Carlo samples")
    first = torch.linalg.vector_norm(samples - target.unsqueeze(0), dim=-1).mean(dim=0)
    values = samples.permute(1, 2, 0, 3)
    pair_sum = torch.zeros(values.shape[:2], device=samples.device, dtype=samples.dtype)
    chunk_size = min(64, sample_count)
    for start in range(0, sample_count, chunk_size):
        pair_sum = pair_sum + torch.cdist(values[:, :, start : start + chunk_size], values).sum(
            dim=(-2, -1)
        )
    second = 0.5 * pair_sum / (sample_count * (sample_count - 1))
    return (first - second).mean()


def graph_huber(
    prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    num_assets = prediction.shape[-1]
    mask = 1.0 - torch.eye(num_assets, device=prediction.device, dtype=prediction.dtype)
    standardized_error = (prediction - target) / scale
    loss = F.smooth_l1_loss(
        standardized_error, torch.zeros_like(standardized_error), reduction="none"
    )
    expanded_mask = mask.view(*([1] * (loss.ndim - 2)), num_assets, num_assets)
    return (loss * expanded_mask).sum() / (
        loss.shape[:-2].numel() * num_assets * (num_assets - 1)
    )


def monotonicity_penalty(risk_slopes: torch.Tensor) -> torch.Tensor:
    """The decoder uses positive slopes structurally; this audits numerical violations."""
    return F.relu(-risk_slopes).square().mean()


def total_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    risk_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    distribution = energy_score(outputs["samples"], batch["y"])
    concept_error = (
        outputs["concept_path"] - batch["c_target"]
    ) / outputs["concept_loss_scale"].view(1, 1, 1, -1)
    concept_element = F.smooth_l1_loss(
        concept_error, torch.zeros_like(concept_error), reduction="none"
    )
    concept_mask = outputs.get(
        "concept_supervision_mask",
        torch.ones(concept_error.shape[-1], device=concept_error.device),
    ).to(concept_error)
    concept = (
        concept_element * concept_mask.view(1, 1, 1, -1)
    ).sum() / (concept_error.shape[:-1].numel() * concept_mask.sum().clamp_min(1.0))
    graph = graph_huber(
        outputs["graph_path"], batch["a_target"], outputs["graph_loss_scale"]
    )
    del risk_indices  # retained in the public signature for loss/config consistency
    monotonicity = monotonicity_penalty(outputs["risk_slopes"])
    gain = F.relu(outputs["gain_radius"] - float(config["gain_target"])).square()
    total = (
        distribution
        + float(config["lambda_concept"]) * concept
        + float(config["lambda_graph"]) * graph
        + float(config["lambda_monotonicity"]) * monotonicity
        + float(config["lambda_gain"]) * gain
    )
    components = {
        "total": float(total.detach()),
        "distribution": float(distribution.detach()),
        "concept": float(concept.detach()),
        "graph": float(graph.detach()),
        "monotonicity": float(monotonicity.detach()),
        "gain": float(gain.detach()),
        "gain_radius": float(outputs["gain_radius"].detach()),
    }
    return total, components
