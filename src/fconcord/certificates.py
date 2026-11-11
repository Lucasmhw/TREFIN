from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fconcord.model import FCONCORD


def _batch_frobenius(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=1)


def _normalize_batch(values: torch.Tensor) -> torch.Tensor:
    norms = _batch_frobenius(values).clamp_min(1e-12)
    return values / norms.view(values.shape[0], *([1] * (values.ndim - 1)))


def _graph_tangent(values: torch.Tensor) -> torch.Tensor:
    values = 0.5 * (values + values.transpose(-1, -2))
    eye = torch.eye(values.shape[-1], device=values.device, dtype=values.dtype)
    return values * (1.0 - eye)


def _block_spectral_norm(
    function: Any,
    point: torch.Tensor,
    iterations: int,
    graph_tangent: bool,
    seed: int,
) -> torch.Tensor:
    """Estimate each batch member's local Jacobian block norm by JVP/VJP power iteration."""
    generator = torch.Generator(device=point.device).manual_seed(seed)
    direction = torch.randn(
        point.shape, generator=generator, device=point.device, dtype=point.dtype
    )
    if graph_tangent:
        direction = _graph_tangent(direction)
    direction = _normalize_batch(direction)
    base = point.detach()
    for _ in range(iterations):
        _, jvp = torch.autograd.functional.jvp(
            function, base, direction, create_graph=False, strict=False
        )
        variable = base.detach().requires_grad_(True)
        output = function(variable)
        adjoint = torch.autograd.grad(
            output,
            variable,
            grad_outputs=jvp.detach(),
            create_graph=False,
            retain_graph=False,
        )[0]
        if graph_tangent:
            adjoint = _graph_tangent(adjoint)
        direction = _normalize_batch(adjoint.detach())
    _, jvp = torch.autograd.functional.jvp(
        function, base, direction, create_graph=False, strict=False
    )
    return _batch_frobenius(jvp).detach()


def local_block_gain_matrix(
    model: FCONCORD,
    concepts: torch.Tensor,
    graph: torch.Tensor,
    market: torch.Tensor,
    lead_index: int,
    iterations: int = 20,
    seed: int = 314159,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empirical [B,2,2] block gains and their nonnegative spectral radii."""
    if iterations < 1:
        raise ValueError("Power iteration count must be positive")
    lead = model._lead(concepts.shape[0], lead_index, concepts.device).detach()
    fixed_c = concepts.detach()
    fixed_a = graph.detach()
    fixed_s = market.detach()

    def concept_from_concept(value: torch.Tensor) -> torch.Tensor:
        return model._step(value, fixed_a, fixed_s, lead)[0]

    def concept_from_graph(value: torch.Tensor) -> torch.Tensor:
        return model._step(fixed_c, value, fixed_s, lead)[0]

    def graph_from_concept(value: torch.Tensor) -> torch.Tensor:
        return model._step(value, fixed_a, fixed_s, lead)[1]

    def graph_from_graph(value: torch.Tensor) -> torch.Tensor:
        return model._step(fixed_c, value, fixed_s, lead)[1]

    m_cc = _block_spectral_norm(
        concept_from_concept, fixed_c, iterations, False, seed
    )
    m_ca = _block_spectral_norm(
        concept_from_graph, fixed_a, iterations, True, seed + 1
    )
    m_ac = _block_spectral_norm(
        graph_from_concept, fixed_c, iterations, False, seed + 2
    )
    m_aa = _block_spectral_norm(
        graph_from_graph, fixed_a, iterations, True, seed + 3
    )
    matrices = torch.stack(
        [torch.stack([m_cc, m_ca], dim=-1), torch.stack([m_ac, m_aa], dim=-1)],
        dim=-2,
    )
    discriminant = torch.sqrt((m_cc - m_aa) ** 2 + 4.0 * m_ca * m_ac)
    radius = 0.5 * (m_cc + m_aa + discriminant)
    return matrices, radius


def empirical_certificate_report(
    model: FCONCORD,
    prepared: dict[str, Any],
    split: str,
    device: torch.device,
    iterations: int = 20,
    max_states: int | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Audit predicted closed-loop states without reading future labels."""
    all_indices = np.asarray(prepared["metadata"]["split_indices"][split], dtype=np.int64)
    if max_states is not None and len(all_indices) > max_states:
        positions = np.unique(
            np.linspace(0, len(all_indices) - 1, max_states, dtype=np.int64)
        )
        indices = all_indices[positions]
    else:
        indices = all_indices
    arrays = prepared["arrays"]
    matrix_rows: list[np.ndarray] = []
    radius_rows: list[np.ndarray] = []
    origin_rows: list[np.ndarray] = []
    lead_rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        concepts = torch.from_numpy(arrays["c0"][batch_indices]).to(device)
        graph = torch.from_numpy(arrays["a0"][batch_indices]).to(device)
        market = torch.from_numpy(arrays["s0"][batch_indices]).to(device)
        for lead_index in range(model.horizon):
            matrices, radii = local_block_gain_matrix(
                model,
                concepts,
                graph,
                market,
                lead_index,
                iterations=iterations,
                seed=314159 + 1009 * lead_index + start,
            )
            matrix_rows.append(matrices.cpu().numpy())
            radius_rows.append(radii.cpu().numpy())
            origin_rows.append(batch_indices.copy())
            lead_rows.append(np.full(len(batch_indices), lead_index, dtype=np.int64))
            with torch.no_grad():
                lead = model._lead(len(batch_indices), lead_index, device)
                concepts, graph = model._step(concepts, graph, market, lead)
    matrices_np = np.concatenate(matrix_rows)
    radii_np = np.concatenate(radius_rows)
    origins_np = np.concatenate(origin_rows)
    leads_np = np.concatenate(lead_rows)
    return {
        "split": split,
        "method": "local JVP/VJP power iteration on four closed-loop blocks",
        "iterations": iterations,
        "audited_origins": len(indices),
        "total_split_origins": len(all_indices),
        "origin_coverage": float(len(indices) / len(all_indices)),
        "audited_state_leads": len(radii_np),
        "radius_mean": float(radii_np.mean()),
        "radius_median": float(np.median(radii_np)),
        "radius_max": float(radii_np.max()),
        "contraction_coverage_rho_lt_1": float(np.mean(radii_np < 1.0)),
        "block_gain_mean": matrices_np.mean(axis=0).tolist(),
        "block_gain_max": matrices_np.max(axis=0).tolist(),
        "origin_indices": origins_np.tolist(),
        "lead_indices": leads_np.tolist(),
        "radii": radii_np.tolist(),
        "matrices": matrices_np.tolist(),
        "scope": (
            "Local empirical diagnostic on the audited predicted states; it does not "
            "replace the separate global structural upper bound."
        ),
    }
