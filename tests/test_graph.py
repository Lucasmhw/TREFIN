import numpy as np
import torch

from fconcord.graph import estimate_partial_correlation, smooth_signed_graph
from fconcord.model import signed_laplacian, typed_graph_operator, unsigned_laplacian


def test_partial_correlation_and_graph_are_symmetric() -> None:
    rng = np.random.default_rng(5)
    residual = rng.normal(size=(120, 5))
    residual[:, 1] += 0.7 * residual[:, 0]
    residual = (residual - residual.mean(0)) / residual.std(0)
    weights = np.ones(120) / 120
    _, precision, partial = estimate_partial_correlation(residual, weights, alpha=0.05)
    graph = smooth_signed_graph(partial, center=0.05, temperature=0.03)
    np.testing.assert_allclose(precision, precision.T, atol=1e-8)
    np.testing.assert_allclose(graph, graph.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(graph), 0.0)


def test_graph_solver_settings_are_audited() -> None:
    rng = np.random.default_rng(13)
    residual = rng.normal(size=(80, 4))
    residual = (residual - residual.mean(0)) / residual.std(0)
    weights = np.ones(80) / 80
    *_, diagnostics = estimate_partial_correlation(
        residual,
        weights,
        alpha=0.1,
        return_diagnostics=True,
        max_iter=321,
        tolerance=2e-4,
    )
    assert diagnostics["solver_max_iter"] == 321
    assert diagnostics["solver_tolerance"] == 2e-4


def test_typed_laplacian_respects_direction_and_magnitude_semantics() -> None:
    graph = torch.tensor([[[0.0, -1.0], [-1.0, 0.0]]])
    concepts = torch.tensor([[[1.0, 1.0], [-1.0, 3.0]]])
    result = typed_graph_operator(concepts, graph, torch.tensor([1.0, 0.0]))
    assert torch.allclose(result[0, :, 0], torch.zeros(2))
    assert torch.allclose(result[0, :, 1], torch.tensor([-2.0, 2.0]))
    for laplacian in [signed_laplacian(graph), unsigned_laplacian(graph)]:
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        assert torch.all(eigenvalues >= -1e-6)
