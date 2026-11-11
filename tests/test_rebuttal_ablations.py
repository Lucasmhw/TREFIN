"""Tests for isolated reviewer-requested structural variants.

Deploy alongside the patched fconcord/model.py.  Only transition propagation is
changed by SingleSigned/SingleUnsigned; decoder structure stays identical.
"""

import pytest
import torch

from fconcord.losses import total_loss
from fconcord.model import FCONCORD


def make_model(variant="Full", *, assets=2, **updates):
    config = {
        "hidden_dim": 8,
        "lead_dim": 3,
        "graph_embedding_dim": 4,
        "delta": 0.2,
        "gamma_min": 0.1,
        "gamma_max": 0.3,
        "diffusion_max": 0.4,
        "force_max": 0.3,
        "graph_refresh": 0.1,
        "future_graph_gate_center": 0.1,
        "future_graph_gate_temperature": 0.05,
        "q_epsilon": 0.02,
        "nu_min": 2.1,
        "residual_scale": 0.01,
        "ablation_variant": variant,
    }
    config.update(updates)
    return FCONCORD(
        num_assets=assets,
        num_concepts=2,
        horizon=2,
        directional_mask=torch.tensor([1.0, 0.0]),
        risk_indices=[1],
        config=config,
        concept_bound=10.0,
        graph_bound=4.0,
        return_scale=torch.ones(assets) * 0.02,
    )


@pytest.mark.parametrize(
    "variant, expected_diffusion",
    [
        ("Full", [[1.0, 0.0], [1.0, 0.0]]),
        ("SingleSigned", [[1.0, 2.0], [1.0, 2.0]]),
        ("SingleUnsigned", [[0.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_negative_edge_has_expected_coordinate_semantics(variant, expected_diffusion):
    model = make_model(variant)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    concepts = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    graph = torch.tensor([[[0.0, -0.5], [-0.5, 0.0]]])
    market = torch.zeros(1, 3)
    lead = model._lead(1, 0, concepts.device)
    next_concepts, next_graph = model._step(concepts, graph, market, lead)
    # Zero neural weights make gamma=0.2, diffusion=0.2, force=residual=0.
    expected = concepts * 0.96 - 0.04 * torch.tensor([expected_diffusion])
    torch.testing.assert_close(next_concepts, expected)
    # Zero graph proposal, so every variant preserves the same graph update.
    torch.testing.assert_close(next_graph, 0.9 * graph)


@pytest.mark.parametrize("variant", ["SingleSigned", "SingleUnsigned", "NoStateSupervision"])
def test_variants_preserve_decoder_and_conditional_risk_guarantee(variant):
    torch.manual_seed(612)
    full = make_model().eval()
    changed = make_model(variant).eval()
    changed.load_state_dict(full.state_dict(), strict=True)
    concepts = torch.tensor([[[0.4, -0.7], [0.2, 1.1]]], requires_grad=True)
    graph = torch.tensor([[[0.0, -0.6], [-0.6, 0.0]]])
    market = torch.tensor([[0.1, -0.3, 0.4]])
    lead = full._lead(1, 1, concepts.device)
    original = full._decode_state(concepts, graph, market, lead)
    ablated = changed._decode_state(concepts, graph, market, lead)
    for name in original:
        torch.testing.assert_close(ablated[name], original[name], rtol=0, atol=0)
    # The signed precision's off-diagonal is positive for a negative edge.
    assert torch.all(ablated["precision"][:, 0, 1] > 0)
    assert torch.linalg.eigvalsh(ablated["precision"]).min() > 0
    slope = torch.autograd.grad(ablated["sigma_minus"].sum(), concepts)[0]
    assert torch.all(slope[:, :, 1] > 0)


@pytest.mark.parametrize("variant, expected", [("SingleSigned", 3.1), ("SingleUnsigned", 2.1)])
def test_gain_regularizer_uses_same_transition_operator(variant, expected):
    model = make_model(
        variant, assets=3, delta=1.0, gamma_min=0.05, gamma_max=0.15,
        diffusion_max=2.0, residual_scale=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # The all-negative triangle is unbalanced: signed Laplacian eigenvalues
    # are 1,1,4; unsigned are 0,3,3. gamma=.1 and diffusion=1.
    graph = (torch.eye(3) - torch.ones(3, 3))[None]
    matrix, _ = model.batch_gain_bound(torch.ones(1, 3, 2), graph, torch.zeros(1, 3))
    torch.testing.assert_close(matrix[0, 0], torch.tensor(expected), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("variant", ["SingleSigned", "SingleUnsigned", "NoStateSupervision"])
def test_strict_checkpoint_reload_reconstructs_transition_mask(variant):
    model = make_model(variant)
    original = make_model("Full").state_dict()
    assert set(model.state_dict()) == set(original)
    assert "transition_directional_mask" not in original
    model.load_state_dict(original, strict=True)
    torch.testing.assert_close(model.directional_mask, torch.tensor([1.0, 0.0]))
    expected = {
        "SingleSigned": [1.0, 1.0],
        "SingleUnsigned": [0.0, 0.0],
        "NoStateSupervision": [1.0, 0.0],
    }[variant]
    torch.testing.assert_close(model.transition_directional_mask, torch.tensor(expected))


def test_no_state_supervision_leaves_forward_map_unchanged_and_removes_label_gradients():
    torch.manual_seed(182)
    full = make_model("Full")
    ablated = make_model("NoStateSupervision")
    ablated.load_state_dict(full.state_dict(), strict=True)
    concepts = torch.randn(2, 2, 2)
    graph = torch.tensor([[[0.0, -0.4], [-0.4, 0.0]]]).expand(2, -1, -1)
    market = torch.randn(2, 3)
    torch.manual_seed(85)
    original = full(concepts, graph, market, num_samples=4)
    torch.manual_seed(85)
    output = ablated(concepts, graph, market, num_samples=4)
    for name in original:
        torch.testing.assert_close(output[name], original[name], rtol=0, atol=0)
    training = {
        "lambda_concept": 0.0,
        "lambda_graph": 0.0,
        "lambda_monotonicity": 0.01,
        "lambda_gain": 0.1,
        "gain_target": 0.995,
    }
    batch = {
        "y": torch.randn(2, 2, 2) * 0.02,
        "c_target": torch.randn_like(output["concept_path"], requires_grad=True),
        "a_target": torch.randn_like(output["graph_path"], requires_grad=True),
    }
    loss, _ = total_loss(output, batch, training, ablated.risk_indices)
    grads = torch.autograd.grad(loss, (batch["c_target"], batch["a_target"]))
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)
    alternate_batch = dict(batch)
    alternate_batch["c_target"] = batch["c_target"].detach() + 100
    alternate_batch["a_target"] = batch["a_target"].detach() - 100
    alternate_loss, _ = total_loss(output, alternate_batch, training, ablated.risk_indices)
    torch.testing.assert_close(loss, alternate_loss, rtol=0, atol=0)
