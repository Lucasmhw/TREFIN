import torch

from fconcord.certificates import local_block_gain_matrix
from fconcord.model import FCONCORD


def make_model(ablation_variant: str = "Full") -> FCONCORD:
    config = {
        "hidden_dim": 16,
        "lead_dim": 4,
        "graph_embedding_dim": 8,
        "delta": 0.2,
        "gamma_min": 0.08,
        "gamma_max": 0.8,
        "diffusion_max": 0.6,
        "force_max": 0.5,
        "graph_refresh": 0.15,
        "future_graph_gate_center": 0.08,
        "future_graph_gate_temperature": 0.06,
        "q_epsilon": 0.05,
        "nu_min": 2.2,
        "ablation_variant": ablation_variant,
    }
    return FCONCORD(
        num_assets=4,
        num_concepts=7,
        horizon=3,
        directional_mask=torch.tensor([1, 0, 0, 1, 0, 1, 1]),
        risk_indices=[1, 2, 4],
        config=config,
        concept_bound=6.0,
        graph_bound=4.0,
        return_scale=torch.tensor([0.010, 0.015, 0.020, 0.025]),
    )


def test_model_shapes_symmetry_and_positive_definiteness() -> None:
    torch.manual_seed(1)
    model = make_model()
    c0 = torch.randn(2, 4, 7)
    a0 = torch.randn(2, 4, 4)
    a0 = 0.5 * (a0 + a0.transpose(-1, -2))
    a0 = a0 * (1.0 - torch.eye(4))
    s0 = torch.randn(2, 3)
    output = model(c0, a0, s0, num_samples=9)
    assert output["concepts"].shape == (2, 3, 4, 7)
    assert output["concept_path"].shape == (2, 4, 4, 7)
    assert output["graphs"].shape == (2, 3, 4, 4)
    assert output["graph_path"].shape == (2, 4, 4, 4)
    assert output["samples"].shape == (9, 2, 3, 4)
    assert torch.allclose(output["graphs"], output["graphs"].transpose(-1, -2), atol=1e-6)
    assert torch.all(torch.diagonal(output["graphs"], dim1=-2, dim2=-1).abs() < 1e-7)
    assert torch.all(torch.linalg.eigvalsh(output["precision"]) > 0)
    assert torch.all(output["nu"] > 2.0)
    assert torch.isfinite(output["gain_matrix"]).all()
    assert output["gain_radius"] >= 0.0


def test_checkpoint_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(2)
    model = make_model().eval()
    inputs = (torch.randn(2, 4, 7), torch.zeros(2, 4, 4), torch.randn(2, 3))
    with torch.no_grad():
        reference = model(*inputs, num_samples=4)["mu"]
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        prediction = restored(*inputs, num_samples=4)["mu"]
    assert torch.equal(reference, prediction)


def test_derived_concept_mask_is_reconstructed_not_persisted() -> None:
    legacy_compatible_state = make_model("Full").state_dict()
    assert "active_concept_mask" not in legacy_compatible_state
    restored = make_model("NoRiskConcepts")
    restored.load_state_dict(legacy_compatible_state, strict=True)
    torch.testing.assert_close(
        restored.active_concept_mask,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    )


def test_signal_shrinkage_scales_only_the_structural_mean() -> None:
    torch.manual_seed(23)
    model = make_model().eval()
    final_mean_layer = next(
        layer for layer in reversed(model.mu_head) if isinstance(layer, torch.nn.Linear)
    )
    with torch.no_grad():
        final_mean_layer.bias.fill_(0.2)
    inputs = (torch.randn(2, 4, 7), torch.zeros(2, 4, 4), torch.randn(2, 3))
    with torch.no_grad():
        unshrunk = model(*inputs, num_samples=3)["mu"]
        model.signal_shrinkage = 0.25
        shrunk = model(*inputs, num_samples=3)["mu"]
    torch.testing.assert_close(shrunk, 0.25 * unshrunk)


def test_downside_scale_is_conditionally_monotone_in_named_risk() -> None:
    torch.manual_seed(31)
    model = make_model().eval()
    concepts = torch.randn(2, 4, 7)
    graph = torch.zeros(2, 4, 4)
    market = torch.randn(2, 3)
    lead = model._lead(2, 1, concepts.device)
    responses = []
    with torch.no_grad():
        for value in torch.linspace(-3.0, 3.0, 25):
            intervened = concepts.clone()
            intervened[:, :, 1] = value
            responses.append(model._decode_state(intervened, graph, market, lead)["sigma_minus"])
    stacked = torch.stack(responses)
    assert torch.all(torch.diff(stacked, dim=0) >= -1e-8)


def test_asymmetric_student_t_mean_correction() -> None:
    torch.manual_seed(37)
    model = make_model().eval()
    mu = torch.tensor([[[0.004, -0.002, 0.001, 0.0]]])
    sigma_plus = torch.tensor([[[0.010, 0.020, 0.015, 0.025]]])
    sigma_minus = torch.tensor([[[0.030, 0.012, 0.020, 0.010]]])
    nu = torch.tensor([[8.0]])
    correlation = torch.eye(4).view(1, 1, 4, 4)
    with torch.no_grad():
        samples = model._sample_distribution(
            mu, sigma_plus, sigma_minus, nu, correlation, num_samples=50_000
        )
    torch.testing.assert_close(samples.mean(dim=0), mu, atol=7e-4, rtol=0.0)


def test_empirical_certificate_returns_four_nonnegative_blocks() -> None:
    torch.manual_seed(41)
    model = make_model().eval()
    concepts = torch.randn(2, 4, 7)
    graph = torch.zeros(2, 4, 4)
    market = torch.randn(2, 3)
    matrices, radii = local_block_gain_matrix(
        model, concepts, graph, market, lead_index=0, iterations=3
    )
    assert matrices.shape == (2, 2, 2)
    assert radii.shape == (2,)
    assert torch.isfinite(matrices).all()
    assert torch.all(matrices >= 0.0)
    assert torch.all(radii >= 0.0)


def test_no_graph_ablation_removes_initial_future_and_decoder_graph() -> None:
    model = make_model("NoGraph").eval()
    c0 = torch.randn(2, 4, 7)
    a0 = torch.randn(2, 4, 4)
    s0 = torch.randn(2, 3)
    output = model(c0, a0, s0, num_samples=3)
    assert torch.count_nonzero(output["graph_path"]) == 0
    expected = torch.diag_embed(torch.diagonal(output["precision"], dim1=-2, dim2=-1))
    torch.testing.assert_close(output["precision"], expected)


def test_no_market_ablation_is_invariant_to_supplied_market_state() -> None:
    model = make_model("NoMarketState").eval()
    c0 = torch.randn(2, 4, 7)
    a0 = torch.zeros(2, 4, 4)
    first = model(c0, a0, torch.randn(2, 3), num_samples=3)["mu"]
    second = model(c0, a0, torch.randn(2, 3), num_samples=3)["mu"]
    torch.testing.assert_close(first, second)


def test_no_risk_ablation_masks_risk_state_and_supervision() -> None:
    model = make_model("NoRiskConcepts").eval()
    output = model(
        torch.randn(2, 4, 7),
        torch.zeros(2, 4, 4),
        torch.randn(2, 3),
        num_samples=3,
    )
    assert torch.count_nonzero(output["concept_path"][..., [1, 2, 4]]) == 0
    torch.testing.assert_close(
        output["concept_supervision_mask"],
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    )


def test_gaussian_symmetric_ablation_has_one_scale_on_both_sides() -> None:
    model = make_model("GaussianSymmetricDecoder").eval()
    output = model(
        torch.randn(2, 4, 7),
        torch.zeros(2, 4, 4),
        torch.randn(2, 3),
        num_samples=3,
    )
    torch.testing.assert_close(output["sigma_plus"], output["sigma_minus"])
