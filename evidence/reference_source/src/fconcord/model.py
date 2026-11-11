from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _batch_frobenius_max(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=1).max()


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, final_tanh: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def signed_laplacian(graph: torch.Tensor) -> torch.Tensor:
    degree = graph.abs().sum(dim=-1)
    return torch.diag_embed(degree) - graph


def unsigned_laplacian(graph: torch.Tensor) -> torch.Tensor:
    absolute = graph.abs()
    degree = absolute.sum(dim=-1)
    return torch.diag_embed(degree) - absolute


def typed_graph_operator(
    concepts: torch.Tensor, graph: torch.Tensor, directional_mask: torch.Tensor
) -> torch.Tensor:
    signed = signed_laplacian(graph) @ concepts
    unsigned = unsigned_laplacian(graph) @ concepts
    mask = directional_mask.view(1, 1, -1).to(concepts)
    return signed * mask + unsigned * (1.0 - mask)


def project_graph(graph: torch.Tensor, bound: float) -> torch.Tensor:
    graph = 0.5 * (graph + graph.transpose(-1, -2))
    num_assets = graph.shape[-1]
    eye = torch.eye(num_assets, device=graph.device, dtype=graph.dtype)
    graph = graph * (1.0 - eye)
    norm = torch.linalg.matrix_norm(graph, ord="fro", dim=(-2, -1), keepdim=True)
    scale = torch.clamp(bound / norm.clamp_min(1e-8), max=1.0)
    return graph * scale


class FCONCORD(nn.Module):
    """Concept-graph closed-loop dynamics with an asymmetric Student-t decoder."""

    def __init__(
        self,
        num_assets: int,
        num_concepts: int,
        horizon: int,
        directional_mask: torch.Tensor,
        risk_indices: list[int],
        config: dict[str, Any],
        concept_bound: float,
        graph_bound: float,
        return_scale: torch.Tensor | None = None,
        concept_lower_bound: torch.Tensor | None = None,
        concept_upper_bound: torch.Tensor | None = None,
        market_lower_bound: torch.Tensor | None = None,
        market_upper_bound: torch.Tensor | None = None,
        concept_loss_scale: torch.Tensor | None = None,
        graph_loss_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.num_concepts = num_concepts
        self.horizon = horizon
        self.graph_bound = float(graph_bound)
        self.delta = float(config["delta"])
        self.gamma_min = float(config["gamma_min"])
        self.gamma_max = float(config["gamma_max"])
        self.diffusion_max = float(config["diffusion_max"])
        self.force_max = float(config["force_max"])
        self.graph_refresh = float(config["graph_refresh"])
        self.graph_gate_center = float(config["future_graph_gate_center"])
        self.graph_gate_temperature = float(config["future_graph_gate_temperature"])
        self.q_epsilon = float(config["q_epsilon"])
        self.nu_min = float(config["nu_min"])
        self.mean_bound_multiplier = float(config.get("mean_bound_multiplier", 5.0))
        self.signal_shrinkage = float(config.get("signal_shrinkage", 1.0))
        if not 0.0 < self.signal_shrinkage <= 1.0:
            raise ValueError("signal_shrinkage must lie in (0, 1]")
        self.scale_floor = float(config.get("scale_floor", 0.05))
        self.risk_strength_max = float(config.get("risk_strength_max", 1.0))
        self.hidden_dim = int(config["hidden_dim"])
        self.lead_dim = int(config["lead_dim"])
        self.graph_embedding_dim = int(config["graph_embedding_dim"])
        self.residual_scale = float(config.get("residual_scale", 0.02))
        self.ablation_variant = str(config.get("ablation_variant", "Full"))
        valid_variants = {
            "Full",
            "NoGraph",
            "NoMarketState",
            "NoRiskConcepts",
            "NoForwardConceptSupervision",
            "GaussianSymmetricDecoder",
        }
        if self.ablation_variant not in valid_variants:
            raise ValueError(f"Unknown ablation variant: {self.ablation_variant}")
        self.use_graph = self.ablation_variant != "NoGraph"
        self.use_market_state = self.ablation_variant != "NoMarketState"
        self.use_risk_concepts = self.ablation_variant != "NoRiskConcepts"
        self.gaussian_symmetric_decoder = (
            self.ablation_variant == "GaussianSymmetricDecoder"
        )

        self.register_buffer("directional_mask", directional_mask.float())
        if concept_lower_bound is None:
            concept_lower_bound = torch.full((num_concepts,), -float(concept_bound))
        if concept_upper_bound is None:
            concept_upper_bound = torch.full((num_concepts,), float(concept_bound))
        if (
            concept_lower_bound.shape != (num_concepts,)
            or concept_upper_bound.shape != (num_concepts,)
            or torch.any(concept_lower_bound >= concept_upper_bound)
        ):
            raise ValueError("Concept bounds must be ordered vectors of length num_concepts")
        self.register_buffer("concept_lower_bound", concept_lower_bound.float())
        self.register_buffer("concept_upper_bound", concept_upper_bound.float())
        self.concept_bound = float(
            torch.maximum(concept_lower_bound.abs(), concept_upper_bound.abs()).max()
        )
        if market_lower_bound is None:
            market_lower_bound = torch.full((3,), -float(concept_bound))
        if market_upper_bound is None:
            market_upper_bound = torch.full((3,), float(concept_bound))
        self.register_buffer("market_lower_bound", market_lower_bound.float())
        self.register_buffer("market_upper_bound", market_upper_bound.float())
        risk_mask = torch.zeros(num_concepts, dtype=torch.float32)
        risk_mask[risk_indices] = 1.0
        self.register_buffer("risk_mask", risk_mask)
        active_concept_mask = (
            torch.ones(num_concepts, dtype=torch.float32)
            if self.use_risk_concepts
            else 1.0 - risk_mask
        )
        # This mask is a deterministic function of ``ablation_variant`` and
        # ``risk_indices``.  Keeping it out of state_dict preserves strict
        # compatibility with pre-ablation F-CONCORD checkpoints while the
        # factory still reconstructs the exact mask on every reload.
        self.register_buffer("active_concept_mask", active_concept_mask, persistent=False)
        self.register_buffer("risk_indices", torch.tensor(risk_indices, dtype=torch.long))
        if return_scale is None:
            return_scale = torch.ones(num_assets, dtype=torch.float32)
        if return_scale.shape != (num_assets,) or torch.any(return_scale <= 0):
            raise ValueError("return_scale must contain one positive value per asset")
        self.register_buffer("return_scale", return_scale.float())
        if concept_loss_scale is None:
            concept_loss_scale = torch.ones(num_concepts, dtype=torch.float32)
        if concept_loss_scale.shape != (num_concepts,) or torch.any(concept_loss_scale <= 0):
            raise ValueError("concept_loss_scale must be positive for each concept")
        self.register_buffer("concept_loss_scale", concept_loss_scale.float())
        if graph_loss_scale <= 0.0:
            raise ValueError("graph_loss_scale must be positive")
        self.register_buffer(
            "graph_loss_scale", torch.tensor(float(graph_loss_scale), dtype=torch.float32)
        )

        self.lead_embedding = nn.Embedding(horizon + 1, self.lead_dim)
        self.gamma_net = _mlp(3, self.hidden_dim, num_concepts)
        self.diffusion_net = _mlp(3, self.hidden_dim, 1)
        self.force_net = _mlp(3, self.hidden_dim, num_concepts)
        local_dim = 2 * num_concepts + 3 + self.lead_dim
        self.residual_net = _mlp(local_dim, self.hidden_dim, num_concepts, final_tanh=True)

        graph_input_dim = num_concepts + 3 + self.lead_dim
        self.similarity_encoder = _mlp(
            graph_input_dim, self.hidden_dim, self.graph_embedding_dim
        )
        self.opposition_encoder = _mlp(
            graph_input_dim, self.hidden_dim, self.graph_embedding_dim
        )

        decoder_dim = 2 * num_concepts + 3 + self.lead_dim
        self.mu_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.up_scale_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.down_scale_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.q_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.lambda_q_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.nu_head = _mlp(decoder_dim, self.hidden_dim, 1)
        self.risk_weights_raw = nn.Parameter(torch.zeros(len(risk_indices)))
        self.risk_strength_raw = nn.Parameter(torch.zeros(()))
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        """Start from a calibrated no-change forecast without freezing learning."""
        for head in (self.mu_head, self.up_scale_head, self.down_scale_head):
            final = next(layer for layer in reversed(head) if isinstance(layer, nn.Linear))
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _lead(self, batch_size: int, h: int, device: torch.device) -> torch.Tensor:
        indices = torch.full((batch_size,), h, device=device, dtype=torch.long)
        return self.lead_embedding(indices)

    def _local_features(
        self, concepts: torch.Tensor, graph: torch.Tensor, market: torch.Tensor, lead: torch.Tensor
    ) -> torch.Tensor:
        batch, num_assets, _ = concepts.shape
        market_nodes = market[:, None, :].expand(batch, num_assets, -1)
        lead_nodes = lead[:, None, :].expand(batch, num_assets, -1)
        neighborhood = graph @ concepts
        return torch.cat([concepts, neighborhood, market_nodes, lead_nodes], dim=-1)

    def _graph_proposal(
        self, concepts: torch.Tensor, market: torch.Tensor, lead: torch.Tensor
    ) -> torch.Tensor:
        batch, num_assets, _ = concepts.shape
        node_input = torch.cat(
            [
                concepts,
                market[:, None, :].expand(batch, num_assets, -1),
                lead[:, None, :].expand(batch, num_assets, -1),
            ],
            dim=-1,
        )
        similarity = self.similarity_encoder(node_input)
        opposition = self.opposition_encoder(node_input)
        scale = math.sqrt(self.graph_embedding_dim)
        score = (
            similarity @ similarity.transpose(-1, -2)
            - opposition @ opposition.transpose(-1, -2)
        ) / scale
        gate = torch.sigmoid(
            (score.abs() - self.graph_gate_center) / self.graph_gate_temperature
        )
        proposal = torch.tanh(score) * gate
        eye = torch.eye(num_assets, device=concepts.device, dtype=concepts.dtype)
        return proposal * (1.0 - eye)

    def _step(
        self,
        concepts: torch.Tensor,
        graph: torch.Tensor,
        market: torch.Tensor,
        lead: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_graph:
            graph = torch.zeros_like(graph)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
            self.gamma_net(market)
        )
        diffusion = self.diffusion_max * torch.sigmoid(self.diffusion_net(market))
        force = self.force_max * torch.tanh(self.force_net(market))
        typed = typed_graph_operator(concepts, graph, self.directional_mask)
        local = self._local_features(concepts, graph, market, lead)
        nonlinear = self.residual_scale * self.residual_net(local)
        concept_next = (
            concepts
            - self.delta * concepts * gamma[:, None, :]
            - self.delta * diffusion[:, None, :] * typed
            + self.delta * force[:, None, :]
            + self.delta * nonlinear
        )
        concept_next = torch.maximum(
            torch.minimum(concept_next, self.concept_upper_bound.view(1, 1, -1)),
            self.concept_lower_bound.view(1, 1, -1),
        )
        concept_next = concept_next * self.active_concept_mask.view(1, 1, -1)

        if self.use_graph:
            proposal = self._graph_proposal(concepts, market, lead)
            graph_next = (1.0 - self.graph_refresh) * graph + self.graph_refresh * proposal
            graph_next = project_graph(graph_next, self.graph_bound)
        else:
            graph_next = torch.zeros_like(graph)
        return concept_next, graph_next

    def _decoder_features(
        self, concepts: torch.Tensor, graph: torch.Tensor, market: torch.Tensor, lead: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_assets, _ = concepts.shape
        risk_mask = self.risk_mask.view(1, 1, -1)
        concepts_free = concepts * (1.0 - risk_mask)
        neighborhood = graph @ concepts
        neighborhood_free = neighborhood * (1.0 - risk_mask)
        market_nodes = market[:, None, :].expand(batch, num_assets, -1)
        lead_nodes = lead[:, None, :].expand(batch, num_assets, -1)
        free_features = torch.cat(
            [concepts_free, neighborhood_free, market_nodes, lead_nodes], dim=-1
        )
        full_features = torch.cat([concepts, neighborhood, market_nodes, lead_nodes], dim=-1)
        return full_features, free_features

    def _decode_state(
        self, concepts: torch.Tensor, graph: torch.Tensor, market: torch.Tensor, lead: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        full, free = self._decoder_features(concepts, graph, market, lead)
        asset_scale = self.return_scale.view(1, -1)
        structural_mean = asset_scale * self.mean_bound_multiplier * torch.tanh(
            self.mu_head(full).squeeze(-1)
        )
        mu = self.signal_shrinkage * structural_mean
        sigma_plus = asset_scale * (
            self.scale_floor + F.softplus(self.up_scale_head(free).squeeze(-1))
        )
        risk_values = concepts.index_select(-1, self.risk_indices)
        risk_weights = (
            self.risk_strength_max
            * torch.sigmoid(self.risk_strength_raw)
            * torch.softmax(self.risk_weights_raw, dim=0)
        )
        risk_signal = (risk_values * risk_weights.view(1, 1, -1)).sum(dim=-1)
        if self.gaussian_symmetric_decoder:
            sigma_minus = sigma_plus
        else:
            sigma_minus = asset_scale * (
                self.scale_floor
                + F.softplus(self.down_scale_head(free).squeeze(-1) + risk_signal)
            )
        q = self.q_epsilon + F.softplus(self.q_head(free).squeeze(-1))
        pooled = free.mean(dim=1)
        lambda_q = 0.01 + F.softplus(self.lambda_q_head(pooled).squeeze(-1))
        nu = self.nu_min + F.softplus(self.nu_head(pooled).squeeze(-1))
        precision = torch.diag_embed(q) + lambda_q[:, None, None] * signed_laplacian(graph)
        eye = torch.eye(self.num_assets, device=concepts.device, dtype=concepts.dtype)
        precision = precision + self.q_epsilon * eye
        covariance = torch.linalg.inv(precision)
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-8)
        correlation = covariance / torch.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
        correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
        correlation = correlation + 1e-5 * eye
        return {
            "mu": mu,
            "sigma_plus": sigma_plus,
            "sigma_minus": sigma_minus,
            "nu": nu,
            "precision": precision,
            "correlation": correlation,
        }

    def _sample_distribution(
        self,
        mu: torch.Tensor,
        sigma_plus: torch.Tensor,
        sigma_minus: torch.Tensor,
        nu: torch.Tensor,
        correlation: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        batch, horizon, num_assets = mu.shape
        eye = torch.eye(num_assets, device=mu.device, dtype=mu.dtype)
        cholesky = torch.linalg.cholesky(correlation + 1e-5 * eye)
        z = torch.randn(num_samples, batch, horizon, num_assets, device=mu.device, dtype=mu.dtype)
        correlated = torch.einsum("bhij,sbhj->sbhi", cholesky, z)
        if self.gaussian_symmetric_decoder:
            return mu[None] + sigma_plus[None] * correlated
        gamma = torch.distributions.Gamma(nu / 2.0, torch.full_like(nu, 0.5))
        chi = gamma.rsample((num_samples,)).clamp_min(1e-8)
        xi = torch.sqrt((nu[None, :, :, None] - 2.0) / chi[:, :, :, None]) * correlated
        log_kappa = (
            0.5 * torch.log(nu - 2.0)
            + torch.lgamma((nu - 1.0) / 2.0)
            - math.log(2.0 * math.sqrt(math.pi))
            - torch.lgamma(nu / 2.0)
        )
        kappa = torch.exp(log_kappa)
        samples = (
            mu[None]
            + sigma_plus[None] * F.relu(xi)
            - sigma_minus[None] * F.relu(-xi)
            - kappa[None, :, :, None] * (sigma_plus - sigma_minus)[None]
        )
        return samples

    def _spectral_product(self, module: nn.Module) -> torch.Tensor:
        product = torch.ones((), device=next(self.parameters()).device)
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                product = product * torch.linalg.matrix_norm(layer.weight, ord=2)
        return product

    def _network_output_norm_bound(self, module: nn.Module, input_norm: torch.Tensor) -> torch.Tensor:
        """Conservative Euclidean output bound using spectral norms and biases."""
        bound = input_norm
        for layer in module:
            if isinstance(layer, nn.Linear):
                bound = torch.linalg.matrix_norm(layer.weight, ord=2) * bound
                if layer.bias is not None:
                    bound = bound + torch.linalg.vector_norm(layer.bias)
            elif isinstance(layer, nn.SiLU):
                # |SiLU(x)| <= |x| coordinatewise.
                continue
            elif isinstance(layer, nn.Tanh):
                bound = torch.minimum(
                    bound,
                    torch.sqrt(torch.tensor(layer.out_features if hasattr(layer, "out_features") else self.num_concepts, device=bound.device, dtype=bound.dtype)),
                )
        return bound

    def structural_gain_bound(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-parameter structural small-gain upper bound from the manual.

        The bound is global over the registered concept/graph/market boxes and
        all lead embeddings.  It is deliberately conservative; unlike a local
        Jacobian diagnostic, a value below one can be used with the system
        theorem's block-Lipschitz assumption.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        silu_lipschitz = torch.tensor(1.1, device=device, dtype=dtype)

        residual_lipschitz = self._spectral_product(self.residual_net) * silu_lipschitz**2
        concept_abs = torch.maximum(
            self.concept_lower_bound.abs(), self.concept_upper_bound.abs()
        )
        b_c = math.sqrt(self.num_assets) * torch.linalg.vector_norm(concept_abs)
        local_c = math.sqrt(1.0 + self.graph_bound**2)
        l_f_c = self.residual_scale * residual_lipschitz * local_c
        l_f_a = self.residual_scale * residual_lipschitz * b_c

        laplacian_max = 2.0 * math.sqrt(max(1, self.num_assets - 1)) * self.graph_bound
        linear_candidates = torch.stack(
            [
                torch.as_tensor(abs(1.0 - self.delta * self.gamma_min), device=device, dtype=dtype),
                torch.as_tensor(abs(1.0 - self.delta * self.gamma_max), device=device, dtype=dtype),
                torch.as_tensor(
                    abs(
                        1.0
                        - self.delta * self.gamma_min
                        - self.delta * self.diffusion_max * laplacian_max
                    ),
                    device=device,
                    dtype=dtype,
                ),
                torch.as_tensor(
                    abs(
                        1.0
                        - self.delta * self.gamma_max
                        - self.delta * self.diffusion_max * laplacian_max
                    ),
                    device=device,
                    dtype=dtype,
                ),
            ]
        )
        m_cc = linear_candidates.max() + self.delta * l_f_c
        l_laplacian_a = math.sqrt(max(1, self.num_assets - 1)) + 1.0
        m_ca = self.delta * (
            self.diffusion_max * l_laplacian_a * b_c + l_f_a
        )

        market_abs = torch.maximum(
            self.market_lower_bound.abs(), self.market_upper_bound.abs()
        )
        lead_bound = torch.linalg.vector_norm(self.lead_embedding.weight, dim=1).max()
        node_input_bound = torch.sqrt(
            torch.linalg.vector_norm(concept_abs) ** 2
            + torch.linalg.vector_norm(market_abs) ** 2
            + lead_bound**2
        )
        sim_bound = math.sqrt(self.num_assets) * self._network_output_norm_bound(
            self.similarity_encoder, node_input_bound
        )
        opp_bound = math.sqrt(self.num_assets) * self._network_output_norm_bound(
            self.opposition_encoder, node_input_bound
        )
        sim_lipschitz = self._spectral_product(self.similarity_encoder) * silu_lipschitz**2
        opp_lipschitz = self._spectral_product(self.opposition_encoder) * silu_lipschitz**2
        score_lipschitz = (
            2.0 * sim_bound * sim_lipschitz
            + 2.0 * opp_bound * opp_lipschitz
        ) / math.sqrt(self.graph_embedding_dim)
        gate_lipschitz = 1.0 + 1.0 / (4.0 * self.graph_gate_temperature)
        l_g_c = gate_lipschitz * score_lipschitz
        m_ac = self.graph_refresh * l_g_c
        m_aa = torch.as_tensor(
            1.0 - self.graph_refresh, device=device, dtype=dtype
        )
        matrix = torch.stack([torch.stack([m_cc, m_ca]), torch.stack([m_ac, m_aa])])
        trace = m_cc + m_aa
        discriminant = torch.sqrt((m_cc - m_aa) ** 2 + 4.0 * m_ca * m_ac)
        radius = 0.5 * (trace + discriminant)
        return matrix, radius

    def batch_gain_bound(
        self, concepts: torch.Tensor, graph: torch.Tensor, market: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable state-conditional block bound used by the gain loss.

        This is the manual's current-batch ``rho-hat`` object.  It uses the
        actually visited state norms and exact spectra of the dissipative
        linear concept operator.  The separate ``structural_gain_bound`` stays
        global over the registered state sets and is never replaced by this
        tighter training diagnostic.
        """
        silu_lipschitz = torch.as_tensor(1.1, device=concepts.device, dtype=concepts.dtype)
        residual_lipschitz = self._spectral_product(self.residual_net) * silu_lipschitz**2
        graph_spectral = torch.linalg.matrix_norm(graph, ord=2).max()
        b_c = _batch_frobenius_max(concepts)
        l_f_c = self.residual_scale * residual_lipschitz * torch.sqrt(
            1.0 + graph_spectral**2
        )
        l_f_a = self.residual_scale * residual_lipschitz * b_c

        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
            self.gamma_net(market)
        )
        diffusion = self.diffusion_max * torch.sigmoid(self.diffusion_net(market))
        signed = signed_laplacian(graph)
        unsigned = unsigned_laplacian(graph)
        selected_laplacian = torch.where(
            self.directional_mask.view(1, -1, 1, 1).bool(),
            signed[:, None],
            unsigned[:, None],
        )
        eye = torch.eye(self.num_assets, device=concepts.device, dtype=concepts.dtype)
        linear = (
            eye.view(1, 1, self.num_assets, self.num_assets)
            * (1.0 - self.delta * gamma)[:, :, None, None]
            - self.delta
            * diffusion[:, None, :, None]
            * selected_laplacian
        )
        linear_norm = torch.linalg.eigvalsh(linear).abs().amax()
        m_cc = linear_norm + self.delta * l_f_c
        if not self.use_graph:
            zero = torch.zeros((), device=concepts.device, dtype=concepts.dtype)
            matrix = torch.stack(
                [torch.stack([m_cc, zero]), torch.stack([zero, zero])]
            )
            return matrix, m_cc
        l_laplacian_a = math.sqrt(max(1, self.num_assets - 1)) + 1.0
        m_ca = self.delta * (
            diffusion.max() * l_laplacian_a * b_c + l_f_a
        )

        batch = concepts.shape[0]
        lead_bounds = self.lead_embedding.weight
        # The graph proposal at a training state may be called at any known
        # lead, so take the maximum across the finite registered embeddings.
        lead_norm = torch.linalg.vector_norm(lead_bounds, dim=1).max()
        market_nodes = market[:, None, :].expand(batch, self.num_assets, -1)
        representative_lead = lead_bounds[
            torch.linalg.vector_norm(lead_bounds, dim=1).argmax()
        ]
        lead_nodes = representative_lead.view(1, 1, -1).expand(
            batch, self.num_assets, -1
        )
        del lead_norm
        node_input = torch.cat([concepts, market_nodes, lead_nodes], dim=-1)
        similarity = self.similarity_encoder(node_input)
        opposition = self.opposition_encoder(node_input)
        sim_bound = _batch_frobenius_max(similarity)
        opp_bound = _batch_frobenius_max(opposition)
        sim_lipschitz = self._spectral_product(self.similarity_encoder) * silu_lipschitz**2
        opp_lipschitz = self._spectral_product(self.opposition_encoder) * silu_lipschitz**2
        score_lipschitz = (
            2.0 * sim_bound * sim_lipschitz
            + 2.0 * opp_bound * opp_lipschitz
        ) / math.sqrt(self.graph_embedding_dim)
        gate_lipschitz = 1.0 + 1.0 / (4.0 * self.graph_gate_temperature)
        m_ac = self.graph_refresh * gate_lipschitz * score_lipschitz
        m_aa = torch.as_tensor(
            1.0 - self.graph_refresh, device=concepts.device, dtype=concepts.dtype
        )
        matrix = torch.stack([torch.stack([m_cc, m_ca]), torch.stack([m_ac, m_aa])])
        discriminant = torch.sqrt((m_cc - m_aa) ** 2 + 4.0 * m_ca * m_ac)
        radius = 0.5 * (m_cc + m_aa + discriminant)
        return matrix, radius

    def gain_proxy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible name for the structural certificate bound."""
        return self.structural_gain_bound()

    def forward(
        self,
        c0: torch.Tensor,
        a0: torch.Tensor,
        s0: torch.Tensor,
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        if not self.use_market_state:
            s0 = torch.zeros_like(s0)
        concepts = c0 * self.active_concept_mask.view(1, 1, -1)
        graph = a0 if self.use_graph else torch.zeros_like(a0)
        sanitized_c0, sanitized_a0 = concepts, graph
        concept_path, graph_path = [], []
        decoded: list[dict[str, torch.Tensor]] = []
        for h in range(self.horizon):
            update_lead = self._lead(c0.shape[0], h, c0.device)
            concepts, graph = self._step(concepts, graph, s0, update_lead)
            concept_path.append(concepts)
            graph_path.append(graph)
            decode_lead = self._lead(c0.shape[0], h + 1, c0.device)
            decoded.append(self._decode_state(concepts, graph, s0, decode_lead))
        outputs = {
            key: torch.stack([item[key] for item in decoded], dim=1)
            for key in decoded[0]
        }
        outputs["concepts"] = torch.stack(concept_path, dim=1)
        outputs["graphs"] = torch.stack(graph_path, dim=1)
        outputs["concept_path"] = torch.cat(
            [sanitized_c0[:, None], outputs["concepts"]], dim=1
        )
        outputs["graph_path"] = torch.cat(
            [sanitized_a0[:, None], outputs["graphs"]], dim=1
        )
        outputs["samples"] = self._sample_distribution(
            outputs["mu"],
            outputs["sigma_plus"],
            outputs["sigma_minus"],
            outputs["nu"],
            outputs["correlation"],
            num_samples,
        )
        gain_matrix, gain_radius = self.batch_gain_bound(
            sanitized_c0, sanitized_a0, s0
        )
        outputs["gain_matrix"] = gain_matrix
        outputs["gain_radius"] = gain_radius
        outputs["concept_loss_scale"] = self.concept_loss_scale
        outputs["concept_supervision_mask"] = self.active_concept_mask
        outputs["graph_loss_scale"] = self.graph_loss_scale
        outputs["risk_slopes"] = (
            self.risk_strength_max
            * torch.sigmoid(self.risk_strength_raw)
            * torch.softmax(self.risk_weights_raw, dim=0)
        )
        return outputs
