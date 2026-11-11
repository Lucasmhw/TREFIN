"""Analytic state/mean Lipschitz bounds on explicitly specified rollout tubes.

These are uniform bounds on a local domain, not estimates from sampled
Jacobians and not replacements for the paper's registered-domain constants.
All quantities are evaluated in float64 with a conservative rounding margin.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def spectral(x):
    return torch.linalg.matrix_norm(x, ord=2)


def head_parts(module, k):
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    assert len(layers) == 3
    first = layers[0].weight
    tail = 1.1**2 * spectral(layers[1].weight) * spectral(layers[2].weight)
    return first[:, :k], first[:, k:2*k], tail


def encoder_lipschitz(module, k):
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return (1.1**2 * spectral(layers[0].weight[:, :k])
            * spectral(layers[1].weight) * spectral(layers[2].weight))


def state_constants(model):
    k = model.num_concepts
    n = model.num_assets
    wc, wn, tail = head_parts(model.residual_net, k)
    mwc, mwn, mtail = head_parts(model.mu_head, k)
    c = model.graph_gate_center
    tau = model.graph_gate_temperature
    assert c >= 0 and tau > 0
    # tanh(x) <= x, sech^2(x) <= 1 and sup SiLU' <= 1.1:
    # g'(x) <= sigmoid(u) + u sigmoid'(u) + (c/tau) sigmoid'(u).
    gate = min(1.0 + 1.0/(4.0*tau), 1.1 + c/(4.0*tau))
    return {
        'wc': wc, 'wn': wn, 'tail': tail,
        'mean_wc': mwc, 'mean_wn': mwn, 'mean_tail': mtail,
        'sim_lip': encoder_lipschitz(model.similarity_encoder, k),
        'opp_lip': encoder_lipschitz(model.opposition_encoder, k),
        'gate_lip': gate,
        'lap_diff_lip': math.sqrt((n-1)/2.0) + math.sqrt((n-1)/n),
        'directional': model.directional_mask.bool(),
    }


def typed_graph_cross_bound(c, directional):
    """Uniform norm of A -> typed L(A) C, including changes of edge sign.

    c: [N,K]. Input graph norm and output state norm are Frobenius.
    The incidence proof covers all symmetric, zero-diagonal graph pairs.
    """
    n = c.shape[0]
    cd = c[:, directional]
    cu = c[:, ~directional]
    # Signed incidence: ||B_s|| <= sqrt(2(N-1)); graph edge coordinates
    # have factor 1/sqrt(2). The two largest magnitudes bound |c_i +/- c_j|.
    pair = cd.abs().topk(2, dim=0).values.sum(dim=0)
    width = cu.max(dim=0).values - cu.min(dim=0).values
    incidence = torch.sqrt((n-1)*pair.square().sum() + (n/2)*width.square().sum())
    # Alternative uniform degree/adjacency bound; unsigned columns can be
    # centered because every unsigned Laplacian annihilates constants.
    center = 0.5*(cu.max(dim=0).values + cu.min(dim=0).values)
    cuc = cu-center
    combined = torch.cat([cd, cuc], dim=1)
    adjacency = torch.sqrt(spectral(cd).square()+spectral(cuc).square())
    degree = math.sqrt(n-1)*torch.linalg.vector_norm(combined,dim=1).max()
    return torch.minimum(incidence, degree+adjacency)


def affine_feature_lip(wc, wn, lower, upper):
    # For symmetric A, diagonalize its asset action. The norm reduces to
    # max_lambda ||W_C + lambda W_N||. Convexity gives endpoint bounds.
    return torch.maximum(spectral(wc+lower*wn), spectral(wc+upper*wn))


def tube_bound(model, c, a, s, lead_index, radius_c, radius_a, constants=None):
    """Return a block transition bound and calibrated-mean readout bound.

    Uniform on the intersection of the registered state domain with
    ||C-c||_F <= radius_c and ||A-a||_F <= radius_a, at fixed s and lead.
    Mean bound uses the decoder at lead_index (transition uses the same
    embedding argument; callers provide the appropriate lead convention).
    """
    q = constants or state_constants(model)
    rc, ra = float(radius_c), float(radius_a)
    assert rc >= 0 and ra >= 0
    n = model.num_assets
    lead = model._lead(1,lead_index,c.device)
    gamma = model.gamma_min+(model.gamma_max-model.gamma_min)*torch.sigmoid(model.gamma_net(s[None]))
    diffusion = model.diffusion_max*torch.sigmoid(model.diffusion_net(s[None]))
    d = diffusion.squeeze()
    gamma = gamma.squeeze(0)
    degree = torch.diag(a.abs().sum(dim=-1))
    ls = degree-a
    lu = degree-a.abs()
    es = torch.linalg.eigvalsh(ls)
    eu = torch.linalg.eigvalsh(lu)
    low = torch.where(q['directional'], es[0], eu[0])
    high = torch.where(q['directional'], es[-1], eu[-1])
    linear = torch.maximum((1-model.delta*gamma-model.delta*d*low).abs(),
                           (1-model.delta*gamma-model.delta*d*high).abs()).max()
    linear = linear+model.delta*d*q['lap_diff_lip']*ra
    eig_a = torch.linalg.eigvalsh(a)
    lower,upper = eig_a[0]-ra,eig_a[-1]+ra
    c_feature = affine_feature_lip(q['wc'],q['wn'],lower,upper)
    c_norm = spectral(c)+rc
    lf_c = model.residual_scale*q['tail']*c_feature
    lf_a = model.residual_scale*q['tail']*spectral(q['wn'])*c_norm
    # The same c_N controls signed and unsigned L differences. Adding a
    # C perturbation of Frobenius radius rc therefore adds at most c_N rc.
    lap_c = typed_graph_cross_bound(c,q['directional'])+q['lap_diff_lip']*rc
    mcc = linear+model.delta*lf_c
    mca = model.delta*(d*lap_c+lf_a)
    node = torch.cat([c,s[None].expand(n,-1),lead.expand(n,-1)],dim=-1)
    sim,opp = model.similarity_encoder(node),model.opposition_encoder(node)
    sim_bound = spectral(sim)+q['sim_lip']*rc
    opp_bound = spectral(opp)+q['opp_lip']*rc
    score = 2*(sim_bound*q['sim_lip']+opp_bound*q['opp_lip'])/math.sqrt(model.graph_embedding_dim)
    mac = model.graph_refresh*q['gate_lip']*score
    maa = c.new_tensor(1-model.graph_refresh)
    matrix = torch.stack([torch.stack([mcc,mca]),torch.stack([mac,maa])])
    # The mean's tanh and elementwise train-only return scale are bounded
    # by one and max_i a_i, respectively. Fixed calibration is positive.
    prefactor = float(model.signal_shrinkage)*model.return_scale.max()*model.mean_bound_multiplier*q['mean_tail']
    mean_c = prefactor*affine_feature_lip(q['mean_wc'],q['mean_wn'],lower,upper)
    mean_a = prefactor*spectral(q['mean_wn'])*c_norm
    # Conservative numerical cushion, separate from the analytic inequalities.
    cushion = 1+1e-10
    return matrix*cushion+1e-12, torch.stack([mean_c,mean_a])*cushion+1e-12


def improved_global_bound(model, constants=None):
    """Tighter analytic choices for the SAME registered-domain block bound."""
    q = constants or state_constants(model)
    n = model.num_assets
    b = torch.maximum(model.concept_lower_bound.abs(),model.concept_upper_bound.abs())
    widths = model.concept_upper_bound-model.concept_lower_bound
    bc = math.sqrt(n)*torch.linalg.vector_norm(b)
    aa = math.sqrt((n-1)/n)*model.graph_bound
    lf_c = model.residual_scale*q['tail']*affine_feature_lip(q['wc'],q['wn'],-aa,aa)
    lf_a = model.residual_scale*q['tail']*spectral(q['wn'])*bc
    lap_max = min(q['lap_diff_lip'],math.sqrt(n))*model.graph_bound
    linear = max(abs(1-model.delta*g-model.delta*d*l)
                 for g in [model.gamma_min,model.gamma_max]
                 for d in [0,model.diffusion_max] for l in [0,lap_max])
    cross = torch.sqrt(4*(n-1)*b[q['directional']].square().sum()
                       +(n/2)*widths[~q['directional']].square().sum())
    mcc = linear+model.delta*lf_c
    mca = model.delta*(model.diffusion_max*cross+lf_a)
    market = torch.maximum(model.market_lower_bound.abs(),model.market_upper_bound.abs())
    lead = torch.linalg.vector_norm(model.lead_embedding.weight,dim=1).max()
    node_bound = torch.sqrt(b.square().sum()+market.square().sum()+lead.square())
    sim = math.sqrt(n)*model._network_output_norm_bound(model.similarity_encoder,node_bound)
    opp = math.sqrt(n)*model._network_output_norm_bound(model.opposition_encoder,node_bound)
    score = 2*(sim*q['sim_lip']+opp*q['opp_lip'])/math.sqrt(model.graph_embedding_dim)
    mac = model.graph_refresh*q['gate_lip']*score
    maa = b.new_tensor(1-model.graph_refresh)
    matrix=torch.stack([torch.stack([mcc,mca]),torch.stack([mac,maa])])
    return matrix*(1+1e-10)+1e-12
