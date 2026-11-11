"""Explicit uniform local W1 bounds for one 32-asset decoder law.

The same analytic formula covers the ideal law and the implemented
chi-floor/double-jitter law. Its bound is on the kernel's state sensitivity,
not its discrepancy from the unknown true conditional return distribution.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from tube_bounds import spectral,affine_feature_lip
from local_mlp_bounds import scalar_mlp_state_bound


def argument_bounds(module,c,a,rc,ra,mask,pooled=False):
    layers=[x for x in module.modules() if isinstance(x,nn.Linear)]
    k=c.shape[1]
    wc=layers[0].weight[:,:k]*mask
    wn=layers[0].weight[:,k:2*k]*mask
    tail=1.1**2*spectral(layers[1].weight)*spectral(layers[2].weight)
    eigen=torch.linalg.eigvalsh(a)
    lc=tail*affine_feature_lip(wc,wn,eigen[0]-ra,eigen[-1]+ra)
    la=tail*spectral(wn)*(spectral(c*mask)+rc)
    if pooled:lc=lc/math.sqrt(c.shape[0]);la=la/math.sqrt(c.shape[0])
    return torch.stack([lc,la])


def softplus_head(module,features,c,a,rc,ra,mask,offset,pooled=False,extra=None,extra_c_lip=0.,network_bound='local'):
    arg=module(features.mean(1) if pooled else features).squeeze()
    lips=argument_bounds(module,c,a,rc,ra,mask,pooled)
    local_interval=None
    if network_bound=='local':
        local,local_interval=scalar_mlp_state_bound(module,features,c,a,rc,ra,mask,pooled)
        lips=torch.minimum(lips,local)
    if extra is not None:arg=arg+extra
    lips=lips+arg.new_tensor([extra_c_lip,0.])
    radius=lips@arg.new_tensor([rc,ra])
    if local_interval is not None:radius=torch.minimum(radius,local_interval+extra_c_lip*rc)
    lo=offset+F.softplus(arg-radius);hi=offset+F.softplus(arg+radius)
    slope=torch.sigmoid(arg+radius).max()
    return lo,hi,lips*slope,{'argument_center':arg,'argument_radius':radius}


def decoder_w1_bound(model,c,a,s,lead_index,rc,ra,law='implemented',network_bound='local'):
    assert law in ['implemented','ideal']
    assert network_bound in ['local','spectral']
    assert not model.gaussian_symmetric_decoder
    n,k=c.shape
    lead=model._lead(1,lead_index,c.device)
    full,free=model._decoder_features(c[None],a[None],s[None],lead)
    ones=torch.ones(k,dtype=c.dtype,device=c.device)
    mask=1-model.risk_mask
    qlo,qhi,lq,_=softplus_head(model.q_head,free,c,a,rc,ra,mask,model.q_epsilon,network_bound=network_bound)
    llo,lhi,ll,_=softplus_head(model.lambda_q_head,free,c,a,rc,ra,mask,.01,pooled=True,network_bound=network_bound)
    nlo,nhi,ln,_=softplus_head(model.nu_head,free,c,a,rc,ra,mask,model.nu_min,pooled=True,network_bound=network_bound)
    plo,phi,lp,_=softplus_head(model.up_scale_head,free,c,a,rc,ra,mask,model.scale_floor,network_bound=network_bound)
    weights=model.risk_strength_max*torch.sigmoid(model.risk_strength_raw)*torch.softmax(model.risk_weights_raw,dim=0)
    risk=(c[:,model.risk_indices]*weights).sum(-1)
    mlo,mhi,lm,_=softplus_head(model.down_scale_head,free,c,a,rc,ra,mask,model.scale_floor,
                              extra=risk,extra_c_lip=float(weights.norm()),network_bound=network_bound)
    scales=model.return_scale
    plo,phi=plo*scales,phi*scales;mlo,mhi=mlo*scales,mhi*scales
    lp=lp*scales.max();lm=lm*scales.max()
    sigma_max=torch.maximum(phi.max(),mhi.max())
    sigma_difference=torch.maximum((phi-mlo).abs(),(mhi-plo).abs()).norm()
    lmu=argument_bounds(model.mu_head,c,a,rc,ra,ones)*(
        float(model.signal_shrinkage)*scales.max()*model.mean_bound_multiplier)

    # Every retained signed edge keeps both its sign and a minimum weight.
    stable=a.sign()*(a.abs()-ra/math.sqrt(2.)).clamp_min(0.)
    stable=stable-torch.diag(stable.diagonal())
    ls_stable=torch.diag(stable.abs().sum(-1))-stable
    qlower=torch.diag(qlo+model.q_epsilon)+llo*ls_stable
    mq=torch.linalg.eigvalsh(qlower)[0]
    degree_upper=a.abs().sum(-1)+math.sqrt((n-1)/2.)*ra
    qdiag_upper=(qhi+model.q_epsilon+lhi*degree_upper).max()
    ls=torch.diag(a.abs().sum(-1))-a
    ls_f_upper=ls.norm()+math.sqrt(n)*ra
    lq_matrix=lq+ll*ls_f_upper
    lq_matrix=lq_matrix+lq_matrix.new_tensor([0.,float(lhi)*math.sqrt(n)])

    # (Q^-1)_ii >= 1/Q_ii. The registered local bounds also apply on
    # the convex segment between any pair of attainable precisions.
    diag_floor=1e-8 if law=='implemented' else 0.
    dmin=torch.maximum(1/qdiag_upper,qdiag_upper.new_tensor(diag_floor))
    factor_lip=1/(2*torch.sqrt(dmin)*mq.pow(1.5))
    jitter=2e-5 if law=='implemented' else 0.
    rmax=torch.minimum(mq.new_tensor(float(n)),1/(mq*dmin))+jitter
    # The floor on chi can only reduce the inverse moment used here.
    ln_scalar=1/(2*(nlo-2))+.5*torch.sqrt(torch.special.polygamma(1,nlo/2))
    ln_isotropic=math.sqrt(n)*ln_scalar
    kappa_lip=.5*ln_scalar
    # Changing plus/minus scales uses symmetry and half of each marginal
    # second moment; the deterministic centering term has kappa<=1/2.
    scale_term=math.sqrt((1+jitter)/2)*torch.sqrt(lp.square()+lm.square())+.5*(lp+lm)
    precision_factor_bound=factor_lip*lq_matrix
    original_precision_factor_bound=precision_factor_bound.clone()
    normalized_precision_factor_bound=None
    normalized_precision_floor=None
    # Correlation is unchanged under Q -> Q/lambda whenever the original
    # covariance-diagonal numerical clamp is inactive. In this equivalent
    # representation lambda affects only the diagonal ratio, not L(A).
    if float(1/qdiag_upper)>1e-8:
        blo=(qlo+model.q_epsilon)/lhi
        bhi=(qhi+model.q_epsilon)/llo
        t_floor=torch.linalg.eigvalsh(ls_stable+torch.diag(blo))[0]
        t_diag=(degree_upper+bhi).max()
        t_factor=1/(2*torch.sqrt(1/t_diag)*t_floor.pow(1.5))
        lb=lq/llo+torch.linalg.vector_norm(qhi+model.q_epsilon)/llo.square()*ll
        lt=lb+lb.new_tensor([0.,math.sqrt(n)])
        normalized_precision_factor_bound=t_factor*lt
        normalized_precision_floor=t_floor
        precision_factor_bound=torch.minimum(precision_factor_bound,normalized_precision_factor_bound)
        rmax=torch.minimum(rmax,torch.minimum(mq.new_tensor(float(n)),t_diag/t_floor)+jitter)
    precision_term=sigma_max*precision_factor_bound
    # Apply the Gamma-score proof directly to G~N(0,R+jI): its trace is
    # bounded by N(1+j), including a possible covariance diagonal clamp.
    ln_correlated=math.sqrt(n*(1+jitter))*ln_scalar
    nu_coefficient=sigma_max*ln_correlated+sigma_difference*kappa_lip
    nu_term=nu_coefficient*ln
    total=lmu+scale_term+precision_term+nu_term
    total=total*(1+1e-10)+1e-12
    return {
        'law':law,'network_bound':network_bound,'state_block_constants':total,'L_D_sum_block_norm':total.max(),
        'mean_term':lmu,'scale_and_centering_term':scale_term,
        'precision_term':precision_term,'degrees_of_freedom_term':nu_term,
        'original_precision_factor_block_bound':original_precision_factor_bound,
        'normalized_precision_factor_block_bound':normalized_precision_factor_bound,
        'normalized_precision_eigenvalue_lower':normalized_precision_floor,
        'precision_block_constants':lq_matrix,'precision_eigenvalue_lower':mq,
        'precision_diagonal_upper':qdiag_upper,'covariance_diagonal_lower':dmin,
        'normalization_clamp_inactive':bool(float(1/qdiag_upper)>1e-8),
        'normalized_precision_factor_lipschitz':factor_lip,
        'correlation_with_jitter_eigenvalue_upper':rmax,
        'nu_lower':nlo,'nu_upper':nhi,'nu_kernel_lipschitz':ln_isotropic,
        'nu_correlated_kernel_lipschitz':ln_correlated,
        'kappa_lipschitz':kappa_lip,'sigma_max':sigma_max,
        'total_correlation_jitter':jitter,'chi_floor':1e-8 if law=='implemented' else 0.,
        'dimension':n,
    }
