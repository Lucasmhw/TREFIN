"""Uniform scalar-MLP derivative bounds from a center plus bounded variation.

This is analytic interval control over a whole state tube. The center
Jacobian alone is never used as a Lipschitz certificate.
"""
import math
import torch
from torch import nn


def silu_derivative(x):
    s=torch.sigmoid(x)
    return s+x*s*(1-s)


def _silu_change_radius(center,radius):
    lo,hi=center-radius,center+radius
    flo,fhi=torch.nn.functional.silu(lo),torch.nn.functional.silu(hi)
    bottom=torch.minimum(flo,fhi)
    # SiLU has one minimum in [-2,-1], and SiLU(x)>=-1/e globally.
    crosses=(lo<=-1)&(hi>=-2)
    bottom=torch.where(crosses,torch.minimum(bottom,bottom.new_tensor(-1/math.e)),bottom)
    top=torch.maximum(flo,fhi)
    value=torch.nn.functional.silu(center)
    interval=torch.maximum((value-bottom).abs(),(top-value).abs())
    return torch.minimum(interval,1.1*radius)


def scalar_mlp_state_bound(module,features,c,a,rc,ra,mask,pooled=False):
    layers=[x for x in module.modules() if isinstance(x,nn.Linear)]
    w1,w2,w3=[x.weight for x in layers]
    assert w3.shape[0]==1
    n,k=c.shape
    wc=w1[:,:k]*mask;wn=w1[:,k:2*k]*mask
    x=features.mean(1) if pooled else features[0]
    if x.ndim==1:x=x[None]
    z1=layers[0](x);act1=torch.nn.functional.silu(z1);z2=layers[1](act1)
    d1=silu_derivative(z1);d2=silu_derivative(z2)
    if pooled:
        v=c.new_ones(n)/n;av=a@v
        first_c=torch.sqrt((v.square().sum()*wc.square().sum(-1)
                          +av.square().sum()*wn.square().sum(-1)
                          +2*(v@av)*(wc*wn).sum(-1)).clamp_min(0))
        w=c@wn.T
        matrices=.5*(v[None,:,None]*w.T[:,None,:]+w.T[:,:,None]*v[None,None,:])
        eye=torch.eye(n,dtype=c.dtype,device=c.device)
        first_a=torch.linalg.matrix_norm(matrices*(1-eye),ord='fro')
        r1=(rc*first_c+ra*first_a+rc*ra*v.norm()*wn.norm(dim=1))[None]
    else:
        first_c=torch.sqrt(wc.square().sum(-1)[None]+a.square().sum(-1)[:,None]*wn.square().sum(-1)[None])
        w=c@wn.T
        first_a=torch.sqrt((w.square().sum(0)[None]-w.square()).clamp_min(0)/2)
        r1=rc*first_c+ra*first_a+rc*ra*wn.norm(dim=1)[None]/math.sqrt(2)
    r2=_silu_change_radius(z1,r1)@w2.abs().T
    # |SiLU''| <= 3/4: 2 sigmoid' <=1/2 and
    # |x| sigmoid'(x) <=1/4, the latter from cosh x >=1+x^2/2.
    e1=torch.minimum(.75*r1,torch.maximum(d1+.1,1.1-d1))
    e2=torch.minimum(.75*r2,torch.maximum(d2+.1,1.1-d2))
    partial=torch.cat([wc,wn],dim=1)
    before1=(w3*d2)@w2
    g0=(before1*d1)@partial
    center_to_first=torch.matmul(w2[None]*d1[:,None,:],partial)
    term1=torch.bmm((w3.abs()*e2)[:,None,:],center_to_first.abs()).squeeze(1)
    term2=(before1.abs()*e1)@partial.abs()
    term3=(((w3.abs()*e2)@w2.abs())*e1)@partial.abs()
    err=term1+term2+term3
    gc,gn=g0[:,:k],g0[:,k:]
    ec,en=err[:,:k].norm(dim=1),err[:,k:].norm(dim=1)
    cnorm=torch.linalg.matrix_norm(c,ord=2)
    anorm=torch.linalg.matrix_norm(a,ord=2)
    if pooled:
        jc=v[:,None]*gc[0]+av[:,None]*gn[0]
        w=c@gn[0]
        ja=.5*(v[:,None]*w[None]+w[:,None]*v[None]);ja=ja-torch.diag(ja.diagonal())
        lc=jc.norm()+v.norm()*ec[0]+(av.norm()+ra*v.norm())*en[0]+ra*v.norm()*gn[0].norm()
        la=ja.norm()+v.norm()*en[0]*(cnorm+rc)+v.norm()*gn[0].norm()*rc
    else:
        eye=torch.eye(n,dtype=c.dtype,device=c.device)
        jc=eye[:,:,None]*gc[:,None,:]+a[:,:,None]*gn[:,None,:]
        ij=torch.triu_indices(n,n,1,device=c.device)
        ja=c.new_zeros(n,len(ij[0]))
        ja[ij[0],torch.arange(len(ij[0]),device=c.device)]=(gn[ij[0]]*c[ij[1]]).sum(-1)/math.sqrt(2)
        ja[ij[1],torch.arange(len(ij[0]),device=c.device)]=(gn[ij[1]]*c[ij[0]]).sum(-1)/math.sqrt(2)
        lc=torch.linalg.matrix_norm(jc.reshape(n,-1),ord=2)+ec.max()+(anorm+ra)*en.max()+ra*gn.norm(dim=1).max()
        la=torch.linalg.matrix_norm(ja,ord=2)+en.max()*(cnorm+rc)+gn.norm(dim=1).max()*rc
    # Direct interval propagation also bounds each argument's displacement.
    r3=_silu_change_radius(z2,r2)@w3.abs().T
    return torch.stack([lc,la]),r3.squeeze(-1)
