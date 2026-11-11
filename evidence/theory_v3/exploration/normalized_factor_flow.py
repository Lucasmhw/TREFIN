"""Exploratory uniform correlation coupling; finalized evidence remains unchanged."""
import math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from decoder_bounds import softplus_head, argument_bounds
from local_mlp_bounds import scalar_mlp_state_bound


def precision_flow_bound(model,c,a,s,lead_index,rc=.1,ra=.05):
    n,k=c.shape;lead=model._lead(1,lead_index,c.device)
    _,free=model._decoder_features(c[None],a[None],s[None],lead);mask=1-model.risk_mask
    qlo,qhi,lq,_=softplus_head(model.q_head,free,c,a,rc,ra,mask,model.q_epsilon)
    llo,lhi,ll,_=softplus_head(model.lambda_q_head,free,c,a,rc,ra,mask,.01,pooled=True)
    aq=torch.minimum(argument_bounds(model.q_head,c,a,rc,ra,mask),scalar_mlp_state_bound(model.q_head,free,c,a,rc,ra,mask)[0])
    al=torch.minimum(argument_bounds(model.lambda_q_head,c,a,rc,ra,mask,True),scalar_mlp_state_bound(model.lambda_q_head,free,c,a,rc,ra,mask,True)[0])
    logq=torch.minimum(aq,lq/(qlo+model.q_epsilon).min())
    logl=torch.minimum(al,ll/llo)
    stable=a.sign()*(a.abs()-ra/math.sqrt(2)).clamp_min(0)
    stable=stable-torch.diag(stable.diagonal())
    blo=(qlo+model.q_epsilon)/lhi;bhi=(qhi+model.q_epsilon)/llo
    lower=torch.diag(stable.abs().sum(-1)+blo)-stable
    low_eig=torch.linalg.eigvalsh(lower)[0]
    inv_lower=torch.linalg.inv(lower)
    degreehi=a.abs().sum(-1)+math.sqrt((n-1)/2)*ra
    diagonalhi=degreehi+bhi
    original_diagonalhi=qhi+model.q_epsilon+lhi*degreehi
    assert float(1/original_diagonalhi.max())>1e-8
    rootdiag=diagonalhi.sqrt()
    rmax=torch.minimum(c.new_tensor(float(n)),torch.linalg.eigvalsh(rootdiag[:,None]*inv_lower*rootdiag[None])[-1])
    vari= min(float(rmax.sqrt()/2),math.sqrt(n/8))
    ij=torch.triu_indices(n,n,1,device=c.device);i,j=ij
    center=a[i,j];fixed=center.abs()>ra/math.sqrt(2)
    ii=inv_lower[i[:,None],i[None,:]];ijj=inv_lower[i[:,None],j[None,:]]
    ji=inv_lower[j[:,None],i[None,:]];jj=inv_lower[j[:,None],j[None,:]]
    envelope=torch.zeros_like(ii)
    for se in [-1.,1.]:
        ae=(~fixed)|(center.sign()==se)
        for sf in [-1.,1.]:
            af=(~fixed)|(center.sign()==sf)
            candidate=(ii-sf*ijj-se*ji+se*sf*jj).square()/2
            envelope=torch.maximum(envelope,torch.where(ae[:,None]&af[None],candidate,torch.zeros_like(candidate)))
    graph_relative=torch.linalg.eigvalsh(envelope)[-1].clamp_min(0).sqrt()
    graph_relative=torch.minimum(graph_relative,math.sqrt(n)/low_eig)
    qterm=vari*logq;lterm=math.sqrt(n)/4*logl
    graphterm=c.new_tensor([0.,vari*float(graph_relative)])
    bound=(qterm+lterm+graphterm)*(1+1e-10)+1e-12
    return {'block_bound':bound,'q_log_block':logq,'lambda_log_block':logl,
            'q_term':qterm,'lambda_term':lterm,'graph_term':graphterm,
            'rmax':float(rmax),'graph_relative':float(graph_relative),'lower_eigenvalue':float(low_eig),
            'allowed_two_sign_edges':int((~fixed).sum()),'edges':len(i)}
