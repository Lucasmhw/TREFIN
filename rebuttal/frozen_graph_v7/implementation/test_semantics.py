"""Deterministic semantic fixtures, not empirical forecasting results."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,types
import torch

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
CONFIG=dict(hidden_dim=8,lead_dim=3,graph_embedding_dim=4,delta=.2,gamma_min=.12,gamma_max=.9,
    diffusion_max=.15,force_max=.4,graph_refresh=.1,future_graph_gate_center=.1,
    future_graph_gate_temperature=.05,q_epsilon=.02,nu_min=2.1,residual_scale=.01,
    signal_shrinkage=1.0,scale_floor=.05,risk_strength_max=1.0)
TRAIN=dict(lambda_concept=.25,lambda_graph=.08,lambda_monotonicity=.01,lambda_gain=.1,gain_target=.995)

def make(module,variant='Full',dtype=torch.float64,device='cpu',**updates):
    config={**CONFIG,**updates,'ablation_variant':variant}
    return module.FCONCORD(num_assets=4,num_concepts=6,horizon=5,
      directional_mask=torch.tensor([1,1,1,0,0,0]),risk_indices=[4,5],config=config,
      concept_bound=4.,graph_bound=3.,return_scale=torch.full((4,),.02)).to(device=device,dtype=dtype)
def inputs(dtype,device):
    c=torch.linspace(-.7,.8,48,device=device,dtype=dtype).reshape(2,4,6)
    a=torch.tensor([[0,.2,-.1,.05],[.2,0,.08,-.12],[-.1,.08,0,.15],[.05,-.12,.15,0]],device=device,dtype=dtype)
    return c,a.expand(2,-1,-1).clone(),torch.tensor([[.2,-.1,.3],[-.15,.1,.2]],device=device,dtype=dtype)
def equal_outputs(a,b):
    assert a.keys()==b.keys()
    for k in a:
        if torch.is_tensor(a[k]):assert torch.equal(a[k],b[k]),k
        else:assert a[k]==b[k],k
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-model',type=Path,required=True);ap.add_argument('--losses',type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    old=load('frozen_test_base',args.base_model);new=load('frozen_test_new',HERE/'model.py');losses=load('frozen_test_losses',args.losses)
    checks={}
    for dtype in [torch.float32,torch.float64]:
        torch.manual_seed(123);base=make(old,dtype=dtype,device=args.device)
        torch.manual_seed(123);full=make(new,dtype=dtype,device=args.device)
        for k,v in base.state_dict().items():assert torch.equal(v,full.state_dict()[k]),k
        c,a,s=inputs(dtype,args.device)
        torch.manual_seed(81);x=base(c,a,s,8)
        torch.manual_seed(81);y=full(c,a,s,8);equal_outputs(x,y)
        batch={'y':torch.full_like(x['mu'],.001),'c_target':x['concept_path'].detach()+.03,'a_target':x['graph_path'].detach()+.01}
        lx,_=losses.total_loss(x,batch,TRAIN,base.risk_indices);ly,_=losses.total_loss(y,batch,TRAIN,full.risk_indices)
        assert torch.equal(lx,ly);lx.backward();ly.backward()
        for (n,p),(n2,q) in zip(base.named_parameters(),full.named_parameters()):
            assert n==n2 and ((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)),n
        for m in [base,full]:
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001).step()
        for k,v in base.state_dict().items():assert torch.equal(v,full.state_dict()[k]),k
        checks[f'Full_bitwise_forward_loss_gradients_optimizer_{dtype}']=True
    torch.manual_seed(123);frozen=make(new,'FrozenGraph',device=args.device)
    c,a,s=inputs(torch.float64,args.device);a.requires_grad_(True)
    def fail(*args,**kwargs):raise AssertionError('Inactive graph proposal was evaluated')
    frozen._graph_proposal=fail
    handles=[m.register_forward_hook(lambda *args:fail()) for m in [frozen.similarity_encoder,frozen.opposition_encoder]]
    torch.manual_seed(91);x=frozen(c,a,s,8)
    assert torch.equal(x['graph_path'],a[:,None].expand(-1,6,-1,-1))
    assert torch.equal(torch.autograd.grad(x['graph_path'].sum(),a,retain_graph=True)[0],torch.full_like(a,6))
    for bound in [frozen.batch_gain_bound(c,a,s),frozen.structural_gain_bound()]:
        matrix,radius=bound;assert matrix[1,0].item()==0 and matrix[1,1].item()==1
        assert torch.equal(radius,torch.maximum(matrix[0,0],torch.ones_like(radius)))
        assert matrix[0,1]>0
    checks['FrozenGraph_identity_all_leads_and_A0_derivative']=True
    checks['FrozenGraph_augmented_gain_row_0_1_and_retained_C_from_A']=True
    lead=frozen._lead(2,0,c.device)
    torch.manual_seed(123);full=make(new,device=args.device)
    fc,fa=frozen._step(c,a,s,lead);bc,ba=full._step(c,a,s,lead)
    assert torch.equal(fc,bc) and torch.equal(fa,a) and not torch.equal(ba,a)
    assert not torch.equal(fc,frozen._step(c,torch.zeros_like(a),s,lead)[0])
    features,_=frozen._decoder_features(c,a,s,lead)
    assert torch.equal(features[...,6:12],a@c)
    dx=frozen._decode_state(c,a,s,lead);dy=frozen._decode_state(c,torch.zeros_like(a),s,lead)
    assert not torch.equal(dx['precision'],dy['precision'])
    with torch.no_grad():frozen.mu_head[-1].weight.fill_(.1)
    assert not torch.equal(frozen._decode_state(c,a,s,lead)['mu'],frozen._decode_state(c,torch.zeros_like(a),s,lead)['mu'])
    checks['FrozenGraph_retains_typed_propagation_AC_mean_and_precision_influence']=True
    # A0 is data during training; inactive proposal parameters must have no loss gradient.
    a=a.detach();torch.manual_seed(91);x=frozen(c,a,s,8)
    batch={'y':torch.full_like(x['mu'],.001),'c_target':x['concept_path'].detach()+.03,'a_target':x['graph_path'].detach()+.01}
    cfg={**TRAIN,'lambda_graph':0.0};loss,_=losses.total_loss(x,batch,cfg,frozen.risk_indices)
    other,_=losses.total_loss(x,{**batch,'a_target':batch['a_target']+1},cfg,frozen.risk_indices)
    assert torch.equal(loss,other);loss.backward()
    for name,p in frozen.named_parameters():
        if name.startswith(('similarity_encoder.','opposition_encoder.')):assert p.grad is None,name
        elif p.grad is not None:assert torch.isfinite(p.grad).all(),name
    assert frozen.mu_head[-1].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in frozen.residual_net.parameters())
    checks['FrozenGraph_graph_loss_zero_and_unused_proposal_gradients_absent']=True
    for handle in handles:handle.remove()
    # Slow refresh and arbitrary unused proposal weights are exact aliases.
    alt=copy.deepcopy(frozen);alt.graph_refresh=.02
    with torch.no_grad():
        for m in [alt.similarity_encoder,alt.opposition_encoder]:
            for p in m.parameters():p.mul_(100).add_(2)
    torch.manual_seed(103);x=frozen(c,a,s,8)
    torch.manual_seed(103);y=alt(c,a,s,8);equal_outputs(x,y)
    checks['FrozenGraph_refresh_and_proposal_parameters_do_not_affect_any_output']=True
    # Finite subgradient at the identity-row eigenvalue tie; no sqrt(0) autograd.
    tie=make(new,'FrozenGraph',device=args.device,gamma_min=0.,gamma_max=0.,diffusion_max=0.,residual_scale=1.)
    probe=torch.zeros((),dtype=torch.float64,device=args.device,requires_grad=True)
    tie._spectral_product=lambda module:probe
    matrix,radius=tie.batch_gain_bound(c,a,s)
    penalty=.1*torch.relu(radius-.995).square();penalty.backward()
    assert radius.item()==1 and abs(penalty.item()-2.5e-6)<1e-15 and torch.isfinite(probe.grad)
    checks['FrozenGraph_gain_floor_and_tie_gradient_finite']=True
    report={'status':'passed','device':args.device,'torch':torch.__version__,'checks':checks,
      'base_model_sha256':sha(args.base_model),'patched_model_sha256':sha(HERE/'model.py'),
      'losses_sha256':sha(args.losses),'scope':'Small deterministic semantic fixtures only; not forecast evidence.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
