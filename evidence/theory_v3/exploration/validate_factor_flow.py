"""Actual-checkpoint verification of exploratory Gaussian flow coupling bounds."""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from normalized_factor_flow import precision_flow_bound
sys.path.insert(0,str(Path('work/server_snapshot/f-concord/src').resolve()))
from fconcord.config import load_config
from fconcord.experiment import _model_factory
from fconcord.model import project_graph

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gaussian_w2(a,b):
    eigen,v=torch.linalg.eigh(a);root=(v*eigen.clamp_min(0).sqrt()[None])@v.T
    middle=root@b@root;cross=torch.linalg.eigvalsh((middle+middle.T)/2).clamp_min(0).sqrt().sum()
    return (a.trace()+b.trace()-2*cross).clamp_min(0).sqrt()

def main():
    root=Path(__file__).resolve().parent;out=root/'validation';out.mkdir(exist_ok=True)
    protocol=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1')
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(2026111001);records=[];start=time.perf_counter()
    for fold in [1,2,3]:
        pp=protocol/'runs'/f'fold_{fold}_seed_1'/'audit';meta=json.loads((pp/'prepared_metadata.json').read_text());n,k=meta['num_assets'],meta['num_concepts'];nc=n*k
        with np.load(pp/'prepared_arrays.npz') as z:arrays={k:z[k] for k in ['c0','a0','s0','directional_mask','risk_indices']}
        ij=torch.triu_indices(n,n,1)
        def pack(c,g):return torch.cat([c.reshape(-1),math.sqrt(2)*g[ij[0],ij[1]]])
        def unpack(x):
            c=x[:nc].reshape(n,k);g=x.new_zeros(n,n).index_put((ij[0],ij[1]),x[nc:]/math.sqrt(2));return c,g+g.T
        for seed in [1,7,19,37,73]:
            dest=out/f'fold_{fold}_seed_{seed}.json'
            if dest.exists():records.append(json.loads(dest.read_text()));continue
            rp=protocol/'runs'/f'fold_{fold}_seed_{seed}';cfg=load_config(rp/'config_resolved.yaml');m=_model_factory({'metadata':meta,'arrays':arrays},cfg).double().eval()
            ck=torch.load(rp/'checkpoints/best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state']);m.requires_grad_(False);m.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model']['signal_shrinkage']))
            state=json.loads((root.parent/f'local_tube_results_v2/fold_{fold}_seed_{seed}.json').read_text())
            rec={'fold':fold,'seed':seed,'checkpoint_sha256':sha(rp/'checkpoints/best.pt'),'dense_log_head_block_checks':0,'finite_gaussian_pairs':0,'finite_log_head_checks':0,'maximum_dense_log_head_ratio':0.,'maximum_log_head_ratio':0.,'maximum_gaussian_parameter_ratio':0.,'maximum_gaussian_state_ratio':0.,'maximum_parameter_to_state_ratio':0.,'edge_sign_changes':0}
            for oi in [0,31]:
                idx=state['origins'][oi]['prepared_index'];c,g,s=[torch.tensor(arrays[key][idx],dtype=torch.float64) for key in ['c0','a0','s0']]
                for h in range(1,6):
                    with torch.no_grad():cc,gg=m._step(c[None],g[None],s[None],m._lead(1,h-1,c.device));c,g=cc[0],gg[0]
                    if h not in [1,5]:continue
                    lead=m._lead(1,h,c.device)
                    def logs(x):
                        cc,gg=unpack(x);_,fr=m._decoder_features(cc[None],gg[None],s[None],lead)
                        q=2*m.q_epsilon+torch.nn.functional.softplus(m.q_head(fr).reshape(-1));lam=.01+torch.nn.functional.softplus(m.lambda_q_head(fr.mean(1)).reshape(-1))
                        return torch.cat([q.log(),lam.log()])
                    x=pack(c,g);centerlog=logs(x);jac=torch.func.jacrev(logs,chunk_size=32)(x)
                    centerR=m._decode_state(c[None],g[None],s[None],lead)['correlation'][0]+1e-5*torch.eye(n,dtype=c.dtype)
                    for rc in [.001,.01,.1,1.]:
                        ra=rc/2;b=precision_flow_bound(m,c,g,s,h,rc,ra);qco=min(math.sqrt(b['rmax'])/2,math.sqrt(n/8));lco=math.sqrt(n)/4;gco=b['graph_term'][1]
                        for sl,key in [(slice(0,n),'q_log_block'),(slice(n,n+1),'lambda_log_block')]:
                            actual=torch.stack([torch.linalg.matrix_norm(jac[sl,:nc],ord=2),torch.linalg.matrix_norm(jac[sl,nc:],ord=2)])
                            assert torch.all(actual<=b[key]+1e-10),(fold,seed,h,rc,key)
                            rec['dense_log_head_block_checks']+=2;rec['maximum_dense_log_head_ratio']=max(rec['maximum_dense_log_head_ratio'],float((actual/b[key]).max()))
                        for direction in range(3):
                            dc=torch.randn(c.shape,generator=rng,dtype=c.dtype);dc/=dc.norm();e=torch.randn(len(ij[0]),generator=rng,dtype=c.dtype);e/=e.norm();dg=torch.zeros_like(g).index_put((ij[0],ij[1]),e/math.sqrt(2));dg+=dg.T.clone()
                            for block in ['C','A','both']:
                                for sign in [-1.,1.]:
                                    cp=c if block=='A' else torch.maximum(torch.minimum(c+sign*rc*dc,m.concept_upper_bound),m.concept_lower_bound)
                                    gp=g if block=='C' else project_graph((g+sign*ra*dg)[None],m.graph_bound)[0]
                                    delta=torch.stack([(cp-c).norm(),(gp-g).norm()]);lp=logs(pack(cp,gp));changes=torch.stack([(lp[:n]-centerlog[:n]).norm(),(lp[n]-centerlog[n]).abs()])
                                    qcap=b['q_log_block']@delta;lcap=b['lambda_log_block']@delta
                                    assert changes[0]<=qcap+1e-10 and changes[1]<=lcap+1e-10
                                    rec['finite_log_head_checks']+=2;rec['maximum_log_head_ratio']=max(rec['maximum_log_head_ratio'],float(changes[0]/qcap),float(changes[1]/lcap))
                                    Rp=m._decode_state(cp[None],gp[None],s[None],lead)['correlation'][0]+1e-5*torch.eye(n,dtype=c.dtype)
                                    actual=gaussian_w2(centerR,Rp);pcap=qco*changes[0]+lco*changes[1]+gco*delta[1];scap=b['block_bound']@delta
                                    assert actual<=pcap+1e-7,(fold,seed,h,rc,block,float(actual),float(pcap))
                                    assert pcap<=scap+1e-9
                                    rec['finite_gaussian_pairs']+=1;rec['maximum_gaussian_parameter_ratio']=max(rec['maximum_gaussian_parameter_ratio'],float(actual/pcap));rec['maximum_gaussian_state_ratio']=max(rec['maximum_gaussian_state_ratio'],float(actual/scap));rec['maximum_parameter_to_state_ratio']=max(rec['maximum_parameter_to_state_ratio'],float(pcap/scap));rec['edge_sign_changes']+=int(((gp[ij[0],ij[1]].sign()!=g[ij[0],ij[1]].sign())&(gp[ij[0],ij[1]]!=g[ij[0],ij[1]])).sum())
            rec['passed']=True;dest.write_text(json.dumps(rec,indent=2)+'\n');records.append(rec);print(json.dumps({'fold':fold,'seed':seed,'passed':True,'pairs':rec['finite_gaussian_pairs'],'seconds':time.perf_counter()-start}),flush=True)
    summary={'passed':all(r['passed'] for r in records),'runs':records,'flow_source_sha256':sha(root/'normalized_factor_flow.py'),'validator_sha256':sha(__file__),'scope':'Actual-checkpoint Gaussian W2 and uniform log-head implementation checks; analytic inequalities establish the complete mixture coupling bound','totals':{k:sum(r[k] for r in records) for k in ['dense_log_head_block_checks','finite_gaussian_pairs','finite_log_head_checks','edge_sign_changes']},'maxima':{k:max(r[k] for r in records) for k in records[0] if k.startswith('maximum_')},'seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
