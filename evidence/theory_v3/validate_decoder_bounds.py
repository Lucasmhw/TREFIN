"""Actual-checkpoint checks of local head and full covariance-factor bounds."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from decoder_bounds import decoder_w1_bound,argument_bounds
from local_mlp_bounds import scalar_mlp_state_bound


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def factor(q):
    eigen,v=torch.linalg.eigh(q)
    invroot=(v*eigen.rsqrt()[None])@v.T
    denominator=invroot.square().sum(-1).clamp_min(1e-8).sqrt()
    return invroot/denominator[:,None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('work/server_snapshot/f-concord/src'))
    p.add_argument('--protocol',type=Path,default=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1'))
    p.add_argument('--state-results',type=Path,default=Path('work/theory_v3/local_tube_results_v2'))
    p.add_argument('--output',type=Path,default=Path('work/theory_v3/decoder_validation'))
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);sys.path.insert(0,str(a.source.resolve()));torch.set_num_threads(4)
    from fconcord.config import load_config
    from fconcord.experiment import _model_factory
    from fconcord.model import project_graph
    rng=torch.Generator().manual_seed(20261109);runs=[];start=time.perf_counter()
    names=['mu_head','up_scale_head','down_scale_head','q_head','lambda_q_head','nu_head']
    for fold in [1,2,3]:
        pp=a.protocol/'runs'/f'fold_{fold}_seed_1'/'audit';meta=json.loads((pp/'prepared_metadata.json').read_text())
        with np.load(pp/'prepared_arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in ['c0','a0','s0','directional_mask','risk_indices']}
        prepared={'metadata':meta,'arrays':arrays};n,k=meta['num_assets'],meta['num_concepts'];nc=n*k;ij=torch.triu_indices(n,n,1)
        def pack(c,g):return torch.cat([c.reshape(-1),2**.5*g[ij[0],ij[1]]])
        def unpack(x):
            c=x[:nc].view(n,k);g=x.new_zeros(n,n).index_put((ij[0],ij[1]),x[nc:]/2**.5)
            return c,g+g.T
        for seed in [1,7,19,37,73]:
            out=a.output/f'fold_{fold}_seed_{seed}.json'
            if out.exists():runs.append(json.loads(out.read_text()));continue
            rp=a.protocol/'runs'/f'fold_{fold}_seed_{seed}';cfg=load_config(rp/'config_resolved.yaml')
            m=_model_factory(prepared,cfg).double().eval();ck=torch.load(rp/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(ck['model_state']);m.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model']['signal_shrinkage']));m.requires_grad_(False)
            assert not m.gaussian_symmetric_decoder
            statespec=json.loads((a.state_results/f'fold_{fold}_seed_{seed}.json').read_text())
            rec={'fold':fold,'seed':seed,'checkpoint_sha256':sha(rp/'checkpoints/best.pt'),'dense_head_blocks':0,
                 'finite_head_pairs':0,'factor_pairs':0,'max_dense_ratio':0.,'max_head_difference_ratio':0.,
                 'max_factor_difference_ratio':0.,'max_covariance_factor_identity_error':0.,
                 'max_precision_lower_violation':0.,'max_precision_diagonal_upper_violation':0.,
                 'max_nu_interval_violation':0.,'normalization_clamp_inactive':True}
            for oi in [0,len(statespec['origins'])-1]:
                idx=statespec['origins'][oi]['prepared_index'];c=torch.tensor(arrays['c0'][idx],dtype=torch.float64);g=torch.tensor(arrays['a0'][idx],dtype=torch.float64);s=torch.tensor(arrays['s0'][idx],dtype=torch.float64)
                for h in range(1,6):
                    with torch.no_grad():c1,g1=m._step(c[None],g[None],s[None],m._lead(1,h-1,c.device));c,g=c1[0],g1[0]
                    if h not in [1,5]:continue
                    lead=m._lead(1,h,c.device)
                    full,free=m._decoder_features(c[None],g[None],s[None],lead)
                    def allargs(x):
                        cc,gg=unpack(x);ff,fr=m._decoder_features(cc[None],gg[None],s[None],lead)
                        return torch.cat([getattr(m,name)((ff if name=='mu_head' else fr).mean(1) if name in ['lambda_q_head','nu_head'] else (ff if name=='mu_head' else fr)).reshape(-1) for name in names])
                    jac=torch.func.jacrev(allargs,chunk_size=64)(pack(c,g))
                    centerdec=m._decode_state(c[None],g[None],s[None],lead)
                    fc=factor(centerdec['precision'][0])
                    ident=float((fc@fc.T+1e-5*torch.eye(n)-centerdec['correlation'][0]).abs().max())
                    rec['max_covariance_factor_identity_error']=max(rec['max_covariance_factor_identity_error'],ident)
                    for rc in [.001,.01,.1,1.]:
                        ra=rc/2;bound=decoder_w1_bound(m,c,g,s,h,rc,ra,'implemented','local')
                        rec['normalization_clamp_inactive'] &= bound['normalization_clamp_inactive']
                        head_bounds=[];position=0
                        for name in names:
                            module=getattr(m,name);feat=full if name=='mu_head' else free;mask=torch.ones(k) if name=='mu_head' else 1-m.risk_mask;pooled=name in ['lambda_q_head','nu_head']
                            local,_=scalar_mlp_state_bound(module,feat,c,g,rc,ra,mask,pooled)
                            hb=torch.minimum(local,argument_bounds(module,c,g,rc,ra,mask,pooled));head_bounds.append(hb)
                            length=1 if pooled else n;jblock=jac[position:position+length];position+=length
                            actual=torch.stack([torch.linalg.matrix_norm(jblock[:,:nc],ord=2),torch.linalg.matrix_norm(jblock[:,nc:],ord=2)])
                            assert torch.all(actual<=hb+1e-9),(fold,seed,h,rc,name,actual,hb)
                            rec['dense_head_blocks']+=2;rec['max_dense_ratio']=max(rec['max_dense_ratio'],float((actual/hb).max()))
                        for direction in range(3):
                            dc=torch.randn(c.shape,generator=rng,dtype=c.dtype);dc=dc/dc.norm()
                            edge=torch.randn(len(ij[0]),generator=rng,dtype=c.dtype);edge=edge/edge.norm()
                            dg=torch.zeros_like(g).index_put((ij[0],ij[1]),edge/2**.5);dg=dg+dg.T
                            for sign in [-1,1]:
                                cp=torch.maximum(torch.minimum(c+sign*rc*dc,m.concept_upper_bound),m.concept_lower_bound)
                                gp=project_graph((g+sign*ra*dg)[None],m.graph_bound)[0]
                                delta=torch.stack([(cp-c).norm(),(gp-g).norm()]);fp,frp=m._decoder_features(cp[None],gp[None],s[None],lead)
                                for name,hb in zip(names,head_bounds):
                                    feat=full if name=='mu_head' else free;featp=fp if name=='mu_head' else frp
                                    if name in ['lambda_q_head','nu_head']:feat,featp=feat.mean(1),featp.mean(1)
                                    diff=(getattr(m,name)(featp)-getattr(m,name)(feat)).norm();cap=hb@delta
                                    assert diff<=cap+1e-9,(fold,seed,h,rc,name,float(diff),float(cap))
                                    rec['finite_head_pairs']+=1;rec['max_head_difference_ratio']=max(rec['max_head_difference_ratio'],float(diff/cap))
                                dp=m._decode_state(cp[None],gp[None],s[None],lead);qp=dp['precision'][0]
                                factor_cap=bound['original_precision_factor_block_bound']
                                if bound['normalized_precision_factor_block_bound'] is not None:factor_cap=torch.minimum(factor_cap,bound['normalized_precision_factor_block_bound'])
                                actual=(factor(qp)-fc).norm();cap=factor_cap@delta
                                assert actual<=cap+1e-9,(fold,seed,h,rc,float(actual),float(cap))
                                rec['factor_pairs']+=1;rec['max_factor_difference_ratio']=max(rec['max_factor_difference_ratio'],float(actual/cap))
                                lo=float(bound['precision_eigenvalue_lower']);hi=float(bound['precision_diagonal_upper']);nu=float(dp['nu'])
                                lower_error=max(0.,lo-float(torch.linalg.eigvalsh(qp)[0]));upper_error=max(0.,float(qp.diagonal().max())-hi)
                                nu_error=max(0.,float(bound['nu_lower'])-nu,nu-float(bound['nu_upper']))
                                rec['max_precision_lower_violation']=max(rec['max_precision_lower_violation'],lower_error)
                                rec['max_precision_diagonal_upper_violation']=max(rec['max_precision_diagonal_upper_violation'],upper_error)
                                rec['max_nu_interval_violation']=max(rec['max_nu_interval_violation'],nu_error)
                                assert max(lower_error,upper_error,nu_error)<1e-9
            assert rec['max_covariance_factor_identity_error']<1e-10
            rec['passed']=True;out.write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8');runs.append(rec)
            print(json.dumps({'fold':fold,'seed':seed,'passed':True,'head_pairs':rec['finite_head_pairs'],'factor_pairs':rec['factor_pairs'],'elapsed':time.perf_counter()-start}),flush=True)
    result={'passed':all(r['passed'] for r in runs),'runs':runs,'script_sha256':sha(__file__),
            'decoder_source_sha256':sha(Path(__file__).with_name('decoder_bounds.py')),
            'local_mlp_source_sha256':sha(Path(__file__).with_name('local_mlp_bounds.py')),
            'scope':'numerical implementation checks of analytic parameter/factor bounds; no Monte Carlo estimate is presented as a Wasserstein certificate',
            'totals':{key:sum(r[key] for r in runs) for key in ['dense_head_blocks','finite_head_pairs','factor_pairs']},
            'wall_seconds':time.perf_counter()-start}
    (a.output/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
