"""Independent numerical checks of the analytic local-tube implementation.

Dense Jacobians and finite perturbations validate the code; they do not define
the uniform bounds. At exactly zero graph edges, the autograd linearization
uses its selected absolute-value subgradient. Finite perturbations include
both signs at those edges and are checked against the analytic bound.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from tube_bounds import state_constants,tube_bound


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('work/theory_v3/local_tube_results_v2'))
    p.add_argument('--protocol',type=Path,default=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1'))
    p.add_argument('--source',type=Path,default=Path('work/server_snapshot/f-concord/src'))
    p.add_argument('--output',type=Path,default=Path('work/theory_v3/validation'))
    p.add_argument('--dense-origins',type=int,default=1)
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,7,19,37,73])
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads);a.output.mkdir(exist_ok=True,parents=True)
    sys.path.insert(0,str(a.source.resolve()))
    from fconcord.config import load_config
    from fconcord.experiment import _model_factory
    from fconcord.model import project_graph
    report={'schema':'trefin-tube-bound-validation-v1','script_sha256':digest(__file__),
            'bound_script_sha256':digest(Path(__file__).with_name('tube_bounds.py')),
            'seed':20261108,'dtype':'float64','dense_origins_per_run':a.dense_origins,
            'dense_scope':'full finite-dimensional autograd linearizations; absolute value uses selected subgradient at zero edges',
            'finite_scope':'random admissible initial perturbations of both signs at all four prespecified radii, first/last sampled origin',
            'dense_jacobians_do_not_define_analytic_bounds':True,'runs':[]}
    generator=torch.Generator().manual_seed(20261108)
    t0=time.perf_counter()
    for fold in a.folds:
        prep=a.protocol/'runs'/f'fold_{fold}_seed_1'/'audit'
        meta=json.loads((prep/'prepared_metadata.json').read_text())
        with np.load(prep/'prepared_arrays.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in ['c0','a0','s0','directional_mask','risk_indices']}
        prepared={'metadata':meta,'arrays':arrays}
        n,k=meta['num_assets'],meta['num_concepts'];nc=n*k
        ij=torch.triu_indices(n,n,1);root2=2**.5
        def pack(c,g):return torch.cat([c.reshape(-1),root2*g[ij[0],ij[1]]])
        def unpack(x):
            cc=x[:nc].view(n,k)
            aa=x.new_zeros(n,n).index_put((ij[0],ij[1]),x[nc:]/root2)
            return cc,aa+aa.T
        for seed in a.seeds:
            out=a.output/f'fold_{fold}_seed_{seed}.json'
            if out.exists():report['runs'].append(json.loads(out.read_text()));continue
            run=a.protocol/'runs'/f'fold_{fold}_seed_{seed}'
            config=load_config(run/'config_resolved.yaml')
            m=_model_factory(prepared,config).double().eval()
            ck=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(ck['model_state']);m.signal_shrinkage=float(ck.get('signal_shrinkage',config['model']['signal_shrinkage']));m.requires_grad_(False)
            constants=state_constants(m)
            recorded=json.loads((a.results/f'fold_{fold}_seed_{seed}.json').read_text())
            dense_positions=set(np.linspace(0,len(recorded['origins'])-1,a.dense_origins,dtype=int).tolist())
            finite_positions={0,len(recorded['origins'])-1}
            rr={'fold':fold,'seed':seed,'checkpoint_sha256':digest(run/'checkpoints/best.pt'),
                'centers_checked':0,'max_box_violation':0.,'max_graph_ball_violation':0.,'max_asymmetry':0.,'max_diagonal':0.,
                'dense':[],'finite':[]}
            for oi,origin in enumerate(recorded['origins']):
                index=origin['prepared_index']
                c=torch.tensor(arrays['c0'][index],dtype=torch.float64)
                g=torch.tensor(arrays['a0'][index],dtype=torch.float64)
                s=torch.tensor(arrays['s0'][index],dtype=torch.float64)
                states=[(c,g)];means=[]
                with torch.no_grad():
                    for h in range(5):
                        c,g=m._step(c[None],g[None],s[None],m._lead(1,h,c.device));c,g=c[0],g[0]
                        states.append((c,g));means.append(m._decode_state(c[None],g[None],s[None],m._lead(1,h+1,c.device))['mu'][0])
                    for c,g in states:
                        rr['centers_checked']+=1
                        rr['max_box_violation']=max(rr['max_box_violation'],float((m.concept_lower_bound-c).clamp_min(0).max()),float((c-m.concept_upper_bound).clamp_min(0).max()))
                        rr['max_graph_ball_violation']=max(rr['max_graph_ball_violation'],max(0.,float(g.norm())-m.graph_bound))
                        rr['max_asymmetry']=max(rr['max_asymmetry'],float((g-g.T).abs().max()))
                        rr['max_diagonal']=max(rr['max_diagonal'],float(g.diagonal().abs().max()))
                if oi in dense_positions:
                    product=torch.eye(nc+len(ij[0]),dtype=torch.float64)
                    for h,(c,g) in enumerate(states[:-1]):
                        lead=m._lead(1,h,c.device)
                        def step(x):
                            cc,aa=unpack(x);cc,aa=m._step(cc[None],aa[None],s[None],lead)
                            return pack(cc[0],aa[0])
                        jac=torch.func.jacrev(step,chunk_size=64)(pack(c,g))
                        product=jac@product
                        blocks=np.array([[float(torch.linalg.svdvals(jac[:nc,:nc])[0]),float(torch.linalg.svdvals(jac[:nc,nc:])[0])],
                                         [float(torch.linalg.svdvals(jac[nc:,:nc])[0]),float(torch.linalg.svdvals(jac[nc:,nc:])[0])]])
                        bound=np.asarray(origin['radii'][2]['block_matrices'][h])
                        assert np.all(blocks<=bound+1e-9),(fold,seed,h,blocks,bound)
                        rr['dense'].append({'test_position':origin['test_position'],'lead':h+1,
                            'zero_offdiagonal_edges':int((g[ij[0],ij[1]]==0).sum()),
                            'exact_svd_block_norms_of_autograd_linearization':blocks.tolist(),
                            'analytic_tube_block_bounds':bound.tolist(),'maximum_block_ratio':float((blocks/bound).max()),
                            'h_step_full_state_linearization_l2_norm':float(torch.linalg.svdvals(product)[0]),
                            'h_step_concept_initialization_to_full_state_l2_norm':float(torch.linalg.svdvals(product[:,:nc])[0])})
                if oi in finite_positions:
                    c0,g0=states[0]
                    for radius in origin['radii']:
                        rc,ra=radius['concept_radius'],radius['graph_radius']
                        for direction_index in range(3):
                            dc=torch.randn(c0.shape,generator=generator,dtype=c0.dtype);dc=dc/dc.norm()
                            edge=torch.randn(len(ij[0]),generator=generator,dtype=c0.dtype);edge=edge/edge.norm()
                            ga=torch.zeros_like(g0).index_put((ij[0],ij[1]),edge/root2);ga=ga+ga.T
                            for sign in [-1,1]:
                                cp=torch.maximum(torch.minimum(c0+sign*.1*rc*dc,m.concept_upper_bound),m.concept_lower_bound)
                                gp=project_graph((g0+sign*.1*ra*ga)[None],m.graph_bound)[0]
                                init=np.array([float((cp-c0).norm()),float((gp-g0).norm())]);current=init.copy();prefix=True
                                for h in range(5):
                                    with torch.no_grad():
                                        cp,gp=m._step(cp[None],gp[None],s[None],m._lead(1,h,cp.device));cp,gp=cp[0],gp[0]
                                        mup=m._decode_state(cp[None],gp[None],s[None],m._lead(1,h+1,cp.device))['mu'][0]
                                    current=np.asarray(radius['block_matrices'][h])@current
                                    errors=np.array([float((cp-states[h+1][0]).norm()),float((gp-states[h+1][1]).norm())])
                                    prefix=prefix and radius['certified_prefixes'][h]
                                    assert prefix
                                    assert np.all(errors<=current+1e-9),(fold,seed,h,errors,current)
                                    assert np.all(errors<=np.array([rc,ra])+1e-9)
                                    _,mean_bound=tube_bound(m,*states[h+1],s,h+1,rc,ra,constants)
                                    forecast_bound=float(mean_bound.numpy()@current)
                                    mean_error=float((mup-means[h]).norm())
                                    assert mean_error<=forecast_bound+1e-10,(fold,seed,h,mean_error,forecast_bound)
                                    rr['finite'].append({'test_position':origin['test_position'],'concept_radius':rc,
                                        'direction':direction_index,'sign':sign,'lead':h+1,'initial_error_blocks':init.tolist(),
                                        'observed_error_blocks':errors.tolist(),'certified_error_blocks':current.tolist(),
                                        'observed_to_certified_ratio':(errors/np.maximum(current,1e-30)).tolist(),
                                        'observed_mean_l2_difference':mean_error,'certified_mean_l2_difference':forecast_bound})
            assert rr['max_box_violation']<1e-10 and rr['max_graph_ball_violation']<1e-10
            assert rr['max_asymmetry']<1e-10 and rr['max_diagonal']<1e-10
            rr['passed']=True
            out.write_text(json.dumps(rr,indent=2)+'\n',encoding='utf-8');report['runs'].append(rr)
            print(json.dumps({'fold':fold,'seed':seed,'passed':True,'dense_checks':len(rr['dense']),
                              'finite_checks':len(rr['finite']),'centers':rr['centers_checked'],
                              'elapsed':time.perf_counter()-t0}),flush=True)
    report['passed']=all(r['passed'] for r in report['runs'])
    report['totals']={'runs':len(report['runs']),'centers':sum(r['centers_checked'] for r in report['runs']),
                      'dense_jacobians':sum(len(r['dense']) for r in report['runs']),
                      'finite_perturbation_leads':sum(len(r['finite']) for r in report['runs']),
                      'wall_seconds':time.perf_counter()-t0}
    report['maximum_dense_block_ratio']=max(x['maximum_block_ratio'] for r in report['runs'] for x in r['dense'])
    report['maximum_finite_state_ratio']=max(max(x['observed_to_certified_ratio']) for r in report['runs'] for x in r['finite'])
    report['maximum_finite_mean_ratio']=max(x['observed_mean_l2_difference']/max(x['certified_mean_l2_difference'],1e-30) for r in report['runs'] for x in r['finite'])
    (a.output/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete',**report['totals']}),flush=True)


if __name__=='__main__':main()
