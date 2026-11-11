"""Explicit local decoder-kernel W1 coefficients on the completed state tubes."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from decoder_bounds import decoder_w1_bound


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return {'n':0,'minimum':None,'median':None,'q95':None,'maximum':None}
    return {'n':int(x.size),'minimum':float(x.min()),'median':float(np.median(x)),
            'q95':float(np.quantile(x,.95)),'maximum':float(x.max())}


VARIANTS=[('ideal','spectral'),('implemented','spectral'),('ideal','local'),('implemented','local')]
FIELDS=['L_C','L_A','L_D_sum_block_norm','mean_C','mean_A','scale_C','scale_A',
        'precision_C','precision_A','nu_C','nu_A','precision_floor','precision_diagonal_upper',
        'covariance_diagonal_lower','nu_lower','nu_upper','nu_kernel_bound','kappa_bound','sigma_max',
        'correlation_eigenvalue_upper','original_precision_factor_C','original_precision_factor_A',
        'normalized_precision_factor_C','normalized_precision_factor_A','normalization_clamp_inactive',
        'propagated_W1_upper_bound']


def vector(d,delta):
    def pair(key):
        value=d[key]
        return [float('nan'),float('nan')] if value is None else [float(x) for x in value]
    scalar=['precision_eigenvalue_lower','precision_diagonal_upper','covariance_diagonal_lower',
            'nu_lower','nu_upper','nu_correlated_kernel_lipschitz','kappa_lipschitz','sigma_max',
            'correlation_with_jitter_eigenvalue_upper']
    result=pair('state_block_constants')+[float(d['L_D_sum_block_norm'])]
    for key in ['mean_term','scale_and_centering_term','precision_term','degrees_of_freedom_term']:result+=pair(key)
    result +=[float(d[key]) for key in scalar]
    result +=pair('original_precision_factor_block_bound')+pair('normalized_precision_factor_block_bound')
    result +=[float(d['normalization_clamp_inactive']),float(d['state_block_constants'].numpy()@delta) if delta is not None else float('nan')]
    assert len(result)==len(FIELDS)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('work/server_snapshot/f-concord/src'))
    p.add_argument('--protocol',type=Path,default=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1'))
    p.add_argument('--state-results',type=Path,default=Path('work/theory_v3/local_tube_results_v2'))
    p.add_argument('--output',type=Path,default=Path('work/theory_v3/decoder_results'))
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3]);p.add_argument('--seeds',nargs='+',type=int,default=[1,7,19,37,73])
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads);a.output.mkdir(exist_ok=True,parents=True)
    sys.path.insert(0,str(a.source.resolve()))
    from fconcord.config import load_config
    from fconcord.experiment import _model_factory
    import fconcord.model
    spec={'schema':'trefin-explicit-local-decoder-W1-v1','folds':a.folds,'seeds':a.seeds,
          'state_protocol_sha256':digest(a.state_results/'protocol.json'),
          'source_model_sha256':digest(fconcord.model.__file__),
          'decoder_bound_source_sha256':digest(Path(__file__).with_name('decoder_bounds.py')),
          'local_mlp_bound_source_sha256':digest(Path(__file__).with_name('local_mlp_bounds.py')),
          'audit_source_sha256':digest(__file__),'dtype':'float64','fields':FIELDS,
          'axes':['origin','radius','lead','law_and_network_bound','field'],
          'variants':[f'{law}_{network}' for law,network in VARIANTS],
          'metric':'W1 with Euclidean cost on one 32-asset lead, state inputs use the sum of Frobenius block norms',
          'scope':'uniform on each registered local state tube, fixed market and lead; no true-law decoder defect estimated',
          'ideal_law':{'correlation_jitter':0,'chi_floor':0},
          'implemented_law':{'correlation_jitter':2e-5,'chi_floor':1e-8,'covariance_diagonal_floor':1e-8},
          'chi_shared_across_assets_within_lead':True,'signal_shrinkage_applies_to_mean_only':True,
          'test_labels_used':False,'model_fitting':False,'all_state_radii_retained':True}
    sp=a.output/'protocol.json'
    if sp.exists():assert json.loads(sp.read_text())==spec
    else:sp.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    began=time.perf_counter();runs=[]
    for fold in a.folds:
        pp=a.protocol/'runs'/f'fold_{fold}_seed_1'/'audit'
        meta=json.loads((pp/'prepared_metadata.json').read_text())
        with np.load(pp/'prepared_arrays.npz',allow_pickle=False) as z:
            arrays={key:z[key] for key in ['c0','a0','s0','directional_mask','risk_indices']}
        prepared={'metadata':meta,'arrays':arrays}
        for seed in a.seeds:
            out=a.output/f'fold_{fold}_seed_{seed}.json';dataout=out.with_suffix('.npz')
            if out.exists():runs.append(json.loads(out.read_text()));continue
            checkpoint_dir=a.protocol/'runs'/f'fold_{fold}_seed_{seed}'
            cfg=load_config(checkpoint_dir/'config_resolved.yaml')
            model=_model_factory(prepared,cfg).double().eval()
            ck=torch.load(checkpoint_dir/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model_state'],strict=True);model.requires_grad_(False)
            model.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model']['signal_shrinkage']))
            assert not model.gaussian_symmetric_decoder
            statepath=a.state_results/f'fold_{fold}_seed_{seed}.json';state=json.loads(statepath.read_text())
            assert state['checkpoint_sha256']==digest(checkpoint_dir/'checkpoints/best.pt')
            values=np.full((len(state['origins']),4,5,len(VARIANTS),len(FIELDS)),np.nan,dtype=np.float64)
            tick=time.perf_counter()
            with torch.no_grad():
                for oi,origin in enumerate(state['origins']):
                    i=origin['prepared_index'];c=torch.tensor(arrays['c0'][i],dtype=torch.float64);g=torch.tensor(arrays['a0'][i],dtype=torch.float64);s=torch.tensor(arrays['s0'][i],dtype=torch.float64)
                    for h in range(5):
                        c1,g1=model._step(c[None],g[None],s[None],model._lead(1,h,c.device));c,g=c1[0],g1[0]
                        for ri,radius in enumerate(origin['radii']):
                            rc,ra=radius['concept_radius'],radius['graph_radius']
                            delta=np.asarray(radius['matrix_products'][h])@(.1*np.array([rc,ra])) if radius['certified_prefixes'][h] else None
                            for vi,(law,network) in enumerate(VARIANTS):
                                result=decoder_w1_bound(model,c,g,s,h+1,rc,ra,law,network)
                                values[oi,ri,h,vi]=vector(result,delta)
                            # Local variation and spectral bounds are both retained;
                            # the former takes valid minima and cannot worsen a bound.
                            assert np.all(values[oi,ri,h,2,:3]<=values[oi,ri,h,0,:3]*(1+1e-9)+1e-9)
                            assert np.all(values[oi,ri,h,3,:3]<=values[oi,ri,h,1,:3]*(1+1e-9)+1e-9)
            np.savez_compressed(dataout,values=values,fields=np.array(FIELDS),variants=np.array(spec['variants']),
                                radii=np.array([r['concept_radius'] for r in state['origins'][0]['radii']]),
                                test_positions=np.array([o['test_position'] for o in state['origins']]))
            record={'fold':fold,'seed':seed,'checkpoint_sha256':state['checkpoint_sha256'],
                    'state_result_sha256':digest(statepath),'arrays_sha256':digest(dataout),
                    'arrays_file':dataout.name,'gaussian_symmetric_decoder':False,'wall_seconds':time.perf_counter()-tick,
                    'normalization_clamp_inactive_everywhere':bool(np.all(values[...,FIELDS.index('normalization_clamp_inactive')]==1)),
                    'primary_implemented_LD':stat(values[:,2,:,3,FIELDS.index('L_D_sum_block_norm')].ravel()),
                    'smallest_implemented_LD':stat(values[:,0,:,3,FIELDS.index('L_D_sum_block_norm')].ravel())}
            out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');runs.append(record)
            print(json.dumps({'fold':fold,'seed':seed,'seconds':record['wall_seconds'],
                              'primary_LD_max':record['primary_implemented_LD']['maximum'],
                              'smallest_LD_max':record['smallest_implemented_LD']['maximum']}),flush=True)
    summary={'protocol':spec,'runs':runs,'wall_seconds':time.perf_counter()-began,'folds':{}}
    for fold in a.folds:
        grouped=[]
        for run in runs:
            if run['fold']==fold:
                with np.load(a.output/run['arrays_file'],allow_pickle=False) as z:grouped.append(z['values'])
        values=np.concatenate(grouped,axis=0);rr=[]
        for ri,rc in enumerate([.001,.01,.1,1.]):
            vv={}
            for vi,name in enumerate(spec['variants']):
                vv[name]={key:stat(values[:,ri,:,vi,fi].ravel()) for fi,key in enumerate(FIELDS)}
                vv[name]['propagated_W1_upper_bound_by_lead']=[stat(values[:,ri,h,vi,FIELDS.index('propagated_W1_upper_bound')]) for h in range(5)]
            rr.append({'concept_radius':rc,'graph_radius':rc*.5,'variants':vv})
        summary['folds'][str(fold)]={'radii':rr}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','runs':len(runs),'seconds':summary['wall_seconds']}),flush=True)


if __name__=='__main__':main()
