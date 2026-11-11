"""Read-only, prespecified analytic rollout-neighborhood audit of 15 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

from tube_bounds import state_constants, tube_bound, improved_global_bound


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def summary(x):
    x=np.asarray(x,dtype=float)
    if x.size==0:
        return {'n':0,'minimum':None,'median':None,'q95':None,'maximum':None}
    return {'n':int(x.size),'minimum':float(x.min()),'median':float(np.median(x)),
            'q95':float(np.quantile(x,.95)),'maximum':float(x.max())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('work/server_snapshot/f-concord/src'))
    p.add_argument('--protocol',type=Path,default=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1'))
    p.add_argument('--output',type=Path,default=Path('work/theory_v3/local_tube_results'))
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,7,19,37,73])
    p.add_argument('--origins',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0,str(a.source.resolve()))
    from fconcord.config import load_config
    from fconcord.experiment import _model_factory
    source=Path(importlib.import_module('fconcord.model').__file__)
    a.output.mkdir(parents=True,exist_ok=True)
    radii=[.001,.01,.1,1.0]
    spec={'schema':'trefin-uniform-local-rollout-tube-v1','source_model':str(source.resolve()),
          'source_model_sha256':digest(source),'bound_script_sha256':digest(Path(__file__).with_name('tube_bounds.py')),
          'audit_script_sha256':digest(__file__),'folds':a.folds,'seeds':a.seeds,'origins_per_fold':a.origins,
          'origin_selection':'evenly spaced chronological held-out positions, same for every seed',
          'concept_radii':radii,'graph_to_concept_radius':.5,'primary_concept_radius':.1,
          'initial_fraction':.1,'state_metric':'weighted maximum of concept/graph Frobenius block norms',
          'held_fixed':['market state','fitted parameters','lead schedule','positive mean scalar'],
          'uses_test_labels':False,'fitting_or_selection':False,'numeric_dtype':'float64',
          'numerical_cushion':{'relative':1e-10,'absolute':1e-12},
          'scope':'uniform analytic local-neighborhood bounds, distinct from registered-domain bounds and empirical point Jacobians',
          'no_reference_transition_defect_or_Wasserstein_constant_claim':True}
    specpath=a.output/'protocol.json'
    if specpath.exists():
        old=json.loads(specpath.read_text())
        if old!=spec:raise ValueError('Existing protocol differs; choose a fresh output directory')
    else:specpath.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter();all_records=[]
    for fold in a.folds:
        prep=a.protocol/'runs'/f'fold_{fold}_seed_1'/'audit'
        metadata=json.loads((prep/'prepared_metadata.json').read_text())
        with np.load(prep/'prepared_arrays.npz',allow_pickle=False) as z:
            arrays={key:z[key] for key in ['c0','a0','s0','directional_mask','risk_indices','origins']}
        prepared={'metadata':metadata,'arrays':arrays}
        tests=np.asarray(metadata['split_indices']['test'],dtype=np.int64)
        positions=np.linspace(0,len(tests)-1,a.origins,dtype=int)
        selected=tests[positions]
        for seed in a.seeds:
            out=a.output/f'fold_{fold}_seed_{seed}.json'
            if out.exists():
                all_records.append(json.loads(out.read_text()));continue
            run=a.protocol/'runs'/f'fold_{fold}_seed_{seed}'
            config=load_config(run/'config_resolved.yaml')
            model=_model_factory(prepared,config).double().eval()
            checkpoint=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model_state'],strict=True)
            model.signal_shrinkage=float(checkpoint.get('signal_shrinkage',config['model']['signal_shrinkage']))
            model.requires_grad_(False)
            constants=state_constants(model)
            with torch.no_grad():
                old_matrix,old_radius=model.structural_gain_bound()
                new_matrix=improved_global_bound(model,constants)
                new_radius=torch.linalg.eigvals(new_matrix).real.max()
            with np.load(run/'predictions/test_predictions.npz',allow_pickle=False) as z:
                stored_mean=z['mu'][positions]
                assert np.array_equal(z['indices'][positions],selected)
            per_origin=[];max_replay=0.
            t0=time.perf_counter()
            with torch.no_grad():
                for pos,index in zip(positions,selected):
                    c=torch.as_tensor(arrays['c0'][index],dtype=torch.float64)
                    graph=torch.as_tensor(arrays['a0'][index],dtype=torch.float64)
                    market=torch.as_tensor(arrays['s0'][index],dtype=torch.float64)
                    states=[(c,graph)]
                    means=[]
                    for h in range(model.horizon):
                        c1,g1=model._step(c[None],graph[None],market[None],model._lead(1,h,c.device))
                        c,graph=c1[0],g1[0];states.append((c,graph))
                        means.append(model._decode_state(c1,g1,market[None],model._lead(1,h+1,c.device))['mu'][0])
                    replay=float(np.abs(torch.stack(means).numpy()-stored_mean[len(per_origin)]).max())
                    max_replay=max(max_replay,replay)
                    assert replay<1e-6,('Archived-mean replay mismatch',fold,seed,pos,replay)
                    radius_records=[]
                    for rc in radii:
                        ra=.5*rc;rv=np.array([rc,ra]);matrix_list=[];mean_list=[]
                        for h in range(model.horizon):
                            m,_=tube_bound(model,*states[h],market,h,rc,ra,constants)
                            _,dm=tube_bound(model,*states[h+1],market,h+1,rc,ra,constants)
                            matrix_list.append(m.numpy());mean_list.append(dm.numpy())
                        product=np.eye(2);products=[];gains=[];q=[];mean_bounds=[]
                        for m,dm in zip(matrix_list,mean_list):
                            product=m@product;products.append(product.copy())
                            gains.append(float((product@rv/rv).max()))
                            q.append(float((m@rv/rv).max()))
                            mean_bounds.append(float(dm@(product@(.1*rv))))
                        prefixes=np.maximum.accumulate(gains)<=10
                        radius_records.append({'concept_radius':rc,'graph_radius':ra,'block_matrices':matrix_list,
                            'matrix_products':products,'weighted_one_step_bounds':q,'weighted_h_step_bounds':gains,
                            'initial_fraction':.1,'all_five_steps_certified':max(gains)<=10,
                            'full_radius_five_step_tube_certified':max(gains)<=1,
                            'same_radius_invariant':max(q)<=1,'uniform_contraction_bound':max(q),
                            'certified_prefixes':prefixes.tolist(),
                            'mean_difference_l2_bounds_from_initial_fraction':[
                                value if valid else None for value,valid in zip(mean_bounds,prefixes)]})
                    per_origin.append({'test_position':int(pos),'prepared_index':int(index),
                                       'original_origin_row':int(arrays['origins'][index]),'radii':radius_records})
            rec={'fold':fold,'seed':seed,'checkpoint_sha256':digest(run/'checkpoints/best.pt'),
                 'config_sha256':digest(run/'config_resolved.yaml'),'prepared_metadata_sha256':digest(prep/'prepared_metadata.json'),
                 'signal_shrinkage':model.signal_shrinkage,'registered_domain_original_matrix':old_matrix.numpy(),
                 'registered_domain_original_radius':float(old_radius),'registered_domain_improved_matrix':new_matrix.numpy(),
                 'registered_domain_improved_radius':float(new_radius),'archived_mean_replay_max_abs':max_replay,
                 'origins':per_origin,'wall_seconds':time.perf_counter()-t0}
            def conv(value):
                if isinstance(value,np.ndarray):return value.tolist()
                raise TypeError(type(value))
            out.write_text(json.dumps(rec,default=conv,indent=2)+'\n',encoding='utf-8')
            all_records.append(json.loads(out.read_text()))
            primary=[o['radii'][2] for o in per_origin]
            print(json.dumps({'fold':fold,'seed':seed,'seconds':rec['wall_seconds'],
                              'old_radius':float(old_radius),'improved_global_radius':float(new_radius),
                              'primary_tubes_certified':sum(r['all_five_steps_certified'] for r in primary),
                              'same_radius_invariant':sum(r['same_radius_invariant'] for r in primary),
                              'primary_max_h5_bound':max(r['weighted_h_step_bounds'][-1] for r in primary)}),flush=True)
            del checkpoint,model
    aggregate={'protocol':spec,'runs':len(all_records),'wall_seconds':time.perf_counter()-started,
               'original_global_radius':summary([r['registered_domain_original_radius'] for r in all_records]),
               'improved_global_radius':summary([r['registered_domain_improved_radius'] for r in all_records]),'folds':{}}
    for fold in a.folds:
        rr=[r for r in all_records if r['fold']==fold];entries=[]
        for ri,rc in enumerate(radii):
            values=[o['radii'][ri] for r in rr for o in r['origins']]
            entries.append({'concept_radius':rc,'graph_radius':rc*.5,'trajectories':len(values),
                'certified_five_step_count':sum(v['all_five_steps_certified'] for v in values),
                'full_radius_five_step_tube_certified_count':sum(v['full_radius_five_step_tube_certified'] for v in values),
                'same_radius_invariant_count':sum(v['same_radius_invariant'] for v in values),
                'uniform_one_step_contraction_bound':summary([v['uniform_contraction_bound'] for v in values]),
                'h_step_amplification_bounds':[summary([v['weighted_h_step_bounds'][h] for v in values]) for h in range(5)],
                'mean_l2_difference_bound_basis_points':[summary([
                    1e4*v['mean_difference_l2_bounds_from_initial_fraction'][h] for v in values
                    if v['certified_prefixes'][h]]) for h in range(5)]})
        aggregate['folds'][str(fold)]={'radii':entries}
    (a.output/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','runs':len(all_records),'output':str(a.output.resolve())}),flush=True)


if __name__=='__main__':main()
