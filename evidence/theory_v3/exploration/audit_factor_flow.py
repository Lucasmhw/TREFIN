"""Isolated full-cohort comparison of the normalized covariance-factor flow."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from normalized_factor_flow import precision_flow_bound

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(x):
    x=np.asarray(x);assert np.isfinite(x).all()
    return dict(n=int(x.size),minimum=float(x.min()),median=float(np.median(x)),q95=float(np.quantile(x,.95)),maximum=float(x.max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--all-radii',action='store_true');a=p.parse_args()
    base=Path(__file__).resolve().parent;root=base.parent;out=base/('all_radius_flow_results' if a.all_radii else 'primary_flow_results');out.mkdir(exist_ok=True)
    source=Path('work/server_snapshot/f-concord/src');protocol=Path('work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1')
    sys.path.insert(0,str(source.resolve()));torch.set_num_threads(4)
    from fconcord.config import load_config
    from fconcord.experiment import _model_factory
    fields=json.loads((root/'decoder_results/protocol.json').read_text())['fields']
    ris=[0,1,2,3] if a.all_radii else [2]
    spec={'scope':'exploratory uniform factor-flow coupling, full existing cohort; no fitting or test labels',
          'radii_indices':ris,'radius_values':[.001,.01,.1,1.],'dtype':'float64','source_sha256':sha(base/'normalized_factor_flow.py'),
          'runner_sha256':sha(__file__),'base_protocol_sha256':sha(root/'decoder_results/protocol.json')}
    file=out/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==spec
    else:file.write_text(json.dumps(spec,indent=2)+'\n')
    summary={'protocol':spec,'folds':{}};tick=time.perf_counter()
    for fold in [1,2,3]:
        pp=protocol/'runs'/f'fold_{fold}_seed_1'/'audit';meta=json.loads((pp/'prepared_metadata.json').read_text())
        with np.load(pp/'prepared_arrays.npz') as z:arrays={k:z[k] for k in ['c0','a0','s0','directional_mask','risk_indices']}
        collected=[]
        for seed in [1,7,19,37,73]:
            dest=out/f'fold_{fold}_seed_{seed}.npz'
            if dest.exists():
                with np.load(dest) as z:collected.append(z['values']);continue
            rp=protocol/'runs'/f'fold_{fold}_seed_{seed}';cfg=load_config(rp/'config_resolved.yaml')
            m=_model_factory({'metadata':meta,'arrays':arrays},cfg).double().eval()
            ck=torch.load(rp/'checkpoints/best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state']);m.requires_grad_(False)
            m.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model']['signal_shrinkage']))
            state=json.loads((root/f'local_tube_results_v2/fold_{fold}_seed_{seed}.json').read_text())
            with np.load(root/f'decoder_results/fold_{fold}_seed_{seed}.npz') as z:old=z['values'][:,:,:,3]
            result=np.zeros((32,len(ris),5,16));start=time.perf_counter()
            with torch.no_grad():
                for oi,origin in enumerate(state['origins']):
                    idx=origin['prepared_index'];c,g,s=[torch.tensor(arrays[k][idx],dtype=torch.float64) for k in ['c0','a0','s0']]
                    for h in range(5):
                        cc,gg=m._step(c[None],g[None],s[None],m._lead(1,h,c.device));c,g=cc[0],gg[0]
                        for ri_index,ri in enumerate(ris):
                            radius=origin['radii'][ri];rc,ra=radius['concept_radius'],radius['graph_radius']
                            new=precision_flow_bound(m,c,g,s,h+1,rc,ra);v=old[oi,ri,h]
                            sigma=v[fields.index('sigma_max')];baseprec=v[[fields.index('precision_C'),fields.index('precision_A')]]
                            flowprec=sigma*new['block_bound'].numpy();prec=np.minimum(baseprec,flowprec)
                            rest=sum(v[[fields.index(f'{key}_C'),fields.index(f'{key}_A')]] for key in ['mean','scale','nu'])
                            total=(rest+prec)*(1+1e-10)+1e-12;original=v[:2]
                            assert np.all(total<=original*(1+1e-10)+1e-10)
                            delta=np.array(radius['matrix_products'][h])@(.1*np.array([rc,ra]))
                            result[oi,ri_index,h]=[*original,*total,*baseprec,*flowprec,new['rmax'],new['graph_relative'],new['lower_eigenvalue'],
                                                  float(total@delta),float(original@delta),*new['q_term'].numpy(),new['lambda_term'][1].item()]
            np.savez_compressed(dest,values=result,fields=np.array(['old_L_C','old_L_A','new_L_C','new_L_A','old_precision_C','old_precision_A','flow_precision_C','flow_precision_A','rmax','graph_relative','precision_floor','new_W1','old_W1','q_C','q_A','lambda_A']))
            (dest.with_suffix('.json')).write_text(json.dumps({'fold':fold,'seed':seed,'checkpoint_sha256':sha(rp/'checkpoints/best.pt'),'base_npz_sha256':sha(root/f'decoder_results/fold_{fold}_seed_{seed}.npz'),'npz_sha256':sha(dest),'seconds':time.perf_counter()-start},indent=2)+'\n')
            collected.append(result);print(json.dumps({'fold':fold,'seed':seed,'old_LD_max':float(result[...,:2].max()),'new_LD_max':float(result[...,2:4].max()),'seconds':time.perf_counter()-start}),flush=True)
        x=np.concatenate(collected);rr=[]
        for rix,ri in enumerate(ris):
            old=x[:,rix,:,:2].max(-1);new=x[:,rix,:,2:4].max(-1)
            rr.append({'concept_radius':spec['radius_values'][ri],'old_LD':stat(old),'new_LD':stat(new),'new_over_old':stat(new/old),
                       'new_lead5_W1_bp':stat(x[:,rix,4,11]*1e4),'old_lead5_W1_bp':stat(x[:,rix,4,12]*1e4)})
        summary['folds'][str(fold)]=rr
    summary['seconds']=time.perf_counter()-tick;(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('complete',flush=True)

if __name__=='__main__':main()
