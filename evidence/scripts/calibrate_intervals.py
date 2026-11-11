"""Positive validation-only innovation scaling, with frozen archived models."""
from pathlib import Path
import argparse, os, json, hashlib, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch, yaml
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist
from fconcord.experiment import load_prepared, _model_factory
from fconcord.trainer import collect_predictions
from fconcord.metrics import evaluate_predictions
from fconcord.utils import seed_everything, json_dump

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(args):
    root=Path(args.root).resolve(); torch.set_num_threads(4)
    device=torch.device(args.device)
    proto=root/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'
    out=Path(args.out).resolve() if args.out else root/'rebuttal/interval_calibration'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for fold in [1,2,3]:
        prepared=load_prepared(proto/f'fold_{fold}_seed_1/audit')
        for seed in [1,7,19,37,73]:
            dest=out/f'fold_{fold}_seed_{seed}'; dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():
                records.append(json.loads((dest/'result.json').read_text())); continue
            started=time.time()
            run=proto/f'fold_{fold}_seed_{seed}'
            cfg=yaml.safe_load((run/'config_resolved.yaml').read_text())
            checkpoint=run/'checkpoints/best.pt'; checkpoint_hash=sha256(checkpoint)
            model=_model_factory(prepared,cfg).to(device)
            ck=torch.load(checkpoint,map_location=device,weights_only=False)
            model.load_state_dict(ck['model_state']); model.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model'].get('signal_shrinkage',1.))); model.eval()
            train_cfg=dict(cfg['training']); train_cfg['validation_samples']=256
            seed_everything(271845,True)
            vp=collect_predictions(model,prepared,'val',train_cfg,device)
            eps=(vp['samples']-vp['mu'][None]).astype(float); error=(vp['y']-vp['mu']).astype(float)
            lo,hi=np.quantile(eps,[.05,.95],axis=0)
            if not np.all((lo < 0)&(hi > 0)): raise RuntimeError('Uncentered marginal quantiles prevent nested intervals')
            required=np.where(error>=0,error/hi,error/lo)
            a90=float(np.clip(np.quantile(required,.90,method='higher'),.25,4.0))
            pair=np.mean([pdist(eps[:,b,h]).sum()/(256*255) for b in range(error.shape[0]) for h in range(error.shape[1])])
            def obj(a): return float(np.linalg.norm(a*eps-error[None],axis=-1).mean()-a*pair)
            opt=minimize_scalar(obj,bounds=(.25,4.),method='bounded',options={'xatol':1e-5})
            if not opt.success: raise RuntimeError(f'ES scale optimization failed: {opt.message}')
            aes=min([.25,1.,4.,float(opt.x)],key=obj)
            if obj(aes)>obj(1.)+1e-12: raise RuntimeError('Validation ES increased after minimization')
            selected={'fold':fold,'seed':seed,'rho':model.signal_shrinkage,'scale_coverage90':a90,'scale_es':aes,
                      'validation_samples':256,'bounds':[.25,4.0],'selection_split':'validation',
                      'coverage90_validation_at_selected':float(np.mean(required<=a90)),
                      'validation_es_base':obj(1.),'validation_es_scaled':obj(aes),
                      'risk_response':'positive constant scales entire mean-centered innovation; derivative multiplied by a>0',
                      'test_used_for_selection':False,'device':str(device),'torch_version':torch.__version__,
                      'validation_random_seed':271845,'optimizer_success':bool(opt.success),
                      'coverage_quantile_method':'higher; pooled marginal origin/lead/asset required-scale quantile',
                      'coverage_scale_at_boundary':bool(a90 in [.25,4.]),
                      'conditional_guarantee_scope':'fixed decoder graph, other concepts, market state and lead; scale is frozen, positive, and independent of intervention',
                      'checkpoint_sha256':checkpoint_hash,'script_sha256':sha256(__file__),
                      'config_sha256':sha256(run/'config_resolved.yaml'),
                      'prepared_arrays_sha256':sha256(proto/f'fold_{fold}_seed_1/audit/prepared_arrays.npz')}
            json_dump(selected,dest/'validation_selection.json')
            del vp,eps
            seed_everything(seed+1,True)
            # Verify means, then use ORIGINAL test scenarios for a matched
            # scaling comparison; CPU and original CUDA RNG streams may differ.
            verify_cfg=dict(cfg['training']);verify_cfg['test_samples']=2
            replay=collect_predictions(model,prepared,'test',verify_cfg,device)
            archive=run/'predictions/test_predictions.npz'
            with np.load(archive,allow_pickle=False) as saved:
                p={key:saved[key] for key in ['mu','samples','y','indices']}
            if not np.array_equal(replay['indices'],p['indices']) or not np.array_equal(replay['y'],p['y']):
                raise RuntimeError('Replayed test origins/targets do not match the archive')
            mean_difference=float(np.abs(replay['mu']-p['mu']).max())
            if mean_difference>=1e-7:raise RuntimeError(f'Replayed test means differ: {mean_difference}')
            selected['test_mean_replay_max_absolute_error']=mean_difference
            selected['test_predictions_sha256']=sha256(archive)
            selected['test_scenario_source']='original archived test scenarios; no new test sampling used in reported metrics'
            del replay
            pred_eps=p['samples']-p['mu'][None]
            results={}
            for name,a in [('original',1.),('validation_ES_scale',aes),('validation_90pct_scale',a90)]:
                results[name]=evaluate_predictions(p['y'],p['mu'],p['mu'][None]+a*pred_eps,[.5,.8,.9,.95,.99])
                results[name]['mse']=float(np.mean((p['mu'].astype(np.float64)-p['y'].astype(np.float64))**2))
            if sha256(checkpoint)!=checkpoint_hash:raise RuntimeError('Checkpoint changed')
            record={**selected,'test_metrics':results,'seconds':time.time()-started}
            json_dump(record,dest/'result.json'); records.append(record)
            print(json.dumps({'fold':fold,'seed':seed,'a90':a90,'aes':aes,'coverages':{k:v['coverage']['0.9'] for k,v in results.items()}}),flush=True)
            del p,pred_eps,model
            if device.type=='cuda':torch.cuda.empty_cache()
    json_dump({'runs':records,'scope':'post-submission validation-only recalibration; original rho and checkpoints fixed'},out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--device',default='cuda');p.add_argument('--out');main(p.parse_args())
