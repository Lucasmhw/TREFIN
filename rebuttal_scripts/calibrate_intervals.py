"""Positive validation-only innovation scaling, with frozen archived models."""
from pathlib import Path
import argparse, os, json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch, yaml
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist
from fconcord.experiment import load_prepared, _model_factory
from fconcord.trainer import collect_predictions
from fconcord.metrics import evaluate_predictions
from fconcord.utils import seed_everything, json_dump

def main(args):
    root=Path(args.root).resolve(); torch.set_num_threads(4)
    proto=root/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'
    out=root/'rebuttal/interval_calibration'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for fold in [1,2,3]:
        prepared=load_prepared(proto/f'fold_{fold}_seed_1/audit')
        for seed in [1,7,19,37,73]:
            dest=out/f'fold_{fold}_seed_{seed}'; dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():
                records.append(json.loads((dest/'result.json').read_text())); continue
            run=proto/f'fold_{fold}_seed_{seed}'
            cfg=yaml.safe_load((run/'config_resolved.yaml').read_text())
            model=_model_factory(prepared,cfg).cuda()
            ck=torch.load(run/'checkpoints/best.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(ck['model_state']); model.signal_shrinkage=float(ck.get('signal_shrinkage',cfg['model']['signal_shrinkage'])); model.eval()
            train_cfg=dict(cfg['training']); train_cfg['validation_samples']=256
            seed_everything(271845,True)
            vp=collect_predictions(model,prepared,'val',train_cfg,torch.device('cuda'))
            eps=(vp['samples']-vp['mu'][None]).astype(float); error=(vp['y']-vp['mu']).astype(float)
            lo,hi=np.quantile(eps,[.05,.95],axis=0)
            if not np.all((lo < 0)&(hi > 0)): raise RuntimeError('Uncentered marginal quantiles prevent nested intervals')
            required=np.where(error>=0,error/hi,error/lo)
            a90=float(np.clip(np.quantile(required,.90,method='higher'),.25,4.0))
            pair=np.mean([pdist(eps[:,b,h]).sum()/(256*255) for b in range(error.shape[0]) for h in range(error.shape[1])])
            def obj(a): return float(np.linalg.norm(a*eps-error[None],axis=-1).mean()-a*pair)
            opt=minimize_scalar(obj,bounds=(.25,4.),method='bounded',options={'xatol':1e-5})
            aes=min([.25,4.,float(opt.x)],key=obj)
            selected={'fold':fold,'seed':seed,'rho':model.signal_shrinkage,'scale_coverage90':a90,'scale_es':aes,
                      'validation_samples':256,'bounds':[.25,4.0],'selection_split':'validation',
                      'coverage90_validation_at_selected':float(np.mean(required<=a90)),
                      'validation_es_base':obj(1.),'validation_es_scaled':obj(aes),
                      'risk_response':'positive constant scales entire mean-centered innovation; derivative multiplied by a>0',
                      'test_used_for_selection':False}
            json_dump(selected,dest/'validation_selection.json')
            del vp,eps
            seed_everything(seed+1,True)
            p=collect_predictions(model,prepared,'test',cfg['training'],torch.device('cuda'))
            pred_eps=p['samples']-p['mu'][None]
            results={}
            for name,a in [('original',1.),('validation_ES_scale',aes),('validation_90pct_scale',a90)]:
                results[name]=evaluate_predictions(p['y'],p['mu'],p['mu'][None]+a*pred_eps,[.5,.8,.9,.95,.99])
            record={**selected,'test_metrics':results}
            json_dump(record,dest/'result.json'); records.append(record)
            print(json.dumps({'fold':fold,'seed':seed,'a90':a90,'aes':aes,'coverages':{k:v['coverage']['0.9'] for k,v in results.items()}}),flush=True)
            del p,pred_eps,model; torch.cuda.empty_cache()
    json_dump({'runs':records,'scope':'post-submission validation-only recalibration; original rho and checkpoints fixed'},out/'summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');main(p.parse_args())
