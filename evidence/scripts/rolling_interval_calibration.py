"""Prespecified causal rolling scale: 126 completed origins, pooled 90% score.

This is a sequential calibration diagnostic, not the frozen static protocol and
not a conformal coverage theorem. Each forecast enters calibration only once all
five of its labels have been realized: source raw origin + H <= current origin.
"""
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import yaml

from fconcord.experiment import load_prepared, _model_factory
from fconcord.trainer import collect_predictions
from fconcord.metrics import evaluate_predictions
from fconcord.utils import seed_everything, json_dump


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def conformity(prediction):
    eps=prediction['samples'].astype(np.float64)-prediction['mu'][None].astype(np.float64)
    lo,hi=np.quantile(eps,[.05,.95],axis=0)
    if not np.all((lo<0)&(hi>0)):raise ValueError('Central intervals must straddle the conditional mean')
    error=prediction['y'].astype(np.float64)-prediction['mu'].astype(np.float64)
    return np.where(error>=0,error/hi,error/lo)


def causal_scales(validation_origins, validation_scores, test_origins, test_scores, horizon=5, window=126):
    vo=np.asarray(validation_origins,dtype=int);to=np.asarray(test_origins,dtype=int)
    if len(np.unique(vo))!=len(vo) or np.any(np.diff(vo)<=0):raise ValueError('Validation origins not chronological')
    if len(np.unique(to))!=len(to) or np.any(np.diff(to)<=0):raise ValueError('Test origins not chronological')
    if vo.max()>=to.min():raise ValueError('Validation must precede test')
    pool=deque(maxlen=window);vp=tp=0;scales=[];audit=[]
    for b,t in enumerate(to):
        while vp<len(vo) and vo[vp]+horizon<=t:
            pool.append(('validation',int(vo[vp]),validation_scores[vp]));vp+=1
        while tp<b and to[tp]+horizon<=t:
            pool.append(('test',int(to[tp]),test_scores[tp]));tp+=1
        if len(pool)!=window:raise ValueError(f'Need {window} completed origins, have {len(pool)} at {t}')
        calibration_origins=np.array([x[1] for x in pool])
        if np.any(calibration_origins+horizon>t):raise ValueError('Future target leaked into calibration')
        if np.any(np.diff(calibration_origins)<=0):raise ValueError('Calibration window not chronological')
        required=np.stack([x[2] for x in pool])
        scale=float(np.clip(np.quantile(required,.90,method='higher'),.25,4.))
        scales.append(scale)
        audit.append({'current_raw_origin':int(t),'window_first_raw_origin':int(calibration_origins[0]),
                      'window_last_raw_origin':int(calibration_origins[-1]),
                      'last_realized_label_row':int(calibration_origins.max()+horizon),
                      'completed_forecast_origins':len(pool),'test_forecast_origins_in_window':sum(x[0]=='test' for x in pool),
                      'scale':scale})
    return np.array(scales),audit


def run(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device(args.device)
    proto=root/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'
    records=[]
    for fold in [1,2,3]:
        prepared=load_prepared(proto/f'fold_{fold}_seed_1/audit')
        raw_origins=prepared['arrays']['origins'];horizon=int(prepared['metadata']['horizon'])
        for seed in [1,7,19,37,73]:
            dest=out/f'fold_{fold}_seed_{seed}';dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():
                records.append(json.loads((dest/'result.json').read_text()));continue
            started=time.time();run_path=proto/f'fold_{fold}_seed_{seed}'
            config=yaml.safe_load((run_path/'config_resolved.yaml').read_text())
            checkpoint=run_path/'checkpoints/best.pt';checkpoint_hash=sha256(checkpoint)
            model=_model_factory(prepared,config).to(device)
            ck=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(ck['model_state'])
            model.signal_shrinkage=float(ck.get('signal_shrinkage',config['model'].get('signal_shrinkage',1.)));model.eval()
            rho=model.signal_shrinkage
            calibration_config=dict(config['training']);calibration_config['validation_samples']=256
            seed_everything(271845,True)
            vp=collect_predictions(model,prepared,'val',calibration_config,device)
            val_scores=conformity(vp);val_origins=raw_origins[vp['indices']]
            del vp,model
            archive=run_path/'predictions/test_predictions.npz'
            with np.load(archive,allow_pickle=False) as f:
                prediction={k:f[k] for k in ['samples','mu','y','indices']}
            test_origins=raw_origins[prediction['indices']];test_scores=conformity(prediction)
            scales,audit=causal_scales(val_origins,val_scores,test_origins,test_scores,horizon,126)
            sample=prediction['mu'][None]+scales[None,:,None,None]*(prediction['samples']-prediction['mu'][None])
            metric=evaluate_predictions(prediction['y'],prediction['mu'],sample,[.5,.8,.9,.95,.99])
            metric['mse']=float(np.mean((prediction['mu'].astype(float)-prediction['y'].astype(float))**2))
            np.savez_compressed(dest/'scores_and_scales.npz',validation_raw_origins=val_origins,validation_scores=val_scores,
                                test_raw_origins=test_origins,test_scores=test_scores,scales=scales,test_indices=prediction['indices'])
            json_dump(audit,dest/'causal_origin_audit.json')
            if sha256(checkpoint)!=checkpoint_hash:raise ValueError('Checkpoint changed')
            record={'fold':fold,'seed':seed,'rho':rho,'method':'causal rolling positive innovation scale',
                    'protocol':'sequential adaptation using only completely realized past five-step forecast errors',
                    'window_completed_origins':126,'quantile':.90,'quantile_method':'higher','bounds':[.25,4.],
                    'selection_rule_fixed_before_test_evaluation':True,'hyperparameter_search':False,
                    'validation_scenarios':256,'validation_seed':271845,'test_scenarios':'original archive, 256 scenarios',
                    'maturity_rule':'source_raw_origin + 5 <= current_raw_origin',
                    'point_forecasts_unchanged':True,'coverage_theorem_claimed':False,
                    'risk_response_scope':'at each current forecast, a_t>0 is determined by past data and held fixed during a state intervention; d(a_t sigma_minus)/dc=a_t d(sigma_minus)/dc',
                    'scale_mean':float(scales.mean()),'scale_min':float(scales.min()),'scale_max':float(scales.max()),
                    'scale_boundary_fraction':float(np.mean((scales==.25)|(scales==4.))),
                    'maximum_availability_violation':max(x['last_realized_label_row']-x['current_raw_origin'] for x in audit),
                    'test_metrics':metric,'checkpoint_sha256':checkpoint_hash,'predictions_sha256':sha256(archive),
                    'script_sha256':sha256(__file__),'device':str(device),'torch_version':torch.__version__,
                    'seconds':time.time()-started}
            json_dump(record,dest/'result.json');records.append(record)
            print(json.dumps({'fold':fold,'seed':seed,'coverage90':metric['coverage']['0.9'],'es':metric['energy_score'],'scale_mean':record['scale_mean']}),flush=True)
            del prediction,sample,test_scores,val_scores
    json_dump({'runs':records,'scope':'prespecified sequential calibration; compare separately from frozen static protocol'},out/'summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    run(p.parse_args())
