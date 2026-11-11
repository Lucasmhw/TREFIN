"""Matched post-review structural retraining; no held-out tuning."""
from pathlib import Path
import os, argparse, json, copy, hashlib, time, shutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml
from scipy.spatial.distance import pdist
from fconcord.experiment import load_prepared, _model_factory
from fconcord.trainer import train_model, checkpoint_reload_check, collect_predictions
from fconcord.metrics import evaluate_predictions
from fconcord.utils import seed_everything, setup_logger, json_dump, environment_snapshot

def state_metrics(p):
    c=p['concepts'][:,1:]; ct=p['c_target'][:,1:]
    a=p['graphs'][:,1:]; at=p['a_target'][:,1:]
    n=a.shape[-1]; off=~np.eye(n,dtype=bool)
    ap=np.abs(a[...,off])>.05; tp=np.abs(at[...,off])>.05
    inter=(ap&tp).sum(-1); union=(ap|tp).sum(-1)
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    return {'concept_mae_lead_coordinate':np.abs(c-ct).mean((0,2)).tolist(),
            'concept_persistence_mae_lead_coordinate':np.abs(p['c0'][:,None]-ct).mean((0,2)).tolist(),
            'graph_iou_lead':iou.mean(0).tolist(),
            'graph_mae_lead':np.abs(a[...,off]-at[...,off]).mean((0,2)).tolist(),
            'graph_threshold':.05}

def origin_losses(p):
    s=p['samples']; y=p['y']; mu=p['mu']
    es=np.linalg.norm(s-y[None],axis=-1).mean(0)
    for b in range(y.shape[0]):
        for h in range(y.shape[1]):
            es[b,h]-=pdist(s[:,b,h].astype(float)).sum()/(s.shape[0]*(s.shape[0]-1))
    return {'indices':p['indices'],'mse':np.mean((mu-y)**2,axis=(1,2)),
            'mae':np.mean(np.abs(mu-y),axis=(1,2)),'es':es.mean(1),'mu':mu,'y':y}

def run(args):
    root=Path(args.root).resolve(); torch.set_num_threads(args.threads)
    prot=root/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'
    out=root/('rebuttal/retraining_tuned' if args.selections else 'rebuttal/retraining'); device=torch.device('cuda')
    selections=json.loads(Path(args.selections).read_text())['selections'] if args.selections else None
    for fold in args.folds:
        prep=load_prepared(prot/f'fold_{fold}_seed_1'/'audit')
        for seed in args.seeds:
            for variant in args.variants:
                d=out/f'fold_{fold}_seed_{seed}'/variant
                if (d/'complete.json').exists(): continue
                if selections and not any(selections[variant]['best_overrides'].values()):
                    fixed=root/'rebuttal/retraining'/f'fold_{fold}_seed_{seed}'/variant
                    if (fixed/'complete.json').exists():
                        shutil.copytree(fixed,d,dirs_exist_ok=True)
                        json_dump({'source':str(fixed),'reason':'selected configuration equals fixed original configuration; exact existing run reused'},d/'audit/reuse.json')
                        continue
                for sub in ['checkpoints','logs','predictions','audit','tables']: (d/sub).mkdir(parents=True,exist_ok=True)
                cfg=yaml.safe_load((prot/f'fold_{fold}_seed_{seed}'/'config_resolved.yaml').read_text())
                if selections:
                    for section, changes in selections[variant]['best_overrides'].items(): cfg[section].update(changes)
                cfg['model']['signal_shrinkage']=1.0
                cfg['model']['ablation_variant']=variant
                cfg['experiment'].update(name=f'rebuttal_fold_{fold}_seed_{seed}_{variant}',seed=seed,device='cuda')
                if variant in ['NoGraph','NoStateSupervision']: cfg['training']['lambda_graph']=0.0
                if variant=='NoStateSupervision': cfg['training']['lambda_concept']=0.0
                cfg['training']['signal_shrinkage_calibration']=True
                (d/'config_before_training.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                source=root/'src/fconcord/model.py'
                json_dump({'model_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                           'prepared_sha256':hashlib.sha256((prot/f'fold_{fold}_seed_1/audit/prepared_arrays.npz').read_bytes()).hexdigest(),
                           'selection':'original 20-epoch ceiling, validation ES, same positive rho for all variants',
                           'test_used_for_selection':False},d/'audit/provenance.json')
                seed_everything(seed,True); logger=setup_logger(d/'logs/run.log')
                factory=lambda:_model_factory(prep,cfg)
                model=factory()
                if selections and fold==1 and seed==1:
                    selection=selections[variant]
                    source_checkpoint=Path(selection['selected_checkpoint'])
                    assert hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()==selection['selected_checkpoint_sha256']
                    shutil.copy2(source_checkpoint,d/'checkpoints/best.pt')
                    source_run=source_checkpoint.parents[1]
                    if (source_run/'checkpoints/last.pt').exists(): shutil.copy2(source_run/'checkpoints/last.pt',d/'checkpoints/last.pt')
                    shutil.copy2(source_run/'logs/training_history.csv',d/'logs/training_history.csv')
                    selected_trial=next(x for x in selection['trials'] if x['candidate']==selection['selected_candidate'])
                    summary=selected_trial['training_summary']
                    ck=torch.load(d/'checkpoints/best.pt',map_location=device,weights_only=False)
                    model=model.to(device); model.load_state_dict(ck['model_state']); model.signal_shrinkage=float(ck['signal_shrinkage'])
                    json_dump({'source':str(source_checkpoint),'reason':'validation-selected seed1 checkpoint reused before its first test evaluation'},d/'audit/reuse.json')
                else:
                    model,history,summary=train_model(model,prep,cfg['training'],d,device,logger)
                cfg['model']['signal_shrinkage']=model.signal_shrinkage
                (d/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                reloaded,check=checkpoint_reload_check(factory,d/'checkpoints/best.pt',model,prep,cfg['training'],device)
                json_dump(check,d/'audit/checkpoint_reload_check.json')
                assert check['passed']
                # Validation artifacts are saved before the final test evaluation.
                seed_everything(271845,True)
                vp=collect_predictions(reloaded,prep,'val',cfg['training'],device)
                np.savez_compressed(d/'predictions/validation_predictions.npz',**vp)
                del vp
                logger.info('final_test_started checkpoint=best.pt')
                seed_everything(seed+1,True)
                p=collect_predictions(reloaded,prep,'test',cfg['training'],device)
                np.savez_compressed(d/'predictions/test_predictions.npz',**p)
                m=evaluate_predictions(p['y'],p['mu'],p['samples'],[0.9])
                rho=model.signal_shrinkage
                un_mu=p['mu']/rho
                # Exact inversion of the positive translation, using frozen innovations.
                un_samples=p['samples']+(un_mu-p['mu'])[None]
                um=evaluate_predictions(p['y'],un_mu,un_samples,[0.9])
                json_dump(m,d/'tables/calibrated.json'); json_dump(um,d/'tables/uncalibrated.json')
                json_dump(state_metrics(p),d/'tables/states.json')
                np.savez_compressed(d/'tables/origin_losses.npz',**origin_losses(p))
                json_dump({'fold':fold,'seed':seed,'variant':variant,'rho':rho,'training':summary,'calibrated':m,'uncalibrated':um},d/'complete.json')
                logger.info('COMPLETE fold=%d seed=%d variant=%s mse=%.9g es=%.9g',fold,seed,variant,m['mse'],m['energy_score'])
                del p,model,reloaded,un_samples; torch.cuda.empty_cache()
    json_dump(environment_snapshot(),out/f'environment_{os.getpid()}.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--folds',nargs='+',type=int,default=[1,2,3]); p.add_argument('--seeds',nargs='+',type=int,default=[1,7,19,37,73]); p.add_argument('--variants',nargs='+',default=['Full','NoGraph','SingleSigned','SingleUnsigned','NoStateSupervision']); p.add_argument('--threads',type=int,default=4); p.add_argument('--selections'); run(p.parse_args())
