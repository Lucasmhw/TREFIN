"""Train/evaluate one registered selected configuration after the six-choice freeze."""
from pathlib import Path
import argparse,json,os,shutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml
from scipy.spatial.distance import pdist
from fconcord.experiment import load_prepared,_model_factory
from fconcord.trainer import train_model,checkpoint_reload_check,collect_predictions
from fconcord.metrics import evaluate_predictions
from fconcord.utils import seed_everything,setup_logger,environment_snapshot
from protocol import (VARIANTS,SEEDS,CANDIDATES,configure,sha256,dump,runtime_sources,
  reference_config,prepared_hashes,effective_fingerprint,train_validation_only,selection_key,validate_frozen)

def state_metrics(p):
    c=p['concepts'][:,1:].astype(np.float64);ct=p['c_target'][:,1:].astype(np.float64)
    a=p['graphs'][:,1:].astype(np.float64);at=p['a_target'][:,1:].astype(np.float64)
    off=~np.eye(a.shape[-1],dtype=bool)
    def graph_metrics(prediction):
        ap=np.abs(prediction[...,off])>.05;tp=np.abs(at[...,off])>.05
        union=(ap|tp).sum(-1);intersection=(ap&tp).sum(-1)
        iou=np.divide(intersection,union,out=np.zeros_like(intersection,dtype=float),where=union>0)
        error=prediction[...,off]-at[...,off]
        return {'iou_lead':iou.mean(0).tolist(),'mae_lead':np.abs(error).mean((0,2)).tolist(),
          'rmse_lead':np.sqrt(np.square(error).mean((0,2))).tolist()}
    g=graph_metrics(a);persistence=graph_metrics(np.broadcast_to(p['a0'][:,None].astype(np.float64),at.shape))
    return {'concept_mae_lead_coordinate':np.abs(c-ct).mean((0,2)).tolist(),
      'concept_persistence_mae_lead_coordinate':np.abs(p['c0'][:,None].astype(np.float64)-ct).mean((0,2)).tolist(),
      'graph_iou_lead':g['iou_lead'],'graph_mae_lead':g['mae_lead'],'graph_rmse_lead':g['rmse_lead'],
      'graph_persistence':persistence,'graph_threshold':.05,'graph_support_rule':'abs(A_ij)>0.05, i!=j; empty union IoU=0',
      'graph_max_abs_departure_from_initial':float(np.max(np.abs(p['graphs']-p['a0'][:,None]))),
      'reduction':'float64; future leads 1..5; concepts average origins/assets; graphs average origins/off-diagonal entries'}

def origin_losses(p,prepared):
    """Float64 per-origin point losses and unbiased-pair Monte Carlo ES."""
    samples=p['samples'];y=p['y'].astype(np.float64);mu=p['mu'].astype(np.float64)
    es=np.empty(y.shape[:2],dtype=np.float64);count=len(samples)
    for b in range(len(y)):
        for h in range(y.shape[1]):
            s=samples[:,b,h].astype(np.float64)
            es[b,h]=np.linalg.norm(s-y[b,h],axis=-1).mean()-pdist(s).sum()/(count*(count-1))
    indices=p['indices']
    return {'indices':indices,'origins':prepared['arrays']['origins'][indices],
      'mse':np.square(mu-y).mean((1,2)),'mae':np.abs(mu-y).mean((1,2)),
      'es':es.mean(1),'es_lead':es,'mu':p['mu'],'y':p['y']}

def verify_complete(directory,frozen_sha):
    item=json.loads((directory/'complete.json').read_text())
    assert item['global_selection_sha256']==frozen_sha
    for relative,digest in item['artifact_sha256'].items():assert sha256(directory/relative)==digest,relative
    return item

def run(args):
    study=args.study_root.resolve();torch.set_num_threads(args.threads);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    frozen,frozen_sha=validate_frozen(study)
    assert runtime_sources()==frozen['source_sha256']
    entry=frozen['selections'][selection_key(args.fold,args.variant)]
    selection=json.loads((study/entry['selection_path']).read_text())
    registration=json.loads((study/entry['registration_path']).read_text())
    assert prepared_hashes(args.prepared_root,args.fold)==registration['prepared_sha256']
    base_path=reference_config(args.root,args.fold,args.seed)
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    candidate=next(c for c in CANDIDATES if c['id']==selection['selected_candidate'])
    cfg=configure(base,args.variant,candidate,fold=args.fold,seed=args.seed,device=args.device)
    trial=next(t for t in selection['trials'] if t['candidate']==candidate['id'])
    assert effective_fingerprint(cfg)==trial['effective_fingerprint'],'Seed configuration differs from selected architecture/training'
    assert args.device==registration['device'] and args.threads==registration['threads'],'Execution setting changed after registration'
    directory=study/'runs'/f'fold_{args.fold}_seed_{args.seed}'/args.variant
    if (directory/'complete.json').exists():
        verify_complete(directory,frozen_sha);print('Existing complete result verified.');return
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Preserve incomplete output before retrying: {directory}')
    for sub in ['checkpoints','logs','audit','tables','predictions']:(directory/sub).mkdir(parents=True,exist_ok=True)
    prepared=load_prepared(args.prepared_root/f'fold_{args.fold}')
    limited,guard=train_validation_only(prepared)
    (directory/'config_before_training.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    dump({'global_selection_sha256':frozen_sha,'selection_sha256':entry['selection_sha256'],
      'registration_sha256':entry['registration_sha256'],'source_sha256':frozen['source_sha256'],
      'base_configuration_sha256':sha256(base_path),'prepared_sha256':registration['prepared_sha256'],
      'seed':args.seed,'fold':args.fold,'variant':args.variant,'candidate':candidate['id'],
      'test_used_for_selection':False,'all_six_selections_frozen':True,
      'effective_fingerprint':effective_fingerprint(cfg)},directory/'audit/provenance.json')
    dump(guard,directory/'audit/train_validation_access_guard.json')
    dump(environment_snapshot(),directory/'audit/environment.json')
    seed_everything(args.seed,True);logger=setup_logger(directory/'logs/run.log')
    factory=lambda:_model_factory(prepared,cfg)
    model=factory()
    if args.seed==1:
        checkpoint=study/selection['selected_checkpoint'];assert sha256(checkpoint)==selection['selected_checkpoint_sha256']
        shutil.copy2(checkpoint,directory/'checkpoints/best.pt')
        source_run=checkpoint.parents[1]
        for relative in ['checkpoints/last.pt','logs/training_history.csv']:
            source=source_run/relative
            if source.exists():shutil.copy2(source,directory/relative)
        saved=torch.load(directory/'checkpoints/best.pt',map_location=device,weights_only=False)
        model=model.to(device);model.load_state_dict(saved['model_state']);model.signal_shrinkage=float(saved['signal_shrinkage'])
        summary=trial['training_summary']
        dump({'source':selection['selected_checkpoint'],'source_sha256':selection['selected_checkpoint_sha256'],
          'reason':'Reuse this fold/variant validation-selected seed1 checkpoint before its first test evaluation'},directory/'audit/reuse.json')
    else:
        model,_,summary=train_model(model,limited,cfg['training'],directory,device,logger)
    assert summary['signal_shrinkage_calibration']['selection_split']=='validation'
    cfg['model']['signal_shrinkage']=float(model.signal_shrinkage)
    (directory/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    reloaded,check=checkpoint_reload_check(factory,directory/'checkpoints/best.pt',model,limited,cfg['training'],device)
    assert check['passed'];dump(check,directory/'audit/checkpoint_reload_check.json')
    seed_everything(271845,True)
    validation=collect_predictions(reloaded,prepared,'val',cfg['training'],device)
    np.savez_compressed(directory/'predictions/validation_predictions.npz',**validation);del validation
    # Held-out evaluation starts only after the immutable global choice gate and validation artifacts.
    logger.info('final_test_started frozen_selection_sha=%s',frozen_sha)
    dump({'global_selection_sha256':frozen_sha,'checkpoint_sha256':sha256(directory/'checkpoints/best.pt'),
      'sampling_seed':args.seed+1,'test_scenarios':256},directory/'audit/test_evaluation_started.json')
    seed_everything(args.seed+1,True)
    prediction=collect_predictions(reloaded,prepared,'test',cfg['training'],device)
    for name,value in prediction.items():assert np.isfinite(value).all(),f'Nonfinite prediction field: {name}'
    np.savez_compressed(directory/'predictions/test_predictions.npz',**prediction)
    calibrated=evaluate_predictions(prediction['y'],prediction['mu'],prediction['samples'],[.9])
    rho=float(model.signal_shrinkage);assert .05<=rho<=1
    raw_mu=prediction['mu']/rho
    raw_samples=prediction['samples']+(raw_mu-prediction['mu'])[None]
    uncalibrated=evaluate_predictions(prediction['y'],raw_mu,raw_samples,[.9])
    states=state_metrics(prediction)
    if args.variant=='FrozenGraph':assert states['graph_max_abs_departure_from_initial']==0.0
    dump(calibrated,directory/'tables/calibrated.json');dump(uncalibrated,directory/'tables/uncalibrated.json')
    dump(states,directory/'tables/states.json')
    np.savez_compressed(directory/'tables/origin_losses.npz',**origin_losses(prediction,prepared))
    artifact_sha256={p.relative_to(directory).as_posix():sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix!='.log'}
    dump({'schema':'trefin-frozen-graph-v7-complete','fold':args.fold,'seed':args.seed,'variant':args.variant,
      'selected_candidate':candidate['id'],'rho':rho,'global_selection_sha256':frozen_sha,
      'training':summary,'calibrated':calibrated,'uncalibrated':uncalibrated,
      'graph_max_abs_departure_from_initial':states['graph_max_abs_departure_from_initial'],
      'origin_losses_precision':'float64; unbiased pair ES; raw sample arrays preserved',
      'artifact_sha256':artifact_sha256},directory/'complete.json')
    verify_complete(directory,frozen_sha)
    logger.info('COMPLETE fold=%d seed=%d variant=%s mse=%.9g es=%.9g',args.fold,args.seed,args.variant,calibrated['mse'],calibrated['energy_score'])
    print(json.dumps({'event':'complete','fold':args.fold,'seed':args.seed,'variant':args.variant,'mse':calibrated['mse'],'es':calibrated['energy_score']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-root',type=Path,required=True)
    p.add_argument('--study-root',type=Path,required=True);p.add_argument('--fold',type=int,choices=[1,2,3],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cuda');run(p.parse_args())
