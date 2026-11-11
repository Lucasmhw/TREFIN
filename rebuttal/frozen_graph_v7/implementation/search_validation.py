"""One fold/variant validation-only worker; no test origins enter its fitting object."""
from pathlib import Path
import argparse, json, os, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml
from fconcord.experiment import _model_factory, load_prepared
from fconcord.trainer import train_model, checkpoint_reload_check
from fconcord.utils import seed_everything, setup_logger, environment_snapshot
from protocol import (VARIANTS,CANDIDATES,sha256,dump,configure,effective_fingerprint,
    train_validation_only,runtime_sources,reference_config,prepared_hashes,implementation_hashes)

def run(args):
    started=time.perf_counter()
    study=args.study_root.resolve(); output=study/'search'/f'fold_{args.fold}'/args.variant
    if (study/'frozen_selections.json').exists():raise RuntimeError('Selections already frozen; search is closed')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    base_path=reference_config(args.root,args.fold)
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    prepared,guard=train_validation_only(load_prepared(args.prepared_root/f'fold_{args.fold}'))
    registration={
      'schema':'trefin-frozen-graph-v7-validation-search','fold':args.fold,'variant':args.variant,'training_seed':1,
      'candidates_in_tie_break_order':CANDIDATES,'base_configuration_sha256':sha256(base_path),
      'prepared_sha256':prepared_hashes(args.prepared_root,args.fold),'source_sha256':runtime_sources(),
      'implementation_sha256':implementation_hashes(),'device':args.device,'threads':args.threads,
      'training_budget':{'epoch_ceiling':20,'patience':6,'batch_size':64,'train_scenarios':16,'validation_scenarios':64},
      'checkpoint_selection':'validation ES before mean calibration; original trainer minimum delta 1e-6',
      'candidate_selection':'minimum calibrated validation ES; exact ties follow registered candidate order',
      'positive_mean_calibration':{'split':'validation','bounds':[.05,1.0],'tolerance':1e-5},
      'nominal_slots':4,'known_duplicate':{'FrozenGraph/slow_graph':'FrozenGraph/base'},
      'duplicate_policy':'skip identical effective candidate; record alias; never reallocate budget',
      'test_evaluation_permitted':False,'test_used_for_selection':False,
      'guard':{k:v for k,v in guard.items() if k!='retained_original_indices'}}
    output.mkdir(parents=True,exist_ok=True)
    registration_path=output/'search_registration.json'
    if registration_path.exists():
        assert json.loads(registration_path.read_text())==registration,'Existing registration differs'
    else:
        dump(registration,registration_path);dump(guard,output/'train_validation_access_guard.json')
        dump(environment_snapshot(),output/'environment.json')
    trials=[];seen={}
    for order,candidate in enumerate(CANDIDATES):
        cfg=configure(base,args.variant,candidate,fold=args.fold,seed=1,device=args.device)
        fingerprint=effective_fingerprint(cfg)
        if fingerprint in seen:
            previous=seen[fingerprint]
            trials.append({'candidate':candidate['id'],'order':order,'executed':False,
              'effective_duplicate_of':previous['candidate'],'effective_fingerprint':fingerprint,
              'calibrated_validation_energy_score':previous['calibrated_validation_energy_score']})
            continue
        directory=output/candidate['id']; complete=directory/'complete.json'
        if complete.exists():
            item=json.loads(complete.read_text())
            assert item['effective_fingerprint']==fingerprint
            assert item['registration_sha256']==sha256(registration_path)
            assert item['checkpoint_sha256']==sha256(directory/'checkpoints/best.pt')
        else:
            if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Preserve incomplete trial; use a new study root: {directory}')
            for sub in ['checkpoints','logs','audit']:(directory/sub).mkdir(parents=True,exist_ok=True)
            (directory/'config_before_training.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            seed_everything(1,True);logger=setup_logger(directory/'logs/run.log')
            logger.info('validation_search_started fold=%d variant=%s candidate=%s test_access=forbidden',args.fold,args.variant,candidate['id'])
            factory=lambda:_model_factory(prepared,cfg)
            model=factory()
            model,_,summary=train_model(model,prepared,cfg['training'],directory,device,logger)
            calibration=summary['signal_shrinkage_calibration']
            assert calibration and calibration['selection_split']=='validation'
            score=float(calibration['calibrated_validation_metrics']['energy_score'])
            assert np.isfinite(score)
            cfg['model']['signal_shrinkage']=float(model.signal_shrinkage)
            reloaded,check=checkpoint_reload_check(factory,directory/'checkpoints/best.pt',model,prepared,cfg['training'],device)
            assert check['passed']
            dump(check,directory/'audit/checkpoint_reload_check.json')
            (directory/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            item={'fold':args.fold,'variant':args.variant,'candidate':candidate['id'],'order':order,
              'executed':True,'effective_fingerprint':fingerprint,'registration_sha256':sha256(registration_path),
              'calibrated_validation_energy_score':score,'selected_signal_shrinkage':float(model.signal_shrinkage),
              'training_summary':summary,'checkpoint_sha256':sha256(directory/'checkpoints/best.pt'),
              'checkpoint_reload':check,'test_access_count':0}
            dump(item,complete);del reloaded,model
            if device.type=='cuda':torch.cuda.empty_cache()
            print(json.dumps({'event':'candidate_complete','fold':args.fold,'variant':args.variant,'candidate':candidate['id'],'validation_es':score}),flush=True)
        trials.append(item);seen[fingerprint]=item
    selected=min(trials,key=lambda x:(x['calibrated_validation_energy_score'],x['order']))
    assert selected['executed'],'Canonical candidate must precede its duplicate'
    checkpoint=output/selected['candidate']/'checkpoints/best.pt'
    record={'fold':args.fold,'variant':args.variant,'selection_training_seed':1,
      'selected_candidate':selected['candidate'],'selected_calibrated_validation_energy_score':selected['calibrated_validation_energy_score'],
      'selected_checkpoint':checkpoint.relative_to(study).as_posix(),'selected_checkpoint_sha256':sha256(checkpoint),
      'registration_sha256':sha256(registration_path),'nominal_slots':4,'executed_unique_candidates':len(seen),
      'trials':trials,'test_access_count':0,'wall_seconds':time.perf_counter()-started}
    dump(record,output/'selection.json')
    print(json.dumps({'event':'selection_complete','fold':args.fold,'variant':args.variant,'candidate':selected['candidate']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-root',type=Path,required=True)
    p.add_argument('--study-root',type=Path,required=True);p.add_argument('--fold',type=int,choices=[1,2,3],required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cuda');run(p.parse_args())
