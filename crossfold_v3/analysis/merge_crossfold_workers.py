"""Strict five-seed merge; full artifacts must exist and be verified remotely."""
import argparse,copy,os,shutil
from pathlib import Path
import numpy as np
from common import SEEDS,FAMILIES,METRICS,sha,read,dump,target_sha,load_npz,normalized_protocol,close,metric_summary


def relative(path,root):return Path(os.path.relpath(path,root)).as_posix()


def validate_seed(worker,seed,fold,family,registered,data,manifest,root):
    pp=worker/'protocol.json';protocol=read(pp)
    assert protocol['seeds']==[seed]
    assert normalized_protocol(protocol)==normalized_protocol(registered),'Protocols differ beyond seeds'
    assert protocol['fold']==fold and protocol['family']==family
    assert protocol['data_sha256']==manifest['data_sha256']
    assert protocol['test_targets_float32_bytes_sha256']==manifest['test_targets_float32_bytes_sha256']
    assert protocol['split_sizes']==manifest['split_sizes']
    assert protocol['learning_rate_candidates']==[.001,.0003]
    assert (protocol['max_epochs'],protocol['patience_epochs'],protocol['batch_size'])==(100,15,64)
    assert (protocol['validation_samples'],protocol['test_samples'])==(64,256)
    if family=='tedm':assert protocol['effective_sampler_origin_batch']==64 and protocol['effective_sampler_track_chunk']==64
    if family=='timemixerpp':assert protocol['outer_point_inference_batch_size']==32
    folder=worker/f'seed_{seed}';row=read(folder/'result.json')
    assert row['family']==family and row['seed']==seed
    assert row['protocol_sha256']==sha(pp) and row['source_data_sha256']==manifest['data_sha256']
    assert 0<=row['reload_validation_ES_difference']<1e-7
    assert row['pairwise_translation_error']<1e-12
    assert [c['learning_rate'] for c in row['candidate_summaries']]==[.001,.0003]
    assert len(row['candidate_summaries'])==2
    chosen=min(row['candidate_summaries'],key=lambda c:(c['best_validation_es'],c['candidate']))
    assert chosen==row['selected_candidate']==read(folder/'selected_candidate.json')
    calibration=read(folder/'calibration.json')
    assert calibration['optimizer_success'] and .05<=row['rho']<=1.
    assert calibration['selected_signal_shrinkage']==row['rho']
    for candidate in row['candidate_summaries']:
        directory=folder/f"candidate_{candidate['candidate']:02d}"
        assert read(directory/'complete.json')==candidate
        checkpoint=worker/candidate['checkpoint']
        assert checkpoint.is_file() and sha(checkpoint)==candidate['checkpoint_sha256']
        assert (directory/'last.pt').is_file() and (directory/'training_history.csv').is_file()
        assert 1<=candidate['best_epoch']<=candidate['epochs_completed']<=100
    prediction=folder/'test_predictions.npz';validation=folder/'validation_predictions.npz'
    assert prediction.is_file() and validation.is_file()
    assert sha(prediction)==row['test_predictions_sha256']
    losses=load_npz(folder/'paired_losses.npz')
    positions=data['test_positions'];target=data['targets'][positions]
    assert np.array_equal(losses['indices'],positions)
    assert row['test_origins']==len(positions)==manifest['split_sizes']['test']
    assert row['test_targets_float32_bytes_sha256']==target_sha(target)
    with np.load(prediction,allow_pickle=False) as f:
        assert np.array_equal(f['indices'],positions) and np.array_equal(f['target'],target)
        mean=f['mean'].astype(np.float64);y=target.astype(np.float64)
        assert mean.shape==y.shape and np.isfinite(mean).all()
        for stage,pred in [('before',mean),('after',row['rho']*mean)]:
            close(np.mean((pred-y)**2,axis=(1,2)),losses[f'mse_{stage}'])
            close(np.mean(np.abs(pred-y),axis=(1,2)),losses[f'mae_{stage}'])
            for loss in ['mse','mae','es']:assert np.array_equal(f[f'{loss}_{stage}'],losses[f'{loss}_{stage}'])
        samples=f['samples']
        assert samples.shape==(256,len(positions),5,32) and np.isfinite(samples).all()
        del samples
    with np.load(validation,allow_pickle=False) as f:
        assert np.array_equal(f['indices'],data['val_positions'])
        assert np.array_equal(f['target'],data['targets'][data['val_positions']])
        samples=f['samples'];assert samples.shape==(64,len(data['val_positions']),5,32) and np.isfinite(samples).all()
        del samples
    for stage in ['before','after']:
        for loss,metric,tolerance in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:
            values=losses[f'{loss}_{stage}'];assert values.shape==positions.shape and np.isfinite(values).all()
            close(float(values.mean()),row[stage][metric],tolerance)
    full_artifacts=[{'source':relative(f,root),'sha256':sha(f),'bytes':f.stat().st_size}
                    for f in sorted(worker.rglob('*')) if f.is_file() and f.suffix.lower() in {'.npz','.pt','.pth'}]
    record={'seed':seed,'source_worker':relative(worker,root),'source_result_sha256':sha(folder/'result.json'),
            'execution_protocol_source':relative(pp,root),'execution_protocol_sha256':sha(pp),
            'paired_losses_sha256':sha(folder/'paired_losses.npz'),
            'full_artifacts':full_artifacts,'all_full_artifacts_present_and_hashed':True,
            'full_prediction_target_indices_and_point_losses_verified':True,
            'test_targets_float32_bytes_sha256':target_sha(target)}
    return row,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True,help='Frozen crossfold_v3 source directory')
    p.add_argument('--fold',type=int,choices=[2,3],required=True)
    p.add_argument('--family',choices=list(FAMILIES),required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--execution-plan',type=Path)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();source=a.source.resolve()
    plan_path=a.execution_plan or source/'execution_plan.json';plan=read(plan_path)
    assert {(j['fold'],j['family'],j['seed']) for j in plan['jobs']}=={(f,m,s) for f in [2,3] for m in FAMILIES for s in SEEDS}
    jobs=[j for j in plan['jobs'] if j['fold']==a.fold and j['family']==a.family]
    assert sorted(j['seed'] for j in jobs)==sorted(SEEDS)
    assert plan['source_adaptation_manifest_sha256']==sha(source/'source_adaptation_manifest.json')
    assert plan['data_registry_sha256']==sha(source/'data/registry.json')
    for record in read(source/'source_adaptation_manifest.json')['records']:
        assert sha(source/record['updated_path'])==record['updated_sha256']
        assert sha(source/record['frozen_copy'])==record['frozen_source_sha256']
    manifest=read(source/f'data/fold_{a.fold}/manifest.json')
    data_path=source/f'data/fold_{a.fold}'/manifest['data_file']
    assert sha(data_path)==manifest['data_sha256'];data=load_npz(data_path)
    workers={s:root/f'fold_{a.fold}_{a.family}_seed_{s}' for s in SEEDS}
    for job in jobs:assert Path(job['result']).resolve()==workers[job['seed']]/f"seed_{job['seed']}"/'result.json'
    registered=read(workers[1]/'protocol.json')
    runner=(source/'official_runners'/('stable_tedm_baselines.py' if a.family=='tedm' else 'fresh_official_baselines.py')) if a.family in ['tedm','timerxl'] else source/'point_bundle'/('fresh_point_baselines_fast.py' if a.family=='timemixerpp' else 'fresh_point_baselines.py')
    assert sha(runner)==registered['script_sha256']
    assert sha(source/'crossfold_data.py')==registered['crossfold_loader_sha256']
    assert registered['data_manifest_sha256']==sha(source/f'data/fold_{a.fold}/manifest.json')
    rows=[];records=[]
    for seed in SEEDS:
        row,record=validate_seed(workers[seed],seed,a.fold,a.family,registered,data,manifest,root)
        rows.append(row);records.append(record)
    summary={'family':a.family,'fold':a.fold,'seeds':SEEDS,'runs':rows,'metrics':metric_summary(rows)}
    if a.verify_only:
        print({'status':'verified','fold':a.fold,'family':a.family,'seeds':SEEDS});return
    out=(a.out or root/'merged'/f'fold_{a.fold}_{a.family}').resolve()
    assert all(not(out==w or out.is_relative_to(w) or w.is_relative_to(out)) for w in workers.values())
    temporary=out.with_name(out.name+'.building')
    assert not out.exists() and not temporary.exists(),'Use a fresh merge output'
    temporary.mkdir(parents=True)
    copies=[]
    def preserve(src,dst,seed=None):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
        copies.append({'seed':seed,'source':relative(src,root),'destination':dst.relative_to(temporary).as_posix(),'sha256':sha(src),'bytes':src.stat().st_size})
    preserve(plan_path,temporary/'execution_plan.json')
    preserve(source/f'data/fold_{a.fold}/manifest.json',temporary/'data_manifest.json')
    preserve(source/'analysis/analysis_registration.json',temporary/'analysis_registration.json')
    for seed in SEEDS:
        worker=workers[seed];folder=worker/f'seed_{seed}'
        for src in sorted(folder.rglob('*')):
            if src.is_file() and (src.suffix.lower() in {'.json','.csv'} or src.name=='paired_losses.npz'):
                preserve(src,temporary/f'seed_{seed}'/src.relative_to(folder),seed)
        preserve(worker/'protocol.json',temporary/f'seed_{seed}/execution_protocol.json',seed)
        if (worker/'numerical_audit.json').exists():preserve(worker/'numerical_audit.json',temporary/f'seed_{seed}/worker_numerical_audit_snapshot.json',seed)
    top=copy.deepcopy(registered);top['seeds']=SEEDS
    top['parallel_seed_execution']={'scope':'Five separately registered seed workers; all fitting settings identical',
        'only_normalized_protocol_field':'seeds','worker_mapping':{str(s):relative(workers[s],root) for s in SEEDS},
        'execution_plan_sha256':sha(plan_path),'merge_script_sha256':sha(__file__),
        'result_protocol_hash_semantics':'Each unchanged result hashes its own seed_S/execution_protocol.json; assembled top protocol has a distinct hash'}
    dump(temporary/'protocol.json',top);dump(temporary/'summary.json',summary)
    merged={'schema':'trefin-crossfold-worker-merge-v3','status':'complete','fold':a.fold,'family':a.family,
        'registered_global_seeds':SEEDS,'all_five_completed':True,'normalization':'Remove seeds only; every other protocol field must compare equal',
        'all_full_artifacts_verified_remotely':True,'execution_workers':records,'copied_artifacts':copies,
        'summary_sha256':sha(temporary/'summary.json'),'assembled_protocol_sha256':sha(temporary/'protocol.json'),
        'source_script_sha256':sha(__file__),'execution_plan_sha256':sha(plan_path),
        'analysis_registration_sha256':sha(source/'analysis/analysis_registration.json'),
        'compact_scope':'All small records and paired losses copied unchanged; full prediction/checkpoint hashes identify retained worker artifacts'}
    dump(temporary/'merge_manifest.json',merged);temporary.rename(out)
    print({'status':'complete','fold':a.fold,'family':a.family,'copied_artifacts':len(copies),'out':str(out)})


if __name__=='__main__':main()
