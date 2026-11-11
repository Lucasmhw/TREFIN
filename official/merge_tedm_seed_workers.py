#!/usr/bin/env python3
"""Merge five completed, identically configured TEDM seed workers without refitting.

python merge_tedm_seed_workers.py --root /experiment/rebuttal
Default workers: fresh_tedm_b64 (seed1), fresh_tedm_seed_{7,19,37,73}.
Default output: fresh_tedm_complete. Sources are read-only. Compact artifacts are
copied byte-for-byte; each result retains its actual worker protocol SHA256.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,os,shutil
from pathlib import Path
import numpy as np

SEEDS=[1,7,19,37,73]
METRICS=['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def normalize(protocol):
    value=copy.deepcopy(protocol)
    if 'seeds' not in value:raise ValueError('Execution protocol lacks registered seeds')
    del value['seeds']
    return value

def aggregate(rows):
    if [r['seed'] for r in rows]!=SEEDS:raise ValueError('All five registered seeds, in order, are required')
    result={'family':'tedm','seeds':SEEDS,'runs':rows,'metrics':{}}
    for stage in ['before','after']:
        result['metrics'][stage]={}
        for key in METRICS:
            values=np.asarray([r[stage][key] for r in rows],dtype=np.float64)
            if not np.isfinite(values).all():raise ValueError(f'Nonfinite {stage}/{key}')
            result['metrics'][stage][key]={'mean':float(values.mean()),'sample_std':float(values.std(ddof=1))}
    return result


def relative(path,root):return Path(os.path.relpath(path,root)).as_posix()

def validate_worker(seed,worker,registered,root):
    protocol_path=worker/'protocol.json';protocol=read(protocol_path)
    expected=SEEDS if seed==1 else [seed]
    if protocol['seeds']!=expected:raise ValueError(f'Seed{seed}: worker seed list differs from {expected}')
    if normalize(protocol)!=normalize(registered):raise ValueError(f'Seed{seed}: a protocol field other than seeds differs')
    if protocol['family']!='tedm' or protocol['effective_sampler_origin_batch']!=64:
        raise ValueError('Only the registered TEDM origin-batch64 protocol may be merged')
    folder=worker/f'seed_{seed}';result_path=folder/'result.json';loss_path=folder/'paired_losses.npz'
    row=read(result_path)
    if row.get('family')!='tedm' or row.get('seed')!=seed:raise ValueError('Result identity mismatch')
    if row['protocol_sha256']!=sha(protocol_path):raise ValueError(f'Seed{seed}: result does not hash its actual worker protocol')
    if row['source_data_sha256']!=protocol['data_sha256']:raise ValueError('Source data mismatch')
    if not 0<=row['reload_validation_ES_difference']<1e-7:raise ValueError('Checkpoint reload verification failed')
    if len(row['candidate_summaries'])!=len(protocol['learning_rate_candidates']):raise ValueError('Missing candidate results')
    if [x['learning_rate'] for x in row['candidate_summaries']]!=protocol['learning_rate_candidates']:raise ValueError('Candidate grid mismatch')
    selected=read(folder/'selected_candidate.json')
    if selected!=row['selected_candidate']:raise ValueError('Selected candidate differs from immutable result')
    expected_selection=min(row['candidate_summaries'],key=lambda x:(x['best_validation_es'],x['candidate']))
    if selected!=expected_selection:raise ValueError('Selected candidate is not the validation ES minimum')
    calibration=read(folder/'calibration.json')
    if not calibration['optimizer_success'] or calibration['selected_signal_shrinkage']!=row['rho']:raise ValueError('Calibration record mismatch')
    candidates=[]
    for candidate in row['candidate_summaries']:
        completed=folder/f"candidate_{candidate['candidate']:02d}"/'complete.json'
        if read(completed)!=candidate:raise ValueError('Candidate completion/result mismatch')
        checkpoint=worker/candidate['checkpoint']
        if checkpoint.exists() and sha(checkpoint)!=candidate['checkpoint_sha256']:raise ValueError('Candidate checkpoint hash mismatch')
        candidates.append({'candidate':candidate['candidate'],'source_checkpoint':relative(checkpoint,root),'checkpoint_sha256':candidate['checkpoint_sha256'],'checkpoint_present_at_merge':checkpoint.is_file()})
    with np.load(loss_path,allow_pickle=False) as source:losses={k:source[k] for k in source.files}
    indices=np.asarray(losses['indices'])
    if len(indices)!=row['test_origins'] or len(indices)!=500 or np.any(np.diff(indices)<=0):raise ValueError('Test origins are not the registered500 chronological origins')
    for stage in ['before','after']:
        for loss,metric,tol in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:
            values=np.asarray(losses[f'{loss}_{stage}'],dtype=np.float64)
            if values.shape!=indices.shape or not np.isfinite(values).all():raise ValueError('Invalid paired origin losses')
            if abs(float(values.mean())-row[stage][metric])>=tol:raise ValueError(f'Seed{seed}: paired {stage}/{metric} mean mismatch')
    full_predictions=folder/'test_predictions.npz'
    if full_predictions.exists() and sha(full_predictions)!=row['test_predictions_sha256']:raise ValueError('Full prediction hash mismatch')
    record={'seed':seed,'source_worker':relative(worker,root),'source_run':relative(folder,root),'source_result_sha256':sha(result_path),'source_paired_losses_sha256':sha(loss_path),'execution_protocol_source':relative(protocol_path,root),'execution_protocol_sha256':sha(protocol_path),'execution_protocol_seeds':protocol['seeds'],'registered_script_sha256':protocol['script_sha256'],'numerical_adapter':protocol.get('numerical_adapter'),'source_test_predictions':relative(full_predictions,root),'test_predictions_sha256':row['test_predictions_sha256'],'full_test_predictions_present_at_merge':full_predictions.is_file(),'candidate_checkpoints':candidates,'test_targets_float32_bytes_sha256':row['test_targets_float32_bytes_sha256']}
    return row,indices,record


def self_test():
    base={'seeds':SEEDS,'learning_rate_candidates':[.001,.0003],'same':{'x':1}}
    other=copy.deepcopy(base);other['seeds']=[7];assert normalize(base)==normalize(other)
    other['same']['x']=2;assert normalize(base)!=normalize(other)
    rows=[{'seed':s,**{stage:{k:float(i+1) for k in METRICS} for stage in ['before','after']}} for i,s in enumerate(SEEDS)]
    summary=aggregate(rows);assert summary['metrics']['after']['mse']['mean']==3.
    assert abs(summary['metrics']['after']['mse']['sample_std']-np.sqrt(2.5))<1e-15
    try:aggregate(rows[:-1])
    except ValueError:pass
    else:raise AssertionError('Incomplete seeds accepted')
    print(json.dumps({'self_test_passed':True,'checks':['seed-only protocol normalization','other settings mismatch','actual five-value mean and sampleSD','incomplete seeds rejected'],'files_created':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--seed1-dir',type=Path);p.add_argument('--worker',action='append',default=[],help='Optional override SEED=PATH')
    p.add_argument('--copy-all-artifacts',action='store_true',help='Also copy full forecast arrays, checkpoints and logs; otherwise small JSON/CSV + paired losses only')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.root is None:p.error('--root is required')
    root=a.root.resolve();out=(a.out or root/'fresh_tedm_complete').resolve()
    workers={seed:root/('fresh_tedm_b64' if seed==1 else f'fresh_tedm_seed_{seed}') for seed in SEEDS}
    if a.seed1_dir:workers[1]=a.seed1_dir.resolve()
    seen=set()
    for item in a.worker:
        seed,path=item.split('=',1);seed=int(seed)
        if seed not in SEEDS or seed in seen:raise ValueError('Unknown or duplicate worker seed')
        seen.add(seed);workers[seed]=Path(path).resolve()
    if len(set(workers.values()))!=5:raise ValueError('Five distinct worker directories are required')
    for worker in workers.values():
        if out==worker or out.is_relative_to(worker) or worker.is_relative_to(out):raise ValueError('Output and source-worker paths must be separate')
    registered_path=workers[1]/'protocol.json';registered=read(registered_path)
    if registered['seeds']!=SEEDS:raise ValueError('Original global registered seed list is missing or changed')
    rows=[];records=[];reference_indices=None;target_hash=None
    for seed in SEEDS:
        row,indices,record=validate_worker(seed,workers[seed],registered,root)
        if reference_indices is None:reference_indices=indices;target_hash=row['test_targets_float32_bytes_sha256']
        elif not np.array_equal(reference_indices,indices) or row['test_targets_float32_bytes_sha256']!=target_hash:raise ValueError('Seeds do not use identical origins/targets')
        rows.append(row);records.append(record)
    summary=aggregate(rows)
    if a.verify_only:
        print(json.dumps({'verified':True,'registered_seeds':SEEDS,'workers':records}));return
    temporary=out.with_name(out.name+'.building')
    if out.exists() or temporary.exists():raise FileExistsError('Use a fresh output directory; existing merged or partial outputs are not overwritten')
    temporary.mkdir(parents=True)
    copy_records=[]
    def preserve(source,destination,seed):
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        before=sha(source);after=sha(destination)
        if before!=after:raise RuntimeError('Copied bytes differ')
        copy_records.append({'seed':seed,'source':relative(source,root),'destination':destination.relative_to(temporary).as_posix(),'sha256':before,'bytes':source.stat().st_size})
    preserve(registered_path,temporary/'global_registered_protocol.json',None)
    for seed in SEEDS:
        worker=workers[seed];folder=worker/f'seed_{seed}';dest=temporary/f'seed_{seed}'
        for source in sorted(folder.rglob('*')):
            if not source.is_file():continue
            include=a.copy_all_artifacts or source.suffix.lower() in {'.json','.csv'} or source.name=='paired_losses.npz'
            if include:preserve(source,dest/source.relative_to(folder),seed)
        preserve(worker/'protocol.json',dest/'execution_protocol.json',seed)
        if (worker/'numerical_audit.json').exists():preserve(worker/'numerical_audit.json',dest/'worker_numerical_audit_snapshot.json',seed)
    top=copy.deepcopy(registered)
    top['seeds']=SEEDS
    top['parallel_seed_execution']={'schema':'trefin-independent-seed-worker-assembly-v1','fit_and_selection_settings_unchanged':True,'only_normalized_protocol_field':'seeds','original_global_seed_list':SEEDS,'global_registered_protocol_file':'global_registered_protocol.json','global_registered_protocol_sha256':sha(registered_path),'worker_mapping':{str(r['seed']):r['source_worker'] for r in records},'result_protocol_hash_semantics':'Each unchanged seed result hashes its own seed_S/execution_protocol.json, not this assembled protocol','merge_script_sha256':sha(__file__),'copy_mode':'all seed artifacts' if a.copy_all_artifacts else 'compact JSON/CSV plus paired origin losses; full predictions/checkpoints remain in source workers'}
    dump(temporary/'protocol.json',top)
    summary['execution']={'mode':'independent seed workers; no refitting or result selection during merge','registered_global_seed_list':SEEDS,'assembled_protocol_sha256':sha(temporary/'protocol.json'),'result_records_preserved_byte_for_byte':True,'all_five_actual_completed_results':True}
    dump(temporary/'summary.json',summary)
    manifest={'schema':'trefin-tedm-seed-worker-merge-v1','status':'complete','registered_global_seeds':SEEDS,'global_registered_protocol_sha256':sha(registered_path),'protocol_normalization':'Remove seeds only; every other JSON value must compare equal','all_test_origins_and_target_hashes_identical':True,'execution_workers':records,'copied_artifacts':copy_records,'summary_sha256':sha(temporary/'summary.json'),'assembled_protocol_sha256':sha(temporary/'protocol.json'),'source_script_sha256':sha(__file__),'compact_artifact_note':'Recorded source checkpoint/prediction paths are relative to --root. They are genuine source artifacts; they are not asserted to exist in this compact assembly.'}
    dump(temporary/'merge_manifest.json',manifest)
    (temporary/'ASSEMBLY_README.md').write_text('This directory assembles five completed independent training seeds. No model was fitted, scored, selected or recalibrated by the merge. Every source protocol is identical after removing only its seeds field. The original all-five-seed registration is preserved in global_registered_protocol.json. Each seed result is byte-identical to its source and retains the SHA256 of seed_S/execution_protocol.json. The top-level protocol adds explicit worker provenance, so its hash intentionally differs. Summary means and sample SD use exactly seeds 1,7,19,37,73. With the default compact option, checkpoint and full-scenario hashes refer to retained source-worker artifacts; they are not fabricated local files. Per-origin paired losses are included for downstream temporal inference. Worker numerical-audit snapshots may contain process-wide counters beyond the selected completed seed; result.json is the completed-run record.\n',encoding='utf8')
    temporary.rename(out)
    print(json.dumps({'status':'complete','output':str(out),'seeds':SEEDS,'copied_artifacts':len(copy_records),'summary_sha256':sha(out/'summary.json')}))

if __name__=='__main__':main()
