"""Independent checks of completed compact records and temporal inference."""
import argparse,hashlib,math
from pathlib import Path
import numpy as np
from common import SEEDS,FAMILIES,METRICS,sha,read,dump,load_npz,target_sha,close

OLD_NAMES={'tedm':'fresh_tedm_complete','timerxl':'fresh_timerxl','timexer':'fresh_timexer','timemixerpp':'fresh_timemixerpp_complete'}


def independent_bootstrap(d,block):
    count=len(d);starts=np.random.default_rng(20261108).integers(0,count,(10000,(count+block-1)//block))
    positions=(starts[:,:,None]+np.arange(block))%count
    means=d[positions.reshape(10000,-1)[:,:count]].mean(1);effect=float(d.mean())
    return effect,np.percentile(means,[2.5,97.5]),float((1+np.count_nonzero(abs(means-effect)>=abs(effect)))/10001)


def independent_dm(d):
    count=len(d);centered=d-d.mean();lag=min(20,count-1)
    covariance=np.array([np.dot(centered[k:],centered[:-k])/count for k in range(1,lag+1)])
    variance=max(0.,float(np.dot(centered,centered)/count+2*np.dot(1-np.arange(1,lag+1)/(lag+1),covariance)))
    se=math.sqrt(variance/count);correction=math.sqrt((count+1-10+20/count)/count)
    if se<=np.finfo(float).eps:stat=0. if abs(d.mean())<=np.finfo(float).eps else math.copysign(math.inf,d.mean())
    else:stat=float(d.mean())/se
    stat*=correction
    return {'mean_difference':float(d.mean()),'hac_long_run_variance':variance,'standard_error':se,
            'harvey_correction':correction,'raw_p':math.erfc(abs(stat)/math.sqrt(2))}


def independent_holm(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable')
    adjusted=np.maximum.accumulate(np.minimum(1.,values[order]*(len(values)-np.arange(len(values)))))
    output=np.empty_like(adjusted);output[order]=adjusted;return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-cache',type=Path,required=True)
    p.add_argument('--reference-repo',type=Path)
    p.add_argument('--results',type=Path)
    p.add_argument('--fold1-root',type=Path)
    p.add_argument('--merged-root',type=Path)
    p.add_argument('--registration',type=Path,default=Path(__file__).with_name('analysis_registration.json'))
    p.add_argument('--reference-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();ref={};ref_sources=[]
    for fold in [1,2,3]:
        folder=a.reference_cache/f'fold_{fold}';origin=read(folder/'origin_series.json');ref[fold]={}
        for seed in SEEDS:
            metadata=read(folder/f'seed_{seed}.json');file=folder/f'seed_{seed}.npz';assert sha(file)==metadata['loss_sha256']
            arrays=load_npz(file);assert np.array_equal(arrays['indices'],origin['indices'])
            assert metadata['target_hash']==origin['target_hash'];close(arrays['es'].mean(),metadata['energy_score_recomputed'],1e-14)
            if a.reference_repo:
                prediction=a.reference_repo/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'/f'fold_{fold}_seed_{seed}'/'predictions/test_predictions.npz'
                assert sha(prediction)==metadata['source_sha256']
                with np.load(prediction,allow_pickle=False) as f:y=f['y'];mu=f['mu'].astype(float);assert np.array_equal(f['indices'],arrays['indices'])
                assert target_sha(y)==metadata['target_hash']
                close(np.square(mu-y.astype(float)).mean((1,2)),arrays['mse'],1e-14)
                close(np.abs(mu-y.astype(float)).mean((1,2)),arrays['mae'],1e-14)
            ref[fold][seed]={**arrays,'target_hash':metadata['target_hash']}
            ref_sources.append({'fold':fold,'seed':seed,'loss_sha256':sha(file),'prediction_sha256':metadata['source_sha256'],
                                'original_prediction_and_point_losses_reverified':bool(a.reference_repo)})
    if a.reference_only:
        dump(a.out/'reference_independent_audit.json',{'status':'passed','reference_sources':ref_sources,
            'calibrated_reference_runs':15,'all_target_hashes_and_indices_checked':True})
        print({'status':'reference_independent_audit_passed','reference_runs':15});return
    assert a.results and a.fold1_root and a.merged_root
    report=read(a.results);registration=read(a.registration)
    assert report['registration']==registration and report['analysis_registration_sha256']==sha(a.registration)
    assert report['completed_baseline_runs']==60 and report['new_baseline_runs']==40
    assert report['source_hashes']['summarizer']==sha(Path(__file__).with_name('summarize_crossfold.py'))
    assert report['source_hashes']['common']==sha(Path(__file__).with_name('common.py'))
    copied=0;completed=0;difference={};families={}
    for fold in [1,2,3]:
        for family,label in FAMILIES.items():
            directory=a.fold1_root/OLD_NAMES[family] if fold==1 else a.merged_root/f'fold_{fold}_{family}'
            summary=read(directory/'summary.json');assert summary['seeds']==SEEDS
            if (directory/'merge_manifest.json').exists():
                manifest=read(directory/'merge_manifest.json');assert manifest['status']=='complete'
                assert sha(directory/'summary.json')==manifest['summary_sha256']
                assert sha(directory/'protocol.json')==manifest['assembled_protocol_sha256']
                for item in manifest['copied_artifacts']:
                    assert sha(directory/item['destination'])==item['sha256'];copied+=1
                if fold>1:
                    assert manifest['source_script_sha256']==sha(Path(__file__).with_name('merge_crossfold_workers.py'))
                    assert manifest['all_full_artifacts_verified_remotely']
                    assert manifest['analysis_registration_sha256']==sha(a.registration)
                    assert all(w['all_full_artifacts_present_and_hashed'] for w in manifest['execution_workers'])
            rows=[];losses=[];protocol_reference=None
            for seed in SEEDS:
                folder=directory/f'seed_{seed}';row=read(folder/'result.json')
                protocol_file=folder/'execution_protocol.json'
                if not protocol_file.exists():protocol_file=directory/'protocol.json'
                protocol=read(protocol_file);normalized={k:v for k,v in protocol.items() if k!='seeds'}
                if protocol_reference is None:protocol_reference=normalized
                else:assert protocol_reference==normalized
                assert row['protocol_sha256']==sha(protocol_file)
                assert row['seed']==seed and row['family']==family
                assert row['test_targets_float32_bytes_sha256']==ref[fold][seed]['target_hash']
                assert row['source_data_sha256']==protocol['data_sha256']
                chosen=min(row['candidate_summaries'],key=lambda c:(c['best_validation_es'],c['candidate']))
                assert row['selected_candidate']==chosen==read(folder/'selected_candidate.json')
                assert [c['learning_rate'] for c in row['candidate_summaries']]==[.001,.0003]
                for c in row['candidate_summaries']:assert c==read(folder/f"candidate_{c['candidate']:02d}/complete.json")
                calibration=read(folder/'calibration.json');assert calibration['optimizer_success'] and calibration['selected_signal_shrinkage']==row['rho']
                arrays=load_npz(folder/'paired_losses.npz');assert np.array_equal(arrays['indices'],ref[fold][seed]['indices'])
                for stage in ['before','after']:
                    for loss,key,tolerance in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:close(arrays[f'{loss}_{stage}'].mean(),row[stage][key],tolerance)
                rows.append(row);losses.append(arrays);completed+=1
            reported=report['results'][str(fold)][label]
            assert reported['runs']==rows==summary['runs']
            for path,expected in reported['source_hashes'].items():assert sha(directory/path)==expected
            for stage in ['before','after']:
                for metric in METRICS:
                    values=np.asarray([r[stage][metric] for r in rows]);expect=[values.mean(),values.std(ddof=1)]
                    for k,v in zip(['mean','sample_std'],expect):close(v,reported['metrics'][stage][metric][k]);close(v,summary['metrics'][stage][metric][k])
            families[f'fold_{fold}/{label}']={'seeds':SEEDS,'target_hash_and_origin_alignment':True,'candidate_calibration_records_verified':True}
            if label in registration['primary_comparators']:
                for metric in ['mse','mae','es']:
                    difference[(f'fold_{fold}/{label}',metric)]=np.mean([v[f'{metric}_after'] for v in losses],axis=0)-np.mean([ref[fold][s][metric] for s in SEEDS],axis=0)
    boot=dm=0;tests=report['paired_tests_holm6']
    for (name,metric),d in difference.items():
        for block in [5,10,20]:
            effect,ci,pvalue=independent_bootstrap(d,block);t=tests[name][metric][f'block{block}']
            close(effect,t['mean_difference'],1e-14);close(ci,t['ci95'],1e-14);close(pvalue,t['raw_p'],1e-14);boot+=1
        expected=independent_dm(d)
        for key,value in expected.items():close(value,tests[name][metric]['dm_hac20'][key],1e-12)
        dm+=1
    names=registration['multiplicity']['comparisons']
    for metric in registration['losses']:
        for key in ['block5','block10','block20','dm_hac20']:
            adjusted=independent_holm([tests[n][metric][key]['raw_p'] for n in names])
            close(adjusted,[tests[n][metric][key]['holm6_p'] for n in names],1e-14)
    result={'status':'passed','completed_baseline_runs':completed,'reference_runs':15,
        'merged_copy_hashes_verified':copied,'independent_bootstrap_comparisons':boot,
        'independent_DM_HAC20_checks':dm,'Holm6_bootstrap_families':9,'Holm6_DM_families':3,
        'source_results_sha256':sha(a.results),'analysis_registration_sha256':sha(a.registration),
        'families':families,'reference_sources':ref_sources,
        'full_artifact_scope':'New baseline full prediction/checkpoint files were required and hash-verified by the remote merge; this audit independently verifies the downloaded compact records and inference.'}
    assert completed==60 and boot==54 and dm==18
    dump(a.out/'crossfold_independent_audit.json',result)
    print({k:result[k] for k in ['status','completed_baseline_runs','independent_bootstrap_comparisons','independent_DM_HAC20_checks']})


if __name__=='__main__':main()
