"""Complete three-fold tables and prespecified six-comparison paired inference."""
import argparse,csv
from pathlib import Path
import numpy as np
from common import SEEDS,FAMILIES,METRICS,sha,read,dump,load_npz,normalized_protocol,close,metric_summary,circular_bootstrap,dm_hac20,holm

OLD_NAMES={'tedm':'fresh_tedm_complete','timerxl':'fresh_timerxl','timexer':'fresh_timexer','timemixerpp':'fresh_timemixerpp_complete'}


def reference(cache):
    manifest=read(cache/'reference_manifest.json');result={};rows={}
    for fold in [1,2,3]:
        folder=cache/f'fold_{fold}';summary=read(folder/'reference_metrics.json')
        assert sha(folder/'reference_metrics.json')==manifest['folds'][str(fold)]['reference_metrics_sha256']
        assert sha(folder/'origin_series.json')==summary['origin_series_sha256']
        origins=read(folder/'origin_series.json');indices=np.asarray(origins['indices'])
        result[fold]={};rows[fold]=summary
        assert summary['seeds']==SEEDS
        for seed in SEEDS:
            metadata=read(folder/f'seed_{seed}.json');file=folder/f'seed_{seed}.npz'
            assert sha(file)==metadata['loss_sha256']
            assert metadata['target_hash']==origins['target_hash']==summary['target_hash']
            arrays=load_npz(file);assert np.array_equal(arrays['indices'],indices)
            close(arrays['es'].mean(),metadata['energy_score_recomputed'],1e-14)
            result[fold][seed]={**arrays,'target_hash':metadata['target_hash']}
    return result,rows


def load_completed(directory,fold,family,ref):
    top=read(directory/'protocol.json');summary=read(directory/'summary.json')
    assert top['seeds']==SEEDS and summary['seeds']==SEEDS
    source_hashes={'protocol.json':sha(directory/'protocol.json'),'summary.json':sha(directory/'summary.json')}
    if (directory/'merge_manifest.json').exists():
        merge=read(directory/'merge_manifest.json');assert merge['status']=='complete'
        assert merge['registered_global_seeds']==SEEDS
        assert sha(directory/'summary.json')==merge['summary_sha256']
        assert sha(directory/'protocol.json')==merge['assembled_protocol_sha256']
        for record in merge['copied_artifacts']:
            assert sha(directory/record['destination'])==record['sha256'],record['destination']
        if fold>1:assert merge['all_full_artifacts_verified_remotely'] and merge['all_five_completed']
        source_hashes['merge_manifest.json']=sha(directory/'merge_manifest.json')
    elif fold>1:raise AssertionError('New folds require a complete verified merge')
    rows=[];losses=[];execution_reference=None
    for seed in SEEDS:
        folder=directory/f'seed_{seed}';row=read(folder/'result.json')
        protocol_file=folder/'execution_protocol.json'
        if not protocol_file.exists():protocol_file=directory/'protocol.json'
        protocol=read(protocol_file)
        assert row['seed']==seed and row['family']==family
        assert protocol['family']==family and seed in protocol['seeds']
        assert row['protocol_sha256']==sha(protocol_file)
        assert row['source_data_sha256']==protocol['data_sha256']
        if fold>1:assert protocol['fold']==fold and protocol['seeds']==[seed]
        if execution_reference is None:execution_reference=normalized_protocol(protocol)
        else:assert execution_reference==normalized_protocol(protocol)
        assert row['test_targets_float32_bytes_sha256']==ref[seed]['target_hash']
        assert row['reload_validation_ES_difference']<1e-7 and row['pairwise_translation_error']<1e-12
        chosen=min(row['candidate_summaries'],key=lambda r:(r['best_validation_es'],r['candidate']))
        assert chosen==row['selected_candidate']==read(folder/'selected_candidate.json')
        assert [c['learning_rate'] for c in row['candidate_summaries']]==[.001,.0003]
        for candidate in row['candidate_summaries']:
            assert candidate==read(folder/f"candidate_{candidate['candidate']:02d}/complete.json")
        calibration=read(folder/'calibration.json');assert calibration['optimizer_success']
        assert calibration['selected_signal_shrinkage']==row['rho'] and .05<=row['rho']<=1.
        arrays=load_npz(folder/'paired_losses.npz')
        assert np.array_equal(arrays['indices'],ref[seed]['indices']) and len(arrays['indices'])==row['test_origins']
        for stage in ['before','after']:
            for loss,metric,tolerance in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:
                values=arrays[f'{loss}_{stage}'];assert values.shape==arrays['indices'].shape and np.isfinite(values).all()
                close(values.mean(),row[stage][metric],tolerance)
        rows.append(row);losses.append(arrays)
        for file in [folder/'result.json',folder/'paired_losses.npz',protocol_file]:source_hashes[file.relative_to(directory).as_posix()]=sha(file)
    assert summary['runs']==rows
    metrics=metric_summary(rows)
    for stage in ['before','after']:
        for key in METRICS:
            for stat in ['mean','sample_std']:close(metrics[stage][key][stat],summary['metrics'][stage][key][stat])
    return {'runs':rows,'metrics':metrics,'source_hashes':source_hashes,'protocol':top},losses,execution_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold1-root',type=Path)
    p.add_argument('--merged-root',type=Path)
    p.add_argument('--reference-cache',type=Path,required=True)
    p.add_argument('--registration',type=Path,default=Path(__file__).with_name('analysis_registration.json'))
    p.add_argument('--original-fold1-results',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-only',action='store_true')
    a=p.parse_args();registration=read(a.registration)
    assert registration['multiplicity']['family_size']==6
    assert registration['bootstrap']['repetitions']==10000 and registration['bootstrap']['rng_seed']==20261108
    ref,reference_rows=reference(a.reference_cache)
    if a.reference_only:
        dump(a.out/'reference_only_verification.json',{'status':'passed','folds':[1,2,3],
            'completed_calibrated_reference_seeds':15,'all_seed_target_hashes_and_indices_verified':True,
            'reference_manifest_sha256':sha(a.reference_cache/'reference_manifest.json'),
            'analysis_registration_sha256':sha(a.registration)})
        print({'status':'reference_only_passed'});return
    assert a.fold1_root and a.merged_root and a.original_fold1_results
    results={};seed_losses={};protocols={};flat=[];execution={}
    for fold in [1,2,3]:
        results[str(fold)]={};seed_losses[fold]={};execution[fold]={}
        for family,label in FAMILIES.items():
            directory=a.fold1_root/OLD_NAMES[family] if fold==1 else a.merged_root/f'fold_{fold}_{family}'
            record,losses,protocol=load_completed(directory,fold,family,ref[fold])
            results[str(fold)][label]=record;seed_losses[fold][label]=losses;execution[fold][label]=protocol
            if fold>1:
                old=execution[1][label]
                allow={'script_sha256','source_manifest_sha256','common_helper_sha256','data_manifest_sha256','data_sha256','selection'}
                for key in old:
                    if key not in allow:assert protocol[key]==old[key],(fold,label,key)
                assert protocol['frozen_fold1_runner_sha256']==old['script_sha256']
            for stage in ['before','after']:
                flat.append({'fold':fold,'method':label,'stage':stage,'seeds':5,
                    **{f'{m}_{stat}':record['metrics'][stage][m][stat] for m in METRICS for stat in ['mean','sample_std']}})
        for stage in ['before','after']:
            flat.append({'fold':fold,'method':'TREFIN','stage':stage,'seeds':5,
                **{f'{m}_{stat}':reference_rows[fold]['metrics'][stage][m][stat] for m in METRICS for stat in ['mean','sample_std']}})
    tests={};differences={}
    for fold in [1,2,3]:
        for label in registration['primary_comparators']:
            name=f'fold_{fold}/{label}';tests[name]={}
            for metric in registration['losses']:
                candidate=np.mean([x[f'{metric}_after'] for x in seed_losses[fold][label]],axis=0)
                reference_average=np.mean([ref[fold][s][metric] for s in SEEDS],axis=0)
                d=candidate-reference_average
                differences[f"fold_{fold}_{label.replace('-','_')}_{metric}"]=d
                tests[name][metric]={f'block{block}':circular_bootstrap(d,block) for block in [5,10,20]}
                tests[name][metric]['dm_hac20']=dm_hac20(d)
    assert set(tests)==set(registration['multiplicity']['comparisons'])
    for metric in registration['losses']:
        for block in [5,10,20]:
            corrected=holm({name:t[metric][f'block{block}']['raw_p'] for name,t in tests.items()})
            for name,value in corrected.items():tests[name][metric][f'block{block}']['holm6_p']=value
        corrected=holm({name:t[metric]['dm_hac20']['raw_p'] for name,t in tests.items()})
        for name,value in corrected.items():tests[name][metric]['dm_hac20']['holm6_p']=value
    # Retain the original fold-1 adjusted results separately and verify that its
    # raw comparisons are unchanged by enlarging the adjustment family.
    old=read(a.original_fold1_results)
    for family in registration['primary_comparators']:
        for metric in registration['losses']:
            current=tests[f'fold_1/{family}'][metric];original=old['paired_tests'][family][metric]
            for block in [5,10,20]:
                b=current[f'block{block}'];o=original[f'block{block}']
                close(b['mean_difference'],o['mean_loss_difference_candidate_minus_reference'],1e-14)
                close(b['ci95'],o['confidence_interval'],1e-14)
                close(b['raw_p'],o['two_sided_bootstrap_p_value'],1e-14)
            close(current['dm_hac20']['raw_p'],original['dm_hac20']['two_sided_p_value'],1e-12)
    report={'schema':'trefin-complete-crossfold-analysis-v3','status':'complete',
        'completed_baseline_runs':60,'new_baseline_runs':40,'folds':[1,2,3],'seeds':SEEDS,
        'registration':registration,'analysis_registration_sha256':sha(a.registration),
        'reference_manifest_sha256':sha(a.reference_cache/'reference_manifest.json'),
        'reference_metrics':{str(k):v for k,v in reference_rows.items()},
        'results':results,'paired_tests_holm6':tests,
        'original_fold1_two_comparator_inference':{'source_sha256':sha(a.original_fold1_results),'paired_tests':old['paired_tests'],
            'raw_bootstrap_comparisons_reproduced_atol':1e-14,'DM_p_values_reproduced_atol':1e-12,
            'adjusted_p_values_retained_separately':True},
        'source_hashes':{'summarizer':sha(__file__),'common':sha(Path(__file__).with_name('common.py'))},
        'interpretation':'All seeds/folds retained. Positive comparator-minus-TREFIN differences favor TREFIN. Mean and sample SD describe seeds; temporal inference averages seeds before resampling origins. No cross-fold or seed pooling. TimeXer/TimeMixer++ remain descriptive.'}
    a.out.mkdir(parents=True,exist_ok=True)
    dump(a.out/'crossfold_results.json',report)
    np.savez_compressed(a.out/'paired_seed_average_differences.npz',**differences)
    with (a.out/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    test_rows=[]
    for name,metrics in tests.items():
        for loss,items in metrics.items():
            for block in [5,10,20]:
                t=items[f'block{block}'];test_rows.append({'comparison':name,'loss':loss,'method':f'circular_block_{block}',
                    'difference':t['mean_difference'],'ci95_low':t['ci95'][0],'ci95_high':t['ci95'][1],
                    'raw_p':t['raw_p'],'holm6_p':t['holm6_p'],'origins':t['origins']})
            t=items['dm_hac20'];test_rows.append({'comparison':name,'loss':loss,'method':'DM_HAC20','difference':t['mean_difference'],
                'ci95_low':'','ci95_high':'','raw_p':t['raw_p'],'holm6_p':t['holm6_p'],'origins':t['origins']})
    with (a.out/'paired_tests.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(test_rows[0]));w.writeheader();w.writerows(test_rows)
    lines=['# Completed three-fold baseline comparison','',
        'All four baseline families have five completed seeds in each fold. The two new folds use the frozen fold-1 fitting protocol. The table reports mean ± sample SD over seeds after validation-only mean calibration. Full before/after eight-metric values are in metrics.csv.','',
        '| Fold | Method | MSE ×10⁴ | MAE ×10² | ES ×10² | 90% coverage |','|---|---|---:|---:|---:|---:|']
    for row in flat:
        if row['stage']!='after':continue
        cells=[f"{row[m+'_mean']*scale:.4f} ± {row[m+'_sample_std']*scale:.4f}" for m,scale in [('mse',1e4),('mae',100),('energy_score',100),('coverage_90',100)]]
        lines.append(f"| {row['fold']} | {row['method']} | "+' | '.join(cells)+' |')
    lines+=['','Paired inference is performed separately within each fold after seed averaging. Holm adjustment covers the six TEDM/Timer-XL × fold comparisons separately for each loss and block choice. The original fold-1 two-comparator adjusted outputs are preserved separately; per-comparison intervals are not simultaneous intervals. TimeXer and TimeMixer++ are descriptive comparisons.']
    (a.out/'crossfold_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print({'status':'complete','baseline_runs':60,'bootstrap_comparisons':54,'DM_comparisons':18})


if __name__=='__main__':main()
