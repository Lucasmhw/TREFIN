#!/usr/bin/env python3
"""Tables and paired temporal tests for completed fresh upstream supplements.

Only per-seed result.json and paired_losses.npz need be downloaded. Historical
TREFIN predictions are read from the local restored repository. Historical
TEDM/Timer-XL summary numbers are never substituted for new run outputs.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def paired_circular_bootstrap(candidate,reference,block_length,repetitions=10000,seed=20261108):
    """Percentile CI; two-sided test of the null-centered paired block means."""
    from summarize_retraining import circular_bootstrap
    candidate=np.asarray(candidate,dtype=np.float64)
    reference=np.asarray(reference,dtype=np.float64)
    if candidate.shape!=reference.shape:raise ValueError('Paired loss shapes differ')
    result=circular_bootstrap(candidate-reference,block_length,repetitions,seed)
    return {
        'mean_loss_difference_candidate_minus_reference':result['mean_difference'],
        'confidence_interval':result['ci95'],
        'two_sided_bootstrap_p_value':result['centered_two_sided_p'],
        'p_value_method':'null-centered circular block means; (1 + count(abs(centered_mean) >= abs(observed_mean))) / (repetitions + 1)',
        'confidence_interval_method':'percentile 95% interval of uncentered paired block means',
        'block_length':result['block_length'],'requested_block_length':block_length,
        'repetitions':repetitions,'seed':seed,'origins':result['origins'],
        'minimum_raw_p_value':1/(repetitions+1)}


def load_reference(repo,cache):
    """Compute float64, exact all-pairs TREFIN ES once per immutable source."""
    from fconcord.statistical_tests import per_origin_energy_score
    cache.mkdir(parents=True,exist_ok=True)
    seeds=[1,7,19,37,73];reference={};rows={'before':[],'after':[]}
    for stage,protocol in [('before','finmultitime_us_rolling_3fold_5seed_v1'),('after','finmultitime_us_calibrated_rolling_3fold_5seed_v1')]:
        for seed in seeds:
            run=repo/'runs/protocols'/protocol/'runs'/f'fold_1_seed_{seed}'
            reported=json.loads((run/'run_summary.json').read_text())['test_metrics']
            rows[stage].append(reported)
            if stage!='after':continue
            source=run/'predictions/test_predictions.npz';source_hash=sha256_file(source)
            meta_file=cache/f'seed_{seed}.json';loss_file=cache/f'seed_{seed}.npz'
            if meta_file.exists() and loss_file.exists():
                metadata=json.loads(meta_file.read_text())
                if metadata.get('source_sha256')!=source_hash or metadata.get('loss_sha256')!=sha256_file(loss_file):raise RuntimeError('Reference cache/source hash mismatch')
                if metadata.get('schema')!='trefin-float64-exact-ES-reference-v1':raise RuntimeError('Reference cache schema mismatch')
                with np.load(loss_file) as f:reference[seed]={k:f[k] for k in f.files}
                reference[seed]['target_hash']=metadata['target_hash']
                continue
            with np.load(source) as f:
                mean=f['mu'].astype(np.float64);target32=f['y'];target=target32.astype(np.float64);indices=f['indices'];samples=f['samples'].astype(np.float64)
            assert samples.shape==(256,500,5,32) and target.shape==(500,5,32)
            es=per_origin_energy_score(samples,target)
            score_difference=abs(float(es.mean())-reported['energy_score'])
            assert score_difference<1e-7
            target_hash=hashlib.sha256(np.ascontiguousarray(target32,dtype=np.float32).tobytes()).hexdigest()
            arrays={'mse':((mean-target)**2).mean((1,2)),'mae':np.abs(mean-target).mean((1,2)),'es':es,'indices':indices}
            np.savez_compressed(loss_file,**arrays)
            metadata={'schema':'trefin-float64-exact-ES-reference-v1','seed':seed,'source':str(source),'source_sha256':source_hash,'loss_sha256':sha256_file(loss_file),'target_hash':target_hash,'energy_score_recomputed':float(es.mean()),'difference_from_archived_float32_reduction':score_difference,'scenarios':len(samples),'origins':len(target),'horizon':target.shape[1],'assets':target.shape[2],'pairwise_method':'scipy.spatial.distance.pdist in float64, all unordered pairs, divisor S*(S-1)'}
            meta_file.write_text(json.dumps(metadata,indent=2),encoding='utf8')
            reference[seed]={**arrays,'target_hash':target_hash}
            del samples,mean,target,target32
            print(json.dumps({'event':'reference_ES_cached','seed':seed,'energy_score':metadata['energy_score_recomputed'],'roundoff_difference':score_difference}),flush=True)
    return reference,rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--tedm',type=Path)
    p.add_argument('--timerxl',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-cache',type=Path)
    p.add_argument('--reference-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.repo.resolve()/'src'))
    from fconcord.statistical_tests import diebold_mariano_hac,holm_adjust
    seeds=[1,7,19,37,73]
    reference,reference_rows=load_reference(a.repo,a.reference_cache or a.out/'reference_cache')
    if a.reference_only:
        print(json.dumps({'event':'reference_complete','seeds':seeds,'metrics':['mse','mae','es']}));return
    if a.tedm is None or a.timerxl is None:p.error('--tedm and --timerxl are required unless --reference-only is used')
    reference_average={metric:np.mean([reference[s][metric] for s in seeds],axis=0) for metric in ['mse','mae','es']}
    results={};tests={};source_protocols={}
    for family,directory in [('TEDM',a.tedm),('Timer-XL',a.timerxl)]:
        source_protocols[family]=json.loads((directory/'protocol.json').read_text())
        assert source_protocols[family]['seeds']==seeds
        rows=[];losses=[]
        for seed in seeds:
            folder=directory/f'seed_{seed}'
            result=json.loads((folder/'result.json').read_text())
            assert result['seed']==seed
            assert result['test_targets_float32_bytes_sha256']==reference[seed]['target_hash']
            assert result['reload_validation_ES_difference']<1e-7
            with np.load(folder/'paired_losses.npz') as f:
                arrays={k:f[k] for k in f.files}
            assert np.array_equal(arrays['indices'],reference[seed]['indices'])
            for stage in ['before','after']:
                assert abs(arrays[f'mse_{stage}'].mean()-result[stage]['mse'])<1e-12
                assert abs(arrays[f'es_{stage}'].mean()-result[stage]['energy_score'])<1e-9
            rows.append(result);losses.append(arrays)
        results[family]={'runs':rows,'metrics':{}}
        for stage in ['before','after']:
            results[family]['metrics'][stage]={key:{'mean':float(np.mean([r[stage][key] for r in rows])),'sample_std':float(np.std([r[stage][key] for r in rows],ddof=1))} for key in ['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']}
        tests[family]={}
        for metric in ['mse','mae','es']:
            candidate=np.mean([v[f'{metric}_after'] for v in losses],axis=0)
            ref=reference_average[metric]
            tests[family][metric]={
                'definition':f'Calibrated {family} minus calibrated TREFIN; five seed losses averaged at each identical origin; positive favors TREFIN',
                'block20':paired_circular_bootstrap(candidate,ref,20,repetitions=10000,seed=20261108),
                'block5':paired_circular_bootstrap(candidate,ref,5,repetitions=10000,seed=20261108),
                'block10':paired_circular_bootstrap(candidate,ref,10,repetitions=10000,seed=20261108),
                'dm_hac20':diebold_mariano_hac(candidate,ref,5,max_lag=20)}
    for metric in ['mse','mae','es']:
        for block in [5,10,20]:
            correction=holm_adjust({family:tests[family][metric][f'block{block}']['two_sided_bootstrap_p_value'] for family in tests})
            for family,value in correction.items():
                tests[family][metric].setdefault('holm_adjusted_bootstrap_p_by_block',{})[str(block)]=value
                if block==20:tests[family][metric]['holm_adjusted_bootstrap_p']=value
        correction=holm_adjust({family:tests[family][metric]['dm_hac20']['two_sided_p_value'] for family in tests})
        for family,value in correction.items():tests[family][metric]['holm_adjusted_dm_p']=value
    from summarize_retraining import __file__ as bootstrap_source
    report={'scope':'New task-trained official-upstream supplements; historical separate-host results are distinct and are not reconstructed here','seed_average_before_temporal_inference':True,'primary_loss':'MSE','secondary_losses':['MAE','ES'],'reference_ES':'Original archived TREFIN NPZ; float64 exact all-pairs pdist computed once and source-hash cached','bootstrap_repetitions':10000,'bootstrap_p_value':'null-centered two-sided test with +1 correction; percentile 95% confidence intervals','primary_block_length':20,'sensitivity_block_lengths':[5,10],'source_sha256':{'summarizer':sha256_file(__file__),'bootstrap_implementation':sha256_file(bootstrap_source)},'paired_tests':tests,'results':results,'protocols':source_protocols}
    report['multiplicity_family']={'comparators':['TEDM','Timer-XL'],'adjustment':'Holm separately within each metric and block length; other baseline families excluded'}
    report['reference_metrics']={stage:{key:{'mean':float(np.mean([r.get(key,r['rmse']**2) for r in reference_rows[stage]])),'sample_std':float(np.std([r.get(key,r['rmse']**2) for r in reference_rows[stage]],ddof=1))} for key in ['mse','rmse','mae','energy_score']} for stage in ['before','after']}
    (a.out/'fresh_official_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# Matched TEDM and Timer-XL comparisons','',
           'The comparison uses the pinned official backbones and the task adaptations documented in the accompanying protocol.','',
           'The five fixed training seeds are 1, 7, 19, 37 and 73. Both methods use the exact fold-1 origins and targets; checkpoint and learning-rate candidate selection use validation ES only. The same positive mean-calibration family is then fitted on validation scenarios. Results below are completed runs, with mean ± sample SD over seeds.','',
           '| Method | Stage | MSE x10^4 | RMSE x10^2 | MAE x10^2 | ES x10^2 |','|---|---|---:|---:|---:|---:|']
    for family,r in results.items():
        for stage in ['before','after']:
            cells=[f"{r['metrics'][stage][key]['mean']*scale:.4f} ± {r['metrics'][stage][key]['sample_std']*scale:.4f}" for key,scale in [('mse',1e4),('rmse',100),('mae',100),('energy_score',100)]]
            lines.append(f"| {family} | {stage} mean calibration | "+' | '.join(cells)+' |')
    for stage in ['before','after']:
        cells=[]
        for key,scale in [('mse',1e4),('rmse',100),('mae',100),('energy_score',100)]:
            v=[r.get(key,r['rmse']**2) for r in reference_rows[stage]]
            cells.append(f'{np.mean(v)*scale:.4f} ± {np.std(v,ddof=1)*scale:.4f}')
        lines.append(f'| TREFIN | {stage} mean calibration | '+' | '.join(cells)+' |')
    lines+=['','## Paired temporal tests','',
            'Losses are averaged over the five seeds at each of the 500 identical test origins before temporal inference. Positive differences favor TREFIN. Primary circular-block bootstrap intervals use blocks of 20 origins and 10,000 resamples; two-sided p values use null-centered bootstrap means with a +1 correction. Blocks of five and ten origins and DM with HAC lag 20 provide sensitivity checks. Holm adjustment is across the two comparisons within each loss and block choice. TREFIN energy scores are recomputed from its original scenarios using float64 exact pairwise distances.','',
            '| Comparator | Loss | Difference | 95% block-20 interval | Holm-adjusted p |','|---|---|---:|---:|---:|']
    for family in tests:
        for metric in ['mse','mae','es']:
            t=tests[family][metric];b=t['block20'];lo,hi=b['confidence_interval']
            pvalue=t['holm_adjusted_bootstrap_p'];pdisplay=f'{pvalue:.4g}'
            lines.append(f"| {family} | {metric.upper()} | {b['mean_loss_difference_candidate_minus_reference']:.8g} | [{lo:.8g}, {hi:.8g}] | {pdisplay} |")
    lines+=['','## Runtime and calibration','', '| Method | Mean selected rho | Total optimizer updates (both candidates, all seeds) | Total candidate training+validation seconds |','|---|---:|---:|---:|']
    for family,r in results.items():
        candidates=[c for row in r['runs'] for c in row['candidate_summaries']]
        lines.append(f"| {family} | {np.mean([row['rho'] for row in r['runs']]):.5f} | {sum(c['optimizer_updates'] for c in candidates)} | {sum(c['training_and_validation_seconds'] for c in candidates):.1f} |")
    lines+=['','All target hashes, test indices, saved per-origin loss means, checkpoint reload comparisons and calibration artifacts were checked. Timer-XL uses the specified training-residual scenario adapter; TEDM uses native stochastic samples. The inputs, normalization, heads, objectives, selection budgets and calibration settings are recorded in the accompanying protocol.']
    (a.out/'fresh_official_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'completed_seeds_per_family':5,'paired_test_count':6,'verified':True}))


if __name__=='__main__':main()
