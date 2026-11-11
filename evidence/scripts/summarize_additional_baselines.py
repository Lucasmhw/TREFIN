#!/usr/bin/env python3
"""Descriptive matched-adapter tables, separate from the TEDM/Timer Holm family.

Example: --family 'TimeXer=path/fresh_timexer' --family 'TimeMixer++=path/fresh_timemixerpp'
--diagnostic-inference is for partial-run QA only: tests are explicitly unadjusted.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from summarize_fresh_official import sha256_file,paired_circular_bootstrap

SEEDS=[1,7,19,37,73]
KEYS=['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']

def load_family(directory,reference_cache):
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text())
    assert protocol['seeds']==SEEDS
    runs=[];losses=[];sources={}
    for seed in SEEDS:
        folder=directory/f'seed_{seed}'
        rp=folder/'result.json';lp=folder/'paired_losses.npz'
        row=json.loads(rp.read_text());meta=json.loads((reference_cache/f'seed_{seed}.json').read_text())
        assert row['seed']==seed and row['test_targets_float32_bytes_sha256']==meta['target_hash']
        assert row['reload_validation_ES_difference']<1e-7
        with np.load(lp) as f:a={k:f[k] for k in f.files}
        with np.load(reference_cache/f'seed_{seed}.npz') as f:assert np.array_equal(a['indices'],f['indices'])
        for stage in ['before','after']:
            for loss,metric,tol in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:
                assert np.isfinite(a[f'{loss}_{stage}']).all()
                assert abs(a[f'{loss}_{stage}'].mean()-row[stage][metric])<tol
        runs.append(row);losses.append(a)
        sources[str(rp)]=sha256_file(rp);sources[str(lp)]=sha256_file(lp)
    metrics={stage:{key:{'mean':float(np.mean([r[stage][key] for r in runs])),
                        'sample_std':float(np.std([r[stage][key] for r in runs],ddof=1))}
                    for key in KEYS} for stage in ['before','after']}
    candidates=[c for r in runs for c in r['candidate_summaries']]
    return {'runs':runs,'metrics':metrics,'mean_rho':float(np.mean([r['rho'] for r in runs])),
            'total_optimizer_updates':sum(c['optimizer_updates'] for c in candidates),
            'total_training_validation_seconds':sum(c['training_and_validation_seconds'] for c in candidates),
            'protocol':protocol,'source_sha256':sources},losses

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',action='append',required=True)
    p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--diagnostic-inference',action='store_true');p.add_argument('--repo',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    reference={}
    for metric in ['mse','mae','es']:
        arrays=[]
        for seed in SEEDS:
            cp=a.reference_cache/f'seed_{seed}.npz';meta=json.loads((a.reference_cache/f'seed_{seed}.json').read_text())
            assert sha256_file(cp)==meta['loss_sha256']
            with np.load(cp) as f:arrays.append(f[metric])
        reference[metric]=np.mean(arrays,axis=0)
    report={'schema':'additional-matched-baseline-descriptive-summary-v1','status':'complete for explicitly listed families only',
            'primary_H5_multiplicity_family':'Not modified: TEDM and Timer-XL are assessed together in summarize_fresh_official.py',
            'inference_status':'None; descriptive calibration table' if not a.diagnostic_inference else 'Partial diagnostic only: all p values are raw/unadjusted; final TEDM/Timer Holm correction pending',
            'source_sha256':sha256_file(__file__),'results':{}}
    if a.diagnostic_inference:
        if not a.repo:p.error('--repo is required for diagnostic DM inference')
        sys.path.insert(0,str(a.repo.resolve()/'src'))
        from fconcord.statistical_tests import diebold_mariano_hac
        report['diagnostic_unadjusted_tests']={}
    for item in a.family:
        name,directory=item.split('=',1)
        if name in report['results']:raise ValueError('Repeated family')
        family,losses=load_family(directory,a.reference_cache);report['results'][name]=family
        if a.diagnostic_inference:
            report['diagnostic_unadjusted_tests'][name]={}
            for metric in ['mse','mae','es']:
                candidate=np.mean([v[f'{metric}_after'] for v in losses],axis=0);ref=reference[metric]
                record={f'block{block}':paired_circular_bootstrap(candidate,ref,block) for block in [5,10,20]}
                record['dm_hac20']=diebold_mariano_hac(candidate,ref,5,max_lag=20)
                report['diagnostic_unadjusted_tests'][name][metric]=record
    (a.out/'additional_baseline_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    lines=['# Matched-adapter descriptive calibration table','',report['inference_status'],'',
           '| Method | Stage | MSE x10^4 | RMSE x10^2 | MAE x10^2 | ES x10^2 |','|---|---|---:|---:|---:|---:|']
    for name,row in report['results'].items():
        for stage in ['before','after']:
            values=[f"{row['metrics'][stage][k]['mean']*s:.4f} ± {row['metrics'][stage][k]['sample_std']*s:.4f}" for k,s in [('mse',1e4),('rmse',100),('mae',100),('energy_score',100)]]
            lines.append(f'| {name} | {stage} | '+' | '.join(values)+' |')
    lines+=['','All rows use five seeds; means and sample SD are over seeds. Target hashes, origin alignment, saved per-origin loss means and checkpoint reload were verified. Source implementations are specified in each recorded protocol.']
    (a.out/'additional_baseline_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'families':list(report['results']),'verified':True,'inference_status':report['inference_status']}))

if __name__=='__main__':main()
