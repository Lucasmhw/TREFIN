#!/usr/bin/env python3
"""Aggregate the supplemental native calibration replay and validate paired losses."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    repo, run = args.repo.resolve(), args.run.resolve()
    sys.path.insert(0, str(repo / 'src'))
    from fconcord.statistical_tests import circular_block_bootstrap_mean_difference, diebold_mariano_hac, holm_adjust, per_origin_energy_score
    summary = json.loads((run/'summary.json').read_text())
    rows = summary['runs']
    assert len(rows) == 45 and not summary['missing']
    seed_set = [1,7,19,37,73]
    trefin_loss = {}
    trefin_targets = {}
    trefin_rows = {'before':[], 'after':[]}
    reference_indices = None
    for stage, protocol in [('before','finmultitime_us_rolling_3fold_5seed_v1'),('after','finmultitime_us_calibrated_rolling_3fold_5seed_v1')]:
        for seed in seed_set:
            source=repo/'runs/protocols'/protocol/'runs'/f'fold_1_seed_{seed}'
            report=json.loads((source/'run_summary.json').read_text())
            trefin_rows[stage].append(report['test_metrics'])
            if stage=='after':
                with np.load(source/'predictions/test_predictions.npz') as f:
                    m=f['mu'].astype(np.float64);y=f['y'].astype(np.float64);indices=f['indices']
                if reference_indices is None: reference_indices=indices
                assert np.array_equal(indices,reference_indices)
                trefin_loss[seed]=np.mean((m-y)**2,axis=(1,2))
                trefin_targets[seed]=y
    trefin_average_loss=np.mean([trefin_loss[s] for s in seed_set],axis=0)
    tests={}
    verification=[]
    for model in sorted({r['model'] for r in rows}):
        losses=[]
        selected=[r for r in rows if r['model']==model]
        assert sorted(r['seed'] for r in selected)==seed_set
        for r in selected:
            source=repo/Path(r['source_predictions'])
            dest=run/f"fold_1_seed_{r['seed']}"/model
            calibration=json.loads((dest/'validation_selection.json').read_text())
            assert calibration['calibrated_validation_metrics']['energy_score'] <= calibration['uncalibrated_validation_metrics']['energy_score']+1e-8
            with np.load(dest/'paired_test_losses.npz') as f:
                assert np.array_equal(reference_indices,f['test_indices'])
                assert abs(f['mse_after'].mean()-r['after']['mse'])<1e-14
                assert abs(f['mse_before'].mean()-r['before']['mse'])<1e-14
                losses.append(f['mse_after'])
                saved_es=f['es_after'][[0,-1]]
            with np.load(source) as f:
                assert np.array_equal(f['target'].astype(np.float64),trefin_targets[r['seed']])
                target=f['target'][[0,-1]].astype(np.float64)
                mean=f['mean'][[0,-1]]
                pool=f['train_residual_pool'];centered=pool-pool.mean(axis=0,keepdims=True)
                samples=(mean[None]+centered[f['scenario_residual_indices'][:,[0,-1]]]).astype(np.float32).astype(np.float64)
            translated=samples+(r['rho']-1)*mean.astype(np.float64)[None]
            exact_es=per_origin_energy_score(translated,target)
            error=float(np.max(np.abs(exact_es-saved_es)))
            assert error<1e-7
            verification.append({'seed':r['seed'],'model':model,'exact_ES_max_abs_difference':error})
        averaged=np.mean(losses,axis=0)
        tests[model]={
            'definition':'mean calibrated native loss minus mean calibrated TREFIN loss, averaged across five training seeds at each of 500 origins',
            'primary_block20':circular_block_bootstrap_mean_difference(averaged,trefin_average_loss,20,repetitions=5000,seed=20261108),
            'sensitivity_block5':circular_block_bootstrap_mean_difference(averaged,trefin_average_loss,5,repetitions=5000,seed=20261108),
            'dm_hac20':diebold_mariano_hac(averaged,trefin_average_loss,5,max_lag=20),
        }
    adjusted=holm_adjust({name:v['primary_block20']['two_sided_bootstrap_p_value'] for name,v in tests.items()})
    for name,value in adjusted.items(): tests[name]['primary_bootstrap_holm_p']=value
    (run/'paired_native_mse_tests.json').write_text(json.dumps(tests,indent=2),encoding='utf8')
    audit={'checked_runs':45,'checked_ES_origins_per_run':2,'all_passed':True,'max_exact_ES_discrepancy':max(r['exact_ES_max_abs_difference'] for r in verification),'max_device_test_mean_drift':max(r['test_mean_replay_max_error'] for r in rows),'max_validation_ES_replay_difference':max(r['validation_score_replay_abs_difference'] for r in rows),'checks':verification}
    (run/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    lines=['# Supplementary native matched-calibration results','',
           'This is an additional comparison of archived native baseline implementations. It is not a recalibration or reproduction of the separate-host TEDM/Timer-XL main table. All rows use fold 1 and seeds 1, 7, 19, 37, 73. Values are mean ± sample SD across training seeds; these SDs do not quantify market-path sampling uncertainty.','',
           'Each historical selected checkpoint is frozen. The same positive mean scalar in [0.05,1] is fitted by validation ES using 64 fixed residual-bootstrap scenarios; all point forecasts and residual distributions stay otherwise fixed. Test ES uses the original 256 residual-bootstrap draws and all-pairs ES, with the invariant pairwise term reused. TREFIN retains its original asymmetric Student-t scenarios. Hence this controls access to mean shrinkage, not the probabilistic adapter family.','',
           '| Model | MSE before x10^4 | MSE after x10^4 | ES before x10^2 | ES after x10^2 | Mean rho |','|---|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        cell=lambda stage,key,scale:f"{g[f'{stage}_{key}_mean']*scale:.4f} ± {g[f'{stage}_{key}_sample_std']*scale:.4f}"
        lines.append(f"| {g['model']} | {cell('before','mse',1e4)} | {cell('after','mse',1e4)} | {cell('before','energy_score',100)} | {cell('after','energy_score',100)} | {g['rho_mean']:.4f} |")
    tcell=lambda stage,key,scale: f"{np.mean([r.get(key,r['rmse']**2) for r in trefin_rows[stage]])*scale:.4f} ± {np.std([r.get(key,r['rmse']**2) for r in trefin_rows[stage]],ddof=1)*scale:.4f}"
    lines.append(f"| TREFIN (archived model) | {tcell('before','mse',1e4)} | {tcell('after','mse',1e4)} | {tcell('before','energy_score',100)} | {tcell('after','energy_score',100)} | archived |")
    lines+=['','## Paired temporal MSE comparisons','',
            'Five seed-specific losses are averaged at each origin before inference. Positive differences favor TREFIN. Primary intervals use circular blocks of 20 origins, 5,000 resamples; five-origin blocks are retained as a sensitivity check. The hypotheses and intervals concern this observed fold and fixed set of trained models.','',
            '| Calibrated native comparator | Difference x10^4 | 95% interval x10^4 | Holm-adjusted bootstrap p |','|---|---:|---:|---:|']
    for name,t in tests.items():
        b=t['primary_block20'];lo,hi=b['confidence_interval']
        lines.append(f"| {name} | {b['mean_loss_difference_candidate_minus_reference']*1e4:.5f} | [{lo*1e4:.5f}, {hi*1e4:.5f}] | {t['primary_bootstrap_holm_p']:.4f} |")
    lines+=['','## Verification','',f"All 45 runs passed. Independent exact all-pairs ES recomputation on two origins per run differed by at most {audit['max_exact_ES_discrepancy']:.3g}. Maximum CPU-versus-archived test-mean drift was {audit['max_device_test_mean_drift']:.3g} return units; archived test means and training residuals were retained exactly. Validation-only selection used recovered CPU checkpoint inference, with maximum validation ES replay difference {audit['max_validation_ES_replay_difference']:.3g}. No training, checkpoint reselection, or test-driven hyperparameter selection was performed."]
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k!='checks'}))


if __name__=='__main__':main()
