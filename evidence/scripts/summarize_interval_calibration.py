"""Aggregate the 15 completed frozen-model interval recalibration diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    runs=[json.loads(f.read_text(encoding='utf-8')) for f in sorted(a.root.glob('fold_*_seed_*/result.json'))]
    if len(runs)!=15:raise ValueError(f'Expected all 15 runs, found {len(runs)}')
    rows=[];report={'folds':{},'audit':{}}
    for fold in [1,2,3]:
        group=[r for r in runs if r['fold']==fold]
        if sorted(r['seed'] for r in group)!=[1,7,19,37,73]:raise ValueError('Seed set incomplete')
        output=[]
        for method in ['original','validation_ES_scale','validation_90pct_scale']:
            vals={
                'mse': [r['test_metrics'][method]['mse'] for r in group],
                'energy_score': [r['test_metrics'][method]['energy_score'] for r in group],
                'coverage90': [r['test_metrics'][method]['coverage']['0.9'] for r in group],
                'width90': [r['test_metrics'][method]['interval_width']['0.9'] for r in group],
                'scale': [1. if method=='original' else r['scale_es' if method=='validation_ES_scale' else 'scale_coverage90'] for r in group]}
            row={'fold':fold,'method':method}
            for k,v in vals.items():row[k+'_mean']=float(np.mean(v));row[k+'_seed_std']=float(np.std(v,ddof=1))
            rows.append(row);output.append(row)
        report['folds'][str(fold)]=output
    report['audit']={'runs':len(runs),'maximum_test_mean_replay_absolute_error':max(r['test_mean_replay_max_absolute_error'] for r in runs),
                     'all_validation_ES_nonincreasing':all(r['validation_es_scaled']<=r['validation_es_base']+1e-12 for r in runs),
                     'all_scales_positive':all(r['scale_es']>0 and r['scale_coverage90']>0 for r in runs),
                     'minimum_validation_coverage_at_coverage_selected_scale':min(r['coverage90_validation_at_selected'] for r in runs),
                     'coverage_scales_on_boundaries':sum(r['coverage_scale_at_boundary'] for r in runs),
                     'test_sampling':'original saved scenarios; scale chosen on 256 CPU validation scenarios only',
                     'point_forecasts':'archived rho and means fixed; no point forecast changes',
                     'scope':'empirical pooled marginal coverage targeting, not a distribution-free or time-shift coverage guarantee'}
    (a.root/'aggregate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.root/'aggregate.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
