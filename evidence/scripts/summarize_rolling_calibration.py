"""Combine original, static validation, and causal sequential calibration tables."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--rolling',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True);a=p.parse_args()
    rr=[json.loads(x.read_text()) for x in sorted(a.rolling.glob('fold_*_seed_*/result.json'))]
    if len(rr)!=15:raise ValueError('All15 runs required')
    fixed=json.loads((a.fixed/'aggregate.json').read_text())
    rows=[]
    for fold in [1,2,3]:
        for row in fixed['folds'][str(fold)]:rows.append(row)
        g=[r for r in rr if r['fold']==fold]
        row={'fold':fold,'method':'causal_rolling_90pct_scale'}
        vals={'mse':[r['test_metrics']['mse'] for r in g],
              'energy_score':[r['test_metrics']['energy_score'] for r in g],
              'coverage90':[r['test_metrics']['coverage']['0.9'] for r in g],
              'width90':[r['test_metrics']['interval_width']['0.9'] for r in g],
              'scale':[r['scale_mean'] for r in g]}
        for k,v in vals.items():row[k+'_mean']=float(np.mean(v));row[k+'_seed_std']=float(np.std(v,ddof=1))
        rows.append(row)
    report={'rows':rows,'protocol_distinction':'Original/static scales use fixed parameters throughout test. Rolling scale uses fully observed past test labels with five-row maturity delay and126 completed-origin window. No coverage theorem or test-based parameter search.',
            'audit':{'maximum_availability_violation':max(r['maximum_availability_violation'] for r in rr),
                     'all_scales_positive':min(r['scale_min'] for r in rr)>0,
                     'maximum_scale_boundary_fraction':max(r['scale_boundary_fraction'] for r in rr),
                     'point_forecasts_unchanged':all(r['point_forecasts_unchanged'] for r in rr)}}
    (a.rolling/'aggregate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (a.rolling/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
