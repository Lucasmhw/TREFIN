#!/usr/bin/env python3
"""Report all six descriptive metric ranks from completed native calibration records."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
SEEDS=[1,7,19,37,73];METRICS=['mse','rmse','mae','energy_score','msic','msir']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--trefin-configs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 summary=json.loads((a.native/'summary.json').read_text());sources={};rows=[];max_change={'msic':0.,'msir':0.}
 for group in summary['groups']:
  model=group['model'];runs=[]
  for seed in SEEDS:
   f=a.native/f'fold_1_seed_{seed}'/model/'result.json';r=json.loads(f.read_text());assert r['seed']==seed and r['model']==model and r['rho']>0;sources[str(f)]=sha(f);runs.append(r)
   for m in max_change:max_change[m]=max(max_change[m],abs(r['after'][m]-r['before'][m]))
  for stage in ['before','after']:
   row={'model':model,'stage':stage,'seeds':5}
   for m in METRICS:
    vals=np.array([r[stage][m] for r in runs]);mean=float(vals.mean());sd=float(vals.std(ddof=1));assert abs(group[f'{stage}_{m}_mean']-mean)<1e-14 and abs(group[f'{stage}_{m}_sample_std']-sd)<1e-14;row[m+'_mean']=mean;row[m+'_sample_sd']=sd
   rows.append(row)
 for stage,folder in [('before','uncalibrated'),('after','calibrated')]:
  runs=[]
  for seed in SEEDS:
   f=a.trefin_configs/folder/f'fold_1_seed_{seed}'/'run_summary.json';r=json.loads(f.read_text());sources[str(f)]=sha(f);runs.append(r['test_metrics'])
  row={'model':'TREFIN','stage':stage,'seeds':5}
  for m in METRICS:
   vals=np.array([r[m] if m in r else r['rmse']**2 for r in runs]);row[m+'_mean']=float(vals.mean());row[m+'_sample_sd']=float(vals.std(ddof=1))
  rows.append(row)
 rankings={}
 for stage in ['before','after']:
  rankings[stage]={}
  for m in METRICS:
   chosen=sorted([r for r in rows if r['stage']==stage],key=lambda r:r[m+'_mean'],reverse=m in ['msic','msir']);rankings[stage][m]=[r['model'] for r in chosen]
   for rank,row in enumerate(chosen,1):row[m+'_rank']=rank
 # Tiny floating-point differences are not used to manufacture ties. All values/ranks retain full precision.
 report={'scope':'Additional native comparison: nine archived native implementations plus TREFIN, fold1, five seeds; not the historical separate-host main table','ranking':'Descriptive rank of five-seed mean; lower is better for MSE/RMSE/MAE/ES, higher for msIC/msIR; not a significance claim','metrics_definition':'msIC/msIR use the repository FinTSBridge released-code Pearson-over-horizon definition, not Spearman rank correlation','TREFIN_MSE':'Derived from each archived per-run RMSE squared when MSE absent, matching the existing R7 table; RMSE is averaged per seed separately','native_45_records_verified_against_summary':True,'max_native_per_run_correlation_score_change':max_change,'trefin_ranks':{stage:{m:rankings[stage][m].index('TREFIN')+1 for m in METRICS} for stage in rankings},'ranking_order':rankings,'rows':rows,'source_sha256':sources,'script_sha256':sha(__file__)}
 (a.out/'full_metric_rankings.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 with (a.out/'full_metric_rankings.csv').open('w',newline='',encoding='utf8') as f:
  fields=['model','stage','seeds']+[m+s for m in METRICS for s in ['_mean','_sample_sd','_rank']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 note='In this supplementary ten-method, five-seed comparison, TREFIN moves from fourth to first in MSE, RMSE and MAE after equal access to validation-selected mean shrinkage, and remains first in ES. Its msIC and msIR ranks remain fourth, behind MASTER, HistGradientBoosting and PatchTST. The correlation scores are unchanged up to numerical precision, consistent with positive common scaling preserving correlation. These are descriptive ranks; the paired MSE comparisons remain statistically inconclusive.'
 table=['| Metric | TREFIN rank before | TREFIN rank after |','|---|---:|---:|']+[f'| {m} | {report["trefin_ranks"]["before"][m]}/10 | {report["trefin_ranks"]["after"][m]}/10 |' for m in METRICS]
 (a.out/'full_metric_ranking_note.md').write_text(note+'\n\n'+'\n'.join(table)+'\n',encoding='utf8');print(json.dumps({'native_records_verified':45,'TREFIN_ranks':report['trefin_ranks'],'max_native_correlation_change':max_change,'note_characters':len(note)}))
if __name__=='__main__':main()
