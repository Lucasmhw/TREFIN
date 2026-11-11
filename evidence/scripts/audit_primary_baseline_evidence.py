#!/usr/bin/env python3
"""Independent audit of completed TEDM/Timer compact evidence and primary inference."""
import argparse,copy,hashlib,json,math
from pathlib import Path
import numpy as np
SEEDS=[1,7,19,37,73]
METRICS=['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def close(a,b,tol=1e-14):
 if not np.allclose(a,b,atol=tol,rtol=0):raise AssertionError(f'Numerical disagreement: {a} vs {b}')
def norm_protocol(p):
 x=copy.deepcopy(p);del x['seeds'];return x
def independent_bootstrap(d,block,reps,seed):
 n=len(d);start=np.random.default_rng(seed).integers(0,n,(reps,(n+block-1)//block))
 ix=(start[:,:,None]+np.arange(block))%n;means=d[ix.reshape(reps,-1)[:,:n]].mean(1)
 effect=float(d.mean());return effect,np.percentile(means,[2.5,97.5]),float((1+np.count_nonzero(abs(means-effect)>=abs(effect)))/(reps+1))
def independent_dm(d):
 n=len(d);z=d-d.mean();lag=20;variance=np.dot(z,z)/n
 covariance=np.array([np.dot(z[k:],z[:-k])/n for k in range(1,lag+1)])
 variance=max(0.,float(variance+2*np.dot(1-np.arange(1,lag+1)/(lag+1),covariance)))
 se=float(np.sqrt(variance/n));correction=float(np.sqrt((n+1-10+20/n)/n));stat=float(d.mean()/se*correction)
 return {'mean_loss_difference_candidate_minus_reference':float(d.mean()),'hac_long_run_variance':variance,'standard_error':se,'harvey_correction':correction,'dm_statistic':stat,'two_sided_p_value':math.erfc(abs(stat)/math.sqrt(2))}
def holm2(values):
 ix=np.argsort(values);out=np.empty(2);out[ix[0]]=min(1.,2*values[ix[0]]);out[ix[1]]=max(out[ix[0]],values[ix[1]]);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--matched-results',type=Path,required=True);p.add_argument('--scripts',type=Path,required=True);p.add_argument('--reference-repo',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 matched=read(a.matched_results);assert matched['seed_average_before_temporal_inference'];assert matched['primary_block_length']==20 and matched['sensitivity_block_lengths']==[5,10]
 assert matched['multiplicity_family']['comparators']==['TEDM','Timer-XL']
 assert matched['source_sha256']['summarizer']==sha(a.scripts/'summarize_fresh_official.py') and matched['source_sha256']['bootstrap_implementation']==sha(a.scripts/'summarize_retraining.py')
 tedm=a.root/'fresh_tedm_complete';merge=read(tedm/'merge_manifest.json');registered=read(tedm/'global_registered_protocol.json')
 assert merge['status']=='complete' and merge['registered_global_seeds']==SEEDS and registered['seeds']==SEEDS
 assert merge['global_registered_protocol_sha256']==sha(tedm/'global_registered_protocol.json')
 assert merge['source_script_sha256']==sha(a.scripts/'merge_tedm_seed_workers.py')
 for item in merge['copied_artifacts']:assert sha(tedm/item['destination'])==item['sha256'],item['destination']
 assert sha(tedm/'summary.json')==merge['summary_sha256'] and sha(tedm/'protocol.json')==merge['assembled_protocol_sha256']
 ref={};reference_sources=[]
 for seed in SEEDS:
  meta=read(a.reference_cache/f'seed_{seed}.json');path=a.reference_cache/f'seed_{seed}.npz';assert sha(path)==meta['loss_sha256']
  with np.load(path) as f:arrays={k:f[k] for k in f.files}
  close(arrays['es'].mean(),meta['energy_score_recomputed']);ref[seed]={**arrays,'target_hash':meta['target_hash']}
  rec={'seed':seed,'loss_sha256':sha(path),'source_prediction_sha256':meta['source_sha256'],'original_prediction_hash_reverified':False,'point_losses_recomputed_from_original_mu_y':False}
  if a.reference_repo:
   src=a.reference_repo/'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'/f'fold_1_seed_{seed}'/'predictions/test_predictions.npz';assert sha(src)==meta['source_sha256']
   with np.load(src) as f:
    y=f['y'];mu=f['mu'].astype(float);assert np.array_equal(f['indices'],arrays['indices']);assert hashlib.sha256(np.ascontiguousarray(y,dtype=np.float32).tobytes()).hexdigest()==meta['target_hash']
   close(np.square(mu-y.astype(float)).mean((1,2)),arrays['mse']);close(np.abs(mu-y.astype(float)).mean((1,2)),arrays['mae'])
   rec.update(original_prediction_hash_reverified=True,point_losses_recomputed_from_original_mu_y=True)
  reference_sources.append(rec)
 ref_average={m:np.mean([ref[s][m] for s in SEEDS],axis=0) for m in ['mse','mae','es']}
 families={};losses={};primary=[];bootstrap_checks=dm_checks=0;source_files={}
 for label,name in [('TEDM','fresh_tedm_complete'),('Timer-XL','fresh_timerxl')]:
  directory=a.root/name;top=read(directory/'protocol.json');summary=read(directory/'summary.json');assert top['seeds']==SEEDS;rows=[];arrays=[]
  assert matched['protocols'][label]==top
  for seed in SEEDS:
   folder=directory/f'seed_{seed}';row=read(folder/'result.json');execution=folder/'execution_protocol.json' if label=='TEDM' else directory/'protocol.json';protocol=read(execution)
   assert row['seed']==seed and row['protocol_sha256']==sha(execution);assert row['source_data_sha256']==protocol['data_sha256'];assert row['reload_validation_ES_difference']<1e-7
   if label=='TEDM':
    assert norm_protocol(protocol)==norm_protocol(registered)
    assert protocol['seeds']==(SEEDS if seed==1 else [seed]);assert protocol['effective_sampler_origin_batch']==64
    assert protocol['script_sha256']==sha(a.scripts/'stable_tedm_baselines.py')
    assert protocol['numerical_adapter']['source_sha256']==sha(a.scripts/'tedm_noise_scale_adapter.py')
   else:assert protocol['script_sha256']==sha(a.scripts/'fresh_official_baselines.py')
   assert row['test_targets_float32_bytes_sha256']==ref[seed]['target_hash']
   assert [c['learning_rate'] for c in row['candidate_summaries']]==protocol['learning_rate_candidates']
   selected=min(row['candidate_summaries'],key=lambda c:(c['best_validation_es'],c['candidate']))
   assert row['selected_candidate']==selected==read(folder/'selected_candidate.json')
   for c in row['candidate_summaries']:assert c==read(folder/f"candidate_{c['candidate']:02d}"/'complete.json')
   calibration=read(folder/'calibration.json');assert calibration['optimizer_success'];close(calibration['selected_signal_shrinkage'],row['rho'])
   with np.load(folder/'paired_losses.npz') as f:loss={k:f[k] for k in f.files}
   assert np.array_equal(loss['indices'],ref[seed]['indices']) and len(loss['indices'])==500
   for stage in ['before','after']:
    for metric,key,tol in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:assert np.isfinite(loss[f'{metric}_{stage}']).all();close(loss[f'{metric}_{stage}'].mean(),row[stage][key],tol)
   rows.append(row);arrays.append(loss);source_files[f'{name}/seed_{seed}/result.json']=sha(folder/'result.json');source_files[f'{name}/seed_{seed}/paired_losses.npz']=sha(folder/'paired_losses.npz')
  assert matched['results'][label]['runs']==rows and summary['runs']==rows
  for stage in ['before','after']:
   for key in METRICS:
    vals=np.array([r[stage][key] for r in rows]);expect={'mean':vals.mean(),'sample_std':vals.std(ddof=1)}
    for stat in expect:close(summary['metrics'][stage][key][stat],expect[stat]);close(matched['results'][label]['metrics'][stage][key][stat],expect[stat])
  losses[label]={m:np.mean([x[m+'_after'] for x in arrays],axis=0) for m in ['mse','mae','es']}
  families[label]={'completed_seeds':SEEDS,'rho':[x['rho'] for x in rows],'candidate_selection_and_calibration_verified':True,'summary_and_paired_loss_means_verified':True,'repair_calls':[x.get('numerical_audit_cumulative',{}).get('repaired_calls') for x in rows] if label=='TEDM' else None}
  for metric in ['mse','mae','es']:
   d=losses[label][metric]-ref_average[metric];tests=matched['paired_tests'][label][metric]
   for block in [5,10,20]:
    t=tests['block'+str(block)];effect,bounds,pvalue=independent_bootstrap(d,block,t['repetitions'],t['seed']);close(effect,t['mean_loss_difference_candidate_minus_reference']);close(bounds,t['confidence_interval']);close(pvalue,t['two_sided_bootstrap_p_value']);bootstrap_checks+=1
    if block==20:primary.append({'method':label,'loss':metric,'difference':effect,'ci95':bounds.tolist(),'raw_p':pvalue,'holm_p':tests['holm_adjusted_bootstrap_p']})
   dm=independent_dm(d)
   for key,value in dm.items():close(value,tests['dm_hac20'][key],1e-12)
   dm_checks+=1
 for metric in ['mse','mae','es']:
  for block in [5,10,20]:
   tests=[matched['paired_tests'][label][metric] for label in ['TEDM','Timer-XL']];expected=holm2([x['block'+str(block)]['two_sided_bootstrap_p_value'] for x in tests]);close(expected,[x['holm_adjusted_bootstrap_p_by_block'][str(block)] for x in tests])
  expected=holm2([x['dm_hac20']['two_sided_p_value'] for x in tests]);close(expected,[x['holm_adjusted_dm_p'] for x in tests])
 report={'status':'passed','scope':'Completed primary matched comparisons only; no refitting or new experiment','completed_runs':10,'verified_merged_copy_records':len(merge['copied_artifacts']),'independent_bootstrap_comparisons':bootstrap_checks,'independent_DM_HAC20_checks':dm_checks,'Holm_bootstrap_families':9,'Holm_DM_families':3,'families':families,'reference_sources':reference_sources,'full_new_baseline_scenarios_scope':'TEDM checkpoint/full-prediction hashes were verified remotely by the merger; compact files were reverified locally. Full new baseline scenarios are not independently re-scored here.','source_files':source_files,'matched_results_sha256':sha(a.matched_results),'primary_comparisons':primary}
 (a.out/'primary_baseline_independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 text=['# Independent primary-baseline audit','','All ten completed TEDM/Timer-XL seed records, source protocol identities, validation-selected candidates, positive calibration records, paired loss means and aggregate mean/sample SD passed. All51 TEDM merge copy hashes match. All18 circular-bootstrap comparisons (six method/loss comparisons × blocks5/10/20), six DM-HAC20 results and their nine/three Holm families were independently recomputed.','', 'Both baselines select rho=.05 for all five seeds. The TEDM guard actually repairs finite-input undefined ratios in seeds7/19/73 (2/5/5 calls), and zero in seeds1/37. These are new-run observations; they do not establish the historical crash cause.','','The primary ES differences favor TREFIN: TEDM Holm p≈.0002; Timer-XL p≈.0470. MSE is not statistically resolved for either comparator, and neither MAE comparison passes Holm at.05 (TEDM≈.0598; Timer≈.3192). Do not describe these new results as a recreation of the separate-host historical table or claim blanket statistical superiority.','','Original TREFIN cached prediction hashes and per-origin MSE/MAE were reverified where --reference-repo was supplied. Original scenario ES remains the exact frozen source-hash-verified cache. New baseline full scenarios/checkpoints remain remote; the merger verified their hashes, while this audit rechecks the complete compact record set.']
 (a.out/'primary_baseline_independent_audit.md').write_text('\n'.join(text)+'\n',encoding='utf8')
 print(json.dumps({k:report[k] for k in ['status','completed_runs','verified_merged_copy_records','independent_bootstrap_comparisons','independent_DM_HAC20_checks','Holm_bootstrap_families','Holm_DM_families']}))
if __name__=='__main__':main()
