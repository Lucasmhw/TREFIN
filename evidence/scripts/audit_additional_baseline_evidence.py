#!/usr/bin/env python3
"""Audit completed TimeMixer++/TimeXer compact records; descriptive comparison only."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_primary_baseline_evidence import SEEDS,METRICS,sha,read,close,norm_protocol

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--additional-results',type=Path,required=True);p.add_argument('--fast-bundle',type=Path,required=True);p.add_argument('--scripts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 report=read(a.additional_results);assert report['inference_status']=='None; descriptive calibration table';assert 'diagnostic_unadjusted_tests' not in report;assert report['source_sha256']==sha(a.scripts/'summarize_additional_baselines.py')
 folder=a.root/'fresh_timemixerpp_complete';merge=read(folder/'merge_manifest.json');reg=read(folder/'global_registered_protocol.json');assert merge['registered_global_seeds']==SEEDS and merge['status']=='complete';assert merge['source_script_sha256']==sha(a.scripts/'merge_timemixerpp_seed_workers.py');assert merge['global_registered_protocol_sha256']==sha(folder/'global_registered_protocol.json')
 for rec in merge['copied_artifacts']:assert sha(folder/rec['destination'])==rec['sha256']
 assert sha(folder/'summary.json')==merge['summary_sha256'] and sha(folder/'protocol.json')==merge['assembled_protocol_sha256']
 fast=read(a.fast_bundle/'manifest.json');accept=read(a.fast_bundle/'numerical_acceptance.json')
 for path,h in fast['files'].items():assert sha(a.fast_bundle/path)==h
 for name,group in fast['upstreams'].items():
  for path,h in group['files'].items():assert sha(a.fast_bundle/'upstream'/name/path)==h
 assert not accept['original_strict_threshold_passed'] and accept['original_failures_retained'];assert accept['production_outer_inference_batch']==32 and accept['training_batch_size']==64
 families={};checked_sources=0
 for label,name in [('TimeMixer++','fresh_timemixerpp_complete'),('TimeXer','fresh_timexer')]:
  directory=a.root/name;top=read(directory/'protocol.json');summary=read(directory/'summary.json');rows=[];assert top['seeds']==SEEDS and report['results'][label]['protocol']==top
  for seed in SEEDS:
   d=directory/f'seed_{seed}';row=read(d/'result.json');ep=d/'execution_protocol.json' if label=='TimeMixer++' else directory/'protocol.json';protocol=read(ep)
   assert row['seed']==seed and row['protocol_sha256']==sha(ep);assert row['reload_validation_ES_difference']<1e-7
   if label=='TimeMixer++':
    assert norm_protocol(protocol)==norm_protocol(reg) and protocol['seeds']==(SEEDS if seed==1 else [seed])
    assert protocol['batch_size']==64 and protocol['outer_point_inference_batch_size']==32 and protocol['effective_native_inference_batch_size']==32
    assert protocol['script_sha256']==fast['files']['fresh_point_baselines_fast.py'] and protocol['source_manifest_sha256']==sha(a.fast_bundle/'manifest.json')
    assert protocol['numerical_acceptance']==accept and protocol['numerical_acceptance_sha256']==sha(a.fast_bundle/'numerical_acceptance.json')
   cache=read(a.reference_cache/f'seed_{seed}.json');assert row['test_targets_float32_bytes_sha256']==cache['target_hash']
   assert row['source_data_sha256']==protocol['data_sha256'];assert [c['learning_rate'] for c in row['candidate_summaries']]==protocol['learning_rate_candidates']
   selected=min(row['candidate_summaries'],key=lambda r:(r['best_validation_es'],r['candidate']));assert row['selected_candidate']==selected==read(d/'selected_candidate.json')
   for c in row['candidate_summaries']:assert c==read(d/f"candidate_{c['candidate']:02d}"/'complete.json')
   calibration=read(d/'calibration.json');assert calibration['optimizer_success'];close(row['rho'],calibration['selected_signal_shrinkage'])
   with np.load(d/'paired_losses.npz') as f, np.load(a.reference_cache/f'seed_{seed}.npz') as ref:
    assert np.array_equal(f['indices'],ref['indices']) and len(f['indices'])==500
    for stage in ['before','after']:
     for metric,key,tol in [('mse','mse',1e-12),('mae','mae',1e-12),('es','energy_score',1e-9)]:assert np.isfinite(f[f'{metric}_{stage}']).all();close(f[f'{metric}_{stage}'].mean(),row[stage][key],tol)
   rows.append(row)
  assert summary['runs']==rows==report['results'][label]['runs']
  for stage in ['before','after']:
   for key in METRICS:
    values=np.array([r[stage][key] for r in rows])
    for stat,value in [('mean',values.mean()),('sample_std',values.std(ddof=1))]:close(summary['metrics'][stage][key][stat],value);close(report['results'][label]['metrics'][stage][key][stat],value)
  for source,h in report['results'][label]['source_sha256'].items():
   normalized=source.replace('\\','/');suffix=normalized.split(name+'/',1)[1];assert sha(directory/suffix)==h;checked_sources+=1
  families[label]={'seeds':SEEDS,'rho':[r['rho'] for r in rows],'protocols_candidates_calibration_verified':True,'eight_metric_means_and_sample_SD_verified':True,'paired_loss_means_and_target_identity_verified':True,'metrics':summary['metrics']}
 output={'status':'passed','scope':'Completed additional descriptive comparison; no new hypothesis tests or change to TEDM/Timer multiplicity family','completed_runs':10,'TimeMixer_merged_artifact_hashes_verified':len(merge['copied_artifacts']),'additional_summary_source_hashes_verified':checked_sources,'fast_frozen_source_and_acceptance_hashes_verified':True,'original_strict_numerical_failure_preserved':True,'bitwise_CUDA_equivalence_claim':False,'families':families,'source_results_sha256':sha(a.additional_results),'full_scenarios_scope':'Remote merger verified full scenario/checkpoint hashes; local audit verifies compact records and saved losses, not omitted scenarios'}
 (a.out/'additional_baseline_independent_audit.json').write_text(json.dumps(output,indent=2),encoding='utf8')
 (a.out/'additional_baseline_independent_audit.md').write_text('All five completed TimeMixer++ seeds and five TimeXer seeds passed independent compact-record validation. All46 TimeMixer++ copied artifacts and20 source hashes in the descriptive report match. Candidate selection, frozen protocol identity, numerical-acceptance identity, target/origin alignment, per-origin loss means and all eight metric means/sample SD were verified. TimeMixer++ uses the frozen inference-batch32/training-batch64 implementation and preserves the original stricter numerical failure. This is numerical agreement, not bitwise CUDA equivalence. The additional summary is descriptive and does not change the TEDM/Timer-XL Holm family. Full scenarios/checkpoints remain remote; the merge verified their hashes.\n',encoding='utf8')
 print(json.dumps({k:output[k] for k in ['status','completed_runs','TimeMixer_merged_artifact_hashes_verified','additional_summary_source_hashes_verified']}))
if __name__=='__main__':main()
