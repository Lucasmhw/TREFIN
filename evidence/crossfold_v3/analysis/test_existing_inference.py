"""Regression checks on real completed fold-1 records; no new forecast results."""
import argparse
from pathlib import Path
import numpy as np
from common import SEEDS,FAMILIES,read,dump,sha,close,holm,circular_bootstrap,dm_hac20
from summarize_crossfold import reference,load_completed,OLD_NAMES
from audit_crossfold import independent_bootstrap,independent_dm,independent_holm


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold1-root',type=Path,required=True)
    p.add_argument('--reference-cache',type=Path,required=True)
    p.add_argument('--original-results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    ref,_=reference(a.reference_cache);old=read(a.original_results);tests={};boot=dm=0
    for family,label in FAMILIES.items():
        _,losses,_=load_completed(a.fold1_root/OLD_NAMES[family],1,family,ref[1])
        if label not in ['TEDM','Timer-XL']:continue
        tests[label]={}
        for metric in ['mse','mae','es']:
            d=np.mean([x[f'{metric}_after'] for x in losses],axis=0)-np.mean([ref[1][s][metric] for s in SEEDS],axis=0)
            tests[label][metric]={}
            for block in [5,10,20]:
                result=circular_bootstrap(d,block);expected=old['paired_tests'][label][metric][f'block{block}']
                close(result['mean_difference'],expected['mean_loss_difference_candidate_minus_reference'],1e-14)
                close(result['ci95'],expected['confidence_interval'],1e-14)
                close(result['raw_p'],expected['two_sided_bootstrap_p_value'],1e-14)
                effect,ci,pvalue=independent_bootstrap(d,block)
                close(effect,result['mean_difference'],1e-14);close(ci,result['ci95'],1e-14);close(pvalue,result['raw_p'],1e-14)
                tests[label][metric][block]=result['raw_p'];boot+=1
            result=dm_hac20(d);independent=independent_dm(d)
            for key,value in independent.items():close(result[key],value,1e-12)
            close(result['raw_p'],old['paired_tests'][label][metric]['dm_hac20']['two_sided_p_value'],1e-12);dm+=1
    for metric in ['mse','mae','es']:
        for block in [5,10,20]:
            correction=holm({label:tests[label][metric][block] for label in tests})
            for label,value in correction.items():close(value,old['paired_tests'][label][metric]['holm_adjusted_bootstrap_p_by_block'][str(block)],1e-14)
    values=[.001,.004,.012,.03,.06,.9];expected=[.006,.020,.048,.09,.12,.9]
    correction=holm(dict(enumerate(values)));close(list(correction.values()),expected,1e-14)
    close(independent_holm(values),expected,1e-14)
    report={'status':'passed','scope':'Regression on existing completed fold-1 evidence; not new forecasts',
        'baseline_seed_records_checked':20,'reference_seed_caches_checked':15,
        'original_fold1_bootstrap_results_reproduced':boot,'original_fold1_DM_results_reproduced':dm,
        'original_fold1_Holm2_families_reproduced':9,'Holm6_known_value_check':True,
        'original_fold1_results_sha256':sha(a.original_results)}
    dump(a.out/'analysis_regression_verification.json',report);print(report)


if __name__=='__main__':main()
