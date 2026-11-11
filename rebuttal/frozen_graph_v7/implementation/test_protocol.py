"""Configuration and access-guard tests; never fit or evaluate forecasting models."""
from pathlib import Path
import argparse,json
import numpy as np
import yaml
from protocol import VARIANTS,SEEDS,CANDIDATES,configure,effective_fingerprint,reference_config,train_validation_only,dump,implementation_hashes

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    configurations={}
    for fold in [1,2,3]:
        expected={}
        for seed in SEEDS:
            base=yaml.safe_load(reference_config(args.root,fold,seed).read_text())
            for variant in VARIANTS:
                configs=[configure(base,variant,c,fold=fold,seed=seed) for c in CANDIDATES]
                fingerprints=[effective_fingerprint(c) for c in configs]
                if seed==1:expected[variant]=fingerprints
                else:assert fingerprints==expected[variant]
                assert len(set(fingerprints))==(4 if variant=='Full' else 3)
                if variant=='FrozenGraph':
                    assert fingerprints[0]==fingerprints[1]
                    assert all(c['training']['lambda_graph']==0 for c in configs)
                assert all(c['model']['signal_shrinkage']==1 for c in configs)
            configurations[f'fold_{fold}_seed_{seed}']=True
    arrays={key:np.arange(12)[:,None].copy() for key in ['c0','a0','s0','c_target','a_target','y','origins']}
    arrays.update(directional_mask=np.array([1,0]),risk_indices=np.array([1]))
    prepared={'arrays':arrays,'metadata':{'split_indices':{'train':[0,1,2,3],'val':[5,6],'test':[8,9,10]},
      'split_date_ranges':{'train':[0,3],'val':[5,6],'test':[8,10]},'num_samples':12}}
    limited,guard=train_validation_only(prepared)
    assert limited['arrays']['origins'].ravel().tolist()==[0,1,2,3,5,6]
    assert limited['metadata']['split_indices']=={'train':[0,1,2,3],'val':[4,5]}
    assert prepared['metadata']['num_samples']==12 and len(prepared['arrays']['origins'])==12
    dump({'status':'passed','configuration_checks':configurations,'effective_unique_candidates':{'Full':4,'FrozenGraph':3},
      'physical_test_exclusion_and_original_input_nonmutation':True,'guard':guard,
      'implementation_sha256':implementation_hashes(),'scope':'Configuration/guard fixtures only; no experimental outcomes'},args.out)
    print('All 15 resolved configurations and physical test-exclusion checks passed.')

if __name__=='__main__':main()
