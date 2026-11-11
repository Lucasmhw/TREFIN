"""Compact, reproducible tables from the completed decoder audit (no fitting)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def stat(x):
    x=np.asarray(x,dtype=float).reshape(-1)
    assert np.isfinite(x).all()
    return dict(n=int(x.size),minimum=float(x.min()),median=float(np.median(x)),q95=float(np.quantile(x,.95)),maximum=float(x.max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('work/theory_v3/decoder_results'));p.add_argument('--output',type=Path,default=Path('work/theory_v3/decoder_summary_compact.json'));a=p.parse_args()
    protocol=json.loads((a.results/'protocol.json').read_text());fields=protocol['fields'];vi=protocol['variants'].index('implemented_local')
    arrays={}
    for fold in [1,2,3]:
        runs=[]
        for seed in [1,7,19,37,73]:
            with np.load(a.results/f'fold_{fold}_seed_{seed}.npz',allow_pickle=False) as z:
                runs.append(z['values'][:,:,:,vi,:])
        arrays[fold]=np.concatenate(runs,axis=0)
    out={'source_summary_sha256':hashlib.sha256((a.results/'summary.json').read_bytes()).hexdigest(),'variant':'implemented_local','metric':'One-lead Euclidean W1 on the 32-asset vector, raw-return units; state sum-block Frobenius norm','radii':[]}
    for ri,rc in enumerate([.001,.01,.1,1.]):
        row={'concept_radius':rc,'graph_radius':rc/2,'initial_fraction':.1,'folds':{}}
        combined=[]
        for fold,v in arrays.items():
            x=v[:,ri];combined.append(x)
            row['folds'][str(fold)]={'L_D':stat(x[...,fields.index('L_D_sum_block_norm')]),'lead5_composed_W1_basis_points':stat(x[:,4,fields.index('propagated_W1_upper_bound')]*1e4)}
        x=np.concatenate(combined,axis=0)
        row['all_folds']={'L_D':stat(x[...,fields.index('L_D_sum_block_norm')]),'lead5_composed_W1_basis_points':stat(x[:,4,fields.index('propagated_W1_upper_bound')]*1e4),'precision_fraction_in_A_bound':stat(x[...,fields.index('precision_A')]/x[...,fields.index('L_A')])}
        out['radii'].append(row)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for row in out['radii']:
        print(row['concept_radius'], ' | '.join(f"F{f}: LD {r['L_D']['median']:.6g}/{r['L_D']['maximum']:.6g}; W1bp {r['lead5_composed_W1_basis_points']['median']:.6g}/{r['lead5_composed_W1_basis_points']['maximum']:.6g}" for f,r in row['folds'].items()))
        print('pooled',row['all_folds'])

if __name__=='__main__':main()
