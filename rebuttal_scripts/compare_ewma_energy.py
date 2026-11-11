"""Pair archived TREFIN energy losses against the new EWMA Student-t baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_archived_evidence import paired_bootstrap, sha256
from ewma_student_baseline import energy_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--ewma-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    result={'definition':'TREFIN minus EWMA Student-t; negative favors TREFIN; seeds averaged within each origin before resampling','folds':{}}
    arrays={}
    for fold in sorted({r['fold'] for r in manifest['runs']}):
        ep=args.ewma_root/fold/'test_predictions.npz'
        with np.load(ep,allow_pickle=False) as f:
            target=f['y'];indices=f['indices'];ewma=f['per_origin_energy_score']
        losses=[];sources=[]
        for row in sorted([r for r in manifest['runs'] if r['fold']==fold],key=lambda x:x['seed']):
            path=Path(row['predictions'])
            if not path.is_absolute():path=args.manifest.resolve().parent/path
            with np.load(path,allow_pickle=False) as f:
                if not np.array_equal(f['indices'],indices) or not np.array_equal(f['y'],target):
                    raise ValueError('Unpaired targets or origins')
                loss=energy_score(f['samples'].transpose(1,2,0,3).astype(np.float64),target.astype(np.float64))
            losses.append(loss);sources.append({'seed':row['seed'],'sha256':sha256(path),'path':str(path)})
            print(fold,row['seed'],float(loss.mean()),flush=True)
        mean=np.mean(losses,axis=0);difference=mean-ewma
        result['folds'][fold]={'trefin_seed_energy_scores':[float(x.mean()) for x in losses],
                               'trefin_mean_energy_score':float(mean.mean()),'ewma_energy_score':float(ewma.mean()),
                               'paired_tests':[paired_bootstrap(difference,b,5000,20261108+i) for i,b in enumerate([5,10,20])],
                               'trefin_sources':sources,'ewma_sha256':sha256(ep)}
        arrays[f'{fold}_difference']=difference;arrays[f'{fold}_trefin_seed_losses']=np.array(losses)
        arrays[f'{fold}_ewma_losses']=ewma;arrays[f'{fold}_indices']=indices
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'energy_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'energy_losses.npz',**arrays)


if __name__=='__main__':main()
