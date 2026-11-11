"""Reconstruct exact target windows from losslessly compacted causal sequences."""
from pathlib import Path
import numpy as np, lzma, pickle, json, shutil
root=Path(__file__).resolve().parents[1]
for src in (root/'compact_prepared').glob('fold_*.xz'):
    fold=int(src.stem.split('_')[1])
    with lzma.open(src,'rb') as f:a=pickle.load(f)
    n=len(a['y'])
    for key,initial in [('c_target','c0'),('a_target','a0')]:
        seq=a.pop(key+'_sequence')
        a[key]=np.stack([seq[h:h+n] for h in range(6)],axis=1)
        a[initial]=a[key][:,0].copy()
    dest=root/f'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs/fold_{fold}_seed_1/audit'
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'prepared_arrays.npz',**a)
    shutil.copy2(src.with_suffix('.json'),dest/'prepared_metadata.json')
    print(json.dumps({'fold':fold,'samples':n,'keys':list(a)}),flush=True)
