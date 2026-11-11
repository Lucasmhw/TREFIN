"""Restore only verified model arrays in the isolated study input directory."""
from pathlib import Path
import hashlib,json,lzma,pickle,shutil
import numpy as np
H=Path(__file__).resolve().parent
src=H/'inputs/compact';manifest=json.loads((H/'input_equivalence.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
result={'status':'passed','folds':{}}
for f in [1,2,3]:
    x=src/f'fold_{f}.xz';spec=manifest['folds'][str(f)]
    assert sha(x)==spec['compact_sha256']
    assert sha(x.with_suffix('.json'))==spec['metadata_sha256']
    with lzma.open(x,'rb') as stream:a=pickle.load(stream)
    n=len(a['y'])
    for key,initial in [('c_target','c0'),('a_target','a0')]:
        seq=a.pop(key+'_sequence');a[key]=np.stack([seq[h:h+n] for h in range(6)],axis=1);a[initial]=a[key][:,0].copy()
    assert set(a)==set(spec['array_fields'])
    for k,v in a.items():
        s=spec['array_fields'][k];assert list(v.shape)==s['shape'] and str(v.dtype)==s['dtype'] and ah(v)==s['sha256'],(f,k)
    dest=H/f'prepared/fold_{f}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'prepared_arrays.npz').exists()
    np.savez_compressed(dest/'prepared_arrays.npz',**a)
    shutil.copy2(x.with_suffix('.json'),dest/'prepared_metadata.json')
    result['folds'][str(f)]={'arrays_sha256':sha(dest/'prepared_arrays.npz'),'metadata_sha256':sha(dest/'prepared_metadata.json'),'model_array_hashes_verified':True,'split_counts':spec['split_counts']}
    print('Fold',f,'restored',spec['split_counts'],flush=True)
    del a
(H/'prepared_verification.json').write_text(json.dumps(result,indent=2)+'\n')
