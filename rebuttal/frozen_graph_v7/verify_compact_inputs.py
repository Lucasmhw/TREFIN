"""Check lossless compact states against all archived reference arrays."""
from pathlib import Path
import json,hashlib,lzma,pickle
import numpy as np
H=Path(__file__).resolve().parent;W=H.parents[1]
P=W/'work/server_snapshot/f-concord/runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
report={'status':'passed','folds':{},'purpose':'Lossless reconstruction of existing frozen measurement arrays; no fitting or selection.'}
for f in [1,2,3]:
    src=W/f'work/compact_prepared/fold_{f}.xz'
    with lzma.open(src,'rb') as stream:a=pickle.load(stream)
    n=len(a['y'])
    for key,initial in [('c_target','c0'),('a_target','a0')]:
        seq=a.pop(key+'_sequence');a[key]=np.stack([seq[h:h+n] for h in range(6)],axis=1);a[initial]=a[key][:,0].copy()
    ref=P/f'fold_{f}_seed_1/audit'
    m=json.loads(src.with_suffix('.json').read_text())
    assert m==json.loads((ref/'prepared_metadata.json').read_text())
    with np.load(ref/'prepared_arrays.npz',allow_pickle=False) as z:
        assert set(a)=={'origins','y','s0','directional_mask','risk_indices','c_target','c0','a_target','a0'}
        assert set(a).issubset(z.files)
        omitted=sorted(set(z.files)-set(a))
        for k in a:assert np.array_equal(a[k],z[k]),(f,k)
    entry={'compact_sha256':sha(src),'metadata_sha256':sha(src.with_suffix('.json')),
      'array_fields':{k:{'sha256':ah(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in a.items()},
      'split_counts':{k:len(v) for k,v in m['split_indices'].items()},'all_model_consumed_fields_equal_original':True,
      'omitted_diagnostic_fields_not_model_inputs':omitted}
    report['folds'][str(f)]=entry
    print('Fold',f,'verified',entry['split_counts'],flush=True)
    del a
(H/'input_equivalence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
