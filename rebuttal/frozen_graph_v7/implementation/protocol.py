"""FrozenGraph v7 configuration and leakage guards; no fitting on import."""
from pathlib import Path
import copy, hashlib, json

VARIANTS = ['Full', 'FrozenGraph']
SEEDS = [1, 7, 19, 37, 73]
CANDIDATES = [
    {'id': 'base', 'model': {}, 'training': {}},
    {'id': 'slow_graph', 'model': {'graph_refresh': .02}, 'training': {}},
    {'id': 'stronger_dynamics', 'model': {'residual_scale': .05, 'graph_refresh': .10}, 'training': {}},
    {'id': 'weaker_state_supervision', 'model': {}, 'training': {'lambda_concept': .10, 'lambda_graph': .02}},
]
PROTOCOL_DIRECTORY = 'runs/protocols/finmultitime_us_calibrated_rolling_3fold_5seed_v1/runs'

def reference_config(root, fold, seed=1):
    return Path(root)/PROTOCOL_DIRECTORY/f'fold_{fold}_seed_{seed}'/'config_resolved.yaml'

def prepared_hashes(prepared_root, fold):
    directory=Path(prepared_root)/f'fold_{fold}'
    return {name:sha256(directory/name) for name in ['prepared_arrays.npz','prepared_metadata.json']}

def implementation_hashes():
    directory=Path(__file__).resolve().parent
    return {name:sha256(directory/name) for name in ['model.py','protocol.py','search_validation.py','freeze_selections.py','run_selected.py']}

def selection_key(fold, variant):
    return f'fold_{fold}/{variant}'

def validate_frozen(study_root):
    """Check all six durable choices and their selected checkpoint hashes."""
    root=Path(study_root)
    path=root/'frozen_selections.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    assert record['schema']=='trefin-frozen-graph-v7-selections' and record['status']=='frozen_before_test'
    assert set(record['selections'])=={selection_key(f,v) for f in [1,2,3] for v in VARIANTS}
    assert record['implementation_sha256']==implementation_hashes(),'Scientific runner source changed after freeze'
    for key,item in record['selections'].items():
        assert sha256(root/item['selection_path'])==item['selection_sha256'],f'Choice changed: {key}'
        assert sha256(root/item['registration_path'])==item['registration_sha256'],f'Registration changed: {key}'
        selection=json.loads((root/item['selection_path']).read_text(encoding='utf-8'))
        assert sha256(root/selection['selected_checkpoint'])==selection['selected_checkpoint_sha256'],f'Checkpoint changed: {key}'
    return record,sha256(path)

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):digest.update(chunk)
    return digest.hexdigest()

def dump(value, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def configure(base, variant, candidate, *, fold, seed, device='cuda'):
    if variant not in VARIANTS:raise ValueError(variant)
    cfg=copy.deepcopy(base)
    cfg['model'].update(candidate['model'])
    cfg['training'].update(candidate['training'])
    cfg['model'].update(ablation_variant=variant,signal_shrinkage=1.0)
    cfg['training'].update(epochs=20,patience=6,batch_size=64,selection_metric='energy_score',
      selection_min_delta=1e-6,signal_shrinkage_calibration=True,
      signal_shrinkage_lower=.05,signal_shrinkage_upper=1.0,signal_shrinkage_tolerance=1e-5,
      train_samples=16,validation_samples=64,test_samples=256,eval_samples=64)
    if variant=='FrozenGraph':cfg['training']['lambda_graph']=0.0
    cfg['experiment'].update(name=f'frozen_graph_v7_fold_{fold}_seed_{seed}_{variant}_{candidate["id"]}',
      seed=seed,device=device,deterministic=True)
    assert cfg['training']['lambda_gain']==.1 and cfg['training']['gain_target']==.995
    return cfg

def effective_fingerprint(cfg):
    effective={k:copy.deepcopy(cfg[k]) for k in ['model','training']}
    if effective['model']['ablation_variant']=='FrozenGraph':
        effective['model'].pop('graph_refresh',None)
    return hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()

def train_validation_only(prepared):
    """Physically exclude every test-origin row, preserving the original split."""
    import numpy as np
    from fconcord.data import FinancialTensorDataset
    metadata=copy.deepcopy(prepared['metadata'])
    train=np.asarray(metadata['split_indices']['train'],dtype=np.int64)
    val=np.asarray(metadata['split_indices']['val'],dtype=np.int64)
    test=np.asarray(metadata['split_indices'].get('test',[]),dtype=np.int64)
    keep=np.sort(np.concatenate([train,val]))
    assert len(np.unique(keep))==len(keep) and not np.intersect1d(keep,test).size
    origin_keys={'c0','a0','s0','c_target','a_target','y','origins'}
    arrays={k:prepared['arrays'][k][keep].copy() for k in origin_keys}
    arrays.update({k:prepared['arrays'][k].copy() for k in ['directional_mask','risk_indices']})
    metadata['split_indices']={'train':np.searchsorted(keep,train).tolist(),'val':np.searchsorted(keep,val).tolist()}
    metadata['num_samples']=len(keep)
    for key in ['split_date_ranges','split_boundaries','purged_indices']:
        if isinstance(metadata.get(key),dict):metadata[key]={k:v for k,v in metadata[key].items() if k!='test'}
    metadata['search_access_scope']='train_and_validation_only'
    limited={'arrays':arrays,'metadata':metadata}
    try:FinancialTensorDataset(limited,'test')
    except KeyError:pass
    else:raise AssertionError('Test dataset construction must be impossible')
    return limited,{'retained_original_indices':keep.tolist(),'train_count':len(train),'validation_count':len(val),
      'excluded_test_origin_count':len(test),'test_split_key_removed':True,
      'test_dataset_construction_raises_KeyError':True,'origin_arrays_physically_subset':True}

def runtime_sources():
    """Require the isolated installed model to be the exact registered patch."""
    import fconcord.model as model
    directory=Path(model.__file__).resolve().parent
    expected=Path(__file__).resolve().with_name('model.py')
    assert sha256(directory/'model.py')==sha256(expected),'PYTHONPATH must use the isolated patched source tree'
    return {name:sha256(directory/name) for name in ['model.py','trainer.py','losses.py','calibration.py','metrics.py','experiment.py','utils.py']}
