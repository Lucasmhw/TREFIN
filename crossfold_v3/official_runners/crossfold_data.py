"""Strict manifest-driven loader; numerical operations match frozen fold 1."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    fold = int(manifest['fold'])
    assert fold in (1, 2, 3)
    name = manifest['data_file']
    assert name == f'fold{fold}_raw_returns.npz'
    data_path = bundle/name
    assert sha256(data_path) == manifest['data_sha256']
    with np.load(data_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    norm = (data['returns']-data['location'])/data['scale']
    data['history'] = np.stack([norm[o-125:o+1] for o in data['origins']]).astype(np.float32)
    data['target_scaled'] = ((data['targets']-data['location'][None,None])/data['scale'][None,None]).astype(np.float32)
    for split in ['train', 'val', 'test']:
        positions = data[f'{split}_positions']
        assert np.all(np.diff(positions) > 0)
        assert len(positions) == manifest['split_sizes'][split]
        assert positions.min() >= 0 and positions.max() < len(data['origins'])
    assert data['origins'][data['train_positions'][-1]]+5 < data['origins'][data['val_positions'][0]]
    assert data['origins'][data['val_positions'][-1]]+5 < data['origins'][data['test_positions'][0]]
    assert data['history'].shape[1:] == (126,32)
    assert data['target_scaled'].shape[1:] == (5,32)
    assert np.isfinite(data['history']).all() and np.isfinite(data['target_scaled']).all()
    return data, manifest


def verify_registered_data(manifest, sources, bundle):
    expected = sources['data_references'][str(manifest['fold'])]
    assert manifest['data_sha256'] == expected['data_sha256'], 'Data differs from registered fold'
    assert manifest['data_file'] == expected['data_file']
    assert manifest['split_sizes'] == expected['split_sizes']
    assert sha256(Path(bundle)/'manifest.json') == expected['manifest_sha256']


def verify_official_upstreams(bundle, manifest):
    for name, record in manifest['upstreams'].items():
        for relative, expected in record['files'].items():
            assert sha256(Path(bundle)/'upstream'/name/relative) == expected, (name, relative)


def enforce_frozen_arguments(args):
    assert [float(s) for s in args.learning_rates.split(',')] == [.001, .0003]
    assert args.max_epochs == 100 and args.patience == 15 and args.batch_size == 64
    seeds = [int(s) for s in args.seeds.split(',')]
    assert seeds and seeds == [s for s in [1,7,19,37,73] if s in seeds]
    if args.family == 'tedm':
        assert args.origin_batch == 64 and args.track_chunk == 64


def protocol_metadata(manifest):
    return {'fold': manifest['fold'], 'data_file': manifest['data_file'],
            'split_sizes': manifest['split_sizes'],
            'split_date_ranges': manifest['split_date_ranges'],
            'normalization_fit_rows_inclusive': manifest['normalization_fit_rows_inclusive'],
            'crossfold_loader_sha256': sha256(__file__),
            'test_targets_float32_bytes_sha256': manifest['test_targets_float32_bytes_sha256']}
