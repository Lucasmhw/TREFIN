"""Read-only independent audit checks; does not train or change adapter sources."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
bundle = root / 'work/official_supplement_bundle'
manifest = json.loads((bundle / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(bundle / 'fold1_raw_returns.npz') == manifest['data_sha256']
source_count = 0
for name, spec in manifest['upstreams'].items():
    for rel, expected in spec['files'].items():
        assert sha(bundle / 'upstream' / name / rel) == expected, (name, rel)
        source_count += 1
adapter = root / 'work/scripts/fresh_official_baselines.py'
assert sha(adapter) == sha(bundle / 'fresh_official_baselines.py')
spec = importlib.util.spec_from_file_location('audited_adapter', adapter)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
data, _ = mod.load_data(bundle)
first, last = manifest['normalization_fit_rows_inclusive']
fit = data['returns'][first:last+1]
assert np.array_equal(fit.mean(0), data['location'])
assert np.array_equal(fit.std(0), data['scale'])
reconstructed = np.stack([data['returns'][o+1:o+6] for o in data['origins']]).astype(np.float32)
assert np.array_equal(reconstructed, data['targets'])
assert data['origins'].min() >= 125
assert np.isfinite(data['history']).all() and np.isfinite(data['target_scaled']).all()
assert set(data['train_positions']).isdisjoint(data['val_positions'])
assert set(data['val_positions']).isdisjoint(data['test_positions'])
assert set(data['train_positions']).isdisjoint(data['test_positions'])

# Temporal-coordinate test: if native sample evolves every coordinate one step,
# final-five extraction must return t+1,...,t+5, never historical or shifted labels.
class UnitTimeAdvance:
    past_k = 13
    padding = 1
    horizon = 5
    clamp = False
    def __init__(self):
        self.conditions = []
    def sample(self, current, condition):
        self.conditions.append(condition.clone())
        return current[..., 1:] + 1, None

history = torch.arange(-125., 1.).reshape(1,126,1).repeat(2,1,3)
history[1] += 1000
fake = UnitTimeAdvance()
pred = mod.tedm_rollout(fake, history, tracks=3)
for s in range(3):
    for b in range(2):
        assert torch.equal(pred[s,b,:,0], torch.arange(1.,6.) + 1000*b)
for step, condition in enumerate(fake.conditions):
    assert torch.equal(condition[0,0], torch.arange(-125.+step,-13.+step))

# Direct independently written energy-score expression, nonzero random arrays.
rng = np.random.default_rng(43)
x = rng.normal(size=(7,4,5,3)).astype(np.float64)
y = rng.normal(size=(4,5,3)).astype(np.float64)
pair = sum(np.linalg.norm(x[s]-x[t],axis=-1) for s in range(7) for t in range(7) if s != t)
reference = (np.linalg.norm(x-y[None],axis=-1).mean(0) - pair/(2*7*6)).mean(1)
actual = mod.fast_es(x,y,torch.device('cpu'))
assert np.max(np.abs(actual-reference)) < 1e-12

report = {
    'status':'passed',
    'adapter_sha256':sha(adapter),
    'upstream_source_hashes_verified':source_count,
    'data_sha256':sha(bundle/'fold1_raw_returns.npz'),
    'normalization_recomputed_exactly':True,
    'target_alignment_float32_exact':True,
    'split_membership_disjoint':True,
    'train_label_end_row':int(data['origins'][data['train_positions'][-1]]+5),
    'first_validation_origin_row':int(data['origins'][data['val_positions'][0]]),
    'validation_label_end_row':int(data['origins'][data['val_positions'][-1]]+5),
    'first_test_origin_row':int(data['origins'][data['test_positions'][0]]),
    'native_rollout_coordinate_mapping_test':'t+1 through t+5 passed; all tracks/origins kept separate',
    'energy_score_max_error':float(np.max(np.abs(actual-reference))),
    'caveat':'Coordinate test establishes task alignment under native one-step evolution, not forecast accuracy.'
}
out = root / 'work/fresh_adapter_independent_checks.json'
out.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
