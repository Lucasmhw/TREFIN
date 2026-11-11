"""Independent complete-cohort verification of the proposed flow tightening."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches_recorded_metadata_hash(path, expected):
    """Accept exact bytes or an explicitly recorded path-only public derivative."""
    path = Path(path).resolve()
    actual = sha(path)
    if actual == expected:
        return True
    for repository in path.parents:
        manifest = repository/'docs/BOOTSTRAP_PROVENANCE.json'
        if not manifest.is_file():
            continue
        public = read(manifest).get('public_anonymization', {})
        if public.get('schema') != 'trefin-scientific-path-anonymization-v1':
            return False
        relative = path.relative_to(repository).as_posix()
        records = [item for item in public.get('records', []) if item.get('path') == relative]
        return (len(records) == 1 and records[0].get('original_sha256') == expected
                and records[0].get('public_sha256') == actual
                and records[0].get('public_bytes') == path.stat().st_size)
    return False


def stats(values):
    values = np.asarray(values)
    assert np.isfinite(values).all()
    return dict(n=int(values.size), minimum=float(values.min()), median=float(np.median(values)),
                q95=float(np.quantile(values, .95)), maximum=float(values.max()))


def compare(stored, expected):
    assert stored['n'] == expected['n']
    for key in ['minimum', 'median', 'q95', 'maximum']:
        np.testing.assert_allclose(stored[key], expected[key], rtol=2e-14, atol=1e-12)


def main():
    p=argparse.ArgumentParser();p.add_argument('--theory-root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--data-root',type=Path);p.add_argument('--model-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    here = args.data_root.resolve() if args.data_root is not None else args.theory_root.resolve()/'exploration'
    root = here.parent
    directory = here/'all_radius_flow_results'
    spec = read(directory/'protocol.json')
    summary = read(directory/'summary.json')
    original_spec = read(root/'decoder_results/protocol.json')
    fields = original_spec['fields']
    ix = {name: i for i, name in enumerate(fields)}
    assert summary['protocol'] == spec
    assert spec['radii_indices'] == [0, 1, 2, 3] and spec['radius_values'] == [.001, .01, .1, 1.]
    assert spec['source_sha256'] == sha(here/'normalized_factor_flow.py')
    assert spec['runner_sha256'] == sha(here/'audit_factor_flow.py')
    assert spec['base_protocol_sha256'] == sha(root/'decoder_results/protocol.json')
    assert original_spec['source_model_sha256'] == sha(args.model_source)
    assert matches_recorded_metadata_hash(root/'local_tube_results_v2/protocol.json', original_spec['state_protocol_sha256'])
    original_review = read(root/'independent_decoder_result_check.json')
    assert original_review['status'] == 'passed'
    assert original_review['summary_sha256'] == sha(root/'decoder_results/summary.json')
    for field, filename in [('decoder_bound_source_sha256', 'decoder_bounds.py'),
                            ('local_mlp_bound_source_sha256', 'local_mlp_bounds.py')]:
        assert original_spec[field] == sha(root/filename)
    grouped = {}
    checkpoint_hashes = {}
    cases = 0
    max_ratio = 0.
    for fold in [1, 2, 3]:
        grouped[fold] = []
        for seed in [1, 7, 19, 37, 73]:
            stem = f'fold_{fold}_seed_{seed}'
            metadata = read(directory/(stem+'.json'))
            original_metadata = read(root/'decoder_results'/(stem+'.json'))
            state = read(root/'local_tube_results_v2'/(stem+'.json'))
            assert metadata['checkpoint_sha256'] == original_metadata['checkpoint_sha256'] == state['checkpoint_sha256']
            checkpoint_hashes[fold, seed] = metadata['checkpoint_sha256']
            assert metadata['npz_sha256'] == sha(directory/(stem+'.npz'))
            assert metadata['base_npz_sha256'] == sha(root/'decoder_results'/(stem+'.npz'))
            assert original_metadata['state_result_sha256'] == sha(root/'local_tube_results_v2'/(stem+'.json'))
            with np.load(directory/(stem+'.npz'), allow_pickle=False) as archive:
                values = archive['values']
                names = archive['fields'].tolist()
            assert names == ['old_L_C', 'old_L_A', 'new_L_C', 'new_L_A', 'old_precision_C', 'old_precision_A',
                             'flow_precision_C', 'flow_precision_A', 'rmax', 'graph_relative', 'precision_floor',
                             'new_W1', 'old_W1', 'q_C', 'q_A', 'lambda_A']
            assert values.shape == (32, 4, 5, 16) and np.isfinite(values).all()
            assert (values[..., 8] >= 1-1e-10).all() and (values[..., 8] <= 32+1e-10).all()
            assert (values[..., 9:11] > 0).all()
            with np.load(root/'decoder_results'/(stem+'.npz'), allow_pickle=False) as archive:
                old = archive['values'][:, :, :, 3]
            np.testing.assert_array_equal(values[..., :2], old[..., :2])
            old_precision = old[..., [ix['precision_C'], ix['precision_A']]]
            np.testing.assert_array_equal(values[..., 4:6], old_precision)
            rest = sum(old[..., [ix[key+'_C'], ix[key+'_A']]] for key in ['mean', 'scale', 'nu'])
            expected = (rest+np.minimum(old_precision, values[..., 6:8]))*(1+1e-10)+1e-12
            np.testing.assert_allclose(values[..., 2:4], expected, rtol=2e-14, atol=1e-12)
            assert (values[..., 2:4] <= values[..., :2]*(1+1e-10)+1e-10).all()
            max_ratio = max(max_ratio, float((values[..., 2:4]/values[..., :2]).max()))
            for oi, origin in enumerate(state['origins']):
                for ri, radius in enumerate(origin['radii']):
                    assert all(radius['certified_prefixes'])
                    initial = .1*np.array([radius['concept_radius'], radius['graph_radius']])
                    for h, product in enumerate(radius['matrix_products']):
                        delta = np.asarray(product)@initial
                        np.testing.assert_allclose(values[oi, ri, h, 11:13],
                                                   [values[oi, ri, h, 2:4]@delta, values[oi, ri, h, :2]@delta],
                                                   rtol=2e-14, atol=1e-12)
                        cases += 1
            grouped[fold].append(values)
    primary = {}
    for fold, arrays in grouped.items():
        values = np.concatenate(arrays, axis=0)
        for ri, entry in enumerate(summary['folds'][str(fold)]):
            assert entry['concept_radius'] == spec['radius_values'][ri]
            old = values[:, ri, :, :2].max(-1)
            new = values[:, ri, :, 2:4].max(-1)
            compare(entry['old_LD'], stats(old))
            compare(entry['new_LD'], stats(new))
            compare(entry['new_over_old'], stats(new/old))
            compare(entry['new_lead5_W1_bp'], stats(values[:, ri, 4, 11]*1e4))
            compare(entry['old_lead5_W1_bp'], stats(values[:, ri, 4, 12]*1e4))
            if ri == 2:
                primary[str(fold)] = entry
    validation = read(here/'validation/summary.json')
    assert validation['passed'] and len(validation['runs']) == 15
    assert validation['flow_source_sha256'] == spec['source_sha256']
    assert validation['validator_sha256'] == sha(here/'validate_factor_flow.py')
    totals = {key: 0 for key in validation['totals']}
    maxima = {key: 0. for key in validation['maxima']}
    for run in validation['runs']:
        assert run['passed'] and run['checkpoint_sha256'] == checkpoint_hashes[run['fold'], run['seed']]
        for key in totals:
            totals[key] += run[key]
        for key in maxima:
            assert np.isfinite(run[key]) and run[key] <= 1+1e-9
            maxima[key] = max(maxima[key], run[key])
    assert totals == validation['totals']
    assert totals['dense_log_head_block_checks'] == 960
    assert totals['finite_gaussian_pairs'] == 4320 and totals['finite_log_head_checks'] == 8640
    assert maxima == validation['maxima']
    parity = read(here/'portable_runner_parity.json')
    assert parity['passed'] and parity['validation_json_exact_equality']
    assert all(parity['npz_array_exact_equality'].values())
    for name, digest in parity['source_hashes'].items():
        assert sha(here/name) == digest
    assert parity['original_audit_arrays_sha256'] == parity['portable_audit_arrays_sha256'] == sha(directory/'fold_1_seed_1.npz')
    result = {'status': 'passed', 'runs': 15, 'same_domain_readout_neighborhoods': cases,
              'all_four_prespecified_radii_retained': True, 'per_cell_minima_and_state_compositions_recomputed': True,
              'all_summary_statistics_recomputed': True, 'maximum_new_to_old_block_ratio': max_ratio,
              'validation_totals': totals, 'validation_maxima': maxima,
              'dense_count_scope': '240 unique log-head C/A Jacobian blocks checked at four radii, giving 960 checks from 60 joint Jacobian evaluations.',
              'portable_runner_parity_verified': True, 'flow_source_sha256': spec['source_sha256'],
              'restored_model_source_sha256_verified': original_spec['source_model_sha256'],
              'summary_sha256': sha(directory/'summary.json'), 'validation_sha256': sha(here/'validation/summary.json'),
              'primary_results': primary,
              'scope': 'Uniform analytic tightening on the identical domains and complete cohort; no true-law decoder defect is estimated.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'primary_results'}, indent=2))


if __name__ == '__main__':
    main()
