"""Recompute the completed decoder audit's hashes, compositions and summaries."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'n': int(values.size), 'minimum': float(values.min()),
            'median': float(np.median(values)), 'q95': float(np.quantile(values, .95)),
            'maximum': float(values.max())}


def compare(observed, expected):
    assert observed['n'] == expected['n']
    for key in ['minimum', 'median', 'q95', 'maximum']:
        np.testing.assert_allclose(observed[key], expected[key], rtol=2e-14, atol=1e-14)


def main():
    root = Path(__file__).resolve().parent
    directory = root/'decoder_results'
    state_directory = root/'local_tube_results_v2'
    protocol = json.loads((directory/'protocol.json').read_text())
    state_protocol = json.loads((state_directory/'protocol.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['protocol'] == protocol
    source_files = {'decoder_bound_source_sha256': root/'decoder_bounds.py',
                    'local_mlp_bound_source_sha256': root/'local_mlp_bounds.py',
                    'audit_source_sha256': root/'audit_decoder_bounds.py',
                    'source_model_sha256': Path(state_protocol['source_model']),
                    'state_protocol_sha256': state_directory/'protocol.json'}
    for key, path in source_files.items():
        assert protocol[key] == digest(path), (key, 'source changed')
    assert protocol['source_model_sha256'] == state_protocol['source_model_sha256']
    assert state_protocol['initial_fraction'] == .1
    fields = protocol['fields']
    ix = {name: i for i, name in enumerate(fields)}
    variants = protocol['variants']
    assert variants == ['ideal_spectral', 'implemented_spectral', 'ideal_local', 'implemented_local']
    all_arrays = {}
    scalar_checks = 0
    vectors = 0
    for record in summary['runs']:
        fold, seed = record['fold'], record['seed']
        run_name = f'fold_{fold}_seed_{seed}'
        state_path = state_directory/(run_name+'.json')
        state = json.loads(state_path.read_text())
        assert record['state_result_sha256'] == digest(state_path)
        assert record['checkpoint_sha256'] == state['checkpoint_sha256']
        assert record['arrays_sha256'] == digest(directory/record['arrays_file'])
        assert not record['gaussian_symmetric_decoder']
        with np.load(directory/record['arrays_file'], allow_pickle=False) as data:
            values = data['values']
            assert data['fields'].tolist() == fields
            assert data['variants'].tolist() == variants
            np.testing.assert_array_equal(data['radii'], state_protocol['concept_radii'])
            np.testing.assert_array_equal(data['test_positions'], [o['test_position'] for o in state['origins']])
        assert values.shape == (32, 4, 5, 4, len(fields))
        # This completed cohort has no inactive alternative or uncertified prefix.
        assert np.isfinite(values).all(), (run_name, 'nonfinite field')
        assert (values[..., ix['normalization_clamp_inactive']] == 1).all()
        assert (values[..., ix['precision_floor']] > 0).all()
        assert (values[..., ix['nu_lower']] > 2).all()
        np.testing.assert_allclose(values[..., ix['L_D_sum_block_norm']],
                                   values[..., :2].max(-1), rtol=0, atol=0)
        terms = sum(values[..., ix[a]:ix[a]+2]
                    for a in ['mean_C', 'scale_C', 'precision_C', 'nu_C'])
        np.testing.assert_allclose(values[..., :2], terms*(1+1e-10)+1e-12, rtol=1e-14, atol=1e-14)
        for oi, origin in enumerate(state['origins']):
            for ri, radius in enumerate(origin['radii']):
                assert all(radius['certified_prefixes'])
                initial = .1*np.array([radius['concept_radius'], radius['graph_radius']])
                for hi, product in enumerate(radius['matrix_products']):
                    delta = np.asarray(product)@initial
                    expected = values[oi, ri, hi, :, :2]@delta
                    np.testing.assert_allclose(values[oi, ri, hi, :, ix['propagated_W1_upper_bound']],
                                               expected, rtol=2e-14, atol=1e-14)
                    scalar_checks += len(variants)
        assert (values[..., 2, :3] <= values[..., 0, :3]*(1+1e-9)+1e-9).all()
        assert (values[..., 3, :3] <= values[..., 1, :3]*(1+1e-9)+1e-9).all()
        compare(record['primary_implemented_LD'], statistics(values[:, 2, :, 3, ix['L_D_sum_block_norm']]))
        compare(record['smallest_implemented_LD'], statistics(values[:, 0, :, 3, ix['L_D_sum_block_norm']]))
        vectors += int(np.prod(values.shape[:-1]))
        all_arrays.setdefault(fold, []).append(values)
    concise = {}
    for fold, arrays in all_arrays.items():
        assert len(arrays) == 5
        values = np.concatenate(arrays, axis=0)
        for ri, radius in enumerate(summary['folds'][str(fold)]['radii']):
            for vi, variant in enumerate(variants):
                stored = radius['variants'][variant]
                for fi, field in enumerate(fields):
                    compare(stored[field], statistics(values[:, ri, :, vi, fi]))
                for h in range(5):
                    compare(stored['propagated_W1_upper_bound_by_lead'][h],
                            statistics(values[:, ri, h, vi, ix['propagated_W1_upper_bound']]))
        concise[str(fold)] = {}
        for ri in [0, 2]:
            vv = summary['folds'][str(fold)]['radii'][ri]['variants']['implemented_local']
            concise[str(fold)][str(state_protocol['concept_radii'][ri])] = {
                'LD_all_leads': vv['L_D_sum_block_norm'],
                'lead5_propagated_W1': vv['propagated_W1_upper_bound_by_lead'][4]}
    compact = json.loads((root/'decoder_summary_compact.json').read_text())
    assert compact['source_summary_sha256'] == digest(directory/'summary.json')
    grouped = {fold: np.concatenate(arrays, axis=0) for fold, arrays in all_arrays.items()}
    for ri, row in enumerate(compact['radii']):
        pooled = []
        for fold, values in grouped.items():
            values = values[:, ri, :, 3, :]
            pooled.append(values)
            compare(row['folds'][str(fold)]['L_D'], statistics(values[..., ix['L_D_sum_block_norm']]))
            compare(row['folds'][str(fold)]['lead5_composed_W1_basis_points'],
                    statistics(values[:, 4, ix['propagated_W1_upper_bound']]*1e4))
        values = np.concatenate(pooled, axis=0)
        compare(row['all_folds']['L_D'], statistics(values[..., ix['L_D_sum_block_norm']]))
        compare(row['all_folds']['lead5_composed_W1_basis_points'],
                statistics(values[:, 4, ix['propagated_W1_upper_bound']]*1e4))
        compare(row['all_folds']['precision_fraction_in_A_bound'],
                statistics(values[..., ix['precision_A']]/values[..., ix['L_A']]))
    validation = json.loads((root/'decoder_validation'/'validation_summary.json').read_text())
    assert validation['passed'] and len(validation['runs']) == 15
    assert validation['script_sha256'] == digest(root/'validate_decoder_bounds.py')
    assert validation['decoder_source_sha256'] == protocol['decoder_bound_source_sha256']
    assert validation['local_mlp_source_sha256'] == protocol['local_mlp_bound_source_sha256']
    checkpoints = {(r['fold'], r['seed']): r['checkpoint_sha256'] for r in summary['runs']}
    maxima = {key: 0. for key in ['max_dense_ratio', 'max_head_difference_ratio', 'max_factor_difference_ratio',
                                  'max_covariance_factor_identity_error']}
    for record in validation['runs']:
        assert record['checkpoint_sha256'] == checkpoints[record['fold'], record['seed']]
        assert record['passed'] and record['normalization_clamp_inactive']
        assert [record[k] for k in ['dense_head_blocks', 'finite_head_pairs', 'factor_pairs']] == [192, 576, 96]
        for key in ['max_precision_lower_violation', 'max_precision_diagonal_upper_violation', 'max_nu_interval_violation']:
            assert record[key] == 0
        for key in maxima:
            maxima[key] = max(maxima[key], record[key])
    assert max(maxima[k] for k in ['max_dense_ratio', 'max_head_difference_ratio', 'max_factor_difference_ratio']) <= 1
    assert maxima['max_covariance_factor_identity_error'] < 1e-10
    assert validation['totals'] == {'dense_head_blocks': 2880, 'finite_head_pairs': 8640, 'factor_pairs': 1440}
    result = {'status': 'passed', 'runs': len(summary['runs']), 'decoder_bound_vectors': vectors,
              'propagated_compositions_recomputed': scalar_checks,
              'all_fields_finite': True, 'all_normalization_clamps_inactive': True,
              'all_summary_fields_recomputed': True,
              'compact_summary_recomputed': True,
              'actual_checkpoint_validation_verified': True,
              'validation_totals': validation['totals'],
              'validation_maxima': maxima,
              'dense_check_count_scope': '720 unique C/A head-Jacobian blocks checked against four radii, giving 2880 block-bound checks; 60 joint head-Jacobian evaluations.',
              'source_hashes_verified': {key: protocol[key] for key in source_files},
              'summary_sha256': digest(directory/'summary.json'),
              'selected_descriptive_results': concise,
              'scope': 'Independent source and complete-cohort consistency audit; not a reference-law accuracy claim.'}
    (root/'independent_decoder_result_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'selected_descriptive_results'}, indent=2))


if __name__ == '__main__':
    main()
