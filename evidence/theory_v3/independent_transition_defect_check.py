"""Independent unrolled-product verification of conditional defect budgets."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    state_directory = root/'local_tube_results_v2'
    result_path = root/'transition_defect_results'/'transition_defect_budgets.json'
    result = json.loads(result_path.read_text())
    assert result['source_state_summary_sha256'] == digest(state_directory/'summary.json')
    assert result['source_script_sha256'] == digest(root/'audit_transition_defect_budgets.py')
    for record in result['source_files']:
        assert record['sha256'] == digest(state_directory/record['file'])
    alpha, beta = result['initial_fraction'], result['fixed_defect_fraction_checked']
    assert alpha == .1 and beta == .05
    states = {}
    for record in result['source_files']:
        source = json.loads((state_directory/record['file']).read_text())
        for origin in source['origins']:
            for radius in origin['radii']:
                states[source['fold'], source['seed'], origin['test_position'], radius['concept_radius']] = radius
    assert len(states) == len(result['records']) == 1920
    checked = []
    for record in result['records']:
        key = record['fold'], record['seed'], record['test_position'], record['concept_radius']
        source = states[key]
        matrices = np.asarray(source['block_matrices'])
        radii = np.array([source['concept_radius'], source['graph_radius']])
        ratios, thresholds = [], []
        # Explicitly sum every tail product instead of using the generator's
        # S_(h+1)=M_h S_h+I recurrence.
        for h in range(1, 6):
            product = np.eye(2)
            for j in range(h):
                product = matrices[j]@product
            summed = np.zeros((2, 2))
            for defect_time in range(h):
                tail = np.eye(2)
                for j in range(defect_time+1, h):
                    tail = matrices[j]@tail
                summed += tail
            baseline = alpha*(product@radii)
            sensitivity = summed@radii
            ratio = (baseline+beta*sensitivity)/radii
            threshold = (radii-baseline)/sensitivity
            ratios.append(ratio)
            thresholds.append(threshold)
        ratios, thresholds = np.asarray(ratios), np.asarray(thresholds)
        assert np.all(thresholds > 0)
        np.testing.assert_allclose(record['fixed_beta_prefix_normalized_error_bounds'], ratios, rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(record['prefix_block_defect_fraction_thresholds'], thresholds, rtol=2e-14, atol=1e-14)
        threshold = float(thresholds.min())
        conservative = max(0., threshold*(1-1e-10)-1e-12)
        np.testing.assert_allclose(record['uniform_defect_fraction_threshold'], threshold, rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(record['conservative_uniform_defect_fraction'], conservative, rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(record['fixed_beta_maximum_normalized_state_error'], ratios.max(), rtol=2e-14, atol=1e-14)
        assert record['fixed_beta_all_prefixes_contained'] == bool((ratios <= 1).all())
        assert ratios.max() <= 1
        checked.append((record['fold'], record['concept_radius'], conservative, float(ratios.max())))
    primary = {}
    for fold, groups in result['folds'].items():
        for group in groups:
            rows = [r for r in checked if r[0] == int(fold) and r[1] == group['concept_radius']]
            thresholds = np.array([r[2] for r in rows])
            assert len(rows) == group['trajectories'] == group['fixed_beta_certified_count'] == 160
            expected = {'n': 160, 'minimum': float(thresholds.min()),
                        'median': float(np.median(thresholds)), 'maximum': float(thresholds.max())}
            for key, value in expected.items():
                np.testing.assert_allclose(group['admissible_fraction'][key], value, rtol=2e-14, atol=1e-14)
            assert group['common_admissible_fraction_rounded_down_6dp'] == math.floor(thresholds.min()*1e6)/1e6
            np.testing.assert_allclose(group['fixed_beta_maximum_normalized_state_error'], max(r[3] for r in rows), rtol=2e-14, atol=1e-14)
            if group['concept_radius'] == .1:
                primary[fold] = group
    output = {'status': 'passed', 'trajectory_radius_cases': len(checked),
              'all_five_prefixes_checked': True, 'all_beta_005_cases_contained': True,
              'verification': 'Independent explicit sum of time-ordered tail products; all source hashes and summaries checked.',
              'source_result_sha256': digest(result_path), 'primary_results': primary,
              'scope': 'Conditional on an admissible comparison trajectory (or projected perturbation updates) with the stated initial and transition-residual budgets. Actual reference/true transition defects are not estimated.'}
    (root/'independent_transition_defect_check.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
