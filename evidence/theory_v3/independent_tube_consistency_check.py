"""Independent arithmetic/provenance checks of the completed analytic audit."""
import hashlib
import json
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parent
results=base/'local_tube_results_v2'
protocol=json.loads((results/'protocol.json').read_text())
aggregate=json.loads((results/'summary.json').read_text())
validation=json.loads((base/'validation/validation_summary.json').read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(base/'tube_bounds.py')==protocol['bound_script_sha256']==validation['bound_script_sha256']
assert sha(base/'audit_local_tubes.py')==protocol['audit_script_sha256']
assert sha(base/'validate_tube_bounds.py')==validation['script_sha256']
assert protocol['folds']==[1,2,3] and protocol['seeds']==[1,7,19,37,73]
assert protocol['concept_radii']==[.001,.01,.1,1.] and protocol['initial_fraction']==.1
records=[]
for fold in protocol['folds']:
    for seed in protocol['seeds']:
        records.append(json.loads((results/f'fold_{fold}_seed_{seed}.json').read_text()))
assert len(records)==15
validated={(r['fold'],r['seed']):r for r in validation['runs']}
matrix_steps=0;tubes=0;prefix_certified=0;one_step_noncontraction=0
for r in records:
    v=validated[r['fold'],r['seed']]
    assert v['checkpoint_sha256']==r['checkpoint_sha256'] and v['passed']
    assert len(r['origins'])==32
    assert all(v[key]<=1e-10 for key in ['max_box_violation','max_graph_ball_violation','max_asymmetry','max_diagonal'])
    for origin in r['origins']:
        assert [x['concept_radius'] for x in origin['radii']]==protocol['concept_radii']
        for radius in origin['radii']:
            tubes+=1
            rv=np.array([radius['concept_radius'],radius['graph_radius']])
            product=np.eye(2);gains=[]
            for h,m in enumerate(radius['block_matrices']):
                matrix_steps+=1;m=np.asarray(m)
                assert np.isfinite(m).all() and (m>=0).all()
                product=m@product
                assert np.allclose(product,radius['matrix_products'][h],rtol=0,atol=1e-12)
                gain=float(np.max(product@rv/rv));gains.append(gain)
                assert abs(gain-radius['weighted_h_step_bounds'][h])<1e-12
                q=float(np.max(m@rv/rv))
                assert abs(q-radius['weighted_one_step_bounds'][h])<1e-12
                one_step_noncontraction+=q>1
                passed=max(gains)<=10
                assert bool(radius['certified_prefixes'][h])==passed
                prefix_certified+=passed
                if not passed: assert radius['mean_difference_l2_bounds_from_initial_fraction'][h] is None
            assert bool(radius['all_five_steps_certified'])==(max(gains)<=10)
            assert bool(radius['full_radius_five_step_tube_certified'])==(max(gains)<=1)
for fold in [1,2,3]:
    for ri in range(4):
        entries=[origin['radii'][ri] for r in records if r['fold']==fold for origin in r['origins']]
        target=aggregate['folds'][str(fold)]['radii'][ri]
        assert target['trajectories']==len(entries)==160
        assert target['certified_five_step_count']==sum(e['all_five_steps_certified'] for e in entries)
        for h in range(5):
            values=np.array([e['weighted_h_step_bounds'][h] for e in entries])
            s=target['h_step_amplification_bounds'][h]
            assert abs(float(values.max())-s['maximum'])<1e-12
            assert abs(float(np.median(values))-s['median'])<1e-12
assert validation['passed'] and validation['totals']['runs']==15
assert validation['totals']['centers']==2880
assert validation['totals']['dense_jacobians']==75
assert validation['totals']['finite_perturbation_leads']==3600
assert validation['maximum_dense_block_ratio']<=1
assert validation['maximum_finite_state_ratio']<=1
assert validation['maximum_finite_mean_ratio']<=1
result={'status':'passed','independent_review':'Analytic derivation and implementation read; stored product arithmetic, source hashes, cohort completeness, and validation summaries independently cross-checked.',
        'runs':len(records),'neighborhoods':tubes,'matrix_steps':matrix_steps,'certified_prefixes':prefix_certified,
        'one_step_bounds_above_one':one_step_noncontraction,
        'primary_five_step_bound':{str(f):aggregate['folds'][str(f)]['radii'][2]['h_step_amplification_bounds'][4] for f in [1,2,3]},
        'validation_totals':validation['totals'],
        'max_validation_ratios':{k:validation[k] for k in ['maximum_dense_block_ratio','maximum_finite_state_ratio','maximum_finite_mean_ratio']},
        'source_hashes':{str(p):sha(p) for p in [base/'tube_bounds.py',base/'tube_derivation.md',base/'audit_local_tubes.py',base/'validate_tube_bounds.py',results/'summary.json',base/'validation/validation_summary.json']},
        'remaining_mathematical_issue':None,
        'interpretation':'Uniform analytic bounds on declared neighborhoods, evaluated in float64; bounded finite-horizon amplification, not universal contraction, unknown-dynamics accuracy, or a numerical distributional W1 certificate.'}
(base/'independent_review_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','primary_five_step_bound']},ensure_ascii=False))
