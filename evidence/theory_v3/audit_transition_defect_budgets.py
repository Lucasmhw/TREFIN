"""Finite-horizon containment with explicit bounded additive transition defects.

This evaluates stored analytic matrices only, for an admissible comparison
trajectory (or projected perturbed updates). It does not estimate actual
reference/true transition defects, fit a model, or inspect prediction outcomes.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def stats(x):
    x=np.asarray(x,dtype=float)
    return {'n':len(x),'minimum':float(x.min()),'median':float(np.median(x)),'maximum':float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();state_summary=read(a.state_results/'summary.json')
    assert state_summary['runs']==15
    alpha=.1;beta=.05;rows=[];sources=[]
    for file in sorted(a.state_results.glob('fold_*_seed_*.json')):
        r=read(file);sources.append({'file':file.name,'sha256':sha(file)})
        for origin in r['origins']:
            for radius in origin['radii']:
                assert radius['initial_fraction']==alpha and radius['all_five_steps_certified']
                R=np.array([radius['concept_radius'],radius['graph_radius']],dtype=float)
                matrices=np.asarray(radius['block_matrices'],dtype=float)
                assert matrices.shape==(5,2,2) and (matrices>=0).all()
                P=np.eye(2);S=np.zeros((2,2));limits=[];ratios=[];direct=alpha*R
                for h,M in enumerate(matrices):
                    P=M@P;S=M@S+np.eye(2)
                    assert np.allclose(P,radius['matrix_products'][h],rtol=1e-12,atol=1e-14)
                    margin=R-alpha*(P@R);sensitivity=S@R
                    assert (margin>0).all() and (sensitivity>0).all()
                    limits.append(margin/sensitivity)
                    bounded=alpha*(P@R)+beta*(S@R)
                    direct=M@direct+beta*R
                    assert np.allclose(bounded,direct,rtol=1e-12,atol=1e-14)
                    ratios.append(bounded/R)
                admissible=float(np.min(limits));conservative=max(0.,admissible*(1-1e-10)-1e-12)
                rows.append({'fold':r['fold'],'seed':r['seed'],'test_position':origin['test_position'],
                    'concept_radius':float(R[0]),'graph_radius':float(R[1]),
                    'uniform_defect_fraction_threshold':admissible,
                    'conservative_uniform_defect_fraction':conservative,
                    'fixed_beta':beta,'fixed_beta_all_prefixes_contained':bool(np.max(ratios)<=1),
                    'fixed_beta_maximum_normalized_state_error':float(np.max(ratios)),
                    'prefix_block_defect_fraction_thresholds':np.asarray(limits).tolist(),
                    'fixed_beta_prefix_normalized_error_bounds':np.asarray(ratios).tolist()})
    assert len(sources)==15 and len(rows)==1920
    summary={}
    for fold in [1,2,3]:
        summary[str(fold)]=[]
        for rc in [.001,.01,.1,1.]:
            group=[x for x in rows if x['fold']==fold and x['concept_radius']==rc]
            assert len(group)==160
            values=[x['conservative_uniform_defect_fraction'] for x in group]
            summary[str(fold)].append({'concept_radius':rc,'graph_radius':rc/2,
                'trajectories':len(group),'admissible_fraction':stats(values),
                'common_admissible_fraction_rounded_down_6dp':math.floor(min(values)*1e6)/1e6,
                'fixed_beta_certified_count':sum(x['fixed_beta_all_prefixes_contained'] for x in group),
                'fixed_beta_maximum_normalized_state_error':max(x['fixed_beta_maximum_normalized_state_error'] for x in group)})
    result={'schema':'trefin-local-transition-defect-budget-v1','status':'complete',
        'initial_fraction':alpha,'fixed_defect_fraction_checked':beta,'registered_radii':[.001,.01,.1,1.],
        'source_state_summary_sha256':sha(a.state_results/'summary.json'),'source_script_sha256':sha(__file__),
        'source_files':sources,'folds':summary,'records':rows,
        'scope':'Conditional sensitivity/error propagation for an admissible comparison trajectory, or projected perturbed updates, inside the same certified state neighborhoods, with fixed market/lead/parameters. The reference path stays in the registered concept and graph domains. No actual reference or true transition defects are estimated.',
        'formula':'P_0=I,S_0=0; P_{h+1}=M_h P_h,S_{h+1}=M_h S_h+I. For an admissible comparison trajectory, initial error<=alpha R and each one-step transition residual<=beta R imply error_h<=alpha P_h R+beta S_h R if all prefixes stay inside R.',
        'threshold_rule':'minimum over all five prefixes and both state blocks of (R-alpha P_h R)/(S_h R); downward numerical cushion then floor6dp for a common reported budget'}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'transition_defect_budgets.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'complete','trajectory_radius_cases':len(rows),'folds':summary},indent=2))

if __name__=='__main__':main()
