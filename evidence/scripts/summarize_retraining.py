#!/usr/bin/env python3
"""Summarize every matched structural variant without selecting favorable results.

python summarize_retraining.py --root /path/rebuttal/retraining --exact-es
python summarize_retraining.py --root /path/rebuttal/retraining_tuned --folds 1

Default: require all five variants and seeds [1,7,19,37,73] in each requested fold.
--allow-incomplete is explicitly diagnostic; no final completeness claim is made.
numpy is required; scipy is required only for --exact-es.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ['Full','NoGraph','SingleSigned','SingleUnsigned','NoStateSupervision']
SEEDS = [1,7,19,37,73]
METRICS = ['mse','rmse','mae','energy_score','msic','msir']
SCALES = {'mse':1e4,'rmse':1e2,'mae':1e2,'energy_score':1e2,'msic':1e2,'msir':1e2}
SEMANTICS = {
    'Full':'Original typed transition and measured-state decoder.',
    'NoGraph':'A channel is zero at input, every transition and decoder; graph-target loss is zero. Market state is retained, including its residual-dependence summary. This does not remove every graph-derived statistic.',
    'SingleSigned':'All transition coordinates use the signed Laplacian; decoder risk masking and registered financial concept definitions remain unchanged.',
    'SingleUnsigned':'All transition coordinates use the unsigned Laplacian; decoder risk masking and registered financial concept definitions remain unchanged.',
    'NoStateSupervision':'Both forward concept and graph target-loss coefficients are zero; the forward state architecture remains unchanged.'}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def dump(value: Any,path: Path) -> None:
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def write_csv(rows,path):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(values):
    v=np.asarray(values,dtype=np.float64)
    if not v.size:return {'mean':None,'sample_sd':None,'n':0,'values':[]}
    if not np.isfinite(v).all():raise ValueError('Nonfinite summary values')
    return {'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)) if len(v)>1 else None,
            'n':len(v),'values':v.tolist()}


def exact_energy_loss(samples,target):
    from scipy.spatial.distance import pdist
    if samples.ndim!=4 or samples.shape[1:]!=target.shape or samples.shape[0]<2:
        raise ValueError('Expected scenarios[S,B,H,N] and target[B,H,N]')
    # Compute all terms in float64. Stream origins to bound temporary memory.
    count=samples.shape[0];loss=np.empty(len(target),dtype=np.float64)
    for b in range(len(target)):
        x=samples[:,b].astype(np.float64)
        first=np.linalg.norm(x-target[b][None],axis=-1).mean(axis=0)
        second=np.array([pdist(x[:,h]).sum()/(count*(count-1)) for h in range(target.shape[1])])
        loss[b]=(first-second).mean()
    return loss


def load_run(directory,fold,seed,variant,exact_es):
    complete_path=directory/'complete.json';complete=read_json(complete_path)
    for key,expected in [('fold',fold),('seed',seed),('variant',variant)]:
        if complete.get(key)!=expected:raise ValueError(f'Run identity mismatch: {directory}, {key}')
    loss_path=directory/'tables/origin_losses.npz';state_path=directory/'tables/states.json'
    if not loss_path.is_file() or not state_path.is_file():raise FileNotFoundError(f'Completed run lacks diagnostics: {directory}')
    with np.load(loss_path,allow_pickle=False) as f:
        indices=np.asarray(f['indices'],dtype=np.int64)
        target=np.asarray(f['y'],dtype=np.float64);mean=np.asarray(f['mu'],dtype=np.float64)
        stored_es=np.asarray(f['es'],dtype=np.float64)
    if target.shape!=mean.shape or target.ndim!=3 or len(indices)!=len(target):raise ValueError('Invalid point-array shape')
    if np.any(np.diff(indices)<=0) or len(np.unique(indices))!=len(indices):raise ValueError('Origins not unique/chronological')
    if target.shape[1]!=5:raise ValueError('This registered experiment expects H=5')
    if not all(np.isfinite(x).all() for x in [target,mean,stored_es]):raise ValueError('Nonfinite losses/predictions')
    if stored_es.shape!=(len(target),):raise ValueError('Stored ES must have one value per origin')
    losses={'mse':np.square(mean-target).mean(axis=(1,2)),
            'mae':np.abs(mean-target).mean(axis=(1,2)),'energy_score':stored_es}
    provenance={str(p.resolve()):sha256(p) for p in [complete_path,loss_path,state_path]}
    precision={'mse':'recomputed in float64 from mu/y','mae':'recomputed in float64 from mu/y',
               'energy_score':'stored origin_losses.npz values cast to float64; source arithmetic may have used float32'}
    es_difference=None
    if exact_es:
        prediction_path=directory/'predictions/test_predictions.npz'
        with np.load(prediction_path,allow_pickle=False) as f:
            if not np.array_equal(f['indices'],indices) or not np.array_equal(f['y'],target) or not np.array_equal(f['mu'],mean):
                raise ValueError(f'Full predictions and origin losses do not align: {directory}')
            es=exact_energy_loss(f['samples'],target)
        es_difference=float(np.max(np.abs(es-stored_es)))
        losses['energy_score']=es
        provenance[str(prediction_path.resolve())]=sha256(prediction_path)
        precision['energy_score']='exact finite-scenario U-statistic recomputed in float64 from original full test scenarios'
    states=read_json(state_path)
    if float(states.get('graph_threshold',-1))!=.05:raise ValueError('Graph threshold differs from manuscript .05')
    for key in ['concept_mae_lead_coordinate','concept_persistence_mae_lead_coordinate']:
        array=np.asarray(states[key],dtype=np.float64)
        if array.ndim!=2 or array.shape[0]!=5 or not np.isfinite(array).all():raise ValueError('Invalid concept-state metrics')
    if np.asarray(states['concept_mae_lead_coordinate']).shape!=np.asarray(states['concept_persistence_mae_lead_coordinate']).shape:
        raise ValueError('Concept persistence shape differs')
    for key in ['graph_iou_lead','graph_mae_lead']:
        array=np.asarray(states[key],dtype=np.float64)
        if array.shape!=(5,) or not np.isfinite(array).all():raise ValueError('Invalid graph metrics')
    checks={}
    for metric in ['mse','mae','energy_score']:
        registered=float(complete['calibrated'][metric]);recomputed=float(losses[metric].mean())
        if not np.isclose(registered,recomputed,rtol=2e-6,atol=1e-7):
            raise ValueError(f'{directory}: {metric} does not reproduce completion record: {registered} vs {recomputed}')
        checks[metric]={'registered':registered,'origin_recomputed_mean':recomputed,'absolute_difference':abs(registered-recomputed)}
    for optional in ['config_resolved.yaml','audit/provenance.json','audit/checkpoint_reload_check.json']:
        path=directory/optional
        if path.exists():provenance[str(path.resolve())]=sha256(path)
    if (directory/'audit/checkpoint_reload_check.json').exists():
        if not read_json(directory/'audit/checkpoint_reload_check.json').get('passed',False):raise ValueError('Failed checkpoint reload')
    return {'fold':fold,'seed':seed,'variant':variant,'directory':str(directory.resolve()),'complete':complete,
            'indices':indices,'target':target,'losses':losses,'states':states,'provenance':provenance,
            'precision':precision,'metric_reproduction':checks,'max_exact_minus_stored_es':es_difference}


def circular_bootstrap(difference,block_length,repetitions,seed):
    d=np.asarray(difference,dtype=np.float64)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():raise ValueError('Invalid origin differences')
    block=min(block_length,len(d));n=len(d);rng=np.random.default_rng(seed)
    offsets=np.arange(block);means=np.empty(repetitions)
    for start in range(0,repetitions,200):
        stop=min(start+200,repetitions)
        positions=rng.integers(n,size=(stop-start,int(np.ceil(n/block))))
        indices=((positions[:,:,None]+offsets[None,None,:])%n).reshape(stop-start,-1)[:,:n]
        means[start:stop]=d[indices].mean(axis=1)
    effect=float(d.mean());ci=np.quantile(means,[.025,.975])
    centered=means-effect
    p=float((1+np.count_nonzero(np.abs(centered)>=abs(effect)))/(repetitions+1))
    return {'mean_difference':effect,'ci95':[float(ci[0]),float(ci[1])],'centered_two_sided_p':p,
            'block_length':block,'requested_block_length':block_length,'repetitions':repetitions,'random_seed':seed,
            'origins':n,'holm_p':None}


def holm(values):
    names=sorted(values,key=lambda x:(values[x],x));out={};running=0.
    for i,name in enumerate(names):
        running=max(running,min(1.,(len(names)-i)*values[name]));out[name]=running
    return out


def summarize(args):
    root=args.root.resolve();output=args.output_dir.resolve() if args.output_dir else root/'summary'
    runs={};missing=[]
    for fold in args.folds:
        for seed in SEEDS:
            for variant in VARIANTS:
                directory=root/f'fold_{fold}_seed_{seed}'/variant
                if not (directory/'complete.json').exists():missing.append({'fold':fold,'seed':seed,'variant':variant,'directory':str(directory)});continue
                runs[(fold,seed,variant)]=load_run(directory,fold,seed,variant,args.exact_es)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f'Missing {len(missing)} of {25*len(args.folds)} required runs. Use --allow-incomplete only for explicitly diagnostic summaries. Missing: '+json.dumps(missing))
    if not runs:raise RuntimeError('No completed runs available')
    output.mkdir(parents=True,exist_ok=True)
    metrics_rows=[];paired_rows=[];state_rows=[];coordinate_rows=[];saved={}
    report={'schema':'trefin-matched-retraining-summary-v1','source_root':str(root),'folds':{},'missing_runs':missing,
            'status':'diagnostic_incomplete' if missing else 'complete','expected_runs':25*len(args.folds),'completed_runs':len(runs),
            'variants':VARIANTS,'required_seeds':SEEDS,'ablation_semantics':SEMANTICS,
            'definitions':{'point_metrics':'Six headline metrics from complete.json; means and sample SD over separately trained seeds. RMSE computed per run before seed aggregation.',
                           'display_multipliers':SCALES,'paired_effect':'candidate variant minus Full; positive loss difference favors Full',
                           'paired_unit':'One chronological origin contains all assets and all five leads; average paired losses across matched seeds before temporal resampling. Seeds are never independent time observations.',
                           'bootstrap':'paired circular block bootstrap; 5000 draws by default; percentile95%CI and centered-null two-sided p',
                           'primary_block_length':20,'sensitivity_block_lengths':[5,10],
                           'multiple_testing':'Holm correction over all four non-Full variants, separately for MSE and ES within each fold and block length. Missing variants receive p=1 in diagnostic correction; MAE is unadjusted.',
                           'concept_MAE':'Stored standardized projected concept MAE per lead/coordinate, averaged over origins and assets; h=0 excluded. Overall coordinate MAE uses equal weights over the five leads.',
                           'graph_IoU':'Strict abs(A_ij)>0.05 on off-diagonal entries, per-origin intersection/union then mean over origins. Empty union is zero; h=0 excluded.',
                           'graph_MAE':'Mean absolute off-diagonal edge-weight error, per future lead.',
                           'energy_score_source':'exact float64 recomputation from complete scenarios' if args.exact_es else 'stored per-origin ES; use --exact-es for final high-precision inference'},
            'provenance':{'script_sha256':sha256(Path(__file__)),'runs':[]}}
    for run in runs.values():
        report['provenance']['runs'].append({k:run[k] for k in ['fold','seed','variant','directory','provenance','precision','metric_reproduction','max_exact_minus_stored_es']})
    for fold in args.folds:
        fr={'metrics':{},'paired_tests':{},'states':{}}
        # Cross-seed and cross-variant target identity is essential to pairing.
        available=[r for key,r in runs.items() if key[0]==fold]
        if available:
            ref=available[0]
            for r in available[1:]:
                if not np.array_equal(ref['indices'],r['indices']) or not np.array_equal(ref['target'],r['target']):
                    raise ValueError(f'Origin/target mismatch within fold{fold}: {r["directory"]}')
            saved[f'fold_{fold}_origin_indices']=ref['indices']
        for variant in VARIANTS:
            group=[runs[(fold,seed,variant)] for seed in SEEDS if (fold,seed,variant) in runs]
            fr['metrics'][variant]={'seeds':[r['seed'] for r in group],'calibrated':{},'uncalibrated':{}}
            for calibration in ['calibrated','uncalibrated']:
                for metric in METRICS:
                    values=[float(r['complete'][calibration][metric]) for r in group if calibration in r['complete']]
                    statistic=stats(values);fr['metrics'][variant][calibration][metric]=statistic
                    metrics_rows.append({'fold':fold,'variant':variant,'calibration':calibration,'metric':metric,'n_seeds':len(values),
                                         'seed_ids':','.join(str(r['seed']) for r in group),'mean_raw':statistic['mean'],'sample_sd_raw':statistic['sample_sd'],
                                         'display_multiplier':SCALES[metric],
                                         'mean_display':statistic['mean']*SCALES[metric] if statistic['mean'] is not None else None,
                                         'sample_sd_display':statistic['sample_sd']*SCALES[metric] if statistic['sample_sd'] is not None else None})
            if group:
                c=np.stack([r['states']['concept_mae_lead_coordinate'] for r in group]);cp=np.stack([r['states']['concept_persistence_mae_lead_coordinate'] for r in group])
                if c.shape!=cp.shape:raise ValueError('Concept coordinate mismatch')
                gs={key:np.array([r['states'][key] for r in group]) for key in ['graph_iou_lead','graph_mae_lead']}
                fr['states'][variant]={'seed_ids':[r['seed'] for r in group],'concept_mae_lead_coordinate_seed_mean':c.mean(0).tolist(),
                                       'concept_mae_lead_coordinate_seed_sd':c.std(0,ddof=1).tolist() if len(group)>1 else None,
                                       'concept_mae_overall':stats(c.mean(axis=(1,2))),
                                       'concept_mae_lead_seed_values':c.mean(axis=2).tolist(),
                                       'concept_persistence_mae_overall':stats(cp.mean(axis=(1,2))),
                                       'graph_iou_overall':stats(gs['graph_iou_lead'].mean(axis=1)),
                                       'graph_iou_lead_seed_values':gs['graph_iou_lead'].tolist(),
                                       'concept_persistence_mae_lead_coordinate_seed_mean':cp.mean(0).tolist(),
                                       'concept_coordinates_better_than_persistence_per_lead':(c.mean(0)<cp.mean(0)).sum(1).tolist(),
                                       'concept_coordinates_better_than_persistence_overall':int((c.mean((0,1))<cp.mean((0,1))).sum()),
                                       'concept_coordinate_count':int(c.shape[-1])}
                for lead in range(5):
                    for key,array in [('concept_mae',c.mean(2)),('concept_persistence_mae',cp.mean(2)),*gs.items()]:
                        st=stats(array[:,lead]);state_rows.append({'fold':fold,'variant':variant,'lead':lead+1,'metric':key,'mean':st['mean'],'seed_sample_sd':st['sample_sd'],'n_seeds':len(group)})
                    for k in range(c.shape[-1]):
                        st=stats(c[:,lead,k]);pst=stats(cp[:,lead,k])
                        coordinate_rows.append({'fold':fold,'variant':variant,'lead':lead+1,'coordinate_index_zero_based':k,
                                                'learned_mae':st['mean'],'learned_seed_sample_sd':st['sample_sd'],'persistence_mae':pst['mean'],
                                                'difference_learned_minus_persistence':st['mean']-pst['mean'],'n_seeds':len(group)})
                for key,array in gs.items():
                    fr['states'][variant][key+'_seed_mean']=array.mean(0).tolist()
                    fr['states'][variant][key+'_seed_sd']=array.std(0,ddof=1).tolist() if len(group)>1 else None
            else:fr['states'][variant]={'seed_ids':[],'status':'missing'}
        for variant in VARIANTS[1:]:
            matched=[seed for seed in SEEDS if (fold,seed,'Full') in runs and (fold,seed,variant) in runs]
            if not matched:
                fr['paired_tests'][variant]={'status':'no_matched_seeds','matched_seeds':[]};continue
            comparison={'status':'complete' if len(matched)==5 else 'diagnostic_incomplete','matched_seeds':matched,'metrics':{}}
            for metric in ['mse','mae','energy_score']:
                by_seed=np.stack([runs[(fold,seed,variant)]['losses'][metric]-runs[(fold,seed,'Full')]['losses'][metric] for seed in matched])
                difference=by_seed.mean(axis=0)
                saved[f'fold_{fold}_{variant}_{metric}_seed_differences']=by_seed
                saved[f'fold_{fold}_{variant}_{metric}_seed_ids']=np.array(matched)
                tests=[]
                for block in [20,5,10]:
                    test=circular_bootstrap(difference,block,args.repetitions,args.seed+fold*100+block)
                    test['primary']=block==20;tests.append(test)
                comparison['metrics'][metric]=tests
            fr['paired_tests'][variant]=comparison
        for metric in ['mse','energy_score']:
            for block in [20,5,10]:
                values={variant:1. for variant in VARIANTS[1:]}
                for variant,comparison in fr['paired_tests'].items():
                    if 'metrics' in comparison:
                        values[variant]=next(t['centered_two_sided_p'] for t in comparison['metrics'][metric] if t['requested_block_length']==block)
                adjusted=holm(values)
                for variant,comparison in fr['paired_tests'].items():
                    if 'metrics' in comparison:
                        next(t for t in comparison['metrics'][metric] if t['requested_block_length']==block)['holm_p']=adjusted[variant]
        for variant,comparison in fr['paired_tests'].items():
            if 'metrics' not in comparison:continue
            for metric,tests in comparison['metrics'].items():
                for test in tests:
                    paired_rows.append({'fold':fold,'candidate':variant,'reference':'Full','metric':metric,
                                        'matched_seeds':','.join(map(str,comparison['matched_seeds'])),'n_seeds':len(comparison['matched_seeds']),
                                        'origins':test['origins'],'block_length':test['block_length'],'primary':test['primary'],
                                        'mean_difference_raw':test['mean_difference'],'ci95_lower_raw':test['ci95'][0],'ci95_upper_raw':test['ci95'][1],
                                        'two_sided_p':test['centered_two_sided_p'],'holm_p_four_variants':test['holm_p'],
                                        'repetitions':test['repetitions'],'random_seed':test['random_seed']})
        report['folds'][str(fold)]=fr
    dump(report,output/'summary.json')
    write_csv(metrics_rows,output/'metrics.csv');write_csv(paired_rows,output/'paired_tests.csv')
    write_csv(state_rows,output/'state_metrics.csv');write_csv(coordinate_rows,output/'concept_coordinates.csv')
    np.savez_compressed(output/'paired_origin_differences.npz',**saved)
    lines=['# Matched structural retraining summary','',f"Status: **{report['status']}**. Completed {len(runs)}/{report['expected_runs']} requested runs.",'',
           'All five variants are retained. Means and sample SD describe training-seed variation. Paired inference averages matched seed losses within each chronological origin; all assets and five leads remain grouped. Positive candidate-minus-Full loss differences favor Full. Primary circular block length is20;5 and10 are sensitivity checks. MSE and ES p values are Holm-adjusted over the four variants separately within each fold/block. MAE p values are unadjusted.','',
           'NoGraph removes the A input/rollout/decoder channel and graph-target loss. It retains the market state, including its residual-dependence summary; it does not remove every graph-derived quantity.','']
    if missing:lines+=['**Incomplete diagnostic only:** missing combinations are listed in summary.json. Pairwise seed sets may differ; this is not the final five-seed result.','']
    for fold,fr in report['folds'].items():
        lines += [f'## Fold{fold} calibrated results','',
                  'MSE is multiplied by10^4; all other displayed metrics are multiplied by10^2. RMSE is computed for each run before seed aggregation.','',
                  '| Variant | Seeds | MSE | RMSE | MAE | ES | msIC | msIR |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant in VARIANTS:
            m=fr['metrics'][variant];cells=[]
            for metric in METRICS:
                st=m['calibrated'][metric]
                cells.append('missing' if st['mean'] is None else f"{st['mean']*SCALES[metric]:.4f} ± "+(f"{st['sample_sd']*SCALES[metric]:.4f}" if st['sample_sd'] is not None else 'n/a'))
            lines.append('| '+variant+' | '+str(len(m['seeds']))+' | '+' | '.join(cells)+' |')
        lines += ['', 'Primary paired comparisons (raw loss units; no benefit claim is inferred from ranking alone):','',
                  '| Candidate − Full | Metric | Matched seeds | Mean difference |95% CI|Raw p|Holm p|',
                  '|---|---|---:|---:|---|---:|---:|']
        for variant in VARIANTS[1:]:
            comparison=fr['paired_tests'][variant]
            if 'metrics' not in comparison:
                lines.append(f'| {variant} | unavailable |0|—|—|—|—|');continue
            for metric in ['mse','energy_score']:
                t=next(t for t in comparison['metrics'][metric] if t['primary'])
                lines.append(f"|{variant}|{metric}|{len(comparison['matched_seeds'])}|{t['mean_difference']:.6g}|[{t['ci95'][0]:.6g},{t['ci95'][1]:.6g}]|{t['centered_two_sided_p']:.4f}|{t['holm_p']:.4f}|")
        lines+=['']
    lines += ['## State metrics and provenance','',
              'state_metrics.csv reports future-lead concept MAE, matched concept persistence MAE, graph IoU and graph edge MAE. concept_coordinates.csv retains every coordinate/lead. Anchored h=0 is excluded. Graph support uses strict |A_ij|>0.05 off the diagonal; an empty union has IoU0. These fidelity measures do not identify a causal market graph.','',
              'All input completion files, loss archives, state tables and available configuration/audit records are SHA256-hashed in summary.json. With --exact-es, full prediction archives are also hashed and the exact finite-scenario ES is recomputed in float64. Without that flag, stored ES retains its original numerical precision; point losses are always recomputed in float64.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'output':str(output),'status':report['status'],'completed':len(runs),'expected':report['expected_runs']}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--folds',nargs='+',type=int,default=[1])
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--exact-es',action='store_true')
    parser.add_argument('--repetitions',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261108)
    args=parser.parse_args()
    if args.repetitions<100:parser.error('At least100 bootstrap repetitions required')
    if len(set(args.folds))!=len(args.folds):parser.error('Folds must be unique')
    summarize(args)
