#!/usr/bin/env python3
"""Plot completed fixed/validation-selected structural summaries, never placeholders.

python plot_structural_comparison.py --fixed-summary fixed/summary/summary.json \
 --tuned-summary tuned/summary/summary.json --fold 1 --output-dir work/figures

Requires the current summarize_retraining.py schema and all25 runs in both inputs.
No figure is produced if evidence is incomplete or per-run state SD is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


VARIANTS=['Full','NoGraph','SingleSigned','SingleUnsigned','NoStateSupervision']
LABELS=['Full','NoGraph','Single\nsigned','Single\nunsigned','No state\nsupervision']


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validated(path,fold):
    summary=json.loads(Path(path).read_text(encoding='utf-8'))
    if summary.get('status')!='complete' or summary.get('missing_runs'):
        raise ValueError(f'Only complete evidence may be plotted: {path}')
    if str(fold) not in summary['folds']:raise ValueError(f'Fold{fold} is absent: {path}')
    f=summary['folds'][str(fold)]
    for variant in VARIANTS:
        metrics=f['metrics'][variant]
        if sorted(metrics['seeds'])!=[1,7,19,37,73]:raise ValueError('Expected five specified seeds for every variant')
        for metric in ['mse','energy_score']:
            s=metrics['calibrated'][metric]
            if s['n']!=5 or s['sample_sd'] is None:raise ValueError('Missing seed dispersion')
        state=f['states'][variant]
        for metric in ['concept_mae_overall','graph_iou_overall']:
            if metric not in state:raise ValueError('Re-run current summarize_retraining.py to retain genuine per-run state averages/SD; coordinate SDs cannot be combined as if independent')
            if state[metric]['n']!=5 or state[metric]['sample_sd'] is None:raise ValueError('Missing per-run state dispersion')
        if variant!='Full':
            test=f['paired_tests'][variant]
            if test['status']!='complete' or len(test['matched_seeds'])!=5:raise ValueError('Incomplete paired comparison')
            for metric in ['mse','energy_score']:
                primary=next(x for x in test['metrics'][metric] if x['primary'])
                if primary['requested_block_length']!=20:raise ValueError('Primary paired intervals must use20-origin blocks')
    return summary,f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed-summary',type=Path,required=True)
    p.add_argument('--tuned-summary',type=Path,required=True)
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    fixed,ff=load_validated(args.fixed_summary,args.fold)
    tuned,tf=load_validated(args.tuned_summary,args.fold)
    protocols=[('Fixed configuration',ff,'#0072B2','o',-.13),
               ('Validation-selected configuration',tf,'#D55E00','s',.13)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
                         'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.8,'xtick.labelsize':9,'ytick.labelsize':10,'svg.fonttype':'none',
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(2,3,figsize=(15.5,8.7));handles=[];rows=[]
    upper=[('mse',1e4,'(a) Point error',r'MSE $\times10^4$'),
           ('energy_score',1e2,'(b) Distribution score',r'Energy score $\times10^2$'),
           ('concept_mae_overall',1.,'(c) Concept prediction','Mean concept MAE (standardized)')]
    for col,(metric,multiplier,title,ylabel) in enumerate(upper):
        axis=ax[0,col]
        for protocol,f,color,marker,offset in protocols:
            values=[f['metrics'][v]['calibrated'][metric] if col<2 else f['states'][v][metric] for v in VARIANTS]
            means=np.array([v['mean'] for v in values])*multiplier
            sd=np.array([v['sample_sd'] for v in values])*multiplier
            artist=axis.errorbar(np.arange(5)+offset,means,yerr=sd,fmt=marker,color=color,
                                markersize=5.5,capsize=3,elinewidth=1.2,capthick=1.1,
                                markeredgecolor='white',markeredgewidth=.5,label=protocol,zorder=3)
            if col==0:handles.append(artist)
            for i,variant in enumerate(VARIANTS):
                rows.append({'fold':args.fold,'protocol':protocol,'panel':title,'variant':variant,'metric':metric,
                             'estimate_display':float(means[i]),'lower_display':float(means[i]-sd[i]),'upper_display':float(means[i]+sd[i]),
                             'interval_type':'mean plus/minus sampleSD across five seeds','display_multiplier':multiplier})
        axis.set(title=title,ylabel=ylabel,xlim=(-.55,4.55))
        axis.set_xticks(np.arange(5),LABELS);axis.grid(axis='y',alpha=.17);axis.tick_params(axis='x',length=0,pad=6)
        axis.margins(y=.15)
    for col,(metric,multiplier,title,ylabel) in enumerate([
        ('mse',1e4,'(d) Paired point-error differences',r'$\Delta$MSE $\times10^4$ (variant $-$ Full)'),
        ('energy_score',1e2,'(e) Paired distribution-score differences',r'$\Delta$ES $\times10^2$ (variant $-$ Full)')]):
        axis=ax[1,col];axis.axhline(0,color='#68717D',lw=1,ls=(0,(4,3)),zorder=1)
        for protocol,f,color,marker,offset in protocols:
            vals=[next(t for t in f['paired_tests'][v]['metrics'][metric] if t['primary']) for v in VARIANTS[1:]]
            point=np.array([v['mean_difference'] for v in vals])*multiplier
            lower=np.array([v['ci95'][0] for v in vals])*multiplier
            upper=np.array([v['ci95'][1] for v in vals])*multiplier
            # Percentile intervals can theoretically exclude their point estimate;
            # draw independent interval segments rather than requiring nonnegative yerr.
            xpos=np.arange(4)+offset
            axis.vlines(xpos,lower,upper,color=color,lw=1.5,zorder=2)
            axis.hlines(lower,xpos-.055,xpos+.055,color=color,lw=1.2,zorder=2)
            axis.hlines(upper,xpos-.055,xpos+.055,color=color,lw=1.2,zorder=2)
            axis.scatter(xpos,point,c=color,marker=marker,s=31,edgecolors='white',linewidths=.5,zorder=3)
            for i,variant in enumerate(VARIANTS[1:]):
                rows.append({'fold':args.fold,'protocol':protocol,'panel':title,'variant':variant,'metric':metric,
                             'estimate_display':float(point[i]),'lower_display':float(lower[i]),'upper_display':float(upper[i]),
                             'interval_type':'95% paired circular-block bootstrapCI; block20; matched seed losses averaged at each origin',
                             'display_multiplier':multiplier})
        axis.set(title=title,ylabel=ylabel,xlim=(-.55,3.55));axis.set_xticks(np.arange(4),LABELS[1:])
        axis.grid(axis='y',alpha=.17);axis.tick_params(axis='x',length=0,pad=6);axis.margins(y=.15)
    axis=ax[1,2]
    for protocol,f,color,marker,offset in protocols:
        vals=[f['states'][v]['graph_iou_overall'] for v in VARIANTS]
        point=np.array([v['mean'] for v in vals]);sd=np.array([v['sample_sd'] for v in vals])
        axis.errorbar(np.arange(5)+offset,point,yerr=sd,fmt=marker,color=color,markersize=5.5,
                      capsize=3,elinewidth=1.2,capthick=1.1,markeredgecolor='white',markeredgewidth=.5,zorder=3)
        for i,variant in enumerate(VARIANTS):
            rows.append({'fold':args.fold,'protocol':protocol,'panel':'(f) Graph support','variant':variant,'metric':'graph_iou_overall',
                         'estimate_display':float(point[i]),'lower_display':float(point[i]-sd[i]),'upper_display':float(point[i]+sd[i]),
                         'interval_type':'mean plus/minus sampleSD across five seeds','display_multiplier':1.})
    axis.set(title='(f) Graph support',ylabel=r'Mean graph IoU ($|A_{ij}|>0.05$)',ylim=(-.04,1.04),xlim=(-.55,4.55))
    axis.set_xticks(np.arange(5),LABELS);axis.grid(axis='y',alpha=.17);axis.tick_params(axis='x',length=0,pad=6)
    fig.legend(handles,[x[0] for x in protocols],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.005),fontsize=11)
    fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.89,wspace=.34,hspace=.43)
    fig.text(.065,.035,'(a–c,f): mean ± seed SD.  (d,e):95% temporal block-bootstrap CI, after averaging matched seed losses per origin.',fontsize=9.5,color='#4B5563')
    fig.text(.065,.012,'Positive paired loss differences favor Full. NoGraph removes the A channel but retains market-state residual-dependence information.',fontsize=9.5,color='#4B5563')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    base=output/'R6_structural_comparison'
    for suffix in ['png','svg','pdf']:fig.savefig(base.with_suffix('.'+suffix),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    with (output/'R6_structural_comparison_data.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    caption=f'''**Figure R6. Matched structural comparisons on fold {args.fold}.** Fixed and validation-selected configurations are shown for all five variants, each trained with seeds 1, 7, 19, 37 and 73 and the same positive validation-only mean-calibration procedure. Panels (a–c,f) show five-seed means and sample SD. Concept MAE averages standardized errors over future leads 1–5 and all coordinates within each run; graph IoU averages future-lead IoU within each run. Seed SDs are calculated from these per-run averages. Graph support uses strict absolute off-diagonal weight >0.05; empty unions have IoU 0. Panels (d,e) show candidate-minus-Full loss differences and 95% paired circular-block-bootstrap intervals with 20-origin blocks. Each origin retains all assets and leads; matched seed losses are averaged before temporal resampling. Positive differences favor Full. NoGraph removes the A input, rollout and decoder channel while retaining the market state, including its residual-dependence summary.
'''
    (output/'structural_caption.md').write_text(caption,encoding='utf-8')
    manifest={'fold':args.fold,'fixed_summary':str(args.fixed_summary.resolve()),'fixed_summary_sha256':sha256(args.fixed_summary),
              'tuned_summary':str(args.tuned_summary.resolve()),'tuned_summary_sha256':sha256(args.tuned_summary),
              'script_sha256':sha256(__file__),'all_required_runs_complete':True,
              'fixed_es_precision':fixed['definitions']['energy_score_source'],'tuned_es_precision':tuned['definitions']['energy_score_source'],
              'outputs':[str(base.with_suffix('.'+s)) for s in ['png','svg','pdf']]}
    (output/'structural_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
