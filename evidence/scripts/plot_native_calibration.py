#!/usr/bin/env python3
"""Plot the completed additional native calibration experiment, without retraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--repo',type=Path,help='Full original repository with archived protocol runs')
    source.add_argument('--trefin-configs',type=Path,help='Packaged configs/table4_trefin with calibrated/uncalibrated summaries')
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,help='Defaults to --run; only figures/data are written here')
    ap.add_argument('--basename',default='native_matched_calibration')
    args=ap.parse_args()
    output=args.output_dir or args.run;output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run/'summary.json').read_text())
    source_files=[args.run/'summary.json']
    groups=summary['groups']
    order=['ShrunkVAR','HistGradientBoosting','RawHistoryMLP','DLinear','PatchTST','iTransformer','GRU','MASTER','StateMLP']
    rows=[next(g for g in groups if g['model']==name) for name in order]
    trefin={'model':'TREFIN'}
    for stage,protocol in [('before','finmultitime_us_rolling_3fold_5seed_v1'),('after','finmultitime_us_calibrated_rolling_3fold_5seed_v1')]:
        paths=[(args.trefin_configs/('uncalibrated' if stage=='before' else 'calibrated')/f'fold_1_seed_{seed}'/'run_summary.json') if args.trefin_configs else (args.repo/'runs/protocols'/protocol/'runs'/f'fold_1_seed_{seed}'/'run_summary.json') for seed in [1,7,19,37,73]]
        source_files.extend(paths)
        metrics=[json.loads(path.read_text())['test_metrics'] for path in paths]
        for key in ['mse','energy_score']:
            values=[m.get(key,m['rmse']**2) for m in metrics]
            trefin[f'{stage}_{key}_mean']=np.mean(values)
            trefin[f'{stage}_{key}_sample_std']=np.std(values,ddof=1)
    rows.append(trefin)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,6.4),sharey=True)
    colors={'before':'#87939B','after':'#176B87'}
    ys=np.arange(len(rows))[::-1]
    for ax,key,scale,label in [(axes[0],'mse',1e4,r'MSE $\times 10^4$ (lower is better)'),(axes[1],'energy_score',100,r'Energy score $\times 10^2$ (lower is better)')]:
        for i,(y,r) in enumerate(zip(ys,rows)):
            b=r[f'before_{key}_mean']*scale;a=r[f'after_{key}_mean']*scale
            ax.plot([a,b],[y,y],color='#C2C9CD',lw=1.7,zorder=1)
            for stage,marker,offset in [('before','o',.06),('after','s',-.06)]:
                ax.errorbar(r[f'{stage}_{key}_mean']*scale,y+offset,xerr=r[f'{stage}_{key}_sample_std']*scale,fmt=marker,color=colors[stage],markersize=5,capsize=2,lw=1.2,label=('Before calibration' if stage=='before' else 'Same validation-only mean calibration') if i==0 else None)
        ax.axhspan(-.5,.48,color='#ECF3F6',zorder=0)
        ax.set_xlabel(label)
        ax.set_ylim(-.6,len(rows)-.35)
        ax.grid(axis='y',visible=False)
    axes[0].set_yticks(ys,[r['model'] for r in rows])
    axes[0].set_title('(a) Point error',loc='left',fontweight='bold')
    axes[1].set_title('(b) Joint distribution score',loc='left',fontweight='bold')
    fig.suptitle('Equal access to mean shrinkage: supplementary native comparison',fontsize=14,fontweight='bold',y=.97)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.57,.92),ncol=2,frameon=False)
    fig.text(.03,.045,'Fold 1; five training seeds; error bars = sample SD. All checkpoints are frozen.\nSupplementary native baselines use training-residual scenarios; TREFIN uses its original decoder.',fontsize=9,color='#424B50',ha='left',va='bottom')
    fig.subplots_adjust(left=.20,right=.98,bottom=.18,top=.80,wspace=.18)
    for suffix in ['png','svg','pdf']:
        fig.savefig(output/f'{args.basename}.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (output/'figure_caption.txt').write_text('Supplementary matched mean calibration of nine archived native baseline implementations. Each previously selected native checkpoint is frozen, and a positive scalar in [0.05,1] is selected using validation energy score and fixed innovations. The panels report point MSE and joint energy score before and after calibration on the same fold-1 test origins. Symbols are five-seed means; error bars denote sample standard deviations over training seeds. Native baseline distributions use centered training residual paths, whereas TREFIN uses its original probabilistic decoder. These additional results use native implementations distinct from the task adaptations underlying the submitted main table.\n',encoding='utf8')

    plot_rows=[]
    for row in rows:
        for stage in ['before','after']:
            plot_rows.append({'model':row['model'],'stage':stage,'mse_mean':float(row[f'{stage}_mse_mean']),'mse_sample_sd':float(row[f'{stage}_mse_sample_std']),'energy_score_mean':float(row[f'{stage}_energy_score_mean']),'energy_score_sample_sd':float(row[f'{stage}_energy_score_sample_std']),'seeds':5})
    with (output/f'{args.basename}_data.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plot_rows[0]));writer.writeheader();writer.writerows(plot_rows)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance={'script_sha256':digest(__file__),'source_files':[{'path':str(path.resolve()),'sha256':digest(path)} for path in source_files],'outputs':[str((output/f'{args.basename}.{suffix}').resolve()) for suffix in ['png','svg','pdf']],'numerical_definition':'All original means/sampleSD unchanged; only optional packaged-summary input and output-path controls added'}
    (output/f'{args.basename}_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')


if __name__=='__main__':main()
