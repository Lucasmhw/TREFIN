"""Plot all completed folds; never render partial or synthetic results."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=['TEDM','Timer-XL','TimeXer','TimeMixer++','TREFIN']
PERIODS={1:'2020–2021',2:'2022–2023',3:'2024–Mar 2025'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.results.read_text());audit=json.loads(a.audit.read_text())
    assert r['status']=='complete' and r['completed_baseline_runs']==60 and r['new_baseline_runs']==40
    assert r['seeds']==[1,7,19,37,73] and r['folds']==[1,2,3]
    assert audit['status']=='passed' and audit['source_results_sha256']==sha(a.results)
    rows=[]
    for fold in [1,2,3]:
        for label in ORDER:
            metrics=(r['reference_metrics'][str(fold)] if label=='TREFIN' else r['results'][str(fold)][label])['metrics']
            for stage in ['before','after']:
                row={'fold':fold,'test_period':PERIODS[fold],'method':label,'stage':stage,'seeds':5}
                for metric in ['mse','energy_score']:
                    for stat in ['mean','sample_std']:
                        value=metrics[stage][metric][stat];assert np.isfinite(value) and value>=0
                        row[metric+'_'+stat]=value
                rows.append(row)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(12.5,10.1))
    palette={'before':'#87939B','after':'#176B87'}
    for fold in [1,2,3]:
        for column,(metric,scale,xlabel) in enumerate([('mse',1e4,r'MSE $\times 10^4$'),('energy_score',1e2,r'Energy score $\times 10^2$')]):
            ax=axes[fold-1,column];ys=np.arange(5)[::-1]
            for y,label in zip(ys,ORDER):
                values={s:next(v for v in rows if v['fold']==fold and v['method']==label and v['stage']==s) for s in ['before','after']}
                ax.plot([values[s][metric+'_mean']*scale for s in ['before','after']],[y,y],color='#C2C9CD',lw=1.7,zorder=1)
                for stage,marker,offset in [('before','o',.065),('after','s',-.065)]:
                    v=values[stage]
                    ax.errorbar(v[metric+'_mean']*scale,y+offset,xerr=v[metric+'_sample_std']*scale,fmt=marker,color=palette[stage],markersize=5,capsize=2,lw=1.2,
                        label=('Before calibration' if stage=='before' else 'Validation-only mean calibration') if label==ORDER[0] else None)
            ax.axhspan(-.48,.48,color='#ECF3F6',zorder=0)
            ax.set_yticks(ys,ORDER if column==0 else ['']*5)
            ax.set_ylim(-.6,4.6);ax.grid(axis='x',alpha=.18)
            ax.set_xlabel(xlabel+' (lower is better)')
            ax.set_title(f"({chr(97+(fold-1)*2+column)}) Fold {fold}: {PERIODS[fold]}",loc='left',fontweight='bold',fontsize=11)
            ax.margins(x=.12)
    fig.suptitle('Matched calibration across three chronological test periods',fontsize=15,fontweight='bold',y=.98)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,.947),ncol=2,frameon=False)
    fig.text(.03,.022,'Five seeds per method and fold; error bars show sample SD. Each fold uses its own training and validation data.\nAll five-step forecasts use 32 assets and 256 test scenarios; model and calibration selection use validation data only.',fontsize=9,color='#424B50',ha='left',va='bottom')
    fig.subplots_adjust(left=.15,right=.98,bottom=.10,top=.87,wspace=.18,hspace=.53)
    a.out.mkdir(parents=True,exist_ok=True);stem='R8_crossfold_matched_calibration'
    for ext in ['png','svg','pdf']:fig.savefig(a.out/(stem+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    with (a.out/(stem+'_data.csv')).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    caption=('**Figure R8. Matched mean calibration across three chronological test periods.** Each panel compares TREFIN with TEDM, Timer-XL, TimeXer and TimeMixer++ under the documented task adapters. '
        'Every method has five training seeds in each fold. Symbols show mean MSE or joint energy score before and after validation-only positive mean shrinkage; error bars are sample standard deviations across seeds. '
        'Training, normalization, checkpoint selection and calibration use only the corresponding fold’s training/validation periods. The frozen fitting and calibration protocol for the four adapters is shared across folds. '
        'All comparisons align the same test origins within each fold, 32 assets and five leads, with 256 test scenarios. TEDM supplies native stochastic scenarios, the three point backbones use centered joint training-residual paths, and TREFIN uses its original probabilistic decoder. '
        'Axis ranges vary by fold; before/after points connected by a line belong to the same fixed predictor. Narrow seed error bars may be smaller than their markers.\n')
    (a.out/(stem+'_caption.txt')).write_text(caption,encoding='utf8')
    files=[p for p in sorted(a.out.glob(stem+'*')) if p.name!=stem+'_manifest.json']
    manifest={'status':'complete','script_sha256':sha(__file__),'source_results_sha256':sha(a.results),'independent_audit_sha256':sha(a.audit),'rows':len(rows),'complete_baseline_runs':60,'original_trefin_reference_runs':15,'files':{p.name:sha(p) for p in files}}
    (a.out/(stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'complete','figure':stem,'data_rows':len(rows)}))

if __name__=='__main__':main()
