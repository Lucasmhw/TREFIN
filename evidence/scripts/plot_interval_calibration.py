"""Publication figure from audited calibration aggregates; no simulated values."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    with a.input.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    a.output.mkdir(parents=True,exist_ok=True)
    methods=[('original','Original','#4B5563','o'),('validation_ES_scale','Fixed validation ES','#0072B2','s'),
             ('validation_90pct_scale','Fixed validation 90%','#D55E00','^'),
             ('causal_rolling_90pct_scale','Causal rolling 90%','#009E73','D')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
                         'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.8,'xtick.labelsize':9.5,'ytick.labelsize':10,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.6))
    panels=[('coverage90','(a) Marginal interval coverage','Coverage (%)',(77.5,96.3),5),
            ('width90','(b) Mean 90% interval width','Width (percentage points of return)',(4.5,8.2),1),
            ('energy_score','(c) Energy score',r'Energy score $\times 10^2$',(6.75,9.2),.5)]
    offsets=np.linspace(-.24,.24,len(methods));handles=[]
    for ax,(metric,title,ylabel,limits,spacing) in zip(axes,panels):
        if metric=='coverage90':ax.axhline(90,color='#65717E',ls=(0,(4,3)),lw=1,zorder=1)
        for i,(key,label,color,marker) in enumerate(methods):
            chosen=[next(r for r in rows if int(r['fold'])==fold and r['method']==key) for fold in [1,2,3]]
            mean=np.array([float(r[metric+'_mean']) for r in chosen])*100
            sd=np.array([float(r[metric+'_seed_std']) for r in chosen])*100
            artist=ax.errorbar(np.arange(3)+offsets[i],mean,yerr=sd,fmt=marker,color=color,
                               ms=6,capsize=3.2,capthick=1.2,elinewidth=1.2,
                               markeredgecolor='white',markeredgewidth=.5,zorder=3,label=label)
            if metric=='coverage90':handles.append(artist)
        ax.set(title=title,ylabel=ylabel,ylim=limits,xlim=(-.6,2.6))
        ax.set_xticks(np.arange(3),['Fold 1\n2020–2021','Fold 2\n2022–2023','Fold 3\n2024–Mar 2025'])
        ax.yaxis.set_major_locator(MultipleLocator(spacing));ax.grid(axis='y',alpha=.17,zorder=0)
        ax.tick_params(axis='x',length=0,pad=8)
    axes[0].text(2.58,90.4,'Nominal 90%',ha='right',va='bottom',color='#56606E',fontsize=9)
    fig.legend(handles,[m[1] for m in methods],loc='upper center',ncol=4,frameon=False,
               bbox_to_anchor=(.51,1.005),columnspacing=1.9,handletextpad=.55,fontsize=10)
    fig.subplots_adjust(left=.065,right=.99,bottom=.22,top=.81,wspace=.36)
    fig.text(.065,.025,'Points show five-seed means; error bars show sample SD across seeds.',fontsize=9,color='#4B5563')
    basename=a.output/'R5_interval_calibration'
    for suffix in ['png','svg','pdf']:fig.savefig(basename.with_suffix('.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)
    with (a.output/'R5_interval_calibration_data.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['fold','method','coverage90_mean_pct','coverage90_seed_sd_pct','width90_mean_return_pct','width90_seed_sd_return_pct','energy_score_mean_x100','energy_score_seed_sd_x100']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow(dict(zip(fields,[r['fold'],r['method'],float(r['coverage90_mean'])*100,float(r['coverage90_seed_std'])*100,
                                              float(r['width90_mean'])*100,float(r['width90_seed_std'])*100,float(r['energy_score_mean'])*100,float(r['energy_score_seed_std'])*100])))
    caption='''**Figure R5. Coverage, sharpness, and distribution score under positive innovation scaling.** Points show means over five separately trained, frozen TREFIN models; error bars show sample standard deviations across seeds. The original validation-selected mean shrinkage and point forecasts are retained. Fixed validation ES selects one positive scale per model by validation energy score; fixed validation 90% selects it to attain pooled 90% validation interval coverage. Both use 256 validation scenarios and freeze the scale for test evaluation. Causal rolling calibration uses the 126 most recent completed forecast origins, pooling their marginal conformity scores; an origin enters the window once all five outcomes have been realized. This scale updates sequentially during test evaluation. All methods reuse the original 256 test scenarios. Coverage is pooled over origins, leads, and assets; width is expressed in percentage points of return. Lower energy score is better. The horizontal reference marks nominal 90% coverage.
'''
    (a.output/'calibration_caption.md').write_text(caption,encoding='utf-8')
    provenance={'input':str(a.input.resolve()),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plotted_seed_count':5,'error_bars':'sample standard deviation, ddof=1',
                'methods':[m[0] for m in methods],'outputs':[str(basename.with_suffix('.'+s).resolve()) for s in ['png','svg','pdf']]}
    (a.output/'calibration_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
