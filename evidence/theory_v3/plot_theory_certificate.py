"""R9: completed global-bound and uniform local-tube evidence, without fitting."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('work/theory_v3/local_tube_results_v2'))
    p.add_argument('--validation',type=Path,default=Path('work/theory_v3/validation/validation_summary.json'))
    p.add_argument('--output',type=Path,default=Path('work/figures'))
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    summary=json.loads((a.results/'summary.json').read_text())
    checks=json.loads(a.validation.read_text())
    assert summary['runs']==15 and checks['passed'] and checks['totals']['runs']==15
    assert checks['bound_script_sha256']==summary['protocol']['bound_script_sha256']
    files=sorted(a.results.glob('fold_*_seed_*.json'))
    runs=[json.loads(f.read_text()) for f in files];assert len(runs)==15
    for fold in ['1','2','3']:
        for r in summary['folds'][fold]['radii']:
            assert r['certified_five_step_count']==r['trajectories']==160
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.5,
                         'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.7,'xtick.labelsize':9,'ytick.labelsize':9,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(14.6,5.2))
    colors=['#0072B2','#D55E00','#009E73'];gray='#7D8791'
    global_rows=[];lead_rows=[];radius_rows=[]
    for fold in [1,2,3]:
        rr=sorted([r for r in runs if r['fold']==fold],key=lambda r:r['seed'])
        for offset,r in zip(np.linspace(-.18,.18,5),rr):
            x=fold+offset;old=r['registered_domain_original_radius'];new=r['registered_domain_improved_radius']
            axes[0].plot([x,x],[new,old],color='#C6CCD1',lw=1.2,zorder=1)
            axes[0].scatter(x,old,s=23,c=gray,marker='o',label='Submitted global bound' if fold==1 and offset==-.18 else None,zorder=3)
            axes[0].scatter(x,new,s=25,c='#176B87',marker='s',label='Tighter global bound' if fold==1 and offset==-.18 else None,zorder=3)
            global_rows.append({'fold':fold,'seed':r['seed'],'original_radius':old,'tighter_radius':new})
        primary=next(r for r in summary['folds'][str(fold)]['radii'] if r['concept_radius']==.1)
        values=[v['maximum'] for v in primary['h_step_amplification_bounds']]
        axes[1].plot(range(1,6),values,'o-',color=colors[fold-1],lw=1.8,ms=4.5,label=f'Fold {fold}')
        for h,v in enumerate(primary['h_step_amplification_bounds'],1):
            lead_rows.append({'fold':fold,'lead':h,'radius_C':.1,'radius_A':.05,**v})
        rad=summary['folds'][str(fold)]['radii'];xs=[r['concept_radius'] for r in rad]
        ys=[r['h_step_amplification_bounds'][-1]['maximum'] for r in rad]
        axes[2].plot(xs,ys,'o-',color=colors[fold-1],lw=1.8,ms=4.5,label=f'Fold {fold}')
        for r in rad:
            radius_rows.append({'fold':fold,'radius_C':r['concept_radius'],'radius_A':r['graph_radius'],
                                'certified_trajectories':r['certified_five_step_count'],
                                **r['h_step_amplification_bounds'][-1]})
    axes[0].set_title('(a) Registered-domain sensitivity',loc='left',fontweight='bold')
    axes[0].set_ylabel('Spectral radius of global bound matrix')
    axes[0].set_xticks([1,2,3],['Fold 1','Fold 2','Fold 3']);axes[0].set_ylim(0,66)
    axes[0].axhline(1,color='#87939B',ls=':',lw=1)
    axes[0].legend(frameon=False,fontsize=8.3,loc='upper left')
    axes[1].set_title('(b) Five-step neighborhood bound',loc='left',fontweight='bold')
    axes[1].set_xlabel('Forecast lead');axes[1].set_xticks(range(1,6))
    axes[1].set_ylabel('Maximum certified amplification')
    axes[1].set_ylim(.8,4.4);axes[1].legend(frameon=False,loc='upper left',fontsize=9)
    axes[1].text(.04,.04,r'$R_C=0.1,\ R_A=0.05$',transform=axes[1].transAxes,fontsize=10,color='#4B5563')
    axes[2].set_title('(c) Complete neighborhood-radius audit',loc='left',fontweight='bold')
    axes[2].set_xscale('log');axes[2].set_xticks([.001,.01,.1,1],['0.001','0.01','0.1','1'])
    axes[2].set_xlabel(r'Concept-neighborhood radius $R_C$')
    axes[2].set_ylabel('Maximum certified five-step amplification')
    axes[2].set_ylim(.8,4.4);axes[2].axvline(.1,color='#A7B0B8',ls=':',lw=1)
    axes[2].text(.1,4.19,'Primary radius',ha='center',fontsize=8.5,color='#66727C')
    for ax in axes:
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E7EBEE',lw=.7);ax.tick_params(length=3)
    fig.suptitle('Finite-horizon sensitivity with explicit neighborhoods',x=.065,y=.965,ha='left',fontsize=15,fontweight='bold')
    fig.text(.065,.91,'15 archived models · 32 held-out origins per fold · all five forecast leads',fontsize=10,color='#4B5563')
    fig.text(.065,.095,r'Neighborhoods use full-block Frobenius radii $R_C$ and $R_A=R_C/2$; initial errors are at most $0.1$ times these radii.',fontsize=9,color='#4B5563')
    fig.text(.065,.05,'Panels (b,c): uniform analytic bounds in the weighted block-max norm; all 480 trajectories satisfy the five-step containment criterion.',fontsize=9,color='#4B5563')
    fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.25,wspace=.40)
    base=a.output/'R9_finite_horizon_certificate'
    for ext in ['png','svg','pdf']:fig.savefig(base.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    for name,rows in [('global_bounds',global_rows),('leadwise_bounds',lead_rows),('radius_audit',radius_rows)]:
        with (a.output/f'R9_{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    caption='''**Figure R9. Global sensitivity constants and uniform finite-horizon neighborhood bounds.** (a) Original and analytically tightened registered-domain sensitivity radii for all 15 frozen checkpoints; each connected pair is one fold-seed model. (b) At each forecast lead, the maximum certified amplification across five seeds and 32 evenly spaced held-out origins in each fold, using concept/graph Frobenius neighborhood radii 0.1/0.05. (c) The corresponding maximum five-step bound for every prespecified concept radius, with graph radius equal to half the concept radius. Nominal trajectories are recomputed in float64 with the observed market state, fitted parameters, and lead schedule fixed. The norm is the maximum of the concept and graph Frobenius errors divided by their neighborhood radii. Analytic block bounds hold uniformly throughout each specified neighborhood; their finite products propagate initial errors bounded by one tenth of the radii. All 480 trajectories satisfy the containment criterion at every audited radius. The registered-domain bounds in (a) and neighborhood bounds in (b,c) have their respective stated domains.\n'''
    (a.output/'R9_finite_horizon_certificate_caption.md').write_text(caption,encoding='utf-8')
    inputs=[a.results/'summary.json',a.results/'protocol.json',a.validation,*files]
    manifest={'figure':'R9_finite_horizon_certificate','script_sha256':sha(__file__),
              'inputs':[{'path':str(x.resolve()),'sha256':sha(x)} for x in inputs],
              'outputs':[{'path':str(base.with_suffix('.'+ext).resolve()),'sha256':sha(base.with_suffix('.'+ext))} for ext in ['png','svg','pdf']],
              'all_checks_passed':True,'all_numeric_values_from_completed_analyses':True}
    (a.output/'R9_finite_horizon_certificate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figure':str(base.resolve()),'formats':['png','svg','pdf']}))


if __name__=='__main__':main()
