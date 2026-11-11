"""Prospectively register and execute validation search followed by selected fits."""
from pathlib import Path
import argparse,concurrent.futures,datetime,json,os,subprocess,sys,threading,time

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--study-root',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();study=args.study_root.resolve();impl=study/'implementation'
    sys.path.insert(0,str(impl))
    from protocol import CANDIDATES,SEEDS,VARIANTS,dump,sha256,implementation_hashes
    if (study/'orchestration_registration.json').exists():raise FileExistsError('Preserve registered run; inspect before any restart')
    assert json.loads((study/'test_semantics_gpu.json').read_text())['status']=='passed'
    assert json.loads((study/'preflight_gpu.json').read_text())['status']=='passed'
    assert not (study/'search').exists() and not (study/'runs').exists()
    registration={'schema':'trefin-frozen-graph-v7-prospective-orchestration','registered_utc':utc(),
      'folds':[1,2,3],'seeds':SEEDS,'variants':VARIANTS,'candidate_order':CANDIDATES,
      'selection_seed':1,'epoch_ceiling':20,'patience':6,'batch_size':64,
      'training_scenarios':16,'validation_scenarios':64,'test_scenarios':256,
      'checkpoint_selection':'uncalibrated validation ES, original trainer min_delta=1e-6',
      'configuration_selection':'minimum validation ES after positive mean calibration; ties in registered order',
      'mean_calibration':{'split':'validation','bounds':[.05,1.0],'tolerance':1e-5,'innovations_unchanged':True},
      'FrozenGraph':'A_h=A_0 in transition and decoder; graph supervision zero; augmented gain [[m_CC,m_CA],[0,1]]',
      'duplicate':'FrozenGraph slow_graph aliases base; no reallocation of unused slot',
      'unique_search_fits':21,'additional_selected_fits':24,'final_evaluations':30,
      'all_six_selections_frozen_before_any_test':True,'selected_seed1_checkpoint_reused':True,
      'implementation_sha256':implementation_hashes(),'orchestrator_sha256':sha256(__file__),
      'input_equivalence_sha256':sha256(study/'input_equivalence.json'),
      'prepared_verification_sha256':sha256(study/'prepared_verification.json'),
      'gpu_semantics_sha256':sha256(study/'test_semantics_gpu.json'),
      'preflight_sha256':sha256(study/'preflight_gpu.json'),
      'execution':{'parallel_workers':args.workers,'threads_per_worker':2,'device':'cuda'},
      'reporting':'All 30 selected evaluations; float64 per-origin point losses and unbiased-pair Monte Carlo ES; five-seed sample SD; same target/origin checks',
      'reference_cohort':'New selected Full/FrozenGraph study; does not replace original forecasting/theory checkpoints'}
    dump(registration,study/'orchestration_registration.json')
    env=os.environ.copy();env.update(PYTHONPATH=str(study/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    common=['--root',str(args.root.resolve()),'--study-root',str(study),'--prepared-root',str(study/'prepared'),'--device','cuda','--threads','2']
    lock=threading.Lock();state={'registered_utc':registration['registered_utc'],'phase':'search','jobs':{},'failed':False}
    def save():dump(state,study/'orchestration_status.json')
    save()
    def job(name,script,extra):
        with lock:state['jobs'][name]={'status':'running','started_utc':utc()};save()
        cmd=[sys.executable,str(impl/script),*common,*extra]
        with (study/'logs'/f'{name}.log').open('wb') as out:
            code=subprocess.run(cmd,cwd=args.root,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
        with lock:
            state['jobs'][name].update(status='complete' if code==0 else 'failed',exit_code=code,ended_utc=utc())
            if code:state['failed']=True
            save()
        print(json.dumps({'event':'job_finished','name':name,'exit_code':code,'utc':utc()}),flush=True)
        return code
    def batch(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            outcomes=list(ex.map(lambda spec:job(*spec),jobs))
        if any(outcomes):raise RuntimeError('One or more jobs failed; outputs preserved and next phase blocked')
    jobs=[(f'search_f{f}_{v}','search_validation.py',['--fold',str(f),'--variant',v]) for f in [1,2,3] for v in VARIANTS]
    batch(jobs)
    subprocess.run([sys.executable,str(impl/'freeze_selections.py'),'--study-root',str(study)],env=env,check=True)
    state.update(phase='selected_evaluation',selection_frozen_utc=utc());save()
    jobs=[(f'final_f{f}_s{s}_{v}','run_selected.py',['--fold',str(f),'--seed',str(s),'--variant',v]) for s in SEEDS for f in [1,2,3] for v in VARIANTS]
    batch(jobs)
    state.update(phase='complete',ended_utc=utc());save()
    print(json.dumps({'event':'study_complete','final_evaluations':30,'utc':utc()}),flush=True)

if __name__=='__main__':main()
