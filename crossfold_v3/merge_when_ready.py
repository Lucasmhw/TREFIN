"""Verify and compact each complete five-seed group without touching training."""
import argparse,datetime,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path

FAMILIES=['tedm','timerxl','timexer','timemixerpp']
SEEDS=[1,7,19,37,73]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();source=a.source.resolve()
    folder=root/'merge_manager';folder.mkdir(exist_ok=True)
    statefile=folder/'status.json';assert not statefile.exists(),'A merge manager has already started'
    states={f'fold_{f}_{m}':{'fold':f,'family':m,'state':'waiting'} for f in [2,3] for m in FAMILIES}
    state={'started_utc':now(),'source_script_sha256':sha(__file__),'groups':states}
    while True:
        scheduler=json.loads((root/'scheduler/status.json').read_text())
        for name,record in states.items():
            if record['state']!='waiting':continue
            f,m=record['fold'],record['family']
            # Require successful process exit as well as its final artifact.
            if not all(scheduler['jobs'][f'fold_{f}_{m}_seed_{s}']['state']=='complete' for s in SEEDS):continue
            if not all((root/f'fold_{f}_{m}_seed_{s}'/f'seed_{s}'/'result.json').is_file() for s in SEEDS):continue
            record.update(state='verifying',started_utc=now());state['updated_utc']=now();save(statefile,state)
            command=[sys.executable,str(source/'analysis/merge_crossfold_workers.py'),'--root',str(root),'--source',str(source),'--fold',str(f),'--family',m]
            try:
                with (folder/(name+'.log')).open('w') as log:
                    run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                record.update(state='complete' if run.returncode==0 else 'failed',returncode=run.returncode,finished_utc=now())
            except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),finished_utc=now())
            print(name,record['state'],flush=True)
        state['updated_utc']=now();save(statefile,state)
        if all(v['state']=='complete' for v in states.values()):break
        if any(v['state']=='failed' for v in states.values()):
            state['status']='merge_failure_requires_review';save(statefile,state);return
        time.sleep(30)
    archive=root/'crossfold_v3_compact_complete.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as z:
        for f in sorted((root/'merged').rglob('*')):
            if f.is_file():z.add(f,arcname=f.relative_to(root))
    state.update(status='all_eight_groups_verified',archive_file=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,finished_utc=now())
    save(statefile,state);print(json.dumps({k:v for k,v in state.items() if k!='groups'}),flush=True)

if __name__=='__main__':main()
