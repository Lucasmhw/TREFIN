"""Execute a frozen list of independent runs within an explicit GPU budget.

This scheduler never reads a forecast score and never modifies training settings.
Run in a detached Linux process. Status is written atomically for read-only polls.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=10)
    p.add_argument('--reservation-budget-mib',type=int,default=75000)
    p.add_argument('--free-headroom-mib',type=int,default=4000)
    a=p.parse_args();a.state.mkdir(parents=True,exist_ok=True)
    lock=(a.state/'scheduler.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(a.plan.read_text())
    assert plan['cwd'].startswith('/') and '\\' not in plan['cwd']
    jobs=plan['jobs'];assert len(jobs)==40 and len({j['id'] for j in jobs})==40
    assert all('\\' not in arg for j in jobs for arg in j['argv'])
    assert all(Path(j['argv'][1]).is_file() for j in jobs)
    state_path=a.state/'status.json'
    if state_path.exists():raise RuntimeError('Existing scheduler state: inspect live processes before resuming')
    states={j['id']:{'state':'queued','fold':j['fold'],'seed':j['seed'],'family':j['family']} for j in jobs}
    processes={};logs={};started=utc()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    reserved=0
    while True:
        for job in jobs:
            jid=job['id']
            if jid not in processes:continue
            process=processes[jid];code=process.poll()
            if code is None:continue
            logs.pop(jid).close();del processes[jid];reserved-=job['reserved_gpu_mib']
            ok=code==0 and Path(job['result']).is_file() and Path(job['summary']).is_file()
            states[jid].update(state='complete' if ok else 'failed',exit_code=code,finished_utc=utc())
            print(json.dumps({'event':states[jid]['state'],'id':jid,'exit_code':code}),flush=True)
        try:
            gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
            free,used,util=[int(x.strip()) for x in gpu_text.split(',')]
        except Exception:
            free,used,util=0,None,None
        for job in jobs:
            jid=job['id']
            if states[jid]['state']!='queued':continue
            need=job['reserved_gpu_mib']
            if len(processes)>=a.max_workers or reserved+need>a.reservation_budget_mib:continue
            if free<need+a.free_headroom_mib:continue
            logfile=Path(job['log']);logfile.parent.mkdir(parents=True,exist_ok=True)
            handle=logfile.open('x')
            process=subprocess.Popen(job['argv'],cwd=plan['cwd'],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            processes[jid]=process;logs[jid]=handle;reserved+=need;free-=need
            states[jid].update(state='running',pid=process.pid,started_utc=utc(),log=str(logfile))
            print(json.dumps({'event':'started','id':jid,'pid':process.pid}),flush=True)
        counts={kind:sum(row['state']==kind for row in states.values()) for kind in ['queued','running','complete','failed']}
        status={'schema':'trefin-crossfold-scheduler-v3','plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                'started_utc':started,'updated_utc':utc(),'scheduler_pid':os.getpid(),'counts':counts,
                'gpu_memory_used_mib':used,'gpu_utilization_percent':util,'reserved_gpu_mib':reserved,'jobs':states}
        dump(state_path,status)
        if counts['queued']==0 and counts['running']==0:
            print(json.dumps({'event':'scheduler_finished','counts':counts}),flush=True)
            return 0 if counts['failed']==0 else 1
        time.sleep(10)


if __name__=='__main__':raise SystemExit(main())
