"""Check the eight fold/model combinations before launching the frozen plan."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path('/root/autodl-tmp/trefin_rebuttal_20261108')
pkg=root/'crossfold_v3'
manifest=json.loads((pkg/'source_adaptation_manifest.json').read_text())
for row in manifest['records']:
    assert hashlib.sha256((pkg/row['updated_path']).read_bytes()).hexdigest()==row['updated_sha256']
plan=json.loads((pkg/'execution_plan.json').read_text())
assert plan['cwd']==str(root)
for job in plan['jobs']:
    assert all('\\' not in arg for arg in job['argv'])
    assert Path(job['argv'][1]).is_file()
jobs=[j for j in plan['jobs'] if j['seed']==1]
assert len(jobs)==8
out=root/'rebuttal/crossfold_v3_cuda_smoke_v2'
out.mkdir(parents=True,exist_ok=True)
def run(job):
    argv=list(job['argv'])
    argv[argv.index('--out')+1]=str(out/job['id'])
    argv+=['--smoke']
    log=out/(job['id']+'.log');started=time.time()
    with log.open('w') as handle:
        code=subprocess.call(argv,stdout=handle,stderr=subprocess.STDOUT)
    row={'id':job['id'],'returncode':code,'seconds':time.time()-started,'log':str(log)}
    print(json.dumps(row),flush=True)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(run,jobs))
report={'status':'passed' if all(r['returncode']==0 for r in rows) else 'failed','checks':rows,
        'execution_plan_sha256':hashlib.sha256((pkg/'execution_plan.json').read_bytes()).hexdigest()}
(out/'cuda_smoke_summary.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['status']=='passed',report
print('ALL_EIGHT_CUDA_SMOKES_PASSED',flush=True)
