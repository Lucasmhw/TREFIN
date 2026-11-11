"""Read-only bounded wait for a scheduler completion/failure change."""
import argparse,json,time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--completed',type=int,required=True)
p.add_argument('--seconds',type=float,default=55)
a=p.parse_args()
assert 0<a.seconds<=55
started=time.monotonic()
while True:
    status=json.loads((a.root/'scheduler/status.json').read_text())
    counts={name:sum(j['state']==name for j in status['jobs'].values())
            for name in ['complete','running','queued','failed']}
    if counts['complete']!=a.completed or counts['failed'] or time.monotonic()-started>=a.seconds:
        print(json.dumps({'counts':counts,'updated_utc':status.get('updated_utc'),
                          'wait_seconds':round(time.monotonic()-started,2)}),flush=True)
        break
    time.sleep(min(2.,max(.01,a.seconds-(time.monotonic()-started))))
