"""Compact read-only status for the registered crossfold experiment plan."""
import json
from pathlib import Path

root=Path('/root/autodl-tmp/trefin_rebuttal_20261108')
status=json.loads((root/'rebuttal/crossfold_v3/scheduler/status.json').read_text())
print(json.dumps({key:status[key] for key in ['counts','gpu_memory_used_mib','gpu_utilization_percent','updated_utc']}))
for key,row in status['jobs'].items():
    if row['state'] not in ['running','failed']:continue
    path=Path(row['log']);lines=path.read_text(errors='replace').splitlines() if path.exists() else []
    events=[]
    for line in lines[-12:]:
        try:events.append(json.loads(line))
        except (ValueError,TypeError):pass
    last=events[-1] if events else {}
    print(key,row['state'],last.get('event','starting'),last.get('candidate',''),last.get('epoch',''))
    if row['state']=='failed':print('\n'.join(lines[-8:]))
