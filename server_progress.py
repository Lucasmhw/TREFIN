"""Compact read-only status of the registered baseline workers."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    jobs=[('TEDM',1,'fresh_tedm_b64'),('TimeMixer++',1,'fresh_timemixerpp_fast')]
    jobs += [(family,seed,f'{prefix}_seed_{seed}') for family,prefix in [('TEDM','fresh_tedm'),('TimeMixer++','fresh_timemixerpp')] for seed in [7,19,37,73]]
    completed=0
    for family,seed,name in jobs:
        result=a.root/name/f'seed_{seed}'/'result.json'
        log=a.root/(name+'.log')
        lines=log.read_text(errors='replace').splitlines() if log.exists() else []
        row=None
        for line in reversed(lines):
            try:row=json.loads(line);break
            except (ValueError,TypeError):continue
        if result.exists():
            try:
                value=json.loads(result.read_text());assert value['seed']==seed
                status='complete';completed+=1
            except (ValueError,AssertionError):status='writing_result'
        elif any('Traceback (most recent call last)' in line for line in lines[-20:]):status='error: '+lines[-1]
        elif row:
            status=row.get('event','running')
            if status=='epoch':status=f"candidate {row['candidate']}, epoch {row['epoch']}"
        else:status='starting'
        print(f'{family:12} seed {seed:2}: {status}')
    print(f'Completed registered workers: {completed}/{len(jobs)}')

if __name__=='__main__':main()
