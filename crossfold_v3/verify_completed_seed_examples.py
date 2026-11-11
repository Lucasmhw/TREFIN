"""Read-only early check of completed seed-1 artifacts while other runs continue."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source/'analysis'))
    from common import read,load_npz,dump
    from merge_crossfold_workers import validate_seed
    status=read(a.root/'scheduler/status.json');checks=[]
    for fold in [2,3]:
        manifest=read(a.source/f'data/fold_{fold}/manifest.json')
        data=load_npz(a.source/f'data/fold_{fold}'/manifest['data_file'])
        for family in ['tedm','timerxl','timexer','timemixerpp']:
            name=f'fold_{fold}_{family}_seed_1'
            if status['jobs'][name]['state']!='complete':continue
            worker=a.root/name;protocol=read(worker/'protocol.json')
            _,record=validate_seed(worker,1,fold,family,protocol,data,manifest,a.root)
            checks.append({'fold':fold,'family':family,'seed':1,'passed':True,'record':record})
    dump(a.root/'early_completed_examples_verification.json',{'status':'passed','scope':'Completed seed-1 artifacts only; not a complete crossfold result','examples_checked':len(checks),'checks':checks})
    print(json.dumps({'status':'passed','completed_examples':[(r['fold'],r['family']) for r in checks]}))

if __name__=='__main__':main()
