#!/usr/bin/env python3
"""Freeze minimal raw-return data and pinned upstream source for new comparisons."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repo=a.repo.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    from fconcord.data import load_panel_csv
    from fconcord.config import load_config
    ref=repo/'runs/protocols/finmultitime_us_rolling_3fold_5seed_v1/runs/fold_1_seed_1'
    cfg=load_config(ref/'config_resolved.yaml')
    panel_path=repo/'data/raw/finmultitime_us_32/panel.csv';factor_path=repo/'data/raw/finmultitime_us_32/factors.csv'
    provenance=json.loads((ref/'audit/data_provenance.json').read_text())
    assert digest(panel_path)==provenance['files']['panel']['sha256']
    assert digest(factor_path)==provenance['files']['factors']['sha256']
    panel=load_panel_csv(panel_path,factor_path,cfg['data'])
    metadata=json.loads((ref/'audit/prepared_metadata.json').read_text())
    assert list(panel['asset_names'])==metadata['asset_names']
    with np.load(ref/'audit/prepared_arrays.npz') as f:
        origins=f['origins'];archived_targets=f['y']
    reconstructed=np.stack([panel['returns'][o+1:o+6] for o in origins]).astype(np.float32)
    assert np.array_equal(reconstructed,archived_targets)
    splits={name:np.asarray(values,dtype=np.int64) for name,values in metadata['split_indices'].items()}
    first=max(1,int(origins[splits['train'][0]])-125);last=int(origins[splits['train'][-1]])+5
    assert first>=1 and last<int(origins[splits['val'][0]])
    fit=panel['returns'][first:last+1]
    location=fit.mean(axis=0);scale=fit.std(axis=0);assert np.all(scale>0)
    np.savez_compressed(out/'fold1_raw_returns.npz',returns=panel['returns'].astype(np.float64),origins=origins,targets=archived_targets,train_positions=splits['train'],val_positions=splits['val'],test_positions=splits['test'],location=location,scale=scale,dates=panel['dates'].astype('datetime64[ns]').astype(np.int64))
    manifest={'schema':'trefin-fresh-official-supplement-data-v1','description':'NEW task-trained comparisons, not reconstruction of historical separate-host results','fold':1,'history':126,'horizon':5,'assets':metadata['asset_names'],'split_sizes':{s:len(v) for s,v in splits.items()},'split_date_ranges':metadata['split_date_ranges'],'normalization':'per-asset mean and population SD fitted to unique raw training rows including purged training labels only','normalization_fit_rows_inclusive':[first,last],'panel_sha256':digest(panel_path),'factors_sha256':digest(factor_path),'reference_prepared_metadata_sha256':digest(ref/'audit/prepared_metadata.json'),'targets_match_archived_exactly':True,'data_sha256':digest(out/'fold1_raw_returns.npz'),'upstreams':{}}
    selections={'tedm':['models/networks.py','models/noise.py','models/precond.py','models/filters.py','training/sampler.py','training/scheduler.py','utils/__init__.py','utils/io.py','utils/seed.py','utils/stats.py','config/custom.yaml','config/exchange.yaml','LICENSE','README.md'], 'OpenLTM':['models/timer_xl.py','layers/Transformer_EncDec.py','layers/SelfAttention_Family.py','layers/Attn_Bias.py','layers/Attn_Projection.py','utils/__init__.py','utils/masking.py','LICENSE','README.md']}
    urls={'tedm':'https://gitlab.com/dlr-dw/tedm','OpenLTM':'https://github.com/thuml/OpenLTM'}
    for name,files in selections.items():
        source=a.upstream/name;revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        hashes={}
        for rel in files:
            target=out/'upstream'/name/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,target);hashes[rel]=digest(target)
        manifest['upstreams'][name]={'url':urls[name],'revision':revision,'files':hashes}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (out/'requirements-supplement.txt').write_text('einops==0.8.1\nomegaconf==2.3.0\nema-pytorch==0.2.1\n',encoding='utf8')
    print(json.dumps({'split_sizes':manifest['split_sizes'],'total_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'data_hash':manifest['data_sha256']}))


if __name__=='__main__':main()
