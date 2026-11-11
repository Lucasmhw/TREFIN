"""Reproduce the study figure from compact verified per-seed CSV, without forecasts."""
from pathlib import Path
import argparse,csv,hashlib,json
from analyze_frozen_graph import make_figure

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--analysis-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
source=a.analysis_dir/'paired_seed_differences.csv'
manifest=json.loads((a.analysis_dir/'manifest.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files'][source.name]
rows=[]
for row in csv.DictReader(source.open(encoding='utf-8')):
    row['fold']=int(row['fold']);row['seed']=int(row['seed'])
    for key in ['full','frozen_graph','difference_frozen_minus_full','display_scale']:row[key]=float(row[key])
    rows.append(row)
assert len({(r['fold'],r['seed']) for r in rows})==15
a.output_dir.mkdir(parents=True,exist_ok=True)
make_figure(rows,a.output_dir)
print('Reproduced paired figure from all matched seed records.')
