"""Freeze minimal primary implementation files; never change upstream code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work/fresh_point_bundle'
OUT.mkdir(exist_ok=True)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selections = {
    'TimeSeriesLibrary': ['models/TimeXer.py','layers/SelfAttention_Family.py','layers/Embed.py','utils/masking.py','LICENSE','README.md'],
    'PyPOTS': ['pypots/nn/modules/timemixerpp/backbone.py','pypots/nn/modules/timemixerpp/layers.py','pypots/nn/modules/timemixerpp/__init__.py','pypots/nn/modules/revin/layers.py','pypots/nn/modules/revin/__init__.py','pypots/nn/modules/inception/layers.py','pypots/nn/modules/inception/__init__.py','pypots/nn/modules/transformer/attention.py','pypots/nn/modules/transformer/embedding.py','pypots/forecasting/timemixerpp/core.py','LICENSE','README.md'],
}
urls = {'TimeSeriesLibrary':'https://github.com/thuml/Time-Series-Library','PyPOTS':'https://github.com/WenjieDu/PyPOTS'}
manifest = {'schema':'trefin-fresh-point-implementations-v1','upstreams':{},'files':{}}
for name, files in selections.items():
    src = ROOT/'work/upstream'/name
    rev = subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
    record = {'url':urls[name],'revision':rev,'license':'MIT' if name=='TimeSeriesLibrary' else 'BSD-3-Clause','files':{}}
    for relative in files:
        dst = OUT/'upstream'/name/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src/relative,dst)
        record['files'][relative] = digest(dst)
    manifest['upstreams'][name] = record
for src, dst in [(ROOT/'work/scripts/fresh_official_baselines.py',OUT/'fresh_official_common_v1.py'),(ROOT/'work/scripts/fresh_point_baselines.py',OUT/'fresh_point_baselines.py')]:
    shutil.copy2(src,dst)
    manifest['files'][dst.name]=digest(dst)
manifest['data_reference']={'path_argument':'--data-bundle','file':'fold1_raw_returns.npz','sha256':json.loads((ROOT/'work/official_supplement_bundle/manifest.json').read_text())['data_sha256']}
manifest['timexer_standalone_model_content_identical'] = (ROOT/'work/upstream/TimeXer/models/TimeXer.py').read_text() == (OUT/'upstream/TimeSeriesLibrary/models/TimeXer.py').read_text()
manifest['timexer_standalone_revision'] = subprocess.check_output(['git','-C',str(ROOT/'work/upstream/TimeXer'),'rev-parse','HEAD'],text=True).strip()
manifest['timemixerpp_provenance'] = {'implementation':'PyPOTS maintained implementation, not author-repository reproduction','author_review_source':'https://anonymous.4open.science/r/TimeMixerPP','author_source_retrieval_status':'repository_expired on 2026-11-08','maintainer_reference':'https://github.com/WenjieDu/PyPOTS/issues/789','forecast_head':'unchanged upstream final/coarsest-scale head; no averaging of heads'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
(OUT/'requirements-point.txt').write_text('einops==0.8.1\nreformer-pytorch==1.4.4\naxial-positional-embedding==0.2.1\nlocal-attention==1.9.15\nproduct-key-memory==0.2.11\ncolt5-attention==0.11.1\n',encoding='utf8')
print(json.dumps({'bundle':str(OUT),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'upstreams':{k:v['revision'] for k,v in manifest['upstreams'].items()}}))
