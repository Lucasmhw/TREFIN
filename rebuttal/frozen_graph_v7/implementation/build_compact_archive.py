"""Verify the complete 30-run study and archive compact evidence, never raw forecasts."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile,zipfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def npz_shape(path,member):
    import numpy as np
    with zipfile.ZipFile(path) as archive,archive.open(member+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError(f'Unsupported NPY header version {version}: {path}/{member}')
    return list(shape),str(dtype)

def run(args):
    study=args.study_root.resolve();repository=args.repository_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(f'Archive exists; preserve it: {output}')
    sys.path.insert(0,str(study/'implementation'))
    from protocol import validate_frozen,VARIANTS,SEEDS
    frozen,frozen_sha=validate_frozen(study)
    assert read(study/'orchestration_status.json')['phase']=='complete','Whole study must finish before archival'
    assert not read(study/'orchestration_status.json')['failed']
    for name in ['test_semantics_gpu.json','preflight_gpu.json']:assert read(study/name)['status']=='passed'
    files={};raw=[];verified=[];prepared=[]
    def include(path,name=None):
        path=Path(path)
        if path.is_symlink():raise ValueError(f'Symlink not permitted: {path}')
        if not path.is_file():raise FileNotFoundError(path)
        name=name or path.relative_to(study).as_posix()
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert name not in files or files[name]['path']==path
        files[name]={'path':path,'bytes':path.stat().st_size,'sha256':sha(path)}
    # Explicit top-level allowlist deliberately excludes SSH/authentication helpers.
    top=['orchestration_registration.json','orchestration_status.json','frozen_selections.json',
      'input_equivalence.json','prepared_verification.json','test_semantics_gpu.json','preflight_gpu.json',
      'orchestrate_study.py','restore_verified_inputs.py','verify_compact_inputs.py']
    for name in top:include(study/name)
    if (study/'original_source_manifest.json').is_file():include(study/'original_source_manifest.json')
    # Final analysis is required and its own manifest must verify before archival.
    analysis=study/'analysis'
    analysis_manifest=read(analysis/'manifest.json')
    assert analysis_manifest['status']=='complete'
    assert analysis_manifest['global_selection_sha256']==frozen_sha
    analyzer=study/'analysis_tools/analyze_frozen_graph.py'
    assert sha(analyzer)==analysis_manifest['script_sha256']
    include(analyzer)
    include(study/'analysis_tools/render_frozen_graph_figure.py')
    for relative,digest in analysis_manifest['files'].items():
        path=(analysis/relative).resolve()
        assert path.is_relative_to(analysis.resolve()) and path.is_file()
        assert sha(path)==digest,f'Analysis artifact changed: {relative}'
    for path in sorted(analysis.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:include(path)
    for path in sorted((study/'implementation').iterdir()):
        if path.is_file() and path.suffix in {'.py','.md','.json','.patch'}:include(path)
    for path in sorted((study/'src').rglob('*.py')):include(path)
    original=repository/'src/fconcord'
    for path in sorted(original.rglob('*.py')):
        include(path,'original_source/fconcord/'+path.relative_to(original).as_posix())
    for fold in [1,2,3]:
        directory=study/'prepared'/f'fold_{fold}'
        include(directory/'prepared_metadata.json')
        prepared.append({'path':f'prepared/fold_{fold}/prepared_arrays.npz','sha256':sha(directory/'prepared_arrays.npz'),
          'bytes':(directory/'prepared_arrays.npz').stat().st_size,'retained_on_server':True,'included':False})
    unique_search=0
    for fold in [1,2,3]:
        for variant in VARIANTS:
            directory=study/'search'/f'fold_{fold}'/variant
            selection=read(directory/'selection.json')
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in {'.json','.yaml','.csv'}:include(path)
            for trial in selection['trials']:
                if not trial['executed']:continue
                unique_search+=1;td=directory/trial['candidate']
                assert sha(td/'checkpoints/best.pt')==trial['checkpoint_sha256']
                assert read(td/'complete.json')['checkpoint_sha256']==trial['checkpoint_sha256']
                for path in sorted(td.rglob('*')):
                    if not path.is_file():continue
                    if path.suffix in {'.json','.yaml','.csv','.log'} or path.relative_to(td).as_posix()=='checkpoints/best.pt':include(path)
    assert unique_search==21
    count=0
    for fold in [1,2,3]:
        for seed in SEEDS:
            for variant in VARIANTS:
                directory=study/'runs'/f'fold_{fold}_seed_{seed}'/variant
                complete=read(directory/'complete.json');count+=1
                assert complete['schema']=='trefin-frozen-graph-v7-complete'
                assert (complete['fold'],complete['seed'],complete['variant'])==(fold,seed,variant)
                assert complete['global_selection_sha256']==frozen_sha
                if variant=='FrozenGraph':assert complete['graph_max_abs_departure_from_initial']==0
                for relative,digest in complete['artifact_sha256'].items():
                    assert sha(directory/relative)==digest,f'Artifact modified: {directory/relative}'
                include(directory/'complete.json')
                for path in sorted(directory.rglob('*')):
                    if not path.is_file() or path.name=='complete.json':continue
                    relative=path.relative_to(directory).as_posix()
                    if relative.startswith('predictions/'):
                        assert path.suffix=='.npz'
                        shape,dtype=npz_shape(path,'samples')
                        assert shape[0]==(256 if path.name=='test_predictions.npz' else 64) and shape[2:]==[5,32]
                        raw.append({'path':path.relative_to(study).as_posix(),'bytes':path.stat().st_size,
                          'sha256':sha(path),'samples_shape':shape,'samples_dtype':dtype,'included':False,'retained_on_server':True})
                    elif path.suffix in {'.json','.yaml','.csv','.log'} or relative in {'checkpoints/best.pt','tables/origin_losses.npz'}:
                        include(path)
                verified.append({'fold':fold,'seed':seed,'variant':variant,'complete_sha256':sha(directory/'complete.json'),
                  'verified_original_artifact_count':len(complete['artifact_sha256'])})
    assert count==30 and len(raw)==60
    # Only registered scientific process logs, not connection/session logs.
    for path in sorted((study/'logs').glob('*.log')):
        if path.name.startswith(('search_f','final_f')):include(path)
    inventory_path=study/'packaging/compact_inventory.json'
    inventory={'schema':'trefin-frozen-graph-v7-compact-inventory','status':'verified_complete',
      'global_selection_sha256':frozen_sha,'unique_search_fits':unique_search,'final_runs':count,
      'verified_runs':verified,'raw_predictions_retained_server':raw,'prepared_arrays_retained_server':prepared,
      'included_files':{name:{k:v for k,v in item.items() if k!='path'} for name,item in sorted(files.items())},
      'omitted':'Full prediction NPZs, prepared arrays, last.pt checkpoints and authentication/connection helpers. Best checkpoints, all final compact scientific records, and every unique search record are included.',
      'archive_builder_sha256':sha(__file__)}
    dump(inventory,inventory_path);include(inventory_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists():raise FileExistsError(temporary)
    with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
        for name,item in sorted(files.items()):archive.add(item['path'],arcname='frozen_graph_v7/'+name,recursive=False)
    with tarfile.open(temporary,'r:gz') as archive:
        members=archive.getmembers();assert len(members)==len(files)
        for member in members:
            assert member.isfile()
            relative=member.name.removeprefix('frozen_graph_v7/')
            stream=archive.extractfile(member);digest=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==files[relative]['sha256']
    temporary.replace(output)
    receipt={'status':'passed','archive':output.name,'archive_sha256':sha(output),'archive_bytes':output.stat().st_size,
      'member_count':len(files),'inventory_sha256':sha(inventory_path),'all_member_hashes_verified':True,
      'all_30_final_runs_original_artifacts_verified':True,'all_21_search_checkpoints_verified':True,
      'final_analysis_manifest_and_files_verified':True,
      'raw_forecast_files_in_archive':0,'authentication_helpers_in_archive':0}
    dump(receipt,output.with_name(output.name+'.receipt.json'));print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--repository-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
