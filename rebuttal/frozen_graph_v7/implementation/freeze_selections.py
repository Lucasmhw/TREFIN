"""Freeze all six validation-only selections before any final test execution."""
from pathlib import Path
import argparse,json
from protocol import VARIANTS,SEEDS,sha256,dump,selection_key,implementation_hashes,validate_frozen

def run(args):
    root=args.study_root.resolve();final=root/'frozen_selections.json'
    if final.exists():
        validate_frozen(root);print('Existing six-choice freeze verified.');return
    if (root/'runs').exists() and any((root/'runs').iterdir()):raise RuntimeError('Final run output exists before global selection freeze')
    records={};source=None;implementation=implementation_hashes();executed=0
    for fold in [1,2,3]:
        for variant in VARIANTS:
            directory=root/'search'/f'fold_{fold}'/variant
            sp=directory/'selection.json';rp=directory/'search_registration.json'
            selection=json.loads(sp.read_text());registration=json.loads(rp.read_text())
            assert selection['fold']==registration['fold']==fold and selection['variant']==registration['variant']==variant
            assert selection['test_access_count']==0 and not registration['test_evaluation_permitted']
            assert selection['registration_sha256']==sha256(rp)
            assert registration['implementation_sha256']==implementation
            if source is None:source=registration['source_sha256']
            assert source==registration['source_sha256']
            assert selection['executed_unique_candidates']==(4 if variant=='Full' else 3)
            for trial in selection['trials']:
                if trial['executed']:
                    checkpoint=directory/trial['candidate']/'checkpoints/best.pt'
                    assert sha256(checkpoint)==trial['checkpoint_sha256']
                    assert trial['registration_sha256']==sha256(rp)
                    assert trial['test_access_count']==0 and trial['checkpoint_reload']['passed']
            best=min(selection['trials'],key=lambda x:(x['calibrated_validation_energy_score'],x['order']))
            assert best['candidate']==selection['selected_candidate']
            assert sha256(root/selection['selected_checkpoint'])==selection['selected_checkpoint_sha256']
            records[selection_key(fold,variant)]={'selection_path':sp.relative_to(root).as_posix(),
              'selection_sha256':sha256(sp),'registration_path':rp.relative_to(root).as_posix(),
              'registration_sha256':sha256(rp),'selected_candidate':selection['selected_candidate']}
            executed+=selection['executed_unique_candidates']
    assert executed==21
    dump({'schema':'trefin-frozen-graph-v7-selections','status':'frozen_before_test',
      'folds':[1,2,3],'variants':VARIANTS,'seeds':SEEDS,'final_runs':30,'unique_search_fits':executed,
      'additional_seed_fits':24,'seed1_reuses_validation_selected_checkpoint':True,
      'implementation_sha256':implementation,'source_sha256':source,'selections':records,
      'test_used_for_selection':False},final)
    validate_frozen(root);print(json.dumps({'event':'all_selections_frozen','sha256':sha256(final),'final_runs':30}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);run(p.parse_args())
