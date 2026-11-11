"""Build exact paired TREFIN losses, reusing the immutable fold-1 cache."""
import argparse,shutil,sys
from pathlib import Path
import numpy as np
from common import SEEDS,METRICS,sha,read,dump,target_sha,load_npz,close,metric_summary

PROTOCOLS={'before':'finmultitime_us_rolling_3fold_5seed_v1',
           'after':'finmultitime_us_calibrated_rolling_3fold_5seed_v1'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--fold1-cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repo=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    from fconcord.statistical_tests import per_origin_energy_score
    cache_summary={'schema':'trefin-crossfold-reference-cache-v3','folds':{},
        'cache_script_sha256':sha(__file__),
        'pairwise_implementation_sha256':sha(repo/'src/fconcord/statistical_tests.py'),
        'precision':'Paired calibrated MSE/MAE/ES use float64 source arrays and exact all-pairs U-statistic ES. Descriptive metrics retain archived summary reductions; absent MSE is derived from each seed RMSE squared.'}
    for fold in [1,2,3]:
        directory=a.out/f'fold_{fold}';directory.mkdir(exist_ok=True)
        manifest=read(a.data_root/f'fold_{fold}/manifest.json')
        data_path=a.data_root/f'fold_{fold}'/manifest['data_file']
        assert sha(data_path)==manifest['data_sha256'];data=load_npz(data_path)
        positions=data['test_positions'];raw_origins=data['origins'][positions]
        expected_targets=data['targets'][positions];expected_hash=target_sha(expected_targets)
        assert expected_hash==manifest['test_targets_float32_bytes_sha256']
        rows={s:{'seed':s} for s in SEEDS};sources=[]
        for stage,protocol in PROTOCOLS.items():
            for seed in SEEDS:
                run=repo/'runs/protocols'/protocol/'runs'/f'fold_{fold}_seed_{seed}'
                summary_file=run/'run_summary.json';reported=read(summary_file)['test_metrics']
                mapped={m:reported[m] for m in ['rmse','mae','energy_score','msic','msir']}
                mapped.update(mse=reported.get('mse',reported['rmse']**2),coverage_90=reported['coverage']['0.9'],width_90=reported['interval_width']['0.9'])
                assert set(mapped)==set(METRICS);rows[seed][stage]=mapped
                sources.append({'seed':seed,'stage':stage,'run_relative_to_repo':run.relative_to(repo).as_posix(),'run_summary_sha256':sha(summary_file)})
                if stage!='after':continue
                prediction=run/'predictions/test_predictions.npz';source_hash=sha(prediction)
                cache_json=directory/f'seed_{seed}.json';cache_npz=directory/f'seed_{seed}.npz'
                if fold==1 and not cache_json.exists():
                    for suffix in ['json','npz']:shutil.copy2(a.fold1_cache/f'seed_{seed}.{suffix}',directory/f'seed_{seed}.{suffix}')
                if cache_json.exists() and cache_npz.exists():
                    metadata=read(cache_json)
                    assert metadata['source_sha256']==source_hash
                    assert metadata['loss_sha256']==sha(cache_npz)
                    assert metadata['target_hash']==expected_hash
                    arrays=load_npz(cache_npz);assert np.array_equal(arrays['indices'],positions)
                    with np.load(prediction,allow_pickle=False) as f:
                        mean=f['mu'].astype(np.float64);y=f['y'];assert np.array_equal(f['indices'],positions)
                    assert np.array_equal(y,expected_targets)
                    close(arrays['mse'],np.mean((mean-y.astype(np.float64))**2,axis=(1,2)),1e-14)
                    close(arrays['mae'],np.mean(np.abs(mean-y.astype(np.float64)),axis=(1,2)),1e-14)
                    close(float(arrays['es'].mean()),metadata['energy_score_recomputed'],1e-14)
                else:
                    with np.load(prediction,allow_pickle=False) as f:
                        mean=f['mu'].astype(np.float64);y32=f['y'];y=y32.astype(np.float64)
                        indices=f['indices'];samples=f['samples'].astype(np.float64)
                    assert np.array_equal(indices,positions) and np.array_equal(y32,expected_targets)
                    assert samples.shape==(256,len(positions),5,32)
                    es=per_origin_energy_score(samples,y)
                    difference=abs(float(es.mean())-reported['energy_score']);assert difference<1e-7
                    arrays={'mse':((mean-y)**2).mean((1,2)),'mae':np.abs(mean-y).mean((1,2)),
                            'es':es,'indices':indices}
                    np.savez_compressed(cache_npz,**arrays)
                    metadata={'schema':'trefin-float64-exact-ES-reference-v1','fold':fold,'seed':seed,
                        'source_relative_to_repo':prediction.relative_to(repo).as_posix(),
                        'source_sha256':source_hash,'loss_sha256':sha(cache_npz),'target_hash':expected_hash,
                        'energy_score_recomputed':float(es.mean()),'difference_from_archived_float32_reduction':difference,
                        'scenarios':256,'origins':len(y),'horizon':5,'assets':32,
                        'pairwise_method':'scipy.spatial.distance.pdist in float64, all unordered pairs, divisor S*(S-1)'}
                    dump(cache_json,metadata);del samples
                sources[-1].update(prediction_sha256=source_hash,cache_json_sha256=sha(cache_json),cache_npz_sha256=sha(cache_npz),targets_and_indices_verified=True)
                print({'event':'reference_verified','fold':fold,'seed':seed,'origins':len(positions)},flush=True)
        origin_record={'fold':fold,'indices':positions.tolist(),'raw_origins':raw_origins.tolist(),
                       'origin_dates_ns':data['dates'][raw_origins].tolist(),'target_hash':expected_hash,
                       'data_manifest_sha256':sha(a.data_root/f'fold_{fold}/manifest.json')}
        dump(directory/'origin_series.json',origin_record)
        fold_summary={'fold':fold,'seeds':SEEDS,'runs':[rows[s] for s in SEEDS],
                      'metrics':metric_summary([rows[s] for s in SEEDS]),'sources':sources,
                      'test_origins':len(positions),'target_hash':expected_hash,
                      'origin_series_sha256':sha(directory/'origin_series.json')}
        dump(directory/'reference_metrics.json',fold_summary)
        cache_summary['folds'][str(fold)]={'reference_metrics_sha256':sha(directory/'reference_metrics.json'),
                                         'test_origins':len(positions),'target_hash':expected_hash}
    dump(a.out/'reference_manifest.json',cache_summary)
    print({'status':'reference_cache_complete','folds':[1,2,3],'seeds':SEEDS})


if __name__=='__main__':main()
