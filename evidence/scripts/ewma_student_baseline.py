"""Validation-selected zero-mean multivariate Student-t / EWMA benchmark.

python ewma_student_baseline.py --data-dir path/data/raw/finmultitime_us_32 \
 --prepared-dir path/run/audit --output new_output --fold fold_1

Dependencies: numpy, scipy. Does not import or modify the model repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chi2


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_returns(data_dir, names):
    prices_by_date={}
    with open(data_dir/'panel.csv',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['asset'] in names:
                if row.get('available_at',row['date'])>row['date']:
                    raise ValueError('Late panel observations unsupported')
                prices_by_date.setdefault(row['date'],{})[row['asset']]=float(row['price'])
    dates=sorted(prices_by_date)
    price=np.array([[prices_by_date[d][a] for a in names] for d in dates],dtype=np.float64)
    factors={}
    with open(data_dir/'factors.csv',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('available_at',row['date'])>row['date']:
                raise ValueError('Late factor observations unsupported')
            factors[row['date']]=float(row['risk_free'])
    rf=np.array([factors[d] for d in dates])
    # In the prepared FinMultiTime artifact this column is ALREADY log(1+RF).
    # Applying log1p again would be a unit mismatch.
    raw=np.zeros_like(price)
    raw[1:]=np.log(price[1:]/price[:-1])
    returns=raw-rf[:,None]
    if not np.isfinite(returns).all():raise ValueError('Nonfinite returns')
    return dates,returns


def ewma_covariances(returns, initial, init_row, requested_origins, decay):
    positions={int(t):i for i,t in enumerate(requested_origins)}
    if len(positions)!=len(requested_origins) or min(positions)<=init_row:
        raise ValueError('Forecast origins must be unique and after training initialization')
    output=np.empty((len(positions),returns.shape[1],returns.shape[1]))
    current=initial.copy()
    identity=np.eye(len(current))
    for t in range(init_row+1,max(positions)+1):
        current=decay*current+(1-decay)*np.outer(returns[t],returns[t])
        if t in positions:
            ridge=1e-6*max(float(np.trace(current)/len(current)),1e-12)
            output[positions[t]]=current+ridge*identity
    return output


def base_random(sample_count, origins, horizon, assets, seed):
    rng=np.random.default_rng(seed)
    normal=rng.standard_normal((origins,horizon,sample_count,assets))
    uniform=rng.uniform(np.finfo(float).eps,1-np.finfo(float).eps,(origins,horizon,sample_count))
    return normal,uniform


def scenarios(covariance, degrees, normal, uniform):
    cholesky=np.linalg.cholesky(covariance)
    gaussian=np.einsum('bhsk,bnk->bhsn',normal,cholesky,optimize=True)
    # Cov(t_nu)=nu/(nu-2)*scale. Scale=(nu-2)/nu*EWMA covariance.
    radial=np.sqrt((degrees-2)/chi2.ppf(uniform,degrees))
    return gaussian*radial[...,None]


def energy_score(samples,target):
    # samples [B,H,S,N]; exact unordered-pair U-statistic for every origin/lead.
    b,h,s,n=samples.shape
    first=np.linalg.norm(samples-target[:,:,None,:],axis=-1).mean(axis=-1)
    second=np.empty((b,h))
    for i in range(b):
        for j in range(h):second[i,j]=pdist(samples[i,j]).sum()/(s*(s-1))
    return (first-second).mean(axis=1)


def metrics(samples,target):
    zero=np.zeros_like(target)
    error=zero-target
    mse=np.square(error).mean(axis=(1,2))
    mae=np.abs(error).mean(axis=(1,2))
    es=energy_score(samples,target)
    coverage={};width={}
    for level in [.50,.80,.90,.95,.99]:
        q=(1-level)/2
        lo,hi=np.quantile(samples,[q,1-q],axis=2)
        coverage[str(level)]=float(((target>=lo)&(target<=hi)).mean())
        width[str(level)]=float((hi-lo).mean())
    report={'mse':float(mse.mean()),'rmse':float(np.sqrt(mse.mean())),
            'mae':float(mae.mean()),'energy_score':float(es.mean()),
            'coverage':coverage,'interval_width':width}
    return report,{'mse':mse,'mae':mae,'energy_score':es}


def run(data_dir,prepared_dir,output,fold,validation_samples=64,test_samples=256,seed=271828):
    started=time.time()
    if (output/'result.json').exists():raise FileExistsError('Completed output already exists')
    output.mkdir(parents=True,exist_ok=True)
    mp=prepared_dir/'prepared_metadata.json'; ap=prepared_dir/'prepared_arrays.npz'
    metadata=json.loads(mp.read_text(encoding='utf-8'))
    with np.load(ap,allow_pickle=False) as f:
        all_origins=f['origins'].copy()
        # NPZ decompresses a whole array, but only training and validation slices
        # enter candidate evaluation. Test labels are not used until selection.
        splits={k:np.array(v,dtype=int) for k,v in metadata['split_indices'].items()}
        y=f['y']
        train_labels=y[splits['train']].astype(np.float64)
        validation_y=y[splits['val']].astype(np.float64)
        del y
    dates,returns=load_returns(data_dir,metadata['asset_names'])
    h=int(metadata['horizon']);n=len(metadata['asset_names'])
    train_times=(all_origins[splits['train'],None]+np.arange(1,h+1)).ravel()
    unique=np.unique(train_times)
    init_row=int(unique.max())
    actual_train=returns[all_origins[splits['train'],None]+np.arange(1,h+1)]
    actual_val=returns[all_origins[splits['val'],None]+np.arange(1,h+1)]
    train_error=float(np.abs(actual_train-train_labels).max())
    val_error=float(np.abs(actual_val-validation_y).max())
    if max(train_error,val_error)>2e-7:
        raise ValueError(f'Excess-log-return alignment mismatch: train {train_error} val {val_error}')
    initial=returns[unique].T@returns[unique]/len(unique)
    initial+=1e-6*np.trace(initial)/n*np.eye(n)
    val_origins=all_origins[splits['val']]
    normal,uniform=base_random(validation_samples,len(val_origins),h,n,seed)
    candidates=[]
    for decay in [.90,.94,.97,.99]:
        covariance=ewma_covariances(returns,initial,init_row,val_origins,decay)
        for degrees in [3,5,8,15,30]:
            sample=scenarios(covariance,degrees,normal,uniform)
            es=float(energy_score(sample,validation_y).mean())
            row={'decay':decay,'degrees_of_freedom':degrees,'validation_energy_score':es}
            candidates.append(row)
            print(json.dumps({'fold':fold,**row}),flush=True)
        (output/'validation_candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    best=min(candidates,key=lambda x:(x['validation_energy_score'],x['decay'],x['degrees_of_freedom']))
    selection={'selected':best,'selection_metric':'validation Energy Score',
               'decay_grid':[.90,.94,.97,.99],'degrees_grid':[3,5,8,15,30],
               'validation_samples':validation_samples,'test_samples':test_samples,
               'common_random_numbers':True,'random_seed':seed,'test_used_for_selection':False}
    (output/'frozen_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    del sample,covariance,normal,uniform
    with np.load(ap,allow_pickle=False) as f:
        test_y=f['y'][splits['test']].astype(np.float64)
    test_origins=all_origins[splits['test']]
    test_error=float(np.abs(returns[test_origins[:,None]+np.arange(1,h+1)]-test_y).max())
    if test_error>2e-7:raise ValueError('Test excess-log-return alignment mismatch')
    covariance=ewma_covariances(returns,initial,init_row,test_origins,best['decay'])
    normal,uniform=base_random(test_samples,len(test_origins),h,n,seed+1)
    sample=scenarios(covariance,best['degrees_of_freedom'],normal,uniform)
    result,per_origin=metrics(sample,test_y)
    np.savez_compressed(output/'test_predictions.npz',samples=sample.transpose(2,0,1,3).astype(np.float32),
                        mu=np.zeros_like(test_y,dtype=np.float32),y=test_y.astype(np.float32),
                        indices=splits['test'],origins=test_origins,covariance=covariance,
                        **{f'per_origin_{key}':value for key,value in per_origin.items()})
    report={'schema':'trefin-zero-mean-student-ewma-v1','fold':fold,'selection':selection,
            'test_metrics':result,'training_covariance':{
                'estimator':'uncentered second moment of unique purged training target dates',
                'observations':len(unique),'initialization_row':init_row,'initialization_date':dates[init_row],
                'ridge':'1e-6 times mean diagonal; covariance saved after each origin update'},
            'forecast_semantics':{
                'mean':'zero for all assets/leads',
                'covariance_update':'Q_t=lambda Q_(t-1)+(1-lambda)r_t r_t^T; observations only through t',
                'horizons':'same conditional marginal covariance at every lead; independent Student-t draws across leads',
                'student_scaling':'sqrt((nu-2)/chi2_nu) times N(0,Q), giving covariance Q',
                'return_units':'decimal excess log return; stored risk_free column already log(1+RF)'},
            'alignment_max_absolute_errors':{'train':train_error,'val':val_error,'test':test_error},
            'asset_order':metadata['asset_names'],'test_origins':len(test_origins),
            'source_hashes':{str(p):sha256(p) for p in [data_dir/'panel.csv',data_dir/'factors.csv',ap,mp]},
            'script_sha256':sha256(__file__),'seconds':time.time()-started}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'validation_candidates.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
    print(json.dumps({'fold':fold,'selected':best,'test':result,'seconds':report['seconds']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fold',required=True)
    p.add_argument('--validation-samples',type=int,default=64)
    p.add_argument('--test-samples',type=int,default=256)
    p.add_argument('--seed',type=int,default=271828)
    a=p.parse_args()
    run(a.data_dir,a.prepared_dir,a.output,a.fold,a.validation_samples,a.test_samples,a.seed)
