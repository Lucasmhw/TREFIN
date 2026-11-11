"""Shared deterministic analysis and immutable-record checks."""
import hashlib,json,math
from pathlib import Path
import numpy as np

SEEDS=[1,7,19,37,73]
FAMILIES={'tedm':'TEDM','timerxl':'Timer-XL','timexer':'TimeXer','timemixerpp':'TimeMixer++'}
METRICS=['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def target_sha(y):return hashlib.sha256(np.ascontiguousarray(y,dtype=np.float32).tobytes()).hexdigest()
def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def normalized_protocol(protocol):return {k:v for k,v in protocol.items() if k!='seeds'}
def close(x,y,tol=1e-12):
    assert np.allclose(x,y,rtol=0,atol=tol),(x,y)
def metric_summary(rows):
    assert [r['seed'] for r in rows]==SEEDS
    result={}
    for stage in ['before','after']:
        result[stage]={}
        for metric in METRICS:
            values=np.asarray([r[stage][metric] for r in rows],dtype=np.float64)
            assert np.isfinite(values).all()
            result[stage][metric]={'mean':float(values.mean()),'sample_std':float(values.std(ddof=1))}
    return result

def circular_bootstrap(difference,block_length,repetitions=10000,seed=20261108):
    # Same arithmetic/RNG grouping as the original fold-1 implementation.
    d=np.asarray(difference,dtype=np.float64)
    assert d.ndim==1 and len(d)>=2 and np.isfinite(d).all()
    block=min(block_length,len(d));n=len(d);rng=np.random.default_rng(seed)
    offsets=np.arange(block);means=np.empty(repetitions)
    for start in range(0,repetitions,200):
        stop=min(start+200,repetitions)
        positions=rng.integers(n,size=(stop-start,int(np.ceil(n/block))))
        indices=((positions[:,:,None]+offsets[None,None,:])%n).reshape(stop-start,-1)[:,:n]
        means[start:stop]=d[indices].mean(axis=1)
    effect=float(d.mean());ci=np.quantile(means,[.025,.975])
    p=float((1+np.count_nonzero(np.abs(means-effect)>=abs(effect)))/(repetitions+1))
    return {'mean_difference':effect,'ci95':ci.tolist(),'raw_p':p,'block_length':block,
            'repetitions':repetitions,'rng_seed':seed,'origins':n}

def dm_hac20(difference):
    d=np.asarray(difference,dtype=np.float64);n=len(d);z=d-d.mean();lag=min(20,n-1)
    variance=float(np.dot(z,z)/n)
    for k in range(1,lag+1):variance+=2*(1-k/(lag+1))*float(np.dot(z[k:],z[:-k])/n)
    variance=max(0.,variance);effect=float(d.mean());se=float(np.sqrt(variance/n))
    correction=float(np.sqrt(max((n+1.-10.+20./n)/n,0.)))
    if se<=np.finfo(np.float64).eps:
        statistic=0. if abs(effect)<=np.finfo(np.float64).eps else math.copysign(math.inf,effect)
    else:statistic=effect/se
    statistic*=correction
    return {'mean_difference':effect,'hac_long_run_variance':variance,'standard_error':se,
            'harvey_correction':correction,'dm_statistic':statistic if np.isfinite(statistic) else None,
            'dm_statistic_nonfinite':None if np.isfinite(statistic) else ('positive_infinity' if statistic>0 else 'negative_infinity'),
            'raw_p':math.erfc(abs(statistic)/math.sqrt(2)),'origins':n,'horizon':5,'HAC_lag':lag}

def holm(pvalues):
    names=sorted(pvalues,key=lambda x:(pvalues[x],x));adjusted={};running=0.
    for rank,name in enumerate(names):
        assert 0<=pvalues[name]<=1
        running=max(running,min(1.,(len(names)-rank)*pvalues[name]));adjusted[name]=running
    return adjusted
