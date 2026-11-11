#!/usr/bin/env python3
"""NEW task-trained TEDM/Timer-XL experiments using pinned official backbones.

One model family per process prevents upstream packages named `models`/`utils`
from colliding. No historical main-table values are used for fitting or selection.
Run --smoke first; it checks native mechanics and the TEDM track-axis correction.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');os.replace(q,p)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def fixed_rng(seed,device):
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        yield


def unflatten_tracks(values,tracks,batch):
    """Input order is track-major because Tensor.repeat(tracks,1,1) is used."""
    if values.shape[0] != tracks*batch:raise ValueError('Track/batch count mismatch')
    return values.reshape(tracks,batch,*values.shape[1:]).permute(0,1,3,2).contiguous()


@torch.no_grad()
def tedm_rollout(sampler,history,tracks):
    """Native TEDM rollout equations, with only final track-axis unpacking fixed.

    Official sampler.rollout repeats B examples S times, then uses an unrelated
    reshape with S as the last axis. Here [S*B,C,L] is unpacked as [S,B,L,C].
    Native sample(), clamping, conditioning update and prefix rule are unchanged.
    """
    batch=history.shape[0];x=history.transpose(2,1)
    x_cond=x[..., :-(sampler.past_k+sampler.padding)].repeat(tracks,1,1)
    x_cur=x[...,sampler.past_k:].repeat(tracks,1,1)
    shift=sampler.past_k+1
    for step in range(sampler.horizon):
        x_next,_=sampler.sample(x_cur,x_cond)
        if sampler.clamp:
            x_next=torch.clamp(x_next,min=x_cur.amin(-1,keepdim=True),max=x_cur.amax(-1,keepdim=True))
        x_cond=torch.cat([x_cond[...,1:],x_cur[...,-shift].unsqueeze(-1)],dim=-1)
        x_cur=torch.cat([x_cur[...,-1:],x_next],dim=-1)
    if sampler.padding:x_cur=x_cur[...,1:]
    return unflatten_tracks(x_cur,tracks,batch)[:,:,-sampler.horizon:,:]


class NativeModel:
    def __init__(self,family,device,config):
        self.family=family;self.device=device;self.config=config
        if family=='timerxl':
            from models.timer_xl import Model
            self.base=Model(SimpleNamespace(**config['timer'])).to(device)
            self.ema=None
        else:
            from models.networks import UNet
            from models.noise import TEDM as Noise
            from models.precond import TEDM as Preconditioner
            from ema_pytorch import EMA
            self.noise=Noise(padding=1,**config['tedm_noise']);self.noise.to(device)
            self.base=Preconditioner(UNet(**config['tedm_network']).to(device),self.noise).to(device)
            self.ema=EMA(self.base,beta=.995,update_every=10).to(device)

    def loss(self,history,target):
        if self.family=='timerxl':
            prediction=self.base(history,None,None)[:,-5:,:]
            return torch.mean((prediction-target)**2)
        # Native shifted-window denoising. All rows below are training data:
        # take a 126-row window ending at the final purged training target.
        x=torch.cat([history,target],dim=1)[:,-126:,:].transpose(2,1)
        x_cond=x[...,:112];y=x[...,13:]
        y,sigma,weight=self.noise(y)
        denoised=self.base(y+sigma*torch.randn_like(y),x_cond,sigma)
        return (weight*(denoised-y)**2).mean()

    def export_state(self):
        return {'base':self.base.state_dict(),'ema':self.ema.state_dict() if self.ema else None}

    def load_state(self,state):
        self.base.load_state_dict(state['base'],strict=True)
        if self.ema:self.ema.load_state_dict(state['ema'],strict=True)

    def eval_model(self):
        if self.ema:
            self.ema.ema_model.eval();return self.ema.ema_model
        self.base.eval();return self.base


def load_data(bundle):
    from crossfold_data import load_data as load_crossfold_data
    return load_crossfold_data(bundle)


@torch.no_grad()
def timer_predict(native,data,positions,batch_size):
    model=native.eval_model();chunks=[]
    for start in range(0,len(positions),batch_size):
        x=torch.from_numpy(data['history'][positions[start:start+batch_size]]).to(native.device)
        pred=model(x,None,None)[:,-5:,:].cpu().numpy()
        chunks.append(pred*data['scale'][None,None]+data['location'][None,None])
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def tedm_predict(native,data,positions,samples,origin_batch,track_chunk,rng_seed):
    from training.sampler import TEDM as Sampler
    model=native.eval_model();noise=native.noise.eval()
    sampler=Sampler(model,noise,horizon=5,seq_len=112,past_k=13,padding=1,backward=True,deterministic=False,clamp=True,n_tracks=samples)
    chunks=[]
    with fixed_rng(rng_seed,native.device):
        for start in range(0,len(positions),origin_batch):
            x=torch.from_numpy(data['history'][positions[start:start+origin_batch]]).to(native.device)
            sample_chunks=[]
            for offset in range(0,samples,track_chunk):
                count=min(track_chunk,samples-offset)
                pred=tedm_rollout(sampler,x,count).cpu().numpy()
                sample_chunks.append(pred*data['scale'][None,None,None]+data['location'][None,None,None])
            chunks.append(np.concatenate(sample_chunks,axis=0))
    samples_array=np.concatenate(chunks,axis=1).astype(np.float32)
    if not np.isfinite(samples_array).all():raise RuntimeError('Non-finite native TEDM samples')
    return samples_array.mean(axis=0,dtype=np.float64).astype(np.float32),samples_array


def forecast(native,data,split,samples,batch,origin_batch,track_chunk,rng_seed):
    positions=data[f'{split}_positions']
    if native.family=='tedm':
        mean,scenarios=tedm_predict(native,data,positions,samples,origin_batch,track_chunk,rng_seed)
        residual_pool=None;draws=None
    else:
        from fconcord.benchmark_training import residual_bootstrap_scenarios
        mean=timer_predict(native,data,positions,batch)
        train_mean=timer_predict(native,data,data['train_positions'],batch)
        residual_pool=data['targets'][data['train_positions']]-train_mean
        scenarios,draws=residual_bootstrap_scenarios(mean,residual_pool,samples,rng_seed)
    return {'mean':mean,'samples':scenarios,'target':data['targets'][positions],'indices':positions,'residual_pool':residual_pool,'draws':draws}


@torch.no_grad()
def fast_es(samples,target,device):
    """Exact unbiased finite-sample cross-sectional ES, chunked on device."""
    values=[];s=len(samples)
    for start in range(0,len(target),16):
        x=torch.from_numpy(samples[:,start:start+16]).to(device=device,dtype=torch.float64)
        y=torch.from_numpy(target[start:start+16]).to(device=device,dtype=torch.float64)
        first=torch.linalg.vector_norm(x-y[None],dim=-1).mean(0)
        p=x.permute(1,2,0,3).reshape(-1,s,x.shape[-1])
        d=torch.cdist(p,p,compute_mode='donot_use_mm_for_euclid_dist')
        # Sum includes zero self-distances and all ordered unequal pairs.
        second=d.sum((-1,-2))/(2*s*(s-1))
        values.append((first-second.reshape(first.shape)).mean(1).cpu().numpy())
    return np.concatenate(values)


def smoke(native,data,out,config):
    from fconcord.statistical_tests import per_origin_energy_score
    seed_all(123);indices=data['train_positions'][:2]
    x=torch.from_numpy(data['history'][indices]).to(native.device);y=torch.from_numpy(data['target_scaled'][indices]).to(native.device)
    native.base.train();loss=native.loss(x,y);assert torch.isfinite(loss)
    loss.backward();grad=sum(float(p.grad.norm()) for p in native.base.parameters() if p.grad is not None);assert np.isfinite(grad) and grad>0
    opt=torch.optim.Adam(native.base.parameters(),lr=1e-4);opt.step();opt.zero_grad()
    if native.ema:native.ema.update()
    ids=data['val_positions'][:2]
    if native.family=='tedm':
        mean,samples=tedm_predict(native,data,ids,3,2,3,999)
        from training.sampler import TEDM as Sampler
        model=native.eval_model();sch=native.noise.eval()
        sampler=Sampler(model,sch,horizon=5,seq_len=112,past_k=13,padding=1,backward=True,deterministic=True,clamp=True)
        history=torch.from_numpy(data['history'][ids]).to(native.device)
        with fixed_rng(18,native.device):native_reference=sampler.rollout(history)[:,-5:,:]
        with fixed_rng(18,native.device):adapted=tedm_rollout(sampler,history,1)[0]
        deterministic_error=float((native_reference-adapted).abs().max());assert deterministic_error==0
        # The unpacking must preserve every track and origin independently.
        sentinel=torch.arange(3*2*4*7).reshape(6,4,7)
        unpacked=unflatten_tracks(sentinel,3,2)
        for s in range(3):
            for b in range(2):assert torch.equal(unpacked[s,b],sentinel[s*2+b].T)
        # Test the entire corrected rollout with origin-specific tagged values.
        class TaggedSampler:
            past_k=13;padding=1;horizon=5;clamp=False
            def sample(self,current,condition):return current[...,1:],None
        tagged=torch.stack([torch.full((126,32),float(b+1)) for b in range(2)])
        check=tedm_rollout(TaggedSampler(),tagged,3)
        for s in range(3):
            for b in range(2):assert torch.all(check[s,b]==b+1)
    else:
        mean=timer_predict(native,data,ids,2)
        samples=np.stack([mean-.001,mean,mean+.001]);deterministic_error=None
    assert mean.shape==(2,5,32) and samples.shape==(3,2,5,32)
    assert np.isfinite(samples).all()
    target=data['targets'][ids];gpu_es=fast_es(samples,target,native.device)
    exact_es=per_origin_energy_score(samples.astype(np.float64),target.astype(np.float64))
    es_error=float(np.max(np.abs(gpu_es-exact_es)));assert es_error<1e-10
    state=native.export_state();torch.save(state,out/'smoke_checkpoint.pt')
    fresh=NativeModel(native.family,native.device,config);fresh.load_state(torch.load(out/'smoke_checkpoint.pt',map_location=native.device,weights_only=False))
    if native.family=='tedm':m2,s2=tedm_predict(fresh,data,ids,3,2,3,999)
    else:m2=timer_predict(fresh,data,ids,2);s2=np.stack([m2-.001,m2,m2+.001])
    reload_error=float(np.max(np.abs(s2-samples)));assert reload_error<1e-7
    report={'family':native.family,'passed':True,'script_sha256':sha256(__file__),'training_loss':float(loss.detach()),'gradient_norm_sum':grad,'mean_shape':list(mean.shape),'sample_shape':list(samples.shape),'upstream_deterministic_rollout_max_error':deterministic_error,'track_origin_axis_checks':'passed' if native.family=='tedm' else 'not applicable','ES_max_error':es_error,'reload_max_error':reload_error,'registered_parameters':sum(p.numel() for p in native.base.parameters())}
    dump(out/'smoke_report.json',report);print(json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=['tedm','timerxl'],required=True)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--extra-pythonpath',type=Path)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--seeds',default='1,7,19,37,73')
    ap.add_argument('--learning-rates',default='0.001,0.0003')
    ap.add_argument('--max-epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--origin-batch',type=int,default=8)
    ap.add_argument('--track-chunk',type=int,default=64)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);bundle=a.bundle.resolve()
    if a.extra_pythonpath:sys.path.insert(0,str(a.extra_pythonpath.resolve()))
    sys.path.insert(0,str(a.repo.resolve()/'src'))
    sys.path.insert(0,str(bundle/'upstream'/('tedm' if a.family=='tedm' else 'OpenLTM')))
    from fconcord.calibration import calibrate_predictable_signal
    from fconcord.metrics import point_metrics
    torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    data,manifest=load_data(bundle)
    from crossfold_data import verify_official_upstreams
    verify_official_upstreams(bundle,manifest)
    lr_candidates=[float(s) for s in a.learning_rates.split(',')];seeds=[int(s) for s in a.seeds.split(',')]
    config={'timer':{'input_token_len':21,'output_token_len':5,'d_model':128,'d_ff':512,'e_layers':3,'n_heads':4,'dropout':.1,'output_attention':False,'covariate':False,'flash_attention':False,'activation':'gelu','use_norm':True},'tedm_network':{'feat_size':32,'seq_len':112,'past_k':13,'model_channels':32,'num_blocks':2,'channel_mult':[1,2,3,4,5],'attn_resolutions':[112,56,28,14],'sigma_from_t':False,'kaiser_size':6,'kaiser_beta':14.77,'dropout':.10},'tedm_noise':{'sigma_data':1.0,'var_min':.001,'var_max':5.0,'scale_min':1.0,'scale_max':8.0,'scale_from_data':True,'stats':'cumulative','sliding_window':60,'ema_beta':.99}}
    protocol={'schema':'trefin-fresh-official-baseline-v1','status':'NEW supplemental experiments; no claim to reconstruct the historical separate-host experiments','family':a.family,'upstream':manifest['upstreams']['tedm' if a.family=='tedm' else 'OpenLTM'],'data_manifest_sha256':sha256(bundle/'manifest.json'),'data_sha256':manifest['data_sha256'],'script_sha256':sha256(__file__),'seeds':seeds,'learning_rate_candidates':lr_candidates,'max_epochs':a.max_epochs,'patience_epochs':a.patience,'batch_size':a.batch_size,'validation_samples':64,'test_samples':256,'evaluation_rng_seed':271828,'configuration':config,'normalization':manifest['normalization'],'selection':f"validation ES on {len(data['val_positions'])} exact fold {manifest['fold']} origins after every epoch; same initialization seed for LR candidates; no test access during training or selection",'mean_calibration':'after checkpoint/candidate selection; rho in [0.05,1] selected by validation ES with frozen innovations','inputs':'126 historical one-period excess log returns for all 32 assets; factor file used only for risk-free-return subtraction when reconstructing labels; no FF5 values, concepts or OHLCV as predictor inputs','timer_adapter':'official Timer-XL point backbone, six21-day input tokens and5-day output head, retain last output token; training-residual joint paths for scenarios','tedm_adapter':'official UNet/noise/preconditioner/native stochastic sampler;112denoised positions+13conditioning offset+1padding=126total observed rows;five rollout steps;lastfive outputs;explicit track-major reshape correction','tedm_training':'official weighted denoising objective on126-row training windows ending at each purged target+5; EMA beta.995 every10updates; Adam betas(.9,.96), no weight decay; native plateau scheduler with100update warmup, factor.9 patience15 threshold.01','timer_training':'point MSE over5futurereturns; Adam betas(.9,.999), no weight decay; epoch cosine scheduler; last-token task adaptation','device':str(device),'effective_sampler_origin_batch':a.origin_batch,'effective_sampler_track_chunk':a.track_chunk}
    from crossfold_data import protocol_metadata, enforce_frozen_arguments
    enforce_frozen_arguments(a)
    protocol.update(protocol_metadata(manifest))
    protocol['frozen_fold1_runner_sha256']=sha256(Path(__file__).parent/'frozen_fold1'/Path(__file__).name)
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous!=protocol:raise RuntimeError('Existing protocol differs; use a new output directory')
    dump(protocol_path,protocol)
    if a.smoke:
        seed_all(1);smoke(NativeModel(a.family,device,config),data,out,config);return
    results=[]
    for seed in seeds:
        seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=True)
        if (seed_dir/'result.json').exists():results.append(json.loads((seed_dir/'result.json').read_text()));continue
        candidates=[]
        for ci,lr in enumerate(lr_candidates,1):
            candidate_dir=seed_dir/f'candidate_{ci:02d}';candidate_dir.mkdir(exist_ok=True)
            if (candidate_dir/'complete.json').exists():candidates.append(json.loads((candidate_dir/'complete.json').read_text()));continue
            seed_all(seed);native=NativeModel(a.family,device,config)
            optimizer=torch.optim.Adam(native.base.parameters(),lr=lr,betas=(.9,.96) if a.family=='tedm' else (.9,.999))
            if a.family=='tedm':
                from training.scheduler import ReduceLROnPlateauWithWarmup
                scheduler=ReduceLROnPlateauWithWarmup(optimizer,factor=.9,patience=15,min_lr=1e-6,threshold=.01,threshold_mode='rel',warmup_lr=lr,warmup=100)
            else:scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.max_epochs,eta_min=1e-8)
            best=float('inf');stale=0;updates=0;start=time.time();history=[];train_gen=np.random.default_rng(seed)
            for epoch in range(1,a.max_epochs+1):
                native.base.train();losses=[];epoch_started=time.time()
                shuffled=train_gen.permutation(data['train_positions'])
                for offset in range(0,len(shuffled),a.batch_size):
                    positions=shuffled[offset:offset+a.batch_size]
                    x=torch.from_numpy(data['history'][positions]).to(device);y=torch.from_numpy(data['target_scaled'][positions]).to(device)
                    optimizer.zero_grad(set_to_none=True);loss=native.loss(x,y)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(native.base.parameters(),1.0);optimizer.step();updates+=1
                    if native.ema:native.ema.update()
                    losses.append(float(loss.detach()))
                    if a.family=='tedm':scheduler.step(losses[-1])
                if a.family=='timerxl':scheduler.step()
                val=forecast(native,data,'val',64,a.batch_size,a.origin_batch,a.track_chunk,271828)
                val_es=float(fast_es(val['samples'],val['target'],device).mean())
                row={'epoch':epoch,'optimizer_updates':updates,'training_loss':float(np.mean(losses)),'validation_energy_score':val_es,'validation_mse':float(np.mean((val['mean'].astype(np.float64)-val['target'])**2)),'learning_rate':optimizer.param_groups[0]['lr'],'epoch_seconds':time.time()-epoch_started}
                history.append(row)
                with (candidate_dir/'training_history.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
                print(json.dumps({'event':'epoch','family':a.family,'seed':seed,'candidate':ci,**row}),flush=True)
                state={'model':native.export_state(),'epoch':epoch,'optimizer_updates':updates,'validation_energy_score':val_es,'config':config,'family':a.family,'seed':seed,'candidate':ci,'optimizer':optimizer.state_dict()}
                torch.save(state,candidate_dir/'last.pt')
                if val_es<best-1e-6:
                    best=val_es;stale=0;torch.save(state,candidate_dir/'best.pt')
                else:stale+=1
                if stale>=a.patience:break
            chosen=torch.load(candidate_dir/'best.pt',map_location=device,weights_only=False)
            item={'candidate':ci,'learning_rate':lr,'best_epoch':chosen['epoch'],'best_validation_es':chosen['validation_energy_score'],'epochs_completed':epoch,'optimizer_updates':updates,'training_and_validation_seconds':time.time()-start,'parameters':sum(p.numel() for p in native.base.parameters()),'checkpoint':str((candidate_dir/'best.pt').relative_to(out)),'checkpoint_sha256':sha256(candidate_dir/'best.pt')}
            dump(candidate_dir/'complete.json',item);candidates.append(item)
            del native,optimizer,val
            if device.type=='cuda':torch.cuda.empty_cache()
        chosen=min(candidates,key=lambda r:(r['best_validation_es'],r['candidate']))
        dump(seed_dir/'selected_candidate.json',chosen)
        native=NativeModel(a.family,device,config)
        checkpoint=torch.load(out/chosen['checkpoint'],map_location=device,weights_only=False);native.load_state(checkpoint['model'])
        val=forecast(native,data,'val',64,a.batch_size,a.origin_batch,a.track_chunk,271828)
        val_es=float(fast_es(val['samples'],val['target'],device).mean())
        if abs(val_es-chosen['best_validation_es'])>1e-7:raise RuntimeError('Reloaded validation score mismatch')
        calibration=calibrate_predictable_signal(val['mean'],val['samples'],val['target'],.05,1.,1e-5)
        assert calibration['optimizer_success'];rho=calibration['selected_signal_shrinkage']
        dump(seed_dir/'calibration.json',calibration)
        np.savez_compressed(seed_dir/'validation_predictions.npz',mean=val['mean'],samples=val['samples'],target=val['target'],indices=val['indices'])
        del val
        print(json.dumps({'event':'final_test_started','family':a.family,'seed':seed,'rho':rho,'selected_candidate':chosen['candidate']}),flush=True)
        test_started=time.time();test=forecast(native,data,'test',256,a.batch_size,a.origin_batch,a.track_chunk,271829)
        target=test['target'].astype(np.float64);mean=test['mean'].astype(np.float64);scenarios=test['samples'].astype(np.float64)
        after_mean=rho*mean;after_samples=scenarios+(rho-1)*mean[None]
        before=point_metrics(target,mean);after=point_metrics(target,after_mean)
        es_before=fast_es(test['samples'],test['target'],device)
        es_after=es_before+(np.linalg.norm(after_samples-target[None],axis=-1)-np.linalg.norm(scenarios-target[None],axis=-1)).mean(axis=(0,2))
        before['energy_score']=float(es_before.mean());after['energy_score']=float(es_after.mean())
        for metrics,s in [(before,scenarios),(after,after_samples)]:
            lo,hi=np.quantile(s,[.05,.95],axis=0);metrics['coverage_90']=float(((target>=lo)&(target<=hi)).mean());metrics['width_90']=float((hi-lo).mean())
        pair_error=float(np.max(np.abs((after_samples[1]-after_samples[0])-(scenarios[1]-scenarios[0]))));assert pair_error<1e-12
        saved={'mean':test['mean'],'samples':test['samples'],'target':test['target'],'indices':test['indices'],'calibrated_mean':after_mean.astype(np.float32),'mse_before':np.mean((mean-target)**2,axis=(1,2)),'mse_after':np.mean((after_mean-target)**2,axis=(1,2)),'mae_before':np.mean(np.abs(mean-target),axis=(1,2)),'mae_after':np.mean(np.abs(after_mean-target),axis=(1,2)),'es_before':es_before,'es_after':es_after}
        if test['residual_pool'] is not None:saved.update(train_residual_pool=test['residual_pool'],residual_draw_indices=test['draws'])
        np.savez_compressed(seed_dir/'test_predictions.npz',**saved)
        np.savez_compressed(seed_dir/'paired_losses.npz',**{key:saved[key] for key in ['indices','mse_before','mse_after','mae_before','mae_after','es_before','es_after']})
        result={'family':a.family,'seed':seed,'scope':protocol['status'],'selected_candidate':chosen,'candidate_summaries':candidates,'rho':rho,'before':before,'after':after,'test_origins':len(target),'test_seconds':time.time()-test_started,'reload_validation_ES_difference':abs(val_es-chosen['best_validation_es']),'pairwise_translation_error':pair_error,'source_data_sha256':manifest['data_sha256'],'protocol_sha256':sha256(protocol_path),'test_predictions_sha256':sha256(seed_dir/'test_predictions.npz'),'test_targets_float32_bytes_sha256':hashlib.sha256(np.ascontiguousarray(test['target'],dtype=np.float32).tobytes()).hexdigest()}
        dump(seed_dir/'result.json',result);results.append(result)
        print(json.dumps({'event':'seed_complete','family':a.family,'seed':seed,'rho':rho,'before':before,'after':after}),flush=True)
        del native,test,scenarios,after_samples,saved
        if device.type=='cuda':torch.cuda.empty_cache()
    aggregate={'family':a.family,'seeds':[r['seed'] for r in results],'runs':results,'metrics':{}}
    for stage in ['before','after']:
        aggregate['metrics'][stage]={key:{'mean':float(np.mean([r[stage][key] for r in results])),'sample_std':float(np.std([r[stage][key] for r in results],ddof=1)) if len(results)>1 else None} for key in ['mse','rmse','mae','energy_score','msic','msir','coverage_90','width_90']}
    dump(out/'summary.json',aggregate)
    print(json.dumps({'event':'all_complete','family':a.family,'completed_seeds':len(results)}),flush=True)


if __name__=='__main__':main()
