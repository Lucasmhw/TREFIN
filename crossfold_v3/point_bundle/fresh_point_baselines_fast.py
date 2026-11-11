#!/usr/bin/env python3
"""Fresh task-trained TimeXer and explicitly attributed PyPOTS TimeMixer++.

Execute the bundled copy; frozen common helpers sit beside this file. One model
family per process. Never presents these experiments as historical reproduction.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,json,sys,time,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fresh_official_common_v1 import dump,sha256,seed_all,load_data,forecast,timer_predict,fast_es


def prepare_imports(bundle,family):
    if family=='timexer':
        sys.path.insert(0,str(bundle/'upstream/TimeSeriesLibrary'))
    else:
        # Bypass unrelated PyPOTS top-level training/data dependencies. All loaded
        # mathematical modules are original unmodified files with relative imports.
        base=bundle/'upstream/PyPOTS'
        for name in ['pypots','pypots.nn','pypots.nn.modules','pypots.nn.modules.transformer']:
            module=types.ModuleType(name);module.__path__=[str(base/Path(*name.split('.')))]
            sys.modules[name]=module


class PointForward(torch.nn.Module):
    def __init__(self,family,config):
        super().__init__();self.family=family
        if family=='timexer':
            from models.TimeXer import Model
            self.native=Model(SimpleNamespace(**config))
        else:
            from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP
            self.native=BackboneTimeMixerPP(**config)
            from timemixerpp_fast_eval import install_independent_origin_eval
            install_independent_origin_eval(self.native,n_assets=32)

    def forward(self,history,_unused1=None,_unused2=None):
        if self.family=='timexer':result=self.native(history,None,None,None)
        else:result=self.native.forecast(history,None)
        if result.shape != (history.shape[0],5,32):raise RuntimeError(f'Unexpected output shape {result.shape}')
        return result


class NativeModel:
    def __init__(self,family,device,config):
        self.family=family;self.device=device;self.config=config;self.ema=None
        self.base=PointForward(family,config).to(device)
    def loss(self,history,target):return ((self.base(history)-target)**2).mean()
    def export_state(self):return {'base':self.base.state_dict(),'ema':None}
    def load_state(self,state):self.base.load_state_dict(state['base'],strict=True)
    def eval_model(self):self.base.eval();return self.base


def smoke(native,data,out,config):
    from fconcord.statistical_tests import per_origin_energy_score
    from fconcord.benchmark_training import residual_bootstrap_scenarios
    seed_all(123);positions=data['train_positions'][:2]
    x=torch.from_numpy(data['history'][positions]).to(native.device)
    y=torch.from_numpy(data['target_scaled'][positions]).to(native.device)
    native.base.train();loss=native.loss(x,y);assert torch.isfinite(loss)
    loss.backward();grad=sum(float(p.grad.norm()) for p in native.base.parameters() if p.grad is not None)
    assert np.isfinite(grad) and grad>0
    optimizer=torch.optim.Adam(native.base.parameters(),lr=1e-4);optimizer.step();optimizer.zero_grad()
    ids=data['val_positions'][:2];mean=timer_predict(native,data,ids,2)
    pool=data['targets'][positions]-timer_predict(native,data,positions,2)
    samples,draws=residual_bootstrap_scenarios(mean,pool,3,999)
    assert samples.shape==(3,2,5,32) and np.isfinite(samples).all()
    es=fast_es(samples,data['targets'][ids],native.device)
    exact=per_origin_energy_score(samples.astype(np.float64),data['targets'][ids].astype(np.float64))
    error=float(np.max(np.abs(es-exact)));assert error<1e-10
    torch.save(native.export_state(),out/'smoke_checkpoint.pt')
    fresh=NativeModel(native.family,native.device,config)
    fresh.load_state(torch.load(out/'smoke_checkpoint.pt',map_location=native.device,weights_only=False))
    reload=timer_predict(fresh,data,ids,2)
    reload_error=float(np.max(np.abs(reload-mean)));assert reload_error<1e-7
    independent=timer_predict(fresh,data,ids,1)
    batch_error=float(np.max(np.abs(independent-mean)))
    norm_difference=(independent.astype(np.float64)-mean)/data['scale'][None,None]
    normalized_mean=(mean.astype(np.float64)-data['location'][None,None])/data['scale'][None,None]
    batch_norm_error=float(np.abs(norm_difference).max())
    batch_relative_rms=float(np.sqrt(np.mean(norm_difference**2))/max(np.sqrt(np.mean(normalized_mean**2)),1e-30))
    assert batch_error<=1e-6 and batch_norm_error<=1e-4 and batch_relative_rms<=1e-4
    report={'family':native.family,'passed':True,'script_sha256':sha256(__file__),'training_loss':float(loss.detach()),'gradient_norm_sum':grad,'mean_shape':list(mean.shape),'sample_shape':list(samples.shape),'ES_max_error':error,'reload_max_error':reload_error,'inference_batch_invariance_error':batch_error,'inference_batch_normalized_max_error':batch_norm_error,'inference_batch_relative_rms_error':batch_relative_rms,'numerical_criteria':'raw<=1e-6, normalizedmax<=1e-4, relativeRMS<=1e-4; see numerical_acceptance.json','registered_parameters':sum(p.numel() for p in native.base.parameters()),'source_manifest_sha256':json.loads((out/'protocol.json').read_text())['source_manifest_sha256']}
    dump(out/'smoke_report.json',report);print(json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=['timexer','timemixerpp'],required=True)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--data-bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--extra-pythonpath',type=Path,action='append',default=[])
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--seeds',default='1,7,19,37,73')
    ap.add_argument('--learning-rates',default='0.001,0.0003')
    ap.add_argument('--max-epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);bundle=a.bundle.resolve()
    a.origin_batch=8;a.track_chunk=64  # unused for point/residual adapters
    a.inference_batch_size=32  # numerically audited independent-origin grouping; training batch unchanged
    for extra in a.extra_pythonpath:sys.path.insert(0,str(extra.resolve()))
    sys.path.insert(0,str(a.repo.resolve()/'src'))
    sources=json.loads((bundle/'manifest.json').read_text())
    for name,record in sources['upstreams'].items():
        for rel,expected in record['files'].items():
            if sha256(bundle/'upstream'/name/rel)!=expected:raise RuntimeError(f'Upstream source hash mismatch {name}/{rel}')
    for rel,expected in sources['files'].items():
        if sha256(bundle/rel)!=expected:raise RuntimeError(f'Adapter/helper hash mismatch {rel}')
    prepare_imports(bundle,a.family)
    from fconcord.calibration import calibrate_predictable_signal
    from fconcord.metrics import point_metrics
    torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    data,manifest=load_data(a.data_bundle.resolve())
    from crossfold_data import verify_registered_data
    verify_registered_data(manifest,sources,a.data_bundle)
    assert sha256(__file__)==sources['files'][Path(__file__).name]
    lr_candidates=[float(s) for s in a.learning_rates.split(',')];seeds=[int(s) for s in a.seeds.split(',')]
    if a.family=='timexer':
        config={'task_name':'short_term_forecast','features':'M','seq_len':126,'pred_len':5,'use_norm':True,'patch_len':21,'enc_in':32,'d_model':128,'d_ff':512,'n_heads':4,'e_layers':3,'dropout':.1,'factor':5,'embed':'timeF','freq':'d','activation':'gelu'}
        adaptation='Unchanged author TimeXer from MIT-licensed TSLib; native multivariate mode; six21-day patches+global token per asset; all32 observed returns also supply variate embeddings; no external covariates or future features; direct5-step head.'
        upstream='TimeSeriesLibrary'
    else:
        config={'task_name':'short_term_forecast','n_steps':126,'n_features':32,'n_pred_steps':5,'n_pred_features':32,'n_layers':2,'d_model':32,'d_ffn':64,'n_heads':2,'dropout':.1,'top_k':3,'n_kernels':3,'channel_mixing':True,'channel_independence':True,'downsampling_layers':2,'downsampling_window':3,'downsampling_method':'avg','use_future_temporal_feature':False,'use_norm':True,'embed':'fixed','freq':'d'}
        adaptation='PyPOTS maintained TimeMixer++ implementation, not an official-author-source reproduction; unchanged forecast/backbone/layers; three exact scales126/42/14; two32-width mixer blocks with64-width FF/inception channels, top3 periods,3 kernels, channel mixing and independence, native RevIN; preserves upstream final/coarsest prediction head. Because native FFT period selection averages its batch, evaluation selects periods independently per origin and groups equal-period convolution/attention branches across origins without mixing information; training retains shuffled minibatches.'
        upstream='PyPOTS'
    protocol={'schema':'trefin-fresh-point-baseline-fast-eval-v1','evaluation_acceleration':'Group independent origins by native per-origin FFT period; unchanged training, architecture, randomization and calibration; equivalent to separate-origin inference up to floating-point reduction roundoff','status':'NEW supplemental experiments; no claim to reconstruct historical separate-host results','family':a.family,'upstream':sources['upstreams'][upstream],'source_manifest_sha256':sha256(bundle/'manifest.json'),'data_manifest_sha256':sha256(a.data_bundle/'manifest.json'),'data_sha256':manifest['data_sha256'],'script_sha256':sha256(__file__),'common_helper_sha256':sha256(bundle/'fresh_official_common_v1.py'),'seeds':seeds,'learning_rate_candidates':lr_candidates,'architecture_candidates':1,'max_epochs':a.max_epochs,'patience_epochs':a.patience,'batch_size':a.batch_size,'validation_samples':64,'test_samples':256,'evaluation_rng_seed':271828,'test_rng_seed':271829,'configuration':config,'normalization':manifest['normalization'],'selection':f"uncalibrated validation ES on all {len(data['val_positions'])} exact fold {manifest['fold']} origins after every epoch; identical initialization/shuffle seed perLR; no test-outcome use in training, selection or calibration",'mean_calibration':'after checkpoint/candidate selection; rho in[.05,1] selected by validation ES with frozen centered-training-residual innovations; exact empirical-adapter predictive mean from backbone, not sample mean','inputs':'126 historical excess log returns for all32 assets; no FF5 values, volume, concepts, graphs or future calendar covariates; differs from TREFIN concept/factor/volume predictors','adapter':adaptation,'scenarios':'whole[5,32] centered training residual paths sampled with replacement, same adapter as fresh Timer-XL; point model itself is not natively probabilistic','training':'pointMSE; Adam(.9,.999), no weight decay; epoch cosine scheduler to1e-8; gradclip1;100full-data epochs cap,15validation-check patience','device':str(device),'outer_point_inference_batch_size':a.inference_batch_size,'effective_native_inference_batch_size':a.inference_batch_size,'FFT_selection_origin_group':1,'numerical_acceptance':json.loads((bundle/'numerical_acceptance.json').read_text()),'numerical_acceptance_sha256':sha256(bundle/'numerical_acceptance.json')}
    from crossfold_data import protocol_metadata, enforce_frozen_arguments
    enforce_frozen_arguments(a)
    protocol.update(protocol_metadata(manifest))
    protocol['frozen_fold1_runner_sha256']=sha256(Path(__file__).parent/'frozen_fold1'/Path(__file__).name)
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Existing protocol differs; use new output directory')
    dump(protocol_path,protocol)
    if a.smoke:
        seed_all(1);smoke(NativeModel(a.family,device,config),data,out,config);return
    # Training/evaluation body copied from frozen common runner, with both families
    # using the point-model cosine branch. No alteration of the running original.
    results=[]
    for seed in seeds:
        seed_dir=out/f'seed_{seed}';seed_dir.mkdir(exist_ok=True)
        if (seed_dir/'result.json').exists():results.append(json.loads((seed_dir/'result.json').read_text()));continue
        candidates=[]
        for ci,lr in enumerate(lr_candidates,1):
            candidate_dir=seed_dir/f'candidate_{ci:02d}';candidate_dir.mkdir(exist_ok=True)
            if (candidate_dir/'complete.json').exists():candidates.append(json.loads((candidate_dir/'complete.json').read_text()));continue
            seed_all(seed);native=NativeModel(a.family,device,config)
            optimizer=torch.optim.Adam(native.base.parameters(),lr=lr,betas=(.9,.999))
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.max_epochs,eta_min=1e-8)
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
                    losses.append(float(loss.detach()))
                scheduler.step()
                val=forecast(native,data,'val',64,a.inference_batch_size,a.origin_batch,a.track_chunk,271828)
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
        val=forecast(native,data,'val',64,a.inference_batch_size,a.origin_batch,a.track_chunk,271828)
        val_es=float(fast_es(val['samples'],val['target'],device).mean())
        if abs(val_es-chosen['best_validation_es'])>1e-7:raise RuntimeError('Reloaded validation score mismatch')
        calibration=calibrate_predictable_signal(val['mean'],val['samples'],val['target'],.05,1.,1e-5)
        assert calibration['optimizer_success'];rho=calibration['selected_signal_shrinkage']
        dump(seed_dir/'calibration.json',calibration)
        np.savez_compressed(seed_dir/'validation_predictions.npz',mean=val['mean'],samples=val['samples'],target=val['target'],indices=val['indices'])
        del val
        print(json.dumps({'event':'final_test_started','family':a.family,'seed':seed,'rho':rho,'selected_candidate':chosen['candidate']}),flush=True)
        test_started=time.time();test=forecast(native,data,'test',256,a.inference_batch_size,a.origin_batch,a.track_chunk,271829)
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
