"""Equivalence and timing checks; validation histories only, no test outcomes.

Run separately on CPU/CUDA. Optional --checkpoint accepts a fitted original
runner best.pt. No performance metric is used and no training protocol is tuned.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--data-bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--origins',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();bundle=a.bundle.resolve();sys.path.insert(0,str(bundle))
    torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=module('point_reference',bundle/'fresh_point_baselines.py')
    fast=module('point_fast',bundle/'fresh_point_baselines_fast.py')
    fast.prepare_imports(bundle,'timemixerpp')
    data,manifest=fast.load_data(a.data_bundle.resolve())
    config={'task_name':'short_term_forecast','n_steps':126,'n_features':32,'n_pred_steps':5,'n_pred_features':32,'n_layers':2,'d_model':32,'d_ffn':64,'n_heads':2,'dropout':.1,'top_k':3,'n_kernels':3,'channel_mixing':True,'channel_independence':True,'downsampling_layers':2,'downsampling_window':3,'downsampling_method':'avg','use_future_temporal_feature':False,'use_norm':True,'embed':'fixed','freq':'d'}
    device=torch.device(a.device)
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    positions=data['val_positions'][np.linspace(0,len(data['val_positions'])-1,a.origins,dtype=int)]
    histories=torch.from_numpy(data['history'][positions]).to(device)
    train_ids=data['train_positions'][:4]
    train_x=torch.from_numpy(data['history'][train_ids]).to(device)
    train_y=torch.from_numpy(data['target_scaled'][train_ids]).to(device)
    cases=[]
    states=[('random_initialization',None),('two_training_updates',None)]
    if a.checkpoint:states.append(('existing_fitted_checkpoint',a.checkpoint))
    for state_name,checkpoint_path in states:
        fast.seed_all(41)
        ref=old.NativeModel('timemixerpp',device,config)
        if checkpoint_path:
            checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
            if 'config' in checkpoint:assert checkpoint['config']==config
            ref.load_state(checkpoint.get('model',checkpoint))
        elif state_name=='two_training_updates':
            ref.base.train();opt=torch.optim.Adam(ref.base.parameters(),lr=.001)
            for _ in range(2):
                opt.zero_grad();loss=ref.loss(train_x,train_y);loss.backward();opt.step()
        accelerated=fast.NativeModel('timemixerpp',device,config)
        accelerated.load_state(ref.export_state())
        assert list(ref.base.state_dict())==list(accelerated.base.state_dict())
        # Training path unchanged, including dropout random-number consumption.
        ref.base.train();accelerated.base.train()
        fast.seed_all(839);ref_loss=ref.loss(train_x,train_y);ref_loss.backward()
        fast.seed_all(839);fast_loss=accelerated.loss(train_x,train_y);fast_loss.backward()
        train_loss_error=abs(float(ref_loss.detach()-fast_loss.detach()))
        gradient_error=0.
        # Clear stale gradients from the explicit two training updates first.
        ref.base.zero_grad(set_to_none=True);accelerated.base.zero_grad(set_to_none=True)
        fast.seed_all(839);ref.loss(train_x,train_y).backward()
        fast.seed_all(839);accelerated.loss(train_x,train_y).backward()
        for (name,p_ref),(name_fast,p_fast) in zip(ref.base.named_parameters(),accelerated.base.named_parameters()):
            assert name==name_fast
            assert (p_ref.grad is None)==(p_fast.grad is None)
            if p_ref.grad is not None:gradient_error=max(gradient_error,float((p_ref.grad-p_fast.grad).abs().max()))
        assert train_loss_error==0 and gradient_error==0,(train_loss_error,gradient_error)
        ref.base.eval();accelerated.base.eval()
        with torch.no_grad():
            # Warm up kernels without using any target.
            ref.base(histories[:2]);accelerated.base(histories[:2]);sync()
            start=time.perf_counter();expected=ref.base(histories);sync();reference_seconds=time.perf_counter()-start
            batch_results=[]
            for batch in [1,4,16,a.origins]:
                sync();start=time.perf_counter()
                got=torch.cat([accelerated.base(histories[j:j+batch]) for j in range(0,len(histories),batch)])
                sync();seconds=time.perf_counter()-start
                error=(got-expected).abs();norm_error=float(error.max())
                raw_error=float((error*torch.as_tensor(data['scale'],device=device)).max())
                assert torch.isfinite(got).all()
                assert norm_error<2e-5 and raw_error<1e-6,(state_name,batch,norm_error,raw_error)
                batch_results.append({'outer_batch':batch,'normalized_max_absolute_error':norm_error,'return_unit_max_absolute_error':raw_error,'seconds':seconds,'speedup_vs_origin_loop':reference_seconds/seconds})
        cases.append({'state':state_name,'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path else None,'unchanged_training_loss_error':train_loss_error,'unchanged_training_gradient_max_error':gradient_error,'reference_seconds':reference_seconds,'comparisons':batch_results})
        del ref,accelerated
    result={'passed':True,'device':str(device),'torch_version':torch.__version__,'origins':a.origins,'positions':positions.tolist(),'raw_origins':data['origins'][positions].tolist(),'data_split':'validation history only; no validation/test target used for equivalence or timing','data_sha256':manifest['data_sha256'],'fast_module_sha256':hashlib.sha256((bundle/'timemixerpp_fast_eval.py').read_bytes()).hexdigest(),'cases':cases}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
