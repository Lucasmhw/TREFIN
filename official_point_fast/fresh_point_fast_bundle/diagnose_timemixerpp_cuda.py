"""Report all CUDA numerical differences; never converts a failure into a pass.

Validation histories and training-only gradient checks. No forecast performance
metric, test outcome or hyperparameter selection is used. Requires a real fitted
checkpoint in addition to initialization and two-update checks.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


class LayerCapture:
    def __init__(self,native):
        self.records=[[] for _ in native.encoder_model]
        self.handles=[]
        for i,block in enumerate(native.encoder_model):
            self.handles.append(block.register_forward_hook(self._hook(i)))

    def _hook(self,index):
        def capture(block,args,output):
            xs=args[0];coarse=xs[-1].detach();origins=len(coarse)//32
            amplitude=torch.fft.rfft(coarse,dim=1).abs()
            if hasattr(block,'_original_forward'):
                frequency=amplitude.reshape(origins,32,*amplitude.shape[1:]).mean(1).mean(-1)
            else:
                if origins!=1:raise ValueError('Reference must be evaluated one origin at a time')
                frequency=amplitude.mean(0).mean(-1).unsqueeze(0)
            frequency[:,0]=0
            top=torch.topk(frequency,min(block.k,frequency.shape[-1]),dim=-1).indices
            self.records[index].append({'input_coarse':coarse.cpu(),'frequency':frequency.cpu(),'top':top.cpu(),'output':[x.detach().cpu() for x in output]})
        return capture

    def close(self):
        for handle in self.handles:handle.remove()
        merged=[]
        for rows in self.records:
            merged.append({key:torch.cat([r[key] for r in rows],dim=0) for key in ['input_coarse','frequency','top']})
            merged[-1]['output']=[torch.cat([r['output'][i] for r in rows],dim=0) for i in range(len(rows[0]['output']))]
        return merged


def difference(value,reference,assets=32):
    delta=(value.double()-reference.double())
    reference_rms=float(reference.double().square().mean().sqrt())
    rms=float(delta.square().mean().sqrt())
    return {'max_absolute':float(delta.abs().max()),'rms':rms,'reference_rms':reference_rms,
            'relative_rms':rms/max(reference_rms,1e-30),
            'origin_max_absolute':delta.abs().reshape(len(delta)//assets,-1).max(-1).values.tolist()}


def layer_report(actual,reference):
    rows=[]
    for index,(got,expected) in enumerate(zip(actual,reference)):
        top_equal=(got['top']==expected['top']).all(-1)
        set_equal=(got['top'].sort(-1).values==expected['top'].sort(-1).values).all(-1)
        sorted_frequency=expected['frequency'].sort(-1,descending=True).values
        k=expected['top'].shape[-1]
        gap=sorted_frequency[:,k-1]-sorted_frequency[:,k]
        rows.append({'block':index,'top_k_rank_mismatch_origins':torch.nonzero(~top_equal).flatten().tolist(),
                     'top_k_membership_mismatch_origins':torch.nonzero(~set_equal).flatten().tolist(),
                     'reference_top_k':expected['top'].tolist(),'fast_top_k':got['top'].tolist(),
                     'minimum_reference_selection_gap':float(gap.min()),
                     'frequency_max_absolute_difference':float((got['frequency']-expected['frequency']).abs().max()),
                     'coarse_input_error':difference(got['input_coarse'],expected['input_coarse']),
                     'scale_output_errors':[difference(x,y) for x,y in zip(got['output'],expected['output'])]})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--data-bundle',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--origins',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();bundle=a.bundle.resolve();sys.path.insert(0,str(bundle))
    torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=module('point_reference',bundle/'fresh_point_baselines.py')
    fast=module('point_fast',bundle/'fresh_point_baselines_fast.py')
    fast.prepare_imports(bundle,'timemixerpp');device=torch.device(a.device)
    data,manifest=fast.load_data(a.data_bundle.resolve())
    checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=False)
    config=checkpoint['config']
    positions=data['val_positions'][np.linspace(0,len(data['val_positions'])-1,a.origins,dtype=int)]
    histories=torch.from_numpy(data['history'][positions]).to(device)
    ids=data['train_positions'][:4]
    train_x=torch.from_numpy(data['history'][ids]).to(device)
    train_y=torch.from_numpy(data['target_scaled'][ids]).to(device)
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    result={'status':'diagnostic_report_only_not_approval','device':str(device),'torch_version':torch.__version__,
            'origins':a.origins,'positions':positions.tolist(),'raw_origins':data['origins'][positions].tolist(),
            'data_sha256':manifest['data_sha256'],'data_split':'validation histories; training-only gradient checks; no test outcome',
            'fast_module_sha256':hashlib.sha256((bundle/'timemixerpp_fast_eval.py').read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'original_tolerances':{'normalized_max_absolute':2e-5,'return_unit_max_absolute':1e-6},'cases':[]}
    def save():
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    save()
    for name in ['random_initialization','two_training_updates','existing_fitted_checkpoint']:
        fast.seed_all(41);ref=old.NativeModel('timemixerpp',device,config)
        if name=='existing_fitted_checkpoint':ref.load_state(checkpoint['model'])
        if name=='two_training_updates':
            ref.base.train();opt=torch.optim.Adam(ref.base.parameters(),lr=.001)
            for _ in range(2):
                opt.zero_grad();ref.loss(train_x,train_y).backward();opt.step()
        accelerated=fast.NativeModel('timemixerpp',device,config);accelerated.load_state(ref.export_state())
        ref.base.train();accelerated.base.train();ref.base.zero_grad(set_to_none=True);accelerated.base.zero_grad(set_to_none=True)
        fast.seed_all(839);rl=ref.loss(train_x,train_y);rl.backward()
        fast.seed_all(839);fl=accelerated.loss(train_x,train_y);fl.backward()
        grad_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(ref.base.parameters(),accelerated.base.parameters()) if p.grad is not None)
        case={'state':name,'training_loss_error':abs(float(rl.detach()-fl.detach())),
              'training_gradient_max_error':grad_error,'comparisons':[]}
        result['cases'].append(case);save()
        ref.base.eval();accelerated.base.eval()
        with torch.no_grad():
            ref.base(histories[:2]);accelerated.base(histories[:2]);sync()
            start=time.perf_counter();expected=ref.base(histories);sync();case['reference_seconds']=time.perf_counter()-start
            capture=LayerCapture(ref.base.native);ref.base(histories);reference_layers=capture.close()
            for batch in sorted(set([1,4,16,a.origins])):
                sync();start=time.perf_counter()
                actual=torch.cat([accelerated.base(histories[j:j+batch]) for j in range(0,len(histories),batch)])
                sync();seconds=time.perf_counter()-start
                error=(actual-expected).abs();norm_error=float(error.max())
                scale=torch.as_tensor(data['scale'],device=device)
                raw_error=float((error*scale).max())
                rms=float((actual.double()-expected.double()).square().mean().sqrt())
                reference_rms=float(expected.double().square().mean().sqrt())
                capture=LayerCapture(accelerated.base.native)
                for j in range(0,len(histories),batch):accelerated.base(histories[j:j+batch])
                layers=layer_report(capture.close(),reference_layers)
                row={'outer_batch':batch,'normalized_max_absolute_error':norm_error,'return_unit_max_absolute_error':raw_error,
                     'return_unit_error_basis_points':raw_error*1e4,'normalized_rms_error':rms,
                     'normalized_reference_rms':reference_rms,'normalized_relative_rms_error':rms/max(reference_rms,1e-30),
                     'finite':bool(torch.isfinite(actual).all()),'within_original_thresholds':norm_error<2e-5 and raw_error<1e-6,
                     'seconds':seconds,'speedup_vs_origin_loop':case['reference_seconds']/seconds,'layers':layers}
                case['comparisons'].append(row);save()
                print(json.dumps({'state':name,**{k:v for k,v in row.items() if k!='layers'},'topk_mismatches':[len(r['top_k_membership_mismatch_origins']) for r in layers]}),flush=True)
        del ref,accelerated,reference_layers
    result['all_original_thresholds_pass']=all(r['within_original_thresholds'] for c in result['cases'] for r in c['comparisons'])
    result['all_training_checks_exact']=all(c['training_loss_error']==0 and c['training_gradient_max_error']==0 for c in result['cases'])
    result['all_topk_memberships_identical']=all(not layer['top_k_membership_mismatch_origins'] for c in result['cases'] for r in c['comparisons'] for layer in r['layers'])
    save();print(json.dumps({k:result[k] for k in ['status','all_original_thresholds_pass','all_training_checks_exact','all_topk_memberships_identical']}),flush=True)


if __name__=='__main__':main()
