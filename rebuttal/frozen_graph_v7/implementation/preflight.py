"""Real training-batch smoke tests; no checkpoint, validation selection or test access."""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml
from fconcord.experiment import load_prepared,_model_factory
from fconcord.trainer import make_loader
from fconcord.losses import total_loss
from fconcord.utils import seed_everything
from protocol import (VARIANTS,SEEDS,CANDIDATES,configure,sha256,dump,reference_config,
  prepared_hashes,effective_fingerprint,train_validation_only,runtime_sources,implementation_hashes)

def run(args):
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    source=runtime_sources();records=[];fingerprints={}
    for fold in [1,2,3]:
        reference={}
        for seed in SEEDS:
            base=yaml.safe_load(reference_config(args.root,fold,seed).read_text())
            for variant in VARIANTS:
                fps=[]
                for candidate in CANDIDATES:
                    cfg=configure(base,variant,candidate,fold=fold,seed=seed,device=args.device)
                    fps.append(effective_fingerprint(cfg))
                    assert cfg['training']['lambda_graph']==(0 if variant=='FrozenGraph' else (.02 if candidate['id']=='weaker_state_supervision' else .08))
                if seed==1:reference[variant]=fps
                else:assert fps==reference[variant],f'Configuration mismatch fold {fold} seed {seed} {variant}'
            fingerprints[f'fold_{fold}_seed_{seed}']={'base_sha256':sha256(reference_config(args.root,fold,seed)),
              'effective_fingerprints':{k:v for k,v in reference.items()}}
        assert len(set(reference['Full']))==4
        assert len(set(reference['FrozenGraph']))==3 and reference['FrozenGraph'][0]==reference['FrozenGraph'][1]
        prepared,guard=train_validation_only(load_prepared(args.prepared_root/f'fold_{fold}'))
        base=yaml.safe_load(reference_config(args.root,fold).read_text())
        for variant in VARIANTS:
            started=time.perf_counter();seed_everything(1,True)
            cfg=configure(base,variant,CANDIDATES[0],fold=fold,seed=1,device=args.device)
            batch={k:v.to(device) for k,v in next(iter(make_loader(prepared,'train',cfg['training'],shuffle=False))).items()}
            model=_model_factory(prepared,cfg).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
            outputs=model(batch['c0'],batch['a0'],batch['s0'],cfg['training']['train_samples'])
            for key,value in outputs.items():
                if torch.is_tensor(value):assert torch.isfinite(value).all(),key
            loss,components=total_loss(outputs,batch,cfg['training'],model.risk_indices)
            assert torch.isfinite(loss);loss.backward()
            active=[];inactive=[]
            for name,parameter in model.named_parameters():
                if parameter.grad is None:inactive.append(name)
                else:assert torch.isfinite(parameter.grad).all(),name;active.append(name)
            if variant=='FrozenGraph':
                assert torch.equal(outputs['graph_path'],batch['a0'][:,None].expand_as(outputs['graph_path']))
                assert all(p.grad is None for m in [model.similarity_encoder,model.opposition_encoder] for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip'])
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            if device.type=='cuda':torch.cuda.synchronize()
            records.append({'fold':fold,'variant':variant,'training_batch_size':len(batch['y']),
              'training_scenarios':cfg['training']['train_samples'],'all_outputs_gradients_parameters_finite':True,
              'frozen_graph_identity_verified':variant=='FrozenGraph','active_gradient_parameters':active,
              'inactive_gradient_parameters':inactive,'initial_loss_components':components,
              'elapsed_seconds':time.perf_counter()-started,'prepared_sha256':prepared_hashes(args.prepared_root,fold),
              'guard':{k:v for k,v in guard.items() if k!='retained_original_indices'}})
            del model,outputs,loss,optimizer,batch
            if device.type=='cuda':torch.cuda.empty_cache()
            print(json.dumps({'event':'preflight_pass','fold':fold,'variant':variant}),flush=True)
    report={'status':'passed','scope':'Six independent one-update TRAIN-batch smoke tests; no fitted study checkpoints or selected/test scores',
      'torch':torch.__version__,'device':args.device,'threads':args.threads,'source_sha256':source,
      'implementation_sha256':implementation_hashes(),'preflight_sha256':sha256(__file__),
      'all_15_configurations_compatible':True,'nominal_and_unique_candidate_counts_verified':True,
      'configurations':fingerprints,'checks':records,'test_access_count':0}
    dump(report,args.out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
