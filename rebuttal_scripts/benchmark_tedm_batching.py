"""Measure validation inference throughput only; no forecast-score selection."""
import argparse,json,sys,time
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();sys.path.insert(0,str(root/'official'));sys.path.insert(0,str(root/'official/upstream/tedm'))
import stable_tedm_baselines as runner
from tedm_noise_scale_adapter import install_noise_scale_adapter
from models.noise import TEDM
install_noise_scale_adapter(TEDM)
protocol=json.loads((root/'rebuttal/fresh_tedm_stable/protocol.json').read_text())
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
runner.seed_all(123);device=torch.device('cuda')
model=runner.NativeModel('tedm',device,protocol['configuration'])
data,_=runner.load_data(root/'official');positions=data['val_positions'][:128]
runner.tedm_predict(model,data,positions[:8],8,8,8,271828)
results=[]
for batch in [8,32,64]:
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    mu,samples=runner.tedm_predict(model,data,positions,64,batch,64,271828)
    torch.cuda.synchronize()
    row={'origin_batch':batch,'validation_origins':128,'scenarios':64,
         'seconds':time.perf_counter()-started,'peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6,
         'shape':list(samples.shape),'score_computed':False}
    results.append(row);print(json.dumps(row),flush=True)
    del mu,samples
(root/'rebuttal/tedm_batching_benchmark.json').write_text(json.dumps(results,indent=2))
