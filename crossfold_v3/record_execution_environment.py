"""Record software versions without hostnames, credentials, or environment secrets."""
import argparse,hashlib,importlib.metadata,json,platform,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    packages={}
    for name in ['torch','numpy','scipy','pandas','matplotlib','pillow','einops','ema-pytorch','PyYAML','scikit-learn']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    result={'python':platform.python_version(),'system':platform.system(),'architecture':platform.machine(),'packages':packages,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        import torch
        result['torch_compiled_cuda_version']=torch.version.cuda
        result['torch_cudnn_version']=torch.backends.cudnn.version()
    except ImportError:pass
    try:
        result['nvidia_gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    except (OSError,subprocess.SubprocessError):pass
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
