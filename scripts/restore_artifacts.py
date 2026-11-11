"""Verify and restore snapshot archives from a local directory or a supplied URL."""
import argparse,hashlib,json,tarfile,urllib.request,urllib.parse
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--part',action='append',help='Restore only a named archive; may be repeated.')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--archive-dir',type=Path,help='Directory containing the provided archive parts; originals are retained.')
    source.add_argument('--base-url',help='Explicit artifact download location; no author-hosted location is embedded.')
    args=p.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'artifacts/server_manifest.json').read_text())
    assets=json.loads((root/'artifacts/release_assets.json').read_text())
    selected=set(args.part or [x['asset'] for x in assets])
    known={x['asset'] for x in assets}
    if not selected <= known:raise SystemExit('Unknown asset name')
    if not args.verify_only:
        if args.archive_dir is None and args.base_url is None:
            p.error('Supply --archive-dir PATH or --base-url URL, or use --verify-only for restored files.')
        cache=root/'.artifact_downloads'
        if args.base_url:cache.mkdir(exist_ok=True)
        for item in assets:
            if item['asset'] not in selected:continue
            if Path(item['asset']).name!=item['asset']:
                raise RuntimeError('Invalid archive filename')
            path=(args.archive_dir if args.archive_dir is not None else cache)/item['asset']
            valid=path.is_file() and path.stat().st_size==item['bytes'] and sha256(path)==item['sha256']
            if args.archive_dir is not None and not valid:
                raise RuntimeError('Missing archive or checksum mismatch: '+item['asset'])
            if args.base_url and not valid:
                print('Downloading',item['asset'],flush=True)
                temp=path.with_suffix('.partial')
                address=args.base_url.rstrip('/')+'/'+urllib.parse.quote(item['asset'],safe='')
                urllib.request.urlretrieve(address,temp)
                if temp.stat().st_size!=item['bytes'] or sha256(temp)!=item['sha256']:
                    raise RuntimeError('Archive checksum mismatch: '+item['asset'])
                temp.replace(path)
            print('Extracting',item['asset'],flush=True)
            with tarfile.open(path,'r:gz') as tar:
                for member in tar.getmembers():
                    dest=(root/member.name).resolve()
                    if root not in dest.parents or not member.isfile():
                        raise RuntimeError('Unsafe archive member')
                tar.extractall(root,filter='data')
            if args.base_url:path.unlink()
    checked=0
    for row in manifest['files']:
        if row['storage']=='release' and row['asset'] not in selected:continue
        path=root/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise RuntimeError('File verification failed: '+row['path'])
        checked+=1
    print('Verified',checked,'original snapshot files.')

if __name__=='__main__':main()
