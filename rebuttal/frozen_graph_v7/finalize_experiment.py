"""Post-training analysis and verified compact archival, with explicit phase records."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--repository-root',type=Path,required=True);a=p.parse_args();root=a.study_root.resolve()
    assert json.loads((root/'orchestration_status.json').read_text())['phase']=='complete'
    report={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':[]}
    path=root/'analysis_archival_status.json'
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    phases=[('analysis',[sys.executable,str(root/'analysis_tools/analyze_frozen_graph.py'),'--study-root',str(root),
      '--output-dir',str(root/'analysis'),'--block-intervals','--replicates','5000']),
      ('compact_archive',[sys.executable,str(root/'implementation/build_compact_archive.py'),'--study-root',str(root),
      '--repository-root',str(a.repository_root.resolve()),'--output',str(root/'frozen_graph_v7_compact.tar.gz')])]
    for name,cmd in phases:
        print('Starting',name,flush=True)
        code=subprocess.run(cmd,cwd=root).returncode
        report['phases'].append({'phase':name,'exit_code':code,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if code:
            report['status']='failed';save();raise SystemExit(code)
        save()
    report['status']='complete';save();print('Analysis and compact archive verified.',flush=True)

if __name__=='__main__':main()
