"""Verify portable path adapters against the original audited computations."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=root/'all_radius_flow_results/fold_1_seed_1.npz';b=root/'portable_parity/audit/fold_1_seed_1.npz'
with np.load(a) as za,np.load(b) as zb:
    assert za.files==zb.files
    checks={k:bool(np.array_equal(za[k],zb[k])) for k in za.files}
    assert all(checks.values())
v1=json.loads((root/'validation/fold_1_seed_1.json').read_text());v2=json.loads((root/'portable_parity/validation/fold_1_seed_1.json').read_text());assert v1==v2
report={'passed':True,'actual_checkpoint':'fold_1_seed_1','radius_count':4,'origin_count':32,'lead_count':5,
        'npz_array_exact_equality':checks,'validation_json_exact_equality':True,
        'original_audit_arrays_sha256':sha(a),'portable_audit_arrays_sha256':sha(b),
        'source_hashes':{k:sha(root/k) for k in ['normalized_factor_flow.py','audit_factor_flow.py','reproduce_factor_flow.py','validate_factor_flow.py','reproduce_factor_flow_validation.py']},
        'scope':'Path/argument adapters only; this parity check compares completed real-checkpoint outputs, not fabricated fixtures.'}
(root/'portable_runner_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
