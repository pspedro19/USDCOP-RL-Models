"""Validate provider run manifests; absence is an explicit production blocker."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
REQUIRED=("run_id","provider","asset","requested_at","completed_at","rows_received","rows_accepted","rows_rejected","sha256","frequency","timezone","available_at_policy","status")
def main():
    paths=list((ROOT/'data/acquisition/manifests').glob('*.json')) if (ROOT/'data/acquisition/manifests').exists() else []
    errors=[]
    for p in paths:
        try: d=json.loads(p.read_text(encoding='utf-8'))
        except Exception as e: errors.append(f'{p.name}: invalid json {e}'); continue
        miss=[k for k in REQUIRED if k not in d]
        if miss: errors.append(f'{p.name}: missing {miss}')
        if d.get('rows_accepted',0)+d.get('rows_rejected',0)>d.get('rows_received',0): errors.append(f'{p.name}: row accounting mismatch')
        if d.get('status')=='success' and len(str(d.get('sha256',''))) != 64: errors.append(f'{p.name}: invalid sha256')
    result={'schema_version':1,'manifests':len(paths),'errors':errors,'decision':'PASS' if paths and not errors else 'REVIEW_REQUIRED','reason':'provider execution evidence is required for promotion'}
    out=ROOT/'.claude/codex/evidence/acquisition-manifest-validation.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
