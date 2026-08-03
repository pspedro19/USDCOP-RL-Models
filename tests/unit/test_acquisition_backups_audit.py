import json, subprocess, sys
def test_acquisition_audit_is_reproducible():
    p=subprocess.run([sys.executable,'scripts/analysis/audit_acquisition_backups.py'],capture_output=True,text=True)
    assert p.returncode==0
    r=json.load(open('.claude/codex/evidence/acquisition-backups-audit.json',encoding='utf-8'))
    assert r['summary']['files'] > 0 and r['summary']['readable'] > 0
    assert r['decision'] in ('PASS','REVIEW_REQUIRED')
