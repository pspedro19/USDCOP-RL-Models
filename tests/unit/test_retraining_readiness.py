import json
import subprocess
import sys

def test_retraining_readiness_is_fail_closed():
    p = subprocess.run([sys.executable, "scripts/validation/retraining_readiness.py"], capture_output=True, text=True)
    assert p.returncode == 1
    report = json.loads(p.stdout)
    assert report["cycle"] == "2026"
    assert report["decision"] == "NO-GO"
    assert not report["ready"]
