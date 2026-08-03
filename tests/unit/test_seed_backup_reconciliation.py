import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_reconciliation_is_read_only_and_reports_both_pairs():
    subprocess.run([sys.executable, "scripts/analysis/reconcile_seed_backups.py"], cwd=ROOT, check=True)
    report = json.loads((ROOT / ".claude/codex/evidence/seed-backup-reconciliation.json").read_text())
    assert report["read_only"] is True
    assert {x["dataset"] for x in report["pairs"]} == {"usdcop_m5", "macro_daily"}
    assert all(x["seed_sha256"] and x["backup_sha256"] for x in report["pairs"])
