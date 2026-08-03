import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_market_data_statistics_audit_runs_and_is_conservative():
    p = subprocess.run([sys.executable, "scripts/analysis/audit_market_data_statistics.py"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    report = json.loads((ROOT / ".claude/codex/evidence/market-data-statistics.json").read_text())
    assert {r["asset"] for r in report["snapshots"]} == {"usdcop", "xauusd", "btcusdt", "spx500"}
    assert report["decision"] == "REVIEW_REQUIRED"
    assert all(r["pit_columns"]["available_at"] for r in report["snapshots"])
