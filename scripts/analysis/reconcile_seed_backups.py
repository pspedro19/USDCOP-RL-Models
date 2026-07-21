"""Read-only reconciliation of canonical seeds against backup copies.

Never mutates either input. Timestamps are normalised in memory to UTC so
differences caused solely by timezone representation are visible separately.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {
    "usdcop_m5": (ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet", ROOT / "data/backups/seeds/usdcop_m5_ohlcv_backup.parquet", "time"),
    "macro_daily": (ROOT / "seeds/latest/macro_indicators_daily.parquet", ROOT / "data/backups/seeds/macro_indicators_daily_backup.parquet", "fecha"),
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def audit(name, seed, backup, time_col):
    a, b = pd.read_parquet(seed), pd.read_parquet(backup)
    at = pd.to_datetime(a[time_col], utc=True, errors="coerce")
    bt = pd.to_datetime(b[time_col], utc=True, errors="coerce")
    aset, bset = set(at.dropna()), set(bt.dropna())
    common = aset & bset
    cols_a, cols_b = set(a.columns), set(b.columns)
    return {
        "dataset": name, "seed": str(seed.relative_to(ROOT)), "backup": str(backup.relative_to(ROOT)),
        "seed_sha256": sha(seed), "backup_sha256": sha(backup),
        "seed_rows": len(a), "backup_rows": len(b), "common_timestamps": len(common),
        "seed_only_timestamps": len(aset - bset), "backup_only_timestamps": len(bset - aset),
        "seed_start_utc": at.min().isoformat() if at.notna().any() else None,
        "seed_end_utc": at.max().isoformat() if at.notna().any() else None,
        "backup_start_utc": bt.min().isoformat() if bt.notna().any() else None,
        "backup_end_utc": bt.max().isoformat() if bt.notna().any() else None,
        "seed_duplicate_timestamps": int(at.duplicated().sum()), "backup_duplicate_timestamps": int(bt.duplicated().sum()),
        "invalid_timestamps": int(at.isna().sum() + bt.isna().sum()),
        "columns_seed_only": sorted(cols_a - cols_b), "columns_backup_only": sorted(cols_b - cols_a),
        "timezone_normalized": True,
        "status": "MATCHED" if aset == bset and cols_a == cols_b else "REVIEW_REQUIRED",
    }

def main():
    report = {"generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "read_only": True,
              "pairs": [audit(n, *v) for n, v in PAIRS.items()]}
    out = ROOT / ".claude/codex/evidence/seed-backup-reconciliation.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "statuses": {x["dataset"]: x["status"] for x in report["pairs"]}}, indent=2))

if __name__ == "__main__": main()
