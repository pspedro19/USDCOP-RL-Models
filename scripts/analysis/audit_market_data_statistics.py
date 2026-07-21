"""Descriptive/statistical audit for public snapshots, seeds and macro tables."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/snapshots/public_daily"
OUT = ROOT / ".claude/codex/evidence/market-data-statistics.json"


def audit_frame(asset: str, df: pd.DataFrame) -> dict:
    # Market snapshots use ``timestamp``; the macro seed uses ``fecha``.
    # Keep ``time`` as a compatibility fallback for older extracts.
    ts_col = next((c for c in ("timestamp", "time", "fecha", "date") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{asset}: no temporal column found")
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.sort_values(ts_col)
    numeric = [c for c in d.select_dtypes(include="number").columns if c not in {"asset_id"}]
    stats = {}
    for c in numeric:
        x = pd.to_numeric(d[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        stats[c] = {
            "n": int(len(x)), "missing_pct": round(float(d[c].isna().mean() * 100), 4),
            "min": float(np.min(x)), "p01": float(np.quantile(x, .01)),
            "median": float(np.median(x)), "mean": float(np.mean(x)),
            "std_sample": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "p99": float(np.quantile(x, .99)), "max": float(np.max(x)),
            "skew": float(skew(x, bias=False)) if len(x) > 2 else None,
            "excess_kurtosis": float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else None,
            "jb_pvalue": float(jarque_bera(x).pvalue) if len(x) > 7 else None,
            "zero_pct": float(np.mean(x == 0) * 100),
        }
    delta = d[ts_col].diff().dropna().dt.total_seconds() / 86400
    expected = "24/7_daily" if asset == "btcusdt" else ("macro_daily" if asset == "macro_indicators_daily" else "session_daily")
    return {
        "asset": asset, "rows": int(len(d)), "columns": list(d.columns),
        "observed_frequency": "daily" if delta.median() <= 1.5 else "irregular",
        "expected_frequency": expected,
        "start": d[ts_col].min().isoformat() if len(d) else None,
        "end": d[ts_col].max().isoformat() if len(d) else None,
        "duplicate_timestamps": int(d[ts_col].duplicated().sum()),
        "invalid_timestamps": int(d[ts_col].isna().sum()),
        "median_gap_days": float(delta.median()) if len(delta) else None,
        "p95_gap_days": float(delta.quantile(.95)) if len(delta) else None,
        "gaps_gt_3d": int((delta > 3).sum()) if len(delta) else 0,
        "missing_by_column": {c: round(float(d[c].isna().mean() * 100), 4) for c in d.columns},
        "numeric": stats,
        "pit_columns": {"available_at": "available_at" in d.columns, "release_date": "release_date" in d.columns},
    }


def main() -> int:
    reports = []
    for p in sorted(SNAP.glob("*_daily.parquet")):
        reports.append(audit_frame(p.stem.removesuffix("_daily"), pd.read_parquet(p)))
    macro_path = ROOT / "seeds/latest/macro_indicators_daily.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else None
    macro_report = audit_frame("macro_indicators_daily", macro) if macro is not None else None
    result = {"schema_version": 1, "generated_at": pd.Timestamp.utcnow().isoformat(),
              "snapshots": reports, "macro": macro_report,
              "decision": "REVIEW_REQUIRED",
              "reasons": ["public snapshots use reconstructed availability", "macro vintages require verification"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"assets": len(reports), "output": str(OUT), "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
