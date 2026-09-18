#!/usr/bin/env python3
"""Cross-check the TwelveData USD/COP daily artifact against Investing.com.

This does not replace the M5 TwelveData feed. It provides an independent daily
coherence check using the declared Investing instrument 2112. No dotenv file or
API key is read by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.ingest_asset_ohlcv import _investing_daily

DEFAULT_TWELVEDATA = ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
INVESTING_ID = 2112
REFERER = "https://www.investing.com/currencies/usd-cop-historical-data"


def _daily_close(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    time_col = next((c for c in ("time", "datetime", "date") if c in frame.columns), None)
    if time_col is None or "close" not in frame.columns:
        raise ValueError("USD/COP daily artifact requires a timestamp and close column")
    dates = pd.to_datetime(frame[time_col], errors="coerce", utc=True).dt.normalize()
    values = pd.to_numeric(frame["close"], errors="coerce")
    result = pd.Series(values.to_numpy(), index=dates).dropna()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if result.empty:
        raise ValueError("USD/COP daily artifact is empty")
    return result


def audit(path: Path = DEFAULT_TWELVEDATA) -> dict:
    tw = _daily_close(path)
    investing = _investing_daily(
        INVESTING_ID, "USD/COP", tw.index.min().date(), tw.index.max().date(),
        referer=REFERER, max_chunks=None, fail_closed=True,
    )
    inv_dates = investing["time"].dt.tz_convert("UTC").dt.normalize()
    inv = pd.Series(investing["close"].to_numpy(), index=inv_dates)
    inv = inv[~inv.index.duplicated(keep="last")].sort_index()
    common = tw.index.intersection(inv.index)
    if len(common) < 20:
        raise ValueError(f"insufficient common daily observations: {len(common)}")
    diff_pct = ((inv.loc[common] / tw.loc[common] - 1.0).abs() * 100.0)
    return {
        "contract": "CTR-USDCOP-DAILY-CROSS-SOURCE-001",
        "primary_artifact": str(path),
        "primary_source": "twelvedata",
        "validation_source": "investing",
        "investing_instrument_id": INVESTING_ID,
        "primary_rows": int(len(tw)),
        "validation_rows": int(len(inv)),
        "common_rows": int(len(common)),
        "primary_first": str(tw.index.min().date()),
        "primary_last": str(tw.index.max().date()),
        "validation_first": str(inv.index.min().date()),
        "validation_last": str(inv.index.max().date()),
        "median_abs_diff_pct": round(float(diff_pct.median()), 6),
        "p95_abs_diff_pct": round(float(diff_pct.quantile(0.95)), 6),
        "max_abs_diff_pct": round(float(diff_pct.max()), 6),
        "agreement_flag": "OK" if float(diff_pct.median()) <= 2.0 else "REVIEW",
        "network_called": True,
        "secrets_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twelvedata-path", type=Path, default=DEFAULT_TWELVEDATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.twelvedata_path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"agreement_flag": report["agreement_flag"],
                      "common_rows": report["common_rows"],
                      "median_abs_diff_pct": report["median_abs_diff_pct"]}, ensure_ascii=False))
    return 0 if report["agreement_flag"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
