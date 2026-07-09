#!/usr/bin/env python3
"""Backfill macro_charts[*].data in the published weekly analysis JSONs from the DB.

Root cause (2026-07-07): the weekly analysis exporter sourced macro chart history from
`MACRO_DAILY_CLEAN.parquet`, which is stale (maxes 2026-03-17) — so every `macro_charts[k].data`
in `public/data/analysis/weekly_2026_W*.json` shipped empty and `/analysis` rendered "Sin datos".
The DB (`macro_indicators_daily`) is fresh; this patches each week's chart arrays from it,
matching the frontend `MacroChartPoint[]` shape ({date,value,sma20,bb_upper,bb_lower,rsi}) and
`macro_analyzer.get_chart_data` math (SMA20, BB(20,2), RSI(14)). Idempotent, JSON-safe.

Usage: python -m scripts.ops.patch_analysis_macro_charts [--lookback-days 180]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import date, timedelta

import pandas as pd
import psycopg2

ANALYSIS_DIR = os.path.join("usdcop-trading-dashboard", "public", "data", "analysis")

# macro_charts key -> macro_indicators_daily column
KEY_TO_COL = {
    "dxy": "fxrt_index_dxy_usa_d_dxy",
    "vix": "volt_vix_usa_d_vix",
    "wti": "comm_oil_wti_glb_d_wti",
    "embi_col": "crsk_spread_embi_col_d_embi",
    "ust10y": "finc_bond_yield10y_usa_d_ust10y",
    "ibr": "finc_rate_ibr_overnight_col_d_ibr",
    "gold": "comm_metal_gold_glb_d_gold",
    "brent": "comm_oil_brent_glb_d_brent",
}


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "usdcop_trading")
    user = os.environ.get("POSTGRES_USER", "admin")
    pw = os.environ.get("POSTGRES_PASSWORD", "")
    return f"host={host} port={port} dbname={db} user={user} password={pw}"


def _safe(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)


def load_macro(conn) -> pd.DataFrame:
    cols = ", ".join(KEY_TO_COL.values())
    df = pd.read_sql(f"SELECT fecha, {cols} FROM macro_indicators_daily ORDER BY fecha", conn)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")


def series_points(s: pd.Series, end: date, lookback: int) -> list[dict]:
    s = s.dropna()
    if s.empty:
        return []
    sma20 = s.rolling(20, min_periods=1).mean()
    std = s.rolling(20, min_periods=1).std()
    bb_up, bb_lo = sma20 + 2 * std, sma20 - 2 * std
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - 100 / (1 + rs)
    start = pd.Timestamp(end - timedelta(days=lookback))
    mask = (s.index >= start) & (s.index <= pd.Timestamp(end))
    out = []
    for dt in s.index[mask]:
        out.append({
            "date": str(dt.date()),
            "value": _safe(s.get(dt)),
            "sma20": _safe(sma20.get(dt)),
            "bb_upper": _safe(bb_up.get(dt)),
            "bb_lower": _safe(bb_lo.get(dt)),
            "rsi": _safe(rsi.get(dt)),
        })
    return out


def week_end(fname: str, doc: dict) -> date:
    # Prefer the ISO week embedded in the filename: weekly_YYYY_WNN.json → that week's Friday.
    base = os.path.basename(fname)
    try:
        yr = int(base.split("_")[1])
        wk = int(base.split("_")[2].lstrip("W").split(".")[0])
        return date.fromisocalendar(yr, wk, 5)  # Friday
    except (IndexError, ValueError):
        end = doc.get("weekly_summary", {}).get("ohlcv", {}).get("end")
        return date.fromisoformat(end) if end else date.today()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=180)
    args = ap.parse_args()

    conn = psycopg2.connect(_db_url())
    macro = load_macro(conn)
    conn.close()

    files = sorted(glob.glob(os.path.join(ANALYSIS_DIR, "weekly_2026_W*.json")))
    patched = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            doc = json.load(fh)
        mc = doc.get("macro_charts")
        if not isinstance(mc, dict):
            continue
        end = week_end(f, doc)
        filled = 0
        for key, entry in mc.items():
            col = KEY_TO_COL.get(key)
            if not col or col not in macro.columns or not isinstance(entry, dict):
                continue
            pts = series_points(macro[col], end, args.lookback_days)
            if pts:
                entry["data"] = pts
                filled += 1
        if filled:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)
            patched += 1
            print(f"{os.path.basename(f)}: filled {filled}/{len(mc)} charts (week end {end})")
    print(f"\nDONE: patched {patched}/{len(files)} weekly files")


if __name__ == "__main__":
    main()
