#!/usr/bin/env python3
"""Seed market_session_calendar 2020-2027 + complete asset_daily_ohlcv (COP, SPX500).

Contract: CTR-MKT-CANON-001 (migrations 060/061)

The calendar is THE arbiter of closed-vs-missing in the wide views. It reuses the calendars
already in the repo rather than inventing new ones:
  - Colombia: airflow/dags/utils/trading_calendar.TradingCalendar (colombian_holidays pkg —
    the same code that resolved July 20 live in the freshness gate).
  - NYSE: the `holidays` package US calendar + DST-aware open/close (09:30-16:00
    America/New_York folded to UTC per date — this is where New York DST lives, as data).
  - 24/7 (BTC) and metals (XAU): every day / every weekday respectively.

Also completes the daily long table so the daily wide view covers all 4 assets:
  - usdcop  <- seeds/latest/usdcop_daily_ohlcv.parquet (source='daily_native')
  - spx500  <- data/snapshots/public_daily/spx500_daily.parquet (source='yahoo_snapshot';
               close = adj_close, total-return per SDD-000 §4; available_at from the
               snapshot's own column — the ONE place historic PIT is real, not fabricated)

Idempotent: calendar upserts on PK, daily rows upsert ON CONFLICT DO NOTHING.
Run (host):  POSTGRES_HOST=localhost POSTGRES_PASSWORD=... python scripts/ops/seed_session_calendar.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
START, END = date(2020, 1, 1), date(2027, 12, 31)
BOG, NY = ZoneInfo("America/Bogota"), ZoneInfo("America/New_York")


def _colombia_calendar():
    spec = importlib.util.spec_from_file_location(
        "trading_calendar", REPO / "airflow" / "dags" / "utils" / "trading_calendar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TradingCalendar()


def _nyse_holidays():
    """NYSE trading holidays, NOT the federal calendar.

    NYSE = federal MINUS Columbus Day and Veterans Day (the exchange trades both)
    PLUS Good Friday (closed, but not a federal holiday). Verified against the wide
    view: the federal calendar mislabeled 8 traded Columbus/Veterans days as closed
    and 2 Good Fridays as missing.
    """
    try:
        import holidays
    except ImportError:
        return set()
    from dateutil.easter import easter

    fed = holidays.US(years=range(2020, 2028))
    nyse = {d for d, name in fed.items()
            if "Columbus" not in name and "Veterans" not in name}
    nyse |= {easter(y) - timedelta(days=2) for y in range(2020, 2028)}  # Good Friday
    # Historical quirks (verified against traded bars in the wide view):
    nyse -= {date(2021, 6, 18)}    # Juneteenth federal since 2021 but NYSE first observed 2022
    nyse -= {date(2021, 12, 31)}   # NYSE rule: Jan 1 on a Saturday is NOT observed Friday
    nyse |= {date(2025, 1, 9)}     # national day of mourning (Jimmy Carter) — NYSE closed
    return nyse


def rows() -> list[tuple]:
    col = _colombia_calendar()
    us = _nyse_holidays()
    out = []
    d = START
    while d <= END:
        wd = d.weekday()
        # usdcop: Colombian calendar, 08:00-12:55 COT
        trading = wd < 5 and not col.is_colombian_holiday(d)
        out.append(("usdcop", d, trading,
                    datetime.combine(d, time(8, 0), BOG) if trading else None,
                    datetime.combine(d, time(12, 55), BOG) if trading else None))
        # spx500: NYSE calendar, 09:30-16:00 NY (DST folds into the UTC instants here)
        trading = wd < 5 and d not in us
        out.append(("spx500", d, trading,
                    datetime.combine(d, time(9, 30), NY) if trading else None,
                    datetime.combine(d, time(16, 0), NY) if trading else None))
        # btcusdt: 24/7
        out.append(("btcusdt", d, True,
                    datetime.combine(d, time(0, 0), ZoneInfo("UTC")),
                    datetime.combine(d + timedelta(days=1), time(0, 0), ZoneInfo("UTC"))))
        # xauusd: metals trade Sun ~22:00 UTC -> Fri ~21:00 UTC; only SATURDAY is closed.
        # (384 real Sunday daily bars existed vs the old weekday rule — the calendar was
        # wrong on one side, not the data.)
        trading = wd != 5
        xau_open = (datetime.combine(d, time(22, 0), ZoneInfo("UTC")) if wd == 6
                    else datetime.combine(d, time(0, 0), ZoneInfo("UTC")))
        out.append(("xauusd", d, trading,
                    xau_open if trading else None,
                    datetime.combine(d + timedelta(days=1), time(0, 0), ZoneInfo("UTC"))
                    if trading else None))
        d += timedelta(days=1)
    return out


def main() -> int:
    import psycopg2
    from psycopg2.extras import execute_values
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))
    cur = conn.cursor()

    cal = rows()
    execute_values(cur, """
        INSERT INTO market_session_calendar
            (asset_id, session_date, is_trading_day, session_open_utc, session_close_utc)
        VALUES %s
        ON CONFLICT (asset_id, session_date) DO UPDATE SET
            is_trading_day = EXCLUDED.is_trading_day,
            session_open_utc = EXCLUDED.session_open_utc,
            session_close_utc = EXCLUDED.session_close_utc
    """, cal, page_size=2000)
    print(f"calendario: {len(cal)} filas upsert (4 activos x {END.year - START.year + 1} anos)")

    # -- daily completion -----------------------------------------------------
    def upsert_daily(df, symbol, source, available_col=None):
        vals = []
        for _, r in df.iterrows():
            vals.append((r["time"], symbol, float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"]), float(r.get("volume", 0) or 0),
                         source, r[available_col] if available_col else None))
        execute_values(cur, """
            INSERT INTO asset_daily_ohlcv (time, symbol, open, high, low, close, volume,
                                           source, available_at)
            VALUES %s ON CONFLICT (time, symbol) DO NOTHING
        """, vals, page_size=2000)
        return len(vals)

    cop = pd.read_parquet(REPO / "seeds/latest/usdcop_daily_ohlcv.parquet")
    cop["time"] = pd.to_datetime(cop["time"])
    n1 = upsert_daily(cop.sort_values("time"), "USD/COP", "daily_native")

    spx = pd.read_parquet(REPO / "data/snapshots/public_daily/spx500_daily.parquet")
    spx = spx.rename(columns={"timestamp": "time"}).sort_values("time")
    # Total-return series drives close (SDD-000 §4): using price-only inflates alpha ~1.8%/yr.
    # Back-adjust the WHOLE bar by the same factor — an adj_close glued onto unadjusted
    # O/H/L lands outside [low, high] and (correctly) trips the OHLC coherence check in
    # market_ohlcv_daily (1,498 'incoherent' rows before this fix).
    factor = spx["adj_close"] / spx["close"]
    for col in ("open", "high", "low"):
        spx[col] = spx[col] * factor
    spx["close"] = spx["adj_close"]
    # Repair path: earlier seeds wrote unadjusted O/H/L; replace SPX rows wholesale.
    cur.execute("DELETE FROM asset_daily_ohlcv WHERE symbol = 'SPX500'")
    n2 = upsert_daily(spx, "SPX500", "yahoo_snapshot", available_col="available_at")

    conn.commit()
    print(f"asset_daily_ohlcv: +{n1} COP, +{n2} SPX500 (ON CONFLICT DO NOTHING)")

    cur.execute("SELECT symbol, COUNT(*), MIN(time)::date, MAX(time)::date "
                "FROM asset_daily_ohlcv GROUP BY symbol ORDER BY symbol")
    for row in cur.fetchall():
        print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
