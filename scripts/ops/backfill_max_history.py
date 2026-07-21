#!/usr/bin/env python3
"""Max-history backfill: TwelveData (FX/metals/SPY) + Binance (BTC) to the archive floor.

Contract: CTR-MKT-CANON-001 (migration 063; operator directive 2026-07-22
"cobertura total desde la maxima cantidad de dias posibles").

Probed archive floors (earliest_timestamp, live 2026-07-21):
  M5    USD/COP 2019-12-18 · XAU 2020-03-16 · USD/MXN|BRL ~2020 · BTC (Binance) 2017-08
  1h/4h USD/COP 2019-09 · XAU 2020-01 · SPY 2020-02 · BTC (Binance) 2017-08
  1day  USD/COP 1989 · XAU 1979 · SPY 1993
  1mo   COP 1989 · XAU 1970 · SPY 1993 · BTC 2017

Honesty rules encoded here:
  - BTC comes from BINANCE (BTC/USDT, same series already in the tables), never from
    TwelveData's multi-exchange BTC/USD — different quote currency, silent basis.
  - SPY is stored as SPY (its own series). It never impersonates spx500: the daily
    total-return backbone stays Yahoo adj_close.
  - USD/COP M5 is filtered to the 8:00-12:55 COT session on Colombian trading days
    (validator contract); off-session provider quotes are NOT inserted by backfill.
  - Pre-1991 COP daily has O=H=L=C (reference rate, not a market) — kept, but the
    source label 'twelvedata_daily_deep' marks the provenance.
  - available_at = now() at insert: for a historic backfill, ingestion time IS the
    honest availability bound for this system. ON CONFLICT DO NOTHING — existing
    rows (with their own provenance) always win.
  - Every request window logs a row in market_ingestion_manifest (rows_received vs
    rows_new, latency, source_tz, checksum) — the PIT wiring's first real consumer.

Rate limits: 8 TwelveData keys x 8 credits/min, rotated with per-key cooldown.
Run:  POSTGRES_HOST=localhost POSTGRES_PASSWORD=... python scripts/ops/backfill_max_history.py [--phase m5|native|daily|monthly|btc|all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time as _time
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
UTC = timezone.utc
BOG = ZoneInfo("America/Bogota")

TD_KEYS = [os.environ.get(f"TWELVEDATA_API_KEY_{i}") for i in range(1, 9)]
TD_KEYS = [k for k in TD_KEYS if k]
_key_last: dict[str, list[float]] = {}


def _td_key() -> str:
    """Round-robin key whose 60s window has the fewest hits (8/min/key hard cap)."""
    now = _time.time()
    best, best_n = None, 99
    for k in TD_KEYS:
        hits = [t for t in _key_last.get(k, []) if now - t < 61]
        _key_last[k] = hits
        if len(hits) < best_n:
            best, best_n = k, len(hits)
    if best_n >= 7:
        _time.sleep(10)
        return _td_key()
    _key_last.setdefault(best, []).append(_time.time())
    return best


def td_get(path: str, **params) -> dict:
    params["apikey"] = _td_key()
    url = f"https://api.twelvedata.com/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                out = json.load(r)
            if out.get("code") == 429:
                _time.sleep(15)
                continue
            return out
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            _time.sleep(5 * (attempt + 1))
    return {}


def _conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))


def _manifest(cur, provider, asset, tf, w0, w1, received, new, ms, tz, payload, err=None):
    chk = hashlib.sha256(payload).hexdigest() if payload else None
    cur.execute("""
        INSERT INTO market_ingestion_manifest
          (provider, asset_id, timeframe, window_start, window_end,
           rows_received, rows_new, latency_ms, source_tz, checksum_sha256, error)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (provider, asset, tf, w0, w1, received, new, ms, tz, chk, err))


def _coherent(o, h, l, c) -> bool:
    return h >= max(o, c) and l <= min(o, c) and h >= l


def _upsert(cur, table: str, rows: list[tuple], cols: str, conflict: str) -> int:
    from psycopg2.extras import execute_values
    if not rows:
        return 0
    execute_values(cur, f"INSERT INTO {table} ({cols}) VALUES %s "
                        f"ON CONFLICT ({conflict}) DO NOTHING", rows, page_size=2000)
    return cur.rowcount


def _cop_session_ok(ts: datetime) -> bool:
    lt = ts.astimezone(BOG)
    return lt.weekday() < 5 and dtime(8, 0) <= lt.time() <= dtime(12, 55)


def td_series(conn, symbol: str, interval: str, start: date, end: date,
              table: str, tf_label: str | None = None, session_filter=None,
              window_days: int = 15, source: str | None = None) -> int:
    """Paginate [start, end) in windows, validate, upsert, manifest each window."""
    cur = conn.cursor()
    src = source or f"twelvedata_{interval}_backfill"
    total_new = 0
    w0 = start
    while w0 < end:
        w1 = min(w0 + timedelta(days=window_days), end)
        t0 = _time.time()
        try:
            r = td_get("time_series", symbol=symbol, interval=interval,
                       start_date=w0.isoformat(), end_date=w1.isoformat(),
                       timezone="UTC", outputsize=5000, order="ASC")
        except Exception as e:  # noqa: BLE001
            _manifest(cur, "twelvedata", symbol, tf_label or interval, w0, w1,
                      None, None, int((_time.time() - t0) * 1000), "UTC", b"", str(e)[:300])
            conn.commit()
            w0 = w1
            continue
        vals = r.get("values") or []
        payload = json.dumps(vals, sort_keys=True).encode()
        rows = []
        for v in vals:
            dt = v["datetime"]
            ts = datetime.fromisoformat(dt).replace(tzinfo=UTC)
            o, h, l, c = (float(v["open"]), float(v["high"]),
                          float(v["low"]), float(v["close"]))
            if not _coherent(o, h, l, c):
                continue
            if session_filter and not session_filter(ts):
                continue
            vol = float(v.get("volume") or 0)
            if table == "asset_native_ohlcv":
                rows.append((ts, symbol, tf_label, o, h, l, c, vol, src,
                             datetime.now(UTC)))
            elif table == "asset_daily_ohlcv":
                rows.append((ts, symbol, o, h, l, c, vol, src, datetime.now(UTC)))
            else:  # usdcop_m5_ohlcv
                rows.append((ts, symbol, o, h, l, c, int(vol), src, datetime.now(UTC)))
        if table == "asset_native_ohlcv":
            n = _upsert(cur, table, rows,
                        "time, symbol, tf, open, high, low, close, volume, source, available_at",
                        "time, symbol, tf")
        elif table == "asset_daily_ohlcv":
            n = _upsert(cur, table, rows,
                        "time, symbol, open, high, low, close, volume, source, available_at",
                        "time, symbol")
        else:
            n = _upsert(cur, table, rows,
                        "time, symbol, open, high, low, close, volume, source, available_at",
                        "time, symbol")
        _manifest(cur, "twelvedata", symbol, tf_label or interval, w0, w1,
                  len(vals), n, int((_time.time() - t0) * 1000), "UTC", payload)
        conn.commit()
        total_new += n
        w0 = w1
    print(f"  {symbol} {tf_label or interval}: +{total_new} filas nuevas", flush=True)
    return total_new


def binance_klines(conn, interval: str, table: str, tf_label: str | None,
                   start: datetime) -> int:
    """BTC/USDT from Binance public API (same series as existing rows), 1000/req."""
    cur = conn.cursor()
    total_new, t = 0, int(start.timestamp() * 1000)
    end_ms = int(datetime.now(UTC).timestamp() * 1000)
    step_ms = {"5m": 300_000, "1h": 3_600_000, "4h": 14_400_000, "1M": None}[interval]
    while t < end_ms:
        t0 = _time.time()
        url = ("https://api.binance.com/api/v3/klines?"
               f"symbol=BTCUSDT&interval={interval}&startTime={t}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                kl = json.load(r)
        except Exception as e:  # noqa: BLE001
            _manifest(cur, "binance", "BTC/USDT", tf_label or interval,
                      datetime.fromtimestamp(t / 1000, UTC), None, None, None,
                      int((_time.time() - t0) * 1000), "UTC", b"", str(e)[:300])
            conn.commit()
            _time.sleep(10)
            continue
        if not kl:
            break
        rows = []
        for k in kl:
            ts = datetime.fromtimestamp(k[0] / 1000, UTC)
            o, h, l, c, v = (float(k[1]), float(k[2]), float(k[3]),
                             float(k[4]), float(k[5]))
            if not _coherent(o, h, l, c):
                continue
            if table == "asset_native_ohlcv":
                rows.append((ts, "BTC/USDT", tf_label, o, h, l, c, v,
                             "binance_btcusdt", datetime.now(UTC)))
            else:
                rows.append((ts, "BTC/USDT", o, h, l, c, int(v),
                             "binance_btcusdt", datetime.now(UTC)))
        if table == "asset_native_ohlcv":
            n = _upsert(cur, table, rows,
                        "time, symbol, tf, open, high, low, close, volume, source, available_at",
                        "time, symbol, tf")
        else:
            n = _upsert(cur, table, rows,
                        "time, symbol, open, high, low, close, volume, source, available_at",
                        "time, symbol")
        _manifest(cur, "binance", "BTC/USDT", tf_label or interval,
                  datetime.fromtimestamp(kl[0][0] / 1000, UTC),
                  datetime.fromtimestamp(kl[-1][0] / 1000, UTC),
                  len(kl), n, int((_time.time() - t0) * 1000), "UTC",
                  json.dumps(kl).encode())
        conn.commit()
        total_new += n
        t = kl[-1][0] + (step_ms or 32 * 24 * 3_600_000)
        _time.sleep(0.35)  # ~170 req/min << 1200 weight/min
    print(f"  BTC/USDT {tf_label or interval} (binance): +{total_new} filas nuevas", flush=True)
    return total_new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["m5", "native", "daily", "monthly", "btc", "all"])
    args = ap.parse_args()
    conn = _conn()
    today = date.today()

    if args.phase in ("m5", "all"):
        print("== M5 (TwelveData) ==", flush=True)
        td_series(conn, "USD/COP", "5min", date(2019, 12, 18), date(2020, 1, 3),
                  "usdcop_m5_ohlcv", session_filter=_cop_session_ok,
                  source="twelvedata_backfill")
        td_series(conn, "XAU/USD", "5min", date(2020, 3, 16), today,
                  "usdcop_m5_ohlcv", window_days=12, source="twelvedata_xauusd")
        td_series(conn, "USD/MXN", "5min", date(2020, 1, 1), date(2026, 3, 16),
                  "usdcop_m5_ohlcv", window_days=15, source="twelvedata_backfill")
        td_series(conn, "USD/BRL", "5min", date(2020, 1, 1), date(2026, 3, 16),
                  "usdcop_m5_ohlcv", window_days=15, source="twelvedata_backfill")

    if args.phase in ("native", "all"):
        print("== 1h/4h nativos (TwelveData) ==", flush=True)
        for sym, d0 in (("USD/COP", date(2019, 9, 21)), ("XAU/USD", date(2020, 1, 24)),
                        ("SPY", date(2020, 2, 10)), ("USD/MXN", date(2020, 1, 1)),
                        ("USD/BRL", date(2020, 1, 1))):
            td_series(conn, sym, "1h", d0, today, "asset_native_ohlcv",
                      tf_label="1h", window_days=200)
            td_series(conn, sym, "4h", d0, today, "asset_native_ohlcv",
                      tf_label="4h", window_days=800)

    if args.phase in ("daily", "all"):
        print("== Diario profundo (TwelveData) ==", flush=True)
        td_series(conn, "USD/COP", "1day", date(1989, 11, 8), date(2020, 1, 2),
                  "asset_daily_ohlcv", window_days=5000, source="twelvedata_daily_deep")
        td_series(conn, "XAU/USD", "1day", date(1979, 12, 26), date(2004, 1, 1),
                  "asset_daily_ohlcv", window_days=5000, source="twelvedata_daily_deep")
        td_series(conn, "SPY", "1day", date(1993, 1, 29), today,
                  "asset_daily_ohlcv", window_days=5000, source="twelvedata_daily_deep")
        td_series(conn, "USD/MXN", "1day", date(1990, 1, 1), today,
                  "asset_daily_ohlcv", window_days=5000, source="twelvedata_daily_deep")
        td_series(conn, "USD/BRL", "1day", date(1994, 7, 1), today,
                  "asset_daily_ohlcv", window_days=5000, source="twelvedata_daily_deep")

    if args.phase in ("monthly", "all"):
        print("== Mensual nativo (TwelveData) ==", flush=True)
        for sym, d0 in (("USD/COP", date(1989, 11, 1)), ("XAU/USD", date(1970, 1, 1)),
                        ("SPY", date(1993, 1, 1)), ("USD/MXN", date(1990, 1, 1)),
                        ("USD/BRL", date(1994, 7, 1))):
            td_series(conn, sym, "1month", d0, today, "asset_native_ohlcv",
                      tf_label="1month", window_days=36500)

    if args.phase in ("btc", "all"):
        print("== BTC/USDT (Binance, misma serie) ==", flush=True)
        binance_klines(conn, "5m", "usdcop_m5_ohlcv", None,
                       datetime(2017, 8, 17, tzinfo=UTC))
        binance_klines(conn, "1h", "asset_native_ohlcv", "1h",
                       datetime(2017, 8, 17, tzinfo=UTC))
        binance_klines(conn, "4h", "asset_native_ohlcv", "4h",
                       datetime(2017, 8, 17, tzinfo=UTC))
        binance_klines(conn, "1M", "asset_native_ohlcv", "1month",
                       datetime(2017, 8, 1, tzinfo=UTC))

    cur = conn.cursor()
    for q, lbl in ((
        "SELECT symbol, count(*), min(time)::date, max(time)::date FROM usdcop_m5_ohlcv GROUP BY 1 ORDER BY 1", "M5"),
        ("SELECT symbol, tf, count(*), min(time)::date, max(time)::date FROM asset_native_ohlcv GROUP BY 1,2 ORDER BY 1,2", "NATIVO"),
        ("SELECT symbol, count(*), min(time)::date, max(time)::date FROM asset_daily_ohlcv GROUP BY 1 ORDER BY 1", "DIARIO")):
        cur.execute(q)
        print(f"-- {lbl} --", flush=True)
        for row in cur.fetchall():
            print("  ", row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
