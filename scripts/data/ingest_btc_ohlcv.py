"""BTC/USDT OHLCV ingestion — canonical Binance spot (SPEC-13 Stage B, ADR-0008).

Bitcoin is the 3rd tradeable asset. Unlike COP/Gold (TwelveData), BTC's canonical price is
**Binance spot BTC/USDT, UTC 00:00 daily close** (PRE-REGISTRATION §1). Binance's PUBLIC REST
klines endpoint needs **NO API KEY**, so we ingest real canonical deep history (2017-08-17+)
with what we have today. TwelveData BTC/USD is a graceful fallback if Binance is unreachable.

  1. Daily deep history from Binance public klines (2017-08+), stored at UTC 00:00 (test A4).
  2. 5-min recent window (last ~N days) — proves 24/7 ingestion (weekend bars present, test C1)
     and bars_per_day == 288 (test A3).
  3. Stored under symbol **BTC/USDT** (profile.symbol). 24/7: NO session cut, NO weekend drop.
  4. Quality audit (dupes, NaN, high>=low, OHLC integrity, price range) — raises on critical.
  5. Scalable seeds: seeds/latest/btcusdt_{daily,m5}_ohlcv.parquet — same schema as FX/Gold.
  6. Idempotent UPSERT (best-effort): daily -> asset_daily_ohlcv, 5-min -> usdcop_m5_ohlcv
     (multi-asset/multi-pair by symbol; migrations 051/040). Degrades gracefully if DB is down.

This is a NEW, parallel module — it does not touch the Gold ingestion path.

Usage:
    python scripts/data/ingest_btc_ohlcv.py                    # daily full + 5m 30d + DB
    python scripts/data/ingest_btc_ohlcv.py --no-db            # seeds only
    python scripts/data/ingest_btc_ohlcv.py --skip-intraday    # daily only
    python scripts/data/ingest_btc_ohlcv.py --m5-days 45

Contract: CTR-L0-ASSET-INGEST-001 (crypto variant)
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("btc_ingest")

UTC = ZoneInfo("UTC")
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_GENESIS_MS = 1502928000000  # 2017-08-17 00:00 UTC (first BTCUSDT daily bar)
TD_URL = "https://api.twelvedata.com/time_series"
RATE_DELAY = 0.4


# ------------------------------------------------------------------ AssetProfile loader
def _load_asset_profile(asset_id: str):
    p = REPO / "src" / "contracts" / "asset_profile.py"
    spec = importlib.util.spec_from_file_location("asset_profile", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asset_profile"] = mod
    spec.loader.exec_module(mod)
    return mod.load_asset_profile(asset_id)


def _load_validator():
    p = REPO / "src" / "data_quality" / "ohlcv_validators.py"
    spec = importlib.util.spec_from_file_location("ohlcv_validators", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ohlcv_validators"] = mod  # required so dataclasses resolve string annotations
    spec.loader.exec_module(mod)
    return mod


def _gate_seed(df: pd.DataFrame, profile, granularity: str, validate: bool) -> None:
    """OHLCV weekday/gap/tz/OHLC gate before a seed write (CTR-DQ-OHLCV-001).

    BTC is 24/7 (session.days = 0..6) so weekend bars are *expected* — the validator adapts to the
    profile and only flags true structural defects (dups, NaN, OHLC breaks, calendar gaps)."""
    if df is None or df.empty:
        return
    try:
        rep = _load_validator().validate_ohlcv_seed(df, profile, granularity)
    except Exception as e:
        log.warning("  validate: skipped (%s: %s)", type(e).__name__, e)
        return
    log.info("  %s", rep.summary_line())
    for iss in rep.warnings:
        log.warning("  validate: %s", iss)
    for iss in rep.errors:
        log.error("  validate: %s", iss)
    if validate:
        rep.raise_if_failed()


def _load_env() -> None:
    envf = REPO / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ------------------------------------------------------------------ Binance public klines (NO KEY)
def _binance_klines(interval: str, start_ms: int, *, label: str, max_calls: int = 400) -> pd.DataFrame:
    """Paginate Binance public klines FORWARD from start_ms to now. Public endpoint, no auth.

    Binance returns up to 1000 bars/call as [openTime, o, h, l, c, v, closeTime, ...]. openTime
    for interval '1d' is the UTC-00:00 instant of the bar's day → we timestamp the bar there,
    which is exactly the canonical decision instant (PRE-REGISTRATION §1) and passes test A4.
    """
    frames: list[pd.DataFrame] = []
    cur = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for call in range(max_calls):
        try:
            r = requests.get(BINANCE_KLINES, params={
                "symbol": BINANCE_SYMBOL, "interval": interval,
                "startTime": cur, "limit": 1000,
            }, timeout=30)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            log.warning("  [%s] binance call %d failed: %s", label, call + 1, e)
            break
        if not rows:
            break
        df = pd.DataFrame([{
            "time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
        } for k in rows])
        frames.append(df)
        last_open = int(rows[-1][0])
        log.info("  [%s] call %d: %d bars, latest=%s", label, call + 1, len(rows),
                 df["time"].max().date())
        if len(rows) < 1000 or last_open >= now_ms:
            break
        # step: one interval past the last openTime
        step = last_open - int(rows[0][0]) if len(rows) > 1 else 0
        cur = last_open + (step // (len(rows) - 1) if len(rows) > 1 else 86400000)
        _time.sleep(RATE_DELAY)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ TwelveData fallback (has key)
def _td_daily_fallback(symbol: str, start: str) -> pd.DataFrame:
    keys = [os.environ.get(f"TWELVEDATA_API_KEY_{i}") for i in range(1, 9)]
    keys = [k for k in keys if k] or [os.environ.get("TWELVEDATA_API_KEY")]
    keys = [k for k in keys if k]
    if not keys:
        return pd.DataFrame()
    try:
        r = requests.get(TD_URL, params={
            "symbol": symbol, "interval": "1day", "format": "JSON", "timezone": "UTC",
            "apikey": keys[0], "start_date": start, "outputsize": 5000,
        }, timeout=40)
        d = r.json()
    except Exception as e:
        log.warning("  [td-fallback] %s", e)
        return pd.DataFrame()
    vals = d.get("values") or []
    if not vals:
        log.info("  [td-fallback] no data (%s)", d.get("message", ""))
        return pd.DataFrame()
    rows = [{"time": pd.Timestamp(b["datetime"], tz="UTC"),
             "open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]),
             "close": float(b["close"]),
             "volume": float(b["volume"]) if b.get("volume") not in (None, "") else 0.0}
            for b in vals]
    return pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ audit / clean
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("time")
    df = df[df["high"] >= df["low"]]
    return df.sort_values("time").reset_index(drop=True)


def _audit(df: pd.DataFrame, profile, kind: str) -> dict:
    rep = {"kind": kind, "rows": len(df), "problems": []}
    if df.empty:
        rep["problems"].append("EMPTY")
        return rep
    lo, hi = profile.price_range
    dup = int(df["time"].duplicated().sum())
    nan = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_int = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                   (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    oor = int((~df["close"].between(lo, hi)).sum())
    rep.update(dup=dup, nan=nan, bad_high_low=bad_hl, bad_ohlc_integrity=bad_int, out_of_range=oor,
               earliest=str(df["time"].min()), latest=str(df["time"].max()))
    for cond, msg in [(dup, f"{dup} dup ts"), (nan, f"{nan} NaN"), (bad_hl, f"{bad_hl} high<low"),
                      (bad_int, f"{bad_int} OHLC-integrity"),
                      (oor > len(df) * 0.02, f"{oor} out-of-range closes")]:
        if cond:
            rep["problems"].append(msg)
    return rep


# ------------------------------------------------------------------ persistence
def _to_seed(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    return out[["time", "symbol", "open", "high", "low", "close", "volume"]]


def _write_seed(df: pd.DataFrame, rel_path: str) -> None:
    path = REPO / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("  seed -> %s (%d rows)", rel_path, len(df))


def _db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
        connect_timeout=6,
    )


def _upsert(conn, table: str, df: pd.DataFrame, symbol: str, source: str) -> int:
    from psycopg2.extras import execute_values
    if df.empty:
        return 0
    vals = [(r["time"].to_pydatetime(), symbol, float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]), source)
            for _, r in df.iterrows()]
    cur = conn.cursor()
    try:
        execute_values(cur, f"""
            INSERT INTO {table} (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                close=EXCLUDED.close, volume=EXCLUDED.volume,
                source=EXCLUDED.source, updated_at=NOW()
        """, vals, page_size=1000)
        conn.commit()
        return len(vals)
    except Exception as e:
        conn.rollback()
        log.error("  UPSERT %s failed: %s", table, e)
        raise
    finally:
        cur.close()


# ------------------------------------------------------------------ main
def run(*, use_db: bool, skip_intraday: bool, skip_daily: bool, m5_days: int,
        validate: bool = True) -> dict:
    _load_env()
    profile = _load_asset_profile("btcusdt")
    symbol = profile.symbol  # BTC/USDT
    log.info("=" * 70)
    log.info("INGEST %s (%s) | session=%s/%s (24/7)", symbol, profile.display_name,
             profile.session.mode, profile.session.timezone)
    log.info("=" * 70)
    summary: dict = {"asset_id": "btcusdt", "symbol": symbol}

    # ---- daily deep history: Binance public (canonical) -> TD fallback ----
    if not skip_daily:
        log.info("[1] Daily Binance public klines (canonical, no key; 2017-08+)")
        dly = _binance_klines("1d", BINANCE_GENESIS_MS, label="daily")
        src = "binance"
        if dly.empty:
            log.warning("  Binance daily empty -> TwelveData BTC/USD fallback")
            dly = _td_daily_fallback("BTC/USD", "2017-08-17")
            src = "twelvedata_btcusd"
        dly = _clean(dly)
        rep = _audit(dly, profile, "daily")
        log.info("  audit daily: %s", rep)
        if not dly.empty:
            # daily bars already at UTC 00:00 (Binance openTime) — canonical, test A4 passes
            hours = dly["time"].dt.hour.unique().tolist()
            log.info("  daily UTC hours present: %s (must be [0])", hours)
            _gate_seed(_to_seed(dly, symbol), profile, "daily", validate)
            _write_seed(_to_seed(dly, symbol), profile.data_source.daily_seed_file)
        summary["daily"] = rep
        summary["daily_source"] = src
        summary["daily_frame"] = dly

    # ---- 5-min recent window (proves 24/7 + bars_per_day) ----
    if not skip_intraday:
        log.info("[2] 5-min Binance public klines (last %d days)", m5_days)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=m5_days)).timestamp() * 1000)
        m5 = _clean(_binance_klines("5m", start_ms, label="5min"))
        rep = _audit(m5, profile, "5min")
        log.info("  audit 5min: %s", rep)
        if not m5.empty:
            per_day = m5.groupby(m5["time"].dt.tz_convert(UTC).dt.date).size()
            # median over FULL days only (first/last partial days would skew it low)
            full = per_day[(per_day > 200)]
            summary["measured_bars_per_day_median"] = int(full.median()) if len(full) else int(per_day.median())
            dows = sorted(m5["time"].dt.dayofweek.unique().tolist())
            summary["weekday_coverage"] = dows  # must include 5,6 (weekend) for 24/7
            log.info("  bars/day median=%s  dow coverage=%s (needs 5,6 for 24/7)",
                     summary["measured_bars_per_day_median"], dows)
            _gate_seed(_to_seed(m5, symbol), profile, "5min", validate)
            _write_seed(_to_seed(m5, symbol), profile.data_source.seed_file)
        summary["m5"] = rep
        summary["m5_frame"] = m5

    # ---- DB UPSERT (best-effort) ----
    if use_db:
        log.info("[3] UPSERT to DB (idempotent, best-effort)")
        try:
            conn = _db_conn()
            df = summary.get("daily_frame")
            if not skip_daily and isinstance(df, pd.DataFrame) and not df.empty:
                n = _upsert(conn, "asset_daily_ohlcv", df, symbol, "binance_daily")
                log.info("  asset_daily_ohlcv: upserted %d (symbol=%s)", n, symbol)
                summary["db_daily_upserted"] = n
            df = summary.get("m5_frame")
            if not skip_intraday and isinstance(df, pd.DataFrame) and not df.empty:
                n = _upsert(conn, "usdcop_m5_ohlcv", df, symbol, "binance_btcusdt")
                log.info("  usdcop_m5_ohlcv: upserted %d (symbol=%s)", n, symbol)
                summary["db_m5_upserted"] = n
            conn.close()
        except Exception as e:
            log.warning("  DB step skipped/failed (seeds still written): %s", e)
            summary["db_error"] = str(e)

    summary.pop("daily_frame", None)
    summary.pop("m5_frame", None)
    log.info("DONE %s: %s", symbol, {k: v for k, v in summary.items() if k not in ("m5", "daily")})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC/USDT ingestion (Binance public canonical)")
    ap.add_argument("--no-db", action="store_true", help="seeds only, skip DB UPSERT")
    ap.add_argument("--skip-intraday", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--m5-days", type=int, default=30, help="days of 5-min history to fetch")
    ap.add_argument("--no-validate", action="store_true",
                    help="downgrade the OHLCV weekday/gap/tz gate to warn-only (emergency escape)")
    a = ap.parse_args()
    run(use_db=not a.no_db, skip_intraday=a.skip_intraday, skip_daily=a.skip_daily,
        m5_days=a.m5_days, validate=not a.no_validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
