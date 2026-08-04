"""Asset-generic OHLCV ingestion (SPEC-01 + SPEC-02 intermediate).

Driven entirely by an ``AssetProfile`` (config/assets/<asset>.yaml) — NOT hardcoded to any
symbol, so it scales to Gold, BTC, or any future asset by adding a profile. For Gold this:

  1. Downloads XAU/USD **5-min** from TwelveData (paginated back to the intraday floor; the
     free/basic tier only serves a recent window — we grab whatever exists).
  2. Downloads XAU/USD **daily** deep history from TwelveData (2004+), with **Investing.com**
     as a best-effort cross-check (graceful degradation if CloudFlare blocks).
  3. **Aligns timezones**: 5-min stored at canonical UTC instants; daily stored at the NY-close
     instant (17:00 America/New_York) per SPEC-02.
  4. **Quality audit** (dupes, NaN, high>=low, OHLC integrity, price range) — raises on critical.
  5. Writes **scalable seeds** (`seeds/latest/<asset>_m5_ohlcv.parquet`, `..._daily_ohlcv.parquet`)
     with the same `[time,symbol,open,high,low,close,volume]` schema as the FX pairs.
  6. **Idempotent UPSERT**: 5-min -> `usdcop_m5_ohlcv` (multi-pair table, symbol=profile.symbol);
     daily -> `asset_daily_ohlcv` (multi-asset daily table, migration 051).

Usage:
    python scripts/data/ingest_asset_ohlcv.py --asset xauusd
    python scripts/data/ingest_asset_ohlcv.py --asset xauusd --no-db          # parquet only
    python scripts/data/ingest_asset_ohlcv.py --asset xauusd --skip-intraday  # daily only
    python scripts/data/ingest_asset_ohlcv.py --asset xauusd --daily-start 2004-01-01

Contract: CTR-L0-ASSET-INGEST-001
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]  # scripts/data/<this> -> repo root (reorg fix)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("asset_ingest")

UTC = ZoneInfo("UTC")
TD_URL = "https://api.twelvedata.com/time_series"
MAX_BARS = 5000

# Granularity → DB table (SSOT for the storage-routing convention, migrations 040 + 051).
# One table per granularity, multi-asset via the `symbol` column — NOT per-asset tables. A new
# granularity (e.g. 1h) is a new entry here + a migration; see architecture-overview.md roadmap.
GRANULARITY_TABLE = {
    "5min": "usdcop_m5_ohlcv",     # multi-pair 5-min table (legacy name kept; symbol-discriminated)
    "daily": "asset_daily_ohlcv",  # multi-asset daily table (migration 051)
}
RATE_DELAY = 1.2

# ------------------------------------------------------------------ AssetProfile loader
def _load_asset_profile(asset_id: str):
    p = REPO / "src" / "contracts" / "asset_profile.py"
    spec = importlib.util.spec_from_file_location("asset_profile", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asset_profile"] = mod
    spec.loader.exec_module(mod)
    return mod.load_asset_profile(asset_id)


def _load_validator():
    """Load the standalone OHLCV validator (leaf module: pandas + AssetProfile only)."""
    p = REPO / "src" / "data_quality" / "ohlcv_validators.py"
    spec = importlib.util.spec_from_file_location("ohlcv_validators", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ohlcv_validators"] = mod  # required so dataclasses can resolve string annotations
    spec.loader.exec_module(mod)
    return mod


def _gate_seed(df: pd.DataFrame, profile, granularity: str, validate: bool) -> None:
    """Weekday/gap/tz/OHLC gate BEFORE a seed is written (CTR-DQ-OHLCV-001).

    This is the check that would have caught the Gold Sunday-pile-up day-shift. Off-session
    weekday bars are a hard ERROR → raises (corrupt data never reaches the seed). ``--no-validate``
    downgrades to a warning-only log for emergencies.
    """
    if df is None or df.empty:
        return
    try:
        rep = _load_validator().validate_ohlcv_seed(df, profile, granularity)
    except Exception as e:  # never let the validator itself break ingestion
        log.warning("  validate: skipped (%s: %s)", type(e).__name__, e)
        return
    log.info("  %s", rep.summary_line())
    for iss in rep.warnings:
        log.warning("  validate: %s", iss)
    for iss in rep.errors:
        log.error("  validate: %s", iss)
    if validate:
        rep.raise_if_failed()


# ------------------------------------------------------------------ env / keys
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


def _api_keys() -> list[str]:
    keys = [os.environ.get(f"TWELVEDATA_API_KEY_{i}") for i in range(1, 9)]
    keys.append(os.environ.get("TWELVEDATA_API_KEY"))
    return [k for k in keys if k]


class _KeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.i = 0

    def next(self) -> str:
        if not self.keys:
            raise RuntimeError("No TWELVEDATA_API_KEY_* configured in .env")
        k = self.keys[self.i % len(self.keys)]
        self.i += 1
        return k


# ------------------------------------------------------------------ TwelveData fetch
def _td_fetch(rot: _KeyRotator, symbol: str, interval: str, start: str, end: str,
              tz: str = "UTC") -> pd.DataFrame:
    """One TwelveData time_series call. Returns tz-aware (UTC) OHLCV frame (may be empty)."""
    params = {
        "symbol": symbol, "interval": interval, "format": "JSON",
        "timezone": tz, "apikey": rot.next(), "start_date": start,
        "end_date": end, "outputsize": MAX_BARS,
    }
    try:
        r = requests.get(TD_URL, params=params, timeout=40)
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # network / json
        log.warning("  TD call failed %s %s..%s: %s", symbol, start, end, e)
        return pd.DataFrame()
    if d.get("status") == "error" or d.get("code") not in (None, 200):
        msg = d.get("message", "")
        if "No data" not in msg:
            log.warning("  TD %s %s..%s -> %s", symbol, start, end, msg)
        return pd.DataFrame()
    vals = d.get("values") or []
    if not vals:
        return pd.DataFrame()
    rows = []
    for b in vals:
        rows.append({
            "time": b["datetime"],
            "open": float(b["open"]), "high": float(b["high"]),
            "low": float(b["low"]), "close": float(b["close"]),
            "volume": float(b["volume"]) if b.get("volume") not in (None, "") else 0.0,
        })
    df = pd.DataFrame(rows)
    # datetime returned in requested tz -> localize to that tz, convert to UTC
    ts = pd.to_datetime(df["time"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    df["time"] = ts.dt.tz_convert("UTC")
    return df.drop_duplicates("time")


def _paginate_back(rot: _KeyRotator, symbol: str, interval: str, floor: date,
                   max_calls: int, label: str) -> pd.DataFrame:
    """Walk end_date backward until the API stops returning older data or floor reached."""
    frames: list[pd.DataFrame] = []
    end = date.today() + timedelta(days=1)
    oldest_seen: datetime | None = None
    for call in range(max_calls):
        start = max(floor, end - timedelta(days=3650))
        df = _td_fetch(rot, symbol, interval, start.isoformat(), end.isoformat())
        if df.empty:
            log.info("  [%s] call %d: empty (%s..%s) -> stop", label, call + 1, start, end)
            break
        frames.append(df)
        batch_oldest = df["time"].min().to_pydatetime()
        log.info("  [%s] call %d: %d bars, oldest=%s", label, call + 1, len(df),
                 batch_oldest.date())
        # converged? oldest not moving back
        if oldest_seen is not None and batch_oldest >= oldest_seen:
            break
        oldest_seen = batch_oldest
        if batch_oldest.date() <= floor:
            break
        end = batch_oldest.date()  # next window ends where this one started
        _time.sleep(RATE_DELAY)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ Investing.com daily
def _investing_daily(instrument_id: int, symbol: str, start: date, end: date,
                     referer: str = "https://www.investing.com/commodities/gold-historical-data",
                     max_chunks: int | None = 6, *, fail_closed: bool = False) -> pd.DataFrame:
    """Fetch Investing.com daily OHLC in API-safe chunks.

    Authoritative profiles set ``fail_closed`` so a blocked or incomplete chunk aborts the run;
    a different vendor can never be silently promoted as fallback data.
    """
    def _num(x):
        if x in (None, "", "-"):
            return None
        try:
            return float(str(x).replace(",", ""))
        except (ValueError, TypeError):
            return None

    try:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "USDCOPResearchBot/1.0 (+local-research)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        })
        url = f"https://api.investing.com/api/financialdata/historical/{instrument_id}"
        # Eighteen calendar years remain below the endpoint's ~5,000 daily-row ceiling.
        chunks, cs = [], start
        while cs <= end:
            ce = min(cs + timedelta(days=6570), end)
            chunks.append((cs, ce))
            cs = ce + timedelta(days=1)
        if max_chunks is not None:
            chunks = chunks[-max_chunks:]
        all_rows = []
        for cstart, cend in chunks:
            r = s.get(url, params={"start-date": cstart.isoformat(), "end-date": cend.isoformat(),
                                   "time-frame": "Daily", "add-missing-rows": "false"},
                      headers={"Accept": "application/json", "Referer": referer, "domain-id": "www"},
                      timeout=90)
            if r.status_code != 200:
                message = f"Investing HTTP {r.status_code} on {cstart}..{cend}"
                if fail_closed:
                    raise RuntimeError(message)
                log.info("  [investing] %s -> cross-check chunk skipped", message)
                if r.status_code in (403, 429):
                    break
                continue
            payload = r.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if not rows and fail_closed:
                raise RuntimeError(f"Investing returned no daily rows on {cstart}..{cend}")
            for it in rows:
                dpart = str(it.get("rowDateTimestamp", it.get("rowDate", "")))
                dpart = dpart.split("T")[0] if "T" in dpart else dpart[:10]
                try:
                    t = pd.Timestamp(dpart)
                except Exception:
                    continue
                o = _num(it.get("last_openRaw", it.get("last_open")))
                h = _num(it.get("last_maxRaw", it.get("last_max")))
                lo = _num(it.get("last_minRaw", it.get("last_min")))
                c = _num(it.get("last_closeRaw", it.get("last_close")))
                if None in (o, h, lo, c):
                    continue
                volume = _num(it.get("volumeRaw", it.get("volume"))) or 0.0
                all_rows.append({"time": t, "open": o, "high": h, "low": lo,
                                 "close": c, "volume": volume})
            _time.sleep(RATE_DELAY)
        if not all_rows:
            if fail_closed:
                raise RuntimeError(f"Investing returned no daily data for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df["time"] = df["time"].dt.tz_localize("America/New_York", nonexistent="shift_forward",
                                               ambiguous="NaT").dt.tz_convert("UTC")
        df = df.dropna(subset=["time"]).drop_duplicates("time")
        mode = "authoritative" if fail_closed else "cross-check"
        log.info("  [investing] fetched %d daily bars (%s, id=%d)", len(df), mode, instrument_id)
        return df
    except Exception as e:
        if fail_closed:
            raise RuntimeError(f"Authoritative Investing daily fetch failed for {symbol}: {e}") from e
        log.info("  [investing] failed (%s) -> cross-check skipped", type(e).__name__)
        return pd.DataFrame()


# ------------------------------------------------------------------ session filter + tz align
def _filter_session(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Keep only session days (weekday set). Metals ~23h => no intraday window cut, just weekends."""
    if df.empty:
        return df
    sess = profile.session
    local = df["time"].dt.tz_convert(ZoneInfo(sess.timezone))
    mask = local.dt.dayofweek.isin(list(sess.days))
    if sess.mode == "exchange_hours" and sess.open and sess.close:
        oh, om = map(int, sess.open.split(":"))
        ch, cm = map(int, sess.close.split(":"))
        mins = local.dt.hour * 60 + local.dt.minute
        mask &= (mins >= oh * 60 + om) & (mins <= ch * 60 + cm)
    out = df[mask].copy()
    dropped = len(df) - len(out)
    if dropped:
        log.info("  session filter: dropped %d off-session bars (mode=%s)", dropped, sess.mode)
    return out


def _daily_to_nyclose(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily bars to the NY-close instant (17:00 ET) in UTC (SPEC-02).

    The *calendar date* of a daily bar is the trading day it belongs to, and that day is carried
    by the source label in **UTC space**: TwelveData daily bars arrive at 00:00 UTC on the label
    date (see ``_td_fetch``); the Investing cross-check localizes the bare date to ET → 04:00/05:00
    UTC — still the same UTC calendar day. So we anchor the date with ``tz_convert("UTC").normalize()``,
    NOT ``tz_convert("America/New_York")``: converting a 00:00-UTC stamp to ET yields 19:00/20:00 of
    the *previous* ET day, and ``normalize`` then snaps every bar back one calendar day (Mon→Sun) —
    the day-shift bug that piled Gold bars onto Sunday and emptied Friday. Anchoring in UTC is
    correct for both sources because ET-midnight is always the same UTC day (ET = UTC-4/-5, never
    crosses back).

    DST-correct: build the *naive* 17:00 wall-clock for the trading date, THEN localize to ET. Adding
    a Timedelta to a tz-aware midnight would add absolute UTC duration and land at 16:00/18:00 ET on
    the two DST-transition days — the classic SPEC-02 hazard.
    """
    if df.empty:
        return df
    d = df.copy()
    # Anchor the trading day in UTC (the source-label date), never in ET (see docstring).
    naive_date = d["time"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    naive_close = naive_date + pd.Timedelta(hours=17)  # pure wall-clock 17:00 (no DST math)
    d["time"] = (naive_close
                 .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous=True)
                 .dt.tz_convert("UTC"))
    return d.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _daily_to_session_close(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Normalize daily labels to the profile's local market-close instant."""
    if df.empty:
        return df
    raw_session = profile.raw.get("session", {})
    close_tz = raw_session.get("daily_close_tz") or profile.session.timezone
    if raw_session.get("daily_close_tz"):
        close_text = raw_session.get("daily_close", "17:00")
    else:
        close_text = profile.session.close or "17:00"
    hour, minute = map(int, close_text.split(":"))
    d = df.copy()
    naive_date = d["time"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    naive_close = naive_date + pd.Timedelta(hours=hour, minutes=minute)
    d["time"] = (naive_close
                 .dt.tz_localize(close_tz, nonexistent="shift_forward", ambiguous=True)
                 .dt.tz_convert("UTC"))
    return d.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ quality audit
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
    rep.update(dup=dup, nan=nan, bad_high_low=bad_hl, bad_ohlc_integrity=bad_int,
               out_of_range=oor,
               earliest=str(df["time"].min()), latest=str(df["time"].max()))
    if dup:
        rep["problems"].append(f"{dup} duplicate timestamps")
    if nan:
        rep["problems"].append(f"{nan} NaN OHLC rows")
    if bad_hl:
        rep["problems"].append(f"{bad_hl} high<low")
    if bad_int:
        rep["problems"].append(f"{bad_int} OHLC-integrity violations")
    if oor > len(df) * 0.02:
        rep["problems"].append(f"{oor} out-of-range closes (>2%)")
    return rep


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("time")
    df = df[df["high"] >= df["low"]]
    bad_integrity = ((df["high"] < df[["open", "close"]].max(axis=1)) |
                     (df["low"] > df[["open", "close"]].min(axis=1)))
    if int(bad_integrity.sum()):
        log.warning("  clean: dropped %d OHLC-integrity row(s)", int(bad_integrity.sum()))
        df = df[~bad_integrity]
    return df.sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ persistence
def _to_seed_schema(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    return out[["time", "symbol", "open", "high", "low", "close", "volume"]]


def _write_seed(df: pd.DataFrame, rel_path: str) -> None:
    path = REPO / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("  seed -> %s (%d rows)", rel_path, len(df))


def _write_seed_manifest(df: pd.DataFrame, rel_path: str, *, provider: str,
                         instrument_id: int, source_url: str) -> None:
    """Write lineage next to an authoritative daily seed."""
    path = REPO / rel_path
    manifest_path = path.with_suffix(".manifest.json")
    payload = {
        "schema_version": 1,
        "provider": provider,
        "authoritative": True,
        "fallback_policy": "fail_closed_no_fallback",
        "instrument_id": instrument_id,
        "source_url": source_url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "rows": len(df),
        "earliest": str(df["time"].min()),
        "latest": str(df["time"].max()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_path": rel_path.replace("\\", "/"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("  lineage -> %s", manifest_path.relative_to(REPO))


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


def _upsert(conn, table: str, df: pd.DataFrame, symbol: str, source: str,
            *, replace_symbols: list[str] | None = None) -> tuple[int, int]:
    """Stage the legacy UPSERT; the caller owns commit/rollback with Fabric writes."""
    from psycopg2.extras import execute_values
    if df.empty:
        return 0, 0
    vals = [
        (r["time"].to_pydatetime(), symbol, float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["volume"]), source)
        for _, r in df.iterrows()
    ]
    cur = conn.cursor()
    try:
        deleted = 0
        if replace_symbols:
            cur.execute(f"DELETE FROM {table} WHERE symbol = ANY(%s)", (replace_symbols,))
            deleted = cur.rowcount
        execute_values(cur, f"""
            INSERT INTO {table} (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                close=EXCLUDED.close, volume=EXCLUDED.volume,
                source=EXCLUDED.source, updated_at=NOW()
        """, vals, page_size=1000)
        return len(vals), deleted
    except Exception as e:
        log.error("  UPSERT %s failed: %s", table, e)
        raise
    finally:
        cur.close()


def _publish_fabric_frame(
    conn,
    df: pd.DataFrame,
    *,
    canonical_symbol: str,
    provider_id: str,
    provider_symbol: str,
    interval_id: str,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Screen and stage raw/canonical/quarantine rows in the caller's transaction."""
    from src.data_quality.ingest_guard import publish_or_declare_gap

    rows = df.to_dict(orient="records")
    result = publish_or_declare_gap(
        conn,
        symbol=canonical_symbol,
        provider_id=provider_id,
        interval_id=interval_id,
        rows=rows,
        source_uri=(
            f"script://ingest_asset_ohlcv/{provider_symbol}/{interval_id}"
        ),
    )
    if result is None:
        log.warning(
            "Fabric quality coverage UNAVAILABLE for %s via %s; "
            "legacy ingestion continues without claiming screening",
            canonical_symbol,
            provider_id,
        )
        return df.copy(), {
            "raw": 0,
            "canonical": 0,
            "quarantined": 0,
            "coverage": "UNAVAILABLE",
        }
    accepted = pd.DataFrame(result.accepted, columns=df.columns)
    return accepted, {
        "raw": result.raw_count,
        "canonical": result.canonical_count,
        "quarantined": result.quarantine_count,
        "coverage": "SCOPED",
    }


# ------------------------------------------------------------------ main
def run(asset_id: str, *, use_db: bool, skip_intraday: bool, skip_daily: bool,
        daily_start: str, intraday_calls: int, daily_calls: int, validate: bool = True) -> dict:
    _load_env()
    profile = _load_asset_profile(asset_id)
    rot = _KeyRotator(_api_keys())
    symbol = profile.symbol
    log.info("=" * 70)
    log.info("INGEST %s (%s) | session=%s/%s | keys=%d",
             symbol, profile.display_name, profile.session.mode, profile.session.timezone,
             len(rot.keys))
    log.info("=" * 70)

    summary: dict = {"asset_id": asset_id, "symbol": symbol}
    raw_source = profile.raw.get("data_source", {})
    if not skip_intraday and raw_source.get("intraday_enabled") is False:
        log.info("[1] Intraday disabled by AssetProfile; daily-only source is enforced")
        skip_intraday = True
        summary["intraday"] = {"status": "disabled_by_profile"}

    # ---- 5-min intraday (TwelveData) ----
    if not skip_intraday:
        log.info("[1] 5-min TwelveData (intraday window)")
        floor = date.today() - timedelta(days=400)  # generous; API caps well inside this
        m5 = _paginate_back(rot, symbol, profile.data_source.interval, floor,
                            intraday_calls, "5min")
        m5 = _filter_session(_clean(m5), profile)
        rep = _audit(m5, profile, "5min")
        log.info("  audit 5min: %s", rep)
        if m5.empty:
            log.warning("  no 5-min data available (TwelveData intraday floor) — continuing")
        else:
            # measure bars_per_day for AssetProfile verification (test A3)
            per_day = m5.groupby(m5["time"].dt.tz_convert(ZoneInfo(profile.session.timezone)).dt.date).size()
            summary["measured_bars_per_day_median"] = int(per_day.median())
            _gate_seed(m5, profile, "5min", validate)
            _write_seed(_to_seed_schema(m5, symbol), profile.data_source.seed_file)
        summary["m5"] = rep
        summary["m5_frame"] = m5

    # ---- daily deep history ----
    if not skip_daily:
        floor = datetime.strptime(daily_start, "%Y-%m-%d").date()
        daily_provider = str(raw_source.get("daily_provider") or
                             profile.data_source.daily_provider or
                             profile.data_source.provider).lower()
        authoritative = daily_provider == "investing"
        inv_id = raw_source.get("investing_pair_id")
        daily_source = "twelvedata_daily"

        if authoritative:
            if not inv_id:
                raise RuntimeError("Investing authoritative profile requires investing_pair_id")
            referer = raw_source.get(
                "investing_referer",
                "https://www.investing.com/indices/us-spx-500-historical-data",
            )
            log.info("[2] Daily Investing.com AUTHORITATIVE (id=%s, from %s)", inv_id, floor)
            dly = _investing_daily(int(inv_id), symbol, floor, date.today(),
                                   referer=referer, max_chunks=None, fail_closed=True)
            dly = _daily_to_session_close(_clean(dly), profile)
            daily_source = "investing_daily"
            summary["daily_authoritative"] = True
        else:
            log.info("[2] Daily TwelveData (deep history from %s)", daily_start)
            dly = _paginate_back(rot, symbol, "1day", floor, daily_calls, "daily")
            dly = _daily_to_nyclose(_clean(dly))

            # Legacy profiles may retain Investing as a best-effort cross-check.
            if inv_id:
                inv = _investing_daily(int(inv_id), symbol, floor, date.today())
                if not inv.empty:
                    inv = _daily_to_nyclose(_clean(inv))
                    ov = pd.merge(dly[["time", "close"]].rename(columns={"close": "td"}),
                                  inv[["time", "close"]].rename(columns={"close": "inv"}), on="time")
                    if not ov.empty:
                        d = ((ov["inv"] / ov["td"] - 1.0).abs() * 100)
                        xrep = {"overlap": len(ov), "median_abs_diff_pct": round(float(d.median()), 3),
                                "max_abs_diff_pct": round(float(d.max()), 3),
                                "flag": "OK" if d.median() < 2.0 else "REVIEW (>2% median divergence)"}
                        log.info("  cross-source (TD vs Investing): %s", xrep)
                        summary["cross_source_agreement"] = xrep
                    merged = pd.concat([dly, inv], ignore_index=True).drop_duplicates("time", keep="first")
                    added = len(merged) - len(dly)
                    if added > 0:
                        log.info("  investing filled %d daily bars TD lacked", added)
                    dly = merged.sort_values("time").reset_index(drop=True)
                    summary["investing_crosscheck_rows"] = len(inv)

        rep = _audit(dly, profile, "daily")
        log.info("  audit daily: %s", rep)

        # post-condition: drop any off-session (weekend) daily bars the sources may still carry.
        # The daily path previously skipped session filtering entirely, so a mis-dated bar could
        # slip through unnoticed (see _daily_to_nyclose day-shift). Weekday mask is cheap insurance.
        dly = _filter_session(dly, profile)
        if not dly.empty:
            _gate_seed(dly, profile, "daily", validate)
            _write_seed(_to_seed_schema(dly, symbol), profile.data_source.daily_seed_file)
            if authoritative:
                _write_seed_manifest(
                    dly,
                    profile.data_source.daily_seed_file,
                    provider="investing.com",
                    instrument_id=int(inv_id),
                    source_url=raw_source.get(
                        "investing_referer",
                        "https://www.investing.com/indices/us-spx-500-historical-data",
                    ),
                )
        summary["daily"] = _audit(dly, profile, "daily_final")
        summary["daily_frame"] = dly
        summary["daily_source"] = daily_source

    # ---- persist to DB (idempotent UPSERT) ----
    if use_db:
        log.info("[3] Atomic Fabric publication + legacy UPSERT (idempotent)")
        conn = _db_conn()
        try:
            if not skip_intraday and isinstance(summary.get("m5_frame"), pd.DataFrame) and not summary["m5_frame"].empty:
                tbl = GRANULARITY_TABLE["5min"]
                accepted, fabric = _publish_fabric_frame(
                    conn,
                    summary["m5_frame"],
                    canonical_symbol=symbol,
                    provider_id=str(profile.data_source.provider).lower(),
                    provider_symbol=str(profile.data_source.provider_symbol),
                    interval_id="PT5M",
                )
                n, _ = _upsert(conn, tbl, accepted, symbol,
                               "twelvedata_" + profile.safe_name)
                log.info("  %s: upserted %d rows (symbol=%s)", tbl, n, symbol)
                summary["db_m5_upserted"] = n
                summary["fabric_m5"] = fabric
            if not skip_daily and isinstance(summary.get("daily_frame"), pd.DataFrame) and not summary["daily_frame"].empty:
                tbl = GRANULARITY_TABLE["daily"]
                replace_symbols = (raw_source.get("replace_db_symbols")
                                   if summary.get("daily_authoritative") else None)
                accepted, fabric = _publish_fabric_frame(
                    conn,
                    summary["daily_frame"],
                    canonical_symbol=symbol,
                    provider_id=str(daily_provider).lower(),
                    provider_symbol=str(profile.data_source.provider_symbol),
                    interval_id="P1D",
                )
                n, deleted = _upsert(conn, tbl, accepted, symbol,
                                     summary.get("daily_source", "twelvedata_daily"),
                                     replace_symbols=replace_symbols)
                log.info("  %s: upserted %d rows (symbol=%s)", tbl, n, symbol)
                summary["db_daily_upserted"] = n
                summary["db_daily_replaced"] = deleted
                summary["fabric_daily"] = fabric
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error("  Atomic DB publication failed: %s", e)
            raise
        finally:
            conn.close()

    # strip frames from returned summary
    summary.pop("m5_frame", None)
    summary.pop("daily_frame", None)
    log.info("DONE %s: %s", symbol, {k: v for k, v in summary.items() if k not in ("m5", "daily")})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Asset-generic OHLCV ingestion (AssetProfile-driven)")
    ap.add_argument("--asset", default="xauusd", help="asset_id (config/assets/<id>.yaml)")
    ap.add_argument("--no-db", action="store_true", help="parquet only, skip DB UPSERT")
    ap.add_argument("--skip-intraday", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--daily-start", default="2004-01-01")
    ap.add_argument("--intraday-calls", type=int, default=6)
    ap.add_argument("--daily-calls", type=int, default=8)
    ap.add_argument("--no-validate", action="store_true",
                    help="downgrade the OHLCV weekday/gap/tz gate to warn-only (emergency escape)")
    a = ap.parse_args()
    run(a.asset, use_db=not a.no_db, skip_intraday=a.skip_intraday, skip_daily=a.skip_daily,
        daily_start=a.daily_start, intraday_calls=a.intraday_calls, daily_calls=a.daily_calls,
        validate=not a.no_validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
