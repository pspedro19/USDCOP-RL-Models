"""
Fill USDBRL 5-min gaps.
Phase 1: TwelveData (daily requests for each gap date)
Phase 2: Yahoo Finance fallback for remaining gaps

Usage:
    python scripts/fill_usdbrl_gaps.py
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SYMBOL_TD = "USD/BRL"
SYMBOL_YF = "USDBRL=X"
INTERVAL = "5min"
COT_OFFSET = -5  # UTC-5
MIN_BARS_PER_DAY = 50  # need at least 50 of 60 expected bars

GAP_DATES_FILE = PROJECT_ROOT / "seeds" / "latest" / "dukascopy" / "usdbrl_gap_dates.txt"
RAW_CSV = PROJECT_ROOT / "seeds" / "latest" / "dukascopy" / "usdbrl_m5.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "seeds" / "latest" / "usdbrl_m5_ohlcv.parquet"


def load_api_keys():
    keys = []
    for i in range(1, 9):
        key = os.environ.get(f"TWELVEDATA_API_KEY_{i}")
        if key:
            keys.append(key)
    for g in range(1, 4):
        for i in range(1, 5):
            key = os.environ.get(f"TWELVEDATA_API_KEY_{g}_{i}")
            if key:
                keys.append(key)
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


class KeyRotator:
    def __init__(self, keys):
        self.keys = keys
        self.idx = 0
        self.cooldown_until = {k: 0 for k in keys}

    def get_key(self):
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.idx]
            self.idx = (self.idx + 1) % len(self.keys)
            if now >= self.cooldown_until[key]:
                return key
        soonest = min(self.cooldown_until.values())
        wait = soonest - now + 1
        if wait > 0:
            print(f"    All keys cooling down, waiting {wait:.0f}s...", flush=True)
            time.sleep(wait)
        return self.get_key()

    def mark_rate_limited(self, key):
        self.cooldown_until[key] = time.time() + 62


def fetch_twelvedata_week(key_rotator, start_date, end_date, max_retries=3):
    """Fetch a date range of 5-min USDBRL from TwelveData. Use DATE-ONLY format."""
    base_url = "https://api.twelvedata.com/time_series"

    for attempt in range(max_retries):
        api_key = key_rotator.get_key()
        params = {
            "symbol": SYMBOL_TD,
            "interval": INTERVAL,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "outputsize": 5000,
            "format": "JSON",
            "timezone": "UTC",
            "apikey": api_key,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()

            if "code" in data and data["code"] == 429:
                key_rotator.mark_rate_limited(api_key)
                continue
            if data.get("status") == "error":
                # 400 = no data for this range, don't retry
                if data.get("code") == 400:
                    return None
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            if "values" not in data or not data["values"]:
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def fetch_yahoo_day(target_date):
    """Fetch one day of 5-min USDBRL from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    ticker = yf.Ticker(SYMBOL_YF)
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)

    try:
        df = ticker.history(start=start, end=end, interval="5m")
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # Rename columns to match
        col_map = {"Datetime": "datetime", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"}
        df = df.rename(columns=col_map)
        if "datetime" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        # Ensure tz-naive UTC
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df[["datetime", "open", "high", "low", "close"]].copy()
    except Exception:
        return None


def fetch_yahoo_batch(dates, batch_days=5):
    """
    Fetch multiple days from Yahoo Finance.
    YF only keeps 5-min data for last ~60 days, so older dates will fail.
    For older dates we try 1h and interpolate is NOT done (just mark as unavailable).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed! Run: pip install yfinance")
        return {}

    results = {}
    # Group dates into small batches
    for i in range(0, len(dates), batch_days):
        batch = dates[i:i + batch_days]
        start = min(batch)
        end = max(batch) + timedelta(days=1)

        try:
            ticker = yf.Ticker(SYMBOL_YF)
            df = ticker.history(start=start, end=end, interval="5m")
            if df is not None and not df.empty:
                df = df.reset_index()
                col_map = {"Datetime": "datetime", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"}
                df = df.rename(columns=col_map)
                if "datetime" not in df.columns and "Date" in df.columns:
                    df = df.rename(columns={"Date": "datetime"})
                if df["datetime"].dt.tz is not None:
                    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

                # Split by date
                df["date"] = df["datetime"].dt.date
                for d, grp in df.groupby("date"):
                    if d in [dd for dd in batch]:
                        results[d] = grp[["datetime", "open", "high", "low", "close"]].copy()
        except Exception as e:
            pass  # silently skip failed batches

        time.sleep(0.5)

    return results


def filter_cot_hours(df):
    """Filter DataFrame to COT 8:00-12:55."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["datetime_cot"] = df["datetime"] + timedelta(hours=COT_OFFSET)
    mask = (
        (df["datetime_cot"].dt.hour >= 8) &
        ((df["datetime_cot"].dt.hour < 12) |
         ((df["datetime_cot"].dt.hour == 12) & (df["datetime_cot"].dt.minute <= 55)))
    )
    return df[mask].drop(columns=["datetime_cot"])


def main():
    # Load gap dates
    gap_dates = []
    with open(GAP_DATES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                gap_dates.append(date.fromisoformat(line))

    print(f"Total gap dates to fill: {len(gap_dates)}")

    # Load existing raw data
    existing = pd.read_csv(RAW_CSV)
    existing["datetime"] = pd.to_datetime(existing["datetime"])
    print(f"Existing raw rows: {len(existing):,}")

    keys = load_api_keys()
    print(f"TwelveData API keys: {len(keys)}")
    key_rotator = KeyRotator(keys)

    # Phase 1: TwelveData (weekly batches for speed)
    print(f"\n{'='*60}")
    print(f"PHASE 1: TwelveData gap filling ({len(gap_dates)} dates)")
    print(f"{'='*60}")

    td_filled = 0
    td_failed_dates = []
    new_rows = []
    gap_dates_set = set(gap_dates)

    # Group gap dates into weekly batches
    weekly_batches = []
    current_batch = [gap_dates[0]]
    for d in gap_dates[1:]:
        if (d - current_batch[0]).days <= 7:
            current_batch.append(d)
        else:
            weekly_batches.append(current_batch)
            current_batch = [d]
    weekly_batches.append(current_batch)

    print(f"  Organized into {len(weekly_batches)} weekly batches")

    for i, batch in enumerate(weekly_batches):
        start_d = batch[0]
        end_d = batch[-1] + timedelta(days=1)

        print(f"  [{i+1}/{len(weekly_batches)}] {start_d} to {batch[-1]} ({len(batch)} gaps)...", end="", flush=True)

        chunk = fetch_twelvedata_week(key_rotator, start_d, end_d)
        if chunk is not None and not chunk.empty:
            # Filter to COT hours
            filtered = filter_cot_hours(chunk)
            if filtered is not None and not filtered.empty:
                # Check which gap dates got filled
                filtered_copy = filtered.copy()
                filtered_copy["dt_cot"] = filtered_copy["datetime"] + timedelta(hours=COT_OFFSET)
                filtered_copy["date_cot"] = filtered_copy["dt_cot"].dt.date
                dates_in_chunk = set(filtered_copy["date_cot"].unique())

                filled_in_batch = dates_in_chunk & gap_dates_set
                unfilled_in_batch = set(batch) - filled_in_batch

                if filled_in_batch:
                    new_rows.append(filtered)
                    td_filled += len(filled_in_batch)
                    print(f" OK (+{len(filled_in_batch)} days, {len(filtered)} rows)", flush=True)
                else:
                    print(f" no gap dates matched", flush=True)

                td_failed_dates.extend(sorted(unfilled_in_batch))
            else:
                td_failed_dates.extend(batch)
                print(f" no COT data", flush=True)
        else:
            td_failed_dates.extend(batch)
            print(f" no data", flush=True)

        time.sleep(0.3)

    td_failed_dates = sorted(set(td_failed_dates))
    print(f"\nTwelveData results: {td_filled} dates filled, {len(td_failed_dates)} failed")

    # Phase 2: Yahoo Finance for remaining gaps
    if td_failed_dates:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Yahoo Finance fallback ({len(td_failed_dates)} dates)")
        print(f"{'='*60}")

        yf_filled = 0
        yf_failed_dates = []

        # Yahoo 5-min only works for recent ~60 days
        recent_cutoff = date.today() - timedelta(days=59)
        recent_dates = [d for d in td_failed_dates if d >= recent_cutoff]
        old_dates = [d for d in td_failed_dates if d < recent_cutoff]

        print(f"  Recent dates (<60 days, 5-min available): {len(recent_dates)}")
        print(f"  Old dates (>60 days, 5-min likely unavailable): {len(old_dates)}")

        # Try recent dates with 5-min
        if recent_dates:
            print(f"\n  Fetching recent 5-min data from Yahoo...")
            yf_results = fetch_yahoo_batch(recent_dates)
            for d, chunk in yf_results.items():
                filtered = filter_cot_hours(chunk)
                if filtered is not None and len(filtered) >= 10:
                    new_rows.append(filtered)
                    yf_filled += 1
                else:
                    yf_failed_dates.append(d)
            # Add dates that weren't returned at all
            for d in recent_dates:
                if d not in yf_results and d not in yf_failed_dates:
                    yf_failed_dates.append(d)

        # For old dates, try anyway (might fail)
        if old_dates:
            print(f"\n  Trying old dates with Yahoo (may fail)...")
            # Process in larger batches
            batch_size = 20
            for i in range(0, len(old_dates), batch_size):
                batch = old_dates[i:i + batch_size]
                if (i // batch_size) % 10 == 0:
                    print(f"    Batch {i//batch_size + 1}/{(len(old_dates) + batch_size - 1)//batch_size}...", flush=True)
                yf_results = fetch_yahoo_batch(batch, batch_days=len(batch))
                for d, chunk in yf_results.items():
                    filtered = filter_cot_hours(chunk)
                    if filtered is not None and len(filtered) >= 10:
                        new_rows.append(filtered)
                        yf_filled += 1
                for d in batch:
                    if d not in yf_results:
                        yf_failed_dates.append(d)

        yf_failed_dates = sorted(set(yf_failed_dates))
        print(f"\nYahoo results: {yf_filled} filled, {len(yf_failed_dates)} still missing")
    else:
        yf_failed_dates = []

    # Combine all data
    print(f"\n{'='*60}")
    print(f"COMBINING DATA")
    print(f"{'='*60}")

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        # Ensure same columns as existing
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = 0 if col == "volume" else np.nan
        new_df = new_df[[c for c in existing.columns if c in new_df.columns]]

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing.copy()

    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"Combined raw rows: {len(combined):,}")

    # Save updated raw CSV
    combined.to_csv(RAW_CSV, index=False)
    print(f"Updated CSV: {RAW_CSV}")

    # Filter to COT and save parquet
    combined["datetime_cot"] = pd.to_datetime(combined["datetime"]) + timedelta(hours=COT_OFFSET)
    mask_wd = combined["datetime_cot"].dt.dayofweek < 5
    mask_hr = (
        (combined["datetime_cot"].dt.hour >= 8) &
        ((combined["datetime_cot"].dt.hour < 12) |
         ((combined["datetime_cot"].dt.hour == 12) & (combined["datetime_cot"].dt.minute <= 55)))
    )
    filtered = combined[mask_wd & mask_hr].copy()

    cols_out = ["datetime"]
    for c in ["open", "high", "low", "close"]:
        if c in filtered.columns:
            cols_out.append(c)
    df_out = filtered[cols_out].copy()
    df_out = df_out.rename(columns={"datetime": "time"})
    if "volume" not in df_out.columns:
        df_out["volume"] = 0
    df_out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Filtered parquet: {OUTPUT_PARQUET} ({len(df_out):,} rows)")

    # Final completeness check
    df_out["time"] = pd.to_datetime(df_out["time"])
    df_out["date"] = (df_out["time"] + timedelta(hours=COT_OFFSET)).dt.date
    days = df_out["date"].nunique()
    bars_per_day = df_out.groupby("date").size()
    good_days = (bars_per_day >= MIN_BARS_PER_DAY).sum()
    print(f"\nFinal: {len(df_out):,} bars, {days} trading days, {good_days} days >= {MIN_BARS_PER_DAY} bars")

    if yf_failed_dates:
        print(f"\nStill missing {len(yf_failed_dates)} dates (no data from either source)")
        # Save remaining gaps
        with open(str(GAP_DATES_FILE).replace('.txt', '_remaining.txt'), 'w') as f:
            for d in yf_failed_dates:
                f.write(f"{d}\n")


if __name__ == "__main__":
    main()
