"""
Download USDBRL 5-min OHLCV data from TwelveData API.
Rotates across multiple API keys to maximize throughput.
Filters to Mon-Fri 8:00 AM - 12:55 PM Colombia time (UTC-5).

Usage:
    python scripts/download_usdbrl_twelvedata.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Config ──────────────────────────────────────────────────────────────
SYMBOL = "USD/BRL"
INTERVAL = "5min"
OUTPUT_SIZE = 5000  # max per request
START_DATE = "2020-01-01"
END_DATE = "2026-02-12"
OUTPUT_DIR = PROJECT_ROOT / "seeds" / "latest" / "dukascopy"
OUTPUT_CSV = OUTPUT_DIR / "usdbrl_m5.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "seeds" / "latest" / "usdbrl_m5_ohlcv.parquet"

# Colombia is UTC-5
COLOMBIA_UTC_OFFSET = -5
MARKET_OPEN_HOUR = 8   # 8:00 AM COT
MARKET_CLOSE_HOUR = 12  # last bar at 12:55 PM COT
MARKET_CLOSE_MIN = 55

# ── API Keys ────────────────────────────────────────────────────────────
def load_api_keys():
    """Load all TwelveData API keys from environment."""
    keys = []
    # Legacy keys (1-8)
    for i in range(1, 9):
        key = os.environ.get(f"TWELVEDATA_API_KEY_{i}")
        if key:
            keys.append(key)
    # Group 1 keys (1_1 to 1_4)
    for i in range(1, 5):
        key = os.environ.get(f"TWELVEDATA_API_KEY_1_{i}")
        if key:
            keys.append(key)
    # Group 2 keys (2_1 to 2_4)
    for i in range(1, 5):
        key = os.environ.get(f"TWELVEDATA_API_KEY_2_{i}")
        if key:
            keys.append(key)
    # Group 3 keys (3_1 to 3_4)
    for i in range(1, 5):
        key = os.environ.get(f"TWELVEDATA_API_KEY_3_{i}")
        if key:
            keys.append(key)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


class KeyRotator:
    """Rotate API keys, tracking rate limits per key."""

    def __init__(self, keys):
        self.keys = keys
        self.idx = 0
        self.calls_per_key = {k: 0 for k in keys}
        self.cooldown_until = {k: 0 for k in keys}

    def get_key(self):
        """Get next available key, respecting cooldowns."""
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.idx]
            self.idx = (self.idx + 1) % len(self.keys)
            if now >= self.cooldown_until[key]:
                self.calls_per_key[key] += 1
                return key
        # All keys on cooldown — wait for the soonest
        soonest = min(self.cooldown_until.values())
        wait = soonest - now + 0.5
        if wait > 0:
            print(f"  All keys on cooldown, waiting {wait:.1f}s...")
            time.sleep(wait)
        return self.get_key()

    def mark_rate_limited(self, key):
        """Put a key on 60s cooldown after rate limit hit."""
        self.cooldown_until[key] = time.time() + 62
        print(f"  Key ...{key[-4:]} rate limited, cooldown 62s")


def fetch_chunk(key_rotator, start_dt, end_dt, max_retries=3):
    """
    Fetch one chunk of data from TwelveData.
    Returns DataFrame or None if no data.
    """
    base_url = "https://api.twelvedata.com/time_series"

    for attempt in range(max_retries):
        api_key = key_rotator.get_key()
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": OUTPUT_SIZE,
            "format": "JSON",
            "apikey": api_key,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()

            if "code" in data and data["code"] == 429:
                key_rotator.mark_rate_limited(api_key)
                continue

            if "code" in data and data["code"] != 200:
                print(f"  API error: {data.get('message', data)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

            if "values" not in data or not data["values"]:
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)

    return None


def generate_monthly_ranges(start_date, end_date):
    """Generate (start, end) date pairs for each month."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    ranges = []
    while current < end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(seconds=1)
        if month_end > end:
            month_end = end
        ranges.append((current, month_end))
        current = (month_end + timedelta(seconds=1)).replace(hour=0, minute=0, second=0)

    return ranges


def filter_colombia_hours(df):
    """
    Filter to Mon-Fri 8:00 AM - 12:55 PM Colombia time (UTC-5).
    TwelveData returns data in exchange timezone (BRL = UTC-3 for B3).
    We convert to UTC first, then to Colombia time.
    """
    if df is None or df.empty:
        return df

    # TwelveData forex data is typically in UTC
    # Convert to Colombia time (UTC-5)
    df = df.copy()
    df["datetime_utc"] = df["datetime"]  # assume UTC from TwelveData for forex
    df["datetime_cot"] = df["datetime_utc"] + timedelta(hours=COLOMBIA_UTC_OFFSET)

    # Filter weekdays (Mon=0, Fri=4)
    mask_weekday = df["datetime_cot"].dt.dayofweek < 5

    # Filter hours: 8:00 AM to 12:55 PM COT
    mask_time = (
        (df["datetime_cot"].dt.hour > MARKET_OPEN_HOUR) |
        ((df["datetime_cot"].dt.hour == MARKET_OPEN_HOUR) & (df["datetime_cot"].dt.minute >= 0))
    ) & (
        (df["datetime_cot"].dt.hour < MARKET_CLOSE_HOUR) |
        ((df["datetime_cot"].dt.hour == MARKET_CLOSE_HOUR) & (df["datetime_cot"].dt.minute <= MARKET_CLOSE_MIN))
    )

    filtered = df[mask_weekday & mask_time].copy()
    filtered = filtered.drop(columns=["datetime_cot"], errors="ignore")

    return filtered


def main():
    keys = load_api_keys()
    if not keys:
        print("ERROR: No TwelveData API keys found in .env")
        sys.exit(1)

    print(f"Loaded {len(keys)} API keys")
    print(f"Downloading {SYMBOL} {INTERVAL} from {START_DATE} to {END_DATE}")
    print(f"Filter: Mon-Fri 8:00-12:55 COT (UTC-5)")
    print()

    key_rotator = KeyRotator(keys)
    monthly_ranges = generate_monthly_ranges(START_DATE, END_DATE)

    all_chunks = []
    total_rows = 0

    for i, (start, end) in enumerate(monthly_ranges):
        label = start.strftime("%Y-%m")
        print(f"[{i+1}/{len(monthly_ranges)}] {label} ... ", end="", flush=True)

        chunk = fetch_chunk(key_rotator, start, end)

        if chunk is not None and not chunk.empty:
            all_chunks.append(chunk)
            total_rows += len(chunk)
            print(f"{len(chunk)} rows (total: {total_rows})")
        else:
            print("no data")

        # Small delay between requests to be respectful
        time.sleep(0.3)

    if not all_chunks:
        print("\nERROR: No data downloaded!")
        sys.exit(1)

    # Combine and deduplicate
    print("\nCombining chunks...")
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"Combined: {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")

    # Filter to Colombia market hours
    print("Filtering to Colombia market hours (Mon-Fri 8:00-12:55 COT)...")
    df_filtered = filter_colombia_hours(df)
    print(f"After filter: {len(df_filtered)} rows")

    # Save raw CSV (before filtering, for reference)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Raw CSV saved: {OUTPUT_CSV}")

    # Save filtered parquet
    df_out = df_filtered[["datetime_utc", "open", "high", "low", "close", "volume"]].copy()
    df_out = df_out.rename(columns={"datetime_utc": "time"})
    df_out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Filtered parquet saved: {OUTPUT_PARQUET}")
    print(f"\nDone! {len(df_out)} bars of {SYMBOL} 5-min data")


if __name__ == "__main__":
    main()
