"""
Resample 5-minute OHLCV to 1-hour bars for USDCOP.

Session: 8:00-12:55 COT = 5 hourly bars per day (8:xx, 9:xx, 10:xx, 11:xx, 12:xx)
Output: seeds/latest/usdcop_1h_ohlcv.parquet

Usage:
    python scripts/resample_5m_to_1h.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resample_5m_to_1h(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Resample 5-min OHLCV to 1-hour bars."""
    print(f"Loading 5-min data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Ensure datetime index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

    # Filter to USD/COP only if symbol column exists
    if 'symbol' in df.columns:
        df = df[df['symbol'] == 'USD/COP'].copy()
        df = df.drop(columns=['symbol'])

    # Remove timezone for resampling, then re-add
    original_tz = df.index.tz
    if original_tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    print(f"  Input: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    # Standard OHLCV resampling to 1H
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        agg['volume'] = 'sum'

    df_1h = df.resample('1h').agg(agg).dropna(subset=['close'])

    # Filter to session hours [8, 9, 10, 11, 12] COT
    session_hours = [8, 9, 10, 11, 12]
    df_1h = df_1h[df_1h.index.hour.isin(session_hours)]

    # Remove weekends
    df_1h = df_1h[df_1h.index.dayofweek < 5]

    # Re-add timezone
    if original_tz is not None:
        df_1h.index = df_1h.index.tz_localize(original_tz)

    # Add symbol column
    df_1h['symbol'] = 'USD/COP'

    # Rename index
    df_1h.index.name = 'time'

    # Validate
    print(f"\n  Output: {len(df_1h)} rows")
    print(f"  Date range: {df_1h.index.min()} to {df_1h.index.max()}")

    hours = df_1h.index.tz_localize(None).hour if df_1h.index.tz else df_1h.index.hour
    assert all(h in session_hours for h in hours), f"Hours outside session: {set(hours) - set(session_hours)}"

    weekdays = df_1h.index.tz_localize(None).dayofweek if df_1h.index.tz else df_1h.index.dayofweek
    assert all(d < 5 for d in weekdays), "Weekend bars found!"

    assert df_1h[['open', 'high', 'low', 'close']].isna().sum().sum() == 0, "NaN in OHLC!"
    assert (df_1h['high'] >= df_1h['low']).all(), "high < low found!"

    # Bars per day stats
    if df_1h.index.tz:
        dates = df_1h.index.tz_localize(None).date
    else:
        dates = df_1h.index.date
    bars_per_day = pd.Series(dates).value_counts()
    print(f"  Median bars/day: {bars_per_day.median():.0f}")
    print(f"  Mean bars/day: {bars_per_day.mean():.1f}")

    # Price range sanity
    print(f"  Price range: {df_1h['close'].min():.2f} - {df_1h['close'].max():.2f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_1h.to_parquet(output_path)
    print(f"\n  Saved to {output_path}")

    return df_1h


if __name__ == "__main__":
    input_path = PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
    output_path = PROJECT_ROOT / "seeds/latest/usdcop_1h_ohlcv.parquet"

    df_1h = resample_5m_to_1h(input_path, output_path)
    print(f"\nDone! {len(df_1h)} hourly bars ready.")
