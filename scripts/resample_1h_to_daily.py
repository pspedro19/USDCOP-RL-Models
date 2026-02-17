"""
Resample hourly OHLCV seed to daily bars.

Usage:
    python scripts/resample_1h_to_daily.py

Input:  seeds/latest/usdcop_1h_ohlcv.parquet (6,858 rows, hourly)
Output: seeds/latest/usdcop_daily_ohlcv.parquet (~1,399 rows, daily)
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "seeds" / "latest" / "usdcop_1h_ohlcv.parquet"
    output_path = project_root / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"

    print(f"Loading hourly OHLCV from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Rows: {len(df)}, Range: {df.index.min()} -> {df.index.max()}")

    # Resample to daily: standard OHLCV aggregation
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])  # Drop days with no data (weekends/holidays)

    # Preserve symbol column
    daily['symbol'] = 'USD/COP'

    # Validation
    print(f"\nDaily OHLCV: {len(daily)} rows")
    print(f"  Range: {daily.index.min()} -> {daily.index.max()}")
    print(f"  NaN per column: {daily.isna().sum().to_dict()}")

    # No weekends
    weekday_mask = daily.index.dayofweek < 5
    weekend_count = (~weekday_mask).sum()
    if weekend_count > 0:
        print(f"  WARNING: {weekend_count} weekend rows, removing...")
        daily = daily[weekday_mask]

    # OHLC sanity
    assert (daily['high'] >= daily['low']).all(), "high < low found!"
    assert (daily['high'] >= daily['open']).all(), "high < open found!"
    assert (daily['high'] >= daily['close']).all(), "high < close found!"

    # No NaN in OHLC
    assert daily[['open', 'high', 'low', 'close']].isna().sum().sum() == 0, "NaN in OHLC!"

    # Price range check
    prices = daily[['open', 'high', 'low', 'close']]
    print(f"  Price range: {prices.min().min():.2f} - {prices.max().max():.2f}")
    assert prices.min().min() > 2000, "Price too low for USDCOP"
    assert prices.max().max() < 6000, "Price too high for USDCOP"

    # Save
    daily.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
    print(f"  {len(daily)} daily bars, {len(daily) / 252:.1f} years of data")

    # Show yearly bar counts
    yearly = daily.groupby(daily.index.year).size()
    print(f"\nBars per year:")
    for year, count in yearly.items():
        print(f"  {year}: {count}")


if __name__ == "__main__":
    main()
