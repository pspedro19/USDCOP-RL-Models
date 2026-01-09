"""
L2 Causal Deseasonalization Module

This module provides causal (non-lookahead) deseasonalization for the L2 pipeline.
Instead of using statistics from the entire dataset, it calculates statistics using
only historical data available at each point in time (expanding window).

Key Functions:
- calculate_expanding_hod_stats: Calculate hour-of-day statistics with expanding window
- apply_causal_deseasonalization: Apply deseasonalization using causal statistics

Author: USDCOP Trading System
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_expanding_hod_stats(df, min_samples=30):
    """
    Calculate hour-of-day statistics using a monthly expanding window (SIMPLIFIED & CAUSAL).

    Divides data into monthly periods and calculates HOD stats using only data
    before each period, ensuring no lookahead bias while providing good coverage.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: time, hour_cot, ret_log_1_winsor, range_bps
    min_samples : int
        Minimum number of samples required per hour to calculate statistics

    Returns:
    --------
    pd.DataFrame
        DataFrame with hourly statistics for each period
    """
    logger.info(f"Calculating causal HOD statistics using monthly windows (min_samples={min_samples})...")

    # Sort by time to ensure causality
    df = df.sort_values('time').reset_index(drop=True)

    # Create year-month column for grouping
    df['year_month'] = pd.to_datetime(df['time']).dt.to_period('M')
    unique_periods = sorted(df['year_month'].unique())

    # Get unique hours
    unique_hours = sorted(df['hour_cot'].unique())
    logger.info(f"Processing {len(unique_periods)} monthly periods x {len(unique_hours)} hours...")

    result_rows = []

    # For each month, calculate stats using only data from PREVIOUS months
    for period_idx, current_period in enumerate(unique_periods):
        # For the first month, use data from that month itself (bootstrapping)
        if period_idx == 0:
            historical_mask = df['year_month'] <= current_period
        else:
            # Use only data from previous months (causal!)
            historical_mask = df['year_month'] < current_period

        historical_df = df[historical_mask]

        # Get timestamp from current period for the result
        period_mask = df['year_month'] == current_period
        if period_mask.sum() == 0:
            continue
        period_start_time = df.loc[period_mask, 'time'].iloc[0]

        # Calculate statistics for each hour
        for hour in unique_hours:
            hour_mask = historical_df['hour_cot'] == hour
            hour_data = historical_df[hour_mask]

            if len(hour_data) >= min_samples:
                # Calculate median return
                median_ret = hour_data['ret_log_1_winsor'].median()

                # Calculate MAD (Median Absolute Deviation)
                deviations = np.abs(hour_data['ret_log_1_winsor'] - median_ret)
                mad = deviations.median()

                # Calculate p95 range
                p95_range = hour_data['range_bps'].quantile(0.95)

                result_rows.append({
                    'time': period_start_time,
                    'hour_cot': hour,
                    'median_ret_log_5m': median_ret,
                    'mad_ret_log_5m': mad,
                    'p95_range_bps': p95_range,
                    'sample_count': len(hour_data)
                })

        logger.info(f"Processed period {period_idx + 1}/{len(unique_periods)}: {current_period}")

    result_df = pd.DataFrame(result_rows)

    logger.info(f"Generated {len(result_df)} causal HOD statistics entries ({len(result_df)//len(unique_hours)} periods)")

    return result_df


def apply_causal_deseasonalization(df, causal_hod_df):
    """
    Apply causal deseasonalization to the dataframe using expanding window statistics (OPTIMIZED).

    Uses pandas merge_asof for efficient lookup of the most recent historical statistics
    for each hour, ensuring no lookahead bias.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to deseasonalize (must have: time, hour_cot, ret_log_1_winsor, range_bps)
    causal_hod_df : pd.DataFrame
        Causal HOD statistics from calculate_expanding_hod_stats

    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns: ret_deseason, range_norm
    """
    logger.info("Applying causal deseasonalization (optimized merge_asof approach)...")

    # Sort both dataframes by time
    df = df.sort_values('time').reset_index(drop=True)
    causal_hod_df = causal_hod_df.sort_values('time').reset_index(drop=True)

    # Initialize new columns
    df['ret_deseason'] = np.nan
    df['range_norm'] = np.nan

    # Get unique hours
    unique_hours = sorted(df['hour_cot'].unique())
    logger.info(f"Processing {len(unique_hours)} unique hours...")

    # For each hour, use merge_asof to efficiently find the most recent stats
    for hour in unique_hours:
        # Get all rows for this hour in main df
        hour_mask = df['hour_cot'] == hour
        hour_df = df[hour_mask].copy()

        # Get all stats for this hour
        hour_stats = causal_hod_df[causal_hod_df['hour_cot'] == hour].copy()

        if len(hour_stats) == 0:
            logger.warning(f"No causal stats available for hour {hour}")
            continue

        # Merge asof: for each row, find the most recent stats where stats.time <= row.time
        merged = pd.merge_asof(
            hour_df[['time', 'ret_log_1_winsor', 'range_bps']],
            hour_stats[['time', 'median_ret_log_5m', 'mad_ret_log_5m', 'p95_range_bps']],
            on='time',
            direction='backward'
        )

        # Now calculate deseasonalized values
        for i, (df_idx, row) in enumerate(zip(hour_df.index, merged.itertuples())):
            # Check if we have stats for this row
            if pd.notna(row.median_ret_log_5m) and pd.notna(row.mad_ret_log_5m):
                # Apply deseasonalization for returns
                if pd.notna(row.ret_log_1_winsor):
                    # Calculate scale with MAD and floor
                    mad_scaled = 1.4826 * row.mad_ret_log_5m

                    # Hour-aware floor: 10.0 bps for hour 8, 8.5 bps for others
                    if hour == 8:
                        floor = 10.0 / 10000  # 10.0 bps
                    else:
                        floor = 8.5 / 10000   # 8.5 bps

                    scale_h = max(float(mad_scaled), floor)

                    # Robust z-score
                    ret_deseason = (row.ret_log_1_winsor - row.median_ret_log_5m) / scale_h
                    df.loc[df_idx, 'ret_deseason'] = ret_deseason

            # Normalize range
            if pd.notna(row.range_bps) and pd.notna(row.p95_range_bps) and row.p95_range_bps > 0:
                range_norm = row.range_bps / row.p95_range_bps
                df.loc[df_idx, 'range_norm'] = range_norm

    # Log coverage statistics
    total_bars = len(df)
    deseasonalized_bars = int((~df['ret_deseason'].isna()).sum())
    range_norm_bars = int((~df['range_norm'].isna()).sum())
    coverage_pct = (deseasonalized_bars / total_bars * 100) if total_bars > 0 else 0
    range_coverage_pct = (range_norm_bars / total_bars * 100) if total_bars > 0 else 0

    logger.info(f"Causal deseasonalization complete:")
    logger.info(f"  Total bars: {total_bars}")
    logger.info(f"  Deseasonalized bars: {deseasonalized_bars} ({coverage_pct:.2f}%)")
    logger.info(f"  Range normalized bars: {range_norm_bars} ({range_coverage_pct:.2f}%)")

    return df
