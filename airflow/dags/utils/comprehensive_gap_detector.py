"""
Comprehensive Gap Detector
==========================
Detects ALL missing trading days in the historical data range,
not just gaps from the last timestamp.

This module scans the entire date range from 2020 to present and identifies:
1. Missing trading days (no data at all)
2. Incomplete trading days (< 60 bars expected)
3. Intermediate gaps (missing days between existing data)

Usage:
    from utils.comprehensive_gap_detector import find_all_missing_days

    missing_ranges = find_all_missing_days(
        start_date=datetime(2020, 1, 1),
        end_date=datetime.now(),
        postgres_config=POSTGRES_CONFIG
    )
"""

from datetime import datetime, timedelta, date
from typing import List, Tuple, Set
import logging
import psycopg2
import pytz

COT_TIMEZONE = pytz.timezone('America/Bogota')


def find_all_missing_days(
    start_date: datetime,
    end_date: datetime,
    postgres_config: dict,
    symbol: str = 'USD/COP'
) -> List[Tuple[datetime, datetime]]:
    """
    Scan entire historical range and find ALL missing trading days.

    Args:
        start_date: Start of historical range to scan
        end_date: End of range (typically today)
        postgres_config: PostgreSQL connection config
        symbol: Trading symbol to check

    Returns:
        List of (start_datetime, end_datetime) tuples for missing ranges
    """

    missing_ranges = []

    try:
        # Get all distinct trading days we have data for
        with psycopg2.connect(**postgres_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT DATE(time AT TIME ZONE 'America/Bogota') as trading_day
                    FROM usdcop_m5_ohlcv
                    WHERE symbol = %s
                      AND EXTRACT(DOW FROM time AT TIME ZONE 'America/Bogota') BETWEEN 1 AND 5
                    ORDER BY trading_day
                """, (symbol,))

                existing_days = set(row[0] for row in cursor.fetchall())

                logging.info(f"📊 Found data for {len(existing_days)} trading days in database")

        # Generate ALL expected trading days (Monday-Friday only)
        expected_days = []
        current = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date

        while current <= end:
            # Monday=0, Sunday=6 in Python's weekday()
            if current.weekday() < 5:  # Mon-Fri only
                expected_days.append(current)
            current += timedelta(days=1)

        logging.info(f"📅 Expected {len(expected_days)} total trading days from {start_date.date() if isinstance(start_date, datetime) else start_date} to {end}")

        # Find missing days
        missing_days = sorted(set(expected_days) - existing_days)

        if not missing_days:
            logging.info("✅ NO GAPS FOUND - All expected trading days have data!")
            return []

        logging.info(f"⚠️  FOUND {len(missing_days)} MISSING TRADING DAYS")

        # Group consecutive missing days into date ranges
        if missing_days:
            range_start = missing_days[0]
            range_end = missing_days[0]

            for i in range(1, len(missing_days)):
                # Check if next day is consecutive (allowing weekend gaps)
                days_diff = (missing_days[i] - range_end).days

                if days_diff <= 3:  # <= 3 days allows Fri → Mon gaps
                    range_end = missing_days[i]
                else:
                    # End current range, start new one
                    missing_ranges.append((
                        datetime.combine(range_start, datetime.min.time()).replace(tzinfo=COT_TIMEZONE),
                        datetime.combine(range_end, datetime.max.time()).replace(tzinfo=COT_TIMEZONE)
                    ))
                    range_start = missing_days[i]
                    range_end = missing_days[i]

            # Add final range
            missing_ranges.append((
                datetime.combine(range_start, datetime.min.time()).replace(tzinfo=COT_TIMEZONE),
                datetime.combine(range_end, datetime.max.time()).replace(tzinfo=COT_TIMEZONE)
            ))

        # Log summary
        logging.info("="*70)
        logging.info("🔍 COMPREHENSIVE HISTORICAL GAP ANALYSIS")
        logging.info(f"📊 Total missing trading days: {len(missing_days)}")
        logging.info(f"📦 Missing date ranges: {len(missing_ranges)}")
        logging.info("="*70)

        # Show first 15 gap ranges
        for i, (start, end) in enumerate(missing_ranges[:15], 1):
            days_in_range = len([d for d in missing_days if start.date() <= d <= end.date()])
            logging.info(f"   Gap {i:2d}: {start.date()} → {end.date()} ({days_in_range} trading days)")

        if len(missing_ranges) > 15:
            logging.info(f"   ... and {len(missing_ranges) - 15} more gap ranges")

        logging.info("="*70)

        return missing_ranges

    except Exception as e:
        logging.error(f"❌ Error in comprehensive gap detection: {e}")
        logging.exception(e)
        return []


def get_incomplete_days(
    postgres_config: dict,
    symbol: str = 'USD/COP',
    expected_bars_per_day: int = 60
) -> List[date]:
    """
    Find trading days that have SOME data but are incomplete.

    Args:
        postgres_config: PostgreSQL connection config
        symbol: Trading symbol
        expected_bars_per_day: Expected number of 5-min bars per trading day

    Returns:
        List of dates with incomplete data
    """

    incomplete_days = []

    try:
        with psycopg2.connect(**postgres_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        DATE(time AT TIME ZONE 'America/Bogota') as trading_day,
                        COUNT(*) as bar_count
                    FROM usdcop_m5_ohlcv
                    WHERE symbol = %s
                      AND EXTRACT(DOW FROM time AT TIME ZONE 'America/Bogota') BETWEEN 1 AND 5
                      AND EXTRACT(HOUR FROM time AT TIME ZONE 'America/Bogota') BETWEEN 8 AND 12
                    GROUP BY DATE(time AT TIME ZONE 'America/Bogota')
                    HAVING COUNT(*) < %s
                    ORDER BY trading_day
                """, (symbol, expected_bars_per_day))

                incomplete_days = [row[0] for row in cursor.fetchall()]

                if incomplete_days:
                    logging.warning(f"⚠️  Found {len(incomplete_days)} days with incomplete data (< {expected_bars_per_day} bars)")
                    for day in incomplete_days[:10]:
                        logging.warning(f"      {day}")

        return incomplete_days

    except Exception as e:
        logging.error(f"❌ Error checking incomplete days: {e}")
        return []
