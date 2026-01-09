#!/usr/bin/env python3
"""
L0 Router - Raw Data Quality Endpoints
========================================
Data source: PostgreSQL (market_data table)
Quality reports: MinIO (l0/quality/)

Endpoints:
- GET /pipeline/l0/statistics - L0 quality metrics from database
- GET /pipeline/l0/extended-statistics - Extended quality (coverage, gaps, OHLC violations)
"""

from fastapi import APIRouter, Query, Request
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta

from app.deps import (
    execute_sql_query,
    execute_sql_scalar,
    get_layer_config,
    read_latest_manifest,
    read_parquet_dataset,
    load_quality_gates,
    calculate_etag
)

router = APIRouter(prefix="/pipeline/l0", tags=["pipeline-l0"])


@router.get("/statistics")
def get_l0_statistics(
    date: Optional[str] = Query(None, description="YYYY-MM-DD (default: today)")
):
    """
    L0: Basic data quality statistics from PostgreSQL

    Returns:
        - Record count
        - Date range
        - Completeness percentage
        - OHLC statistics
    """
    config = get_layer_config("l0")

    # Default to today if not specified
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    # Query database
    query = """
        SELECT
            COUNT(*) as total_records,
            MIN(datetime) as earliest_datetime,
            MAX(datetime) as latest_datetime,
            COUNT(DISTINCT DATE(datetime)) as trading_days,
            AVG(close) as avg_price,
            STDDEV(close) as price_std,
            MIN(close) as price_min,
            MAX(close) as price_max
        FROM market_data
        WHERE symbol = 'USDCOP'
          AND DATE(datetime) = :date
    """

    df = execute_sql_query(query, {"date": date})

    if df.empty or df.iloc[0]["total_records"] == 0:
        return {
            "status": "NOT_FOUND",
            "date": date,
            "message": "No data found for this date"
        }

    row = df.iloc[0].to_dict()

    return {
        "status": "OK",
        "date": date,
        "layer": "l0",
        "backend": config["backend"],
        "table": config["table"],
        "statistics": {
            "total_records": int(row["total_records"]),
            "trading_days": int(row["trading_days"]),
            "date_range": {
                "earliest": str(row["earliest_datetime"]),
                "latest": str(row["latest_datetime"])
            },
            "price_stats": {
                "avg": float(row["avg_price"]) if row["avg_price"] else None,
                "std": float(row["price_std"]) if row["price_std"] else None,
                "min": float(row["price_min"]) if row["price_min"] else None,
                "max": float(row["price_max"]) if row["price_max"] else None
            }
        }
    }


@router.get("/extended-statistics")
def get_l0_extended_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    L0: Extended quality metrics (coverage, gaps, OHLC violations, duplicates, stale rate)

    GO/NO-GO Criteria:
    - Coverage >= 95%
    - OHLC violations = 0
    - Duplicates = 0
    - Stale rate <= 2%
    - Gaps (>1 missing bar) = 0
    """
    config = get_layer_config("l0")
    gates = load_quality_gates().get("l0", {})

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Coverage calculation
    coverage_query = """
        WITH expected AS (
            SELECT
                generate_series(
                    :start_date::timestamp,
                    :end_date::timestamp,
                    interval '5 minutes'
                ) as expected_time
        ),
        actual AS (
            SELECT datetime
            FROM market_data
            WHERE symbol = 'USDCOP'
              AND datetime BETWEEN :start_date AND :end_date
        )
        SELECT
            COUNT(DISTINCT expected.expected_time) as expected_points,
            COUNT(DISTINCT actual.datetime) as actual_points,
            (COUNT(DISTINCT actual.datetime)::float / COUNT(DISTINCT expected.expected_time)::float * 100) as coverage_pct
        FROM expected
        LEFT JOIN actual ON expected.expected_time = actual.datetime
    """

    coverage_df = execute_sql_query(coverage_query, {
        "start_date": start_date,
        "end_date": end_date
    })

    coverage_pct = float(coverage_df.iloc[0]["coverage_pct"]) if not coverage_df.empty else 0.0

    # OHLC violations (high < low, close not in [low, high])
    violations_query = """
        SELECT COUNT(*) as violations
        FROM market_data
        WHERE symbol = 'USDCOP'
          AND datetime BETWEEN :start_date AND :end_date
          AND (
              high < low
              OR close < low
              OR close > high
              OR open < low
              OR open > high
          )
    """

    violations_count = execute_sql_scalar(violations_query, {
        "start_date": start_date,
        "end_date": end_date
    })

    # Duplicates
    duplicates_query = """
        SELECT COUNT(*) as duplicates
        FROM (
            SELECT datetime, symbol, COUNT(*) as cnt
            FROM market_data
            WHERE symbol = 'USDCOP'
              AND datetime BETWEEN :start_date AND :end_date
            GROUP BY datetime, symbol
            HAVING COUNT(*) > 1
        ) sub
    """

    duplicates_count = execute_sql_scalar(duplicates_query, {
        "start_date": start_date,
        "end_date": end_date
    })

    # Stale rate (repeated OHLC values)
    stale_query = """
        WITH lagged AS (
            SELECT
                datetime,
                open, high, low, close,
                LAG(open) OVER (ORDER BY datetime) as prev_open,
                LAG(high) OVER (ORDER BY datetime) as prev_high,
                LAG(low) OVER (ORDER BY datetime) as prev_low,
                LAG(close) OVER (ORDER BY datetime) as prev_close
            FROM market_data
            WHERE symbol = 'USDCOP'
              AND datetime BETWEEN :start_date AND :end_date
            ORDER BY datetime
        )
        SELECT
            COUNT(*) as total_bars,
            SUM(CASE WHEN open = prev_open AND high = prev_high AND low = prev_low AND close = prev_close THEN 1 ELSE 0 END) as stale_bars,
            (SUM(CASE WHEN open = prev_open AND high = prev_high AND low = prev_low AND close = prev_close THEN 1 ELSE 0 END)::float / COUNT(*)::float * 100) as stale_pct
        FROM lagged
        WHERE prev_open IS NOT NULL
    """

    stale_df = execute_sql_query(stale_query, {
        "start_date": start_date,
        "end_date": end_date
    })

    stale_pct = float(stale_df.iloc[0]["stale_pct"]) if not stale_df.empty else 0.0

    # Gaps (missing > 1 consecutive bar)
    gaps_query = """
        SELECT COUNT(*) as gaps_gt1
        FROM (
            SELECT
                datetime,
                LEAD(datetime) OVER (ORDER BY datetime) as next_datetime,
                EXTRACT(EPOCH FROM (LEAD(datetime) OVER (ORDER BY datetime) - datetime)) / 60 as gap_minutes
            FROM market_data
            WHERE symbol = 'USDCOP'
              AND datetime BETWEEN :start_date AND :end_date
            ORDER BY datetime
        ) sub
        WHERE gap_minutes > 10  -- More than 2 bars (5min * 2 = 10min)
    """

    gaps_count = execute_sql_scalar(gaps_query, {
        "start_date": start_date,
        "end_date": end_date
    })

    # GO/NO-GO evaluation
    passed = (
        coverage_pct >= 95.0 and
        violations_count == 0 and
        duplicates_count == 0 and
        stale_pct <= 2.0 and
        gaps_count == 0
    )

    return {
        "status": "OK",
        "layer": "l0",
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days
        },
        "quality_metrics": {
            "coverage_pct": round(coverage_pct, 2),
            "ohlc_violations": int(violations_count),
            "duplicates": int(duplicates_count),
            "stale_rate_pct": round(stale_pct, 2),
            "gaps_gt1": int(gaps_count)
        },
        "quality_gates": gates,
        "pass": passed,
        "details": {
            "expected_bars": int(coverage_df.iloc[0]["expected_points"]) if not coverage_df.empty else 0,
            "actual_bars": int(coverage_df.iloc[0]["actual_points"]) if not coverage_df.empty else 0,
            "stale_bars": int(stale_df.iloc[0]["stale_bars"]) if not stale_df.empty else 0
        }
    }


@router.get("/health")
def get_l0_health():
    """Quick health check for L0 layer"""
    try:
        # Check if data exists
        query = "SELECT COUNT(*) as cnt FROM market_data LIMIT 1"
        count = execute_sql_scalar(query)

        return {
            "status": "healthy",
            "layer": "l0",
            "backend": "postgres",
            "has_data": count > 0
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "layer": "l0",
            "error": str(e)
        }
