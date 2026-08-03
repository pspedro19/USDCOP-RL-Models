#!/usr/bin/env python3
"""Audit Twelve Data candles captured before their objective close boundary.

Twelve Data labels a candle with its opening timestamp.  A row whose
``available_at`` predates the end of the interval plus the ingestion buffer is
therefore a partial candle and must never enter causal features or backtests.
The audit is intentionally read-only and emits machine-readable JSON.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from scripts.ops.backfill_max_history import _conn


UTC = timezone.utc


def audit(lookback_days: int) -> dict:
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    queries = {
        "asset_native_ohlcv": """
            SELECT symbol, tf, count(*)::bigint, min(time), max(time)
            FROM asset_native_ohlcv
            WHERE time >= %s
              AND source LIKE 'twelvedata%%'
              AND (
                (tf = '1h' AND available_at < time + interval '65 minutes')
                OR (tf = '4h' AND available_at < time + interval '245 minutes')
                OR (
                  tf = '1month'
                  AND available_at
                      < date_trunc('month', time) + interval '1 month 5 minutes'
                )
              )
            GROUP BY symbol, tf
            ORDER BY symbol, tf
        """,
        "usdcop_m5_ohlcv": """
            SELECT symbol, '5min' AS tf, count(*)::bigint, min(time), max(time)
            FROM usdcop_m5_ohlcv
            WHERE time >= %s
              AND source LIKE 'twelvedata%%'
              AND available_at < time + interval '10 minutes'
            GROUP BY symbol
            ORDER BY symbol
        """,
        "asset_daily_ohlcv": """
            SELECT symbol, '1day' AS tf, count(*)::bigint, min(time), max(time)
            FROM asset_daily_ohlcv
            WHERE time >= %s
              AND source LIKE 'twelvedata%%'
              AND available_at < time + interval '1 day 5 minutes'
            GROUP BY symbol
            ORDER BY symbol
        """,
    }
    conn = _conn()
    try:
        cur = conn.cursor()
        groups = []
        for table, query in queries.items():
            cur.execute(query, (cutoff,))
            for symbol, timeframe, count, first_time, last_time in cur.fetchall():
                groups.append(
                    {
                        "table": table,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "partial_rows": int(count),
                        "first_time": first_time.isoformat(),
                        "last_time": last_time.isoformat(),
                    }
                )
        return {
            "audit_time_utc": datetime.now(UTC).isoformat(),
            "lookback_days": lookback_days,
            "partial_rows": sum(g["partial_rows"] for g in groups),
            "groups": groups,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()
    result = audit(args.lookback_days)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["partial_rows"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
