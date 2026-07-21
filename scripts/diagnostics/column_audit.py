#!/usr/bin/env python3
"""Column-by-column audit of the canonical market tables (CTR-MKT-CANON-001).

For every column of every table: null rate, distinct count, and type-appropriate
descriptives (min/max/avg/stddev for numerics, min/max for timestamps, mode for text).
One aggregate pass per table — no sampling, the WHOLE table is measured.

Run: POSTGRES_HOST=localhost POSTGRES_PASSWORD=... PYTHONIOENCODING=utf-8 \
     python scripts/diagnostics/column_audit.py [table ...]
"""
from __future__ import annotations

import os
import sys

TABLES = [
    "usdcop_m5_ohlcv", "asset_native_ohlcv", "asset_daily_ohlcv",
    "dim_asset", "market_session_calendar", "market_ingestion_manifest",
    "macro_indicators_daily", "macro_indicators_monthly", "macro_indicators_quarterly",
]

NUM = {"numeric", "bigint", "integer", "double precision", "real", "smallint"}
TS = {"timestamp with time zone", "timestamp without time zone", "date"}


def _conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.4g}"
    s = str(v)
    return s[:19] if len(s) > 19 else s


def audit(cur, table: str) -> None:
    cur.execute("""SELECT column_name, data_type FROM information_schema.columns
                   WHERE table_name=%s ORDER BY ordinal_position""", (table,))
    cols = cur.fetchall()
    if not cols:
        print(f"\n== {table}: NO EXISTE ==")
        return
    cur.execute(f"SELECT count(*) FROM {table}")
    total = cur.fetchone()[0]
    print(f"\n== {table} — {total:,} filas, {len(cols)} columnas ==")
    print(f"{'columna':34s} {'tipo':12s} {'nulos':>7s} {'distintos':>9s} "
          f"{'min':>19s} {'max':>19s} {'media':>12s} {'desv':>12s}")
    for name, dtype in cols:
        short = dtype.replace("timestamp with time zone", "timestamptz") \
                     .replace("timestamp without time zone", "timestamp") \
                     .replace("double precision", "float8")
        if total == 0:
            print(f"{name:34s} {short:12s} {'-':>7s}")
            continue
        if dtype in NUM:
            cur.execute(f"""SELECT count(*)-count({name}), count(DISTINCT {name}),
                            min({name}), max({name}), avg({name}), stddev({name})
                            FROM {table}""")
            nulls, dist, mn, mx, avg, sd = cur.fetchone()
            print(f"{name:34s} {short:12s} {nulls/total:7.1%} {dist:9,d} "
                  f"{fmt(float(mn) if mn is not None else None):>19s} "
                  f"{fmt(float(mx) if mx is not None else None):>19s} "
                  f"{fmt(float(avg) if avg is not None else None):>12s} "
                  f"{fmt(float(sd) if sd is not None else None):>12s}")
        elif dtype in TS:
            cur.execute(f"""SELECT count(*)-count({name}), count(DISTINCT {name}),
                            min({name}), max({name}) FROM {table}""")
            nulls, dist, mn, mx = cur.fetchone()
            print(f"{name:34s} {short:12s} {nulls/total:7.1%} {dist:9,d} "
                  f"{fmt(mn):>19s} {fmt(mx):>19s} {'-':>12s} {'-':>12s}")
        elif dtype == "boolean":
            cur.execute(f"""SELECT count(*)-count({name}),
                            count(*) FILTER (WHERE {name}) FROM {table}""")
            nulls, true_n = cur.fetchone()
            print(f"{name:34s} {short:12s} {nulls/total:7.1%} {'2':>9s} "
                  f"{'true=' + format(true_n, ','):>19s} "
                  f"{'false=' + format(total - nulls - true_n, ','):>19s} "
                  f"{'-':>12s} {'-':>12s}")
        else:  # text
            cur.execute(f"""SELECT count(*)-count({name}), count(DISTINCT {name}),
                            mode() WITHIN GROUP (ORDER BY {name}) FROM {table}""")
            nulls, dist, top = cur.fetchone()
            print(f"{name:34s} {short:12s} {nulls/total:7.1%} {dist:9,d} "
                  f"{('moda=' + str(top))[:40]:>40s}")


def main() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    for t in (sys.argv[1:] or TABLES):
        audit(cur, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
