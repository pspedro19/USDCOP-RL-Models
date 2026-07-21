#!/usr/bin/env python3
"""Seed macro_indicators_monthly / _quarterly from the CLEAN MASTER parquets.

Contract: CTR-MKT-CANON-001 (migration 062)

Both DB tables existed (FrequencyRoutedUpsertService schema) but were EMPTY — every
monthly series was only reachable through its daily-ffilled projection, which erases the
distinction between "the January print" and "January's value smeared across February".
The monthly/quarterly wide views need the native-frequency rows, so this loads them from
data/pipeline/04_cleaning/output/MACRO_{MONTHLY,QUARTERLY}_CLEAN.parquet (the MASTER
backups governance forbids deleting — they ARE the macro source of record).

Column mapping is resolved by DB introspection (lowercase match, then a `_q`-suffix
fallback for the quarterly GDP naming drift) instead of a hand-written dict that would
silently rot. publication_date stays NULL for these historic rows — unknown is the honest
value; the views expose a CONSERVATIVE availability bound instead (see 062).

Idempotent: upsert ON CONFLICT (fecha) DO UPDATE.
Run (host): POSTGRES_HOST=localhost POSTGRES_PASSWORD=... python scripts/ops/seed_macro_monthly.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "pipeline" / "04_cleaning" / "output"


def _db_columns(cur, table: str) -> set[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))
    return {r[0] for r in cur.fetchall()}


def seed_table(cur, table: str, parquet: Path, normalize_month: bool = False) -> tuple[int, list[str]]:
    df = pd.read_parquet(parquet)
    df.index = pd.to_datetime(df.index)
    if normalize_month:
        # The MONTHLY MASTER mixes anchors: US series at month START (FRED
        # convention, 2025-11-01) and Colombian series at month END (2025-10-31),
        # splitting every month across two rows. Both anchors mean "the value OF
        # that reference month", so truncating to month start unifies them; the
        # groupby keeps the first non-null per series (never invents data).
        df = df.groupby(df.index.to_period("M").to_timestamp()).first()
    db_cols = _db_columns(cur, table)

    mapping, skipped = {}, []
    for col in df.columns:
        lower = col.lower()
        if lower in db_cols:
            mapping[col] = lower
        elif f"{lower}_q" in db_cols:          # gdpp_real_gdp_usa_q_gdp -> ..._gdp_q drift
            mapping[col] = f"{lower}_q"
        else:
            skipped.append(col)

    cols_sql = ", ".join(mapping.values())
    # COALESCE merge: a reseed must never blank a series another anchor filled.
    updates = ", ".join(f"{c} = COALESCE(EXCLUDED.{c}, {table}.{c})" for c in mapping.values())
    n = 0
    for fecha, row in df.iterrows():
        vals = [None if pd.isna(row[src]) else float(row[src]) for src in mapping]
        cur.execute(
            f"INSERT INTO {table} (fecha, {cols_sql}, is_complete) "
            f"VALUES (%s, {', '.join(['%s'] * len(vals))}, TRUE) "
            f"ON CONFLICT (fecha) DO UPDATE SET {updates}, updated_at = now()",
            [fecha.date(), *vals])
        n += 1
    return n, skipped


def main() -> int:
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))
    cur = conn.cursor()

    # Monthly rows are normalized to month-start; drop any pre-normalization
    # month-end strays from earlier seeder runs (this table is seed-fed only).
    cur.execute("DELETE FROM macro_indicators_monthly "
                "WHERE fecha <> date_trunc('month', fecha)::date")
    if cur.rowcount:
        print(f"macro_indicators_monthly: {cur.rowcount} filas month-end eliminadas (pre-normalizacion)")

    for table, parquet, norm in (
        ("macro_indicators_monthly", CLEAN / "MACRO_MONTHLY_CLEAN.parquet", True),
        ("macro_indicators_quarterly", CLEAN / "MACRO_QUARTERLY_CLEAN.parquet", False),
    ):
        n, skipped = seed_table(cur, table, parquet, normalize_month=norm)
        print(f"{table}: {n} filas upsert" + (f" | sin columna DB: {skipped}" if skipped else ""))

    conn.commit()
    for t in ("macro_indicators_monthly", "macro_indicators_quarterly"):
        cur.execute(f"SELECT COUNT(*), MIN(fecha), MAX(fecha) FROM {t}")
        print("  ", t, cur.fetchone())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
