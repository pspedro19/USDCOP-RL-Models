#!/usr/bin/env python3
"""
Feature-data backup & restore (news / analysis / H5 / multi-asset daily)
========================================================================
The daily seed backup (`airflow/dags/l0_seed_backup.py`) only dumps OHLCV + macro
daily. This module adds the *derived / feature* tables so a clean-slate
`docker compose down -v` boot comes back with the REAL news, analysis, H5
production, and multi-asset daily data — not an empty schema that must be
re-scraped/re-derived from zero.

Design (mirrors `data/backups/seeds/*.parquet` + init-script restore):
  * BACKUP  — dump each present table to `data/backups/features/<table>.parquet`
              (atomic .tmp->rename) + `feature_backup_manifest.json`
              (rows, sha256, max-timestamp per table).
  * RESTORE — for each parquet whose table EXISTS and is EMPTY, bulk-insert the
              rows (column-intersection with the live table, so schema drift is
              tolerated). Non-empty tables are SKIPPED (never clobber live data) —
              the forward pipelines (scrapers/backfill DAGs) fill the gap to today.

Per-asset framing: `asset_daily_ohlcv` carries an `asset_id`/`symbol` column and
`news_*` carry `source_id`, so "backup de noticias por activo" is inherent in the
row data — no partitioning needed to preserve it.

Idempotent both ways. Safe to run repeatedly. Never raises on a missing table.

Contract: CTR-L0-FEATURE-BACKUP-001   ·   Version: 1.0.0   ·   Date: 2026-07-05

Usage:
    python -m scripts.ops.backup.feature_data_backup --mode backup
    python -m scripts.ops.backup.feature_data_backup --mode restore
    python -m scripts.ops.backup.feature_data_backup --mode backup --dir data/backups/features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [feature-backup] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DIR = "data/backups/features"

# Tables to snapshot, with the column used to report "latest" in the manifest
# (best-effort — skipped if the column is absent). Grouped by domain.
TABLES: list[tuple[str, str | None]] = [
    # News Engine
    ("news_articles", "published_at"),
    ("news_feature_snapshots", "snapshot_date"),
    # Analysis module
    ("weekly_analysis", "created_at"),
    ("daily_analysis", "analysis_date"),
    # H5 production (the promoted strategy's live tables)
    ("forecast_h5_predictions", "created_at"),
    ("forecast_h5_signals", "signal_date"),
    ("forecast_h5_executions", "created_at"),
    ("forecast_h5_subtrades", "created_at"),
    ("forecast_h5_paper_trading", "created_at"),
    # Multi-asset daily OHLCV (Gold/BTC/… — per-asset via asset_id column)
    ("asset_daily_ohlcv", "time"),
    # Crypto-native derivatives (BTC perp funding/OI/long-short — migration 052)
    ("crypto_derivatives_daily", "date"),
    # Macro monthly/quarterly (daily already covered by l0_seed_backup)
    ("macro_indicators_monthly", "fecha"),
    ("macro_indicators_quarterly", "fecha"),
    # Compliance: append-only audit trail (Vote-2, kills, fan-outs, go-live, plan changes).
    # Found lost on cold boot 2026-07-07 — an audit log that vanishes on rebuild defeats
    # its purpose (CTR-RBAC-001 §11). Restore is empty-table-only like everything else.
    ("audit_log", "created_at"),
]


def _connect():
    """psycopg2 connection from standard env (works in airflow + data-seeder + host)."""
    import psycopg2

    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", os.environ.get("PGHOST", "localhost")),
        port=int(os.environ.get("POSTGRES_PORT", os.environ.get("PGPORT", "5432"))),
        dbname=os.environ.get("POSTGRES_DB", os.environ.get("PGDATABASE", "usdcop_trading")),
        user=os.environ.get("POSTGRES_USER", os.environ.get("PGUSER", "admin")),
        password=os.environ.get("POSTGRES_PASSWORD", os.environ.get("PGPASSWORD", "")),
    )


def _table_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
    return cur.fetchone()[0] is not None


def _table_columns(cur, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def _row_count(cur, table: str) -> int:
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    return cur.fetchone()[0]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(out_dir: Path) -> dict:
    """Dump every present table to parquet + write a manifest. Returns the manifest."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    manifest: dict = {"created_at": datetime.now(timezone.utc).isoformat(), "tables": {}}
    try:
        cur = conn.cursor()
        for table, ts_col in TABLES:
            if not _table_exists(cur, table):
                log.info("skip %-28s (table absent)", table)
                manifest["tables"][table] = {"status": "absent"}
                continue

            df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            final = out_dir / f"{table}.parquet"
            tmp = out_dir / f"{table}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, final)  # atomic

            latest = None
            if ts_col and ts_col in df.columns and len(df):
                try:
                    latest = str(pd.to_datetime(df[ts_col]).max())
                except Exception:  # noqa: BLE001
                    latest = None
            entry = {
                "status": "ok",
                "rows": int(len(df)),
                "sha256": _sha256(final),
                "latest": latest,
                "file": final.name,
            }
            manifest["tables"][table] = entry
            log.info("backed up %-28s rows=%-8d latest=%s", table, len(df), latest)

        man_path = out_dir / "feature_backup_manifest.json"
        tmp_man = out_dir / "feature_backup_manifest.json.tmp"
        tmp_man.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp_man, man_path)
        log.info("manifest -> %s", man_path)
        return manifest
    finally:
        conn.close()


def restore(in_dir: Path) -> dict:
    """Bulk-insert parquet rows into EMPTY matching tables only. Returns a report."""
    import pandas as pd
    from psycopg2.extras import execute_values

    report: dict = {"restored": {}, "skipped": {}}
    if not in_dir.exists():
        log.warning("restore dir %s absent — nothing to restore", in_dir)
        return report

    conn = _connect()
    try:
        cur = conn.cursor()
        for table, _ts in TABLES:
            pq = in_dir / f"{table}.parquet"
            if not pq.exists():
                report["skipped"][table] = "no backup file"
                continue
            if not _table_exists(cur, table):
                report["skipped"][table] = "table absent"
                log.info("skip %-28s (table absent)", table)
                continue
            existing = _row_count(cur, table)
            if existing > 0:
                report["skipped"][table] = f"already has {existing} rows"
                log.info("skip %-28s (already populated: %d rows)", table, existing)
                continue

            df = pd.read_parquet(pq)
            if df.empty:
                report["skipped"][table] = "backup empty"
                continue

            # Column-intersection => tolerate schema drift between backup and live table.
            live_cols = _table_columns(cur, table)
            cols = [c for c in df.columns if c in live_cols]
            if not cols:
                report["skipped"][table] = "no overlapping columns"
                log.warning("skip %-28s (no overlapping columns)", table)
                continue

            # Per-column postgres data_type — decides how list/array values bind:
            # 'ARRAY' (e.g. TEXT[]) needs a raw Python list; 'jsonb'/'json' needs Json().
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s", (table,))
            coltypes = {r[0]: r[1] for r in cur.fetchall()}

            import numpy as np
            from psycopg2.extras import Json

            def _coerce(v, dtype):
                # pd.isna on an array/dict returns an array (ambiguous truth value),
                # so guard NA to scalars. list/dict/ndarray bind per the column type.
                if v is None:
                    return None
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                if isinstance(v, (list, dict)):
                    if dtype == "ARRAY" and isinstance(v, list):
                        return v  # psycopg2 adapts a Python list -> postgres array
                    return Json(v)  # jsonb / json
                try:
                    na = pd.isna(v)
                except (ValueError, TypeError):
                    return v
                if isinstance(na, bool):
                    return None if na else v
                return v  # non-scalar — pass through

            rows = [tuple(_coerce(v, coltypes.get(c)) for c, v in zip(cols, rec))
                    for rec in df[cols].itertuples(index=False, name=None)]
            col_sql = ", ".join(f'"{c}"' for c in cols)
            # Per-table isolation: one table's adaptation error must not abort the
            # rest (the derived tables are independent; best-effort restore).
            try:
                execute_values(
                    cur,
                    f'INSERT INTO "{table}" ({col_sql}) VALUES %s ON CONFLICT DO NOTHING',
                    rows,
                    page_size=1000,
                )
                conn.commit()
                report["restored"][table] = len(rows)
                log.info("restored %-28s rows=%d", table, len(rows))
            except Exception as e:  # noqa: BLE001
                conn.rollback()
                report["skipped"][table] = f"insert failed: {e}"
                log.warning("skip %-28s (insert failed: %s)", table, e)
        return report
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature-data backup/restore")
    ap.add_argument("--mode", required=True, choices=["backup", "restore"])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="backup directory (repo-relative or absolute)")
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_absolute():
        # repo root = three parents up from scripts/ops/backup/
        d = Path(__file__).resolve().parents[3] / a.dir

    if a.mode == "backup":
        m = backup(d)
        ok = sum(1 for t in m["tables"].values() if t.get("status") == "ok")
        log.info("BACKUP done: %d tables dumped -> %s", ok, d)
    else:
        r = restore(d)
        log.info("RESTORE done: restored=%d skipped=%d",
                 len(r["restored"]), len(r["skipped"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
