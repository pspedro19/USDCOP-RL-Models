#!/usr/bin/env python3
"""Run the official USD/COP forward-looking PIT ingestion outside Airflow."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.usdcop_forward_macro import ForwardMacroScraper, upsert_pit_rows


def _database_connection():
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Install the database extra: pip install -e '.[database]'") from exc
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "usdcop_trading"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill official BanRep/SFC forward-looking data with PIT timestamps."
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--sources",
        default="daily_forward,forward_history,monthly_derivatives,eme,sfc",
        help=(
            "Comma-separated: daily_forward,forward_history,monthly_derivatives,"
            "eme,sfc,sfc_socrata,sfc_formato_415"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Redownload cached documents")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use archived documents only; report cache misses without network access",
    )
    parser.add_argument("--database", action="store_true", help="Also upsert extracted rows to PostgreSQL")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "usdcop_forward_macro_sources.yaml",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    selected = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    scraper = ForwardMacroScraper(
        config_path=args.config,
        force=args.force,
        offline=args.offline,
    )
    result = scraper.run(args.start, args.end, selected)
    summary = result.summary()
    if args.database and not result.extracted.empty:
        connection = _database_connection()
        try:
            summary["database_rows_upserted"] = upsert_pit_rows(connection, result.extracted)
        finally:
            connection.close()
    print(json.dumps(summary, indent=2, default=str))
    if result.errors:
        print(json.dumps({"sample_errors": result.errors[:20]}, indent=2))
    # A scheduled snapshot with no changed/new values is a healthy idempotent
    # run as long as the durable PIT ledger remains populated.
    return 0 if not result.combined.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
