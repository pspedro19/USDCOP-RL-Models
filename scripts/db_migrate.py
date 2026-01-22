#!/usr/bin/env python3
"""
Database Migration System
=========================

Tracks and runs database migrations idempotently.
Prevents schema drift by ensuring all init scripts run exactly once.

Features:
- Tracks executed migrations in `_migrations` table
- Runs pending migrations in order (by filename)
- Idempotent: safe to run multiple times
- Validates required tables exist after migration

Usage:
    python scripts/db_migrate.py              # Run all pending migrations
    python scripts/db_migrate.py --status     # Show migration status
    python scripts/db_migrate.py --validate   # Validate required tables exist
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Migration scripts directory
MIGRATIONS_DIR = PROJECT_ROOT / "init-scripts"

# Required tables (table_name -> description)
REQUIRED_TABLES = {
    # Core OHLCV data
    "public.usdcop_m5_ohlcv": "5-minute OHLCV price data",

    # Macro indicators
    "public.macro_indicators_daily": "Daily macroeconomic indicators",

    # Trading/Paper trading
    "public.trades_history": "Historical trades for backtest/replay",
    "public.trading_state": "Current trading state per model",
    "public.equity_snapshots": "Equity curve snapshots",

    # Model configuration
    "config.models": "Model configurations (SSOT)",
    "config.feature_definitions": "Feature definitions",

    # Events/Signals
    "events.signals_stream": "Trading signals stream",

    # Metrics
    "metrics.model_performance": "Model performance metrics",
}


async def get_connection():
    """Get database connection."""
    import asyncpg

    return await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "admin123"),
        database=os.getenv("POSTGRES_DB", "usdcop_trading"),
    )


async def ensure_migrations_table(conn) -> None:
    """Create migrations tracking table if not exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL UNIQUE,
            checksum VARCHAR(64) NOT NULL,
            executed_at TIMESTAMPTZ DEFAULT NOW(),
            execution_time_ms INTEGER,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_migrations_filename
        ON _migrations(filename)
    """)


def get_file_checksum(filepath: Path) -> str:
    """Calculate MD5 checksum of file."""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


async def get_executed_migrations(conn) -> Dict[str, str]:
    """Get dict of executed migrations {filename: checksum}."""
    rows = await conn.fetch("""
        SELECT filename, checksum FROM _migrations WHERE success = TRUE
    """)
    return {row["filename"]: row["checksum"] for row in rows}


def get_migration_files() -> List[Path]:
    """Get all SQL migration files in order."""
    if not MIGRATIONS_DIR.exists():
        logger.warning(f"Migrations directory not found: {MIGRATIONS_DIR}")
        return []

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return files


async def run_migration(conn, filepath: Path, checksum: str) -> Tuple[bool, Optional[str]]:
    """Run a single migration file."""
    sql = filepath.read_text(encoding="utf-8")

    start_time = datetime.now()
    try:
        await conn.execute(sql)
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Record successful migration
        await conn.execute("""
            INSERT INTO _migrations (filename, checksum, execution_time_ms, success)
            VALUES ($1, $2, $3, TRUE)
            ON CONFLICT (filename) DO UPDATE SET
                checksum = $2,
                executed_at = NOW(),
                execution_time_ms = $3,
                success = TRUE,
                error_message = NULL
        """, filepath.name, checksum, execution_time)

        return True, None

    except Exception as e:
        error_msg = str(e)
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Record failed migration
        try:
            await conn.execute("""
                INSERT INTO _migrations (filename, checksum, execution_time_ms, success, error_message)
                VALUES ($1, $2, $3, FALSE, $4)
                ON CONFLICT (filename) DO UPDATE SET
                    checksum = $2,
                    executed_at = NOW(),
                    execution_time_ms = $3,
                    success = FALSE,
                    error_message = $4
            """, filepath.name, checksum, execution_time, error_msg)
        except:
            pass

        return False, error_msg


async def table_exists(conn, full_table_name: str) -> bool:
    """Check if table exists (handles schema.table format)."""
    if "." in full_table_name:
        schema, table = full_table_name.split(".", 1)
    else:
        schema, table = "public", full_table_name

    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = $1 AND table_name = $2
        )
    """, schema, table)
    return result


async def run_migrations() -> bool:
    """Run all pending migrations."""
    logger.info("=" * 60)
    logger.info("USDCOP Database Migration System")
    logger.info("=" * 60)

    try:
        conn = await get_connection()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        return False

    try:
        # Ensure migrations table exists
        await ensure_migrations_table(conn)

        # Get already executed migrations
        executed = await get_executed_migrations(conn)
        logger.info(f"Previously executed migrations: {len(executed)}")

        # Get all migration files
        migration_files = get_migration_files()
        logger.info(f"Total migration files: {len(migration_files)}")

        # Find pending migrations
        pending = []
        for filepath in migration_files:
            checksum = get_file_checksum(filepath)

            if filepath.name not in executed:
                pending.append((filepath, checksum, "new"))
            elif executed[filepath.name] != checksum:
                pending.append((filepath, checksum, "modified"))

        if not pending:
            logger.info("No pending migrations. Database is up to date.")
            return True

        logger.info(f"Pending migrations: {len(pending)}")

        # Run pending migrations
        success_count = 0
        error_count = 0

        for filepath, checksum, reason in pending:
            logger.info(f"Running [{reason}]: {filepath.name}")

            success, error = await run_migration(conn, filepath, checksum)

            if success:
                logger.info(f"  ✓ Success")
                success_count += 1
            else:
                logger.error(f"  ✗ Error: {error}")
                error_count += 1

        logger.info("-" * 60)
        logger.info(f"Migrations complete: {success_count} succeeded, {error_count} failed")

        return error_count == 0

    finally:
        await conn.close()


async def show_status() -> None:
    """Show migration status."""
    logger.info("=" * 60)
    logger.info("Migration Status")
    logger.info("=" * 60)

    try:
        conn = await get_connection()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        return

    try:
        await ensure_migrations_table(conn)

        # Get executed migrations
        rows = await conn.fetch("""
            SELECT filename, checksum, executed_at, execution_time_ms, success, error_message
            FROM _migrations
            ORDER BY executed_at
        """)

        if not rows:
            logger.info("No migrations have been executed yet.")
        else:
            for row in rows:
                status = "✓" if row["success"] else "✗"
                logger.info(f"  {status} {row['filename']} ({row['execution_time_ms']}ms) - {row['executed_at']}")
                if row["error_message"]:
                    logger.info(f"      Error: {row['error_message'][:100]}")

        # Get pending
        executed = await get_executed_migrations(conn)
        migration_files = get_migration_files()

        pending = [f for f in migration_files if f.name not in executed]
        if pending:
            logger.info(f"\nPending migrations: {len(pending)}")
            for f in pending:
                logger.info(f"  - {f.name}")

    finally:
        await conn.close()


async def validate_tables() -> bool:
    """Validate all required tables exist."""
    logger.info("=" * 60)
    logger.info("Validating Required Tables")
    logger.info("=" * 60)

    try:
        conn = await get_connection()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        return False

    try:
        missing = []
        present = []

        for table_name, description in REQUIRED_TABLES.items():
            exists = await table_exists(conn, table_name)
            if exists:
                present.append(table_name)
                logger.info(f"  ✓ {table_name}")
            else:
                missing.append(table_name)
                logger.warning(f"  ✗ {table_name} - MISSING ({description})")

        logger.info("-" * 60)
        logger.info(f"Present: {len(present)}, Missing: {len(missing)}")

        if missing:
            logger.error("Run 'python scripts/db_migrate.py' to create missing tables")
            return False

        logger.info("All required tables exist!")
        return True

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Database Migration System")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--validate", action="store_true", help="Validate required tables")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.validate:
        success = asyncio.run(validate_tables())
        sys.exit(0 if success else 1)
    else:
        success = asyncio.run(run_migrations())
        # Also validate after running migrations
        if success:
            success = asyncio.run(validate_tables())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
