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
    python scripts/ops/db_migrate.py --plan legacy-init
    python scripts/ops/db_migrate.py --plan fabric-v1 --plan-digest
    python scripts/ops/db_migrate.py --plan fabric-v1 \
        --reviewed-digest sha256:<reviewed-plan-digest>
    python scripts/ops/db_migrate.py --plan fabric-v1 --status
    python scripts/ops/db_migrate.py --plan fabric-v1 --validate
"""

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root (scripts/ops/db_migrate.py -> repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Explicit migration plans.  New migrations are never discovered by a broad
# production glob: adding a file does not make it deployable.
MIGRATION_PLANS = {
    "legacy-init": tuple(sorted((PROJECT_ROOT / "init-scripts").glob("*.sql"))),
    "commerce-v1": tuple(
        PROJECT_ROOT / "database" / "migrations" / name
        for name in (
            "059_checkout_order_ledger.sql",
            "082_checkout_order_retry_transition.sql",
        )
    ),
    "commerce-surface-v1": (
        PROJECT_ROOT / "database" / "migrations" / "057_catalog_watchlist_cart.sql",
    ),
    "identity-admin-v1": tuple(
        PROJECT_ROOT / "database" / "migrations" / name
        for name in (
            "056_admin_console_is_test.sql",
            "056_rbac_dynamic_roles.sql",
        )
    ),
    "h5-identity-v1": tuple(
        PROJECT_ROOT / "database" / "migrations" / name
        for name in (
            "064_h5_strategy_id.sql",
            "083_h5_strategy_performance_view.sql",
        )
    ),
    # Fresh-clone platform schema.  This deliberately uses the consolidated H5
    # migration (050) instead of replaying its superseded 043/044 path, and
    # keeps optional extensions such as pgvector (047) out of the baseline.
    "platform-bootstrap-v1": tuple(
        PROJECT_ROOT / "database" / "migrations" / name
        for name in (
            "045_newsengine_initial.sql",
            "046_weekly_analysis_tables.sql",
            "050_consolidated_h5_ddl.sql",
            "051_asset_daily_ohlcv.sql",
            "053_sb_user_approval.sql",
            "054_h5_subtrades_unique.sql",
            "055_rbac_monetization.sql",
        )
    ),
    "fabric-v1": tuple(
        PROJECT_ROOT / "database" / "migrations" / name
        for name in (
            "070_fabric_control_plane.sql",
            "071_forecast_schema_roles.sql",
            "072_reference_identity.sql",
            "073_market_quality.sql",
            "074_exec_event_sourcing.sql",
            "075_fact_position_pnl.sql",
            "076_lineage_graph.sql",
            "077_portfolio_control.sql",
            "078_exec_reconciliation.sql",
            "079_fabric_integrity_remediation.sql",
            "080_market_physical_profile.sql",
            "081_synthetic_demo_isolation.sql",
        )
    ),
}
REVIEW_GATED_PLANS = frozenset(
    {
        "commerce-v1",
        "commerce-surface-v1",
        "h5-identity-v1",
        "identity-admin-v1",
        "platform-bootstrap-v1",
        "fabric-v1",
    }
)
PLAN_PREREQUISITE_TABLES = {
    "commerce-surface-v1": ("public.sb_users",),
    "h5-identity-v1": (
        "public.forecast_h5_signals",
        "public.forecast_h5_executions",
        "public.forecast_h5_paper_trading",
    ),
    "identity-admin-v1": ("public.sb_users",),
    "platform-bootstrap-v1": (
        "public.sb_users",
        "public.usdcop_m5_ohlcv",
        "public.macro_indicators_daily",
    ),
}
# These values are changed only in the same reviewed commit that changes a
# gated migration plan.  A digest supplied by the operator is a second factor,
# not a way for modified on-disk SQL to authorize itself.
PINNED_PLAN_DIGESTS = {
    "commerce-v1": (
        "sha256:3fdb2d845fbe90e26b294bd09fb1d02b889b17ee71d4836ce47af9b60d49a261"
    ),
    "platform-bootstrap-v1": (
        "sha256:9d6e2d40fa974aca3474c70336b5e0912c04f5b75c7474e62ea172a01388c06d"
    ),
    "fabric-v1": (
        "sha256:023ebffaa5afcb9af83942f3bc23ea74282eec120047d77407a23eccfe6ee1eb"
    ),
}

LEGACY_REQUIRED_TABLES = {
    # Core OHLCV data
    "public.usdcop_m5_ohlcv": "5-minute OHLCV price data",

    # Macro indicators
    "public.macro_indicators_daily": "Daily macroeconomic indicators",
    "public.macro_indicators_pit": "Forward-looking macro publication vintages",

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

PLATFORM_BOOTSTRAP_REQUIRED_TABLES = {
    "public.forecast_h5_signals": "H5 ensemble signals",
    "public.forecast_h5_subtrades": "H5 subtrade ledger",
    "public.news_articles": "NewsEngine articles",
    "public.weekly_analysis": "Weekly analysis output",
    "public.asset_daily_ohlcv": "Multi-asset daily bars",
    "public.audit_log": "Append-only authorization audit ledger",
    "public.user_exchange_keys": "Per-user encrypted exchange-key records",
}

FABRIC_REQUIRED_TABLES = {
    "control.strategy_declaration": "Constitutional strategy declarations",
    "control.strategy_declaration_event": "Strategy transition ledger",
    "control.strategy_sleeve": "Materialized strategy-to-sleeve mapping",
    "control.incident": "Operational incident ledger",
    "control.artifact_identity": "Canonical artifact identity spine",
    "control.metric_event": "Governed metric event ledger",
    "control.legacy_metric_observation": "Quarantined legacy metrics",
    "forecast.forecast_output": "Diagnostic forecast output",
    "forecast.forecast_score": "Immutable forecast score events",
    "forecast.model_horizon_result": "Model/horizon diagnostic results",
    "forecast.calibration_result": "Forecast calibration results",
    "reference.calendar": "Canonical calendars",
    "reference.asset": "Canonical assets",
    "reference.instrument": "Canonical instruments",
    "reference.provider": "Canonical providers",
    "reference.provider_symbol": "Provider symbol aliases",
    "reference.instrument_alias": "Instrument aliases",
    "reference.bar_interval": "Canonical bar intervals",
    "market.raw_bar": "Append-only raw market bars",
    "market.canonical_bar": "Canonical market bars",
    "market.canonical_bar_source": "Canonical-bar lineage",
    "quality.quarantine_event": "Market-data quarantine events",
    "quality.correction_event": "Market-data correction events",
    "quality.feature_status": "Feature quality status",
    "exec.order_header": "Canonical order headers",
    "exec.order_status_event": "Append-only order statuses",
    "exec.order_dispatch": "Fenced broker dispatch claims",
    "exec.order_dispatch_event": "Immutable dispatch attempts",
    "exec.fill_event": "Immutable broker fills",
    "exec.fill_correction_event": "Fill correction chain",
    "exec.reconciliation_event": "Broker reconciliation facts",
    "fact.position": "Canonical position facts",
    "fact.pnl": "Canonical PnL facts",
    "lineage.node": "Lineage nodes",
    "lineage.edge": "Lineage edges",
    "lineage.strategy_node": "Strategy lineage projection",
    "lineage.revision_event": "Typed revision events",
    "portfolio.snapshot": "Point-in-time book snapshots",
    "portfolio.snapshot_signal": "Snapshot signals",
    "portfolio.allocation": "Materialized allocations",
    "portfolio.target": "Aggregate portfolio targets",
    "portfolio.target_exposure": "Target exposure children",
    "portfolio.pretrade_decision": "Pre-trade decision ledger",
    "portfolio.kill_switch_event": "Kill-switch event ledger",
    "portfolio.kill_switch_action": "Fenced kill-switch effects",
    "portfolio.kill_switch_action_event": "Kill-switch attempt events",
    "demo.synthetic_model": "Synthetic models isolated from real performance",
}

REQUIRED_TABLES_BY_PLAN = {
    "legacy-init": LEGACY_REQUIRED_TABLES,
    "platform-bootstrap-v1": PLATFORM_BOOTSTRAP_REQUIRED_TABLES,
    "commerce-v1": {
        "public.checkout_orders": "Immutable sealed checkout quotes",
        "public.billing_events": "Provider-event idempotency ledger",
    },
    "commerce-surface-v1": {
        "public.user_watchlist": "Per-user catalog watchlist",
        "public.user_cart": "Per-user add-on cart",
    },
    "identity-admin-v1": {
        "public.rbac_role_permissions": "Dynamic role-permission assignments",
        "public.rbac_user_overrides": "Per-user RBAC overrides",
    },
    "h5-identity-v1": {
        "public.v_h5_performance_summary": "Strategy-safe H5 performance view",
    },
    "fabric-v1": FABRIC_REQUIRED_TABLES,
}
REQUIRED_COLUMNS_BY_PLAN = {
    "commerce-surface-v1": {
        "public.user_watchlist": {
            "user_id": "User ownership",
            "asset_id": "Catalog asset identity",
            "created_at": "Insertion timestamp",
        },
        "public.user_cart": {
            "user_id": "User ownership",
            "asset_id": "Catalog asset identity",
            "created_at": "Insertion timestamp",
        },
    },
    "h5-identity-v1": {
        "public.forecast_h5_signals": {"strategy_id": "H5 signal identity"},
        "public.forecast_h5_executions": {
            "strategy_id": "H5 execution identity"
        },
        "public.forecast_h5_paper_trading": {
            "strategy_id": "H5 paper-trading identity"
        },
        "public.v_h5_performance_summary": {
            "strategy_id": "Strategy-safe H5 performance projection"
        },
    },
    "identity-admin-v1": {
        "public.sb_users": {
            "is_test": "Admin-console test-user classification",
        },
    },
}
# Compatibility alias for old importers. CLI callers must select a plan.
REQUIRED_TABLES = LEGACY_REQUIRED_TABLES


class MigrationDriftError(RuntimeError):
    """An applied migration no longer matches its immutable reviewed bytes."""


def migration_lock_id(name: str) -> int:
    """Return a stable signed bigint for PostgreSQL advisory locks."""
    digest = hashlib.sha256(f"usdcop-migration:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


async def get_connection():
    """Get database connection."""
    import asyncpg

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return await asyncpg.connect(dsn=database_url)

    password = os.getenv("POSTGRES_PASSWORD")
    if password is None:
        raise RuntimeError(
            "database connection is not configured: set DATABASE_URL or "
            "POSTGRES_PASSWORD with POSTGRES_HOST/PORT/USER/DB"
        )
    return await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=password,
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


def get_migration_files(plan: str = "legacy-init") -> List[Path]:
    """Return the audited allowlist for a named plan."""
    try:
        files = list(MIGRATION_PLANS[plan])
    except KeyError as exc:
        raise ValueError(f"unknown migration plan {plan!r}") from exc
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"migration plan {plan!r} references missing files: {missing}"
        )
    return files


def get_plan_digest(plan: str) -> str:
    """Content-address the ordered plan reviewed by an operator."""
    digest = hashlib.sha256()
    for path in get_migration_files(plan):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def created_tables_for_plan(plan: str) -> set[str]:
    """Return statically declared tables, resolving bare names to ``public``.

    This is a source-contract guard, not a SQL interpreter: it does not resolve
    control flow or dynamic SQL. Runtime ``--validate`` remains authoritative
    for objects whose creation depends on execution.
    """
    ddl = "\n".join(
        path.read_text(encoding="utf-8") for path in get_migration_files(plan)
    )
    ddl = re.sub(r"/\*.*?\*/", "", ddl, flags=re.DOTALL)
    ddl = re.sub(r"--[^\n]*", "", ddl)
    created: set[str] = set()
    pattern = re.compile(
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?P<name>(?:\"?[a-z_]\w*\"?\.)?\"?[a-z_]\w*\"?)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(ddl):
        name = match.group("name").replace('"', "").lower()
        created.add(name if "." in name else f"public.{name}")
    return created


def classify_migrations(
    plan: str, executed: Dict[str, str]
) -> List[Tuple[Path, str, str]]:
    """Return new migrations and fail closed on immutable checksum drift."""
    pending: List[Tuple[Path, str, str]] = []
    drift: List[Tuple[str, str, str]] = []
    for filepath in get_migration_files(plan):
        checksum = get_file_checksum(filepath)
        recorded = executed.get(filepath.name)
        if recorded is None:
            pending.append((filepath, checksum, "new"))
        elif recorded != checksum:
            drift.append((filepath.name, recorded, checksum))
    if drift:
        details = ", ".join(
            f"{name} recorded={recorded} current={current}"
            for name, recorded, current in drift
        )
        raise MigrationDriftError(
            f"applied migration checksum drift in plan {plan!r}: {details}"
        )
    return pending


def plan_is_authorized(plan: str, reviewed_digest: str | None) -> bool:
    if plan not in REVIEW_GATED_PLANS:
        return True
    pinned = PINNED_PLAN_DIGESTS.get(plan)
    current = get_plan_digest(plan)
    if pinned is None:
        logger.error(
            "Plan %s is review-gated but has no pinned reviewed digest",
            plan,
        )
        return False
    if current != pinned:
        logger.error(
            "Plan %s differs from its pinned reviewed digest: pinned=%s current=%s",
            plan,
            pinned,
            current,
        )
        return False
    if reviewed_digest != pinned:
        logger.error(
            "Plan %s is review-gated. Expected --reviewed-digest %s; got %r",
            plan,
            pinned,
            reviewed_digest,
        )
        return False
    return True


async def run_migration(conn, filepath: Path, checksum: str) -> Tuple[bool, Optional[str]]:
    """Run a single migration file."""
    sql = filepath.read_text(encoding="utf-8")

    start_time = datetime.now()
    try:
        async with conn.transaction():
            should_execute = await claim_migration_attempt(
                conn, filepath.name, checksum
            )
            if not should_execute:
                return True, None
            await conn.execute(sql)
            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            await conn.execute("""
                INSERT INTO _migrations (
                    filename, checksum, execution_time_ms, success
                )
                VALUES ($1, $2, $3, TRUE)
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
                    checksum = EXCLUDED.checksum,
                    executed_at = NOW(),
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    success = FALSE,
                    error_message = EXCLUDED.error_message
                WHERE _migrations.success = FALSE
            """, filepath.name, checksum, execution_time, error_msg)
        except Exception:
            pass

        return False, error_msg


async def claim_migration_attempt(
    conn, filename: str, checksum: str
) -> bool:
    """Fence one filename and recover a prior failed attempt.

    The lock is transaction-scoped, so two callers cannot both execute DDL for
    the same file.  A successful row is immutable.  A failed row is diagnostic
    history, not a permanent tombstone: it is removed inside the retry
    transaction and recreated as either SUCCESS or the latest FAILED record.
    """
    await conn.fetchval(
        "SELECT pg_advisory_xact_lock($1::bigint)",
        migration_lock_id(filename),
    )
    row = await conn.fetchrow(
        """
        SELECT checksum, success
        FROM _migrations
        WHERE filename = $1
        FOR UPDATE
        """,
        filename,
    )
    if row is None:
        return True
    if bool(row["success"]):
        if row["checksum"] != checksum:
            raise MigrationDriftError(
                f"immutable migration checksum drift for {filename}: "
                f"recorded={row['checksum']}, current={checksum}"
            )
        return False
    await conn.execute(
        "DELETE FROM _migrations WHERE filename = $1 AND success = FALSE",
        filename,
    )
    return True


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


async def column_exists(conn, full_table_name: str, column_name: str) -> bool:
    """Check whether a required column exists on a schema-qualified table."""
    if "." in full_table_name:
        schema, table = full_table_name.split(".", 1)
    else:
        schema, table = "public", full_table_name

    return await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2 AND column_name = $3
        )
    """, schema, table, column_name)


async def get_missing_required_columns(conn, plan: str) -> list[str]:
    """Return and report every absent postcondition column for a plan."""
    missing = []
    for table, columns in REQUIRED_COLUMNS_BY_PLAN.get(plan, {}).items():
        for column, description in columns.items():
            if not await column_exists(conn, table, column):
                missing.append(f"{table}.{column}")
                logger.warning(
                    "  ✗ %s.%s - MISSING (%s)", table, column, description
                )
    return missing


async def validate_required_columns(conn, plan: str) -> bool:
    """Fail closed when a plan's postcondition columns are absent."""
    return not await get_missing_required_columns(conn, plan)


async def validate_plan_prerequisites(conn, plan: str) -> bool:
    """Fail closed before plan DDL when an upstream schema is absent."""
    missing = [
        table
        for table in PLAN_PREREQUISITE_TABLES.get(plan, ())
        if not await table_exists(conn, table)
    ]
    if missing:
        logger.error(
            "Plan %s prerequisites are missing: %s",
            plan,
            ", ".join(missing),
        )
        return False
    return True


async def run_migrations(
    plan: str = "legacy-init", reviewed_digest: str | None = None
) -> bool:
    """Run all pending migrations."""
    if not plan_is_authorized(plan, reviewed_digest):
        return False
    logger.info("=" * 60)
    logger.info("USDCOP Database Migration System")
    logger.info("=" * 60)

    try:
        conn = await get_connection()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        return False

    plan_lock_acquired = False
    try:
        if not await validate_plan_prerequisites(conn, plan):
            return False
        # Ensure migrations table exists
        await ensure_migrations_table(conn)
        await conn.fetchval(
            "SELECT pg_advisory_lock($1::bigint)",
            migration_lock_id(f"plan:{plan}"),
        )
        plan_lock_acquired = True

        # Get already executed migrations
        executed = await get_executed_migrations(conn)
        logger.info(f"Previously executed migrations: {len(executed)}")

        # Get all migration files
        migration_files = get_migration_files(plan)
        logger.info(f"Total migration files: {len(migration_files)}")

        try:
            pending = classify_migrations(plan, executed)
        except MigrationDriftError as exc:
            logger.error("%s", exc)
            return False

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
        if plan_lock_acquired:
            try:
                await conn.fetchval(
                    "SELECT pg_advisory_unlock($1::bigint)",
                    migration_lock_id(f"plan:{plan}"),
                )
            except Exception as exc:
                logger.warning("Could not explicitly release migration plan lock: %s", exc)
        await conn.close()


async def show_status(plan: str = "legacy-init") -> bool:
    """Show migration status."""
    logger.info("=" * 60)
    logger.info("Migration Status")
    logger.info("=" * 60)

    try:
        conn = await get_connection()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        return False

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
        migration_files = get_migration_files(plan)

        try:
            pending = [
                filepath
                for filepath, _checksum, _reason in classify_migrations(
                    plan, executed
                )
            ]
        except MigrationDriftError as exc:
            logger.error("%s", exc)
            return False
        if pending:
            logger.info(f"\nPending migrations: {len(pending)}")
            for f in pending:
                logger.info(f"  - {f.name}")
        else:
            logger.info("Selected plan is up to date and checksum-clean.")
        return True

    finally:
        await conn.close()


async def validate_tables(plan: str = "legacy-init") -> bool:
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

        for table_name, description in REQUIRED_TABLES_BY_PLAN[plan].items():
            exists = await table_exists(conn, table_name)
            if exists:
                present.append(table_name)
                logger.info(f"  ✓ {table_name}")
            else:
                missing.append(table_name)
                logger.warning(f"  ✗ {table_name} - MISSING ({description})")

        missing_columns = await get_missing_required_columns(conn, plan)

        logger.info("-" * 60)
        logger.info(
            "Present tables: %d, Missing tables: %d, Missing columns: %d",
            len(present),
            len(missing),
            len(missing_columns),
        )

        if missing or missing_columns:
            logger.error(
                "Run scripts/ops/db_migrate.py with --plan %s and its reviewed "
                "digest to create missing tables",
                plan,
            )
            return False

        logger.info("All required tables exist!")
        return True

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(description="Database Migration System")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--validate", action="store_true", help="Validate required tables")
    parser.add_argument(
        "--plan",
        choices=tuple(MIGRATION_PLANS),
        required=True,
        help="Explicit allowlisted migration plan",
    )
    parser.add_argument(
        "--plan-digest",
        action="store_true",
        help="Print the ordered content digest without connecting to a database",
    )
    parser.add_argument(
        "--reviewed-digest",
        help="Required exact digest for review-gated plans",
    )
    args = parser.parse_args()

    if args.plan_digest:
        print(get_plan_digest(args.plan))
    elif args.status:
        success = asyncio.run(show_status(args.plan))
        sys.exit(0 if success else 1)
    elif args.validate:
        success = asyncio.run(validate_tables(args.plan))
        sys.exit(0 if success else 1)
    else:
        success = asyncio.run(run_migrations(args.plan, args.reviewed_digest))
        # Also validate after running migrations
        if success:
            success = asyncio.run(validate_tables(args.plan))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
