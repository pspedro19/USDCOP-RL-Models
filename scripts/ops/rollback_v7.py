#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.1 Rollback Script
====================
Safely rolls back V7.1 event-driven architecture to V6 polling-based system.

Operations:
1. Verify current state (is V7.1 active?)
2. Disable event triggers
3. Clear event tables
4. Restart sensors in polling mode
5. Verify rollback success

Usage:
    # Dry run (no changes)
    python scripts/rollback_v7.py --dry-run

    # Execute rollback
    python scripts/rollback_v7.py --rollback

    # Rollback with SQL execution
    python scripts/rollback_v7.py --rollback --execute-sql

Author: Trading Team
Version: 1.0.0
Created: 2026-01-31
Contract: CTR-V7-ROLLBACK
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
ROLLBACK_SQL_PATH = PROJECT_ROOT / "database" / "migrations" / "rollback_033_event_triggers.sql"


def get_connection_string() -> str:
    """Get database connection string."""
    return os.environ.get(
        "DATABASE_URL",
        os.environ.get("TIMESCALE_URL", "postgresql://postgres:postgres@localhost:5432/trading")
    )


# =============================================================================
# ROLLBACK CHECKER
# =============================================================================

class RollbackChecker:
    """Checks if V7.1 is currently active."""

    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.conn = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.conn_string)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def check_v7_active(self) -> Dict[str, Any]:
        """Check if V7.1 components are active."""
        if not self.conn:
            return {"error": "Not connected"}

        status = {
            "v7_active": False,
            "triggers_present": [],
            "tables_present": [],
            "events_in_dlq": 0,
            "processed_events": 0,
        }

        cur = self.conn.cursor()

        try:
            # Check triggers
            cur.execute("""
                SELECT tgname FROM pg_trigger t
                JOIN pg_class c ON t.tgrelid = c.oid
                WHERE tgname LIKE 'trg_notify_%'
            """)
            status["triggers_present"] = [row[0] for row in cur.fetchall()]

            # Check tables
            for table in ['event_dead_letter_queue', 'event_processed_log', 'circuit_breaker_state']:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    )
                """, (table,))
                if cur.fetchone()[0]:
                    status["tables_present"].append(table)

            # Check DLQ size
            if 'event_dead_letter_queue' in status["tables_present"]:
                cur.execute("SELECT COUNT(*) FROM event_dead_letter_queue")
                status["events_in_dlq"] = cur.fetchone()[0]

            # Check processed events
            if 'event_processed_log' in status["tables_present"]:
                cur.execute("SELECT COUNT(*) FROM event_processed_log")
                status["processed_events"] = cur.fetchone()[0]

            # Determine if V7 is active
            status["v7_active"] = len(status["triggers_present"]) > 0

        except Exception as e:
            logger.error(f"Error checking V7 status: {e}")
            status["error"] = str(e)

        finally:
            cur.close()

        return status

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# =============================================================================
# ROLLBACK EXECUTOR
# =============================================================================

class RollbackExecutor:
    """Executes V7.1 rollback operations."""

    def __init__(self, conn_string: str, dry_run: bool = True):
        self.conn_string = conn_string
        self.dry_run = dry_run
        self.conn = None
        self.operations_log: List[str] = []

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.conn_string)
            self.conn.autocommit = False
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def log_operation(self, operation: str, result: str = "pending"):
        """Log an operation."""
        entry = f"[{datetime.utcnow().isoformat()}] {operation}: {result}"
        self.operations_log.append(entry)
        logger.info(entry)

    def drop_triggers(self) -> bool:
        """Drop V7.1 NOTIFY triggers."""
        self.log_operation("Drop OHLCV trigger")
        self.log_operation("Drop Features trigger")

        if self.dry_run:
            logger.info("  [DRY RUN] Would drop triggers")
            return True

        cur = self.conn.cursor()
        try:
            cur.execute("DROP TRIGGER IF EXISTS trg_notify_new_ohlcv_bar ON usdcop_m5_ohlcv")
            cur.execute("DROP TRIGGER IF EXISTS trg_notify_features_ready ON inference_features_5m")
            self.log_operation("Triggers dropped", "success")
            return True
        except Exception as e:
            self.log_operation("Trigger drop failed", str(e))
            return False
        finally:
            cur.close()

    def drop_functions(self) -> bool:
        """Drop V7.1 functions."""
        functions = [
            "notify_new_ohlcv_bar()",
            "notify_features_ready()",
            "emit_heartbeat(TEXT)",
            "dlq_enqueue(VARCHAR, VARCHAR, VARCHAR, JSONB, TEXT)",
            "is_event_processed(VARCHAR)",
            "mark_event_processed(VARCHAR, VARCHAR, VARCHAR, VARCHAR)",
            "cleanup_old_processed_events()",
        ]

        self.log_operation(f"Drop {len(functions)} functions")

        if self.dry_run:
            logger.info("  [DRY RUN] Would drop functions")
            return True

        cur = self.conn.cursor()
        try:
            for func in functions:
                cur.execute(f"DROP FUNCTION IF EXISTS {func}")
            self.log_operation("Functions dropped", "success")
            return True
        except Exception as e:
            self.log_operation("Function drop failed", str(e))
            return False
        finally:
            cur.close()

    def clear_event_tables(self) -> bool:
        """Clear and optionally drop event tables."""
        tables = ['event_dead_letter_queue', 'event_processed_log', 'circuit_breaker_state']

        self.log_operation(f"Clear {len(tables)} event tables")

        if self.dry_run:
            logger.info("  [DRY RUN] Would clear tables")
            return True

        cur = self.conn.cursor()
        try:
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            self.log_operation("Tables cleared", "success")
            return True
        except Exception as e:
            self.log_operation("Table clear failed", str(e))
            return False
        finally:
            cur.close()

    def drop_views(self) -> bool:
        """Drop V7.1 views."""
        views = ['v_event_system_health', 'v_dlq_summary']

        self.log_operation(f"Drop {len(views)} views")

        if self.dry_run:
            logger.info("  [DRY RUN] Would drop views")
            return True

        cur = self.conn.cursor()
        try:
            for view in views:
                cur.execute(f"DROP VIEW IF EXISTS {view}")
            self.log_operation("Views dropped", "success")
            return True
        except Exception as e:
            self.log_operation("View drop failed", str(e))
            return False
        finally:
            cur.close()

    def execute_rollback(self) -> Tuple[bool, List[str]]:
        """Execute full rollback sequence."""
        logger.info("=" * 60)
        logger.info("V7.1 ROLLBACK EXECUTION")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        steps = [
            ("Drop triggers", self.drop_triggers),
            ("Drop functions", self.drop_functions),
            ("Drop views", self.drop_views),
            ("Clear event tables", self.clear_event_tables),
        ]

        all_success = True
        for step_name, step_func in steps:
            logger.info(f"\nStep: {step_name}")
            if not step_func():
                all_success = False
                logger.error(f"  FAILED: {step_name}")
                break
            logger.info(f"  OK: {step_name}")

        if all_success and not self.dry_run:
            try:
                self.conn.commit()
                self.log_operation("COMMIT", "success")
            except Exception as e:
                self.log_operation("COMMIT", f"failed: {e}")
                self.conn.rollback()
                all_success = False

        return all_success, self.operations_log

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()


# =============================================================================
# POST-ROLLBACK VALIDATOR
# =============================================================================

class PostRollbackValidator:
    """Validates rollback was successful."""

    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.conn = None

    def connect(self) -> bool:
        """Connect to database."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.conn_string)
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def validate(self) -> Dict[str, Any]:
        """Run validation checks."""
        results = {
            "valid": True,
            "triggers_removed": True,
            "tables_removed": True,
            "functions_removed": True,
            "polling_working": None,
            "issues": []
        }

        cur = self.conn.cursor()

        try:
            # Check triggers removed
            cur.execute("""
                SELECT COUNT(*) FROM pg_trigger t
                JOIN pg_class c ON t.tgrelid = c.oid
                WHERE tgname LIKE 'trg_notify_%'
            """)
            if cur.fetchone()[0] > 0:
                results["triggers_removed"] = False
                results["valid"] = False
                results["issues"].append("NOTIFY triggers still present")

            # Check tables removed
            for table in ['event_dead_letter_queue', 'event_processed_log', 'circuit_breaker_state']:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    )
                """, (table,))
                if cur.fetchone()[0]:
                    results["tables_removed"] = False
                    results["valid"] = False
                    results["issues"].append(f"Table {table} still exists")

            # Check core data tables still exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'usdcop_m5_ohlcv'
                )
            """)
            if not cur.fetchone()[0]:
                results["issues"].append("WARNING: Core table usdcop_m5_ohlcv missing")

        except Exception as e:
            results["issues"].append(f"Validation error: {e}")
            results["valid"] = False

        finally:
            cur.close()

        return results

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V7.1 Rollback Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check current V7.1 status
    python rollback_v7.py --status

    # Dry run (no changes)
    python rollback_v7.py --dry-run

    # Execute rollback
    python rollback_v7.py --rollback

    # Execute SQL rollback file directly
    python rollback_v7.py --execute-sql
        """
    )

    parser.add_argument("--status", action="store_true", help="Check V7.1 status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--rollback", action="store_true", help="Execute rollback")
    parser.add_argument("--execute-sql", action="store_true", help="Execute SQL rollback file")
    parser.add_argument("--validate", action="store_true", help="Validate post-rollback")
    parser.add_argument("--connection", type=str, help="Database connection string")

    args = parser.parse_args()

    conn_string = args.connection or get_connection_string()

    # Status check
    if args.status:
        logger.info("Checking V7.1 status...")
        checker = RollbackChecker(conn_string)
        if checker.connect():
            status = checker.check_v7_active()
            checker.close()

            print("\n" + "=" * 60)
            print("V7.1 STATUS")
            print("=" * 60)
            print(f"V7 Active:        {status['v7_active']}")
            print(f"Triggers:         {status['triggers_present']}")
            print(f"Tables:           {status['tables_present']}")
            print(f"Events in DLQ:    {status['events_in_dlq']}")
            print(f"Processed events: {status['processed_events']}")
            print("=" * 60)
        return

    # Dry run
    if args.dry_run:
        executor = RollbackExecutor(conn_string, dry_run=True)
        if executor.connect():
            success, log = executor.execute_rollback()
            executor.close()

            print("\n" + "=" * 60)
            print("DRY RUN COMPLETE")
            print("=" * 60)
            print("No changes were made.")
            print("Run with --rollback to execute.")
        return

    # Execute rollback
    if args.rollback:
        print("\n" + "=" * 60)
        print("⚠️  V7.1 ROLLBACK")
        print("=" * 60)
        print("This will:")
        print("  - Remove all NOTIFY triggers")
        print("  - Drop event tables (DLQ, processed log)")
        print("  - Revert to V6 polling mode")
        print("-" * 60)

        confirm = input("Type 'ROLLBACK' to confirm: ")
        if confirm != "ROLLBACK":
            print("Aborted.")
            return

        executor = RollbackExecutor(conn_string, dry_run=False)
        if executor.connect():
            success, log = executor.execute_rollback()
            executor.close()

            print("\n" + "=" * 60)
            if success:
                print("✅ ROLLBACK SUCCESSFUL")
                print("System reverted to V6 (polling mode)")
                print("Restart Airflow to apply changes.")
            else:
                print("❌ ROLLBACK FAILED")
                print("Check logs for details.")
            print("=" * 60)
        return

    # Execute SQL file
    if args.execute_sql:
        if not ROLLBACK_SQL_PATH.exists():
            logger.error(f"SQL file not found: {ROLLBACK_SQL_PATH}")
            return

        print(f"Executing: {ROLLBACK_SQL_PATH}")
        try:
            result = subprocess.run(
                ["psql", conn_string, "-f", str(ROLLBACK_SQL_PATH)],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
        return

    # Validate
    if args.validate:
        validator = PostRollbackValidator(conn_string)
        if validator.connect():
            results = validator.validate()
            validator.close()

            print("\n" + "=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            print(f"Valid:            {'✅' if results['valid'] else '❌'}")
            print(f"Triggers removed: {'✅' if results['triggers_removed'] else '❌'}")
            print(f"Tables removed:   {'✅' if results['tables_removed'] else '❌'}")

            if results['issues']:
                print("\nIssues:")
                for issue in results['issues']:
                    print(f"  - {issue}")
            print("=" * 60)
        return

    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
