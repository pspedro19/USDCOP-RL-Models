#!/usr/bin/env python3
"""
Database Backup Script
=======================

Creates complete PostgreSQL/TimescaleDB backups for project replication.
Supports both full SQL dumps and table-by-table CSV exports.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14

Usage:
    python scripts/backup/backup_database.py [--output-dir PATH] [--format sql|csv|both]
"""

import os
import sys
import gzip
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


class DatabaseBackup:
    """PostgreSQL/TimescaleDB backup manager."""

    # Critical tables that MUST be backed up
    CRITICAL_TABLES = [
        # Schema: public
        ("public", "usdcop_m5_ohlcv"),
        ("public", "macro_indicators_daily"),
        ("public", "trading_state"),
        ("public", "trades_history"),
        ("public", "equity_snapshots"),
        ("public", "inference_features_5m"),
        ("public", "users"),
        ("public", "trading_sessions"),
        ("public", "trading_metrics"),
        # Schema: config
        ("config", "models"),
        ("config", "feature_definitions"),
        ("config", "model_registry"),
        # Schema: trading
        ("trading", "model_inferences"),
        ("trading", "model_trades"),
        ("trading", "model_states"),
        # Schema: events
        ("events", "signals_stream"),
        # Schema: metrics
        ("metrics", "model_performance"),
        # Schema: dw (data warehouse)
        ("dw", "fact_rl_inference"),
        ("dw", "fact_strategy_performance"),
        ("dw", "fact_equity_curve_realtime"),
    ]

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """Initialize database connection parameters."""
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "usdcop_trading")
        self.user = user or os.getenv("POSTGRES_USER", "admin")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

    def _get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def _run_psql(self, command: str, output_file: Optional[str] = None) -> str:
        """Execute psql command."""
        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        cmd = [
            "psql",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "-c", command
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            raise Exception(f"psql error: {result.stderr}")

        return result.stdout

    def backup_full_sql(self, output_path: Path) -> Path:
        """
        Create full SQL dump of the database.

        Args:
            output_path: Directory to save backup

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = output_path / f"usdcop_full_backup_{timestamp}.sql.gz"

        print(f"Creating full SQL backup: {backup_file}")

        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        # Use pg_dump with compression
        cmd = [
            "pg_dump",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "--format=plain",
            "--no-owner",
            "--no-privileges",
            "--clean",
            "--if-exists"
        ]

        with gzip.open(backup_file, "wt", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

        if result.returncode != 0:
            raise Exception(f"pg_dump error: {result.stderr}")

        print(f"  Size: {backup_file.stat().st_size / 1024 / 1024:.2f} MB")
        return backup_file

    def backup_schema_ddl(self, output_path: Path) -> Path:
        """Export database schema (DDL only, no data)."""
        backup_file = output_path / "schema_ddl.sql.gz"

        print(f"Exporting schema DDL: {backup_file}")

        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        cmd = [
            "pg_dump",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "--schema-only",
            "--no-owner",
            "--no-privileges"
        ]

        with gzip.open(backup_file, "wt", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

        if result.returncode != 0:
            raise Exception(f"pg_dump error: {result.stderr}")

        return backup_file

    def backup_table_csv(
        self,
        schema: str,
        table: str,
        output_path: Path
    ) -> Optional[Path]:
        """
        Export single table to compressed CSV.

        Args:
            schema: Database schema name
            table: Table name
            output_path: Directory to save backup

        Returns:
            Path to backup file or None if table doesn't exist
        """
        backup_file = output_path / f"{schema}_{table}.csv.gz"
        full_table = f"{schema}.{table}"

        print(f"  Backing up {full_table}...", end=" ")

        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        # Check if table exists
        check_cmd = [
            "psql",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "-tAc",
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='{schema}' AND table_name='{table}'"
        ]

        result = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
        if result.stdout.strip() != "1":
            print("SKIP (not found)")
            return None

        # Get row count
        count_cmd = [
            "psql",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "-tAc",
            f"SELECT COUNT(*) FROM {full_table}"
        ]
        result = subprocess.run(count_cmd, capture_output=True, text=True, env=env)
        row_count = int(result.stdout.strip() or 0)

        # Export to CSV
        export_cmd = [
            "psql",
            "-h", self.host,
            "-p", str(self.port),
            "-U", self.user,
            "-d", self.database,
            "-c",
            f"\\COPY {full_table} TO STDOUT WITH CSV HEADER"
        ]

        with gzip.open(backup_file, "wt", encoding="utf-8") as f:
            result = subprocess.run(
                export_cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return None

        size_kb = backup_file.stat().st_size / 1024
        print(f"OK ({row_count:,} rows, {size_kb:.1f} KB)")

        return backup_file

    def backup_all_tables_csv(self, output_path: Path) -> Dict:
        """
        Backup all critical tables to CSV files.

        Args:
            output_path: Directory to save backups

        Returns:
            Dictionary with backup results
        """
        print("\n" + "="*60)
        print("BACKING UP DATABASE TABLES")
        print("="*60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "database": self.database,
            "host": self.host,
            "tables": {}
        }

        for schema, table in self.CRITICAL_TABLES:
            backup_file = self.backup_table_csv(schema, table, output_path)
            if backup_file:
                results["tables"][f"{schema}.{table}"] = {
                    "file": backup_file.name,
                    "size_bytes": backup_file.stat().st_size
                }

        return results

    def create_manifest(self, output_path: Path, results: Dict) -> Path:
        """Create backup manifest with metadata."""
        manifest_file = output_path / "MANIFEST.json"

        manifest = {
            "created_at": datetime.now().isoformat(),
            "database": {
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "user": self.user
            },
            "backup_contents": results,
            "restore_instructions": [
                "1. Ensure PostgreSQL/TimescaleDB is running",
                "2. Create database: createdb -U admin usdcop_trading",
                "3. Restore schema: gunzip -c schema_ddl.sql.gz | psql -U admin -d usdcop_trading",
                "4. Restore data: Use restore_database.py script",
                "5. Verify with: SELECT COUNT(*) FROM public.usdcop_m5_ohlcv"
            ]
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return manifest_file


def main():
    """Main backup function."""
    parser = argparse.ArgumentParser(description="Database Backup Tool")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "backups" / "database"),
        help="Output directory for backups"
    )
    parser.add_argument(
        "--format",
        choices=["sql", "csv", "both"],
        default="both",
        help="Backup format: sql (full dump), csv (table exports), both"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run inside Docker container"
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"backup_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DATABASE BACKUP TOOL")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Format: {args.format}")
    print()

    # Initialize backup manager
    if args.docker:
        backup = DatabaseBackup(host="postgres")
    else:
        backup = DatabaseBackup()

    results = {"files": []}

    try:
        # Schema DDL (always)
        ddl_file = backup.backup_schema_ddl(output_path)
        results["files"].append(str(ddl_file.name))

        if args.format in ["sql", "both"]:
            # Full SQL dump
            sql_file = backup.backup_full_sql(output_path)
            results["files"].append(str(sql_file.name))

        if args.format in ["csv", "both"]:
            # Table-by-table CSV exports
            table_results = backup.backup_all_tables_csv(output_path)
            results["tables"] = table_results["tables"]

        # Create manifest
        manifest = backup.create_manifest(output_path, results)

        print("\n" + "="*60)
        print("BACKUP COMPLETE")
        print("="*60)
        print(f"Location: {output_path}")
        print(f"Manifest: {manifest}")
        print(f"Files created: {len(results.get('files', [])) + len(results.get('tables', {}))}")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
