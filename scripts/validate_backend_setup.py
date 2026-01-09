#!/usr/bin/env python3
"""
Multi-Model Backend Validation Script
======================================

Validates the complete backend setup including:
1. Database tables exist
2. Models are configured
3. Features are loaded
4. Redis is accessible
5. API endpoints respond
6. Model files exist (MinIO)

Usage:
    python scripts/validate_backend_setup.py

    # With verbose output
    python scripts/validate_backend_setup.py -v

    # JSON output for CI/CD
    python scripts/validate_backend_setup.py --json

Author: USDCOP Trading Team
Version: 1.0.0
Date: 2025-12-26
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Status(Enum):
    """Validation status enum."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    status: Status
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_checks: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    results: List[ValidationResult]
    overall_status: str


class BackendValidator:
    """Validates multi-model backend setup."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

        # Configuration
        self.db_config = {
            'host': os.environ.get('POSTGRES_HOST', 'localhost'),
            'port': int(os.environ.get('POSTGRES_PORT', '5432')),
            'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
            'user': os.environ.get('POSTGRES_USER', 'admin'),
            'password': os.environ.get('POSTGRES_PASSWORD', 'admin123')
        }

        self.redis_config = {
            'host': os.environ.get('REDIS_HOST', 'localhost'),
            'port': int(os.environ.get('REDIS_PORT', '6379')),
            'password': os.environ.get('REDIS_PASSWORD', '')
        }

        self.minio_config = {
            'endpoint': os.environ.get('MINIO_ENDPOINT', 'localhost:9000'),
            'access_key': os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
            'secret_key': os.environ.get('MINIO_SECRET_KEY', 'minioadmin123'),
            'secure': False
        }

        self.api_endpoints = {
            'trading': 'http://localhost:8000',
            'multi_model': 'http://localhost:8006',
            'analytics': 'http://localhost:8001'
        }

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")

    def add_result(self, name: str, status: Status, message: str, details: Dict = None):
        """Add a validation result."""
        result = ValidationResult(name=name, status=status, message=message, details=details)
        self.results.append(result)

        status_icon = {
            Status.PASS: "[PASS]",
            Status.FAIL: "[FAIL]",
            Status.WARN: "[WARN]",
            Status.SKIP: "[SKIP]"
        }

        print(f"{status_icon[status]} {name}: {message}")

    # =========================================================================
    # DATABASE VALIDATIONS
    # =========================================================================

    def check_database_connection(self) -> bool:
        """Check if database is accessible."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            self.add_result(
                "Database Connection",
                Status.PASS,
                f"Connected to {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            return True
        except ImportError:
            self.add_result(
                "Database Connection",
                Status.SKIP,
                "psycopg2 not installed"
            )
            return False
        except Exception as e:
            self.add_result(
                "Database Connection",
                Status.FAIL,
                f"Connection failed: {str(e)}"
            )
            return False

    def check_core_tables(self) -> bool:
        """Check if core tables exist."""
        required_tables = [
            ('public', 'usdcop_m5_ohlcv'),
            ('public', 'macro_indicators_daily'),
            ('public', 'users'),
            ('public', 'trading_metrics'),
        ]

        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            missing_tables = []
            existing_tables = []

            for schema, table in required_tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                    )
                """, (schema, table))
                exists = cur.fetchone()[0]

                if exists:
                    existing_tables.append(f"{schema}.{table}")
                else:
                    missing_tables.append(f"{schema}.{table}")

            cur.close()
            conn.close()

            if missing_tables:
                self.add_result(
                    "Core Tables",
                    Status.FAIL,
                    f"Missing tables: {', '.join(missing_tables)}",
                    {'existing': existing_tables, 'missing': missing_tables}
                )
                return False
            else:
                self.add_result(
                    "Core Tables",
                    Status.PASS,
                    f"All {len(required_tables)} core tables exist",
                    {'tables': existing_tables}
                )
                return True

        except Exception as e:
            self.add_result(
                "Core Tables",
                Status.FAIL,
                f"Error checking tables: {str(e)}"
            )
            return False

    def check_dw_schema_tables(self) -> bool:
        """Check if DW schema tables exist."""
        required_tables = [
            'dim_strategy',
            'fact_strategy_signals',
            'fact_equity_curve',
            'fact_strategy_performance',
            'fact_strategy_positions',
            'fact_rl_inference'
        ]

        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check if dw schema exists
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'dw')")
            schema_exists = cur.fetchone()[0]

            if not schema_exists:
                self.add_result(
                    "DW Schema Tables",
                    Status.WARN,
                    "DW schema does not exist - multi-model features unavailable"
                )
                cur.close()
                conn.close()
                return False

            missing_tables = []
            existing_tables = []

            for table in required_tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'dw' AND table_name = %s
                    )
                """, (table,))
                exists = cur.fetchone()[0]

                if exists:
                    existing_tables.append(table)
                else:
                    missing_tables.append(table)

            cur.close()
            conn.close()

            if missing_tables:
                self.add_result(
                    "DW Schema Tables",
                    Status.WARN,
                    f"Missing DW tables: {', '.join(missing_tables)}",
                    {'existing': existing_tables, 'missing': missing_tables}
                )
                return False
            else:
                self.add_result(
                    "DW Schema Tables",
                    Status.PASS,
                    f"All {len(required_tables)} DW tables exist",
                    {'tables': existing_tables}
                )
                return True

        except Exception as e:
            self.add_result(
                "DW Schema Tables",
                Status.FAIL,
                f"Error checking DW tables: {str(e)}"
            )
            return False

    def check_ohlcv_data(self) -> bool:
        """Check if OHLCV data exists."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    MIN(time) as earliest,
                    MAX(time) as latest
                FROM usdcop_m5_ohlcv
                WHERE symbol = 'USD/COP'
            """)

            result = cur.fetchone()
            total, earliest, latest = result

            cur.close()
            conn.close()

            if total == 0:
                self.add_result(
                    "OHLCV Data",
                    Status.WARN,
                    "No OHLCV data found - run data seeder or L0 pipeline"
                )
                return False
            else:
                self.add_result(
                    "OHLCV Data",
                    Status.PASS,
                    f"{total:,} bars from {earliest} to {latest}",
                    {'total': total, 'earliest': str(earliest), 'latest': str(latest)}
                )
                return True

        except Exception as e:
            self.add_result(
                "OHLCV Data",
                Status.FAIL,
                f"Error checking OHLCV data: {str(e)}"
            )
            return False

    def check_strategy_configuration(self) -> bool:
        """Check if strategies are configured in dim_strategy."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check if table exists first
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'dw' AND table_name = 'dim_strategy'
                )
            """)

            if not cur.fetchone()[0]:
                self.add_result(
                    "Strategy Configuration",
                    Status.SKIP,
                    "dim_strategy table does not exist"
                )
                cur.close()
                conn.close()
                return False

            cur.execute("""
                SELECT strategy_code, strategy_name, strategy_type, is_active
                FROM dw.dim_strategy
                ORDER BY strategy_id
            """)

            strategies = cur.fetchall()
            cur.close()
            conn.close()

            if len(strategies) == 0:
                self.add_result(
                    "Strategy Configuration",
                    Status.WARN,
                    "No strategies configured in dim_strategy"
                )
                return False

            active_count = sum(1 for s in strategies if s[3])

            strategy_list = [
                {'code': s[0], 'name': s[1], 'type': s[2], 'active': s[3]}
                for s in strategies
            ]

            self.add_result(
                "Strategy Configuration",
                Status.PASS,
                f"{len(strategies)} strategies configured ({active_count} active)",
                {'strategies': strategy_list}
            )
            return True

        except Exception as e:
            self.add_result(
                "Strategy Configuration",
                Status.FAIL,
                f"Error checking strategies: {str(e)}"
            )
            return False

    def check_macro_indicators(self) -> bool:
        """Check macro indicators data."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Try different possible column names
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'macro_indicators_daily'
                AND column_name IN ('fecha', 'date')
                LIMIT 1
            """)

            date_col_result = cur.fetchone()
            if not date_col_result:
                self.add_result(
                    "Macro Indicators",
                    Status.WARN,
                    "macro_indicators_daily table exists but date column not found"
                )
                cur.close()
                conn.close()
                return False

            date_col = date_col_result[0]

            cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    MIN({date_col}) as earliest,
                    MAX({date_col}) as latest
                FROM macro_indicators_daily
            """)

            result = cur.fetchone()
            total, earliest, latest = result

            cur.close()
            conn.close()

            if total == 0:
                self.add_result(
                    "Macro Indicators",
                    Status.WARN,
                    "No macro data - run macro pipeline DAG"
                )
                return False
            else:
                self.add_result(
                    "Macro Indicators",
                    Status.PASS,
                    f"{total:,} days from {earliest} to {latest}",
                    {'total': total, 'earliest': str(earliest), 'latest': str(latest)}
                )
                return True

        except Exception as e:
            self.add_result(
                "Macro Indicators",
                Status.FAIL,
                f"Error checking macro data: {str(e)}"
            )
            return False

    # =========================================================================
    # REDIS VALIDATIONS
    # =========================================================================

    def check_redis_connection(self) -> bool:
        """Check if Redis is accessible."""
        try:
            import redis
            r = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config['password'] or None,
                decode_responses=True
            )

            pong = r.ping()
            info = r.info('server')

            self.add_result(
                "Redis Connection",
                Status.PASS,
                f"Connected to Redis {info.get('redis_version', 'unknown')} at {self.redis_config['host']}:{self.redis_config['port']}"
            )
            return True

        except ImportError:
            self.add_result(
                "Redis Connection",
                Status.SKIP,
                "redis-py not installed"
            )
            return False
        except Exception as e:
            self.add_result(
                "Redis Connection",
                Status.FAIL,
                f"Connection failed: {str(e)}"
            )
            return False

    def check_redis_streams(self) -> bool:
        """Check if Redis streams exist."""
        expected_streams = [
            'trading:signals',
            'trading:prices',
            'trading:positions'
        ]

        try:
            import redis
            r = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config['password'] or None,
                decode_responses=True
            )

            existing_streams = []
            missing_streams = []

            for stream in expected_streams:
                try:
                    info = r.xinfo_stream(stream)
                    existing_streams.append({
                        'name': stream,
                        'length': info.get('length', 0)
                    })
                except redis.ResponseError:
                    missing_streams.append(stream)

            if existing_streams:
                self.add_result(
                    "Redis Streams",
                    Status.PASS,
                    f"{len(existing_streams)} streams configured",
                    {'streams': existing_streams}
                )
                return True
            else:
                self.add_result(
                    "Redis Streams",
                    Status.WARN,
                    "No trading streams configured (will be created on first use)"
                )
                return False

        except Exception as e:
            self.add_result(
                "Redis Streams",
                Status.WARN,
                f"Could not check streams: {str(e)}"
            )
            return False

    # =========================================================================
    # API VALIDATIONS
    # =========================================================================

    def check_api_endpoint(self, name: str, base_url: str) -> bool:
        """Check if an API endpoint is responding."""
        try:
            import requests

            # Try health endpoint
            health_url = f"{base_url}/api/health"
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')

                self.add_result(
                    f"API: {name}",
                    Status.PASS,
                    f"Healthy at {base_url} (status: {status})",
                    {'url': base_url, 'response': data}
                )
                return True
            else:
                self.add_result(
                    f"API: {name}",
                    Status.FAIL,
                    f"Unhealthy - HTTP {response.status_code}"
                )
                return False

        except ImportError:
            self.add_result(
                f"API: {name}",
                Status.SKIP,
                "requests library not installed"
            )
            return False
        except requests.exceptions.ConnectionError:
            self.add_result(
                f"API: {name}",
                Status.FAIL,
                f"Connection refused at {base_url}"
            )
            return False
        except requests.exceptions.Timeout:
            self.add_result(
                f"API: {name}",
                Status.FAIL,
                f"Timeout connecting to {base_url}"
            )
            return False
        except Exception as e:
            self.add_result(
                f"API: {name}",
                Status.FAIL,
                f"Error: {str(e)}"
            )
            return False

    def check_all_apis(self) -> bool:
        """Check all API endpoints."""
        all_healthy = True

        for name, url in self.api_endpoints.items():
            if not self.check_api_endpoint(name, url):
                all_healthy = False

        return all_healthy

    # =========================================================================
    # MINIO VALIDATIONS
    # =========================================================================

    def check_minio_connection(self) -> bool:
        """Check if MinIO is accessible."""
        try:
            from minio import Minio

            client = Minio(
                self.minio_config['endpoint'],
                access_key=self.minio_config['access_key'],
                secret_key=self.minio_config['secret_key'],
                secure=self.minio_config['secure']
            )

            # List buckets to verify connection
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets]

            self.add_result(
                "MinIO Connection",
                Status.PASS,
                f"Connected - {len(buckets)} buckets found",
                {'buckets': bucket_names}
            )
            return True

        except ImportError:
            self.add_result(
                "MinIO Connection",
                Status.SKIP,
                "minio library not installed"
            )
            return False
        except Exception as e:
            self.add_result(
                "MinIO Connection",
                Status.FAIL,
                f"Connection failed: {str(e)}"
            )
            return False

    def check_model_files(self) -> bool:
        """Check if model files exist in MinIO."""
        expected_buckets = [
            '99-common-trading-models',
            '05-l5-ds-usdcop-serving',
            '04-l4-ds-usdcop-rlready'
        ]

        try:
            from minio import Minio

            client = Minio(
                self.minio_config['endpoint'],
                access_key=self.minio_config['access_key'],
                secret_key=self.minio_config['secret_key'],
                secure=self.minio_config['secure']
            )

            existing_buckets = []
            missing_buckets = []
            model_count = 0

            for bucket in expected_buckets:
                if client.bucket_exists(bucket):
                    existing_buckets.append(bucket)
                    # Count objects in bucket
                    objects = list(client.list_objects(bucket, recursive=True))
                    model_count += len(objects)
                else:
                    missing_buckets.append(bucket)

            if existing_buckets:
                self.add_result(
                    "Model Files",
                    Status.PASS,
                    f"{len(existing_buckets)} buckets, {model_count} objects",
                    {'buckets': existing_buckets, 'missing': missing_buckets, 'object_count': model_count}
                )
                return True
            else:
                self.add_result(
                    "Model Files",
                    Status.WARN,
                    "No model buckets found - run minio-init"
                )
                return False

        except Exception as e:
            self.add_result(
                "Model Files",
                Status.WARN,
                f"Could not check model files: {str(e)}"
            )
            return False

    # =========================================================================
    # FEATURE VALIDATIONS
    # =========================================================================

    def check_feature_configuration(self) -> bool:
        """Check if feature configuration file exists."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'feature_config.json'
        )

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                feature_count = len(config.get('features', []))

                self.add_result(
                    "Feature Configuration",
                    Status.PASS,
                    f"Config loaded with {feature_count} features",
                    {'path': config_path, 'feature_count': feature_count}
                )
                return True
            except Exception as e:
                self.add_result(
                    "Feature Configuration",
                    Status.FAIL,
                    f"Error parsing config: {str(e)}"
                )
                return False
        else:
            self.add_result(
                "Feature Configuration",
                Status.WARN,
                f"Config file not found at {config_path}"
            )
            return False

    def check_inference_view(self) -> bool:
        """Check if inference features view exists."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Check for materialized view
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_matviews
                    WHERE matviewname = 'inference_features_5m'
                )
            """)

            exists = cur.fetchone()[0]

            if exists:
                # Get row count
                cur.execute("SELECT COUNT(*) FROM inference_features_5m")
                count = cur.fetchone()[0]

                self.add_result(
                    "Inference Features View",
                    Status.PASS,
                    f"Materialized view exists with {count:,} rows"
                )
                cur.close()
                conn.close()
                return True
            else:
                self.add_result(
                    "Inference Features View",
                    Status.WARN,
                    "Materialized view not created - features calculated on-the-fly"
                )
                cur.close()
                conn.close()
                return False

        except Exception as e:
            self.add_result(
                "Inference Features View",
                Status.WARN,
                f"Could not check view: {str(e)}"
            )
            return False

    # =========================================================================
    # MAIN VALIDATION
    # =========================================================================

    def run_all_validations(self) -> ValidationReport:
        """Run all validation checks and generate report."""
        print("\n" + "=" * 60)
        print("Multi-Model Backend Validation")
        print("=" * 60 + "\n")

        print("1. DATABASE CHECKS")
        print("-" * 40)
        db_connected = self.check_database_connection()

        if db_connected:
            self.check_core_tables()
            self.check_dw_schema_tables()
            self.check_ohlcv_data()
            self.check_strategy_configuration()
            self.check_macro_indicators()
            self.check_inference_view()

        print("\n2. REDIS CHECKS")
        print("-" * 40)
        redis_connected = self.check_redis_connection()

        if redis_connected:
            self.check_redis_streams()

        print("\n3. API CHECKS")
        print("-" * 40)
        self.check_all_apis()

        print("\n4. MINIO CHECKS")
        print("-" * 40)
        minio_connected = self.check_minio_connection()

        if minio_connected:
            self.check_model_files()

        print("\n5. FEATURE CHECKS")
        print("-" * 40)
        self.check_feature_configuration()

        # Generate report
        passed = sum(1 for r in self.results if r.status == Status.PASS)
        failed = sum(1 for r in self.results if r.status == Status.FAIL)
        warnings = sum(1 for r in self.results if r.status == Status.WARN)
        skipped = sum(1 for r in self.results if r.status == Status.SKIP)

        if failed > 0:
            overall = "FAIL"
        elif warnings > 0:
            overall = "WARN"
        else:
            overall = "PASS"

        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_checks=len(self.results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            results=[asdict(r) for r in self.results],
            overall_status=overall
        )

        return report

    def print_summary(self, report: ValidationReport):
        """Print summary of validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  Total Checks:  {report.total_checks}")
        print(f"  Passed:        {report.passed}")
        print(f"  Failed:        {report.failed}")
        print(f"  Warnings:      {report.warnings}")
        print(f"  Skipped:       {report.skipped}")
        print("-" * 60)
        print(f"  Overall Status: {report.overall_status}")
        print("=" * 60 + "\n")

        if report.failed > 0:
            print("FAILED CHECKS:")
            for r in self.results:
                if r.status == Status.FAIL:
                    print(f"  - {r.name}: {r.message}")
            print()

        if report.warnings > 0:
            print("WARNINGS:")
            for r in self.results:
                if r.status == Status.WARN:
                    print(f"  - {r.name}: {r.message}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Multi-Model Backend Setup"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    validator = BackendValidator(verbose=args.verbose)
    report = validator.run_all_validations()

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        validator.print_summary(report)

    # Exit with appropriate code
    if report.overall_status == "FAIL":
        sys.exit(1)
    elif report.overall_status == "WARN":
        sys.exit(0)  # Warnings don't fail CI
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
