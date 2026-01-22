#!/usr/bin/env python3
"""
Experiment Launch Readiness Validator

Validates all prerequisites for launching an experiment:
1. Infrastructure (PostgreSQL, Redis, MinIO)
2. Database tables and data
3. Contracts (Features, XCom, Rewards)
4. Experiment configuration
5. Dataset availability

Usage:
    python scripts/validate_experiment_readiness.py
    python scripts/validate_experiment_readiness.py --experiment exp1_curriculum_aggressive_v1
    python scripts/validate_experiment_readiness.py --verbose
    python scripts/validate_experiment_readiness.py --fix  # Attempt auto-fixes

Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

import os
import sys
import argparse
import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None
    severity: str = "error"  # error, warning, info
    fix_available: bool = False
    fix_command: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    experiment_name: str
    timestamp: str
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed or r.severity != "error" for r in self.results)

    @property
    def errors(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]


class ExperimentReadinessValidator:
    """Validates system readiness for experiment launch."""

    def __init__(self, experiment_name: str = "exp1_curriculum_aggressive_v1", verbose: bool = False):
        self.experiment_name = experiment_name
        self.verbose = verbose
        self.project_root = PROJECT_ROOT
        self.results: List[ValidationResult] = []

    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"  → {message}")

    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)

        # Print status
        status = "✓" if result.passed else ("⚠" if result.severity == "warning" else "✗")
        color = "\033[92m" if result.passed else ("\033[93m" if result.severity == "warning" else "\033[91m")
        reset = "\033[0m"
        print(f"{color}{status}{reset} {result.name}: {result.message}")

        if result.details and self.verbose:
            print(f"    Details: {result.details}")

        if not result.passed and result.fix_command:
            print(f"    Fix: {result.fix_command}")

    # =========================================================================
    # PHASE 1: Environment Validation
    # =========================================================================

    def validate_env_file(self) -> ValidationResult:
        """Check .env file exists and has required variables."""
        env_file = self.project_root / ".env"

        if not env_file.exists():
            return ValidationResult(
                name="Environment File",
                passed=False,
                message=".env file not found",
                fix_available=True,
                fix_command="cp .env.example .env && edit .env"
            )

        required_vars = [
            "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
            "REDIS_PASSWORD",
            "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"
        ]

        with open(env_file) as f:
            content = f.read()

        missing = [var for var in required_vars if f"{var}=" not in content]

        if missing:
            return ValidationResult(
                name="Environment Variables",
                passed=False,
                message=f"Missing variables: {', '.join(missing)}",
                details="Add these to .env file"
            )

        return ValidationResult(
            name="Environment File",
            passed=True,
            message="All required variables present"
        )

    # =========================================================================
    # PHASE 2: Database Validation
    # =========================================================================

    def validate_database_connection(self) -> ValidationResult:
        """Check PostgreSQL connection."""
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "usdcop_trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            conn.close()

            return ValidationResult(
                name="PostgreSQL Connection",
                passed=True,
                message="Connected successfully"
            )
        except ImportError:
            return ValidationResult(
                name="PostgreSQL Connection",
                passed=False,
                message="psycopg2 not installed",
                fix_command="pip install psycopg2-binary"
            )
        except Exception as e:
            return ValidationResult(
                name="PostgreSQL Connection",
                passed=False,
                message=f"Connection failed: {str(e)[:50]}",
                fix_command="docker-compose up -d postgres"
            )

    def validate_required_tables(self) -> ValidationResult:
        """Check required tables exist."""
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "usdcop_trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )

            required_tables = [
                "usdcop_m5_ohlcv",
                "macro_indicators_daily",
                "model_registry"
            ]

            cur = conn.cursor()
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            existing = {row[0] for row in cur.fetchall()}
            cur.close()
            conn.close()

            missing = [t for t in required_tables if t not in existing]

            if missing:
                return ValidationResult(
                    name="Required Tables",
                    passed=False,
                    message=f"Missing: {', '.join(missing)}",
                    fix_command="Run init-scripts in PostgreSQL"
                )

            return ValidationResult(
                name="Required Tables",
                passed=True,
                message=f"All {len(required_tables)} required tables exist"
            )
        except Exception as e:
            return ValidationResult(
                name="Required Tables",
                passed=False,
                message=f"Check failed: {str(e)[:50]}"
            )

    def validate_ohlcv_data(self) -> ValidationResult:
        """Check OHLCV data availability."""
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "usdcop_trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )

            cur = conn.cursor()
            cur.execute("""
                SELECT
                    MIN(time)::DATE as first_bar,
                    MAX(time)::DATE as last_bar,
                    COUNT(*) as total_bars
                FROM usdcop_m5_ohlcv
            """)
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result[2] == 0:
                return ValidationResult(
                    name="OHLCV Data",
                    passed=False,
                    message="No OHLCV data found",
                    fix_command="Trigger v3.l0_ohlcv_backfill DAG"
                )

            first_bar, last_bar, total = result

            # Check date range covers experiment_training
            required_start = datetime(2023, 1, 1).date()
            required_end = datetime(2024, 12, 31).date()

            if first_bar > required_start or last_bar < required_end:
                return ValidationResult(
                    name="OHLCV Data",
                    passed=False,
                    message=f"Insufficient range: {first_bar} to {last_bar}",
                    details=f"Need: {required_start} to {required_end}",
                    fix_command="Trigger v3.l0_ohlcv_backfill DAG"
                )

            return ValidationResult(
                name="OHLCV Data",
                passed=True,
                message=f"{total:,} bars from {first_bar} to {last_bar}"
            )
        except Exception as e:
            return ValidationResult(
                name="OHLCV Data",
                passed=False,
                message=f"Check failed: {str(e)[:50]}"
            )

    def validate_macro_data(self) -> ValidationResult:
        """Check macro data availability."""
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "usdcop_trading"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )

            cur = conn.cursor()
            cur.execute("""
                SELECT
                    MIN(fecha) as first_date,
                    MAX(fecha) as last_date,
                    COUNT(*) as total_days,
                    COUNT(fxrt_index_dxy_usa_d_dxy) as dxy_count,
                    COUNT(volt_vix_usa_d_vix) as vix_count
                FROM macro_indicators_daily
            """)
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result[2] == 0:
                return ValidationResult(
                    name="Macro Data",
                    passed=False,
                    message="No macro data found",
                    fix_command="Trigger v3.l0_macro_unified DAG"
                )

            first_date, last_date, total, dxy_count, vix_count = result

            # Check critical variables
            fill_rate = min(dxy_count, vix_count) / total if total > 0 else 0

            if fill_rate < 0.9:
                return ValidationResult(
                    name="Macro Data",
                    passed=False,
                    message=f"Low fill rate: {fill_rate:.1%} for DXY/VIX",
                    severity="warning",
                    fix_command="Trigger v3.l0_macro_unified DAG"
                )

            return ValidationResult(
                name="Macro Data",
                passed=True,
                message=f"{total:,} days from {first_date} to {last_date}"
            )
        except Exception as e:
            return ValidationResult(
                name="Macro Data",
                passed=False,
                message=f"Check failed: {str(e)[:50]}"
            )

    # =========================================================================
    # PHASE 3: Contract Validation
    # =========================================================================

    def validate_feature_contract(self) -> ValidationResult:
        """Check feature contract is valid."""
        try:
            from src.core.contracts.feature_contract import (
                FEATURE_ORDER, OBSERVATION_DIM, FEATURE_ORDER_HASH
            )

            if len(FEATURE_ORDER) != 15:
                return ValidationResult(
                    name="Feature Contract",
                    passed=False,
                    message=f"Expected 15 features, got {len(FEATURE_ORDER)}"
                )

            if OBSERVATION_DIM != 15:
                return ValidationResult(
                    name="Feature Contract",
                    passed=False,
                    message=f"OBSERVATION_DIM should be 15, got {OBSERVATION_DIM}"
                )

            return ValidationResult(
                name="Feature Contract",
                passed=True,
                message=f"15 features, hash={FEATURE_ORDER_HASH[:8]}..."
            )
        except ImportError as e:
            return ValidationResult(
                name="Feature Contract",
                passed=False,
                message=f"Import error: {e}"
            )

    def validate_xcom_contracts(self) -> ValidationResult:
        """Check XCom contracts are complete."""
        try:
            from airflow.dags.contracts.xcom_contracts import (
                L0XComKeysEnum, L1XComKeysEnum, L2XComKeysEnum,
                L3XComKeysEnum, L5XComKeysEnum, L6XComKeysEnum,
                L0Output, L1Output, L2Output, L3Output, L5Output, L6Output
            )

            # Check all enums have members
            enums = {
                "L0": len(L0XComKeysEnum.__members__),
                "L1": len(L1XComKeysEnum.__members__),
                "L2": len(L2XComKeysEnum.__members__),
                "L3": len(L3XComKeysEnum.__members__),
                "L5": len(L5XComKeysEnum.__members__),
                "L6": len(L6XComKeysEnum.__members__),
            }

            empty = [k for k, v in enums.items() if v == 0]
            if empty:
                return ValidationResult(
                    name="XCom Contracts",
                    passed=False,
                    message=f"Empty enums: {empty}"
                )

            return ValidationResult(
                name="XCom Contracts",
                passed=True,
                message=f"L0-L6 contracts valid ({sum(enums.values())} keys)"
            )
        except ImportError as e:
            return ValidationResult(
                name="XCom Contracts",
                passed=False,
                message=f"Import error: {e}"
            )

    def validate_canonical_builder(self) -> ValidationResult:
        """Check CanonicalFeatureBuilder is complete."""
        try:
            from src.feature_store.builders.canonical_feature_builder import CanonicalFeatureBuilder

            builder = CanonicalFeatureBuilder()

            # Check required attributes/methods
            required = ["VERSION", "FEATURE_ORDER", "compute_features", "validate_features"]
            missing = [r for r in required if not hasattr(builder, r)]

            if missing:
                return ValidationResult(
                    name="CanonicalFeatureBuilder",
                    passed=False,
                    message=f"Missing: {missing}"
                )

            return ValidationResult(
                name="CanonicalFeatureBuilder",
                passed=True,
                message=f"VERSION={builder.VERSION}, {len(builder.FEATURE_ORDER)} features"
            )
        except ImportError as e:
            return ValidationResult(
                name="CanonicalFeatureBuilder",
                passed=False,
                message=f"Import error: {e}"
            )

    def validate_reward_components(self) -> ValidationResult:
        """Check reward components are importable."""
        try:
            from src.training.reward_components import (
                DifferentialSharpeRatio,
                SortinoCalculator,
                StableRegimeDetector,
                OilCorrelationTracker,
                BanrepInterventionDetector,
                HoldingDecay,
                InactivityTracker,
                ChurnTracker,
                BiasDetector,
            )

            return ValidationResult(
                name="Reward Components",
                passed=True,
                message="All 9 components importable"
            )
        except ImportError as e:
            return ValidationResult(
                name="Reward Components",
                passed=False,
                message=f"Import error: {e}"
            )

    # =========================================================================
    # PHASE 4: Experiment Config Validation
    # =========================================================================

    def validate_experiment_config(self) -> ValidationResult:
        """Check experiment YAML is valid."""
        config_path = self.project_root / "config" / "experiments" / f"{self.experiment_name}.yaml"

        if not config_path.exists():
            return ValidationResult(
                name="Experiment Config",
                passed=False,
                message=f"File not found: {config_path.name}"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required sections
            required = ["experiment", "model", "training", "reward", "data"]
            missing = [s for s in required if s not in config]

            if missing:
                return ValidationResult(
                    name="Experiment Config",
                    passed=False,
                    message=f"Missing sections: {missing}"
                )

            # Check reward weights sum to 1
            weights = config.get("reward", {}).get("weights", {})
            total = sum(weights.values())

            if abs(total - 1.0) > 0.01:
                return ValidationResult(
                    name="Experiment Config",
                    passed=False,
                    message=f"Reward weights sum to {total}, should be 1.0"
                )

            return ValidationResult(
                name="Experiment Config",
                passed=True,
                message=f"Valid config: {config['experiment']['name']}"
            )
        except yaml.YAMLError as e:
            return ValidationResult(
                name="Experiment Config",
                passed=False,
                message=f"YAML parse error: {e}"
            )

    def validate_date_ranges(self) -> ValidationResult:
        """Check date_ranges.yaml is valid."""
        config_path = self.project_root / "config" / "date_ranges.yaml"

        if not config_path.exists():
            return ValidationResult(
                name="Date Ranges",
                passed=False,
                message="config/date_ranges.yaml not found"
            )

        try:
            with open(config_path) as f:
                dates = yaml.safe_load(f)

            # Check required sections
            required = ["experiment_training", "validation", "test"]
            missing = [s for s in required if s not in dates]

            if missing:
                return ValidationResult(
                    name="Date Ranges",
                    passed=False,
                    message=f"Missing sections: {missing}"
                )

            exp_start = dates["experiment_training"]["start"]
            exp_end = dates["experiment_training"]["end"]

            return ValidationResult(
                name="Date Ranges",
                passed=True,
                message=f"experiment_training: {exp_start} to {exp_end}"
            )
        except Exception as e:
            return ValidationResult(
                name="Date Ranges",
                passed=False,
                message=f"Error: {e}"
            )

    # =========================================================================
    # Main Validation
    # =========================================================================

    def run_all_validations(self) -> ValidationReport:
        """Run all validation checks."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT READINESS VALIDATION")
        print(f"Experiment: {self.experiment_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Phase 1: Environment
        print("PHASE 1: Environment\n" + "-"*40)
        self.add_result(self.validate_env_file())

        # Phase 2: Database
        print("\nPHASE 2: Database\n" + "-"*40)
        db_result = self.validate_database_connection()
        self.add_result(db_result)

        if db_result.passed:
            self.add_result(self.validate_required_tables())
            self.add_result(self.validate_ohlcv_data())
            self.add_result(self.validate_macro_data())

        # Phase 3: Contracts
        print("\nPHASE 3: Contracts\n" + "-"*40)
        self.add_result(self.validate_feature_contract())
        self.add_result(self.validate_xcom_contracts())
        self.add_result(self.validate_canonical_builder())
        self.add_result(self.validate_reward_components())

        # Phase 4: Experiment Config
        print("\nPHASE 4: Experiment Config\n" + "-"*40)
        self.add_result(self.validate_experiment_config())
        self.add_result(self.validate_date_ranges())

        # Create report
        report = ValidationReport(
            experiment_name=self.experiment_name,
            timestamp=datetime.now().isoformat(),
            results=self.results
        )

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for r in self.results if r.passed)
        warnings = len(report.warnings)
        errors = len(report.errors)

        print(f"Total checks: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Warnings: {warnings}")
        print(f"Errors: {errors}")

        if report.passed:
            print(f"\n\033[92m✓ READY TO LAUNCH EXPERIMENT\033[0m")
            print(f"\nNext step:")
            print(f"  python scripts/run_experiment.py --config config/experiments/{self.experiment_name}.yaml")
        else:
            print(f"\n\033[91m✗ NOT READY - Fix {errors} error(s) first\033[0m")
            print(f"\nErrors to fix:")
            for err in report.errors:
                print(f"  - {err.name}: {err.message}")
                if err.fix_command:
                    print(f"    Fix: {err.fix_command}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Validate experiment readiness")
    parser.add_argument(
        "--experiment", "-e",
        default="exp1_curriculum_aggressive_v1",
        help="Experiment name (default: exp1_curriculum_aggressive_v1)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    validator = ExperimentReadinessValidator(
        experiment_name=args.experiment,
        verbose=args.verbose
    )

    report = validator.run_all_validations()

    # Exit code based on result
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
