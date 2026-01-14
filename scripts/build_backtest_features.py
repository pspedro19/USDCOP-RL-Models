#!/usr/bin/env python3
"""
CLI Script: Build Backtest Features
Usage: python scripts/build_backtest_features.py --start 2025-01-01 --end 2025-12-31

Follows clean code principles:
- Single entry point
- Clear argument parsing
- Structured logging
- Fail fast on validation errors
"""

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.backtest_feature_builder import BacktestFeatureBuilder, FeatureBuildConfig
from src.validation.backtest_data_validator import BacktestDataValidator, DataValidationError


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_connection_string() -> str:
    """Get database connection string from environment."""
    # Try environment variables first
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DB", "usdcop")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build inference features for backtest period",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build features for 2025
  python scripts/build_backtest_features.py --start 2025-01-01 --end 2025-12-31

  # Dry run to see what would be built
  python scripts/build_backtest_features.py --start 2025-01-01 --end 2025-12-31 --dry-run

  # Skip validation (not recommended)
  python scripts/build_backtest_features.py --start 2025-01-01 --end 2025-12-31 --skip-validation
        """
    )

    parser.add_argument(
        "--start",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
        required=True,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be built without building"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation (not recommended)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("BACKTEST FEATURE BUILDER")
    logger.info("=" * 60)
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip validation: {args.skip_validation}")

    try:
        conn_string = get_connection_string()
        builder = BacktestFeatureBuilder(conn_string)

        result = builder.build_features(
            start_date=args.start,
            end_date=args.end,
            validate_first=not args.skip_validation,
            dry_run=args.dry_run
        )

        logger.info("=" * 60)
        logger.info("BUILD RESULT")
        logger.info("=" * 60)
        for key, value in result.items():
            logger.info(f"  {key}: {value}")

        return 0

    except DataValidationError as e:
        logger.error("=" * 60)
        logger.error("DATA VALIDATION FAILED")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("Recommendations:")
        for issue in e.result.blocking_issues:
            if issue.recommendation:
                logger.error(f"  - {issue.recommendation}")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
