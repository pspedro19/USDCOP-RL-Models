#!/usr/bin/env python
"""
Promote Experiment CLI
======================

Command-line tool for promoting models from experiment zone to production.

Usage:
    # Promote latest model version
    python scripts/promote_experiment.py --experiment-id baseline_v2

    # Promote specific version
    python scripts/promote_experiment.py --experiment-id baseline_v2 --version 20260118_123456

    # Dry run (validate only)
    python scripts/promote_experiment.py --experiment-id baseline_v2 --dry-run

    # Skip validation (use with caution)
    python scripts/promote_experiment.py --experiment-id baseline_v2 --skip-validation

    # Custom validation thresholds
    python scripts/promote_experiment.py --experiment-id baseline_v2 --min-sharpe 0.5 --max-drawdown 0.3

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_workflow.experiment_manager import ExperimentManager
from src.ml_workflow.promotion_service import PromotionService, PromotionStatus
from src.ml_workflow.promotion_gate import (
    PromotionGate,
    ValidationSeverity,
    DEFAULT_GATE_CONFIG,
)
from src.core.factories.storage_factory import StorageFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Promote ML models from experiment zone to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Promote latest model
  %(prog)s --experiment-id baseline_v2

  # Promote specific version
  %(prog)s --experiment-id baseline_v2 --version 20260118_123456

  # Dry run validation
  %(prog)s --experiment-id baseline_v2 --dry-run

  # Custom model ID
  %(prog)s --experiment-id baseline_v2 --model-id production_v3
        """,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-id", "-e",
        required=True,
        help="Experiment identifier (e.g., baseline_v2)",
    )

    # Optional arguments
    parser.add_argument(
        "--version", "-v",
        default=None,
        help="Model version to promote (latest if not specified)",
    )

    parser.add_argument(
        "--model-id", "-m",
        default=None,
        help="Custom production model ID (auto-generated if not specified)",
    )

    # Validation options
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Validate only, don't actually promote",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks (use with caution)",
    )

    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=DEFAULT_GATE_CONFIG["min_sharpe"],
        help=f"Minimum Sharpe ratio (default: {DEFAULT_GATE_CONFIG['min_sharpe']})",
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=DEFAULT_GATE_CONFIG["max_drawdown_limit"],
        help=f"Maximum drawdown (default: {DEFAULT_GATE_CONFIG['max_drawdown_limit']})",
    )

    parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_GATE_CONFIG["min_trades"],
        help=f"Minimum trades for significance (default: {DEFAULT_GATE_CONFIG['min_trades']})",
    )

    parser.add_argument(
        "--require-backtest",
        action="store_true",
        help="Require backtest results",
    )

    # Output options
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON",
    )

    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def list_available_models(experiment_id: str) -> None:
    """List available model versions for an experiment."""
    manager = ExperimentManager(experiment_id)

    try:
        versions = manager.list_model_versions()

        if not versions:
            logger.info(f"No models found for experiment: {experiment_id}")
            return

        print(f"\nAvailable models for {experiment_id}:")
        print("-" * 80)
        print(f"{'Version':<25} {'Hash':<18} {'Created':<20} {'Sharpe':<10}")
        print("-" * 80)

        for v in versions[:10]:  # Show last 10
            sharpe = f"{v.test_sharpe:.2f}" if v.test_sharpe else "N/A"
            created = v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "N/A"
            print(f"{v.version:<25} {v.model_hash[:16]:<18} {created:<20} {sharpe:<10}")

        if len(versions) > 10:
            print(f"... and {len(versions) - 10} more")

        print("-" * 80)

    except Exception as e:
        logger.error(f"Failed to list models: {e}")


def run_promotion(args) -> int:
    """Run the promotion workflow."""
    experiment_id = args.experiment_id
    version = args.version

    # Build validation config
    validation_config = {
        "min_sharpe": args.min_sharpe,
        "max_drawdown_limit": args.max_drawdown,
        "min_trades": args.min_trades,
        "require_backtest": args.require_backtest,
        "expected_observation_dim": 15,
        "expected_action_space": 3,
    }

    logger.info(f"Starting promotion for {experiment_id}")
    if version:
        logger.info(f"Version: {version}")
    else:
        logger.info("Version: latest")

    if args.dry_run:
        logger.info("DRY RUN - will validate only")

    if args.skip_validation:
        logger.warning("VALIDATION SKIPPED - use with caution!")

    # Get model version if not specified
    if version is None:
        manager = ExperimentManager(experiment_id)
        versions = manager.list_model_versions()

        if not versions:
            logger.error(f"No models found for experiment: {experiment_id}")
            list_available_models(experiment_id)
            return 1

        version = versions[0].version
        logger.info(f"Using latest version: {version}")

    # Create promotion service
    service = PromotionService()

    # Run promotion
    result = service.promote(
        experiment_id=experiment_id,
        model_version=version,
        model_id=args.model_id,
        validation_config=validation_config,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
    )

    # Output result
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print_result(result, verbose=args.verbose)

    # Return exit code
    if result.status == PromotionStatus.COMPLETED:
        return 0
    else:
        return 1


def print_result(result, verbose: bool = False) -> None:
    """Print promotion result in human-readable format."""
    print("\n" + "=" * 80)
    print("PROMOTION RESULT")
    print("=" * 80)

    status_emoji = {
        PromotionStatus.COMPLETED: "SUCCESS",
        PromotionStatus.FAILED: "FAILED",
        PromotionStatus.ROLLED_BACK: "ROLLED BACK",
    }.get(result.status, result.status.value.upper())

    print(f"Status: {status_emoji}")
    print(f"Experiment: {result.experiment_id}")
    print(f"Version: {result.model_version}")

    if result.model_id:
        print(f"Model ID: {result.model_id}")

    if result.production_uri:
        print(f"Production URI: {result.production_uri}")

    if result.promoted_at:
        print(f"Promoted at: {result.promoted_at.isoformat()}")

    # Validation details
    print("\nValidation:")
    print(f"  Passed: {'Yes' if result.validation_passed else 'No'}")

    if result.validation_errors:
        print(f"  Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors:
            print(f"    - {error}")

    if result.error_message:
        print(f"\nError: {result.error_message}")

    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        return run_promotion(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Promotion failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
