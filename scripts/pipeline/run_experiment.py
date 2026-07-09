#!/usr/bin/env python3
"""
Run Experiment CLI
==================

Command-line tool for running experiments from YAML configuration files.

Usage:
    # Run experiment
    python scripts/run_experiment.py --config config/experiments/baseline_ppo_v1.yaml

    # Dry run (validate only)
    python scripts/run_experiment.py --config config/experiments/my_exp.yaml --dry-run

    # Run with custom output directory
    python scripts/run_experiment.py --config config/experiments/my_exp.yaml --output-dir models/my_exp

    # List available experiments
    python scripts/run_experiment.py --list

    # Validate configuration
    python scripts/run_experiment.py --config config/experiments/my_exp.yaml --validate

    # Create new experiment from template
    python scripts/run_experiment.py --new my_new_exp --template baseline_ppo_v1

Author: Trading Team
Date: 2026-01-17
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments import (
    load_experiment_config,
    validate_experiment_config,
    list_available_experiments,
    ExperimentRunner,
)
from src.experiments.experiment_loader import create_experiment_from_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run experiments from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run an experiment:
    python scripts/run_experiment.py --config config/experiments/baseline_ppo_v1.yaml

  Validate config only:
    python scripts/run_experiment.py --config config/experiments/my_exp.yaml --validate

  List available experiments:
    python scripts/run_experiment.py --list

  Create new experiment from template:
    python scripts/run_experiment.py --new my_new_exp --template baseline_ppo_v1
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration only",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--new",
        type=str,
        metavar="NAME",
        help="Create new experiment with given name",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="baseline_ppo_v1",
        help="Template experiment for --new (default: baseline_ppo_v1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List experiments
    if args.list:
        list_experiments()
        return 0

    # Create new experiment
    if args.new:
        create_new_experiment(args.new, args.template)
        return 0

    # Require config for other operations
    if not args.config:
        parser.error("--config is required for run/validate operations")
        return 1

    config_path = Path(args.config)

    # Validate only
    if args.validate:
        return validate_config(config_path)

    # Run experiment
    return run_experiment(
        config_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


def list_experiments() -> None:
    """List all available experiments."""
    print("\n" + "=" * 60)
    print("Available Experiments")
    print("=" * 60)

    experiments = list_available_experiments()

    if not experiments:
        print("No experiments found in config/experiments/")
        print("\nCreate one with:")
        print("  python scripts/run_experiment.py --new my_exp --template baseline_ppo_v1")
        return

    for exp in experiments:
        status = "[VALID]" if exp["valid"] else "[INVALID]"
        tags = ", ".join(exp.get("tags", [])) or "no tags"
        print(f"\n{exp['name']} v{exp['version']} {status}")
        print(f"  Path: {exp['path']}")
        print(f"  Tags: {tags}")
        if exp.get("description"):
            print(f"  Description: {exp['description'][:60]}...")
        if exp.get("errors"):
            for error in exp["errors"][:3]:
                print(f"  ERROR: {error}")

    print("\n" + "=" * 60)
    print(f"Total: {len(experiments)} experiments")
    valid = sum(1 for e in experiments if e["valid"])
    print(f"Valid: {valid}, Invalid: {len(experiments) - valid}")


def validate_config(config_path: Path) -> int:
    """Validate experiment configuration."""
    print(f"\nValidating: {config_path}")
    print("-" * 60)

    errors = validate_experiment_config(config_path)

    if not errors:
        print("Configuration is VALID")
        print()

        # Show summary
        config = load_experiment_config(config_path)
        print(f"Experiment: {config.experiment.name} v{config.experiment.version}")
        print(f"Algorithm: {config.model.algorithm}")
        print(f"Total Timesteps: {config.training.total_timesteps:,}")
        print(f"Primary Metric: {config.evaluation.primary_metric}")

        return 0
    else:
        print("Configuration is INVALID")
        print()
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1


def run_experiment(
    config_path: Path,
    output_dir: str = None,
    dry_run: bool = False,
) -> int:
    """Run experiment from configuration."""
    print("\n" + "=" * 60)
    print("USDCOP Experiment Runner")
    print("=" * 60)

    try:
        # Load configuration
        print(f"\nLoading configuration: {config_path}")
        config = load_experiment_config(config_path)

        print(f"Experiment: {config.experiment.name} v{config.experiment.version}")
        print(f"Algorithm: {config.model.algorithm}")
        print(f"Total Timesteps: {config.training.total_timesteps:,}")

        if dry_run:
            print("\n[DRY RUN MODE - No training will be performed]")

        # Create runner
        runner = ExperimentRunner(
            config,
            output_dir=Path(output_dir) if output_dir else None,
            dry_run=dry_run,
        )

        print(f"\nRun ID: {runner.run_id}")
        print(f"Output Directory: {runner.output_dir}")

        # Run experiment
        print("\n" + "-" * 60)
        print("Starting experiment...")
        print("-" * 60)

        result = runner.run()

        # Show results
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nStatus: {result.status}")
        print(f"Duration: {result.duration_seconds:.1f} seconds")

        if result.status == "success":
            print("\nBacktest Metrics:")
            for metric, value in result.backtest_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

            print(f"\nModel saved: {result.model_path}")
            print(f"Results saved: {runner.output_dir / 'result.json'}")

            if result.mlflow_run_id:
                print(f"\nMLflow Run ID: {result.mlflow_run_id}")

            return 0

        elif result.status == "dry_run":
            print("\nDry run completed - configuration is valid")
            return 0

        else:
            print(f"\nError: {result.error}")
            return 1

    except FileNotFoundError as e:
        print(f"\nError: Configuration file not found: {e}")
        return 1
    except ValueError as e:
        print(f"\nError: Invalid configuration: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Experiment failed")
        return 1


def create_new_experiment(name: str, template: str) -> None:
    """Create new experiment from template."""
    print(f"\nCreating new experiment: {name}")
    print(f"From template: {template}")

    try:
        output_path = create_experiment_from_template(name, template)
        print(f"\nCreated: {output_path}")
        print("\nEdit the configuration and run with:")
        print(f"  python scripts/run_experiment.py --config {output_path}")
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nAvailable templates:")
        for exp in list_available_experiments():
            if exp["valid"]:
                print(f"  - {exp['name']}")


if __name__ == "__main__":
    sys.exit(main())
