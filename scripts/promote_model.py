#!/usr/bin/env python
"""
Script para promover modelos con validaciones obligatorias.

This script handles model promotion between MLflow stages with mandatory validations:
- Smoke test: Verify model loads and works correctly
- Dataset hash: Verify dataset_hash is logged in MLflow params
- Staging time: For Production, check minimum days in Staging

Usage:
    # Promote model to Staging
    python scripts/promote_model.py usdcop_ppo 1 Staging --reason "Initial staging"

    # Promote to Production (requires min_staging_days)
    python scripts/promote_model.py usdcop_ppo 1 Production --reason "Ready for production"

    # Skip validations (use with caution)
    python scripts/promote_model.py usdcop_ppo 1 Staging --reason "Emergency" --skip-smoke-test

Author: Trading Team
Date: 2026-01-17
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.models.validation.smoke_test import run_smoke_test
from src.config.trading_flags import get_trading_flags


def get_staging_days(client: MlflowClient, model_name: str, version: int) -> int:
    """
    Calculate days model has been in Staging stage.

    Uses the last_updated_timestamp from the model version, which gets updated
    when the stage changes. This gives us the time since the model entered its
    current stage.

    Args:
        client: MLflow client instance
        model_name: Name of the registered model
        version: Version number of the model

    Returns:
        Number of days (integer) the model has been in Staging
    """
    mv = client.get_model_version(model_name, str(version))

    # Check version history for when it entered Staging
    # For simplicity, use last_updated_timestamp
    staging_start = mv.last_updated_timestamp / 1000  # Convert ms to s
    now = datetime.now().timestamp()
    days = (now - staging_start) / 86400
    return int(days)


def promote_model(
    model_name: str,
    version: int,
    to_stage: str,
    reason: str,
    skip_smoke_test: bool = False,
    skip_dataset_hash: bool = False,
    skip_staging_time: bool = False,
) -> Dict[str, Any]:
    """
    Promote model with mandatory validations.

    This function performs up to three validations before promoting a model:
    1. Smoke test: Verify model loads and produces valid outputs
    2. Dataset hash: Verify dataset_hash is logged in MLflow params
    3. Staging time: For Production, check minimum days in Staging

    Args:
        model_name: Name of the model in MLflow registry
        version: Version number to promote
        to_stage: Target stage (Staging, Production, or Archived)
        reason: Reason for the promotion (required for audit trail)
        skip_smoke_test: Skip smoke test validation
        skip_dataset_hash: Skip dataset_hash validation
        skip_staging_time: Skip staging time validation

    Returns:
        dict with promotion result:
            - success: bool indicating if promotion succeeded
            - stage: stage where failure occurred (if failed)
            - errors: list of error messages (if failed)
            - model_name, version, stage, reason (if succeeded)
    """
    flags = get_trading_flags()
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{version}"

    print(f"\n{'='*60}")
    print(f"PROMOTING MODEL: {model_name} v{version} -> {to_stage}")
    print(f"{'='*60}\n")

    # =========================================================================
    # VALIDATION 1: Smoke Test
    # =========================================================================
    if flags.require_smoke_test and not skip_smoke_test:
        print("[VALIDATION 1] Running smoke test...")
        result = run_smoke_test(model_uri)

        if not result.passed:
            print(f"\n[FAILED] Smoke test FAILED ({result.duration_ms:.1f}ms):")
            for error in result.errors:
                print(f"   - {error}")
            return {
                "success": False,
                "stage": "smoke_test",
                "errors": result.errors,
            }

        print(f"[PASSED] Smoke test PASSED ({len(result.checks)} checks, {result.duration_ms:.1f}ms)")
        for check in result.checks:
            status = "[OK]" if check.passed else "[FAIL]"
            print(f"   {status} {check.name}: {check.message}")
    else:
        print("[VALIDATION 1] Smoke test SKIPPED")

    # =========================================================================
    # VALIDATION 2: Dataset Hash
    # =========================================================================
    if flags.require_dataset_hash and not skip_dataset_hash:
        print("\n[VALIDATION 2] Validating dataset_hash...")
        mv = client.get_model_version(model_name, str(version))
        run = client.get_run(mv.run_id)

        if "dataset_hash" not in run.data.params:
            print("[FAILED] Model missing dataset_hash - promotion blocked")
            return {
                "success": False,
                "stage": "dataset_hash",
                "errors": ["dataset_hash not logged in MLflow"],
            }

        dataset_hash = run.data.params["dataset_hash"]
        print(f"[PASSED] Dataset hash: {dataset_hash[:16]}...")
    else:
        print("\n[VALIDATION 2] Dataset hash validation SKIPPED")

    # =========================================================================
    # VALIDATION 3: Staging Time (only for Production)
    # =========================================================================
    if to_stage == "Production" and not skip_staging_time:
        print("\n[VALIDATION 3] Checking staging time...")
        staging_days = get_staging_days(client, model_name, version)

        if staging_days < flags.min_staging_days:
            print(f"[FAILED] Model has been in Staging for {staging_days} days.")
            print(f"   Minimum required: {flags.min_staging_days} days")
            return {
                "success": False,
                "stage": "staging_time",
                "errors": [f"Only {staging_days} days in staging, need {flags.min_staging_days}"],
            }

        print(f"[PASSED] Staging time: {staging_days} days (min: {flags.min_staging_days})")
    elif to_stage == "Production":
        print("\n[VALIDATION 3] Staging time check SKIPPED")

    # =========================================================================
    # EXECUTE PROMOTION
    # =========================================================================
    print(f"\n[PROMOTING] {model_name} v{version} to {to_stage}...")

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=to_stage,
        archive_existing_versions=(to_stage == "Production"),
    )

    # Log promotion event to MLflow
    with mlflow.start_run(run_name=f"promotion_{model_name}_v{version}"):
        mlflow.log_params({
            "model_name": model_name,
            "model_version": version,
            "to_stage": to_stage,
            "reason": reason,
            "promoted_by": "promote_model.py",
            "promoted_at": datetime.now().isoformat(),
        })

    print(f"\n{'='*60}")
    print(f"[SUCCESS] {model_name} v{version} -> {to_stage}")
    print(f"{'='*60}\n")

    return {
        "success": True,
        "model_name": model_name,
        "version": version,
        "stage": to_stage,
        "reason": reason,
    }


def main():
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(
        description="Promote ML model with mandatory validations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Promote to Staging
  python scripts/promote_model.py usdcop_ppo 1 Staging --reason "Initial staging"

  # Promote to Production (requires min_staging_days)
  python scripts/promote_model.py usdcop_ppo 1 Production --reason "Ready for prod"

  # Archive a model
  python scripts/promote_model.py usdcop_ppo 1 Archived --reason "Superseded by v2"

  # Skip validations (use with caution)
  python scripts/promote_model.py usdcop_ppo 1 Staging --reason "Emergency" --skip-smoke-test
"""
    )

    parser.add_argument(
        "model_name",
        help="Name of the model in MLflow registry"
    )
    parser.add_argument(
        "version",
        type=int,
        help="Version number to promote"
    )
    parser.add_argument(
        "stage",
        choices=["Staging", "Production", "Archived"],
        help="Target stage for the model"
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Reason for promotion (required for audit trail)"
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip smoke test validation (use with caution)"
    )
    parser.add_argument(
        "--skip-dataset-hash",
        action="store_true",
        help="Skip dataset_hash validation (use with caution)"
    )
    parser.add_argument(
        "--skip-staging-time",
        action="store_true",
        help="Skip staging time validation for Production (use with caution)"
    )

    args = parser.parse_args()

    result = promote_model(
        model_name=args.model_name,
        version=args.version,
        to_stage=args.stage,
        reason=args.reason,
        skip_smoke_test=args.skip_smoke_test,
        skip_dataset_hash=args.skip_dataset_hash,
        skip_staging_time=args.skip_staging_time,
    )

    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
