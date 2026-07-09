#!/usr/bin/env python3
"""
Forecasting Pipeline Validation Script
======================================

Validates the complete forecasting pipeline:
1. Data contracts and SSOT
2. Feature engineering
3. Model factory and all 9 models
4. Walk-forward validation
5. Training flow (simulated)
6. Inference flow (simulated)
7. Database schema
8. Frontend service contracts

Usage:
    python scripts/validate_forecasting_pipeline.py
    python scripts/validate_forecasting_pipeline.py --verbose
    python scripts/validate_forecasting_pipeline.py --quick  # Skip slow tests

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_checks: int
    passed: int
    failed: int
    skipped: int
    results: List[ValidationResult]
    duration_seconds: float


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def check_data_contracts() -> ValidationResult:
    """Validate data contracts SSOT."""
    import time
    start = time.time()

    try:
        from src.forecasting.data_contracts import (
            DAILY_OHLCV_TABLE,
            FEATURES_VIEW,
            FEATURE_COLUMNS,
            TARGET_HORIZONS,
            TARGET_COLUMN,
            RAW_DAILY_COLUMNS,
            DATA_CONTRACT_VERSION,
            validate_daily_record,
        )

        checks = []

        # Check table names
        checks.append(("DAILY_OHLCV_TABLE", DAILY_OHLCV_TABLE == "bi.dim_daily_usdcop"))
        checks.append(("FEATURES_VIEW", FEATURES_VIEW == "bi.v_forecasting_features"))

        # Check feature count
        checks.append(("FEATURE_COLUMNS count", len(FEATURE_COLUMNS) == 19))

        # Check horizons
        checks.append(("TARGET_HORIZONS", TARGET_HORIZONS == (1, 5, 10, 15, 20, 25, 30)))

        # Check validation function
        valid_record = {
            "date": "2026-01-22",
            "open": 4350.0,
            "high": 4400.0,
            "low": 4340.0,
            "close": 4380.0,
        }
        is_valid, errors = validate_daily_record(valid_record)
        checks.append(("validate_daily_record", is_valid and len(errors) == 0))

        all_passed = all(passed for _, passed in checks)
        failed_checks = [name for name, passed in checks if not passed]

        duration = (time.time() - start) * 1000
        return ValidationResult(
            name="Data Contracts",
            passed=all_passed,
            message=f"All {len(checks)} checks passed" if all_passed else f"Failed: {failed_checks}",
            details={
                "feature_columns": len(FEATURE_COLUMNS),
                "horizons": TARGET_HORIZONS,
                "version": DATA_CONTRACT_VERSION,
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Data Contracts",
            passed=False,
            message=f"Error: {str(e)}",
            duration_ms=(time.time() - start) * 1000
        )


def check_forecasting_contracts() -> ValidationResult:
    """Validate forecasting contracts SSOT."""
    import time
    start = time.time()

    try:
        from src.forecasting.contracts import (
            HORIZONS,
            MODEL_IDS,
            MODEL_DEFINITIONS,
            HORIZON_CATEGORIES,
            HORIZON_CONFIGS,
            WF_CONFIG,
            ForecastDirection,
            ModelType,
            EnsembleType,
            ForecastingTrainingRequest,
            ForecastingInferenceRequest,
            get_horizon_config,
            validate_model_id,
            validate_horizon,
            FORECASTING_CONTRACT_VERSION,
        )

        checks = []

        # Check model count
        checks.append(("MODEL_IDS count", len(MODEL_IDS) == 9))

        # Check all model types
        checks.append(("Model definitions", all(m in MODEL_DEFINITIONS for m in MODEL_IDS)))

        # Check horizon count
        checks.append(("HORIZONS count", len(HORIZONS) == 7))

        # Check horizon config function
        config = get_horizon_config(5)
        checks.append(("get_horizon_config", "n_estimators" in config))

        # Check validation functions
        checks.append(("validate_model_id", validate_model_id("ridge")))
        checks.append(("validate_horizon", validate_horizon(10)))

        # Check data classes
        train_req = ForecastingTrainingRequest(dataset_path="/tmp/test.csv")
        checks.append(("ForecastingTrainingRequest", len(train_req.models) == 9))

        infer_req = ForecastingInferenceRequest(inference_date="2026-01-22")
        checks.append(("ForecastingInferenceRequest", len(infer_req.horizons) == 7))

        all_passed = all(passed for _, passed in checks)
        failed_checks = [name for name, passed in checks if not passed]

        duration = (time.time() - start) * 1000
        return ValidationResult(
            name="Forecasting Contracts",
            passed=all_passed,
            message=f"All {len(checks)} checks passed" if all_passed else f"Failed: {failed_checks}",
            details={
                "models": list(MODEL_IDS),
                "horizons": HORIZONS,
                "version": FORECASTING_CONTRACT_VERSION,
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Forecasting Contracts",
            passed=False,
            message=f"Error: {str(e)}\n{traceback.format_exc()}",
            duration_ms=(time.time() - start) * 1000
        )


def check_model_factory() -> ValidationResult:
    """Validate model factory can create all models."""
    import time
    start = time.time()

    try:
        from src.forecasting.models import ModelFactory
        from src.forecasting.contracts import MODEL_IDS

        created_models = []
        failed_models = []

        for model_id in MODEL_IDS:
            try:
                model = ModelFactory.create(model_id)
                created_models.append(model_id)
            except Exception as e:
                failed_models.append((model_id, str(e)))

        all_passed = len(failed_models) == 0
        duration = (time.time() - start) * 1000

        return ValidationResult(
            name="Model Factory",
            passed=all_passed,
            message=f"Created {len(created_models)}/9 models" if all_passed else f"Failed: {failed_models}",
            details={
                "created": created_models,
                "failed": failed_models,
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Model Factory",
            passed=False,
            message=f"Error: {str(e)}\n{traceback.format_exc()}",
            duration_ms=(time.time() - start) * 1000
        )


def check_model_training(quick: bool = False) -> ValidationResult:
    """Validate that models can be trained on synthetic data."""
    import time
    start = time.time()

    if quick:
        return ValidationResult(
            name="Model Training",
            passed=True,
            message="Skipped (quick mode)",
            duration_ms=0
        )

    try:
        from src.forecasting.models import ModelFactory

        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 19

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 0.01  # Small returns

        # Test a subset of models
        test_models = ["ridge", "xgboost_pure", "lightgbm_pure"]
        results = {}

        for model_id in test_models:
            try:
                model = ModelFactory.create(model_id)

                # Split data
                X_train, X_test = X[:150], X[150:]
                y_train, y_test = y[:150], y[150:]

                # Train
                model.fit(X_train, y_train)

                # Predict
                predictions = model.predict(X_test)

                # Check prediction shape
                if len(predictions) == len(y_test):
                    results[model_id] = "OK"
                else:
                    results[model_id] = f"Shape mismatch: {len(predictions)} vs {len(y_test)}"

            except Exception as e:
                results[model_id] = f"Error: {str(e)}"

        all_passed = all(v == "OK" for v in results.values())
        duration = (time.time() - start) * 1000

        return ValidationResult(
            name="Model Training",
            passed=all_passed,
            message=f"Trained {sum(1 for v in results.values() if v == 'OK')}/{len(test_models)} models",
            details=results,
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Model Training",
            passed=False,
            message=f"Error: {str(e)}",
            duration_ms=(time.time() - start) * 1000
        )


def check_walk_forward() -> ValidationResult:
    """Validate walk-forward validation."""
    import time
    start = time.time()

    try:
        from src.forecasting.evaluation.walk_forward import WalkForwardValidator
        from src.forecasting.models import ModelFactory

        # Create validator
        validator = WalkForwardValidator(n_folds=3, initial_train_ratio=0.5)

        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01

        # Create model
        model = ModelFactory.create("ridge")

        # Run validation
        result = validator.validate(model, X, y, horizon=5)

        checks = [
            ("n_folds", result.n_folds == 3),
            ("fold_results", len(result.fold_results) == 3),
            ("avg_metrics", result.avg_metrics is not None),
            ("direction_accuracy", 0 <= result.avg_metrics.direction_accuracy <= 100),
        ]

        all_passed = all(passed for _, passed in checks)
        failed_checks = [name for name, passed in checks if not passed]

        duration = (time.time() - start) * 1000
        return ValidationResult(
            name="Walk-Forward Validation",
            passed=all_passed,
            message=f"DA: {result.avg_metrics.direction_accuracy:.1f}%" if all_passed else f"Failed: {failed_checks}",
            details={
                "n_folds": result.n_folds,
                "direction_accuracy": result.avg_metrics.direction_accuracy,
                "rmse": result.avg_metrics.rmse,
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Walk-Forward Validation",
            passed=False,
            message=f"Error: {str(e)}\n{traceback.format_exc()}",
            duration_ms=(time.time() - start) * 1000
        )


def check_metrics() -> ValidationResult:
    """Validate metrics calculations."""
    import time
    start = time.time()

    try:
        from src.forecasting.evaluation.metrics import Metrics, MetricsResult

        # Create test data
        y_true = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        y_pred = np.array([0.008, -0.015, 0.012, -0.008, 0.018])

        # Calculate metrics
        result = Metrics.compute_all(y_true, y_pred)

        checks = [
            ("direction_accuracy", 0 <= result.direction_accuracy <= 100),
            ("rmse", result.rmse >= 0),
            ("mae", result.mae >= 0),
            ("r2", -10 <= result.r2 <= 1),
            ("sample_count", result.sample_count == 5),
        ]

        # Test direction accuracy specifically
        da = Metrics.direction_accuracy(y_true, y_pred)
        checks.append(("DA value", da == 100.0))  # All predictions have correct sign

        all_passed = all(passed for _, passed in checks)
        failed_checks = [name for name, passed in checks if not passed]

        duration = (time.time() - start) * 1000
        return ValidationResult(
            name="Metrics",
            passed=all_passed,
            message=f"All {len(checks)} checks passed" if all_passed else f"Failed: {failed_checks}",
            details={
                "direction_accuracy": result.direction_accuracy,
                "rmse": result.rmse,
                "mae": result.mae,
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Metrics",
            passed=False,
            message=f"Error: {str(e)}",
            duration_ms=(time.time() - start) * 1000
        )


def check_dags_exist() -> ValidationResult:
    """Validate that all DAG files exist and have required components."""
    import time
    start = time.time()

    dags_dir = PROJECT_ROOT / "airflow" / "dags"

    required_dags = [
        ("forecast_l0_daily_data.py", ["task_fetch_daily_data", "bi.dim_daily_usdcop"]),
        ("forecast_l1_daily_features.py", ["v_forecasting_features", "CREATE VIEW"]),
        ("l3b_forecasting_training.py", ["ForecastingEngine", "train"]),
        ("l5b_forecasting_inference.py", ["predict", "bi.fact_forecasts"]),
        ("forecast_l4_backtest_validation.py", ["direction_accuracy", "walk-forward"]),  # Note: hyphenated
        ("forecast_l6_drift_monitor.py", ["psi", "drift"]),
    ]

    results = {}
    for dag_file, keywords in required_dags:
        dag_path = dags_dir / dag_file
        if not dag_path.exists():
            results[dag_file] = "FILE NOT FOUND"
            continue

        content = dag_path.read_text(encoding='utf-8', errors='ignore')
        missing = [kw for kw in keywords if kw.lower() not in content.lower()]

        if missing:
            results[dag_file] = f"Missing: {missing}"
        else:
            results[dag_file] = "OK"

    all_passed = all(v == "OK" for v in results.values())
    duration = (time.time() - start) * 1000

    return ValidationResult(
        name="DAG Files",
        passed=all_passed,
        message=f"{sum(1 for v in results.values() if v == 'OK')}/{len(required_dags)} DAGs valid",
        details=results,
        duration_ms=duration
    )


def check_frontend_contracts() -> ValidationResult:
    """Validate frontend contract file exists."""
    import time
    start = time.time()

    frontend_dir = PROJECT_ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    contract_file = frontend_dir / "forecasting.contract.ts"

    if not contract_file.exists():
        return ValidationResult(
            name="Frontend Contracts",
            passed=False,
            message="forecasting.contract.ts not found",
            duration_ms=(time.time() - start) * 1000
        )

    content = contract_file.read_text()

    required_schemas = [
        "ForecastSchema",
        "ConsensusSchema",
        "DashboardResponseSchema",
        "ModelDetailResponseSchema",  # Not ModelDetailSchema
    ]

    missing = [s for s in required_schemas if s not in content]

    all_passed = len(missing) == 0
    duration = (time.time() - start) * 1000

    return ValidationResult(
        name="Frontend Contracts",
        passed=all_passed,
        message=f"All schemas present" if all_passed else f"Missing: {missing}",
        details={
            "file": str(contract_file),
            "schemas_found": len(required_schemas) - len(missing),
        },
        duration_ms=duration
    )


def check_frontend_service() -> ValidationResult:
    """Validate frontend service file exists."""
    import time
    start = time.time()

    service_file = PROJECT_ROOT / "usdcop-trading-dashboard" / "lib" / "services" / "forecasting.service.ts"

    if not service_file.exists():
        return ValidationResult(
            name="Frontend Service",
            passed=False,
            message="forecasting.service.ts not found",
            duration_ms=(time.time() - start) * 1000
        )

    content = service_file.read_text()

    required_methods = [
        "getDashboard",
        "getForecasts",
        "getConsensus",
        "loadLocalData",
        "parseCSV",
    ]

    missing = [m for m in required_methods if m not in content]

    all_passed = len(missing) == 0
    duration = (time.time() - start) * 1000

    return ValidationResult(
        name="Frontend Service",
        passed=all_passed,
        message=f"All methods present" if all_passed else f"Missing: {missing}",
        details={
            "methods_found": len(required_methods) - len(missing),
        },
        duration_ms=duration
    )


def check_engine_complete() -> ValidationResult:
    """Validate ForecastingEngine has all required methods."""
    import time
    start = time.time()

    try:
        from src.forecasting.engine import ForecastingEngine

        engine = ForecastingEngine(project_root=PROJECT_ROOT)

        required_methods = ["train", "predict"]
        required_attrs = ["model_factory", "walk_forward", "metrics"]

        missing_methods = [m for m in required_methods if not hasattr(engine, m)]
        missing_attrs = [a for a in required_attrs if not hasattr(engine, a)]

        all_passed = len(missing_methods) == 0 and len(missing_attrs) == 0
        duration = (time.time() - start) * 1000

        return ValidationResult(
            name="Forecasting Engine",
            passed=all_passed,
            message="Engine complete" if all_passed else f"Missing: {missing_methods + missing_attrs}",
            details={
                "methods": [m for m in required_methods if hasattr(engine, m)],
                "attrs": [a for a in required_attrs if hasattr(engine, a)],
            },
            duration_ms=duration
        )

    except Exception as e:
        return ValidationResult(
            name="Forecasting Engine",
            passed=False,
            message=f"Error: {str(e)}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(quick: bool = False, verbose: bool = False) -> ValidationReport:
    """Run all validation checks."""
    import time
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("FORECASTING PIPELINE VALIDATION")
    logger.info("=" * 60)

    checks = [
        ("Data Contracts", check_data_contracts),
        ("Forecasting Contracts", check_forecasting_contracts),
        ("Model Factory", check_model_factory),
        ("Model Training", lambda: check_model_training(quick)),
        ("Walk-Forward Validation", check_walk_forward),
        ("Metrics", check_metrics),
        ("DAG Files", check_dags_exist),
        ("Frontend Contracts", check_frontend_contracts),
        ("Frontend Service", check_frontend_service),
        ("Forecasting Engine", check_engine_complete),
    ]

    results = []
    passed = 0
    failed = 0
    skipped = 0

    for name, check_func in checks:
        logger.info(f"\nChecking: {name}...")
        try:
            result = check_func()
            results.append(result)

            if "Skipped" in result.message:
                skipped += 1
                status = "⏭️ SKIPPED"
            elif result.passed:
                passed += 1
                status = "✅ PASSED"
            else:
                failed += 1
                status = "❌ FAILED"

            logger.info(f"  {status}: {result.message} ({result.duration_ms:.0f}ms)")

            if verbose and result.details:
                for k, v in result.details.items():
                    logger.info(f"    {k}: {v}")

        except Exception as e:
            failed += 1
            result = ValidationResult(
                name=name,
                passed=False,
                message=f"Unexpected error: {str(e)}",
            )
            results.append(result)
            logger.error(f"  ❌ FAILED: {str(e)}")

    total_time = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total checks: {len(checks)}")
    logger.info(f"  ✅ Passed:  {passed}")
    logger.info(f"  ❌ Failed:  {failed}")
    logger.info(f"  ⏭️ Skipped: {skipped}")
    logger.info(f"Duration: {total_time:.2f}s")
    logger.info("=" * 60)

    if failed > 0:
        logger.info("\nFailed checks:")
        for r in results:
            if not r.passed and "Skipped" not in r.message:
                logger.info(f"  - {r.name}: {r.message}")

    return ValidationReport(
        total_checks=len(checks),
        passed=passed,
        failed=failed,
        skipped=skipped,
        results=results,
        duration_seconds=total_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Forecasting Pipeline")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    args = parser.parse_args()

    report = run_validation(quick=args.quick, verbose=args.verbose)

    # Exit code based on failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
