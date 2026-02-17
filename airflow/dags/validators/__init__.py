# -*- coding: utf-8 -*-
"""
L0 Data Validators Package
==========================
Validation strategies for L0 macro data quality assurance.

Contract: CTR-L0-VALIDATOR-001

Validators:
- SchemaValidator: Column types and NOT NULL constraints
- RangeValidator: Values within expected ranges
- CompletenessValidator: Data coverage percentage
- LeakageValidator: Future data detection
- FreshnessValidator: Data recency checks
- AntiLeakageValidator: Comprehensive ML anti-leakage validation

Usage:
    from validators import ValidationPipeline, ValidationReport

    pipeline = ValidationPipeline()
    report = pipeline.validate(df, variable='comm_oil_brent_glb_d_brent')

    if not report.overall_passed:
        for error in report.get_all_errors():
            print(error)

    # Anti-leakage validation for ML pipelines
    from validators import AntiLeakageValidator, validate_anti_leakage

    validator = AntiLeakageValidator()
    result = validator.validate_macro_shift_t1(df, 'IBR')
    result = validator.validate_dataset_splits(train_df, val_df, test_df)

Version: 1.1.0
"""

from .data_validators import (
    ValidationResult,
    ValidationReport,
    DataValidator,
    SchemaValidator,
    RangeValidator,
    CompletenessValidator,
    LeakageValidator,
    FreshnessValidator,
    ValidationPipeline,
    ValidationSeverity,
)

from .anti_leakage_validator import (
    AntiLeakageValidator,
    AntiLeakageReport,
    validate_anti_leakage,
)

__all__ = [
    # Core validators
    'ValidationResult',
    'ValidationReport',
    'ValidationSeverity',
    'DataValidator',
    'SchemaValidator',
    'RangeValidator',
    'CompletenessValidator',
    'LeakageValidator',
    'FreshnessValidator',
    'ValidationPipeline',
    # Anti-leakage validators
    'AntiLeakageValidator',
    'AntiLeakageReport',
    'validate_anti_leakage',
]
