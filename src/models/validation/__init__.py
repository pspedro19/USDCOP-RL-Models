"""
Model Validation Module
========================

This module provides validation utilities for ML models including:
- Smoke tests for model health validation
- Input/output shape validation
- Latency benchmarking
- Determinism checks

Classes:
    TestResult: Result of a single validation check
    SmokeTestResult: Complete smoke test result with all checks
    ModelSmokeTest: Main smoke test orchestrator

Functions:
    run_smoke_test: Convenience function to run a full smoke test

Example:
    >>> from src.models.validation import run_smoke_test
    >>> result = run_smoke_test("models:/my_model/Production")
    >>> if result.passed:
    ...     print("Model passed all smoke tests!")
    >>> else:
    ...     print(f"Failures: {result.errors}")
"""

from .smoke_test import (
    TestResult,
    SmokeTestResult,
    ModelSmokeTest,
    run_smoke_test,
)

__all__ = [
    "TestResult",
    "SmokeTestResult",
    "ModelSmokeTest",
    "run_smoke_test",
]

__version__ = "1.0.0"
