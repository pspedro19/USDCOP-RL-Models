"""
Smoke Test Module
=================

Pre-deployment validation for models and inference pipeline.

This module provides smoke testing capabilities to verify:
- Model can be loaded
- Model produces valid outputs
- Inference latency is acceptable
- All required features are available

Components:
- SmokeTestConfig: Configuration for smoke tests
- SmokeTestResult: Result container with pass/fail status
- SmokeTest: Main smoke testing class

Author: USD/COP Trading System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import time
import numpy as np


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeTestConfig:
    """Configuration for smoke tests."""

    # Model validation
    check_model_loadable: bool = True
    check_model_output_valid: bool = True
    check_model_deterministic: bool = True

    # Performance validation
    check_inference_latency: bool = True
    max_inference_latency_ms: float = 100.0
    inference_warmup_runs: int = 3
    inference_test_runs: int = 10

    # Feature validation
    check_features_available: bool = True
    check_feature_ranges: bool = True
    required_features: List[str] = field(default_factory=list)

    # Output validation
    check_output_range: bool = True
    expected_output_shape: Optional[tuple] = None
    output_min: float = -1.0
    output_max: float = 1.0

    # General
    fail_fast: bool = False
    verbose: bool = False


@dataclass
class SmokeTestResult:
    """Result of a smoke test run."""

    # Overall status
    passed: bool = False
    status: str = "not_run"

    # Individual checks
    checks: List[ValidationCheck] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    warning_checks: int = 0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_check(self, check: ValidationCheck) -> None:
        """Add a validation check result."""
        self.checks.append(check)
        if check.status == ValidationStatus.PASSED:
            self.passed_checks += 1
        elif check.status == ValidationStatus.FAILED:
            self.failed_checks += 1
            self.errors.append(f"{check.name}: {check.message}")
        elif check.status == ValidationStatus.SKIPPED:
            self.skipped_checks += 1
        elif check.status == ValidationStatus.WARNING:
            self.warning_checks += 1
            self.warnings.append(f"{check.name}: {check.message}")


class SmokeTest:
    """Smoke testing for model deployment validation.

    This class provides comprehensive pre-deployment validation
    to catch issues before models are deployed to production.

    Example:
        smoke_test = SmokeTest(config)
        result = smoke_test.run(model, sample_input)
        if result.passed:
            deploy_model(model)
    """

    def __init__(self, config: Optional[SmokeTestConfig] = None):
        """Initialize smoke test.

        Args:
            config: Configuration for smoke tests
        """
        self.config = config or SmokeTestConfig()
        self._model = None
        self._sample_input = None

    def run(
        self,
        model: Any,
        sample_input: np.ndarray,
        feature_provider: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> SmokeTestResult:
        """Run smoke tests on a model.

        Args:
            model: Model to test (must have predict method)
            sample_input: Sample input array for testing
            feature_provider: Optional function to get current features

        Returns:
            SmokeTestResult with pass/fail status and details
        """
        result = SmokeTestResult()
        result.start_time = datetime.now()
        result.status = "running"

        self._model = model
        self._sample_input = sample_input

        try:
            # Run all configured checks
            if self.config.check_model_loadable:
                self._check_model_loadable(result)
                if self.config.fail_fast and result.failed_checks > 0:
                    return self._finalize_result(result)

            if self.config.check_model_output_valid:
                self._check_model_output_valid(result)
                if self.config.fail_fast and result.failed_checks > 0:
                    return self._finalize_result(result)

            if self.config.check_model_deterministic:
                self._check_model_deterministic(result)
                if self.config.fail_fast and result.failed_checks > 0:
                    return self._finalize_result(result)

            if self.config.check_inference_latency:
                self._check_inference_latency(result)
                if self.config.fail_fast and result.failed_checks > 0:
                    return self._finalize_result(result)

            if self.config.check_features_available and feature_provider:
                self._check_features_available(result, feature_provider)
                if self.config.fail_fast and result.failed_checks > 0:
                    return self._finalize_result(result)

            if self.config.check_output_range:
                self._check_output_range(result)

        except Exception as e:
            result.errors.append(f"Smoke test failed with exception: {str(e)}")
            result.add_check(ValidationCheck(
                name="smoke_test_execution",
                status=ValidationStatus.FAILED,
                message=str(e),
            ))

        return self._finalize_result(result)

    def _finalize_result(self, result: SmokeTestResult) -> SmokeTestResult:
        """Finalize and return result."""
        result.end_time = datetime.now()
        if result.start_time:
            result.total_duration_ms = (
                (result.end_time - result.start_time).total_seconds() * 1000
            )

        result.passed = result.failed_checks == 0
        result.status = "passed" if result.passed else "failed"

        return result

    def _check_model_loadable(self, result: SmokeTestResult) -> None:
        """Check that model is properly loaded."""
        start = time.perf_counter()
        check = ValidationCheck(name="model_loadable", status=ValidationStatus.PASSED)

        try:
            # Check model has predict method
            if not hasattr(self._model, 'predict'):
                check.status = ValidationStatus.FAILED
                check.message = "Model does not have 'predict' method"
            else:
                check.message = "Model loaded and has predict method"

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error checking model: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)

    def _check_model_output_valid(self, result: SmokeTestResult) -> None:
        """Check that model produces valid output."""
        start = time.perf_counter()
        check = ValidationCheck(name="model_output_valid", status=ValidationStatus.PASSED)

        try:
            output = self._model.predict(self._sample_input)

            # Check output is not None
            if output is None:
                check.status = ValidationStatus.FAILED
                check.message = "Model returned None"
            # Check output is numpy array or can be converted
            elif not isinstance(output, np.ndarray):
                try:
                    output = np.array(output)
                except Exception:
                    check.status = ValidationStatus.FAILED
                    check.message = "Model output cannot be converted to numpy array"

            # Check for NaN/Inf
            if check.status == ValidationStatus.PASSED:
                if np.any(np.isnan(output)):
                    check.status = ValidationStatus.FAILED
                    check.message = "Model output contains NaN values"
                elif np.any(np.isinf(output)):
                    check.status = ValidationStatus.FAILED
                    check.message = "Model output contains Inf values"
                else:
                    check.message = "Model output is valid"
                    check.details['output_shape'] = output.shape if hasattr(output, 'shape') else None
                    check.details['output_dtype'] = str(output.dtype) if hasattr(output, 'dtype') else None

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error during prediction: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)

    def _check_model_deterministic(self, result: SmokeTestResult) -> None:
        """Check that model produces deterministic outputs."""
        start = time.perf_counter()
        check = ValidationCheck(name="model_deterministic", status=ValidationStatus.PASSED)

        try:
            outputs = []
            for _ in range(3):
                output = self._model.predict(self._sample_input)
                outputs.append(output)

            # Check all outputs are identical
            for i in range(1, len(outputs)):
                if not np.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-8):
                    check.status = ValidationStatus.WARNING
                    check.message = "Model outputs are not deterministic"
                    check.details['max_diff'] = float(np.max(np.abs(outputs[0] - outputs[i])))
                    break
            else:
                check.message = "Model produces deterministic outputs"

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error checking determinism: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)

    def _check_inference_latency(self, result: SmokeTestResult) -> None:
        """Check that inference latency is acceptable."""
        start = time.perf_counter()
        check = ValidationCheck(name="inference_latency", status=ValidationStatus.PASSED)

        try:
            # Warmup runs
            for _ in range(self.config.inference_warmup_runs):
                self._model.predict(self._sample_input)

            # Timed runs
            latencies = []
            for _ in range(self.config.inference_test_runs):
                run_start = time.perf_counter()
                self._model.predict(self._sample_input)
                run_end = time.perf_counter()
                latencies.append((run_end - run_start) * 1000)

            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            min_latency = np.min(latencies)

            check.details['avg_latency_ms'] = float(avg_latency)
            check.details['max_latency_ms'] = float(max_latency)
            check.details['min_latency_ms'] = float(min_latency)
            check.details['std_latency_ms'] = float(np.std(latencies))

            if avg_latency > self.config.max_inference_latency_ms:
                check.status = ValidationStatus.FAILED
                check.message = (
                    f"Average latency {avg_latency:.2f}ms exceeds "
                    f"limit {self.config.max_inference_latency_ms}ms"
                )
            else:
                check.message = f"Average latency {avg_latency:.2f}ms is acceptable"

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error measuring latency: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)

    def _check_features_available(
        self,
        result: SmokeTestResult,
        feature_provider: Callable[[], Dict[str, float]]
    ) -> None:
        """Check that required features are available."""
        start = time.perf_counter()
        check = ValidationCheck(name="features_available", status=ValidationStatus.PASSED)

        try:
            features = feature_provider()

            if not features:
                check.status = ValidationStatus.FAILED
                check.message = "No features available"
            else:
                # Check required features
                missing = []
                for required in self.config.required_features:
                    if required not in features:
                        missing.append(required)

                if missing:
                    check.status = ValidationStatus.FAILED
                    check.message = f"Missing required features: {missing}"
                    check.details['missing_features'] = missing
                else:
                    check.message = f"All {len(features)} features available"
                    check.details['feature_count'] = len(features)

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error checking features: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)

    def _check_output_range(self, result: SmokeTestResult) -> None:
        """Check that model output is within expected range."""
        start = time.perf_counter()
        check = ValidationCheck(name="output_range", status=ValidationStatus.PASSED)

        try:
            output = self._model.predict(self._sample_input)

            if isinstance(output, np.ndarray):
                min_val = float(np.min(output))
                max_val = float(np.max(output))

                check.details['output_min'] = min_val
                check.details['output_max'] = max_val

                if min_val < self.config.output_min or max_val > self.config.output_max:
                    check.status = ValidationStatus.WARNING
                    check.message = (
                        f"Output range [{min_val:.4f}, {max_val:.4f}] "
                        f"outside expected [{self.config.output_min}, {self.config.output_max}]"
                    )
                else:
                    check.message = f"Output range [{min_val:.4f}, {max_val:.4f}] is valid"

        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.message = f"Error checking output range: {str(e)}"

        check.duration_ms = (time.perf_counter() - start) * 1000
        result.add_check(check)


def run_smoke_test(
    model: Any,
    sample_input: np.ndarray,
    max_latency_ms: float = 100.0,
) -> SmokeTestResult:
    """Convenience function to run smoke tests with default config.

    Args:
        model: Model to test
        sample_input: Sample input array
        max_latency_ms: Maximum allowed inference latency

    Returns:
        SmokeTestResult
    """
    config = SmokeTestConfig(max_inference_latency_ms=max_latency_ms)
    smoke_test = SmokeTest(config)
    return smoke_test.run(model, sample_input)


__all__ = [
    'ValidationStatus',
    'ValidationCheck',
    'SmokeTestConfig',
    'SmokeTestResult',
    'SmokeTest',
    'run_smoke_test',
]
