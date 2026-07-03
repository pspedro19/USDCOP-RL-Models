"""
Model Smoke Test Validation System
===================================

Provides comprehensive smoke testing for ML models before deployment.
Validates model loading, input/output shapes, determinism, latency,
and artifact integrity.

This module implements Contract CTR-SMOKE-001:
- All models must pass smoke tests before production deployment
- Tests cover loading, shape validation, determinism, and performance

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND STUBS
# =============================================================================

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models that can be tested."""

    def predict(self, observation: np.ndarray) -> Any:
        """Run inference on an observation."""
        ...


class ModelLoaderStub:
    """
    Stub ModelLoader for when the actual implementation is not available.

    This stub provides a minimal interface for testing. In production,
    it delegates to MLflow or the actual ModelLoader.
    """

    @staticmethod
    def load(uri: str) -> ModelProtocol:
        """
        Load a model from a URI.

        Supports:
        - MLflow URIs: models:/model_name/stage or runs:/run_id/artifacts/model
        - Local paths: /path/to/model.zip or /path/to/model.onnx

        Args:
            uri: Model URI

        Returns:
            Model with predict(observation) method

        Raises:
            RuntimeError: If model cannot be loaded
        """
        try:
            import mlflow

            # Check if it's an MLflow URI
            if uri.startswith("models:/") or uri.startswith("runs:/"):
                logger.info(f"Loading model from MLflow: {uri}")
                model = mlflow.pyfunc.load_model(uri)
                return _MLflowModelWrapper(model)
            else:
                # Try loading from local path
                return _load_local_model(uri)

        except ImportError:
            logger.warning("MLflow not available, trying local load")
            return _load_local_model(uri)


class _MLflowModelWrapper:
    """Wrapper to provide predict() interface for MLflow models."""

    def __init__(self, model: Any):
        self._model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run prediction, handling MLflow's DataFrame expectations."""
        import pandas as pd

        # Ensure 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # MLflow pyfunc expects DataFrame or ndarray
        try:
            result = self._model.predict(observation)
        except Exception:
            # Try with DataFrame
            df = pd.DataFrame(observation)
            result = self._model.predict(df)

        return np.atleast_1d(result)


def _load_local_model(path: str) -> ModelProtocol:
    """Load model from local path."""
    from pathlib import Path

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".onnx":
        return _load_onnx_model(str(path))
    elif suffix == ".zip":
        return _load_sb3_model(str(path))
    else:
        raise ValueError(f"Unsupported model format: {suffix}")


def _load_onnx_model(path: str) -> ModelProtocol:
    """Load ONNX model."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return _ONNXModelWrapper(session)
    except ImportError:
        raise RuntimeError("onnxruntime not installed")


def _load_sb3_model(path: str) -> ModelProtocol:
    """Load Stable-Baselines3 model."""
    try:
        from stable_baselines3 import PPO, SAC, TD3, A2C, DQN

        # Try each algorithm
        for algo_class in [PPO, SAC, TD3, A2C, DQN]:
            try:
                model = algo_class.load(path)
                return _SB3ModelWrapper(model)
            except Exception:
                continue

        raise RuntimeError(f"Could not load SB3 model from {path}")
    except ImportError:
        raise RuntimeError("stable-baselines3 not installed")


class _ONNXModelWrapper:
    """Wrapper for ONNX models."""

    def __init__(self, session: Any):
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        observation = observation.astype(np.float32)

        outputs = self._session.run(None, {self._input_name: observation})
        return outputs[0]


class _SB3ModelWrapper:
    """Wrapper for SB3 models."""

    def __init__(self, model: Any):
        self._model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run SB3 inference."""
        if observation.ndim == 2 and observation.shape[0] == 1:
            observation = observation.squeeze(0)
        observation = observation.astype(np.float32)

        action, _ = self._model.predict(observation, deterministic=True)
        return np.atleast_1d(action)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestResult:
    """
    Result of a single validation check.

    Attributes:
        name: Name of the test (e.g., "model_loads", "input_shape")
        passed: Whether the test passed
        message: Human-readable result message
        value: Optional value produced by the test (for debugging)
    """
    name: str
    passed: bool
    message: str
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class SmokeTestResult:
    """
    Complete result of a model smoke test.

    Attributes:
        passed: Whether all tests passed
        checks: List of individual test results
        errors: List of error messages for failed tests
        duration_ms: Total test duration in milliseconds
    """
    passed: bool
    checks: List[TestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with all test results and metadata
        """
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.passed),
                "failed": sum(1 for c in self.checks if not c.passed),
            }
        }


# =============================================================================
# SMOKE TEST CLASS
# =============================================================================

class ModelSmokeTest:
    """
    Comprehensive smoke test suite for ML models.

    Validates that a model:
    1. Loads correctly
    2. Accepts the expected input shape (15 features)
    3. Produces valid outputs (action in [0,1,2], confidence in [0,1])
    4. Is deterministic (same input -> same output)
    5. Meets latency requirements
    6. Has required artifacts (norm_stats)
    7. Has logged hashes for reproducibility

    Usage:
        >>> smoke_test = ModelSmokeTest("models:/my_model/Production")
        >>> result = smoke_test.run()
        >>> if result.passed:
        ...     print("Model ready for deployment!")
        >>> else:
        ...     for error in result.errors:
        ...         print(f"FAIL: {error}")

    Args:
        model_uri: URI to the model (MLflow or local path)
        max_latency_ms: Maximum acceptable inference latency in milliseconds
    """

    # Expected number of input features for USDCOP trading models
    EXPECTED_FEATURES = 15

    # Valid discrete actions: 0=HOLD, 1=BUY, 2=SELL
    VALID_ACTIONS = {0, 1, 2}

    def __init__(
        self,
        model_uri: str,
        max_latency_ms: float = 100.0
    ):
        """
        Initialize the smoke test.

        Args:
            model_uri: URI to the model
            max_latency_ms: Maximum acceptable inference latency
        """
        self.model_uri = model_uri
        self.max_latency_ms = max_latency_ms
        self._model: Optional[ModelProtocol] = None
        self._run_info: Optional[Any] = None

    def run(self) -> SmokeTestResult:
        """
        Execute all smoke test checks.

        Returns:
            SmokeTestResult with all check results
        """
        start_time = time.perf_counter()

        checks: List[TestResult] = []
        errors: List[str] = []

        logger.info(f"Starting smoke test for: {self.model_uri}")

        # Run checks in order (some depend on previous checks)
        check_methods = [
            self._check_model_loads,
            self._check_input_shape,
            self._check_output_valid,
            self._check_deterministic,
            self._check_latency,
            self._check_norm_stats_exist,
            self._check_hashes_logged,
        ]

        for check_method in check_methods:
            try:
                result = check_method()
                checks.append(result)

                if not result.passed:
                    errors.append(f"{result.name}: {result.message}")
                    logger.warning(f"Check failed: {result.name} - {result.message}")
                else:
                    logger.info(f"Check passed: {result.name}")

            except Exception as e:
                error_msg = f"{check_method.__name__}: {str(e)}"
                checks.append(TestResult(
                    name=check_method.__name__.replace("_check_", ""),
                    passed=False,
                    message=f"Exception: {str(e)}",
                ))
                errors.append(error_msg)
                logger.error(f"Check exception: {error_msg}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        passed = len(errors) == 0

        result = SmokeTestResult(
            passed=passed,
            checks=checks,
            errors=errors,
            duration_ms=duration_ms,
        )

        logger.info(
            f"Smoke test completed: {'PASSED' if passed else 'FAILED'} "
            f"({len(checks) - len(errors)}/{len(checks)} checks passed, "
            f"{duration_ms:.1f}ms)"
        )

        return result

    def _check_model_loads(self) -> TestResult:
        """
        Verify model loads correctly.

        Tests:
        - Model can be loaded from URI
        - Model has a predict method
        """
        try:
            self._model = ModelLoaderStub.load(self.model_uri)

            if self._model is None:
                return TestResult(
                    name="model_loads",
                    passed=False,
                    message="Model loaded as None",
                )

            # Verify predict method exists
            if not hasattr(self._model, "predict"):
                return TestResult(
                    name="model_loads",
                    passed=False,
                    message="Model lacks predict() method",
                )

            return TestResult(
                name="model_loads",
                passed=True,
                message=f"Model loaded successfully from {self.model_uri}",
                value=type(self._model).__name__,
            )

        except Exception as e:
            return TestResult(
                name="model_loads",
                passed=False,
                message=f"Failed to load model: {str(e)}",
            )

    def _check_input_shape(self) -> TestResult:
        """
        Verify model accepts 15 features.

        Tests:
        - Model accepts a (1, 15) shaped input
        - Model produces output without error
        """
        if self._model is None:
            return TestResult(
                name="input_shape",
                passed=False,
                message="Model not loaded (skipped)",
            )

        try:
            # Create test input with expected shape
            test_input = np.random.randn(1, self.EXPECTED_FEATURES).astype(np.float32)

            # Try prediction
            output = self._model.predict(test_input)

            return TestResult(
                name="input_shape",
                passed=True,
                message=f"Model accepts {self.EXPECTED_FEATURES} features",
                value={"input_shape": test_input.shape, "output_shape": np.array(output).shape},
            )

        except Exception as e:
            return TestResult(
                name="input_shape",
                passed=False,
                message=f"Model failed with {self.EXPECTED_FEATURES} features: {str(e)}",
            )

    def _check_output_valid(self) -> TestResult:
        """
        Verify output is valid: action in [0,1,2], confidence in [0,1].

        Tests:
        - Action is one of: 0 (HOLD), 1 (BUY), 2 (SELL)
        - If confidence is returned, it's in [0, 1]
        """
        if self._model is None:
            return TestResult(
                name="output_valid",
                passed=False,
                message="Model not loaded (skipped)",
            )

        try:
            test_input = np.random.randn(1, self.EXPECTED_FEATURES).astype(np.float32)
            output = self._model.predict(test_input)
            output = np.atleast_1d(output)

            # Extract action (first element or only element)
            action = int(output.flatten()[0])

            # Validate action is in valid set
            if action not in self.VALID_ACTIONS:
                return TestResult(
                    name="output_valid",
                    passed=False,
                    message=f"Invalid action: {action}. Expected one of {self.VALID_ACTIONS}",
                    value={"action": action, "output": output.tolist()},
                )

            # Check confidence if present (second element)
            if len(output.flatten()) > 1:
                confidence = float(output.flatten()[1])
                if not (0.0 <= confidence <= 1.0):
                    return TestResult(
                        name="output_valid",
                        passed=False,
                        message=f"Invalid confidence: {confidence}. Expected [0, 1]",
                        value={"confidence": confidence},
                    )

            return TestResult(
                name="output_valid",
                passed=True,
                message=f"Output valid: action={action}",
                value={"action": action, "output": output.tolist()},
            )

        except Exception as e:
            return TestResult(
                name="output_valid",
                passed=False,
                message=f"Output validation failed: {str(e)}",
            )

    def _check_deterministic(self) -> TestResult:
        """
        Verify model is deterministic: 5 predictions with same input yield same result.

        Tests:
        - Running predict() 5 times with identical input produces identical output
        """
        if self._model is None:
            return TestResult(
                name="deterministic",
                passed=False,
                message="Model not loaded (skipped)",
            )

        try:
            # Use fixed seed for reproducible test input
            np.random.seed(42)
            test_input = np.random.randn(1, self.EXPECTED_FEATURES).astype(np.float32)

            # Run 5 predictions
            outputs = []
            for _ in range(5):
                output = self._model.predict(test_input.copy())
                outputs.append(np.array(output).flatten())

            # Check all outputs are identical
            first_output = outputs[0]
            all_same = all(np.allclose(first_output, o) for o in outputs[1:])

            if not all_same:
                return TestResult(
                    name="deterministic",
                    passed=False,
                    message="Model produces different outputs for same input",
                    value={"outputs": [o.tolist() for o in outputs]},
                )

            return TestResult(
                name="deterministic",
                passed=True,
                message="Model is deterministic (5/5 predictions identical)",
                value={"output": first_output.tolist()},
            )

        except Exception as e:
            return TestResult(
                name="deterministic",
                passed=False,
                message=f"Determinism check failed: {str(e)}",
            )

    def _check_latency(self) -> TestResult:
        """
        Verify inference latency is within acceptable bounds.

        Tests:
        - Warmup with 10 iterations
        - Measure 100 iterations
        - Check mean, p50, p95, p99 latencies
        """
        if self._model is None:
            return TestResult(
                name="latency",
                passed=False,
                message="Model not loaded (skipped)",
            )

        try:
            test_input = np.random.randn(1, self.EXPECTED_FEATURES).astype(np.float32)

            # Warmup: 10 iterations
            for _ in range(10):
                self._model.predict(test_input.copy())

            # Measure: 100 iterations
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                self._model.predict(test_input.copy())
                latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

            latencies = np.array(latencies)
            mean_latency = float(np.mean(latencies))
            p50_latency = float(np.percentile(latencies, 50))
            p95_latency = float(np.percentile(latencies, 95))
            p99_latency = float(np.percentile(latencies, 99))

            stats = {
                "mean_ms": round(mean_latency, 3),
                "p50_ms": round(p50_latency, 3),
                "p95_ms": round(p95_latency, 3),
                "p99_ms": round(p99_latency, 3),
                "max_allowed_ms": self.max_latency_ms,
            }

            # Check p95 against max latency
            if p95_latency > self.max_latency_ms:
                return TestResult(
                    name="latency",
                    passed=False,
                    message=f"p95 latency {p95_latency:.2f}ms exceeds max {self.max_latency_ms}ms",
                    value=stats,
                )

            return TestResult(
                name="latency",
                passed=True,
                message=f"Latency OK: mean={mean_latency:.2f}ms, p95={p95_latency:.2f}ms",
                value=stats,
            )

        except Exception as e:
            return TestResult(
                name="latency",
                passed=False,
                message=f"Latency check failed: {str(e)}",
            )

    def _check_norm_stats_exist(self) -> TestResult:
        """
        Verify norm_stats artifact exists in MLflow.

        Tests:
        - Model has associated norm_stats.json artifact
        """
        # Only applicable for MLflow URIs
        if not (self.model_uri.startswith("models:/") or self.model_uri.startswith("runs:/")):
            return TestResult(
                name="norm_stats_exist",
                passed=True,
                message="Skipped (local model, not MLflow)",
                value={"skipped": True},
            )

        try:
            import mlflow

            # Get run info from the model URI
            if self.model_uri.startswith("models:/"):
                # Parse: models:/model_name/stage_or_version
                parts = self.model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                stage_or_version = parts[1] if len(parts) > 1 else "Production"

                client = mlflow.tracking.MlflowClient()

                # Try to get by stage first
                try:
                    versions = client.get_latest_versions(model_name, stages=[stage_or_version])
                    if versions:
                        run_id = versions[0].run_id
                    else:
                        # Try as version number
                        version_info = client.get_model_version(model_name, stage_or_version)
                        run_id = version_info.run_id
                except Exception:
                    return TestResult(
                        name="norm_stats_exist",
                        passed=False,
                        message=f"Could not resolve model URI: {self.model_uri}",
                    )
            else:
                # runs:/run_id/artifacts/...
                run_id = self.model_uri.split("/")[1]

            # Check for norm_stats artifact
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id)
            artifact_names = [a.path for a in artifacts]

            # Look for norm_stats in various locations
            norm_stats_found = any(
                "norm_stats" in name.lower()
                for name in artifact_names
            )

            # Also check in subdirectories
            if not norm_stats_found:
                for artifact in artifacts:
                    if artifact.is_dir:
                        sub_artifacts = client.list_artifacts(run_id, artifact.path)
                        norm_stats_found = any(
                            "norm_stats" in a.path.lower()
                            for a in sub_artifacts
                        )
                        if norm_stats_found:
                            break

            if norm_stats_found:
                return TestResult(
                    name="norm_stats_exist",
                    passed=True,
                    message="norm_stats artifact found",
                    value={"artifacts": artifact_names},
                )
            else:
                return TestResult(
                    name="norm_stats_exist",
                    passed=False,
                    message="norm_stats artifact not found",
                    value={"artifacts": artifact_names},
                )

        except ImportError:
            return TestResult(
                name="norm_stats_exist",
                passed=True,
                message="Skipped (MLflow not installed)",
                value={"skipped": True},
            )
        except Exception as e:
            return TestResult(
                name="norm_stats_exist",
                passed=False,
                message=f"Could not check artifacts: {str(e)}",
            )

    def _check_hashes_logged(self) -> TestResult:
        """
        Verify norm_stats_hash and dataset_hash are logged in MLflow params.

        Tests:
        - Run has norm_stats_hash param
        - Run has dataset_hash param
        """
        # Only applicable for MLflow URIs
        if not (self.model_uri.startswith("models:/") or self.model_uri.startswith("runs:/")):
            return TestResult(
                name="hashes_logged",
                passed=True,
                message="Skipped (local model, not MLflow)",
                value={"skipped": True},
            )

        try:
            import mlflow

            # Get run info
            if self.model_uri.startswith("models:/"):
                parts = self.model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                stage_or_version = parts[1] if len(parts) > 1 else "Production"

                client = mlflow.tracking.MlflowClient()

                try:
                    versions = client.get_latest_versions(model_name, stages=[stage_or_version])
                    if versions:
                        run_id = versions[0].run_id
                    else:
                        version_info = client.get_model_version(model_name, stage_or_version)
                        run_id = version_info.run_id
                except Exception:
                    return TestResult(
                        name="hashes_logged",
                        passed=False,
                        message=f"Could not resolve model URI: {self.model_uri}",
                    )
            else:
                run_id = self.model_uri.split("/")[1]

            # Get run params
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            params = run.data.params

            # Check for required hashes
            missing_hashes = []
            found_hashes = {}

            if "norm_stats_hash" not in params:
                missing_hashes.append("norm_stats_hash")
            else:
                found_hashes["norm_stats_hash"] = params["norm_stats_hash"]

            if "dataset_hash" not in params:
                missing_hashes.append("dataset_hash")
            else:
                found_hashes["dataset_hash"] = params["dataset_hash"]

            if missing_hashes:
                return TestResult(
                    name="hashes_logged",
                    passed=False,
                    message=f"Missing hashes: {missing_hashes}",
                    value={"missing": missing_hashes, "found": found_hashes},
                )

            return TestResult(
                name="hashes_logged",
                passed=True,
                message="All required hashes logged",
                value=found_hashes,
            )

        except ImportError:
            return TestResult(
                name="hashes_logged",
                passed=True,
                message="Skipped (MLflow not installed)",
                value={"skipped": True},
            )
        except Exception as e:
            return TestResult(
                name="hashes_logged",
                passed=False,
                message=f"Could not check params: {str(e)}",
            )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_smoke_test(
    model_uri: str,
    max_latency_ms: float = 100.0
) -> SmokeTestResult:
    """
    Run a full smoke test on a model.

    Convenience function that creates a ModelSmokeTest instance and runs it.

    Args:
        model_uri: URI to the model (MLflow or local path)
        max_latency_ms: Maximum acceptable inference latency in milliseconds

    Returns:
        SmokeTestResult with all check results

    Example:
        >>> result = run_smoke_test("models:/usdcop_ppo/Production")
        >>> print(f"Passed: {result.passed}")
        >>> for check in result.checks:
        ...     print(f"  {check.name}: {'PASS' if check.passed else 'FAIL'}")
    """
    smoke_test = ModelSmokeTest(model_uri, max_latency_ms)
    return smoke_test.run()
