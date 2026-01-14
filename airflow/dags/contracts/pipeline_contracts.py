"""
Pipeline Contracts - Pre/Post-condition validation for DAG tasks.

This module provides:
- Pre-condition validation before task execution
- Post-condition validation after task completion
- Artifact contracts for inter-DAG data exchange

Design Patterns:
- Contract Pattern: Pre/post-conditions as executable specifications
- Decorator Pattern: Wrap tasks with validation
- Factory Pattern: Create validators from configuration

SOLID Principles:
- SRP: Each contract validates one concern
- OCP: Add new validators without modifying existing code
- DIP: Tasks depend on contract abstractions

Contract: CTR-PIPELINE-001
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

class ValidationStatus(str, Enum):
    """Status of contract validation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a contract validation."""

    status: ValidationStatus
    contract_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed or is just a warning."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "contract_name": self.contract_name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# CONTRACT PROTOCOL
# =============================================================================

class Contract(Protocol):
    """Protocol for all pipeline contracts."""

    @property
    def name(self) -> str:
        """Contract name for identification."""
        ...

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate the contract.

        Args:
            context: Task context including XCom, config, etc.

        Returns:
            ValidationResult with status and details
        """
        ...


# =============================================================================
# BASE CONTRACTS
# =============================================================================

class BaseContract(ABC):
    """Abstract base for all contracts."""

    def __init__(self, name: str, required: bool = True):
        self._name = name
        self._required = required

    @property
    def name(self) -> str:
        return self._name

    @property
    def required(self) -> bool:
        return self._required

    @abstractmethod
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        pass

    def _passed(self, message: str, details: Optional[Dict] = None) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.PASSED,
            contract_name=self._name,
            message=message,
            details=details or {}
        )

    def _failed(self, message: str, details: Optional[Dict] = None) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            contract_name=self._name,
            message=message,
            details=details or {}
        )

    def _warning(self, message: str, details: Optional[Dict] = None) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.WARNING,
            contract_name=self._name,
            message=message,
            details=details or {}
        )


# =============================================================================
# DATA FRESHNESS CONTRACTS
# =============================================================================

class DataFreshnessContract(BaseContract):
    """Validates that data is not stale."""

    def __init__(
        self,
        name: str,
        table_name: str,
        max_staleness_minutes: int,
        timestamp_column: str = "time"
    ):
        super().__init__(name)
        self.table_name = table_name
        self.max_staleness_minutes = max_staleness_minutes
        self.timestamp_column = timestamp_column

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Check if data is fresh enough."""
        try:
            # Get database connection from context
            conn = context.get("db_connection")
            if not conn:
                return self._failed("No database connection in context")

            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX({self.timestamp_column}))) / 60
                FROM {self.table_name}
            """)
            result = cursor.fetchone()
            cursor.close()

            if not result or result[0] is None:
                return self._failed(f"No data in {self.table_name}")

            age_minutes = float(result[0])

            if age_minutes > self.max_staleness_minutes:
                return self._failed(
                    f"Data is {age_minutes:.1f} min old (max: {self.max_staleness_minutes})",
                    {"age_minutes": age_minutes, "threshold": self.max_staleness_minutes}
                )

            return self._passed(
                f"Data is fresh ({age_minutes:.1f} min old)",
                {"age_minutes": age_minutes}
            )

        except Exception as e:
            return self._failed(f"Validation error: {e}")


class OHLCVFreshnessContract(DataFreshnessContract):
    """OHLCV-specific freshness validation."""

    def __init__(self, max_staleness_minutes: int = 10):
        super().__init__(
            name="ohlcv_freshness",
            table_name="usdcop_m5_ohlcv",
            max_staleness_minutes=max_staleness_minutes,
            timestamp_column="time"
        )


class MacroFreshnessContract(DataFreshnessContract):
    """Macro data freshness validation (daily data)."""

    def __init__(self, max_staleness_hours: int = 48):
        super().__init__(
            name="macro_freshness",
            table_name="macro_indicators_daily",
            max_staleness_minutes=max_staleness_hours * 60,
            timestamp_column="fecha"
        )


# =============================================================================
# MODEL CONTRACTS
# =============================================================================

class ModelLoadedContract(BaseContract):
    """Validates that a model is properly loaded."""

    def __init__(self, model_id: str):
        super().__init__(f"model_loaded_{model_id}")
        self.model_id = model_id

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Check if model is loaded and valid."""
        model_registry = context.get("model_registry")
        if not model_registry:
            return self._failed("No model registry in context")

        model = model_registry.get_model(self.model_id)
        if model is None:
            return self._failed(f"Model {self.model_id} not loaded")

        return self._passed(f"Model {self.model_id} loaded")


class ModelHashContract(BaseContract):
    """Validates model file hash matches expected value."""

    def __init__(self, model_path: str, expected_hash: Optional[str] = None):
        super().__init__("model_hash")
        self.model_path = model_path
        self.expected_hash = expected_hash

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate model file hash."""
        path = Path(self.model_path)

        if not path.exists():
            return self._failed(f"Model file not found: {self.model_path}")

        # Calculate hash
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        details = {"file_hash": file_hash}

        if self.expected_hash and file_hash != self.expected_hash:
            return self._failed(
                f"Hash mismatch: {file_hash} != {self.expected_hash}",
                details
            )

        return self._passed(f"Model hash verified: {file_hash[:16]}...", details)


class FeatureConfigHashContract(BaseContract):
    """Validates feature config matches training config."""

    def __init__(self, config_path: str = "config/feature_registry.yaml"):
        super().__init__("feature_config_hash")
        self.config_path = config_path

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate feature config hash matches model's expected config."""
        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            # Calculate config hash
            config_str = yaml.dump(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()

            # Get expected hash from model metadata
            expected_hash = context.get("expected_feature_config_hash")

            if expected_hash and config_hash != expected_hash:
                return self._warning(
                    f"Feature config may differ from training: {config_hash[:16]}",
                    {"current_hash": config_hash, "expected_hash": expected_hash}
                )

            return self._passed(
                f"Feature config hash: {config_hash[:16]}",
                {"config_hash": config_hash}
            )

        except Exception as e:
            return self._failed(f"Cannot validate feature config: {e}")


# =============================================================================
# INFERENCE PRE-CONDITIONS
# =============================================================================

@dataclass
class InferencePreConditions:
    """Pre-conditions that must be met before inference.

    All conditions must pass for inference to proceed.
    """

    model_loaded: bool = False
    ohlcv_fresh: bool = False
    macro_fresh: bool = False
    within_trading_hours: bool = False
    risk_limits_ok: bool = False
    warmup_complete: bool = False

    def all_passed(self) -> bool:
        """Check if all pre-conditions passed."""
        return all([
            self.model_loaded,
            self.ohlcv_fresh,
            self.macro_fresh,
            self.within_trading_hours,
            self.risk_limits_ok,
            self.warmup_complete
        ])

    def get_failures(self) -> List[str]:
        """Get list of failed pre-conditions."""
        failures = []
        if not self.model_loaded:
            failures.append("model_loaded")
        if not self.ohlcv_fresh:
            failures.append("ohlcv_fresh")
        if not self.macro_fresh:
            failures.append("macro_fresh")
        if not self.within_trading_hours:
            failures.append("within_trading_hours")
        if not self.risk_limits_ok:
            failures.append("risk_limits_ok")
        if not self.warmup_complete:
            failures.append("warmup_complete")
        return failures


class InferencePreConditionsContract(BaseContract):
    """Comprehensive pre-condition validation for inference."""

    def __init__(
        self,
        model_id: str,
        max_ohlcv_staleness_minutes: int = 10,
        max_macro_staleness_hours: int = 48,
        min_warmup_bars: int = 50
    ):
        super().__init__("inference_preconditions")
        self.model_id = model_id
        self.max_ohlcv_staleness_minutes = max_ohlcv_staleness_minutes
        self.max_macro_staleness_hours = max_macro_staleness_hours
        self.min_warmup_bars = min_warmup_bars

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate all inference pre-conditions."""
        conditions = InferencePreConditions()

        # Check model loaded
        model_registry = context.get("model_registry")
        if model_registry:
            conditions.model_loaded = model_registry.get_model(self.model_id) is not None

        # Check data freshness (would need DB connection in context)
        conn = context.get("db_connection")
        if conn:
            try:
                cursor = conn.cursor()

                # OHLCV freshness
                cursor.execute("""
                    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(time))) / 60
                    FROM usdcop_m5_ohlcv
                """)
                ohlcv_age = cursor.fetchone()[0] or 999
                conditions.ohlcv_fresh = ohlcv_age <= self.max_ohlcv_staleness_minutes

                # Macro freshness
                cursor.execute("""
                    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(fecha))) / 3600
                    FROM macro_indicators_daily
                """)
                macro_age = cursor.fetchone()[0] or 999
                conditions.macro_fresh = macro_age <= self.max_macro_staleness_hours

                # Warmup bars
                cursor.execute("""
                    SELECT COUNT(*) FROM usdcop_m5_ohlcv
                    WHERE time > NOW() - INTERVAL '1 day'
                """)
                bar_count = cursor.fetchone()[0] or 0
                conditions.warmup_complete = bar_count >= self.min_warmup_bars

                cursor.close()
            except Exception as e:
                logger.warning(f"Error checking data freshness: {e}")

        # Check trading hours
        conditions.within_trading_hours = context.get("within_trading_hours", False)

        # Check risk limits
        risk_manager = context.get("risk_manager")
        if risk_manager:
            conditions.risk_limits_ok = not risk_manager.is_blocked()
        else:
            conditions.risk_limits_ok = True  # No risk manager means no limits

        # Build result
        if conditions.all_passed():
            return self._passed(
                "All inference pre-conditions passed",
                {"conditions": conditions.__dict__}
            )
        else:
            failures = conditions.get_failures()
            return self._failed(
                f"Pre-conditions failed: {', '.join(failures)}",
                {"conditions": conditions.__dict__, "failures": failures}
            )


# =============================================================================
# ARTIFACT CONTRACTS
# =============================================================================

@dataclass
class ModelArtifact:
    """Artifact produced by training pipeline."""

    model_id: str
    model_path: str
    model_hash: str
    feature_config_hash: str
    training_timestamp: datetime
    metrics: Dict[str, float]
    version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "feature_config_hash": self.feature_config_hash,
            "training_timestamp": self.training_timestamp.isoformat(),
            "metrics": self.metrics,
            "version": self.version
        }


@dataclass
class BacktestArtifact:
    """Artifact produced by backtest validation pipeline."""

    model_id: str
    backtest_timestamp: datetime
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    passed_validation: bool
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backtest_timestamp": self.backtest_timestamp.isoformat(),
            "total_trades": self.total_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "passed_validation": self.passed_validation,
            "validation_errors": self.validation_errors
        }


class ArtifactContract(BaseContract):
    """Validates artifact from upstream pipeline."""

    def __init__(
        self,
        artifact_type: str,
        required_fields: List[str],
        xcom_key: str = "artifact"
    ):
        super().__init__(f"artifact_{artifact_type}")
        self.artifact_type = artifact_type
        self.required_fields = required_fields
        self.xcom_key = xcom_key

    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate artifact from upstream task."""
        artifact = context.get(self.xcom_key)

        if not artifact:
            return self._failed(f"No {self.artifact_type} artifact in context")

        # Check required fields
        missing = [f for f in self.required_fields if f not in artifact]

        if missing:
            return self._failed(
                f"Artifact missing fields: {missing}",
                {"missing_fields": missing, "artifact_type": self.artifact_type}
            )

        return self._passed(
            f"{self.artifact_type} artifact validated",
            {"artifact_type": self.artifact_type, "fields": list(artifact.keys())}
        )


# =============================================================================
# CONTRACT VALIDATOR
# =============================================================================

class ContractValidator:
    """
    Orchestrates validation of multiple contracts.

    Usage:
        validator = ContractValidator()
        validator.add_contract(OHLCVFreshnessContract())
        validator.add_contract(ModelLoadedContract("ppo_primary"))

        results = validator.validate_all(context)
        if not validator.all_passed(results):
            raise ContractViolation(results)
    """

    def __init__(self):
        self._contracts: List[BaseContract] = []

    def add_contract(self, contract: BaseContract) -> "ContractValidator":
        """Add a contract to validate."""
        self._contracts.append(contract)
        return self

    def validate_all(self, context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate all contracts and return results."""
        results = []

        for contract in self._contracts:
            try:
                result = contract.validate(context)
                results.append(result)

                if not result.is_valid and contract.required:
                    logger.error(f"Contract {contract.name} failed: {result.message}")
                elif not result.is_valid:
                    logger.warning(f"Contract {contract.name} failed (optional): {result.message}")
                else:
                    logger.info(f"Contract {contract.name} passed: {result.message}")

            except Exception as e:
                results.append(ValidationResult(
                    status=ValidationStatus.FAILED,
                    contract_name=contract.name,
                    message=f"Validation error: {e}"
                ))

        return results

    def all_passed(self, results: List[ValidationResult]) -> bool:
        """Check if all required contracts passed."""
        for i, result in enumerate(results):
            contract = self._contracts[i]
            if contract.required and not result.is_valid:
                return False
        return True

    def get_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results."""
        passed = sum(1 for r in results if r.is_valid)
        failed = sum(1 for r in results if not r.is_valid)

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "all_passed": self.all_passed(results),
            "results": [r.to_dict() for r in results]
        }


# =============================================================================
# TASK DECORATOR
# =============================================================================

def validate_preconditions(*contracts: BaseContract):
    """
    Decorator to validate pre-conditions before task execution.

    Usage:
        @validate_preconditions(
            OHLCVFreshnessContract(max_staleness_minutes=10),
            ModelLoadedContract("ppo_primary")
        )
        def run_inference(**context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(**context):
            validator = ContractValidator()
            for contract in contracts:
                validator.add_contract(contract)

            results = validator.validate_all(context)

            if not validator.all_passed(results):
                summary = validator.get_summary(results)
                failures = [r for r in results if not r.is_valid]
                raise ContractViolationError(
                    f"Pre-conditions failed: {[f.contract_name for f in failures]}",
                    summary
                )

            return func(**context)
        return wrapper
    return decorator


class ContractViolationError(Exception):
    """Raised when a contract validation fails."""

    def __init__(self, message: str, summary: Dict[str, Any]):
        self.message = message
        self.summary = summary
        super().__init__(message)
