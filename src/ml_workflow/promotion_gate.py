"""
Promotion Gate
==============

Validation gate for model promotion to production.

This module provides:
- PromotionGate: Configurable validation rules
- Built-in validators for common criteria
- Extensible validator interface

Contract: CTR-PROMOTION-GATE-001
- All models must pass gate before promotion
- Configurable validation rules
- Audit trail of validation results

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from src.core.contracts.storage_contracts import ModelSnapshot, BacktestSnapshot

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT
# =============================================================================


class ValidationSeverity(str, Enum):
    """Severity of validation issue."""
    ERROR = "error"      # Blocks promotion
    WARNING = "warning"  # Logged but doesn't block
    INFO = "info"        # Informational


@dataclass
class ValidationIssue:
    """Single validation issue."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    passed: bool
    issues: List[ValidationIssue]
    validated_at: datetime = field(default_factory=datetime.utcnow)
    model_version: Optional[str] = None
    experiment_id: Optional[str] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "validated_at": self.validated_at.isoformat(),
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
        }


# =============================================================================
# VALIDATOR INTERFACE
# =============================================================================


class IValidator(ABC):
    """Abstract validator interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name."""
        pass

    @abstractmethod
    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """
        Validate model against rules.

        Args:
            model: Model snapshot to validate
            backtest: Optional backtest results
            config: Validation configuration

        Returns:
            List of validation issues
        """
        pass


# =============================================================================
# BUILT-IN VALIDATORS
# =============================================================================


class HashIntegrityValidator(IValidator):
    """Validates model has required hashes."""

    @property
    def name(self) -> str:
        return "hash_integrity"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        if not model.model_hash:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Model hash is missing",
            ))

        if not model.norm_stats_hash:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Norm stats hash is missing",
            ))

        if not model.config_hash:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Config hash is missing (recommended)",
            ))

        return issues


class FeatureContractValidator(IValidator):
    """Validates model feature contract compliance."""

    @property
    def name(self) -> str:
        return "feature_contract"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        # Check feature order exists
        if not model.feature_order:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Feature order is missing",
            ))
            return issues

        # Check feature order hash
        if not model.feature_order_hash:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Feature order hash is missing",
            ))

        # Check observation dimension
        expected_dim = config.get("expected_observation_dim", 15)
        if model.observation_dim != expected_dim:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Observation dimension mismatch",
                actual_value=model.observation_dim,
                expected_value=expected_dim,
            ))

        # Check action space
        expected_action_space = config.get("expected_action_space", 3)
        if model.action_space != expected_action_space:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Action space mismatch",
                actual_value=model.action_space,
                expected_value=expected_action_space,
            ))

        return issues


class PerformanceValidator(IValidator):
    """Validates model performance metrics."""

    @property
    def name(self) -> str:
        return "performance"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        if backtest is None:
            if config.get("require_backtest", False):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="Backtest required but not provided",
                ))
            return issues

        # Sharpe ratio check
        min_sharpe = config.get("min_sharpe", 0.0)
        if backtest.sharpe_ratio < min_sharpe:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Sharpe ratio below minimum",
                actual_value=round(backtest.sharpe_ratio, 2),
                expected_value=f">= {min_sharpe}",
            ))

        # Max drawdown check
        max_drawdown_limit = config.get("max_drawdown_limit", 0.5)
        if backtest.max_drawdown > max_drawdown_limit:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Max drawdown exceeds limit",
                actual_value=f"{backtest.max_drawdown:.2%}",
                expected_value=f"<= {max_drawdown_limit:.2%}",
            ))

        # Win rate check (warning only)
        min_win_rate = config.get("min_win_rate", 0.3)
        if backtest.win_rate < min_win_rate:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Win rate below recommended",
                actual_value=f"{backtest.win_rate:.2%}",
                expected_value=f">= {min_win_rate:.2%}",
            ))

        # Trade count check
        min_trades = config.get("min_trades", 10)
        if backtest.total_trades < min_trades:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Insufficient trades for statistical significance",
                actual_value=backtest.total_trades,
                expected_value=f">= {min_trades}",
            ))

        return issues


class LineageValidator(IValidator):
    """Validates model lineage is complete."""

    @property
    def name(self) -> str:
        return "lineage"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        # Check dataset snapshot exists
        if model.dataset_snapshot is None:
            if config.get("require_dataset_lineage", False):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="Dataset lineage required but missing",
                ))
            else:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message="Dataset lineage is missing (recommended)",
                ))

        # Check MLflow tracking
        if not model.mlflow_run_id:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="MLflow run ID is missing (recommended for reproducibility)",
            ))

        return issues


class MLflowFirstValidator(IValidator):
    """
    Validates model follows MLflow-First principle.

    Enforces:
    - Model must have MLflow run ID
    - Model must be registered in MLflow Model Registry
    - Model URI must point to MLflow (models:/ or runs:/)
    """

    @property
    def name(self) -> str:
        return "mlflow_first"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        # MLflow-First is mandatory
        enforce_mlflow_first = config.get("enforce_mlflow_first", True)

        if enforce_mlflow_first:
            # Must have MLflow run ID
            if not model.mlflow_run_id:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="MLflow-First: Model must have mlflow_run_id",
                ))

            # Must be registered in Model Registry
            model_uri = getattr(model, 'model_uri', None) or getattr(model, 's3_uri', None)
            if model_uri:
                if not (model_uri.startswith("models:/") or model_uri.startswith("runs:/")):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message="MLflow-First: Model must be in MLflow Registry",
                        actual_value=model_uri[:50] if len(model_uri) > 50 else model_uri,
                        expected_value="models:/<name>/<version> or runs:/<run_id>/...",
                    ))
            else:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="MLflow-First: Model URI is required",
                ))

        return issues


class DVCTrackedValidator(IValidator):
    """
    Validates dataset follows DVC-Tracked principle.

    Enforces:
    - Dataset must have DVC tag
    - Dataset hash must be present
    """

    @property
    def name(self) -> str:
        return "dvc_tracked"

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot],
        config: Dict[str, Any],
    ) -> List[ValidationIssue]:
        issues = []

        # DVC tracking is mandatory
        enforce_dvc_tracking = config.get("enforce_dvc_tracking", True)

        if enforce_dvc_tracking and model.dataset_snapshot:
            dataset = model.dataset_snapshot

            # Must have DVC tag
            dvc_tag = getattr(dataset, 'dvc_tag', None)
            if not dvc_tag:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="DVC-Tracked: Dataset must have dvc_tag",
                ))

            # Must have content hash
            content_hash = getattr(dataset, 'content_hash', None) or getattr(dataset, 'hash', None)
            if not content_hash:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="DVC-Tracked: Dataset must have content hash",
                ))

        return issues


# =============================================================================
# PROMOTION GATE
# =============================================================================


class PromotionGate:
    """
    Configurable validation gate for model promotion.

    Example:
        >>> gate = PromotionGate(config={
        ...     "min_sharpe": 0.5,
        ...     "max_drawdown_limit": 0.3,
        ...     "require_backtest": True,
        ... })
        >>>
        >>> result = gate.validate(model_snapshot, backtest_snapshot)
        >>> if result.passed:
        ...     promote_model(...)
        >>> else:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
    """

    DEFAULT_VALIDATORS: List[IValidator] = [
        HashIntegrityValidator(),
        FeatureContractValidator(),
        PerformanceValidator(),
        LineageValidator(),
        MLflowFirstValidator(),
        DVCTrackedValidator(),
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        validators: Optional[List[IValidator]] = None,
    ):
        """
        Initialize promotion gate.

        Args:
            config: Validation configuration
            validators: Custom validators (uses defaults if None)
        """
        self._config = config or {}
        self._validators = validators or self.DEFAULT_VALIDATORS.copy()

    def add_validator(self, validator: IValidator) -> None:
        """Add a custom validator."""
        self._validators.append(validator)

    def validate(
        self,
        model: ModelSnapshot,
        backtest: Optional[BacktestSnapshot] = None,
    ) -> ValidationResult:
        """
        Validate model against all rules.

        Args:
            model: Model snapshot to validate
            backtest: Optional backtest results

        Returns:
            ValidationResult with pass/fail and issues
        """
        all_issues: List[ValidationIssue] = []

        for validator in self._validators:
            try:
                issues = validator.validate(model, backtest, self._config)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                all_issues.append(ValidationIssue(
                    rule_name=validator.name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator error: {e}",
                ))

        # Gate passes if no errors (warnings are OK)
        passed = not any(
            i.severity == ValidationSeverity.ERROR
            for i in all_issues
        )

        result = ValidationResult(
            passed=passed,
            issues=all_issues,
            model_version=model.version,
            experiment_id=model.experiment_id,
        )

        # Log result
        if passed:
            logger.info(
                f"Model {model.experiment_id}/{model.version} passed validation "
                f"({len(result.warnings)} warnings)"
            )
        else:
            logger.warning(
                f"Model {model.experiment_id}/{model.version} failed validation "
                f"({len(result.errors)} errors, {len(result.warnings)} warnings)"
            )

        return result


# =============================================================================
# DEFAULT GATE CONFIGURATION
# =============================================================================


DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    # Feature contract
    "expected_observation_dim": 15,
    "expected_action_space": 3,

    # Performance thresholds
    "min_sharpe": 0.0,        # Minimum acceptable Sharpe ratio
    "max_drawdown_limit": 0.5,  # Maximum acceptable drawdown (50%)
    "min_win_rate": 0.3,      # Minimum win rate (warning only)
    "min_trades": 10,         # Minimum trades for significance

    # Lineage requirements
    "require_backtest": False,        # Require backtest results
    "require_dataset_lineage": False,  # Require dataset snapshot

    # MLflow-First + DVC-Tracked Principle
    "enforce_mlflow_first": True,     # Model must be in MLflow Registry
    "enforce_dvc_tracking": True,     # Dataset must have DVC tag
}


def create_default_gate() -> PromotionGate:
    """Create gate with default configuration."""
    return PromotionGate(config=DEFAULT_GATE_CONFIG)


def create_strict_gate() -> PromotionGate:
    """Create gate with strict configuration."""
    strict_config = {
        **DEFAULT_GATE_CONFIG,
        "min_sharpe": 0.5,
        "max_drawdown_limit": 0.3,
        "min_win_rate": 0.4,
        "min_trades": 50,
        "require_backtest": True,
        "require_dataset_lineage": True,
    }
    return PromotionGate(config=strict_config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums and Results
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    # Validator interface
    "IValidator",
    # Built-in validators
    "HashIntegrityValidator",
    "FeatureContractValidator",
    "PerformanceValidator",
    "LineageValidator",
    "MLflowFirstValidator",
    "DVCTrackedValidator",
    # Gate
    "PromotionGate",
    "DEFAULT_GATE_CONFIG",
    "create_default_gate",
    "create_strict_gate",
]
