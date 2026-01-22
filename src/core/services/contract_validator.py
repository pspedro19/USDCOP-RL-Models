"""
Contract Validator Service
==========================

Service for validating contracts across pipeline stages (L1→L3→L5).
Implements GAP 3, 10: Feature order validation at transitions.

Design Patterns:
- Chain of Responsibility: Validate at each stage
- Strategy Pattern: Different validators per stage
- Template Method: Common validation flow

SOLID Principles:
- Single Responsibility: Only handles contract validation
- Open/Closed: Easy to add new validators
- Liskov Substitution: All validators are interchangeable

Author: Trading Team
Date: 2026-01-17
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Imports from SSOT Contracts
# =============================================================================

# Import from SSOT - these are the source of truth
from src.core.contracts import (
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    OBSERVATION_DIM,
)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class PipelineStage(Enum):
    """Pipeline stages for contract validation."""
    L1_FEATURES = "L1_features"
    L3_TRAINING = "L3_training"
    L5_INFERENCE = "L5_inference"


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    ERROR = "error"      # Must fix, blocks execution
    WARNING = "warning"  # Should fix, continues with warning
    INFO = "info"        # Informational only


@dataclass
class ValidationError:
    """Represents a contract validation error."""
    code: str
    message: str
    severity: ValidationSeverity
    stage: PipelineStage
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "stage": self.stage.value,
            "expected": str(self.expected) if self.expected else None,
            "actual": str(self.actual) if self.actual else None,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Result of contract validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    stage: Optional[PipelineStage] = None
    validated_at: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, error: ValidationError) -> None:
        """Add error and mark as invalid."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge with another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            stage=self.stage,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "stage": self.stage.value if self.stage else None,
            "validated_at": self.validated_at,
        }

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            error_msgs = "\n".join(str(e) for e in self.errors)
            raise ContractValidationError(
                f"Contract validation failed at {self.stage}:\n{error_msgs}"
            )


class ContractValidationError(Exception):
    """Exception raised when contract validation fails."""
    pass


# =============================================================================
# Base Validator (Template Method Pattern)
# =============================================================================

class BaseContractValidator(ABC):
    """
    Base class for contract validators.

    Uses Template Method pattern - defines validation flow,
    subclasses implement specific checks.
    """

    def __init__(self, stage: PipelineStage):
        self.stage = stage

    def validate(self, **kwargs) -> ValidationResult:
        """
        Template method for validation.

        Calls abstract methods that subclasses must implement.
        """
        result = ValidationResult(is_valid=True, stage=self.stage)

        # Pre-validation hook
        self._pre_validate(result, **kwargs)

        # Core validations
        self._validate_feature_order(result, **kwargs)
        self._validate_observation_dim(result, **kwargs)
        self._validate_stage_specific(result, **kwargs)

        # Post-validation hook
        self._post_validate(result, **kwargs)

        return result

    def _pre_validate(self, result: ValidationResult, **kwargs) -> None:
        """Hook before validation. Override if needed."""
        pass

    def _post_validate(self, result: ValidationResult, **kwargs) -> None:
        """Hook after validation. Override if needed."""
        pass

    @abstractmethod
    def _validate_feature_order(self, result: ValidationResult, **kwargs) -> None:
        """Validate feature order matches SSOT."""
        pass

    @abstractmethod
    def _validate_observation_dim(self, result: ValidationResult, **kwargs) -> None:
        """Validate observation dimension matches SSOT."""
        pass

    @abstractmethod
    def _validate_stage_specific(self, result: ValidationResult, **kwargs) -> None:
        """Stage-specific validations."""
        pass


# =============================================================================
# Stage-Specific Validators (Strategy Pattern)
# =============================================================================

class L1FeatureValidator(BaseContractValidator):
    """Validator for L1 (Feature Generation) stage."""

    def __init__(self):
        super().__init__(PipelineStage.L1_FEATURES)

    def _validate_feature_order(self, result: ValidationResult, **kwargs) -> None:
        """Validate generated features match FEATURE_ORDER."""
        feature_columns = kwargs.get("feature_columns", [])

        if not feature_columns:
            return

        if tuple(feature_columns) != FEATURE_ORDER:
            result.add_error(ValidationError(
                code="L1_FEATURE_ORDER_MISMATCH",
                message="Generated features don't match FEATURE_ORDER SSOT",
                severity=ValidationSeverity.ERROR,
                stage=self.stage,
                expected=list(FEATURE_ORDER),
                actual=feature_columns,
            ))

    def _validate_observation_dim(self, result: ValidationResult, **kwargs) -> None:
        """Validate feature count matches OBSERVATION_DIM."""
        n_features = kwargs.get("n_features", 0)

        if n_features and n_features != OBSERVATION_DIM:
            result.add_error(ValidationError(
                code="L1_DIMENSION_MISMATCH",
                message=f"Feature count {n_features} != OBSERVATION_DIM {OBSERVATION_DIM}",
                severity=ValidationSeverity.ERROR,
                stage=self.stage,
                expected=OBSERVATION_DIM,
                actual=n_features,
            ))

    def _validate_stage_specific(self, result: ValidationResult, **kwargs) -> None:
        """L1-specific: Validate feature hash is computed."""
        computed_hash = kwargs.get("feature_order_hash")

        if computed_hash and computed_hash != FEATURE_ORDER_HASH:
            result.add_error(ValidationError(
                code="L1_HASH_MISMATCH",
                message="Computed feature order hash doesn't match SSOT",
                severity=ValidationSeverity.ERROR,
                stage=self.stage,
                expected=FEATURE_ORDER_HASH,
                actual=computed_hash,
            ))


class L3TrainingValidator(BaseContractValidator):
    """Validator for L3 (Training) stage."""

    def __init__(self):
        super().__init__(PipelineStage.L3_TRAINING)

    def _validate_feature_order(self, result: ValidationResult, **kwargs) -> None:
        """Validate training uses correct feature order."""
        input_feature_order_hash = kwargs.get("feature_order_hash")

        if input_feature_order_hash and input_feature_order_hash != FEATURE_ORDER_HASH:
            result.add_error(ValidationError(
                code="L3_FEATURE_ORDER_HASH_MISMATCH",
                message="Training input feature order hash doesn't match SSOT",
                severity=ValidationSeverity.ERROR,
                stage=self.stage,
                expected=FEATURE_ORDER_HASH,
                actual=input_feature_order_hash,
            ))

    def _validate_observation_dim(self, result: ValidationResult, **kwargs) -> None:
        """Validate observation shape matches."""
        observation = kwargs.get("observation")

        if observation is not None:
            if isinstance(observation, np.ndarray):
                obs_dim = observation.shape[-1] if observation.ndim > 1 else observation.shape[0]
                if obs_dim != OBSERVATION_DIM:
                    result.add_error(ValidationError(
                        code="L3_OBSERVATION_DIM_MISMATCH",
                        message=f"Observation dimension {obs_dim} != {OBSERVATION_DIM}",
                        severity=ValidationSeverity.ERROR,
                        stage=self.stage,
                        expected=OBSERVATION_DIM,
                        actual=obs_dim,
                    ))

    def _validate_stage_specific(self, result: ValidationResult, **kwargs) -> None:
        """L3-specific: Validate model metadata includes hashes."""
        model_metadata = kwargs.get("model_metadata", {})

        # Check that feature_order_hash will be stored with model
        if "feature_order_hash" not in model_metadata:
            result.add_error(ValidationError(
                code="L3_MISSING_FEATURE_HASH",
                message="Model metadata missing feature_order_hash",
                severity=ValidationSeverity.WARNING,
                stage=self.stage,
                context={"metadata_keys": list(model_metadata.keys())},
            ))


class L5InferenceValidator(BaseContractValidator):
    """Validator for L5 (Inference) stage."""

    def __init__(self):
        super().__init__(PipelineStage.L5_INFERENCE)

    def _validate_feature_order(self, result: ValidationResult, **kwargs) -> None:
        """
        CRITICAL: Validate inference uses same feature order as training.

        This is the key validation that prevents silent failures when
        feature schema changes between training and inference.
        """
        model_feature_order_hash = kwargs.get("model_feature_order_hash")
        current_feature_order_hash = FEATURE_ORDER_HASH

        if model_feature_order_hash and model_feature_order_hash != current_feature_order_hash:
            result.add_error(ValidationError(
                code="L5_FEATURE_ORDER_MISMATCH",
                message=(
                    "CRITICAL: Model was trained with different feature order! "
                    "Inference results will be INCORRECT."
                ),
                severity=ValidationSeverity.ERROR,
                stage=self.stage,
                expected=model_feature_order_hash,
                actual=current_feature_order_hash,
                context={"action": "Retrain model with current feature order"},
            ))

    def _validate_observation_dim(self, result: ValidationResult, **kwargs) -> None:
        """Validate inference observation dimension."""
        observation = kwargs.get("observation")

        if observation is not None:
            if isinstance(observation, np.ndarray):
                obs_dim = observation.shape[-1] if observation.ndim > 1 else observation.shape[0]
                if obs_dim != OBSERVATION_DIM:
                    result.add_error(ValidationError(
                        code="L5_OBSERVATION_DIM_MISMATCH",
                        message=f"Inference observation has {obs_dim} features, expected {OBSERVATION_DIM}",
                        severity=ValidationSeverity.ERROR,
                        stage=self.stage,
                        expected=OBSERVATION_DIM,
                        actual=obs_dim,
                    ))

    def _validate_stage_specific(self, result: ValidationResult, **kwargs) -> None:
        """L5-specific: Validate norm_stats hash matches training."""
        model_norm_stats_hash = kwargs.get("model_norm_stats_hash")
        current_norm_stats_hash = kwargs.get("current_norm_stats_hash")

        if model_norm_stats_hash and current_norm_stats_hash:
            if model_norm_stats_hash != current_norm_stats_hash:
                result.add_error(ValidationError(
                    code="L5_NORM_STATS_MISMATCH",
                    message=(
                        "Normalization statistics changed since model training. "
                        "Feature scaling may be incorrect."
                    ),
                    severity=ValidationSeverity.WARNING,
                    stage=self.stage,
                    expected=model_norm_stats_hash,
                    actual=current_norm_stats_hash,
                ))


# =============================================================================
# Contract Validator (Facade)
# =============================================================================

class ContractValidator:
    """
    Unified contract validator for the ML pipeline.

    Validates contracts at each stage transition (L1→L3→L5).

    Example:
        validator = ContractValidator()

        # Validate at L3 training
        result = validator.validate_l3_training(
            feature_order_hash=input_hash,
            observation=training_obs,
        )
        result.raise_if_invalid()

        # Validate at L5 inference
        result = validator.validate_l5_inference(
            model_feature_order_hash=model_metadata["feature_order_hash"],
            observation=inference_obs,
        )
        if not result.is_valid:
            logger.error("Inference blocked due to contract violation")
    """

    def __init__(self):
        """Initialize validators for each stage."""
        self._validators = {
            PipelineStage.L1_FEATURES: L1FeatureValidator(),
            PipelineStage.L3_TRAINING: L3TrainingValidator(),
            PipelineStage.L5_INFERENCE: L5InferenceValidator(),
        }

    def validate_l1_features(
        self,
        feature_columns: List[str],
        n_features: Optional[int] = None,
        feature_order_hash: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate L1 feature generation output.

        Args:
            feature_columns: List of generated feature names
            n_features: Number of features generated
            feature_order_hash: Computed hash of feature order

        Returns:
            ValidationResult
        """
        return self._validators[PipelineStage.L1_FEATURES].validate(
            feature_columns=feature_columns,
            n_features=n_features or len(feature_columns),
            feature_order_hash=feature_order_hash,
        )

    def validate_l3_training(
        self,
        feature_order_hash: Optional[str] = None,
        observation: Optional[np.ndarray] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate L3 training inputs.

        Args:
            feature_order_hash: Hash from L1 features
            observation: Sample observation array
            model_metadata: Metadata to store with model

        Returns:
            ValidationResult
        """
        return self._validators[PipelineStage.L3_TRAINING].validate(
            feature_order_hash=feature_order_hash,
            observation=observation,
            model_metadata=model_metadata or {},
        )

    def validate_l5_inference(
        self,
        model_feature_order_hash: Optional[str] = None,
        model_norm_stats_hash: Optional[str] = None,
        current_norm_stats_hash: Optional[str] = None,
        observation: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Validate L5 inference inputs.

        CRITICAL: This must be called before every inference to ensure
        feature schema hasn't changed since model training.

        Args:
            model_feature_order_hash: Hash stored with model
            model_norm_stats_hash: Norm stats hash from training
            current_norm_stats_hash: Current norm stats hash
            observation: Observation to predict on

        Returns:
            ValidationResult
        """
        return self._validators[PipelineStage.L5_INFERENCE].validate(
            model_feature_order_hash=model_feature_order_hash,
            model_norm_stats_hash=model_norm_stats_hash,
            current_norm_stats_hash=current_norm_stats_hash,
            observation=observation,
        )

    def validate_feature_subset(
        self,
        feature_columns: List[str],
        variant_name: str,
        expected_observation_dim: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate that a feature subset is a valid subset of FEATURE_ORDER.

        Used for A/B experiments with reduced feature sets (e.g., MACRO_CORE vs MACRO_FULL).
        Validates:
        1. All features exist in SSOT FEATURE_ORDER
        2. Feature order is preserved (relative ordering)
        3. No duplicate features
        4. Observation dimension matches if provided

        Args:
            feature_columns: List of feature names in the subset
            variant_name: Name of the variant (for error messages)
            expected_observation_dim: Expected total dimension (features + state)

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult(is_valid=True, stage=PipelineStage.L3_TRAINING)

        # Check for duplicates
        if len(feature_columns) != len(set(feature_columns)):
            duplicates = [f for f in feature_columns if feature_columns.count(f) > 1]
            result.add_error(ValidationError(
                code="SUBSET_DUPLICATE_FEATURES",
                message=f"Variant '{variant_name}' has duplicate features: {set(duplicates)}",
                severity=ValidationSeverity.ERROR,
                stage=PipelineStage.L3_TRAINING,
                actual=duplicates,
                context={"variant": variant_name},
            ))
            return result  # Can't continue with duplicates

        # Check all features exist in SSOT
        ssot_features = set(FEATURE_ORDER)
        unknown_features = [f for f in feature_columns if f not in ssot_features]

        if unknown_features:
            result.add_error(ValidationError(
                code="SUBSET_UNKNOWN_FEATURES",
                message=f"Variant '{variant_name}' contains features not in SSOT: {unknown_features}",
                severity=ValidationSeverity.ERROR,
                stage=PipelineStage.L3_TRAINING,
                expected=list(FEATURE_ORDER),
                actual=unknown_features,
                context={"variant": variant_name},
            ))

        # Check order is preserved
        if not unknown_features:  # Only check if all features are valid
            last_idx = -1
            order_violations = []

            for feature in feature_columns:
                current_idx = FEATURE_ORDER.index(feature)
                if current_idx < last_idx:
                    order_violations.append({
                        "feature": feature,
                        "expected_after": FEATURE_ORDER[last_idx],
                        "ssot_index": current_idx,
                    })
                last_idx = current_idx

            if order_violations:
                result.add_error(ValidationError(
                    code="SUBSET_ORDER_VIOLATION",
                    message=f"Variant '{variant_name}' has features in wrong order relative to SSOT",
                    severity=ValidationSeverity.ERROR,
                    stage=PipelineStage.L3_TRAINING,
                    expected="Features must preserve SSOT FEATURE_ORDER ordering",
                    actual=order_violations,
                    context={"variant": variant_name, "ssot_order": list(FEATURE_ORDER)},
                ))

        # Check observation dimension if provided
        # Note: observation_dim typically includes state features (position, time_normalized)
        STATE_FEATURES_COUNT = 2  # position, time_normalized
        if expected_observation_dim is not None:
            expected_market_features = expected_observation_dim - STATE_FEATURES_COUNT
            actual_market_features = len(feature_columns)

            if expected_market_features != actual_market_features:
                result.add_error(ValidationError(
                    code="SUBSET_DIMENSION_MISMATCH",
                    message=(
                        f"Variant '{variant_name}' observation_dim mismatch: "
                        f"config says {expected_observation_dim} total ({expected_market_features} market + {STATE_FEATURES_COUNT} state), "
                        f"but feature_columns has {actual_market_features} market features"
                    ),
                    severity=ValidationSeverity.ERROR,
                    stage=PipelineStage.L3_TRAINING,
                    expected=expected_market_features,
                    actual=actual_market_features,
                    context={"variant": variant_name, "observation_dim": expected_observation_dim},
                ))

        # Log success for valid subsets
        if result.is_valid:
            logger.info(
                f"Feature subset '{variant_name}' validated: "
                f"{len(feature_columns)} features (subset of {len(FEATURE_ORDER)})"
            )

        return result

    def validate_transition(
        self,
        from_stage: PipelineStage,
        to_stage: PipelineStage,
        from_hashes: Dict[str, str],
        to_hashes: Dict[str, str],
    ) -> ValidationResult:
        """
        Validate stage transition (e.g., L1→L3, L3→L5).

        Ensures hashes match between stages.

        Args:
            from_stage: Source stage
            to_stage: Target stage
            from_hashes: Hashes from source stage
            to_hashes: Hashes at target stage

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, stage=to_stage)

        # Feature order hash must match
        from_hash = from_hashes.get("feature_order_hash")
        to_hash = to_hashes.get("feature_order_hash")

        if from_hash and to_hash and from_hash != to_hash:
            result.add_error(ValidationError(
                code="TRANSITION_FEATURE_ORDER_MISMATCH",
                message=f"Feature order hash changed from {from_stage.value} to {to_stage.value}",
                severity=ValidationSeverity.ERROR,
                stage=to_stage,
                expected=from_hash,
                actual=to_hash,
                context={
                    "from_stage": from_stage.value,
                    "to_stage": to_stage.value,
                },
            ))

        # SSOT hash must match current contract
        if to_hash and to_hash != FEATURE_ORDER_HASH:
            result.add_error(ValidationError(
                code="TRANSITION_SSOT_MISMATCH",
                message="Feature order hash doesn't match current SSOT",
                severity=ValidationSeverity.ERROR,
                stage=to_stage,
                expected=FEATURE_ORDER_HASH,
                actual=to_hash,
            ))

        return result


# =============================================================================
# Decorator for Automatic Validation
# =============================================================================

def validate_contract(stage: PipelineStage):
    """
    Decorator to automatically validate contracts at function entry.

    Usage:
        @validate_contract(PipelineStage.L5_INFERENCE)
        def predict(self, observation: np.ndarray, model_metadata: Dict):
            ...
    """
    validator = ContractValidator()

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract observation from args/kwargs
            observation = kwargs.get("observation")
            if observation is None and len(args) > 1:
                observation = args[1]  # Assuming self is first arg

            # Get model metadata for L5
            model_metadata = kwargs.get("model_metadata", {})

            # Validate based on stage
            if stage == PipelineStage.L5_INFERENCE:
                result = validator.validate_l5_inference(
                    model_feature_order_hash=model_metadata.get("feature_order_hash"),
                    observation=observation,
                )
            elif stage == PipelineStage.L3_TRAINING:
                result = validator.validate_l3_training(
                    observation=observation,
                )
            else:
                result = ValidationResult(is_valid=True)

            # Raise if invalid
            result.raise_if_invalid()

            # Log warnings
            for warning in result.warnings:
                logger.warning(str(warning))

            return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Factory Function
# =============================================================================

def create_contract_validator() -> ContractValidator:
    """Factory function to create ContractValidator."""
    return ContractValidator()


__all__ = [
    "ContractValidator",
    "ValidationResult",
    "ValidationError",
    "ValidationSeverity",
    "PipelineStage",
    "ContractValidationError",
    "validate_contract",
    "create_contract_validator",
]
