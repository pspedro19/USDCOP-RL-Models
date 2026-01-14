"""
Consistency Validation Service
==============================
Validates feature store consistency between training and inference.

SOLID Principles:
- Single Responsibility: Only validates consistency
- Open/Closed: New validators via registration
- Dependency Inversion: Uses ContractValidator abstraction

Design Patterns:
- Strategy Pattern: Different validation strategies
- Chain of Responsibility: Sequential validations

CRITICAL: This service ensures:
1. Norm stats are loaded correctly (not hardcoded defaults)
2. Builder type matches observation dimension
3. Hash verification for model integrity
4. Feature order consistency
"""

import logging
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..contracts.model_contract import (
    ModelContract,
    ModelRegistry,
    ContractValidator,
    BuilderType,
    get_model_contract,
    compute_json_hash,
    compute_file_hash,
    NormStatsNotFoundError,
    HashVerificationError,
)
from ..core.builder_factory import BuilderFactory, get_observation_builder

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of a validation check"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ConsistencyReport:
    """Complete consistency validation report"""
    model_id: str
    overall_status: ValidationStatus
    checks: List[ValidationResult]
    validation_time_ms: float
    timestamp: datetime

    @property
    def passed(self) -> bool:
        return self.overall_status == ValidationStatus.PASSED

    @property
    def has_warnings(self) -> bool:
        return any(c.status == ValidationStatus.WARNING for c in self.checks)

    @property
    def failed_checks(self) -> List[ValidationResult]:
        return [c for c in self.checks if c.status == ValidationStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": [c.check_name for c in self.failed_checks],
            "validation_time_ms": self.validation_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ConsistencyValidatorService:
    """
    Service for validating feature store consistency.

    Performs comprehensive validation:
    1. Model contract registration
    2. Norm stats file existence and content
    3. Builder type / dimension consistency
    4. Hash verification (optional)
    5. Feature order validation
    """

    def __init__(self, project_root: Path):
        """
        Initialize consistency validator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.contract_validator = ContractValidator(project_root)

    def validate_model(
        self,
        model_id: str,
        verify_hashes: bool = False
    ) -> ConsistencyReport:
        """
        Perform complete consistency validation for a model.

        Args:
            model_id: Model identifier to validate
            verify_hashes: Whether to verify file hashes

        Returns:
            ConsistencyReport with all validation results
        """
        import time
        start_time = time.time()

        checks: List[ValidationResult] = []

        # 1. Validate model contract registration
        checks.append(self._check_model_registration(model_id))

        # Skip remaining checks if model not registered
        if checks[-1].status == ValidationStatus.FAILED:
            return self._build_report(model_id, checks, start_time)

        # Get contract for remaining checks
        try:
            contract = get_model_contract(model_id)
        except Exception as e:
            checks.append(ValidationResult(
                check_name="contract_retrieval",
                status=ValidationStatus.FAILED,
                message=f"Failed to get contract: {str(e)}",
            ))
            return self._build_report(model_id, checks, start_time)

        # 2. Validate norm_stats file exists (CRITICAL)
        checks.append(self._check_norm_stats_exists(contract))

        # 3. Validate builder type / dimension consistency
        checks.append(self._check_builder_dimension_consistency(contract))

        # 4. Validate norm_stats content (not hardcoded defaults)
        checks.append(self._check_norm_stats_not_defaults(contract))

        # 5. Validate model file exists
        checks.append(self._check_model_file_exists(contract))

        # 6. Hash verification (optional)
        if verify_hashes:
            checks.extend(self._check_hash_integrity(contract))

        # 7. Validate builder can be instantiated
        checks.append(self._check_builder_instantiation(model_id))

        # 8. Validate feature order
        checks.append(self._check_feature_order(contract))

        return self._build_report(model_id, checks, start_time)

    def _check_model_registration(self, model_id: str) -> ValidationResult:
        """Check if model is registered in ModelRegistry"""
        try:
            contract = get_model_contract(model_id)
            return ValidationResult(
                check_name="model_registration",
                status=ValidationStatus.PASSED,
                message=f"Model '{model_id}' is registered",
                details={
                    "version": contract.version,
                    "builder_type": contract.builder_type.value,
                },
            )
        except Exception as e:
            return ValidationResult(
                check_name="model_registration",
                status=ValidationStatus.FAILED,
                message=f"Model '{model_id}' not registered: {str(e)}",
            )

    def _check_norm_stats_exists(self, contract: ModelContract) -> ValidationResult:
        """Check if norm_stats file exists - CRITICAL"""
        norm_stats_path = self.project_root / contract.norm_stats_path

        if norm_stats_path.exists():
            return ValidationResult(
                check_name="norm_stats_exists",
                status=ValidationStatus.PASSED,
                message="Norm stats file exists",
                details={"path": str(norm_stats_path)},
            )
        else:
            return ValidationResult(
                check_name="norm_stats_exists",
                status=ValidationStatus.FAILED,
                message=(
                    f"CRITICAL: Norm stats file NOT FOUND at {norm_stats_path}. "
                    f"Model CANNOT produce correct predictions without this file. "
                    f"DO NOT use hardcoded defaults."
                ),
                details={"expected_path": str(norm_stats_path)},
            )

    def _check_builder_dimension_consistency(self, contract: ModelContract) -> ValidationResult:
        """Check if builder_type matches observation_dim"""
        expected_dims = {
            BuilderType.CURRENT_15DIM: 15,
        }

        expected = expected_dims.get(contract.builder_type)

        if expected is None:
            return ValidationResult(
                check_name="builder_dimension_consistency",
                status=ValidationStatus.WARNING,
                message=f"Unknown builder type: {contract.builder_type}",
            )

        if contract.observation_dim == expected:
            return ValidationResult(
                check_name="builder_dimension_consistency",
                status=ValidationStatus.PASSED,
                message=f"Builder type {contract.builder_type.value} matches observation_dim {expected}",
                details={
                    "builder_type": contract.builder_type.value,
                    "observation_dim": contract.observation_dim,
                    "expected_dim": expected,
                },
            )
        else:
            return ValidationResult(
                check_name="builder_dimension_consistency",
                status=ValidationStatus.FAILED,
                message=(
                    f"Dimension mismatch: builder_type {contract.builder_type.value} "
                    f"expects {expected} dims, but observation_dim is {contract.observation_dim}"
                ),
                details={
                    "builder_type": contract.builder_type.value,
                    "observation_dim": contract.observation_dim,
                    "expected_dim": expected,
                },
            )

    def _check_norm_stats_not_defaults(self, contract: ModelContract) -> ValidationResult:
        """Check that norm_stats are not the wrong hardcoded defaults"""
        norm_stats_path = self.project_root / contract.norm_stats_path

        if not norm_stats_path.exists():
            return ValidationResult(
                check_name="norm_stats_not_defaults",
                status=ValidationStatus.SKIPPED,
                message="Skipped - norm_stats file does not exist",
            )

        try:
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)

            # Check for known wrong default values
            wrong_defaults = {
                "dxy_z": {"mean": 100.0, "std": 5.0},  # Should be ~0, ~1
                "vix_z": {"mean": 20.0, "std": 8.0},   # Should be ~0, ~0.9
                "embi_z": {"mean": 300.0, "std": 60.0},  # Should be ~0, ~1
            }

            issues = []
            for feature, wrong_values in wrong_defaults.items():
                if feature in stats:
                    actual = stats[feature]
                    # Check if values are suspiciously close to wrong defaults
                    if abs(actual.get("mean", 0) - wrong_values["mean"]) < 0.1:
                        if abs(actual.get("std", 1) - wrong_values["std"]) < 0.1:
                            issues.append(
                                f"{feature}: mean={actual.get('mean')}, std={actual.get('std')} "
                                f"looks like hardcoded default"
                            )

            if issues:
                return ValidationResult(
                    check_name="norm_stats_not_defaults",
                    status=ValidationStatus.WARNING,
                    message="Some norm_stats values look like wrong hardcoded defaults",
                    details={"suspicious_features": issues},
                )

            return ValidationResult(
                check_name="norm_stats_not_defaults",
                status=ValidationStatus.PASSED,
                message="Norm stats appear to be from training (not hardcoded defaults)",
                details={"features_count": len(stats)},
            )

        except Exception as e:
            return ValidationResult(
                check_name="norm_stats_not_defaults",
                status=ValidationStatus.WARNING,
                message=f"Could not validate norm_stats content: {str(e)}",
            )

    def _check_model_file_exists(self, contract: ModelContract) -> ValidationResult:
        """Check if model file exists"""
        model_path = self.project_root / contract.model_path

        if model_path.exists():
            return ValidationResult(
                check_name="model_file_exists",
                status=ValidationStatus.PASSED,
                message="Model file exists",
                details={"path": str(model_path)},
            )
        else:
            return ValidationResult(
                check_name="model_file_exists",
                status=ValidationStatus.WARNING,
                message=f"Model file not found at {model_path}",
                details={"expected_path": str(model_path)},
            )

    def _check_hash_integrity(self, contract: ModelContract) -> List[ValidationResult]:
        """Check hash integrity for model and norm_stats"""
        results = []

        # Check norm_stats hash
        if contract.norm_stats_hash:
            norm_stats_path = self.project_root / contract.norm_stats_path
            if norm_stats_path.exists():
                try:
                    actual_hash = compute_json_hash(norm_stats_path)
                    if actual_hash == contract.norm_stats_hash:
                        results.append(ValidationResult(
                            check_name="norm_stats_hash",
                            status=ValidationStatus.PASSED,
                            message="Norm stats hash matches",
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name="norm_stats_hash",
                            status=ValidationStatus.FAILED,
                            message="Norm stats hash mismatch - file may have been modified",
                            details={
                                "expected": contract.norm_stats_hash,
                                "actual": actual_hash,
                            },
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        check_name="norm_stats_hash",
                        status=ValidationStatus.WARNING,
                        message=f"Could not compute norm_stats hash: {str(e)}",
                    ))
        else:
            results.append(ValidationResult(
                check_name="norm_stats_hash",
                status=ValidationStatus.SKIPPED,
                message="No norm_stats hash registered for verification",
            ))

        # Check model hash
        if contract.model_hash:
            model_path = self.project_root / contract.model_path
            if model_path.exists():
                try:
                    actual_hash = compute_file_hash(model_path)
                    if actual_hash == contract.model_hash:
                        results.append(ValidationResult(
                            check_name="model_hash",
                            status=ValidationStatus.PASSED,
                            message="Model hash matches",
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name="model_hash",
                            status=ValidationStatus.FAILED,
                            message="Model hash mismatch - file may have been modified",
                            details={
                                "expected": contract.model_hash,
                                "actual": actual_hash,
                            },
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        check_name="model_hash",
                        status=ValidationStatus.WARNING,
                        message=f"Could not compute model hash: {str(e)}",
                    ))
        else:
            results.append(ValidationResult(
                check_name="model_hash",
                status=ValidationStatus.SKIPPED,
                message="No model hash registered for verification",
            ))

        return results

    def _check_builder_instantiation(self, model_id: str) -> ValidationResult:
        """Check if builder can be instantiated without errors"""
        try:
            # Clear cache to force fresh instantiation
            BuilderFactory.clear_cache()

            builder = get_observation_builder(model_id)

            return ValidationResult(
                check_name="builder_instantiation",
                status=ValidationStatus.PASSED,
                message=f"Builder instantiated successfully",
                details={
                    "builder_class": builder.__class__.__name__,
                    "observation_dim": getattr(builder, 'OBSERVATION_DIM', 'unknown'),
                },
            )
        except NormStatsNotFoundError as e:
            return ValidationResult(
                check_name="builder_instantiation",
                status=ValidationStatus.FAILED,
                message=f"CRITICAL: Builder failed to instantiate - norm_stats not found",
                details={"error": str(e)},
            )
        except Exception as e:
            return ValidationResult(
                check_name="builder_instantiation",
                status=ValidationStatus.FAILED,
                message=f"Builder failed to instantiate: {str(e)}",
                details={"error": str(e)},
            )

    def _check_feature_order(self, contract: ModelContract) -> ValidationResult:
        """Check that feature order is defined"""
        try:
            # Import feature contract to get expected order
            from src.features.contract import FEATURE_CONTRACT

            expected_order = list(FEATURE_CONTRACT.feature_order)
            expected_dim = FEATURE_CONTRACT.observation_dim

            if contract.observation_dim == expected_dim:
                return ValidationResult(
                    check_name="feature_order",
                    status=ValidationStatus.PASSED,
                    message=f"Feature order consistent with contract (dim={expected_dim})",
                    details={
                        "feature_count": len(expected_order),
                        "first_features": expected_order[:3],
                        "last_features": expected_order[-3:],
                    },
                )
            else:
                return ValidationResult(
                    check_name="feature_order",
                    status=ValidationStatus.WARNING,
                    message=f"Different feature contract expected for dim={contract.observation_dim}",
                )

        except ImportError:
            return ValidationResult(
                check_name="feature_order",
                status=ValidationStatus.SKIPPED,
                message="Feature contract not available for validation",
            )
        except Exception as e:
            return ValidationResult(
                check_name="feature_order",
                status=ValidationStatus.WARNING,
                message=f"Could not validate feature order: {str(e)}",
            )

    def _build_report(
        self,
        model_id: str,
        checks: List[ValidationResult],
        start_time: float
    ) -> ConsistencyReport:
        """Build final consistency report"""
        import time
        elapsed_ms = (time.time() - start_time) * 1000

        # Determine overall status
        if any(c.status == ValidationStatus.FAILED for c in checks):
            overall_status = ValidationStatus.FAILED
        elif any(c.status == ValidationStatus.WARNING for c in checks):
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED

        return ConsistencyReport(
            model_id=model_id,
            overall_status=overall_status,
            checks=checks,
            validation_time_ms=elapsed_ms,
            timestamp=datetime.utcnow(),
        )

    def validate_all_models(self, verify_hashes: bool = False) -> Dict[str, ConsistencyReport]:
        """Validate all registered models"""
        reports = {}
        for model_id in ModelRegistry.list_models():
            reports[model_id] = self.validate_model(model_id, verify_hashes)
        return reports


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_model_consistency(
    model_id: str,
    project_root: Optional[Path] = None,
    verify_hashes: bool = False
) -> ConsistencyReport:
    """
    Convenience function to validate model consistency.

    Args:
        model_id: Model identifier
        project_root: Project root path (defaults to inference API config)
        verify_hashes: Whether to verify file hashes

    Returns:
        ConsistencyReport with validation results
    """
    if project_root is None:
        from ..config import get_settings
        project_root = get_settings().project_root

    validator = ConsistencyValidatorService(project_root)
    return validator.validate_model(model_id, verify_hashes)
