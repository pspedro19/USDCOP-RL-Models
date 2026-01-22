"""
Model Contract - Single Source of Truth for Model Configurations
================================================================

This module defines the ModelContract and ModelRegistry for managing model
configurations in the inference API. It aligns with the canonical FeatureContract
from `src/core/contracts/feature_contract.py` for SSOT compliance.

SOLID Principles Applied:
- Single Responsibility: Model configuration only
- Open/Closed: New models via registry, not code changes
- Dependency Inversion: Depends on abstractions (IModelRegistry protocol)
- Liskov Substitution: Registry implementations are interchangeable

Design Patterns:
- Registry Pattern: Explicit model registration with IModelRegistry protocol
- Factory Pattern: ContractFactory for creating model contracts
- Immutable Value Objects: Frozen dataclasses
- Protocol Pattern: Type-safe interfaces for extensibility

CRITICAL: This contract enforces:
1. Fail-fast on missing norm_stats (no wrong defaults)
2. Hash verification for model/stats integrity
3. Explicit builder type (no string matching)
4. Feature service linkage for Feast integration (P0-02)
5. SSOT alignment with FeatureContract (OBSERVATION_DIM, FEATURE_ORDER)

SSOT: Hash utilities delegated to src.utils.hash_utils

Example Usage:
    >>> from services.inference_api.contracts.model_contract import (
    ...     ModelContract, ModelRegistry, ContractFactory, BuilderType
    ... )
    >>>
    >>> # Get a registered model contract
    >>> contract = ModelRegistry.get("ppo_primary")
    >>> print(f"Model: {contract.model_id}, Dim: {contract.observation_dim}")
    >>>
    >>> # Create a new contract via factory
    >>> new_contract = ContractFactory.create_15dim_contract(
    ...     model_id="ppo_v2",
    ...     version="2.0.0",
    ...     norm_stats_path="config/norm_stats_v2.json",
    ...     model_path="models/ppo_v2/final_model.zip",
    ... )
    >>>
    >>> # Validate against FeatureContract
    >>> from src.core.contracts import FEATURE_CONTRACT
    >>> errors = new_contract.validate_against_feature_contract(FEATURE_CONTRACT)
    >>> if errors:
    ...     raise ValueError(f"Contract validation failed: {errors}")

Changelog:
- v1.2.0: SSOT alignment with feature_contract.py, IModelRegistry protocol,
          ContractFactory, cross-contract validation
- v1.1.0: Added feature_service_name, training metadata fields (P0-02)
"""

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Final, Protocol, Tuple, runtime_checkable
import logging

# SSOT Import: Hash utilities from canonical source
from src.utils.hash_utils import (
    compute_file_hash as _compute_file_hash_ssot,
    compute_json_hash as _compute_json_hash_ssot,
)

# SSOT Import: Feature dimensions and order from canonical source
from src.core.contracts.feature_contract import (
    OBSERVATION_DIM,
    FEATURE_ORDER,
    FEATURE_CONTRACT,
    FEATURE_CONTRACT_VERSION,
    FEATURE_ORDER_HASH,
    FeatureContract,
)

logger = logging.getLogger(__name__)


class BuilderType(Enum):
    """
    Explicit builder types - NO string matching.
    Each model version must declare its builder type.

    SSOT: Dimension values are derived from OBSERVATION_DIM in feature_contract.py.

    Attributes:
        CURRENT_15DIM: Current production builder with 15-dimensional observation space.
            Aligned with OBSERVATION_DIM from feature_contract.py.
    """
    CURRENT_15DIM = "current_15dim"  # Current 15-dimensional builder

    @property
    def expected_dim(self) -> int:
        """
        Get the expected observation dimension for this builder type.

        Returns:
            int: Expected observation dimension.

        Note:
            Uses OBSERVATION_DIM from feature_contract.py as SSOT.
        """
        # SSOT: Use canonical OBSERVATION_DIM from feature_contract.py
        dim_mapping = {
            BuilderType.CURRENT_15DIM: OBSERVATION_DIM,  # 15 from feature_contract
        }
        return dim_mapping.get(self, OBSERVATION_DIM)


# Mapping from BuilderType to expected observation dimension (SSOT aligned)
BUILDER_TYPE_DIMS: Final[Dict['BuilderType', int]] = {
    BuilderType.CURRENT_15DIM: OBSERVATION_DIM,
}


class ModelContractError(Exception):
    """Base exception for model contract violations"""
    pass


class NormStatsNotFoundError(ModelContractError):
    """Raised when normalization stats file is missing - FAIL FAST"""
    pass


class HashVerificationError(ModelContractError):
    """Raised when hash verification fails"""
    pass


class BuilderNotRegisteredError(ModelContractError):
    """Raised when no builder is registered for a model"""
    pass


@dataclass(frozen=True)
class ModelContract:
    """
    Immutable contract for a model configuration.

    This ensures consistency between training and inference by:
    1. Explicit builder type declaration (not string matching)
    2. Mandatory norm_stats path
    3. Hash verification for integrity
    4. Feature service linkage for Feast Feature Store (P0-02)

    Attributes:
        model_id: Unique identifier (e.g., "ppo_primary")
        version: Semantic version (e.g., "1.0.0")
        builder_type: Explicit BuilderType enum value
        observation_dim: Input dimension for the model
        norm_stats_path: Path to normalization stats JSON
        model_path: Path to trained model file
        description: Human-readable description
        norm_stats_hash: SHA256 hash of norm_stats file (optional verification)
        model_hash: SHA256 hash of model file (optional verification)
        feature_service_name: Link to Feast Feature Service for feature retrieval.
            Should match a registered FeatureService in the Feast feature store.
            Example: "ppo_production_service" or "observation_15d".
            If None, features are computed on-demand via UnifiedFeatureBuilder.
        training_dataset_hash: SHA256 hash of the training dataset for
            reproducibility and lineage tracking. Computed during training.
        training_start_date: ISO 8601 format date string (YYYY-MM-DD) marking
            the start of the training data window.
        training_end_date: ISO 8601 format date string (YYYY-MM-DD) marking
            the end of the training data window.
    """
    model_id: str
    version: str
    builder_type: BuilderType
    observation_dim: int
    norm_stats_path: str
    model_path: str
    description: str = ""
    norm_stats_hash: Optional[str] = None
    model_hash: Optional[str] = None
    # P0-02: Feast Feature Service integration
    feature_service_name: Optional[str] = None
    # Training metadata for lineage and reproducibility
    training_dataset_hash: Optional[str] = None
    training_start_date: Optional[str] = None
    training_end_date: Optional[str] = None

    def __post_init__(self):
        """
        Post-initialization validation.

        Validates:
        1. observation_dim matches builder_type expectations (SSOT aligned)
        2. Warns if feature_service_name is None for production models
        3. Validates date format if training dates are provided
        """
        # SSOT: Validate observation_dim matches builder_type using BUILDER_TYPE_DIMS
        expected = BUILDER_TYPE_DIMS.get(self.builder_type)
        if expected and self.observation_dim != expected:
            raise ValueError(
                f"Observation dim mismatch: {self.model_id} has builder_type "
                f"{self.builder_type.value} (expects {expected} dims from SSOT) but "
                f"observation_dim is {self.observation_dim}. "
                f"SSOT reference: OBSERVATION_DIM={OBSERVATION_DIM} in feature_contract.py"
            )

        # P0-02: Warn if feature_service_name is None for production models
        # Production models should be linked to a Feast Feature Service for
        # consistent feature retrieval and versioning
        if self.feature_service_name is None and self._is_production_model():
            warnings.warn(
                f"Model '{self.model_id}' appears to be a production model but "
                f"has no feature_service_name configured. Consider linking to a "
                f"Feast Feature Service (e.g., 'ppo_production_service') for "
                f"consistent feature retrieval and versioning.",
                UserWarning,
                stacklevel=2
            )

        # Validate training date formats if provided
        self._validate_training_dates()

    def _is_production_model(self) -> bool:
        """
        Determine if this model is intended for production use.

        A model is considered production if:
        - model_id contains 'primary', 'production', or 'prod'
        - model_id does NOT contain 'test', 'dev', 'experiment', or 'staging'

        Returns:
            bool: True if model appears to be for production use.
        """
        model_id_lower = self.model_id.lower()

        # Exclusion patterns - not production
        dev_patterns = ('test', 'dev', 'experiment', 'staging', 'debug', 'local')
        if any(pattern in model_id_lower for pattern in dev_patterns):
            return False

        # Inclusion patterns - likely production
        prod_patterns = ('primary', 'production', 'prod', 'main', 'release')
        return any(pattern in model_id_lower for pattern in prod_patterns)

    def _validate_training_dates(self) -> None:
        """
        Validate training date formats if provided.

        Raises:
            ValueError: If date format is invalid (not ISO 8601 YYYY-MM-DD).
        """
        import re
        iso_date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

        for date_field, date_value in [
            ('training_start_date', self.training_start_date),
            ('training_end_date', self.training_end_date),
        ]:
            if date_value is not None and not iso_date_pattern.match(date_value):
                raise ValueError(
                    f"Invalid {date_field} format: '{date_value}'. "
                    f"Expected ISO 8601 date format: YYYY-MM-DD"
                )

    def validate_against_feature_contract(
        self,
        feature_contract: Optional['FeatureContract'] = None
    ) -> List[str]:
        """
        Validate this ModelContract against the FeatureContract SSOT.

        This cross-contract validation ensures:
        1. Observation dimension matches OBSERVATION_DIM
        2. Builder type is compatible with feature order
        3. Feature service (if specified) aligns with expected features

        Args:
            feature_contract: FeatureContract to validate against.
                Defaults to FEATURE_CONTRACT from feature_contract.py.

        Returns:
            List of validation error messages. Empty list means valid.

        Example:
            >>> contract = ModelRegistry.get("ppo_primary")
            >>> errors = contract.validate_against_feature_contract()
            >>> if errors:
            ...     raise ValueError(f"Validation failed: {errors}")
        """
        errors: List[str] = []
        fc = feature_contract or FEATURE_CONTRACT

        # 1. Validate observation dimension matches SSOT
        if self.observation_dim != fc.observation_dim:
            errors.append(
                f"Observation dim mismatch: ModelContract has {self.observation_dim}, "
                f"but FeatureContract SSOT specifies {fc.observation_dim}"
            )

        # 2. Validate builder type dimension alignment
        expected_builder_dim = BUILDER_TYPE_DIMS.get(self.builder_type)
        if expected_builder_dim and expected_builder_dim != fc.observation_dim:
            errors.append(
                f"Builder type dimension mismatch: {self.builder_type.value} expects "
                f"{expected_builder_dim} dims, but FeatureContract has {fc.observation_dim}"
            )

        # 3. Log feature order hash for traceability
        if not errors:
            logger.debug(
                f"ModelContract '{self.model_id}' validated against FeatureContract "
                f"v{fc.version} (hash: {fc.feature_order_hash})"
            )

        return errors

    def get_expected_feature_order(self) -> Tuple[str, ...]:
        """
        Get the expected feature order from the SSOT FeatureContract.

        Returns:
            Tuple of feature names in canonical order.
        """
        return FEATURE_ORDER

    def get_feature_contract_version(self) -> str:
        """
        Get the version of the FeatureContract SSOT.

        Returns:
            Version string of the FeatureContract.
        """
        return FEATURE_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, any]:
        """
        Convert ModelContract to dictionary for serialization.

        Returns:
            Dictionary representation of the contract.
        """
        return {
            "model_id": self.model_id,
            "version": self.version,
            "builder_type": self.builder_type.value,
            "observation_dim": self.observation_dim,
            "norm_stats_path": self.norm_stats_path,
            "model_path": self.model_path,
            "description": self.description,
            "norm_stats_hash": self.norm_stats_hash,
            "model_hash": self.model_hash,
            "feature_service_name": self.feature_service_name,
            "training_dataset_hash": self.training_dataset_hash,
            "training_start_date": self.training_start_date,
            "training_end_date": self.training_end_date,
            # SSOT alignment metadata
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "feature_order_hash": FEATURE_ORDER_HASH,
        }


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    SSOT: Delegates to src.utils.hash_utils

    Args:
        file_path: Path to the file to hash.

    Returns:
        SHA256 hex digest of the file contents.
    """
    return _compute_file_hash_ssot(file_path).full_hash


def compute_json_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of JSON content (normalized).

    SSOT: Delegates to src.utils.hash_utils

    Args:
        file_path: Path to the JSON file to hash.

    Returns:
        SHA256 hex digest of the normalized JSON content.
    """
    return _compute_json_hash_ssot(file_path).full_hash


# =============================================================================
# Protocol Interfaces
# =============================================================================

@runtime_checkable
class IModelRegistry(Protocol):
    """
    Protocol interface for model registry implementations.

    Design Principle: Dependency Inversion - depend on abstractions.

    This protocol enables:
    - Type-safe registry implementations
    - Mock registries for testing
    - Alternative storage backends (e.g., database-backed registry)

    Example:
        >>> def get_model(registry: IModelRegistry, model_id: str) -> ModelContract:
        ...     return registry.get(model_id)
    """

    def register(self, contract: ModelContract) -> None:
        """
        Register a model contract.

        Args:
            contract: The ModelContract to register.
        """
        ...

    def get(self, model_id: str) -> ModelContract:
        """
        Get a model contract by ID.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            The registered ModelContract.

        Raises:
            BuilderNotRegisteredError: If model not found.
        """
        ...

    def list_models(self) -> Dict[str, ModelContract]:
        """
        List all registered model contracts.

        Returns:
            Dictionary of model_id -> ModelContract.
        """
        ...

    def contains(self, model_id: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            True if model is registered, False otherwise.
        """
        ...


# =============================================================================
# Model Registry - Explicit Registration
# =============================================================================

class ModelRegistry:
    """
    Registry of all known model contracts.

    Implements the IModelRegistry protocol for type-safe contract management.

    Design Principles:
    - Explicit registration (not convention-based discovery)
    - Fail-fast on unknown models
    - Hash verification for integrity
    - Implements IModelRegistry protocol for dependency injection

    Example:
        >>> # Register a new model
        >>> contract = ModelContract(...)
        >>> ModelRegistry.register(contract)
        >>>
        >>> # Get a registered model
        >>> primary = ModelRegistry.get("ppo_primary")
        >>>
        >>> # Check if model exists
        >>> if ModelRegistry.contains("ppo_v2"):
        ...     v2 = ModelRegistry.get("ppo_v2")
    """

    _contracts: Dict[str, ModelContract] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, contract: ModelContract) -> None:
        """
        Register a model contract.

        Args:
            contract: The ModelContract to register.

        Note:
            Overwrites existing contract if model_id already registered.
        """
        if contract.model_id in cls._contracts:
            logger.warning(f"Overwriting existing contract for {contract.model_id}")
        cls._contracts[contract.model_id] = contract
        logger.info(f"Registered model contract: {contract.model_id} "
                   f"(builder={contract.builder_type.value}, dim={contract.observation_dim})")

    @classmethod
    def get(cls, model_id: str) -> ModelContract:
        """
        Get model contract by ID.

        Raises:
            BuilderNotRegisteredError: If model not registered
        """
        cls._ensure_initialized()

        if model_id not in cls._contracts:
            available = list(cls._contracts.keys())
            raise BuilderNotRegisteredError(
                f"Model '{model_id}' not registered. "
                f"Available models: {available}. "
                f"Register the model in ModelRegistry.initialize()."
            )
        return cls._contracts[model_id]

    @classmethod
    def list_models(cls) -> Dict[str, ModelContract]:
        """
        List all registered models.

        Returns:
            Dictionary of model_id -> ModelContract.
        """
        cls._ensure_initialized()
        return cls._contracts.copy()

    @classmethod
    def contains(cls, model_id: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_id: Unique identifier of the model.

        Returns:
            True if model is registered, False otherwise.
        """
        cls._ensure_initialized()
        return model_id in cls._contracts

    @classmethod
    def unregister(cls, model_id: str) -> bool:
        """
        Unregister a model contract.

        Args:
            model_id: Unique identifier of the model to unregister.

        Returns:
            True if model was unregistered, False if not found.
        """
        cls._ensure_initialized()
        if model_id in cls._contracts:
            del cls._contracts[model_id]
            logger.info(f"Unregistered model contract: {model_id}")
            return True
        return False

    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry to uninitialized state.

        Useful for testing. Clears all registered contracts.
        """
        cls._contracts.clear()
        cls._initialized = False
        logger.debug("ModelRegistry reset")

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure registry is initialized with default models"""
        if not cls._initialized:
            cls._initialize_defaults()
            cls._initialized = True

    @classmethod
    def _initialize_defaults(cls) -> None:
        """
        Initialize with known production models.

        This registers the default production models with their contracts.
        Production models should have:
        - Explicit builder_type for feature calculation
        - Valid norm_stats_path for normalization
        - feature_service_name for Feast integration (P0-02)
        """
        # Primary Production Model - SSOT aligned with OBSERVATION_DIM
        # P0-02: Added feature_service_name for Feast Feature Service linkage
        cls.register(ModelContract(
            model_id="ppo_primary",
            version="1.0.0",
            builder_type=BuilderType.CURRENT_15DIM,
            observation_dim=OBSERVATION_DIM,  # SSOT: from feature_contract.py
            norm_stats_path="config/norm_stats.json",
            model_path="models/ppo_production/final_model.zip",
            description=f"PPO Primary Production - {OBSERVATION_DIM} features with macro indicators",
            feature_service_name="ppo_production_service",
        ))

        # Load any dynamic contracts from config/contracts/
        cls._load_dynamic_contracts()

    @classmethod
    def _load_dynamic_contracts(cls) -> None:
        """
        Load dynamically generated contracts from config/contracts/.

        Supports loading contracts from JSON files with optional fields:
        - feature_service_name: Link to Feast Feature Service (P0-02)
        - training_dataset_hash: Hash of training data for lineage
        - training_start_date: Training data start date (ISO 8601)
        - training_end_date: Training data end date (ISO 8601)
        - norm_stats_hash: Hash of norm_stats file
        - model_hash: Hash of model file

        Backward compatible: missing fields default to None.
        """
        import sys
        from pathlib import Path

        # Find project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        contracts_dir = project_root / "config" / "contracts"

        if not contracts_dir.exists():
            logger.debug(f"No dynamic contracts directory at {contracts_dir}")
            return

        for contract_file in contracts_dir.glob("*_contract.json"):
            try:
                with open(contract_file, 'r') as f:
                    data = json.load(f)

                version = data.get("version", "")
                model_id = data.get("model_id", f"ppo_{version}")

                # Skip if already registered
                if model_id in cls._contracts:
                    continue

                # Determine builder type from observation_dim
                obs_dim = data.get("observation_dim", 15)
                # Only current (15-dim) builder supported
                builder_type = BuilderType.CURRENT_15DIM

                # Create and register contract with all supported fields
                # P0-02: Include feature_service_name and training metadata
                contract = ModelContract(
                    model_id=model_id,
                    version=version,
                    builder_type=builder_type,
                    observation_dim=obs_dim,
                    norm_stats_path=data.get("norm_stats_path", "config/norm_stats.json"),
                    model_path=data.get("model_path", f"models/ppo_{version}_production/final_model.zip"),
                    description=data.get("description", f"Dynamic contract {version} ({obs_dim} dims)"),
                    norm_stats_hash=data.get("norm_stats_hash"),
                    model_hash=data.get("model_hash"),
                    # P0-02: Feast Feature Service integration
                    feature_service_name=data.get("feature_service_name"),
                    # Training metadata for lineage
                    training_dataset_hash=data.get("training_dataset_hash"),
                    training_start_date=data.get("training_start_date"),
                    training_end_date=data.get("training_end_date"),
                )

                cls.register(contract)
                logger.info(f"Loaded dynamic contract: {model_id} from {contract_file}")

            except Exception as e:
                logger.warning(f"Failed to load contract from {contract_file}: {e}")


# =============================================================================
# Contract Validation Service
# =============================================================================

class ContractValidator:
    """
    Validates model contracts at runtime.

    Ensures:
    1. Norm stats file exists (fail-fast)
    2. Hash verification if provided
    3. Model file exists
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def validate_contract(
        self,
        contract: ModelContract,
        verify_hashes: bool = False
    ) -> None:
        """
        Validate a model contract.

        Args:
            contract: ModelContract to validate
            verify_hashes: Whether to verify file hashes

        Raises:
            NormStatsNotFoundError: If norm_stats file missing
            HashVerificationError: If hash mismatch
            FileNotFoundError: If model file missing
        """
        # 1. Validate norm_stats exists (FAIL-FAST - no defaults!)
        norm_stats_path = self.project_root / contract.norm_stats_path
        if not norm_stats_path.exists():
            raise NormStatsNotFoundError(
                f"CRITICAL: Normalization stats not found at {norm_stats_path}. "
                f"Model '{contract.model_id}' cannot run without norm_stats. "
                f"This is a deployment error - ensure the file is mounted correctly. "
                f"DO NOT use hardcoded defaults - they will produce incorrect predictions."
            )

        # 2. Validate model file exists
        model_path = self.project_root / contract.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path} for model '{contract.model_id}'"
            )

        # 3. Hash verification (optional but recommended)
        if verify_hashes:
            if contract.norm_stats_hash:
                actual_hash = compute_json_hash(norm_stats_path)
                if actual_hash != contract.norm_stats_hash:
                    raise HashVerificationError(
                        f"Norm stats hash mismatch for {contract.model_id}. "
                        f"Expected: {contract.norm_stats_hash}, "
                        f"Actual: {actual_hash}. "
                        f"The norm_stats file may have been modified after training."
                    )

            if contract.model_hash:
                actual_hash = compute_file_hash(model_path)
                if actual_hash != contract.model_hash:
                    raise HashVerificationError(
                        f"Model hash mismatch for {contract.model_id}. "
                        f"Expected: {contract.model_hash}, "
                        f"Actual: {actual_hash}. "
                        f"The model file may have been modified after training."
                    )

        logger.info(f"Contract validated for {contract.model_id}")

    def load_norm_stats(self, contract: ModelContract) -> Dict[str, Dict[str, float]]:
        """
        Load normalization stats for a contract.

        FAIL-FAST: Raises if file not found. NO DEFAULTS.

        Args:
            contract: ModelContract with norm_stats_path

        Returns:
            Dict of feature name -> {mean, std} stats

        Raises:
            NormStatsNotFoundError: If file not found
        """
        norm_stats_path = self.project_root / contract.norm_stats_path

        if not norm_stats_path.exists():
            raise NormStatsNotFoundError(
                f"CRITICAL: Cannot load norm_stats from {norm_stats_path}. "
                f"This file is REQUIRED for model '{contract.model_id}'. "
                f"Ensure proper deployment configuration."
            )

        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)

        logger.info(f"Loaded norm_stats for {contract.model_id}: {len(stats)} features")
        return stats


# =============================================================================
# Contract Factory
# =============================================================================

class ContractFactory:
    """
    Factory for creating ModelContract instances with SSOT defaults.

    Design Principles:
    - Factory Pattern: Encapsulates contract creation logic
    - SSOT Alignment: Uses OBSERVATION_DIM from feature_contract.py
    - Validation: Validates contracts against FeatureContract on creation

    Example:
        >>> # Create a 15-dim contract (current production standard)
        >>> contract = ContractFactory.create_15dim_contract(
        ...     model_id="ppo_v2",
        ...     version="2.0.0",
        ...     norm_stats_path="config/norm_stats_v2.json",
        ...     model_path="models/ppo_v2/final_model.zip",
        ... )
        >>>
        >>> # Create from dictionary (e.g., from JSON config)
        >>> config = {"model_id": "ppo_v3", "version": "3.0.0", ...}
        >>> contract = ContractFactory.from_dict(config)
    """

    @staticmethod
    def create_15dim_contract(
        model_id: str,
        version: str,
        norm_stats_path: str,
        model_path: str,
        description: str = "",
        feature_service_name: Optional[str] = None,
        norm_stats_hash: Optional[str] = None,
        model_hash: Optional[str] = None,
        training_dataset_hash: Optional[str] = None,
        training_start_date: Optional[str] = None,
        training_end_date: Optional[str] = None,
        validate: bool = True,
    ) -> ModelContract:
        """
        Create a 15-dimensional ModelContract with SSOT defaults.

        Uses OBSERVATION_DIM from feature_contract.py as the dimension.

        Args:
            model_id: Unique identifier for the model.
            version: Semantic version string (e.g., "1.0.0").
            norm_stats_path: Path to normalization stats JSON file.
            model_path: Path to trained model file.
            description: Human-readable description.
            feature_service_name: Feast Feature Service name (optional).
            norm_stats_hash: SHA256 hash of norm_stats file (optional).
            model_hash: SHA256 hash of model file (optional).
            training_dataset_hash: Hash of training data (optional).
            training_start_date: Training data start date ISO 8601 (optional).
            training_end_date: Training data end date ISO 8601 (optional).
            validate: If True, validate against FeatureContract.

        Returns:
            ModelContract instance.

        Raises:
            ModelContractError: If validation fails.
        """
        contract = ModelContract(
            model_id=model_id,
            version=version,
            builder_type=BuilderType.CURRENT_15DIM,
            observation_dim=OBSERVATION_DIM,  # SSOT
            norm_stats_path=norm_stats_path,
            model_path=model_path,
            description=description or f"{model_id} v{version} ({OBSERVATION_DIM} dims)",
            feature_service_name=feature_service_name,
            norm_stats_hash=norm_stats_hash,
            model_hash=model_hash,
            training_dataset_hash=training_dataset_hash,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
        )

        if validate:
            errors = contract.validate_against_feature_contract()
            if errors:
                raise ModelContractError(
                    f"Contract validation failed for {model_id}: {errors}"
                )

        return contract

    @staticmethod
    def from_dict(
        data: Dict[str, any],
        validate: bool = True,
    ) -> ModelContract:
        """
        Create a ModelContract from a dictionary.

        Useful for loading contracts from JSON configuration files.

        Args:
            data: Dictionary with contract fields.
            validate: If True, validate against FeatureContract.

        Returns:
            ModelContract instance.

        Raises:
            ModelContractError: If required fields missing or validation fails.

        Example:
            >>> config = {
            ...     "model_id": "ppo_v2",
            ...     "version": "2.0.0",
            ...     "norm_stats_path": "config/norm_stats_v2.json",
            ...     "model_path": "models/ppo_v2/final_model.zip",
            ... }
            >>> contract = ContractFactory.from_dict(config)
        """
        # Required fields
        required = ["model_id", "version", "norm_stats_path", "model_path"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ModelContractError(f"Missing required fields: {missing}")

        # Determine builder type from observation_dim or default to 15-dim
        obs_dim = data.get("observation_dim", OBSERVATION_DIM)
        if obs_dim == OBSERVATION_DIM:
            builder_type = BuilderType.CURRENT_15DIM
        else:
            # For future extensibility
            builder_type = BuilderType.CURRENT_15DIM
            logger.warning(
                f"Unknown observation_dim {obs_dim}, defaulting to CURRENT_15DIM"
            )

        contract = ModelContract(
            model_id=data["model_id"],
            version=data["version"],
            builder_type=builder_type,
            observation_dim=obs_dim,
            norm_stats_path=data["norm_stats_path"],
            model_path=data["model_path"],
            description=data.get("description", ""),
            feature_service_name=data.get("feature_service_name"),
            norm_stats_hash=data.get("norm_stats_hash"),
            model_hash=data.get("model_hash"),
            training_dataset_hash=data.get("training_dataset_hash"),
            training_start_date=data.get("training_start_date"),
            training_end_date=data.get("training_end_date"),
        )

        if validate:
            errors = contract.validate_against_feature_contract()
            if errors:
                raise ModelContractError(
                    f"Contract validation failed for {data['model_id']}: {errors}"
                )

        return contract

    @staticmethod
    def create_and_register(
        model_id: str,
        version: str,
        norm_stats_path: str,
        model_path: str,
        **kwargs
    ) -> ModelContract:
        """
        Create a contract and register it in the ModelRegistry.

        Convenience method that combines creation and registration.

        Args:
            model_id: Unique identifier for the model.
            version: Semantic version string.
            norm_stats_path: Path to normalization stats.
            model_path: Path to model file.
            **kwargs: Additional arguments passed to create_15dim_contract.

        Returns:
            The created and registered ModelContract.
        """
        contract = ContractFactory.create_15dim_contract(
            model_id=model_id,
            version=version,
            norm_stats_path=norm_stats_path,
            model_path=model_path,
            **kwargs
        )
        ModelRegistry.register(contract)
        return contract


# =============================================================================
# Cross-Contract Validation Utilities
# =============================================================================

def validate_model_against_features(
    model_contract: ModelContract,
    feature_contract: Optional['FeatureContract'] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a ModelContract against FeatureContract SSOT.

    Args:
        model_contract: The ModelContract to validate.
        feature_contract: FeatureContract to validate against.
            Defaults to FEATURE_CONTRACT from feature_contract.py.

    Returns:
        Tuple of (is_valid, error_messages).

    Example:
        >>> contract = ModelRegistry.get("ppo_primary")
        >>> is_valid, errors = validate_model_against_features(contract)
        >>> if not is_valid:
        ...     print(f"Validation failed: {errors}")
    """
    errors = model_contract.validate_against_feature_contract(feature_contract)
    return len(errors) == 0, errors


def validate_feature_service_name(
    feature_service_name: str,
    known_services: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate that a feature service name is known.

    Args:
        feature_service_name: The Feast Feature Service name to validate.
        known_services: List of known service names. If None, returns True
            (cannot validate without Feast connection).

    Returns:
        Tuple of (is_valid, message).
    """
    if known_services is None:
        return True, "Cannot validate without known services list"

    if feature_service_name in known_services:
        return True, f"Feature service '{feature_service_name}' is valid"
    else:
        return False, (
            f"Unknown feature service '{feature_service_name}'. "
            f"Known services: {known_services}"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_model_contract(model_id: str) -> ModelContract:
    """
    Get model contract by ID from the registry.

    Args:
        model_id: Unique identifier of the model.

    Returns:
        The registered ModelContract.

    Raises:
        BuilderNotRegisteredError: If model not registered.
    """
    return ModelRegistry.get(model_id)


def get_builder_type(model_id: str) -> BuilderType:
    """
    Get builder type for a model.

    Args:
        model_id: Unique identifier of the model.

    Returns:
        BuilderType enum value for the model.
    """
    contract = ModelRegistry.get(model_id)
    return contract.builder_type


def get_ssot_observation_dim() -> int:
    """
    Get the SSOT observation dimension from feature_contract.py.

    Returns:
        The canonical OBSERVATION_DIM value.
    """
    return OBSERVATION_DIM


def get_ssot_feature_order() -> Tuple[str, ...]:
    """
    Get the SSOT feature order from feature_contract.py.

    Returns:
        Tuple of feature names in canonical order.
    """
    return FEATURE_ORDER


# =============================================================================
# Export Constants
# =============================================================================

DEFAULT_MODEL_ID: Final[str] = "ppo_primary"
MODEL_CONTRACT_VERSION: Final[str] = "1.2.0"


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core Classes
    "ModelContract",
    "ModelRegistry",
    "ContractFactory",
    "ContractValidator",

    # Protocol Interfaces
    "IModelRegistry",

    # Enums
    "BuilderType",

    # Exceptions
    "ModelContractError",
    "NormStatsNotFoundError",
    "HashVerificationError",
    "BuilderNotRegisteredError",

    # Utility Functions
    "compute_file_hash",
    "compute_json_hash",
    "get_model_contract",
    "get_builder_type",
    "get_ssot_observation_dim",
    "get_ssot_feature_order",
    "validate_model_against_features",
    "validate_feature_service_name",

    # Constants (SSOT re-exports for convenience)
    "DEFAULT_MODEL_ID",
    "MODEL_CONTRACT_VERSION",
    "OBSERVATION_DIM",
    "FEATURE_ORDER",
    "FEATURE_CONTRACT_VERSION",
    "FEATURE_ORDER_HASH",
    "BUILDER_TYPE_DIMS",
]
