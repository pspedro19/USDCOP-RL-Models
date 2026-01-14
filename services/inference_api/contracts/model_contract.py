"""
Model Contract - Single Source of Truth for Model Configurations
================================================================
SOLID Principles Applied:
- Single Responsibility: Model configuration only
- Open/Closed: New models via registry, not code changes
- Dependency Inversion: Depends on abstractions

Design Patterns:
- Registry Pattern: Explicit model registration
- Factory Pattern: Model configuration factory
- Immutable Value Objects: Frozen dataclasses

CRITICAL: This contract enforces:
1. Fail-fast on missing norm_stats (no wrong defaults)
2. Hash verification for model/stats integrity
3. Explicit builder type (no string matching)
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Final
import logging

logger = logging.getLogger(__name__)


class BuilderType(Enum):
    """
    Explicit builder types - NO string matching.
    Each model version must declare its builder type.
    """
    CURRENT_15DIM = "current_15dim"  # Current 15-dimensional builder


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

    def __post_init__(self):
        # Validate observation_dim matches builder_type
        expected_dims = {
            BuilderType.CURRENT_15DIM: 15,
        }
        expected = expected_dims.get(self.builder_type)
        if expected and self.observation_dim != expected:
            raise ValueError(
                f"Observation dim mismatch: {self.model_id} has builder_type "
                f"{self.builder_type.value} (expects {expected} dims) but "
                f"observation_dim is {self.observation_dim}"
            )


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_json_hash(file_path: Path) -> str:
    """Compute SHA256 hash of JSON content (normalized)"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Normalize JSON to ensure consistent hashing
    normalized = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode()).hexdigest()


# =============================================================================
# Model Registry - Explicit Registration
# =============================================================================

class ModelRegistry:
    """
    Registry of all known model contracts.

    Design Principles:
    - Explicit registration (not convention-based discovery)
    - Fail-fast on unknown models
    - Hash verification for integrity
    """

    _contracts: Dict[str, ModelContract] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, contract: ModelContract) -> None:
        """Register a model contract"""
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
        """List all registered models"""
        cls._ensure_initialized()
        return cls._contracts.copy()

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure registry is initialized with default models"""
        if not cls._initialized:
            cls._initialize_defaults()
            cls._initialized = True

    @classmethod
    def _initialize_defaults(cls) -> None:
        """Initialize with known production models"""
        # Primary Production Model (15-dim)
        cls.register(ModelContract(
            model_id="ppo_primary",
            version="1.0.0",
            builder_type=BuilderType.CURRENT_15DIM,
            observation_dim=15,
            norm_stats_path="config/norm_stats.json",
            model_path="models/ppo_production/final_model.zip",
            description="PPO Primary Production - 15 features with macro indicators",
        ))

        # Load any dynamic contracts from config/contracts/
        cls._load_dynamic_contracts()

    @classmethod
    def _load_dynamic_contracts(cls) -> None:
        """Load dynamically generated contracts from config/contracts/"""
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
                model_id = f"ppo_{version}"

                # Skip if already registered
                if model_id in cls._contracts:
                    continue

                # Determine builder type from observation_dim
                obs_dim = data.get("observation_dim", 15)
                # Only current (15-dim) builder supported
                builder_type = BuilderType.CURRENT_15DIM

                # Create and register contract
                contract = ModelContract(
                    model_id=model_id,
                    version=version,
                    builder_type=builder_type,
                    observation_dim=obs_dim,
                    norm_stats_path=data.get("norm_stats_path", "config/norm_stats.json"),
                    model_path=data.get("model_path", f"models/ppo_{version}_production/final_model.zip"),
                    description=f"Dynamic contract {version} ({obs_dim} dims)",
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
# Convenience Functions
# =============================================================================

def get_model_contract(model_id: str) -> ModelContract:
    """Get model contract by ID"""
    return ModelRegistry.get(model_id)


def get_builder_type(model_id: str) -> BuilderType:
    """Get builder type for a model"""
    contract = ModelRegistry.get(model_id)
    return contract.builder_type


# Export constants
DEFAULT_MODEL_ID: Final = "ppo_primary"
