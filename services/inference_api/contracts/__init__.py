"""
Contracts Module
================
Single Source of Truth for all contracts in the inference API.

Exports:
- ModelContract: Model configuration contract
- ModelRegistry: Registry of all model contracts
- ContractValidator: Runtime validation service
- BuilderType: Explicit builder type enum
"""

from .model_contract import (
    # Types
    ModelContract,
    BuilderType,
    # Registry
    ModelRegistry,
    # Validator
    ContractValidator,
    # Exceptions
    ModelContractError,
    NormStatsNotFoundError,
    HashVerificationError,
    BuilderNotRegisteredError,
    # Functions
    get_model_contract,
    get_builder_type,
    compute_file_hash,
    compute_json_hash,
    # Constants
    DEFAULT_MODEL_ID,
)

__all__ = [
    "ModelContract",
    "BuilderType",
    "ModelRegistry",
    "ContractValidator",
    "ModelContractError",
    "NormStatsNotFoundError",
    "HashVerificationError",
    "BuilderNotRegisteredError",
    "get_model_contract",
    "get_builder_type",
    "compute_file_hash",
    "compute_json_hash",
    "DEFAULT_MODEL_ID",
]
