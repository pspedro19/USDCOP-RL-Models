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
    # Constants
    DEFAULT_MODEL_ID,
    BuilderNotRegisteredError,
    BuilderType,
    # Validator
    ContractValidator,
    HashVerificationError,
    # Types
    ModelContract,
    # Exceptions
    ModelContractError,
    # Registry
    ModelRegistry,
    NormStatsNotFoundError,
    compute_file_hash,
    compute_json_hash,
    get_builder_type,
    # Functions
    get_model_contract,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "BuilderNotRegisteredError",
    "BuilderType",
    "ContractValidator",
    "HashVerificationError",
    "ModelContract",
    "ModelContractError",
    "ModelRegistry",
    "NormStatsNotFoundError",
    "compute_file_hash",
    "compute_json_hash",
    "get_builder_type",
    "get_model_contract",
]
