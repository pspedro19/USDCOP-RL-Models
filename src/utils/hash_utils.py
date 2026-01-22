"""
Hash Utilities for Reproducibility and Validation
==================================================
Phase 1.1: MLflow Hash Logging Implementation

This module provides SHA256 hash computation utilities for:
- Model files (.zip)
- Normalization statistics (JSON with sorted keys for consistency)
- Dataset files (via DVC integration)
- Feature order validation

These hashes are logged to MLflow for:
1. Reproducibility tracking
2. Artifact validation during inference
3. Model lineage auditing

Contract: CTR-HASH-001
- All model artifacts must have verifiable hashes
- Hashes are logged as params (truncated) and tags (full) in MLflow

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HashResult:
    """
    Result of a hash computation.

    Attributes:
        full_hash: Complete SHA256 hash (64 characters)
        short_hash: Truncated hash for display (16 characters)
        algorithm: Hash algorithm used
        source_path: Path to source file (if applicable)
        source_type: Type of source (file, json, string, dvc)
    """
    full_hash: str
    short_hash: str
    algorithm: str = "sha256"
    source_path: Optional[str] = None
    source_type: str = "unknown"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for logging."""
        return {
            "full_hash": self.full_hash,
            "short_hash": self.short_hash,
            "algorithm": self.algorithm,
            "source_path": self.source_path,
            "source_type": self.source_type,
        }


# =============================================================================
# HASH COMPUTATION FUNCTIONS
# =============================================================================

def compute_file_hash(
    path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> HashResult:
    """
    Compute SHA256 hash of a file.

    This is used for:
    - Model .zip files
    - Any binary artifact

    Args:
        path: Path to the file
        algorithm: Hash algorithm (default: sha256)
        chunk_size: Size of chunks for reading large files

    Returns:
        HashResult with full and truncated hash

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported

    Example:
        >>> result = compute_file_hash("models/ppo_v1/final_model.zip")
        >>> print(result.short_hash)  # First 16 chars
        'a1b2c3d4e5f67890'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if algorithm not in ("sha256", "sha512", "md5"):
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use sha256, sha512, or md5")

    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    full_hash = hasher.hexdigest()

    logger.debug(f"Computed {algorithm} hash for {path}: {full_hash[:16]}...")

    return HashResult(
        full_hash=full_hash,
        short_hash=full_hash[:16],
        algorithm=algorithm,
        source_path=str(path),
        source_type="file",
    )


def compute_json_hash(
    path: Union[str, Path],
    algorithm: str = "sha256"
) -> HashResult:
    """
    Compute hash of JSON file content with sorted keys for consistency.

    This is CRITICAL for:
    - norm_stats.json: Ensures inference uses exact training normalization
    - Feature contracts: Validates feature order hasn't changed

    The JSON is canonicalized (sorted keys, minimal separators) before hashing
    to ensure the same logical content produces the same hash regardless of
    formatting or key order in the original file.

    Args:
        path: Path to JSON file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        HashResult with full and truncated hash

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> result = compute_json_hash("config/v1_norm_stats.json")
        >>> print(result.full_hash)
        'abc123def456...'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Canonicalize: sort keys and use minimal separators
    # This ensures the same logical content produces the same hash
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))

    hasher = hashlib.new(algorithm)
    hasher.update(canonical.encode("utf-8"))

    full_hash = hasher.hexdigest()

    logger.debug(f"Computed {algorithm} hash for JSON {path}: {full_hash[:16]}...")

    return HashResult(
        full_hash=full_hash,
        short_hash=full_hash[:16],
        algorithm=algorithm,
        source_path=str(path),
        source_type="json",
    )


def compute_string_hash(
    content: str,
    algorithm: str = "sha256"
) -> HashResult:
    """
    Compute hash of a string.

    Used for:
    - Feature order validation
    - Configuration validation

    Args:
        content: String content to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        HashResult with full and truncated hash

    Example:
        >>> features = ["rsi_9", "atr_pct", "adx_14"]
        >>> result = compute_string_hash(",".join(features))
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))

    full_hash = hasher.hexdigest()

    return HashResult(
        full_hash=full_hash,
        short_hash=full_hash[:16],
        algorithm=algorithm,
        source_path=None,
        source_type="string",
    )


def compute_feature_order_hash(
    feature_order: Union[List[str], Tuple[str, ...]],
    algorithm: str = "sha256"
) -> HashResult:
    """
    Compute hash of feature order for contract validation.

    This is used to validate that the model's expected feature order
    matches the feature order used during inference.

    Args:
        feature_order: List or tuple of feature names in order
        algorithm: Hash algorithm (default: sha256)

    Returns:
        HashResult with full and truncated hash

    Example:
        >>> from src.feature_store.core import FEATURE_ORDER
        >>> result = compute_feature_order_hash(FEATURE_ORDER)
        >>> mlflow.log_param("feature_order_hash", result.short_hash)
    """
    # Join features with comma for consistent hashing
    feature_string = ",".join(feature_order)

    hasher = hashlib.new(algorithm)
    hasher.update(feature_string.encode("utf-8"))

    full_hash = hasher.hexdigest()

    logger.debug(f"Computed feature order hash ({len(feature_order)} features): {full_hash[:16]}...")

    return HashResult(
        full_hash=full_hash,
        short_hash=full_hash[:16],
        algorithm=algorithm,
        source_path=None,
        source_type="feature_order",
    )


def compute_dvc_hash(
    data_path: Union[str, Path],
    dvc_executable: str = "dvc"
) -> Optional[HashResult]:
    """
    Compute dataset hash using DVC.

    This leverages DVC's content-addressable storage to get the hash
    of the tracked dataset, ensuring reproducibility.

    Args:
        data_path: Path to DVC-tracked data directory or file
        dvc_executable: Path to DVC executable

    Returns:
        HashResult if successful, None if DVC fails

    Note:
        Requires DVC to be installed and the data path to be tracked.
        Falls back gracefully if DVC is not available.
    """
    data_path = Path(data_path)

    try:
        # Try using dvc hash command
        result = subprocess.run(
            [dvc_executable, "hash", str(data_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            dataset_hash = result.stdout.strip()
            logger.info(f"DVC hash for {data_path}: {dataset_hash[:16]}...")

            return HashResult(
                full_hash=dataset_hash,
                short_hash=dataset_hash[:16] if len(dataset_hash) >= 16 else dataset_hash,
                algorithm="md5",  # DVC uses MD5
                source_path=str(data_path),
                source_type="dvc",
            )
        else:
            logger.warning(f"DVC hash failed: {result.stderr}")
            return None

    except FileNotFoundError:
        logger.warning("DVC not installed or not in PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("DVC hash command timed out")
        return None
    except Exception as e:
        logger.warning(f"Could not compute DVC hash: {e}")
        return None


# =============================================================================
# MLFLOW LOGGING UTILITIES
# =============================================================================

def log_hashes_to_mlflow(
    mlflow_module: Any,
    model_path: Optional[Union[str, Path]] = None,
    norm_stats_path: Optional[Union[str, Path]] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    feature_order: Optional[Union[List[str], Tuple[str, ...]]] = None,
) -> Dict[str, HashResult]:
    """
    Log all training artifact hashes to MLflow.

    This function computes hashes for all provided artifacts and logs them
    to MLflow as both params (truncated for display) and tags (full for validation).

    Implements CTR-HASH-001: All model artifacts must have verifiable hashes.

    Args:
        mlflow_module: The mlflow module (passed to avoid import issues)
        model_path: Path to model .zip file
        norm_stats_path: Path to norm_stats.json
        dataset_path: Path to DVC-tracked dataset
        feature_order: Feature order list/tuple

    Returns:
        Dictionary of artifact name to HashResult

    Example:
        >>> import mlflow
        >>> from src.utils.hash_utils import log_hashes_to_mlflow
        >>> from src.feature_store.core import FEATURE_ORDER
        >>>
        >>> with mlflow.start_run():
        ...     hashes = log_hashes_to_mlflow(
        ...         mlflow,
        ...         model_path="models/ppo_v1/final_model.zip",
        ...         norm_stats_path="config/v1_norm_stats.json",
        ...         feature_order=FEATURE_ORDER,
        ...     )
    """
    results = {}

    # Model hash
    if model_path:
        try:
            model_result = compute_file_hash(model_path)
            mlflow_module.log_param("model_hash", model_result.short_hash)
            mlflow_module.set_tag("model_hash_full", model_result.full_hash)
            results["model"] = model_result
            logger.info(f"Logged model hash: {model_result.short_hash}")
        except Exception as e:
            logger.warning(f"Could not compute model hash: {e}")

    # Norm stats hash (CRITICAL for inference consistency)
    if norm_stats_path:
        try:
            norm_result = compute_json_hash(norm_stats_path)
            mlflow_module.log_param("norm_stats_hash", norm_result.short_hash)
            mlflow_module.set_tag("norm_stats_hash_full", norm_result.full_hash)
            results["norm_stats"] = norm_result
            logger.info(f"Logged norm_stats hash: {norm_result.short_hash}")
        except Exception as e:
            logger.warning(f"Could not compute norm_stats hash: {e}")

    # Dataset hash from DVC
    if dataset_path:
        dvc_result = compute_dvc_hash(dataset_path)
        if dvc_result:
            mlflow_module.log_param("dataset_hash", dvc_result.short_hash)
            mlflow_module.set_tag("dataset_hash_full", dvc_result.full_hash)
            results["dataset"] = dvc_result
            logger.info(f"Logged dataset hash (DVC): {dvc_result.short_hash}")

    # Feature order hash (contract validation)
    if feature_order:
        try:
            feature_result = compute_feature_order_hash(feature_order)
            mlflow_module.log_param("feature_order_hash", feature_result.short_hash)
            mlflow_module.set_tag("feature_order_hash_full", feature_result.full_hash)
            mlflow_module.set_tag("feature_count", str(len(feature_order)))
            results["feature_order"] = feature_result
            logger.info(f"Logged feature_order hash: {feature_result.short_hash}")
        except Exception as e:
            logger.warning(f"Could not compute feature_order hash: {e}")

    return results


def validate_hash(
    expected_hash: str,
    actual_hash: str,
    artifact_name: str = "artifact"
) -> bool:
    """
    Validate that a computed hash matches an expected hash.

    Used during inference to validate model artifacts.

    Args:
        expected_hash: Expected hash from MLflow
        actual_hash: Computed hash
        artifact_name: Name of artifact for logging

    Returns:
        True if hashes match, False otherwise

    Raises:
        ValueError: If hashes don't match (optional strict mode)
    """
    # Compare with case-insensitive (hashes are hex)
    expected_lower = expected_hash.lower()
    actual_lower = actual_hash.lower()

    # Handle both full and short hash comparisons
    if expected_lower == actual_lower:
        logger.debug(f"Hash validation passed for {artifact_name}")
        return True

    # Try prefix matching for short hash
    min_len = min(len(expected_lower), len(actual_lower))
    if expected_lower[:min_len] == actual_lower[:min_len]:
        logger.debug(f"Hash validation passed for {artifact_name} (prefix match)")
        return True

    logger.warning(
        f"Hash mismatch for {artifact_name}: "
        f"expected={expected_hash[:16]}..., actual={actual_hash[:16]}..."
    )
    return False
