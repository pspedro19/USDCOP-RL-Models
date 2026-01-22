#!/usr/bin/env python3
"""
Dataset Reproduction from MLflow Run ID.

This script enables complete reproduction of training datasets from MLflow run metadata.
It validates data integrity through hash verification and ensures reproducibility for
model auditing and debugging purposes.

Implements P0-03 from the remediation plan:
- Extract dataset metadata from MLflow runs
- Checkout DVC-versioned data
- Verify dataset hashes for integrity
- Validate feature order consistency

Contract: CTR-REPRO-001
- All reproduced datasets must match original hashes
- Feature order must be validated against run metadata
- Complete reproduction metadata saved alongside output

Usage:
    python scripts/reproduce_dataset_from_run.py --run-id <run_id> --output <output_dir>
    python scripts/reproduce_dataset_from_run.py --run-id abc123 --output ./reproduced/ --skip-hash-verify

Author: Trading Team
Date: 2026-01-17
Version: 1.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Version info
__version__ = "1.1.0"
__author__ = "Trading Team"

# Import constants from centralized location
# These are imported lazily in methods that need them to avoid import errors
# when feature_contract module is not available
try:
    from src.core.constants import (
        MAX_API_TIMEOUT,
        DEFAULT_API_TIMEOUT,
    )
except ImportError:
    # Fallback defaults if constants module is not available
    MAX_API_TIMEOUT = 300
    DEFAULT_API_TIMEOUT = 60

# DVC timeout uses max API timeout constant
DVC_TIMEOUT_SECONDS: int = MAX_API_TIMEOUT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXIT CODES
# =============================================================================

EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1
EXIT_VALIDATION_ERROR: int = 2
EXIT_INTERRUPTED: int = 130


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class DatasetReproductionError(Exception):
    """Base exception for all dataset reproduction errors."""

    def __init__(self, message: str, run_id: Optional[str] = None) -> None:
        self.run_id = run_id
        super().__init__(message)


class RunNotFoundError(DatasetReproductionError):
    """Raised when the specified MLflow run cannot be found."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"MLflow run not found: {run_id}", run_id=run_id)


class HashMismatchError(DatasetReproductionError):
    """Raised when dataset hash does not match expected value."""

    def __init__(
        self, expected: str, computed: str, run_id: Optional[str] = None
    ) -> None:
        self.expected = expected
        self.computed = computed
        super().__init__(
            f"Hash mismatch: expected={expected[:16]}..., computed={computed[:16]}...",
            run_id=run_id,
        )


class FeatureOrderMismatchError(DatasetReproductionError):
    """Raised when feature order does not match expected value."""

    def __init__(
        self, expected: str, current: str, run_id: Optional[str] = None
    ) -> None:
        self.expected = expected
        self.current = current
        super().__init__(
            f"Feature order mismatch: expected={expected}, current={current}",
            run_id=run_id,
        )


class DVCCheckoutError(DatasetReproductionError):
    """Raised when DVC checkout operation fails."""

    def __init__(self, message: str, dataset_path: Optional[str] = None) -> None:
        self.dataset_path = dataset_path
        super().__init__(f"DVC checkout failed: {message}")


class DataLoadError(DatasetReproductionError):
    """Raised when data loading fails."""

    def __init__(self, message: str, data_path: Optional[Path] = None) -> None:
        self.data_path = data_path
        super().__init__(f"Data load failed: {message}")


class MLflowConnectionError(DatasetReproductionError):
    """Raised when connection to MLflow server fails."""

    def __init__(self, tracking_uri: str, cause: Optional[str] = None) -> None:
        self.tracking_uri = tracking_uri
        msg = f"Cannot connect to MLflow at {tracking_uri}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


# =============================================================================
# PROTOCOLS FOR DEPENDENCY INJECTION
# =============================================================================


class MLflowClientProtocol(Protocol):
    """Protocol defining the MLflow client interface for dependency injection."""

    def get_run(self, run_id: str) -> Any:
        """Get a run by ID."""
        ...

    def get_experiment(self, experiment_id: str) -> Any:
        """Get an experiment by ID."""
        ...

    def search_experiments(self, max_results: int = 1) -> List[Any]:
        """Search for experiments."""
        ...


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class RunMetadata:
    """
    Metadata extracted from an MLflow run.

    Contains all information needed to reproduce the training dataset.
    """

    run_id: str
    experiment_id: str
    experiment_name: str
    run_name: Optional[str]
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]

    # Dataset information
    dataset_hash: Optional[str] = None
    dataset_path: Optional[str] = None
    dvc_version: Optional[str] = None

    # Date range
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    val_start_date: Optional[str] = None
    val_end_date: Optional[str] = None

    # Feature information
    feature_order_hash: Optional[str] = None
    feature_count: Optional[int] = None
    feature_contract_version: Optional[str] = None

    # Normalization
    norm_stats_hash: Optional[str] = None
    norm_stats_path: Optional[str] = None

    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        return result


@dataclass
class ReproductionResult:
    """
    Result of a dataset reproduction attempt.

    Implements the Result pattern for explicit success/failure handling.
    Use factory methods `success()` and `failure()` to create instances.
    """

    success: bool
    run_id: str
    output_path: Optional[Path]
    dataset_rows: int = 0
    dataset_columns: int = 0
    hash_verified: bool = False
    feature_order_verified: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reproduction_time: Optional[float] = None
    metadata: Optional[RunMetadata] = None

    @classmethod
    def create_success(
        cls,
        run_id: str,
        output_path: Path,
        dataset_rows: int,
        dataset_columns: int,
        hash_verified: bool,
        feature_order_verified: bool,
        reproduction_time: float,
        warnings: Optional[List[str]] = None,
        metadata: Optional[RunMetadata] = None,
    ) -> "ReproductionResult":
        """Factory method to create a successful result."""
        return cls(
            success=True,
            run_id=run_id,
            output_path=output_path,
            dataset_rows=dataset_rows,
            dataset_columns=dataset_columns,
            hash_verified=hash_verified,
            feature_order_verified=feature_order_verified,
            errors=[],
            warnings=warnings or [],
            reproduction_time=reproduction_time,
            metadata=metadata,
        )

    @classmethod
    def create_failure(
        cls,
        run_id: str,
        error: str,
        warnings: Optional[List[str]] = None,
        metadata: Optional[RunMetadata] = None,
    ) -> "ReproductionResult":
        """Factory method to create a failed result."""
        return cls(
            success=False,
            run_id=run_id,
            output_path=None,
            dataset_rows=0,
            dataset_columns=0,
            hash_verified=False,
            feature_order_verified=False,
            errors=[error],
            warnings=warnings or [],
            reproduction_time=None,
            metadata=metadata,
        )

    @property
    def is_success(self) -> bool:
        """Check if reproduction was successful."""
        return self.success

    @property
    def is_failure(self) -> bool:
        """Check if reproduction failed."""
        return not self.success

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.output_path:
            result["output_path"] = str(self.output_path)
        # Exclude metadata from asdict (handle separately)
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result


# =============================================================================
# MAIN CLASS
# =============================================================================


class DatasetReproducer:
    """
    Reproduces training datasets from MLflow run metadata.

    This class implements complete dataset reproduction workflow:
    1. Extract run metadata from MLflow
    2. Checkout DVC-versioned data
    3. Load and filter data by date range
    4. Verify hash integrity
    5. Validate feature order

    Supports dependency injection for the MLflow client to enable
    easier testing and flexibility.

    Attributes:
        mlflow_tracking_uri: URI of the MLflow tracking server
        client: MLflow tracking client instance

    Example:
        >>> reproducer = DatasetReproducer("http://localhost:5000")
        >>> result = reproducer.reproduce("abc123def456", Path("./output"))
        >>> if result.is_success:
        ...     print(f"Dataset reproduced at: {result.output_path}")
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        client: Optional[MLflowClientProtocol] = None,
    ) -> None:
        """
        Initialize the DatasetReproducer.

        Args:
            mlflow_tracking_uri: URI of the MLflow tracking server
                                 (e.g., "http://localhost:5000")
            client: Optional pre-configured MLflow client for dependency injection.
                    If not provided, a new client will be created.

        Raises:
            ImportError: If mlflow is not installed
            MLflowConnectionError: If cannot connect to MLflow server
        """
        self._tracking_uri = mlflow_tracking_uri
        self._client: Optional[MLflowClientProtocol] = client
        if self._client is None:
            self._initialize_mlflow_client()

    def _initialize_mlflow_client(self) -> None:
        """Initialize the MLflow client connection."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self._tracking_uri)
            self._client = MlflowClient(tracking_uri=self._tracking_uri)

            # Test connection by listing experiments
            self._client.search_experiments(max_results=1)
            logger.info("Connected to MLflow at %s", self._tracking_uri)

        except ImportError as e:
            raise ImportError(
                "mlflow is required for dataset reproduction. "
                "Install with: pip install mlflow"
            ) from e
        except Exception as e:
            logger.warning(
                "Could not verify MLflow connection: %s. "
                "Operations may fail if server is unavailable.",
                e,
            )

    def get_run_metadata(self, run_id: str) -> RunMetadata:
        """
        Extract dataset metadata from an MLflow run.

        Retrieves all relevant metadata including:
        - Dataset hash and path
        - Date ranges for train/val splits
        - Feature order hash
        - Normalization statistics hash

        Args:
            run_id: The MLflow run ID to extract metadata from

        Returns:
            RunMetadata object containing all extracted information

        Raises:
            RunNotFoundError: If run_id is not found
            DatasetReproductionError: If metadata extraction fails
        """
        logger.debug("Extracting metadata for run: %s", run_id)

        try:
            run = self._client.get_run(run_id)
        except Exception as e:
            raise RunNotFoundError(run_id) from e

        # Extract basic run info
        run_info = run.info
        run_data = run.data

        # Parse start/end times
        start_time = None
        end_time = None
        if run_info.start_time:
            start_time = datetime.fromtimestamp(run_info.start_time / 1000)
        if run_info.end_time:
            end_time = datetime.fromtimestamp(run_info.end_time / 1000)

        # Get experiment name
        experiment = self._client.get_experiment(run_info.experiment_id)
        experiment_name = experiment.name if experiment else "unknown"

        # Extract parameters and tags
        params = dict(run_data.params)
        tags = dict(run_data.tags)

        # Build metadata object
        metadata = RunMetadata(
            run_id=run_id,
            experiment_id=run_info.experiment_id,
            experiment_name=experiment_name,
            run_name=tags.get("mlflow.runName"),
            status=run_info.status,
            start_time=start_time,
            end_time=end_time,
            # Dataset information from params/tags
            dataset_hash=params.get("dataset_hash") or tags.get("dataset_hash_full"),
            dataset_path=params.get("data_processed_dir") or params.get("dataset_path"),
            dvc_version=tags.get("dvc_version") or params.get("dvc_version"),
            # Date ranges
            train_start_date=params.get("train_start_date"),
            train_end_date=params.get("train_end_date"),
            val_start_date=params.get("val_start_date"),
            val_end_date=params.get("val_end_date"),
            # Feature information
            feature_order_hash=(
                params.get("feature_order_hash")
                or tags.get("feature_order_hash_full")
            ),
            feature_count=int(tags.get("feature_count", 15)),
            feature_contract_version=tags.get("feature_contract_version"),
            # Normalization
            norm_stats_hash=(
                params.get("norm_stats_hash") or tags.get("norm_stats_hash_full")
            ),
            norm_stats_path=params.get("data_norm_stats_path"),
            # Store all params and tags for reference
            parameters=params,
            tags=tags,
        )

        logger.debug(
            "Extracted metadata: dataset_hash=%s, feature_order_hash=%s",
            metadata.dataset_hash,
            metadata.feature_order_hash,
        )

        return metadata

    def checkout_dvc_data(
        self, dataset_path: str, dvc_version: Optional[str] = None
    ) -> Path:
        """
        Checkout DVC-versioned data.

        Uses DVC to checkout the specific version of data that was used
        during training. If no version is specified, uses the current version.

        Args:
            dataset_path: Path to the DVC-tracked data directory or file
            dvc_version: Optional DVC version/commit hash to checkout

        Returns:
            Path to the checked out data

        Raises:
            DVCCheckoutError: If DVC checkout fails
            FileNotFoundError: If dataset_path doesn't exist after checkout
        """
        data_path = PROJECT_ROOT / dataset_path
        logger.debug("Checking out DVC data: %s", data_path)

        if dvc_version:
            self._checkout_specific_version(dataset_path, dvc_version)
        else:
            self._ensure_data_available(data_path, dataset_path)

        # Verify data exists
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data not found at {data_path}. "
                "Ensure DVC is configured and data is tracked."
            )

        return data_path

    def _checkout_specific_version(
        self, dataset_path: str, dvc_version: str
    ) -> None:
        """Checkout a specific DVC version of the data."""
        logger.info("Checking out DVC version: %s", dvc_version)
        try:
            # First, try git checkout for the DVC version
            result = subprocess.run(
                ["git", "checkout", dvc_version, "--", f"{dataset_path}.dvc"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=60,
            )

            if result.returncode != 0:
                logger.warning(
                    "Git checkout failed: %s. Trying DVC fetch instead.",
                    result.stderr,
                )

            # Pull the data from DVC remote
            self._run_dvc_pull(dataset_path)

        except subprocess.TimeoutExpired as e:
            raise DVCCheckoutError(
                f"DVC checkout timed out after {DVC_TIMEOUT_SECONDS} seconds",
                dataset_path,
            ) from e
        except FileNotFoundError as e:
            raise DVCCheckoutError(
                "DVC not installed. Install with: pip install dvc", dataset_path
            ) from e

    def _ensure_data_available(self, data_path: Path, dataset_path: str) -> None:
        """Ensure data is available, pulling if necessary."""
        if data_path.exists():
            logger.debug("Using existing data at: %s", data_path)
        else:
            logger.info("Running DVC pull to fetch data...")
            try:
                self._run_dvc_pull(dataset_path)
            except DVCCheckoutError as e:
                logger.warning("DVC pull failed: %s", e)

    def _run_dvc_pull(self, dataset_path: str) -> None:
        """Execute DVC pull command."""
        try:
            result = subprocess.run(
                ["dvc", "pull", str(dataset_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=DVC_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise DVCCheckoutError(result.stderr, dataset_path)
        except subprocess.TimeoutExpired as e:
            raise DVCCheckoutError(
                f"DVC pull timed out after {DVC_TIMEOUT_SECONDS} seconds",
                dataset_path,
            ) from e
        except FileNotFoundError as e:
            raise DVCCheckoutError(
                "DVC not installed", dataset_path
            ) from e

    def load_and_filter_data(
        self,
        data_path: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data and filter by date range.

        Loads the dataset from the specified path and filters it to the
        date range used during training.

        Args:
            data_path: Path to the data file or directory
            start_date: Start date for filtering (ISO format: YYYY-MM-DD)
            end_date: End date for filtering (ISO format: YYYY-MM-DD)

        Returns:
            Filtered pandas DataFrame

        Raises:
            DataLoadError: If data file not found or cannot be loaded
        """
        logger.debug("Loading data from: %s", data_path)

        data_file = self._resolve_data_file(data_path)
        df = self._read_data_file(data_file)

        logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), data_file.name)

        df = self._filter_by_date_range(df, start_date, end_date)

        return df

    def _resolve_data_file(self, data_path: Path) -> Path:
        """Resolve the data file path from a directory or file."""
        if not data_path.is_dir():
            return data_path

        # Look for common data file patterns in priority order
        priority_files = [
            "train_features.parquet",
            "training_data.parquet",
            "dataset.parquet",
            "features.parquet",
        ]

        for filename in priority_files:
            candidate = data_path / filename
            if candidate.exists():
                return candidate

        # Fall back to first parquet/csv found
        for pattern in ["*.parquet", "*.csv"]:
            files = list(data_path.glob(pattern))
            if files:
                return files[0]

        raise DataLoadError(
            f"No data files found in {data_path}. Expected .parquet or .csv files.",
            data_path,
        )

    def _read_data_file(self, data_file: Path) -> pd.DataFrame:
        """Read data from file based on its format."""
        logger.debug("Loading file: %s", data_file)

        try:
            if data_file.suffix == ".parquet":
                return pd.read_parquet(data_file)
            elif data_file.suffix == ".csv":
                return pd.read_csv(data_file)
            else:
                raise DataLoadError(
                    f"Unsupported file format: {data_file.suffix}", data_file
                )
        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(str(e), data_file) from e

    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Filter DataFrame by date range if dates are specified."""
        timestamp_col = self._find_timestamp_column(df)

        if timestamp_col is None:
            if start_date or end_date:
                logger.warning(
                    "No timestamp column found; cannot filter by date range"
                )
            return df

        if not (start_date or end_date):
            return df

        logger.debug(
            "Filtering by date range: %s to %s",
            start_date or "start",
            end_date or "end",
        )

        # Convert timestamp column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df[timestamp_col] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df[timestamp_col] <= end_dt]

        logger.info("After date filtering: %d rows", len(df))

        return df

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the timestamp column in a DataFrame."""
        for col in ["timestamp", "time", "datetime", "date", "index"]:
            if col in df.columns:
                return col

        # Try index if no timestamp column found
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            return df.columns[0]

        return None

    def verify_hash(
        self,
        df: pd.DataFrame,
        expected_hash: str,
        raise_on_mismatch: bool = False,
    ) -> bool:
        """
        Verify that the reproduced dataset matches the expected hash.

        Computes a SHA256 hash of the DataFrame content and compares it
        to the expected hash from the MLflow run metadata.

        Args:
            df: The reproduced DataFrame to verify
            expected_hash: Expected hash from MLflow metadata
            raise_on_mismatch: If True, raise HashMismatchError on failure

        Returns:
            True if hashes match, False otherwise

        Raises:
            HashMismatchError: If raise_on_mismatch is True and hashes don't match

        Note:
            The hash is computed on the sorted, canonical representation
            of the DataFrame to ensure consistency.
        """
        logger.debug("Verifying dataset hash...")

        # Compute hash of DataFrame content
        computed_hash = self._compute_dataframe_hash(df)

        if self._hashes_match(expected_hash, computed_hash):
            logger.info("Hash verification PASSED: %s...", computed_hash[:16])
            return True

        logger.warning(
            "Hash verification FAILED: expected=%s..., computed=%s...",
            expected_hash[:16],
            computed_hash[:16],
        )

        if raise_on_mismatch:
            raise HashMismatchError(expected_hash, computed_hash)

        return False

    def _hashes_match(self, expected: str, computed: str) -> bool:
        """Compare two hashes, handling truncated hashes."""
        expected_lower = expected.lower()
        computed_lower = computed.lower()

        # Try exact match first
        if expected_lower == computed_lower:
            return True

        # Try prefix match (for truncated hashes)
        min_len = min(len(expected_lower), len(computed_lower))
        return min_len >= 8 and expected_lower[:min_len] == computed_lower[:min_len]

    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Compute a deterministic hash of a DataFrame.

        The hash is computed on a canonical representation to ensure
        reproducibility regardless of row/column ordering.

        Args:
            df: DataFrame to hash

        Returns:
            SHA256 hash as hex string
        """
        # Sort columns for deterministic ordering
        df_sorted = df.reindex(sorted(df.columns), axis=1)

        # Sort rows if there's a timestamp column
        timestamp_cols = ["timestamp", "time", "datetime", "date"]
        for col in timestamp_cols:
            if col in df_sorted.columns:
                df_sorted = df_sorted.sort_values(col)
                break

        # Convert to CSV string for hashing (deterministic format)
        csv_content = df_sorted.to_csv(index=False)

        # Compute SHA256
        hasher = hashlib.sha256()
        hasher.update(csv_content.encode("utf-8"))

        return hasher.hexdigest()

    def verify_feature_order(
        self,
        feature_order_hash: str,
        raise_on_mismatch: bool = False,
    ) -> bool:
        """
        Verify that the current feature order matches the training run.

        Computes the hash of the current feature order and compares it
        to the hash stored in the MLflow run metadata.

        Args:
            feature_order_hash: Expected feature order hash from MLflow
            raise_on_mismatch: If True, raise FeatureOrderMismatchError on failure

        Returns:
            True if feature order matches, False otherwise

        Raises:
            FeatureOrderMismatchError: If raise_on_mismatch is True and orders don't match
        """
        logger.debug("Verifying feature order...")

        try:
            from src.core.contracts.feature_contract import (
                FEATURE_ORDER,
                FEATURE_ORDER_HASH,
            )

            if self._hashes_match(feature_order_hash, FEATURE_ORDER_HASH):
                logger.info("Feature order verification PASSED: %s", FEATURE_ORDER_HASH)
                return True

            # Log the mismatch details
            logger.warning(
                "Feature order verification FAILED: expected=%s, current=%s, features=%s",
                feature_order_hash,
                FEATURE_ORDER_HASH,
                list(FEATURE_ORDER),
            )

            if raise_on_mismatch:
                raise FeatureOrderMismatchError(feature_order_hash, FEATURE_ORDER_HASH)

            return False

        except ImportError as e:
            logger.warning(
                "Could not import feature contract: %s. Feature order verification skipped.",
                e,
            )
            return False

    def reproduce(
        self,
        run_id: str,
        output_dir: Path,
        verify_hash: bool = True,
    ) -> ReproductionResult:
        """
        Reproduce a complete training dataset from an MLflow run.

        This is the main entry point for dataset reproduction. It:
        1. Extracts metadata from the MLflow run
        2. Checks out the correct DVC version of data
        3. Loads and filters the data by date range
        4. Verifies hash integrity (if enabled)
        5. Validates feature order
        6. Saves the reproduced dataset and metadata

        Args:
            run_id: MLflow run ID to reproduce
            output_dir: Directory to save reproduced dataset
            verify_hash: Whether to verify dataset hash (default: True)

        Returns:
            ReproductionResult containing success status, output path, and diagnostics
        """
        import time

        start_time = time.time()
        warnings: List[str] = []

        logger.info("Starting dataset reproduction for run: %s", run_id)
        logger.debug("Output directory: %s", output_dir)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract metadata
        metadata = self._extract_metadata_safely(run_id, output_dir)
        if metadata is None:
            return ReproductionResult.create_failure(
                run_id=run_id,
                error="Failed to extract run metadata",
            )

        # Step 2: Checkout DVC data
        data_path, dvc_warnings = self._checkout_data_safely(metadata)
        warnings.extend(dvc_warnings)

        if data_path is None:
            self._save_failure_report(
                output_dir, run_id, "Could not locate dataset", metadata
            )
            return ReproductionResult.create_failure(
                run_id=run_id,
                error="Could not locate dataset",
                warnings=warnings,
                metadata=metadata,
            )

        # Step 3: Load and filter data
        df = self._load_data_safely(data_path, metadata, output_dir, run_id)
        if df is None:
            return ReproductionResult.create_failure(
                run_id=run_id,
                error="Failed to load data",
                warnings=warnings,
                metadata=metadata,
            )

        # Step 4: Verify hash (if enabled and hash available)
        hash_verified, hash_warning = self._verify_hash_safely(
            df, metadata.dataset_hash, verify_hash
        )
        if hash_warning:
            warnings.append(hash_warning)

        # Step 5: Verify feature order
        feature_order_verified, order_warning = self._verify_feature_order_safely(
            metadata.feature_order_hash
        )
        if order_warning:
            warnings.append(order_warning)

        # Step 6: Save reproduced dataset
        output_file = output_dir / f"reproduced_dataset_{run_id[:8]}.parquet"
        df.to_parquet(output_file, index=False)
        logger.info("Saved reproduced dataset to: %s", output_file)

        # Calculate reproduction time
        reproduction_time = time.time() - start_time

        # Create result object
        result = ReproductionResult.create_success(
            run_id=run_id,
            output_path=output_file,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            hash_verified=hash_verified,
            feature_order_verified=feature_order_verified,
            reproduction_time=reproduction_time,
            warnings=warnings,
            metadata=metadata,
        )

        # Save reproduction metadata
        self._save_reproduction_metadata(output_dir, metadata, result)

        # Log summary
        self._log_reproduction_summary(result)

        return result

    def _extract_metadata_safely(
        self, run_id: str, output_dir: Path
    ) -> Optional[RunMetadata]:
        """Extract metadata with error handling."""
        try:
            return self.get_run_metadata(run_id)
        except RunNotFoundError as e:
            logger.error("Run not found: %s", run_id)
            self._save_failure_report(output_dir, run_id, str(e))
            return None
        except Exception as e:
            logger.error("Failed to extract metadata: %s", e)
            self._save_failure_report(output_dir, run_id, str(e))
            return None

    def _checkout_data_safely(
        self, metadata: RunMetadata
    ) -> Tuple[Optional[Path], List[str]]:
        """Checkout DVC data with error handling and fallback."""
        warnings: List[str] = []
        data_path: Optional[Path] = None

        if metadata.dataset_path:
            try:
                data_path = self.checkout_dvc_data(
                    metadata.dataset_path,
                    metadata.dvc_version,
                )
            except DVCCheckoutError as e:
                logger.warning("DVC checkout failed: %s", e)
                warnings.append(f"DVC checkout failed: {e}")

                # Try default path as fallback
                default_path = PROJECT_ROOT / "data" / "processed"
                if default_path.exists():
                    data_path = default_path
                    warnings.append(f"Using default path: {default_path}")
        else:
            # Try default path
            data_path = PROJECT_ROOT / "data" / "processed"
            if not data_path.exists():
                logger.error("No dataset path in metadata and default path not found")
                data_path = None

        return data_path, warnings

    def _load_data_safely(
        self,
        data_path: Path,
        metadata: RunMetadata,
        output_dir: Path,
        run_id: str,
    ) -> Optional[pd.DataFrame]:
        """Load and filter data with error handling."""
        try:
            return self.load_and_filter_data(
                data_path,
                start_date=metadata.train_start_date,
                end_date=metadata.train_end_date,
            )
        except DataLoadError as e:
            logger.error("Failed to load data: %s", e)
            self._save_failure_report(output_dir, run_id, str(e), metadata)
            return None
        except Exception as e:
            logger.error("Unexpected error loading data: %s", e)
            self._save_failure_report(output_dir, run_id, str(e), metadata)
            return None

    def _verify_hash_safely(
        self,
        df: pd.DataFrame,
        dataset_hash: Optional[str],
        should_verify: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Verify hash with warning generation."""
        if not should_verify:
            return False, None

        if not dataset_hash:
            return False, "No dataset hash in metadata; skipping verification"

        hash_verified = self.verify_hash(df, dataset_hash)
        if not hash_verified:
            return False, f"Dataset hash mismatch. Expected: {dataset_hash}"

        return True, None

    def _verify_feature_order_safely(
        self, feature_order_hash: Optional[str]
    ) -> Tuple[bool, Optional[str]]:
        """Verify feature order with warning generation."""
        if not feature_order_hash:
            return False, "No feature order hash in metadata; skipping verification"

        verified = self.verify_feature_order(feature_order_hash)
        if not verified:
            return False, f"Feature order mismatch. Expected: {feature_order_hash}"

        return True, None

    def _log_reproduction_summary(self, result: ReproductionResult) -> None:
        """Log a summary of the reproduction result."""
        logger.info("=" * 60)
        logger.info("REPRODUCTION SUMMARY")
        logger.info("=" * 60)
        logger.info("Run ID: %s", result.run_id)
        logger.info("Rows: %d", result.dataset_rows)
        logger.info("Columns: %d", result.dataset_columns)
        logger.info("Hash Verified: %s", result.hash_verified)
        logger.info("Feature Order Verified: %s", result.feature_order_verified)
        if result.reproduction_time:
            logger.info("Time: %.2fs", result.reproduction_time)
        logger.info("Output: %s", result.output_path)

        if result.warnings:
            logger.warning("Warnings: %d", len(result.warnings))
            for w in result.warnings:
                logger.warning("  - %s", w)

        logger.info("=" * 60)

    def _save_reproduction_metadata(
        self,
        output_dir: Path,
        run_metadata: RunMetadata,
        result: ReproductionResult,
    ) -> None:
        """
        Save reproduction metadata JSON alongside the dataset.

        Args:
            output_dir: Output directory
            run_metadata: Original run metadata
            result: Reproduction result
        """
        metadata_file = output_dir / f"reproduction_metadata_{run_metadata.run_id[:8]}.json"

        combined_metadata = {
            "reproduction": {
                "timestamp": datetime.now().isoformat(),
                "reproducer_version": __version__,
                "result": result.to_dict(),
            },
            "original_run": run_metadata.to_dict(),
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2, default=str)

        logger.info("Saved reproduction metadata to: %s", metadata_file)

    def _save_failure_report(
        self,
        output_dir: Path,
        run_id: str,
        error_message: str,
        metadata: Optional[RunMetadata] = None,
    ) -> None:
        """
        Save a failure report when reproduction fails.

        Args:
            output_dir: Output directory
            run_id: Run ID that failed
            error_message: Error message
            metadata: Optional metadata if extraction succeeded
        """
        report_file = output_dir / f"reproduction_failure_{run_id[:8]}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "success": False,
            "error": error_message,
            "metadata": metadata.to_dict() if metadata else None,
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Saved failure report to: %s", report_file)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce training dataset from MLflow run ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic reproduction
  python scripts/reproduce_dataset_from_run.py --run-id abc123 --output ./reproduced/

  # Skip hash verification
  python scripts/reproduce_dataset_from_run.py --run-id abc123 --output ./reproduced/ --skip-hash-verify

  # Custom MLflow server
  python scripts/reproduce_dataset_from_run.py --run-id abc123 --output ./reproduced/ --tracking-uri http://mlflow.example.com:5000

Exit Codes:
  0 - Success
  1 - General failure
  2 - Validation error (hash or feature order mismatch)
  130 - Interrupted by user

Version: {__version__}
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version and exit",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        metavar="RUN_ID",
        help="MLflow run ID to reproduce (required)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="DIR",
        help="Output directory for reproduced dataset (required)",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        metavar="URI",
        help=(
            "MLflow tracking server URI. Can also be set via MLFLOW_TRACKING_URI "
            "environment variable (default: http://localhost:5000)"
        ),
    )

    parser.add_argument(
        "--skip-hash-verify",
        action="store_true",
        help="Skip dataset hash verification (not recommended for production)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 2 if hash or feature order verification fails",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code:
        - 0: Success
        - 1: General failure
        - 2: Validation error (hash or feature order mismatch in strict mode)
        - 130: Interrupted by user
    """
    args = parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _print_banner(args)

    try:
        # Initialize reproducer
        reproducer = DatasetReproducer(args.tracking_uri)

        # Reproduce dataset
        result = reproducer.reproduce(
            run_id=args.run_id,
            output_dir=Path(args.output),
            verify_hash=not args.skip_hash_verify,
        )

        return _handle_result(result, args.strict)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return EXIT_INTERRUPTED

    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return EXIT_FAILURE

    except MLflowConnectionError as e:
        logger.error("MLflow connection failed: %s", e)
        return EXIT_FAILURE

    except DatasetReproductionError as e:
        logger.error("Reproduction failed: %s", e)
        return EXIT_FAILURE

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return EXIT_FAILURE


def _print_banner(args: argparse.Namespace) -> None:
    """Print the startup banner with configuration details."""
    print("=" * 60)
    print(f"Dataset Reproduction from MLflow Run (v{__version__})")
    print("=" * 60)
    print(f"Run ID:       {args.run_id}")
    print(f"Output:       {args.output}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Verify Hash:  {not args.skip_hash_verify}")
    print(f"Strict Mode:  {args.strict}")
    print("=" * 60)


def _handle_result(result: ReproductionResult, strict: bool) -> int:
    """
    Handle the reproduction result and return appropriate exit code.

    Args:
        result: The reproduction result
        strict: If True, return EXIT_VALIDATION_ERROR on verification failures

    Returns:
        Exit code
    """
    if result.is_failure:
        print(f"\nFailed to reproduce dataset: {result.errors}")
        return EXIT_FAILURE

    print(f"\nSuccess! Dataset reproduced at: {result.output_path}")

    # In strict mode, check for verification failures
    if strict:
        if not result.hash_verified:
            print("WARNING: Hash verification failed (strict mode enabled)")
            return EXIT_VALIDATION_ERROR
        if not result.feature_order_verified:
            print("WARNING: Feature order verification failed (strict mode enabled)")
            return EXIT_VALIDATION_ERROR

    # Log warnings even in non-strict mode
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
