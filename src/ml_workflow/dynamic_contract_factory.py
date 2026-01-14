"""
Dynamic Feature Contract Factory
================================
Generates feature contracts dynamically from training data.

SOLID Principles:
- Single Responsibility: Only creates contracts from data
- Open/Closed: New feature sets via configuration, not code changes
- Dependency Inversion: Depends on abstractions (DataFrames, configs)
- Liskov Substitution: All contracts implement same interface

Design Patterns:
- Factory Pattern: Creates contracts from configuration
- Builder Pattern: Builds contracts step-by-step
- Registry Pattern: Stores and retrieves contracts

This eliminates manual contract creation when training new model versions.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Final
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Dynamic Feature Contract (NOT frozen - built at runtime)
# =============================================================================

@dataclass
class DynamicFeatureContract:
    """
    Feature contract generated dynamically from training data.

    Unlike the frozen static contract, this can be created at runtime
    for new model versions without editing source code.
    """
    version: str
    observation_dim: int
    feature_order: Tuple[str, ...]
    norm_stats_path: str
    model_path: str
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    # Technical indicator periods
    rsi_period: int = 9
    atr_period: int = 10
    adx_period: int = 14
    warmup_bars: int = 14

    # Trading hours
    trading_hours_start: str = "13:00"
    trading_hours_end: str = "17:55"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_from_dataset: Optional[str] = None
    sample_count: int = 0

    # Hash for integrity verification
    contract_hash: Optional[str] = None

    def __post_init__(self):
        """Validate and compute hash after initialization"""
        if self.observation_dim != len(self.feature_order):
            raise ValueError(
                f"observation_dim ({self.observation_dim}) must match "
                f"len(feature_order) ({len(self.feature_order)})"
            )

        # Compute contract hash
        if self.contract_hash is None:
            self.contract_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of contract for integrity verification"""
        content = json.dumps({
            "version": self.version,
            "observation_dim": self.observation_dim,
            "feature_order": list(self.feature_order),
            "clip_range": list(self.clip_range),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "version": self.version,
            "observation_dim": self.observation_dim,
            "feature_order": list(self.feature_order),
            "norm_stats_path": self.norm_stats_path,
            "model_path": self.model_path,
            "clip_range": list(self.clip_range),
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "adx_period": self.adx_period,
            "warmup_bars": self.warmup_bars,
            "trading_hours_start": self.trading_hours_start,
            "trading_hours_end": self.trading_hours_end,
            "created_at": self.created_at,
            "created_from_dataset": self.created_from_dataset,
            "sample_count": self.sample_count,
            "contract_hash": self.contract_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicFeatureContract":
        """Create contract from dictionary"""
        return cls(
            version=data["version"],
            observation_dim=data["observation_dim"],
            feature_order=tuple(data["feature_order"]),
            norm_stats_path=data["norm_stats_path"],
            model_path=data["model_path"],
            clip_range=tuple(data.get("clip_range", (-5.0, 5.0))),
            rsi_period=data.get("rsi_period", 9),
            atr_period=data.get("atr_period", 10),
            adx_period=data.get("adx_period", 14),
            warmup_bars=data.get("warmup_bars", 14),
            trading_hours_start=data.get("trading_hours_start", "13:00"),
            trading_hours_end=data.get("trading_hours_end", "17:55"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            created_from_dataset=data.get("created_from_dataset"),
            sample_count=data.get("sample_count", 0),
            contract_hash=data.get("contract_hash"),
        )


# =============================================================================
# Norm Stats Calculator
# =============================================================================

@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    mean: float
    std: float
    min: float
    max: float
    count: int
    null_pct: float
    percentile_1: float
    percentile_99: float


class NormStatsCalculator:
    """
    Calculates normalization statistics from training data.

    This replaces manual norm_stats.json creation.
    """

    def __init__(self, clip_range: Tuple[float, float] = (-5.0, 5.0)):
        self.clip_range = clip_range

    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        exclude_state_features: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate normalization statistics from DataFrame.

        Args:
            df: Training data DataFrame
            feature_columns: List of feature column names
            exclude_state_features: Features to exclude (e.g., position, time)

        Returns:
            Dict of feature name -> {mean, std, min, max, count, null_pct}
        """
        exclude_state_features = exclude_state_features or ["position", "time_normalized"]

        stats = {}

        for col in feature_columns:
            if col in exclude_state_features:
                continue

            if col not in df.columns:
                logger.warning(f"Feature '{col}' not found in DataFrame")
                continue

            values = df[col].dropna()

            if len(values) == 0:
                logger.warning(f"Feature '{col}' has no valid values")
                continue

            stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(len(values)),
                "null_pct": float(df[col].isna().mean()),
                "percentile_1": float(np.percentile(values, 1)),
                "percentile_99": float(np.percentile(values, 99)),
            }

            # Validate std is not zero
            if stats[col]["std"] == 0:
                logger.warning(f"Feature '{col}' has zero std, setting to 1.0")
                stats[col]["std"] = 1.0

        logger.info(f"Calculated norm_stats for {len(stats)} features from {len(df)} samples")
        return stats

    def save_to_json(
        self,
        stats: Dict[str, Dict[str, float]],
        output_path: Path
    ) -> str:
        """
        Save norm_stats to JSON file and return hash.

        Returns:
            SHA256 hash of saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Compute hash
        with open(output_path, 'r') as f:
            content = f.read()

        file_hash = hashlib.sha256(content.encode()).hexdigest()

        logger.info(f"Saved norm_stats to {output_path} (hash: {file_hash[:16]})")
        return file_hash


# =============================================================================
# Contract Factory
# =============================================================================

class ContractFactory:
    """
    Factory for creating feature contracts dynamically.

    Usage:
        factory = ContractFactory(project_root)
        contract = factory.create_from_dataset(
            dataset_path="data/datasets/my_training_data.csv",
            version="v1",
            feature_columns=[...],
        )
        # Contract and norm_stats are automatically created
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.models_dir = project_root / "models"
        self.contracts_dir = project_root / "config" / "contracts"

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)

    def create_from_dataset(
        self,
        dataset_path: Path,
        version: str,
        feature_columns: List[str],
        state_features: List[str] = None,
        technical_periods: Dict[str, int] = None,
        trading_hours: Dict[str, str] = None,
    ) -> DynamicFeatureContract:
        """
        Create a complete feature contract from training dataset.

        This method:
        1. Loads the dataset
        2. Calculates norm_stats automatically
        3. Saves norm_stats to config/
        4. Creates and saves the contract

        Args:
            dataset_path: Path to training CSV
            version: Version string (e.g., "v1")
            feature_columns: List of feature column names in order
            state_features: Features that are not normalized (e.g., position)
            technical_periods: Dict of indicator periods
            trading_hours: Dict with start/end hours

        Returns:
            DynamicFeatureContract ready for training
        """
        state_features = state_features or ["position", "time_normalized"]
        technical_periods = technical_periods or {"rsi": 9, "atr": 10, "adx": 14}
        trading_hours = trading_hours or {"start": "13:00", "end": "19:00"}

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Calculate norm_stats
        calculator = NormStatsCalculator()
        norm_stats = calculator.calculate_from_dataframe(
            df=df,
            feature_columns=feature_columns,
            exclude_state_features=state_features,
        )

        # Save norm_stats
        norm_stats_filename = f"{version}_norm_stats.json"
        norm_stats_path = self.config_dir / norm_stats_filename
        norm_stats_hash = calculator.save_to_json(norm_stats, norm_stats_path)

        # Create contract
        contract = DynamicFeatureContract(
            version=version,
            observation_dim=len(feature_columns),
            feature_order=tuple(feature_columns),
            norm_stats_path=str(norm_stats_path.relative_to(self.project_root)),
            model_path=f"models/ppo_{version}_production/final_model.zip",
            rsi_period=technical_periods.get("rsi", 9),
            atr_period=technical_periods.get("atr", 10),
            adx_period=technical_periods.get("adx", 14),
            warmup_bars=max(technical_periods.values()),
            trading_hours_start=trading_hours.get("start", "13:00"),
            trading_hours_end=trading_hours.get("end", "17:55"),
            created_from_dataset=str(dataset_path),
            sample_count=len(df),
        )

        # Save contract
        self._save_contract(contract)

        logger.info(
            f"Created contract {version} with {contract.observation_dim} features "
            f"from {contract.sample_count} samples"
        )

        return contract

    def create_from_config(
        self,
        version: str,
        feature_config: Dict[str, Any],
        norm_stats: Dict[str, Dict[str, float]],
    ) -> DynamicFeatureContract:
        """
        Create contract from explicit configuration (no dataset needed).

        Args:
            version: Version string
            feature_config: Configuration dict with feature_order, etc.
            norm_stats: Pre-computed normalization statistics

        Returns:
            DynamicFeatureContract
        """
        # Save norm_stats
        calculator = NormStatsCalculator()
        norm_stats_path = self.config_dir / f"{version}_norm_stats.json"
        calculator.save_to_json(norm_stats, norm_stats_path)

        # Create contract
        contract = DynamicFeatureContract(
            version=version,
            observation_dim=len(feature_config["feature_order"]),
            feature_order=tuple(feature_config["feature_order"]),
            norm_stats_path=str(norm_stats_path.relative_to(self.project_root)),
            model_path=feature_config.get("model_path", f"models/ppo_{version}/model.zip"),
            clip_range=tuple(feature_config.get("clip_range", (-5.0, 5.0))),
            rsi_period=feature_config.get("rsi_period", 9),
            atr_period=feature_config.get("atr_period", 10),
            adx_period=feature_config.get("adx_period", 14),
        )

        self._save_contract(contract)
        return contract

    def load_contract(self, version: str) -> DynamicFeatureContract:
        """
        Load a saved contract by version.

        Args:
            version: Version string (e.g., "v1")

        Returns:
            DynamicFeatureContract

        Raises:
            FileNotFoundError: If contract not found
        """
        contract_path = self.contracts_dir / f"{version}_contract.json"

        if not contract_path.exists():
            raise FileNotFoundError(
                f"Contract for version '{version}' not found at {contract_path}. "
                f"Use create_from_dataset() or create_from_config() first."
            )

        with open(contract_path, 'r') as f:
            data = json.load(f)

        return DynamicFeatureContract.from_dict(data)

    def list_contracts(self) -> List[str]:
        """List all available contract versions"""
        contracts = []
        for path in self.contracts_dir.glob("*_contract.json"):
            version = path.stem.replace("_contract", "")
            contracts.append(version)
        return sorted(contracts)

    def _save_contract(self, contract: DynamicFeatureContract) -> None:
        """Save contract to JSON file"""
        contract_path = self.contracts_dir / f"{contract.version}_contract.json"

        with open(contract_path, 'w') as f:
            json.dump(contract.to_dict(), f, indent=2)

        logger.info(f"Saved contract to {contract_path}")


# =============================================================================
# Contract Registry (Runtime)
# =============================================================================

class ContractRegistry:
    """
    Runtime registry for loaded contracts.

    Provides unified access to both static and dynamic contracts.
    """

    _contracts: Dict[str, DynamicFeatureContract] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, project_root: Path) -> None:
        """Initialize registry with all available contracts"""
        if cls._initialized:
            return

        factory = ContractFactory(project_root)

        # Load static contract
        try:
            from src.features.contract import FEATURE_CONTRACT
            cls._contracts["current"] = DynamicFeatureContract(
                version="current",
                observation_dim=FEATURE_CONTRACT.observation_dim,
                feature_order=FEATURE_CONTRACT.feature_order,
                norm_stats_path=FEATURE_CONTRACT.norm_stats_path,
                model_path="models/ppo_production/final_model.zip",
                clip_range=FEATURE_CONTRACT.clip_range,
                rsi_period=FEATURE_CONTRACT.rsi_period,
                atr_period=FEATURE_CONTRACT.atr_period,
                adx_period=FEATURE_CONTRACT.adx_period,
            )
            logger.info("Loaded static contract")
        except ImportError:
            logger.warning("Static contract not available")

        # Load all dynamic contracts
        for version in factory.list_contracts():
            if version not in cls._contracts:
                try:
                    cls._contracts[version] = factory.load_contract(version)
                    logger.info(f"Loaded dynamic contract: {version}")
                except Exception as e:
                    logger.warning(f"Failed to load contract {version}: {e}")

        cls._initialized = True

    @classmethod
    def get(cls, version: str) -> DynamicFeatureContract:
        """Get contract by version"""
        if version not in cls._contracts:
            raise KeyError(
                f"Contract '{version}' not registered. "
                f"Available: {list(cls._contracts.keys())}"
            )
        return cls._contracts[version]

    @classmethod
    def register(cls, contract: DynamicFeatureContract) -> None:
        """Register a new contract"""
        cls._contracts[contract.version] = contract
        logger.info(f"Registered contract: {contract.version}")

    @classmethod
    def list_versions(cls) -> List[str]:
        """List all registered versions"""
        return list(cls._contracts.keys())


# =============================================================================
# Convenience Functions
# =============================================================================

def create_contract_from_training(
    project_root: Path,
    dataset_path: Path,
    version: str,
    feature_columns: List[str],
    **kwargs
) -> DynamicFeatureContract:
    """
    Convenience function to create contract from training data.

    Example:
        contract = create_contract_from_training(
            project_root=Path("."),
            dataset_path=Path("data/training.csv"),
            version="v1",
            feature_columns=[
                "log_ret_5m", "log_ret_1h", "log_ret_4h",
                "rsi_9", "atr_pct", "adx_14",
                "dxy_z", "vix_z", "embi_z",
                "position", "time_normalized"
            ]
        )
    """
    factory = ContractFactory(project_root)
    return factory.create_from_dataset(
        dataset_path=dataset_path,
        version=version,
        feature_columns=feature_columns,
        **kwargs
    )


def get_contract(version: str, project_root: Path = None) -> DynamicFeatureContract:
    """
    Get contract by version (unified access).

    Supports both static and dynamic contracts.
    """
    if project_root:
        ContractRegistry.initialize(project_root)
    return ContractRegistry.get(version)
