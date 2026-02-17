"""
Experiment Configuration Schema
===============================

Pydantic models for experiment configuration validation and serialization.

Part of EXP-04 remediation from Experimentation Audit.

This module defines:
- ExperimentConfig: Complete experiment configuration
- DatasetConfig: Dataset configuration
- HyperparametersConfig: PPO hyperparameters
- NetworkConfig: Neural network architecture
- EnvironmentConfig: Trading environment settings

Usage:
    from src.core.schemas.experiment_config import ExperimentConfig

    # Load and validate config
    config = ExperimentConfig.from_yaml("experiments/baseline/config.yaml")

    # Get experiment hash
    print(f"Hash: {config.experiment_hash}")

    # Export to dict
    config_dict = config.model_dump()

Author: Trading Team
Date: 2026-01-17
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# =============================================================================
# Sub-configurations
# =============================================================================

class DateRangeConfig(BaseModel):
    """Date range configuration for datasets."""

    train_start: str = Field(..., description="Training data start date (YYYY-MM-DD)")
    train_end: str = Field(..., description="Training data end date (YYYY-MM-DD)")
    validation_start: Optional[str] = Field(None, description="Validation data start date")
    validation_end: Optional[str] = Field(None, description="Validation data end date")
    test_start: Optional[str] = Field(None, description="Test data start date")
    test_end: Optional[str] = Field(None, description="Test data end date")


class DatasetConfig(BaseModel):
    """Dataset configuration for experiments."""

    version: str = Field(..., description="Dataset version string")
    dvc_tag: Optional[str] = Field(None, description="DVC tag for this dataset version")
    source: str = Field("data/processed/", description="Data source directory")
    date_range: DateRangeConfig = Field(..., description="Date range configuration")
    train_ratio: float = Field(0.7, ge=0, le=1, description="Training split ratio")
    val_ratio: float = Field(0.15, ge=0, le=1, description="Validation split ratio")
    test_ratio: float = Field(0.15, ge=0, le=1, description="Test split ratio")

    @model_validator(mode="after")
    def validate_ratios(self) -> "DatasetConfig":
        """Validate that split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class NormalizationConfig(BaseModel):
    """Feature normalization configuration."""

    method: str = Field("zscore", description="Normalization method")
    clip_percentile: float = Field(99.5, ge=0, le=100, description="Clip percentile")
    outlier_std: float = Field(4.0, gt=0, description="Outlier std threshold")


class FeaturesConfig(BaseModel):
    """Feature configuration for experiments."""

    feature_order: List[str] = Field(..., min_length=1, description="Ordered list of features")
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    contract_version: str = Field("1.0.0", description="Feature contract version")

    @computed_field
    @property
    def feature_count(self) -> int:
        """Number of features."""
        return len(self.feature_order)

    @computed_field
    @property
    def feature_order_hash(self) -> str:
        """Deterministic hash of feature order."""
        features_str = ",".join(self.feature_order)
        return hashlib.sha256(features_str.encode()).hexdigest()[:16]


class HyperparametersConfig(BaseModel):
    """PPO hyperparameters configuration."""

    # Learning parameters
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate")
    n_steps: int = Field(2048, gt=0, description="Steps per update")
    batch_size: int = Field(128, gt=0, description="Mini-batch size")
    n_epochs: int = Field(10, gt=0, description="Number of epochs per update")

    # Discount and advantage
    gamma: float = Field(0.95, ge=0, le=1, description="Discount factor (SSOT default)")
    gae_lambda: float = Field(0.95, ge=0, le=1, description="GAE lambda")

    # PPO-specific
    clip_range: float = Field(0.2, gt=0, le=1, description="PPO clip range")
    ent_coef: float = Field(0.05, ge=0, description="Entropy coefficient")
    vf_coef: float = Field(0.5, ge=0, description="Value function coefficient")
    max_grad_norm: float = Field(0.5, gt=0, description="Max gradient norm")

    # Training duration
    total_timesteps: int = Field(500000, gt=0, description="Total training timesteps")

    # Reproducibility
    random_seed: int = Field(42, description="Random seed for reproducibility")


class NetworkLayerConfig(BaseModel):
    """Neural network layer configuration."""

    layers: List[int] = Field([256, 256], min_length=1, description="Layer sizes")
    activation: str = Field("Tanh", description="Activation function")


class NetworkConfig(BaseModel):
    """Neural network architecture configuration."""

    policy_network: NetworkLayerConfig = Field(default_factory=NetworkLayerConfig)
    value_network: NetworkLayerConfig = Field(default_factory=NetworkLayerConfig)


class EnvironmentConfig(BaseModel):
    """Trading environment configuration."""

    episode_length: int = Field(1200, gt=0, description="Episode length in steps")
    initial_balance: float = Field(10000, gt=0, description="Initial trading balance")
    max_position: float = Field(1.0, gt=0, description="Maximum position size")
    max_drawdown_pct: float = Field(15.0, gt=0, le=100, description="Max drawdown percentage")
    transaction_cost_bps: float = Field(2.5, ge=0, description="Transaction cost in bps (MEXC: 2.5)")
    slippage_bps: float = Field(2.5, ge=0, description="Slippage in bps (MEXC: 2.5)")
    trading_start_hour: int = Field(13, ge=0, le=23, description="Trading start hour (UTC)")
    trading_end_hour: int = Field(17, ge=0, le=23, description="Trading end hour (UTC)")
    trading_end_minute: int = Field(55, ge=0, le=59, description="Trading end minute")


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = Field(True, description="Enable early stopping")
    patience: int = Field(10, gt=0, description="Patience epochs")
    min_delta: float = Field(0.01, ge=0, description="Minimum improvement")
    monitor: str = Field("episode_reward_mean", description="Metric to monitor")


class TrainingConfig(BaseModel):
    """Training process configuration."""

    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    log_interval: int = Field(1000, gt=0, description="Logging interval")
    eval_interval: int = Field(5000, gt=0, description="Evaluation interval")
    save_interval: int = Field(10000, gt=0, description="Model save interval")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    n_episodes: int = Field(100, gt=0, description="Number of evaluation episodes")
    deterministic: bool = Field(True, description="Use deterministic policy")
    metrics: List[str] = Field(
        default=[
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
        ],
        description="Metrics to compute",
    )


class PromotionThresholds(BaseModel):
    """Model promotion thresholds."""

    min_sharpe_ratio: float = Field(0.5, description="Minimum Sharpe ratio")
    min_win_rate: float = Field(0.45, ge=0, le=1, description="Minimum win rate")
    max_drawdown: float = Field(-0.15, le=0, description="Maximum drawdown (negative)")
    min_trades: int = Field(50, gt=0, description="Minimum number of trades")
    min_staging_days: Optional[int] = Field(None, description="Minimum days in staging")


class PromotionConfig(BaseModel):
    """Promotion configuration."""

    staging: PromotionThresholds = Field(default_factory=PromotionThresholds)
    production: PromotionThresholds = Field(
        default_factory=lambda: PromotionThresholds(
            min_sharpe_ratio=1.0,
            min_win_rate=0.50,
            max_drawdown=-0.10,
            min_trades=100,
            min_staging_days=7,
        )
    )


class MLflowConfig(BaseModel):
    """MLflow configuration."""

    experiment_name: str = Field("usdcop-rl-training", description="MLflow experiment name")
    model_name: str = Field("usdcop-ppo-model", description="Model registry name")
    tracking_uri: str = Field("http://localhost:5000", description="MLflow tracking URI")
    tags: Dict[str, str] = Field(default_factory=dict, description="MLflow tags")


class ExperimentMetadata(BaseModel):
    """Experiment metadata."""

    id: str = Field(..., description="Unique experiment ID")
    name: str = Field(..., description="Human-readable experiment name")
    description: str = Field("", description="Experiment description")
    version: str = Field("1.0.0", description="Experiment version")
    author: str = Field("Trading Team", description="Experiment author")
    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Creation date",
    )
    parent_experiment_id: Optional[str] = Field(
        None, description="Parent experiment ID for lineage"
    )
    tags: List[str] = Field(default_factory=list, description="Experiment tags")


class ExpectedResults(BaseModel):
    """Expected results from training."""

    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    notes: Optional[str] = None


# =============================================================================
# Main Experiment Configuration
# =============================================================================

class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    This is the root configuration model that contains all settings
    needed to fully reproduce an experiment.

    Example:
        config = ExperimentConfig.from_yaml("experiments/baseline/config.yaml")
        print(config.experiment_hash)
    """

    experiment: ExperimentMetadata
    dataset: DatasetConfig
    features: FeaturesConfig
    hyperparameters: HyperparametersConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    expected_results: Optional[ExpectedResults] = None

    @computed_field
    @property
    def experiment_hash(self) -> str:
        """
        Compute deterministic hash of experiment configuration.

        This hash uniquely identifies the experiment based on all
        configuration that affects training outcomes.
        """
        # Include only training-affecting fields
        hash_dict = {
            "dataset": {
                "version": self.dataset.version,
                "date_range": self.dataset.date_range.model_dump(),
                "train_ratio": self.dataset.train_ratio,
                "val_ratio": self.dataset.val_ratio,
            },
            "features": {
                "feature_order": self.features.feature_order,
                "normalization": self.features.normalization.model_dump(),
            },
            "hyperparameters": self.hyperparameters.model_dump(),
            "network": self.network.model_dump(),
            "environment": {
                "episode_length": self.environment.episode_length,
                "initial_balance": self.environment.initial_balance,
                "max_position": self.environment.max_position,
                "max_drawdown_pct": self.environment.max_drawdown_pct,
                "transaction_cost_bps": self.environment.transaction_cost_bps,
                "slippage_bps": self.environment.slippage_bps,
            },
        }

        # Deterministic JSON serialization
        json_str = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @computed_field
    @property
    def experiment_id_with_hash(self) -> str:
        """Get experiment ID including hash."""
        return f"{self.experiment.id}_{self.experiment_hash}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated ExperimentConfig instance
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)

    def diff(self, other: "ExperimentConfig") -> Dict[str, Tuple[Any, Any]]:
        """
        Get differences between this config and another.

        Args:
            other: Another ExperimentConfig to compare against

        Returns:
            Dictionary of {path: (self_value, other_value)} for different values
        """
        diffs = {}

        def compare_dicts(d1: Dict, d2: Dict, prefix: str = "") -> None:
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                path = f"{prefix}.{key}" if prefix else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if isinstance(v1, dict) and isinstance(v2, dict):
                    compare_dicts(v1, v2, path)
                elif v1 != v2:
                    diffs[path] = (v1, v2)

        compare_dicts(
            self.model_dump(exclude_none=True),
            other.model_dump(exclude_none=True),
        )

        return diffs

    def validate_for_training(self) -> List[str]:
        """
        Validate configuration is complete for training.

        Returns:
            List of validation warnings (empty if all good)
        """
        warnings = []

        # Check feature count matches observation dim
        expected_dim = 13  # From feature contract
        if self.features.feature_count != expected_dim:
            warnings.append(
                f"Feature count ({self.features.feature_count}) != expected ({expected_dim})"
            )

        # Check dataset version
        if not self.dataset.dvc_tag:
            warnings.append("Dataset DVC tag not specified (may affect reproducibility)")

        # Check parent experiment for non-baseline
        if self.experiment.id != "baseline" and not self.experiment.parent_experiment_id:
            warnings.append("Non-baseline experiment without parent_experiment_id")

        return warnings


# =============================================================================
# Factory Functions
# =============================================================================

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load and validate experiment configuration.

    Args:
        path: Path to config YAML file

    Returns:
        Validated ExperimentConfig
    """
    return ExperimentConfig.from_yaml(path)


def create_experiment_from_baseline(
    baseline_path: str | Path,
    experiment_id: str,
    name: str,
    description: str = "",
    **overrides: Any,
) -> ExperimentConfig:
    """
    Create a new experiment configuration from baseline.

    Args:
        baseline_path: Path to baseline config
        experiment_id: New experiment ID
        name: New experiment name
        description: Experiment description
        **overrides: Configuration overrides (nested dicts)

    Returns:
        New ExperimentConfig with overrides applied
    """
    baseline = ExperimentConfig.from_yaml(baseline_path)

    # Create new config data
    data = baseline.model_dump()

    # Update experiment metadata
    data["experiment"]["id"] = experiment_id
    data["experiment"]["name"] = name
    data["experiment"]["description"] = description
    data["experiment"]["parent_experiment_id"] = baseline.experiment.id
    data["experiment"]["created_at"] = datetime.now().strftime("%Y-%m-%d")

    # Apply overrides
    def apply_overrides(target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                apply_overrides(target[key], value)
            else:
                target[key] = value

    apply_overrides(data, overrides)

    return ExperimentConfig.model_validate(data)


__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "DatasetConfig",
    "DateRangeConfig",
    "FeaturesConfig",
    "HyperparametersConfig",
    "NetworkConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PromotionConfig",
    "MLflowConfig",
    "ExpectedResults",
    "load_experiment_config",
    "create_experiment_from_baseline",
]
