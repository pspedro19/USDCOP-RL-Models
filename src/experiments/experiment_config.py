"""
Experiment Configuration Models
===============================

Pydantic models for experiment configuration validation.
Enforces SSOT contracts and provides type-safe configuration access.

Author: Trading Team
Date: 2026-01-17
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
from enum import Enum

# Import SSOT contracts
from src.core.contracts import OBSERVATION_DIM, FEATURE_ORDER


class Algorithm(str, Enum):
    """Supported RL algorithms."""
    PPO = "PPO"
    A2C = "A2C"
    SAC = "SAC"
    TD3 = "TD3"
    DQN = "DQN"


class PolicyType(str, Enum):
    """Supported policy networks."""
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"
    MULTI_INPUT = "MultiInputPolicy"


class ActivationFn(str, Enum):
    """Supported activation functions."""
    TANH = "tanh"
    RELU = "relu"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"


class RewardFunction(str, Enum):
    """Supported reward functions."""
    SHARPE = "sharpe"
    PNL = "pnl"
    RISK_ADJUSTED = "risk_adjusted"
    CUSTOM = "custom"


class DataSource(str, Enum):
    """Supported data sources."""
    FEAST = "feast"
    PARQUET = "parquet"
    CSV = "csv"
    DATABASE = "database"


class Metric(str, Enum):
    """Supported evaluation metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    AVG_TRADE_PNL = "avg_trade_pnl"
    TRADE_COUNT = "trade_count"
    AVG_HOLDING_PERIOD = "avg_holding_period"


# =============================================================================
# Sub-configurations
# =============================================================================

class ExperimentMetadata(BaseModel):
    """Experiment metadata configuration."""
    name: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-z0-9_-]+$")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = Field(None, max_length=500)
    hypothesis: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list)
    baseline_experiment: Optional[str] = None
    owner: Optional[str] = None


class PolicyKwargs(BaseModel):
    """Policy network configuration."""
    net_arch: List[int] = Field(default=[64, 64])
    activation_fn: ActivationFn = ActivationFn.TANH
    ortho_init: bool = True

    @field_validator("net_arch")
    @classmethod
    def validate_net_arch(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("net_arch must have at least one layer")
        if any(x <= 0 for x in v):
            raise ValueError("All layer sizes must be positive")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    algorithm: Algorithm = Algorithm.PPO
    policy: PolicyType = PolicyType.MLP
    policy_kwargs: Optional[PolicyKwargs] = None


class LearningRateSchedule(BaseModel):
    """Learning rate schedule configuration."""
    type: Literal["linear", "exponential", "cosine"]
    initial: float = Field(..., gt=0, le=1)
    final: float = Field(..., ge=0, le=1)


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""
    total_timesteps: int = Field(..., ge=1000)
    learning_rate: Union[float, LearningRateSchedule] = 0.0003
    n_steps: int = Field(2048, ge=1)
    batch_size: int = Field(64, ge=1)
    n_epochs: int = Field(10, ge=1)
    gamma: float = Field(0.95, ge=0, le=1)  # SSOT: experiment_ssot.yaml
    gae_lambda: float = Field(0.95, ge=0, le=1)
    clip_range: float = Field(0.2, ge=0, le=1)
    clip_range_vf: Optional[float] = None
    ent_coef: float = Field(0.0, ge=0)
    vf_coef: float = Field(0.5, ge=0)
    max_grad_norm: float = Field(0.5, ge=0)
    seed: Optional[int] = None
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v):
        if isinstance(v, (int, float)):
            if not 0 < v <= 1:
                raise ValueError("learning_rate must be between 0 and 1")
        return v


class RewardKwargs(BaseModel):
    """Reward function configuration."""
    risk_free_rate: float = 0.0
    annualization_factor: float = 252
    penalty_holding: float = 0.0
    penalty_trading: float = 0.0001


class NormalizationConfig(BaseModel):
    """Feature normalization configuration."""
    enabled: bool = True
    stats_path: str = "config/norm_stats.json"
    clip_range: float = 10.0


class ImputationStrategy(str, Enum):
    """Allowed imputation strategies (NO bfill/ffill to prevent data leakage)."""
    NONE = "none"           # No imputation - fail on NaN
    MEAN = "mean"           # Use feature mean (computed on train only)
    MEDIAN = "median"       # Use feature median (computed on train only)
    ZERO = "zero"           # Fill with zeros
    DROP = "drop"           # Drop rows with NaN


class PreprocessingConfig(BaseModel):
    """
    Preprocessing configuration with data leakage prevention.

    CRITICAL: Enforces best practices to prevent look-ahead bias:
    - NO forward fill (ffill) - causes look-ahead bias
    - NO backward fill (bfill) - causes look-ahead bias
    - Normalization stats computed on train set ONLY
    - No future information in feature computation

    GAP 7: Validates that preprocessing won't cause data leakage.
    """

    # Imputation strategy (no ffill/bfill allowed)
    imputation_strategy: ImputationStrategy = Field(
        ImputationStrategy.NONE,
        description="Strategy for handling missing values. bfill/ffill NOT allowed."
    )

    # Normalization scope - MUST be train_only
    normalization_scope: Literal["train_only", "all_data"] = Field(
        "train_only",
        description="Scope for computing normalization stats. MUST be 'train_only'."
    )

    # Explicit flags for dangerous operations (must be False)
    allow_forward_fill: bool = Field(
        False,
        description="MUST be False. Forward fill causes look-ahead bias."
    )
    allow_backward_fill: bool = Field(
        False,
        description="MUST be False. Backward fill causes data leakage."
    )

    # Optional: max NaN percentage allowed
    max_nan_percentage: float = Field(
        0.05,
        ge=0,
        le=1.0,
        description="Maximum allowed NaN percentage per feature (0.05 = 5%)"
    )

    # Optional: outlier handling
    clip_outliers: bool = Field(
        True,
        description="Whether to clip outliers to clip_range standard deviations"
    )
    clip_range_std: float = Field(
        10.0,
        ge=1.0,
        le=100.0,
        description="Number of standard deviations for outlier clipping"
    )

    @model_validator(mode="after")
    def validate_no_data_leakage(self):
        """
        CRITICAL: Ensure preprocessing won't cause data leakage.

        This validator blocks configurations that could cause:
        - Look-ahead bias (using future data in features)
        - Data leakage (using test data in training)
        """
        # Block forward fill
        if self.allow_forward_fill:
            raise ValueError(
                "BLOCKED: allow_forward_fill=True causes look-ahead bias. "
                "Forward fill uses future values to fill past NaNs, "
                "leaking future information into training data."
            )

        # Block backward fill
        if self.allow_backward_fill:
            raise ValueError(
                "BLOCKED: allow_backward_fill=True causes data leakage. "
                "Backward fill uses past values incorrectly in time-series context."
            )

        # Block all_data normalization
        if self.normalization_scope == "all_data":
            raise ValueError(
                "BLOCKED: normalization_scope='all_data' causes data leakage. "
                "Normalization statistics MUST be computed on train set only. "
                "Using all_data means test data statistics leak into training."
            )

        return self

    @field_validator("imputation_strategy")
    @classmethod
    def validate_imputation_strategy(cls, v: ImputationStrategy) -> ImputationStrategy:
        """Validate imputation strategy is safe."""
        # These are explicitly NOT in the enum, but double-check
        unsafe_strategies = ["ffill", "bfill", "forward_fill", "backward_fill", "interpolate"]
        if v.value.lower() in unsafe_strategies:
            raise ValueError(
                f"BLOCKED: Imputation strategy '{v.value}' causes data leakage. "
                f"Use one of: {[s.value for s in ImputationStrategy]}"
            )
        return v


class EnvironmentConfig(BaseModel):
    """Trading environment configuration."""
    observation_dim: int = Field(default=OBSERVATION_DIM)  # Validated by validator below
    action_type: Literal["continuous", "discrete"] = "continuous"
    reward_function: RewardFunction = RewardFunction.SHARPE
    reward_kwargs: Optional[RewardKwargs] = None
    normalization: Optional[NormalizationConfig] = None

    @field_validator("observation_dim")
    @classmethod
    def validate_observation_dim(cls, v: int) -> int:
        if v != OBSERVATION_DIM:
            raise ValueError(
                f"observation_dim must match SSOT OBSERVATION_DIM={OBSERVATION_DIM}"
            )
        return v


class DataConfig(BaseModel):
    """Data configuration with preprocessing validation."""
    source: DataSource = DataSource.FEAST
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    validation_split: float = Field(0.1, ge=0, le=0.5)
    feature_set: Literal["v1", "v2", "custom"] = "v1"
    custom_features: Optional[List[str]] = None

    # Preprocessing config (GAP 7: validates no bfill/ffill)
    preprocessing: Optional[PreprocessingConfig] = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing configuration with data leakage prevention"
    )

    @model_validator(mode="after")
    def validate_dates(self):
        if self.train_start and self.train_end:
            if self.train_start >= self.train_end:
                raise ValueError("train_start must be before train_end")
        return self

    @model_validator(mode="after")
    def validate_custom_features(self):
        if self.feature_set == "custom" and not self.custom_features:
            raise ValueError("custom_features required when feature_set='custom'")
        return self

    @model_validator(mode="after")
    def validate_preprocessing_defaults(self):
        """Ensure preprocessing config exists with safe defaults."""
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        return self


class BacktestConfig(BaseModel):
    """Backtest configuration."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = Field(100000, ge=0)
    position_size: float = Field(1.0, ge=0, le=1)
    slippage_bps: float = Field(5, ge=0)
    commission_bps: float = Field(10, ge=0)


class StatisticalTestsConfig(BaseModel):
    """Statistical testing configuration."""
    significance_level: float = Field(0.05, ge=0, le=1)
    min_samples: int = Field(30, ge=1)
    bootstrap_iterations: int = Field(1000, ge=100)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: List[Metric] = Field(..., min_length=1)
    primary_metric: Metric = Metric.SHARPE_RATIO
    backtest: Optional[BacktestConfig] = None
    statistical_tests: Optional[StatisticalTestsConfig] = None

    @model_validator(mode="after")
    def validate_primary_metric(self):
        if self.primary_metric not in self.metrics:
            self.metrics.append(self.primary_metric)
        return self


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""
    experiment_name: Optional[str] = None
    tracking_uri: str = "http://localhost:5000"
    log_model: bool = True
    log_artifacts: List[str] = Field(default=["model", "config", "metrics"])


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    enabled: bool = False
    patience: int = Field(10, ge=1)
    min_delta: float = Field(0.01, ge=0)


class CallbacksConfig(BaseModel):
    """Training callbacks configuration."""
    eval_freq: int = Field(10000, ge=100)
    save_freq: int = Field(50000, ge=100)
    early_stopping: Optional[EarlyStoppingConfig] = None
    tensorboard: bool = True


# =============================================================================
# Main Experiment Configuration
# =============================================================================

class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    This is the main configuration class that contains all settings
    for running a training experiment.

    Example:
        config = ExperimentConfig.from_yaml("config/experiments/my_exp.yaml")
        print(config.experiment.name)
        print(config.training.total_timesteps)
    """

    experiment: ExperimentMetadata
    model: ModelConfig
    training: TrainingConfig
    environment: Optional[EnvironmentConfig] = None
    data: Optional[DataConfig] = None
    evaluation: EvaluationConfig
    mlflow: Optional[MLflowConfig] = None
    callbacks: Optional[CallbacksConfig] = None

    @model_validator(mode="after")
    def set_defaults(self):
        """Set default sub-configs if not provided."""
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.mlflow is None:
            self.mlflow = MLflowConfig(experiment_name=self.experiment.name)
        if self.callbacks is None:
            self.callbacks = CallbacksConfig()
        return self

    def get_mlflow_experiment_name(self) -> str:
        """Get MLflow experiment name."""
        if self.mlflow and self.mlflow.experiment_name:
            return self.mlflow.experiment_name
        return self.experiment.name

    def get_run_name(self) -> str:
        """Get unique run name."""
        return f"{self.experiment.name}_v{self.experiment.version}"

    def to_training_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for stable_baselines3."""
        kwargs = {
            "learning_rate": self.training.learning_rate
            if isinstance(self.training.learning_rate, float)
            else self.training.learning_rate.initial,
            "n_steps": self.training.n_steps,
            "batch_size": self.training.batch_size,
            "n_epochs": self.training.n_epochs,
            "gamma": self.training.gamma,
            "gae_lambda": self.training.gae_lambda,
            "clip_range": self.training.clip_range,
            "ent_coef": self.training.ent_coef,
            "vf_coef": self.training.vf_coef,
            "max_grad_norm": self.training.max_grad_norm,
            "seed": self.training.seed,
            "device": self.training.device,
        }

        if self.training.clip_range_vf is not None:
            kwargs["clip_range_vf"] = self.training.clip_range_vf

        if self.model.policy_kwargs:
            kwargs["policy_kwargs"] = {
                "net_arch": self.model.policy_kwargs.net_arch,
                "activation_fn": self._get_activation_fn(),
                "ortho_init": self.model.policy_kwargs.ortho_init,
            }

        return kwargs

    def _get_activation_fn(self):
        """Get torch activation function."""
        import torch.nn as nn

        mapping = {
            ActivationFn.TANH: nn.Tanh,
            ActivationFn.RELU: nn.ReLU,
            ActivationFn.ELU: nn.ELU,
            ActivationFn.LEAKY_RELU: nn.LeakyReLU,
        }
        if self.model.policy_kwargs:
            return mapping.get(self.model.policy_kwargs.activation_fn, nn.Tanh)
        return nn.Tanh

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_default = True


__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "ModelConfig",
    "TrainingConfig",
    "EnvironmentConfig",
    "DataConfig",
    "EvaluationConfig",
    "MLflowConfig",
    "CallbacksConfig",
    "Algorithm",
    "PolicyType",
    "Metric",
    # Preprocessing (GAP 7)
    "PreprocessingConfig",
    "ImputationStrategy",
]
