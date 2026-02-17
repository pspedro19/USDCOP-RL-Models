"""
Forecasting Contracts (SSOT)
============================

Single Source of Truth for forecasting pipeline contracts.
All forecasting-related constants, types, and configurations.

Follows same pattern as src/core/contracts/feature_contract.py

@version 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

_logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS (SSOT)
# =============================================================================

class ForecastDirection(str, Enum):
    """Direction of price movement prediction."""
    UP = "UP"
    DOWN = "DOWN"


class ModelType(str, Enum):
    """Model category for forecasting."""
    LINEAR = "linear"
    BOOSTING = "boosting"
    HYBRID = "hybrid"


class HorizonCategory(str, Enum):
    """Horizon classification."""
    SHORT = "short"    # 1-5 days
    MEDIUM = "medium"  # 10-20 days
    LONG = "long"      # 25-30 days


class EnsembleType(str, Enum):
    """Ensemble strategy types."""
    BEST_OF_BREED = "best_of_breed"
    TOP_3 = "top_3"
    TOP_6_MEAN = "top_6_mean"
    CONSENSUS = "consensus"


# =============================================================================
# CONSTANTS (SSOT) — YAML-first with hardcoded fallback
# =============================================================================

# Try loading from forecasting_ssot.yaml; fall back to hardcoded values
try:
    from src.forecasting.ssot_config import ForecastingSSOTConfig as _SSOTCfg
    _cfg = _SSOTCfg.load()
    HORIZONS: Tuple[int, ...] = _cfg.get_horizons()
    _logger.debug("[contracts] Loaded HORIZONS from YAML: %s", HORIZONS)
except Exception as _e:
    _logger.debug("[contracts] YAML load failed (%s), using hardcoded fallback", _e)
    # Hardcoded fallback (same values as forecasting_ssot.yaml)
    HORIZONS: Tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30)

# Model definitions
MODEL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Linear Models
    "ridge": {
        "name": "Ridge Regression",
        "type": ModelType.LINEAR,
        "requires_scaling": True,
        "supports_early_stopping": False,
    },
    "bayesian_ridge": {
        "name": "Bayesian Ridge",
        "type": ModelType.LINEAR,
        "requires_scaling": True,
        "supports_early_stopping": False,
    },
    "ard": {
        "name": "ARD Regression",
        "type": ModelType.LINEAR,
        "requires_scaling": True,
        "supports_early_stopping": False,
    },
    # Pure Boosting Models
    "xgboost_pure": {
        "name": "XGBoost",
        "type": ModelType.BOOSTING,
        "requires_scaling": False,
        "supports_early_stopping": True,
    },
    "lightgbm_pure": {
        "name": "LightGBM",
        "type": ModelType.BOOSTING,
        "requires_scaling": False,
        "supports_early_stopping": True,
    },
    "catboost_pure": {
        "name": "CatBoost",
        "type": ModelType.BOOSTING,
        "requires_scaling": False,
        "supports_early_stopping": True,
    },
    # Hybrid Models
    "hybrid_xgboost": {
        "name": "XGBoost Hybrid",
        "type": ModelType.HYBRID,
        "requires_scaling": True,
        "supports_early_stopping": True,
    },
    "hybrid_lightgbm": {
        "name": "LightGBM Hybrid",
        "type": ModelType.HYBRID,
        "requires_scaling": True,
        "supports_early_stopping": True,
    },
    "hybrid_catboost": {
        "name": "CatBoost Hybrid",
        "type": ModelType.HYBRID,
        "requires_scaling": True,
        "supports_early_stopping": True,
    },
}

# All model IDs
MODEL_IDS: Tuple[str, ...] = tuple(MODEL_DEFINITIONS.keys())

# Horizon labels
HORIZON_LABELS: Dict[int, str] = {
    1: "1 day",
    5: "5 days",
    10: "10 days",
    15: "15 days",
    20: "20 days",
    25: "25 days",
    30: "30 days",
}

# Horizon categories
HORIZON_CATEGORIES: Dict[int, HorizonCategory] = {
    1: HorizonCategory.SHORT,
    5: HorizonCategory.SHORT,
    10: HorizonCategory.MEDIUM,
    15: HorizonCategory.MEDIUM,
    20: HorizonCategory.MEDIUM,
    25: HorizonCategory.LONG,
    30: HorizonCategory.LONG,
}

# Walk-forward validation config
try:
    WF_CONFIG = _cfg.get_wf_config()
except Exception:
    WF_CONFIG = {
        "n_windows": 5,
        "min_train_pct": 0.4,
        "gap_days": 30,
        "step_pct": 0.1,
    }

# Horizon-adaptive hyperparameters (SSOT)
try:
    HORIZON_CONFIGS: Dict[str, Dict[str, Any]] = {
        cat: dict(_cfg.raw["horizons"]["configs"][cat])
        for cat in ["short", "medium", "long"]
    }
except Exception:
    HORIZON_CONFIGS: Dict[str, Dict[str, Any]] = {
        "short": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "min_samples_leaf": 20,
        },
        "medium": {
            "n_estimators": 30,
            "max_depth": 2,
            "learning_rate": 0.08,
            "colsample_bytree": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "min_samples_leaf": 30,
        },
        "long": {
            "n_estimators": 20,
            "max_depth": 1,
            "learning_rate": 0.1,
            "colsample_bytree": 0.4,
            "reg_alpha": 2.0,
            "reg_lambda": 3.0,
            "min_samples_leaf": 50,
        },
    }


def get_horizon_config(horizon: int) -> Dict[str, Any]:
    """Get horizon-specific hyperparameters."""
    category = HORIZON_CATEGORIES.get(horizon, HorizonCategory.MEDIUM)
    return HORIZON_CONFIGS[category.value]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForecastingTrainingRequest:
    """Request to train forecasting models."""
    dataset_path: Optional[str] = None  # Path to dataset file (if not using SSOT)
    use_db: bool = False  # v2.0: If True, load from SSOT (PostgreSQL/Parquet)
    version: str = "auto"
    experiment_name: str = "forecasting_usdcop"
    models: Optional[List[str]] = None  # None = all models
    horizons: Optional[List[int]] = None  # None = all horizons
    mlflow_enabled: bool = True
    mlflow_tracking_uri: Optional[str] = None
    minio_enabled: bool = True
    db_connection_string: Optional[str] = None
    walk_forward_windows: int = 5

    def __post_init__(self):
        if self.models is None:
            self.models = list(MODEL_IDS)
        if self.horizons is None:
            self.horizons = list(HORIZONS)
        # Validate: either dataset_path or use_db must be specified
        if not self.dataset_path and not self.use_db:
            raise ValueError("Either dataset_path or use_db=True must be specified")


@dataclass
class ForecastingTrainingResult:
    """Result of forecasting training."""
    success: bool
    version: str
    models_trained: int
    total_combinations: int  # models × horizons
    best_model_per_horizon: Dict[int, str]
    metrics_summary: Dict[str, Dict[int, float]]  # model → horizon → DA
    model_artifacts_path: str
    mlflow_experiment_id: Optional[str] = None
    mlflow_run_ids: Dict[str, str] = field(default_factory=dict)
    minio_artifacts_uri: Optional[str] = None
    training_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "version": self.version,
            "models_trained": self.models_trained,
            "total_combinations": self.total_combinations,
            "best_model_per_horizon": self.best_model_per_horizon,
            "metrics_summary": self.metrics_summary,
            "model_artifacts_path": self.model_artifacts_path,
            "mlflow_experiment_id": self.mlflow_experiment_id,
            "mlflow_run_ids": self.mlflow_run_ids,
            "minio_artifacts_uri": self.minio_artifacts_uri,
            "training_duration_seconds": self.training_duration_seconds,
            "errors": self.errors,
        }


@dataclass
class ForecastingInferenceRequest:
    """Request to run forecasting inference."""
    inference_date: str  # YYYY-MM-DD
    models: Optional[List[str]] = None
    horizons: Optional[List[int]] = None
    generate_ensembles: bool = True
    upload_images: bool = True
    persist_to_db: bool = True

    def __post_init__(self):
        if self.models is None:
            self.models = list(MODEL_IDS)
        if self.horizons is None:
            self.horizons = list(HORIZONS)


@dataclass
class ForecastPrediction:
    """Single forecast prediction."""
    model_id: str
    horizon: int
    inference_date: str
    target_date: str
    base_price: float
    predicted_price: float
    predicted_return_pct: float
    direction: ForecastDirection
    signal: int  # -1=SELL, 0=HOLD, 1=BUY
    confidence: Optional[float] = None


@dataclass
class ForecastingInferenceResult:
    """Result of forecasting inference."""
    success: bool
    inference_date: str
    inference_week: int
    inference_year: int
    predictions: List[ForecastPrediction]
    ensembles: Dict[str, Dict[int, ForecastPrediction]]
    consensus_by_horizon: Dict[int, Dict[str, Any]]
    minio_week_path: Optional[str] = None
    images_uploaded: int = 0
    forecasts_persisted: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class ModelMetrics:
    """Metrics for a model/horizon combination."""
    model_id: str
    horizon: int
    direction_accuracy: float
    rmse: float
    mae: float
    mape: Optional[float] = None
    r2: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    sample_count: int = 0


# =============================================================================
# HASH UTILITIES
# =============================================================================

def compute_contract_hash() -> str:
    """Compute hash of this contract for versioning."""
    contract_data = {
        "horizons": HORIZONS,
        "model_ids": MODEL_IDS,
        "wf_config": WF_CONFIG,
        "horizon_configs": HORIZON_CONFIGS,
    }
    content = json.dumps(contract_data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Contract version and hash
FORECASTING_CONTRACT_VERSION = "1.0.0"
FORECASTING_CONTRACT_HASH = compute_contract_hash()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model_id(model_id: str) -> bool:
    """Check if model_id is valid."""
    return model_id in MODEL_IDS


def validate_horizon(horizon: int) -> bool:
    """Check if horizon is valid."""
    return horizon in HORIZONS


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model definition by ID."""
    return MODEL_DEFINITIONS.get(model_id)


__all__ = [
    # Enums
    "ForecastDirection",
    "ModelType",
    "HorizonCategory",
    "EnsembleType",
    # Constants
    "HORIZONS",
    "MODEL_IDS",
    "MODEL_DEFINITIONS",
    "HORIZON_LABELS",
    "HORIZON_CATEGORIES",
    "HORIZON_CONFIGS",
    "WF_CONFIG",
    "FORECASTING_CONTRACT_VERSION",
    "FORECASTING_CONTRACT_HASH",
    # Data classes
    "ForecastingTrainingRequest",
    "ForecastingTrainingResult",
    "ForecastingInferenceRequest",
    "ForecastingInferenceResult",
    "ForecastPrediction",
    "ModelMetrics",
    # Functions
    "get_horizon_config",
    "validate_model_id",
    "validate_horizon",
    "get_model_info",
    "compute_contract_hash",
]
