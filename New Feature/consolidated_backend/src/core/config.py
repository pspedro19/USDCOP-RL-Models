# pipeline_limpio_regresion/core/config.py
"""
Centralized configuration for ML-only pipeline with Optuna optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

HORIZONS = [1, 5, 10, 15, 20, 25, 30]
RANDOM_STATE = 42

# Model registry - ML ONLY (5 models - MLP excluded)
# MLP excluido: R²=-6859 indica varianza explosiva (variance_ratio=41x)
ML_MODELS = ['ridge', 'bayesian_ridge', 'xgboost', 'lightgbm', 'catboost']

# Best model (updated after Optuna optimization)
BEST_MODEL = 'ridge'  # Ridge is near-optimal for mean-reverting markets

# Ensemble weights (will be optimized based on DA)
# MLP excluido - pesos redistribuidos proporcionalmente
ENSEMBLE_WEIGHTS = {
    'ridge': 0.29,
    'bayesian_ridge': 0.24,
    'xgboost': 0.14,
    'lightgbm': 0.14,
    'catboost': 0.19
}

# Dynamic weights per horizon (Phase 3 Optimization)
# Updated: Ridge/Bayesian Ridge favored at long horizons (H>=15) based on expert analysis
# Boosting models collapse at longer horizons, linear models remain near-optimal
# MLP excluido - pesos redistribuidos proporcionalmente por horizonte
ENSEMBLE_WEIGHTS_BY_HORIZON = {
    1: {'ridge': 0.24, 'bayesian_ridge': 0.18, 'xgboost': 0.14, 'lightgbm': 0.14, 'catboost': 0.30},
    5: {'ridge': 0.29, 'bayesian_ridge': 0.21, 'xgboost': 0.12, 'lightgbm': 0.12, 'catboost': 0.26},
    10: {'ridge': 0.33, 'bayesian_ridge': 0.26, 'xgboost': 0.09, 'lightgbm': 0.09, 'catboost': 0.23},
    15: {'ridge': 0.38, 'bayesian_ridge': 0.33, 'xgboost': 0.07, 'lightgbm': 0.07, 'catboost': 0.15},
    20: {'ridge': 0.38, 'bayesian_ridge': 0.35, 'xgboost': 0.06, 'lightgbm': 0.06, 'catboost': 0.15},
    25: {'ridge': 0.38, 'bayesian_ridge': 0.38, 'xgboost': 0.06, 'lightgbm': 0.06, 'catboost': 0.12},
    30: {'ridge': 0.38, 'bayesian_ridge': 0.39, 'xgboost': 0.06, 'lightgbm': 0.06, 'catboost': 0.11}
}


class ModelType(Enum):
    """Model type enumeration."""
    ML = 'ml'
    ENSEMBLE = 'ensemble'


@dataclass
class OptunaConfig:
    """
    Optuna optimization configuration.
    OPTIMIZED: Stronger overfitting control, early stopping, dynamic gap.
    """
    n_trials: int = 50               # Trials per model/horizon
    n_splits: int = 5                # CV folds
    gap: int = 10                    # INCREASED from 5 (more conservative)
    min_gap: int = 10                # NEW: Minimum gap regardless of horizon
    metric: str = 'direction_accuracy'  # Primary metric

    # Overfitting control - MORE AGGRESSIVE
    # AUDITORIA 2025-01: Tolerancia reducida de 35% a 15% para mejor control
    overfitting_penalty: float = 0.5   # INCREASED from 0.3 (exponential penalty)
    overfitting_threshold: float = 0.15  # Max acceptable train-val gap (reducido de 0.35)

    # Timeouts
    timeout: int = 900               # 15 min per model

    # Early stopping for Optuna search
    early_stopping: bool = True
    early_stopping_patience: int = 8   # REDUCED from 15 (faster convergence)

    # Early stopping for boosting models - REDUCED para detener antes del overfitting
    model_early_stopping_rounds: int = 15  # CHANGED from 50 to 15

    n_jobs: int = 1                  # Parallel jobs (-1 for all cores)
    show_progress: bool = True

    def get_gap_for_horizon(self, horizon: int) -> int:
        """Dynamic gap: more conservative for short horizons."""
        return max(self.min_gap, horizon + self.gap)


@dataclass
class ModelConfig:
    """
    Configuration for a single model.
    """
    name: str
    model_type: ModelType
    requires_scaling: bool = False
    supports_optuna: bool = True
    default_params: Dict = field(default_factory=dict)
    optuna_space: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


@dataclass
class PipelineConfig:
    """
    Pipeline configuration - Single source of truth.
    """
    # Data configuration
    target_col: str = 'close'
    date_col: str = 'date'

    # Temporal split
    train_size: float = 0.8
    validation_size: float = 0.1

    # Prediction horizons
    horizons: List[int] = field(default_factory=lambda: HORIZONS)

    # Models to train (ML only)
    ml_models: List[str] = field(default_factory=lambda: ML_MODELS)

    # Feature selection - REDUCED to prevent overfitting (10-15 max for ~1500 samples)
    n_features: int = 15  # Era 50, ahora 15

    # Optuna configuration
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    use_optuna: bool = True  # Enable/disable Optuna tuning

    # Ensemble
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: ENSEMBLE_WEIGHTS)

    # Data validation
    min_samples: int = 252  # Minimum 1 year of data
    max_missing_pct: float = 0.05

    # Random state
    random_state: int = RANDOM_STATE

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path('results/regression'))

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.train_size < 1, "train_size must be between 0 and 1"
        assert len(self.horizons) > 0, "At least one horizon required"
        assert self.n_features > 0, "n_features must be positive"

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if isinstance(self.optuna, dict):
            self.optuna = OptunaConfig(**self.optuna)


# =============================================================================
# MODEL CONFIGURATIONS REGISTRY
# =============================================================================

ML_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'ridge': ModelConfig(
        name='ridge',
        model_type=ModelType.ML,
        requires_scaling=True,
        supports_optuna=True,
        default_params={
            'alpha': 1.0,
            'fit_intercept': True,
            'random_state': RANDOM_STATE
        },
        optuna_space={
            'alpha': ('float_log', 0.1, 100.0)  # Rango razonable para evitar over-shrinkage
        }
    ),
    'bayesian_ridge': ModelConfig(
        name='bayesian_ridge',
        model_type=ModelType.ML,
        requires_scaling=True,
        supports_optuna=True,
        default_params={
            'max_iter': 300,
            'alpha_1': 1e-6,   # Prior menos restrictivo para permitir varianza en predicciones
            'alpha_2': 1e-6,   # Prior menos restrictivo
            'lambda_1': 1e-6,  # Prior menos restrictivo
            'lambda_2': 1e-6,  # Prior menos restrictivo
            'fit_intercept': True,
            'compute_score': True
        },
        optuna_space={
            'alpha_1': ('float_log', 1e-8, 1e-4),  # Rango extendido hacia priors mas debiles
            'alpha_2': ('float_log', 1e-8, 1e-4),
            'lambda_1': ('float_log', 1e-8, 1e-4),
            'lambda_2': ('float_log', 1e-8, 1e-4),
            'max_iter': ('int', 100, 500)
        }
    ),
    'mlp': ModelConfig(
        name='mlp',
        model_type=ModelType.ML,
        requires_scaling=True,  # Critical for neural networks
        supports_optuna=True,
        default_params={
            'hidden_layer_sizes': (16,),       # REDUCED from (64, 32) - prevents overfitting with ~1500 samples
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.5,                      # INCREASED from 0.01 - stronger L2 regularization
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.0001,      # REDUCED from 0.001 - more stable training
            'max_iter': 200,                   # REDUCED from 500 - faster convergence
            'early_stopping': True,
            'validation_fraction': 0.2,        # INCREASED from 0.1 - better validation
            'n_iter_no_change': 5,             # REDUCED from 10 - earlier stopping
            'random_state': RANDOM_STATE,
            'verbose': False
        },
        optuna_space={
            # Conservative architecture options for small datasets
            'hidden_layer_sizes': ('categorical', [(8,), (16,), (24,), (16, 8)]),
            'alpha': ('float_log', 0.1, 2.0),              # Strong regularization range
            'learning_rate_init': ('float_log', 1e-5, 1e-3),  # Lower learning rates
            'max_iter': ('int', 100, 300),                 # Reduced iterations
            'validation_fraction': ('float', 0.15, 0.25)   # Larger validation split
        }
    ),
    'xgboost': ModelConfig(
        name='xgboost',
        model_type=ModelType.ML,
        requires_scaling=False,
        supports_optuna=True,
        default_params={
            'n_estimators': 100,         # RELAXED from 40 (was collapsing)
            'max_depth': 4,              # RELAXED from 2 (was collapsing)
            'learning_rate': 0.05,       # RELAXED from 0.01 (was too slow)
            'random_state': RANDOM_STATE,
            'verbosity': 0,
            # Anti-overfitting defaults - RELAXED
            'reg_alpha': 0.5,            # RELAXED from 1.0
            'reg_lambda': 0.5,           # RELAXED from 1.0
            'subsample': 0.8,            # Row sampling
            'colsample_bytree': 0.8,     # Feature sampling
            'min_child_weight': 3,       # RELAXED from 5
            'gamma': 0.1                 # Min loss reduction to split
        },
        optuna_space={
            'n_estimators': ('int', 50, 150),      # RELAXED range
            'max_depth': ('int', 3, 6),            # RELAXED range
            'learning_rate': ('float_log', 0.01, 0.1),  # RELAXED range
            'subsample': ('float', 0.6, 0.9),
            'colsample_bytree': ('float', 0.6, 0.9),
            'reg_alpha': ('float_log', 0.1, 5.0),  # RELAXED max
            'reg_lambda': ('float_log', 0.1, 5.0), # RELAXED max
            'min_child_weight': ('int', 1, 7),     # RELAXED range
            'gamma': ('float_log', 0.01, 0.5)      # RELAXED max
        }
    ),
    'lightgbm': ModelConfig(
        name='lightgbm',
        model_type=ModelType.ML,
        requires_scaling=False,
        supports_optuna=True,
        default_params={
            'n_estimators': 100,         # RELAXED from 40 (was collapsing)
            'max_depth': 4,              # RELAXED from 2 (was collapsing)
            'learning_rate': 0.05,       # RELAXED from 0.01 (was too slow)
            'random_state': RANDOM_STATE,
            'verbose': -1,
            # Anti-overfitting defaults - RELAXED
            'lambda_l1': 0.5,            # RELAXED from 1.0 (reg_alpha)
            'lambda_l2': 0.5,            # RELAXED from 1.0 (reg_lambda)
            'subsample': 0.8,            # Row sampling
            'colsample_bytree': 0.8,     # Feature sampling
            'min_data_in_leaf': 5,       # RELAXED from 10 (min_child_samples)
            'min_split_gain': 0.1,       # Min gain to split
            'num_leaves': 31             # RELAXED from 15
        },
        optuna_space={
            'n_estimators': ('int', 50, 150),      # RELAXED range
            'max_depth': ('int', 3, 6),            # RELAXED range
            'learning_rate': ('float_log', 0.01, 0.1),  # RELAXED range
            'subsample': ('float', 0.6, 0.9),
            'colsample_bytree': ('float', 0.6, 0.9),
            'lambda_l1': ('float_log', 0.1, 5.0),  # RELAXED max
            'lambda_l2': ('float_log', 0.1, 5.0),  # RELAXED max
            'min_data_in_leaf': ('int', 3, 15),    # RELAXED range
            'num_leaves': ('int', 15, 63)          # RELAXED range
        }
    ),
    'catboost': ModelConfig(
        name='catboost',
        model_type=ModelType.ML,
        requires_scaling=False,
        supports_optuna=True,
        default_params={
            'iterations': 100,           # RELAXED from 40 (was collapsing)
            'depth': 4,                  # RELAXED from 2 (was collapsing)
            'learning_rate': 0.05,       # RELAXED from 0.01 (was too slow)
            'random_state': RANDOM_STATE,
            'verbose': False,
            'allow_writing_files': False,
            # Anti-overfitting defaults - RELAXED
            'l2_leaf_reg': 1.0,          # RELAXED from 3.0
            'random_strength': 1.0,      # Noise injection
            'bagging_temperature': 0.5,  # Bayesian bootstrap
            'subsample': 0.8,            # Row sampling
            'min_data_in_leaf': 5        # RELAXED from 10
        },
        optuna_space={
            'iterations': ('int', 50, 150),        # RELAXED range
            'depth': ('int', 3, 6),                # RELAXED range
            'learning_rate': ('float_log', 0.01, 0.1),  # RELAXED range
            'l2_leaf_reg': ('float_log', 0.5, 5.0),     # RELAXED range
            'random_strength': ('float_log', 0.5, 3.0), # RELAXED
            'bagging_temperature': ('float', 0.2, 1.0),
            'subsample': ('float', 0.6, 0.9),
            'min_data_in_leaf': ('int', 3, 15)     # RELAXED range
        }
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration.

    Args:
        model_name: Name of the model

    Returns:
        ModelConfig for the specified model

    Raises:
        ValueError: If model not found
    """
    if model_name in ML_MODEL_CONFIGS:
        return ML_MODEL_CONFIGS[model_name]
    else:
        raise ValueError(
            f"Model not found: {model_name}. "
            f"Available: {list(ML_MODEL_CONFIGS.keys())}"
        )


def get_all_model_names() -> List[str]:
    """Get all available model names."""
    return list(ML_MODEL_CONFIGS.keys())
