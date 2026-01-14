# USD/COP RL Trading System V19.2 Enhanced
# =========================================
# Main training package with curriculum learning, enhanced features, and factory pattern

__version__ = '19.2.0'
__author__ = 'Claude Code'

# Preprocessing (existing)
from .preprocessing import (
    DataPreprocessor,
    DataQualityConfig,
    ALL_FEATURES_CLEAN,
    diagnose_dataset,
)

# V19 Base Components
from .environment_v19 import (
    TradingEnvironmentV19,
    SETFXCostModel,
    VolatilityScaler,
    create_training_env,
    create_validation_env,
)

from .train_v19 import TrainingPipelineV19

# V19.1 Enhanced Components
from .environment_v19_enhanced import (
    TradingEnvironmentV19Enhanced,
    create_enhanced_training_env,
    create_enhanced_validation_env,
)

from .feedback_tracker import (
    FeedbackTracker,
    RegimeFeatureGenerator,
)

from .multi_seed_ensemble import (
    MultiSeedEnsemble,
    train_multi_seed_ensemble,
)

from .sortino_reward import (
    SortinoCalculator,
    SortinoRewardFunction,
    SortinoConfig,
    HybridSharpesortinoReward,
)

from .horizon_features import (
    MultiHorizonFeatures,
    DirectionalSignal,
    TrendRegimeClassifier,
    add_horizon_features,
)

from .risk_manager import (
    RiskManager,
    RiskLimits,
    RiskStatus,
    TradingLoopWithRiskManager,
)

# V19.2 Factory Pattern
from .factories import (
    EnvironmentFactory,
    EnvironmentConfig,
    CostModelConfig,
    VolatilityScalingConfig,
    RegimeDetectionConfig,
    EnhancedFeaturesConfig,
    RiskManagerConfig,
)

__all__ = [
    # Preprocessing
    'DataPreprocessor',
    'DataQualityConfig',
    'ALL_FEATURES_CLEAN',
    'diagnose_dataset',
    # V19 Base
    'TradingEnvironmentV19',
    'SETFXCostModel',
    'VolatilityScaler',
    'create_training_env',
    'create_validation_env',
    'TrainingPipelineV19',
    # V19.1 Enhanced Environment
    'TradingEnvironmentV19Enhanced',
    'create_enhanced_training_env',
    'create_enhanced_validation_env',
    # Feedback & Regime Features
    'FeedbackTracker',
    'RegimeFeatureGenerator',
    # Multi-Seed Ensemble
    'MultiSeedEnsemble',
    'train_multi_seed_ensemble',
    # Sortino Reward
    'SortinoCalculator',
    'SortinoRewardFunction',
    'SortinoConfig',
    'HybridSharpesortinoReward',
    # Horizon Features
    'MultiHorizonFeatures',
    'DirectionalSignal',
    'TrendRegimeClassifier',
    'add_horizon_features',
    # Risk Manager
    'RiskManager',
    'RiskLimits',
    'RiskStatus',
    'TradingLoopWithRiskManager',
    # V19.2 Factory Pattern
    'EnvironmentFactory',
    'EnvironmentConfig',
    'CostModelConfig',
    'VolatilityScalingConfig',
    'RegimeDetectionConfig',
    'EnhancedFeaturesConfig',
    'RiskManagerConfig',
]
