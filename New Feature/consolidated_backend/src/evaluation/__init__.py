# usdcop_forecasting_clean/backend/src/evaluation/__init__.py
"""
Evaluation and validation module.

Components:
- PurgedKFold, WalkForwardPurged: Temporal CV without look-ahead bias
- walk_forward_backtest: Walk-forward backtest with expanding window
- TrainingReporter: Visualization and reporting for training results
- Backtester: Legacy backtesting
- Metrics: DA, custom objectives for XGBoost/LightGBM/CatBoost
"""

from .backtest import Backtester
from .metrics import (
    direction_accuracy,
    xgb_direction_accuracy,
    lgb_direction_accuracy,
    CatBoostDirectionAccuracy
)
from .purged_kfold import (
    PurgedKFold,
    WalkForwardPurged,
    get_cv_for_horizon
)

# Walk-forward backtest
try:
    from .walk_forward_backtest import (
        walk_forward_backtest,
        backtest_all_models,
        create_backtest_summary_df,
        BacktestResult,
        WindowResult
    )
except ImportError:
    walk_forward_backtest = None
    backtest_all_models = None
    create_backtest_summary_df = None
    BacktestResult = None
    WindowResult = None

# Visualization and reporting
try:
    from .visualization import (
        TrainingReporter,
        create_training_report,
        TrainingMetrics
    )
except ImportError:
    TrainingReporter = None
    create_training_report = None
    TrainingMetrics = None

__all__ = [
    # Legacy
    'Backtester',
    # Metrics
    'direction_accuracy',
    'xgb_direction_accuracy',
    'lgb_direction_accuracy',
    'CatBoostDirectionAccuracy',
    # Cross-validation
    'PurgedKFold',
    'WalkForwardPurged',
    'get_cv_for_horizon',
    # Walk-forward backtest
    'walk_forward_backtest',
    'backtest_all_models',
    'create_backtest_summary_df',
    'BacktestResult',
    'WindowResult',
    # Visualization
    'TrainingReporter',
    'create_training_report',
    'TrainingMetrics',
]
