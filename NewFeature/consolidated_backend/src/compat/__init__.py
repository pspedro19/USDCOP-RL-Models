"""
Compatibility Layer for NewFeature → SSOT Migration
====================================================

This module redirects imports from NewFeature's local modules to the
main project's SSOT (Single Source of Truth) modules.

IMPORTANT: Do NOT add new code here. All new development should go
directly to src/ modules.

Migration Status:
    - minio_client → src.mlops.minio_client (SSOT)
    - walk_forward_backtest → src.forecasting.evaluation.walk_forward (SSOT)
    - metrics → src.forecasting.evaluation.metrics (SSOT)
    - model_factory → src.forecasting.models.factory (SSOT)

Usage:
    # Instead of:
    from src.mlops.minio_client import MinIOClient

    # Use:
    from compat import MinIOClient  # Redirects to SSOT

Author: Trading Team
Date: 2026-01-22
"""

import sys
import warnings
from pathlib import Path

# Add main project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Up to USDCOP-RL-Models
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MINIO CLIENT - Redirect to SSOT
# =============================================================================

def _deprecation_warning(old_module: str, new_module: str):
    """Emit deprecation warning for redirected imports."""
    warnings.warn(
        f"Importing from '{old_module}' is deprecated. "
        f"Use '{new_module}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


try:
    from src.mlops.minio_client import MinIOClient
    _deprecation_warning("NewFeature.src.mlops.minio_client", "src.mlops.minio_client")
except ImportError:
    MinIOClient = None


# =============================================================================
# FORECASTING CONTRACTS - Redirect to SSOT
# =============================================================================

try:
    from src.forecasting.contracts import (
        HORIZONS,
        MODEL_IDS,
        MODEL_DEFINITIONS,
        HORIZON_LABELS,
        HORIZON_CONFIGS,
        WF_CONFIG,
        ForecastDirection,
        ModelType,
        HorizonCategory,
        EnsembleType,
        ForecastingTrainingRequest,
        ForecastingTrainingResult,
        ForecastingInferenceRequest,
        ForecastingInferenceResult,
        ForecastPrediction,
        ModelMetrics,
        get_horizon_config,
        validate_model_id,
        validate_horizon,
        get_model_info,
        FORECASTING_CONTRACT_VERSION,
        FORECASTING_CONTRACT_HASH,
    )
    _deprecation_warning("NewFeature.src.contracts", "src.forecasting.contracts")
except ImportError:
    pass


# =============================================================================
# FORECASTING ENGINE - Redirect to SSOT
# =============================================================================

try:
    from src.forecasting.engine import ForecastingEngine
    _deprecation_warning("NewFeature.src.engine", "src.forecasting.engine")
except ImportError:
    ForecastingEngine = None


# =============================================================================
# EVALUATION METRICS - Redirect to SSOT
# =============================================================================

try:
    from src.forecasting.evaluation.metrics import (
        calculate_direction_accuracy,
        calculate_rmse,
        calculate_mae,
        calculate_mape,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_profit_factor,
    )
    _deprecation_warning("NewFeature.src.evaluation.metrics", "src.forecasting.evaluation.metrics")
except ImportError:
    pass


# =============================================================================
# WALK-FORWARD VALIDATION - Redirect to SSOT
# =============================================================================

try:
    from src.forecasting.evaluation.walk_forward import (
        WalkForwardValidator,
        walk_forward_split,
        expanding_window_split,
    )
    _deprecation_warning("NewFeature.src.evaluation.walk_forward", "src.forecasting.evaluation.walk_forward")
except ImportError:
    pass


# =============================================================================
# MODEL FACTORY - Redirect to SSOT
# =============================================================================

try:
    from src.forecasting.models.factory import (
        ModelFactory,
        create_model,
    )
    _deprecation_warning("NewFeature.src.models.factory", "src.forecasting.models.factory")
except ImportError:
    pass


__all__ = [
    # MinIO
    "MinIOClient",
    # Contracts
    "HORIZONS",
    "MODEL_IDS",
    "MODEL_DEFINITIONS",
    "HORIZON_LABELS",
    "HORIZON_CONFIGS",
    "WF_CONFIG",
    "ForecastDirection",
    "ModelType",
    "HorizonCategory",
    "EnsembleType",
    "ForecastingTrainingRequest",
    "ForecastingTrainingResult",
    "ForecastingInferenceRequest",
    "ForecastingInferenceResult",
    "ForecastPrediction",
    "ModelMetrics",
    "get_horizon_config",
    "validate_model_id",
    "validate_horizon",
    "get_model_info",
    "FORECASTING_CONTRACT_VERSION",
    "FORECASTING_CONTRACT_HASH",
    # Engine
    "ForecastingEngine",
    # Evaluation
    "calculate_direction_accuracy",
    "calculate_rmse",
    "calculate_mae",
    "calculate_mape",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_profit_factor",
    "WalkForwardValidator",
    "walk_forward_split",
    "expanding_window_split",
    # Models
    "ModelFactory",
    "create_model",
]
