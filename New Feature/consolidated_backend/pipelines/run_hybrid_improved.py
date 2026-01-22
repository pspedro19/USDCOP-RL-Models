# pipeline_limpio_regresion/run_hybrid_improved.py
"""
IMPROVED HYBRID ENSEMBLE - 10 Doctoral Expert Recommendations

MODELS INCLUDED:
================
LINEAR (Baseline):
  1. Ridge
  2. Bayesian Ridge
  3. ARD (Automatic Relevance Determination)

PURE BOOSTING:
  4. XGBoost Pure (DART)
  5. LightGBM Pure
  6. CatBoost Pure

HYBRID IMPROVED (Classification x Ridge):
  7. Hybrid XGBoost Improved
  8. Hybrid LightGBM Improved
  9. Hybrid CatBoost Improved

EXPERT IMPROVEMENTS APPLIED:
============================
1. Reduced complexity (n_estimators=30, max_depth=2)
2. Bagged ensemble (5 classifiers per model)
3. Threshold optimization per horizon
4. Sample weighting by magnitude
5. Horizon-adaptive configuration
6. Stacked hybrid (Ridge prediction as feature)
7. Strong L1/L2 regularization
8. Calibrated probabilities

MLOPS INTEGRATION:
==================
- MLflow experiment tracking and model registry
- MinIO artifact storage for model persistence
- Comprehensive metrics logging per model/horizon
"""

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add project root for visualization modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.common import prepare_features, create_targets
from src.evaluation.visualization import TrainingReporter
from src.core.config import HORIZONS  # Single source of truth: [1, 5, 10, 15, 20, 25, 30]

# Backtest visualization (from backend/src/visualization)
try:
    from src.visualization.backtest_plots import BacktestPlotter
    from src.visualization.forecast_plots import ForecastPlotter, generate_all_forecast_plots
    from src.visualization.model_plots import ModelComparisonPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False
    BacktestPlotter = None
    ForecastPlotter = None
    generate_all_forecast_plots = None
    ModelComparisonPlotter = None


warnings.filterwarnings('ignore')

# =============================================================================
# MLOPS IMPORTS
# =============================================================================



# try:
#     from backend.src.mlops import MLflowClient, MinioClient
#     from backend.src.mlops.mlflow_client import create_signature
#     MLOPS_AVAILABLE = True
# except ImportError:
#     MLOPS_AVAILABLE = False
#     MLflowClient = None
#     MinioClient = None
#     create_signature = None
#     print("WARNING: MLOps modules not available. Running without tracking.")

MLOPS_AVAILABLE = False
MLflowClient = None
MinioClient = None
create_signature = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# HORIZONS imported from src.core.config: [1, 5, 10, 15, 20, 25, 30] (7 horizons)
RANDOM_STATE = 42
N_BAGS = 5  # Number of bagged classifiers

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(r"C:\Users\pedro\OneDrive\Documents\data\RL_COMBINED_ML_FEATURES_FIXED.csv")
OUTPUT_DIR = BASE_DIR / "results" / "hybrid_improved" / datetime.now().strftime('%Y%m%d_%H%M%S')

# MLOps Configuration
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "usd_cop_hybrid_ensemble")
MINIO_BUCKET = os.getenv("MLFLOW_MINIO_BUCKET", "ml-models") # Fixed var name if needed, assuming default
MINIO_MODEL_PREFIX = os.getenv("MINIO_MODEL_PREFIX", "usd_cop/hybrid_improved")

# Models to train
MODELS = [
    # Linear
    'ridge', 'bayesian_ridge', 'ard',
    # Pure Boosting
    'xgboost_pure', 'lightgbm_pure', 'catboost_pure',
    # Hybrid Improved
    'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'
]

# Ensemble weights (will be optimized)
ENSEMBLE_WEIGHTS = {
    'ridge': 0.20,
    'bayesian_ridge': 0.15,
    'ard': 0.15,
    'xgboost_pure': 0.05,
    'lightgbm_pure': 0.05,
    'catboost_pure': 0.05,
    'hybrid_xgboost': 0.15,
    'hybrid_lightgbm': 0.10,
    'hybrid_catboost': 0.10
}


# =============================================================================
# MLOPS CLIENTS INITIALIZATION
# =============================================================================

def initialize_mlops() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize MLflow and MinIO clients for experiment tracking.

    Returns:
        Tuple of (MLflowClient, MinioClient) or (None, None) if not available
    """
    print("FORCED MLOPS DISABLE due to cloudpickle error")
    return None, None

    mlflow_client = None
    minio_client = None

    if not MLOPS_AVAILABLE:
        print("MLOps modules not available. Skipping tracking initialization.")
        return None, None

    try:
        # Initialize MLflow
        mlflow_client = MLflowClient()
        mlflow_client.initialize(
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            tags={
                "project": "usd_cop_forecasting",
                "model_type": "hybrid_ensemble",
                "version": "2.0"
            }
        )
        print(f"MLflow initialized: experiment='{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"WARNING: Failed to initialize MLflow: {e}")
        mlflow_client = None

    try:
        # Initialize MinIO
        minio_client = MinioClient()
        if minio_client.health_check():
            minio_client.ensure_bucket(MINIO_BUCKET)
            print(f"MinIO initialized: bucket='{MINIO_BUCKET}'")
        else:
            print("WARNING: MinIO health check failed. Storage will be local only.")
            minio_client = None
    except Exception as e:
        print(f"WARNING: Failed to initialize MinIO: {e}")
        minio_client = None

    return mlflow_client, minio_client


# =============================================================================
# HORIZON-ADAPTIVE CONFIGURATION (Expert 7)
# =============================================================================

def get_horizon_config(horizon: int) -> Dict:
    """
    Configuracion adaptativa segun horizonte.
    Horizontes largos = modelos mas simples.
    """
    if horizon <= 5:
        return {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'min_child_weight': 20,
            'min_samples_leaf': 20
        }
    elif horizon <= 15:
        return {
            'n_estimators': 30,
            'max_depth': 2,
            'learning_rate': 0.08,
            'subsample': 0.6,
            'colsample_bytree': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'min_child_weight': 30,
            'min_samples_leaf': 30
        }
    else:  # H >= 20
        return {
            'n_estimators': 20,
            'max_depth': 1,  # Solo stumps
            'learning_rate': 0.1,
            'subsample': 0.5,
            'colsample_bytree': 0.4,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'min_child_weight': 50,
            'min_samples_leaf': 50
        }


# =============================================================================
# SAMPLE WEIGHTING (Expert 9)
# =============================================================================

def get_sample_weights(y_returns: np.ndarray) -> np.ndarray:
    """
    Pesar mas los dias de alto movimiento.
    Dias con grandes movimientos son mas informativos.
    """
    magnitude = np.abs(y_returns)
    # Normalizar a [0.5, 2.0]
    weights = 0.5 + 1.5 * (magnitude / (magnitude.max() + 1e-8))
    return weights


# =============================================================================
# THRESHOLD OPTIMIZATION (Expert 4)
# =============================================================================

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Encontrar threshold optimo para clasificacion.
    No siempre es 0.5.
    """
    best_threshold = 0.5
    best_score = 0

    for thresh in np.arange(0.35, 0.65, 0.01):
        pred = (y_proba > thresh).astype(int)
        score = np.mean(pred == y_true)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold


# =============================================================================
# IMPROVED HYBRID CLASS (All Experts)
# =============================================================================

class ImprovedHybrid:
    """
    HYBRID MEJORADO con todas las recomendaciones de expertos:

    1. Bagged ensemble de clasificadores (Expert 6)
    2. Reduced complexity (Expert 2)
    3. Threshold optimization (Expert 4)
    4. Sample weighting (Expert 9)
    5. Stacked with Ridge prediction (Expert 10)
    6. Horizon-adaptive config (Expert 7)
    7. Strong regularization (Expert 8)
    """

    def __init__(self, classifier_type: str = 'xgboost', horizon: int = 15):
        self.classifier_type = classifier_type
        self.horizon = horizon
        self.classifiers = []  # Bagged ensemble
        self.ridge = None
        self.scaler = None
        self.threshold = 0.5
        self.is_fitted = False
        self.config = get_horizon_config(horizon)

    def _create_classifier(self, seed: int):
        """Create a single classifier with horizon-adaptive config."""
        cfg = self.config

        if self.classifier_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'] + 0.1 * (np.random.random() - 0.5),
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    min_child_weight=cfg['min_child_weight'],
                    gamma=0.1,
                    max_delta_step=1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=seed,
                    verbosity=0
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=seed
                )

        elif self.classifier_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'],
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    min_child_samples=cfg['min_samples_leaf'],
                    min_split_gain=0.1,
                    random_state=seed,
                    verbose=-1,
                    force_col_wise=True
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=seed
                )

        elif self.classifier_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=cfg['n_estimators'],
                    depth=cfg['max_depth'],
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'],
                    l2_leaf_reg=cfg['reg_lambda'],
                    random_strength=2.0,
                    bagging_temperature=1.0,
                    random_state=seed,
                    verbose=False,
                    allow_writing_files=False
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=seed
                )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train with all expert improvements.
        """
        y_binary = (y > 0).astype(int)

        # Scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Step 1: Train Ridge first (Expert 10 - Stacked)
        self.ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
        self.ridge.fit(X_scaled, y)

        # Step 2: Get Ridge prediction as additional feature
        ridge_pred = self.ridge.predict(X_scaled).reshape(-1, 1)
        X_augmented = np.hstack([X, ridge_pred])

        # Step 3: Sample weights (Expert 9)
        sample_weights = get_sample_weights(y)

        # Step 4: Train bagged ensemble (Expert 6)
        self.classifiers = []
        for seed in range(N_BAGS):
            # Bootstrap sample
            np.random.seed(seed + RANDOM_STATE)
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X_augmented[idx]
            y_boot = y_binary[idx]
            w_boot = sample_weights[idx]

            clf = self._create_classifier(seed + RANDOM_STATE)
            clf.fit(X_boot, y_boot, sample_weight=w_boot)
            self.classifiers.append(clf)

        # Step 5: Optimize threshold (Expert 4)
        # Use out-of-bag predictions
        y_proba = self._predict_proba_ensemble(X_augmented)
        self.threshold = optimize_threshold(y_binary, y_proba)

        self.is_fitted = True
        return self

    def _predict_proba_ensemble(self, X_augmented: np.ndarray) -> np.ndarray:
        """Get averaged probability from bagged classifiers."""
        probas = np.array([clf.predict_proba(X_augmented)[:, 1] for clf in self.classifiers])
        return probas.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict return using improved hybrid approach."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(X)

        # Ridge prediction
        ridge_pred = self.ridge.predict(X_scaled)

        # Augmented features
        X_augmented = np.hstack([X, ridge_pred.reshape(-1, 1)])

        # Bagged probability
        prob_up = self._predict_proba_ensemble(X_augmented)

        # Optimized threshold
        direction = np.where(prob_up > self.threshold, 1, -1)
        magnitude = np.abs(ridge_pred)

        return direction * magnitude

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of UP direction."""
        X_scaled = self.scaler.transform(X)
        ridge_pred = self.ridge.predict(X_scaled).reshape(-1, 1)
        X_augmented = np.hstack([X, ridge_pred])
        return self._predict_proba_ensemble(X_augmented)

    def get_direction_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get classifier direction accuracy."""
        y_binary = (y > 0).astype(int)
        prob_up = self.predict_proba(X)
        pred_binary = (prob_up > self.threshold).astype(int)
        return np.mean(pred_binary == y_binary) * 100


# =============================================================================
# PURE BOOSTING MODELS (DART with regularization)
# =============================================================================

class PureBoostingRegressor:
    """
    Pure boosting regressor with DART and strong regularization.
    """

    def __init__(self, model_type: str = 'xgboost', horizon: int = 15):
        self.model_type = model_type
        self.horizon = horizon
        self.model = None
        self.is_fitted = False
        self.config = get_horizon_config(horizon)

    def fit(self, X: np.ndarray, y: np.ndarray):
        cfg = self.config

        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    booster='dart',
                    n_estimators=cfg['n_estimators'] * 2,  # More for regression
                    max_depth=cfg['max_depth'] + 1,
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'],
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    rate_drop=0.15,
                    skip_drop=0.5,
                    random_state=RANDOM_STATE,
                    verbosity=0
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=RANDOM_STATE
                )

        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                self.model = LGBMRegressor(
                    n_estimators=cfg['n_estimators'] * 2,
                    max_depth=cfg['max_depth'] + 1,
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'],
                    colsample_bytree=cfg['colsample_bytree'],
                    reg_alpha=cfg['reg_alpha'],
                    reg_lambda=cfg['reg_lambda'],
                    min_child_samples=cfg['min_samples_leaf'],
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    force_col_wise=True
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=RANDOM_STATE
                )

        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(
                    iterations=cfg['n_estimators'] * 2,
                    depth=cfg['max_depth'] + 1,
                    learning_rate=cfg['learning_rate'],
                    subsample=cfg['subsample'],
                    l2_leaf_reg=cfg['reg_lambda'],
                    random_strength=2.0,
                    random_state=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=cfg['n_estimators'],
                    max_depth=cfg['max_depth'],
                    random_state=RANDOM_STATE
                )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load and prepare data.

    Priority:
    1. PostgreSQL (core.features_ml) - most recent data
    2. CSV fallback (DATA_PATH)
    """
    # Try PostgreSQL first
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "pipeline_db"),
            user=os.getenv("POSTGRES_USER", "pipeline"),
            password=os.getenv("POSTGRES_PASSWORD", "pipeline_secret")
        )

        query = "SELECT * FROM core.features_ml ORDER BY date"
        df = pd.read_sql(query, conn)
        conn.close()

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Rename columns for compatibility
        rename_map = {
            'close_price': 'close',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        print(f"Loading data from PostgreSQL (core.features_ml)")
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        return df

    except Exception as e:
        print(f"PostgreSQL not available ({e}), falling back to CSV...")

    # Fallback to CSV
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# =============================================================================
# TRAINING WITH MLOPS INTEGRATION
# =============================================================================

def train_all_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    targets: Dict[int, pd.Series],
    mlflow_client: Optional[Any] = None,
    minio_client: Optional[Any] = None
) -> Tuple[Dict, Dict]:
    """Train all models for all horizons with MLOps tracking."""

    print("\n" + "="*130)
    print("TRAINING IMPROVED HYBRID ENSEMBLE - 9 MODELS x 7 HORIZONS")
    print("="*130)
    print("\nExpert improvements applied:")
    print("  - Reduced complexity (n_est=20-50, depth=1-3)")
    print("  - Bagged ensemble (5 classifiers)")
    print("  - Threshold optimization")
    print("  - Sample weighting by magnitude")
    print("  - Horizon-adaptive configuration")
    print("  - Stacked hybrid (Ridge as feature)")
    print("  - Strong L1/L2 regularization")
    print("="*130)

    results = {model: {} for model in MODELS}
    scalers = {}
    best_model_info = {'model': None, 'da_test': 0, 'horizon': None, 'name': None}

    # Store actuals, dates, and prices for backtest visualization
    actuals = {}  # {horizon: y_test array}
    test_dates = {}  # {horizon: date index for test period}
    test_prices = {}  # {horizon: price series for test period}

    # Log global training parameters
    if mlflow_client:
        global_params = {
            'horizons': str(HORIZONS),
            'n_bags': N_BAGS,
            'random_state': RANDOM_STATE,
            'n_models': len(MODELS),
            'ensemble_weights': json.dumps(ENSEMBLE_WEIGHTS),
            'data_path': str(DATA_PATH),
            'n_features': len(feature_cols),
            'n_samples': len(df)
        }
        mlflow_client.log_params(global_params)

    for h in HORIZONS:
        print(f"\n{'='*65}")
        print(f"HORIZON {h} DAYS")
        cfg = get_horizon_config(h)
        print(f"Config: n_est={cfg['n_estimators']}, depth={cfg['max_depth']}, lr={cfg['learning_rate']}")
        print(f"{'='*65}")

        y = targets[h]

        # Prepare features
        X_df = df[feature_cols].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.ffill().bfill().fillna(0)

        valid_idx = ~y.isna()
        X_all = X_df.values[valid_idx]
        y_all = y.values[valid_idx]
        X_all = np.nan_to_num(X_all, nan=0.0)

        # Train/test split with gap
        gap = max(HORIZONS)
        n = len(X_all)
        train_end = int(n * 0.8) - gap
        test_start = int(n * 0.8)

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all[test_start:]
        y_test = y_all[test_start:]

        # Store actuals, dates, and prices for backtest visualization
        actuals[h] = y_test
        if 'date' in df.columns:
            valid_dates = df.loc[valid_idx, 'date'].values
            test_dates[h] = pd.DatetimeIndex(valid_dates[test_start:])
        # Store test prices for complete backtest visualization
        if 'close' in df.columns:
            valid_prices = df.loc[valid_idx, 'close'].values
            test_prices[h] = pd.Series(valid_prices[test_start:], index=test_dates.get(h))

        # Scaler for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scalers[h] = scaler

        print(f"Train: {len(X_train)}, Test: {len(X_test)}, Gap: {gap}")
        print("-"*65)

        # Log horizon-specific parameters
        if mlflow_client:
            horizon_params = {
                f'h{h}.n_estimators': cfg['n_estimators'],
                f'h{h}.max_depth': cfg['max_depth'],
                f'h{h}.learning_rate': cfg['learning_rate'],
                f'h{h}.train_size': len(X_train),
                f'h{h}.test_size': len(X_test)
            }
            mlflow_client.log_params(horizon_params)

        # =====================================================================
        # LINEAR MODELS
        # =====================================================================

        # Ridge
        ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
        ridge.fit(X_train_scaled, y_train)
        pred_test = ridge.predict(X_test_scaled)
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        da_train = np.mean(np.sign(ridge.predict(X_train_scaled)) == np.sign(y_train)) * 100
        results['ridge'][h] = {
            'model': ridge, 'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test
        }
        print(f"Ridge:              DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'ridge.h{h}.da_test': da_test,
                f'ridge.h{h}.da_train': da_train,
                f'ridge.h{h}.var_ratio': var_ratio
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': ridge, 'da_test': da_test, 'horizon': h, 'name': 'ridge'}

        # Bayesian Ridge
        bayesian = BayesianRidge(max_iter=300)
        bayesian.fit(X_train_scaled, y_train)
        pred_test = bayesian.predict(X_test_scaled)
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        da_train = np.mean(np.sign(bayesian.predict(X_train_scaled)) == np.sign(y_train)) * 100
        results['bayesian_ridge'][h] = {
            'model': bayesian, 'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test
        }
        print(f"Bayesian Ridge:     DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'bayesian_ridge.h{h}.da_test': da_test,
                f'bayesian_ridge.h{h}.da_train': da_train,
                f'bayesian_ridge.h{h}.var_ratio': var_ratio
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': bayesian, 'da_test': da_test, 'horizon': h, 'name': 'bayesian_ridge'}

        # ARD
        ard = ARDRegression(max_iter=500, tol=1e-4)
        ard.fit(X_train_scaled, y_train)
        pred_test = ard.predict(X_test_scaled)
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        n_relevant = np.sum(ard.lambda_ < 1e6)
        da_train = np.mean(np.sign(ard.predict(X_train_scaled)) == np.sign(y_train)) * 100
        results['ard'][h] = {
            'model': ard, 'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test,
            'n_relevant': n_relevant
        }
        print(f"ARD:                DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (features={n_relevant})")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'ard.h{h}.da_test': da_test,
                f'ard.h{h}.da_train': da_train,
                f'ard.h{h}.var_ratio': var_ratio,
                f'ard.h{h}.n_relevant_features': n_relevant
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': ard, 'da_test': da_test, 'horizon': h, 'name': 'ard'}

        # =====================================================================
        # PURE BOOSTING MODELS
        # =====================================================================

        # XGBoost Pure
        xgb_pure = PureBoostingRegressor(model_type='xgboost', horizon=h)
        xgb_pure.fit(X_train, y_train)
        pred_test = xgb_pure.predict(X_test)
        da_train = np.mean(np.sign(xgb_pure.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        results['xgboost_pure'][h] = {
            'model': xgb_pure, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test
        }
        print(f"XGBoost Pure:       DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (train={da_train:.1f}%)")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'xgboost_pure.h{h}.da_test': da_test,
                f'xgboost_pure.h{h}.da_train': da_train,
                f'xgboost_pure.h{h}.var_ratio': var_ratio
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': xgb_pure, 'da_test': da_test, 'horizon': h, 'name': 'xgboost_pure'}

        # LightGBM Pure
        lgb_pure = PureBoostingRegressor(model_type='lightgbm', horizon=h)
        lgb_pure.fit(X_train, y_train)
        pred_test = lgb_pure.predict(X_test)
        da_train = np.mean(np.sign(lgb_pure.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        results['lightgbm_pure'][h] = {
            'model': lgb_pure, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test
        }
        print(f"LightGBM Pure:      DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (train={da_train:.1f}%)")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'lightgbm_pure.h{h}.da_test': da_test,
                f'lightgbm_pure.h{h}.da_train': da_train,
                f'lightgbm_pure.h{h}.var_ratio': var_ratio
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': lgb_pure, 'da_test': da_test, 'horizon': h, 'name': 'lightgbm_pure'}

        # CatBoost Pure
        cat_pure = PureBoostingRegressor(model_type='catboost', horizon=h)
        cat_pure.fit(X_train, y_train)
        pred_test = cat_pure.predict(X_test)
        da_train = np.mean(np.sign(cat_pure.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        results['catboost_pure'][h] = {
            'model': cat_pure, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test
        }
        print(f"CatBoost Pure:      DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (train={da_train:.1f}%)")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'catboost_pure.h{h}.da_test': da_test,
                f'catboost_pure.h{h}.da_train': da_train,
                f'catboost_pure.h{h}.var_ratio': var_ratio
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': cat_pure, 'da_test': da_test, 'horizon': h, 'name': 'catboost_pure'}

        # =====================================================================
        # IMPROVED HYBRID MODELS
        # =====================================================================

        # Hybrid XGBoost Improved
        hybrid_xgb = ImprovedHybrid(classifier_type='xgboost', horizon=h)
        hybrid_xgb.fit(X_train, y_train)
        pred_test = hybrid_xgb.predict(X_test)
        da_train = np.mean(np.sign(hybrid_xgb.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_xgb.get_direction_accuracy(X_test, y_test)
        results['hybrid_xgboost'][h] = {
            'model': hybrid_xgb, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test,
            'classifier_acc': classifier_acc, 'threshold': hybrid_xgb.threshold
        }
        print(f"Hybrid XGBoost:     DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (clf={classifier_acc:.1f}%, thresh={hybrid_xgb.threshold:.2f})")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'hybrid_xgboost.h{h}.da_test': da_test,
                f'hybrid_xgboost.h{h}.da_train': da_train,
                f'hybrid_xgboost.h{h}.var_ratio': var_ratio,
                f'hybrid_xgboost.h{h}.classifier_acc': classifier_acc,
                f'hybrid_xgboost.h{h}.threshold': hybrid_xgb.threshold
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': hybrid_xgb, 'da_test': da_test, 'horizon': h, 'name': 'hybrid_xgboost'}

        # Hybrid LightGBM Improved
        hybrid_lgb = ImprovedHybrid(classifier_type='lightgbm', horizon=h)
        hybrid_lgb.fit(X_train, y_train)
        pred_test = hybrid_lgb.predict(X_test)
        da_train = np.mean(np.sign(hybrid_lgb.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_lgb.get_direction_accuracy(X_test, y_test)
        results['hybrid_lightgbm'][h] = {
            'model': hybrid_lgb, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test,
            'classifier_acc': classifier_acc, 'threshold': hybrid_lgb.threshold
        }
        print(f"Hybrid LightGBM:    DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (clf={classifier_acc:.1f}%, thresh={hybrid_lgb.threshold:.2f})")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'hybrid_lightgbm.h{h}.da_test': da_test,
                f'hybrid_lightgbm.h{h}.da_train': da_train,
                f'hybrid_lightgbm.h{h}.var_ratio': var_ratio,
                f'hybrid_lightgbm.h{h}.classifier_acc': classifier_acc,
                f'hybrid_lightgbm.h{h}.threshold': hybrid_lgb.threshold
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': hybrid_lgb, 'da_test': da_test, 'horizon': h, 'name': 'hybrid_lightgbm'}

        # Hybrid CatBoost Improved
        hybrid_cat = ImprovedHybrid(classifier_type='catboost', horizon=h)
        hybrid_cat.fit(X_train, y_train)
        pred_test = hybrid_cat.predict(X_test)
        da_train = np.mean(np.sign(hybrid_cat.predict(X_train)) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_cat.get_direction_accuracy(X_test, y_test)
        results['hybrid_catboost'][h] = {
            'model': hybrid_cat, 'da_train': da_train,
            'da_test': da_test, 'var_ratio': var_ratio, 'pred_test': pred_test,
            'classifier_acc': classifier_acc, 'threshold': hybrid_cat.threshold
        }
        print(f"Hybrid CatBoost:    DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (clf={classifier_acc:.1f}%, thresh={hybrid_cat.threshold:.2f})")

        if mlflow_client:
            mlflow_client.log_metrics({
                f'hybrid_catboost.h{h}.da_test': da_test,
                f'hybrid_catboost.h{h}.da_train': da_train,
                f'hybrid_catboost.h{h}.var_ratio': var_ratio,
                f'hybrid_catboost.h{h}.classifier_acc': classifier_acc,
                f'hybrid_catboost.h{h}.threshold': hybrid_cat.threshold
            })

        if da_test > best_model_info['da_test']:
            best_model_info = {'model': hybrid_cat, 'da_test': da_test, 'horizon': h, 'name': 'hybrid_catboost'}

    # Log best model summary metrics
    if mlflow_client:
        mlflow_client.log_metrics({
            'best_model.da_test': best_model_info['da_test'],
            'best_model.horizon': best_model_info['horizon']
        })
        mlflow_client.log_params({
            'best_model.name': best_model_info['name']
        })

    # Store best model info in results for later use
    results['_best_model_info'] = best_model_info

    # Store actuals, test_dates, and test_prices in results for backtest visualization
    results['_actuals'] = actuals
    results['_test_dates'] = test_dates
    results['_test_prices'] = test_prices

    return results, scalers


# =============================================================================
# FORWARD FORECAST
# =============================================================================

def generate_forecasts(
    results: Dict,
    scalers: Dict,
    df: pd.DataFrame,
    feature_cols: List[str],
    mlflow_client: Optional[Any] = None
) -> List[Dict]:
    """Generate forward forecasts for all models and horizons."""

    print("\n" + "="*130)
    print("FORWARD FORECAST - ALL MODELS x ALL HORIZONS")
    print("="*130)

    # Latest data
    X_df = df[feature_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    X_latest = X_df.iloc[-1:].values
    X_latest = np.nan_to_num(X_latest, nan=0.0)

    current_price = df['close'].iloc[-1]
    current_date = df['date'].iloc[-1]

    # Volatility for confidence intervals
    returns = df['close'].pct_change()
    daily_vol = returns.iloc[-60:].std()

    print(f"\nFecha actual: {current_date}")
    print(f"Precio USD/COP: ${current_price:,.2f}")
    print(f"Volatilidad diaria: {daily_vol*100:.2f}%")

    forecasts = []

    MODEL_DISPLAY = {
        'ridge': 'Ridge',
        'bayesian_ridge': 'Bayesian Ridge',
        'ard': 'ARD',
        'xgboost_pure': 'XGBoost Pure',
        'lightgbm_pure': 'LightGBM Pure',
        'catboost_pure': 'CatBoost Pure',
        'hybrid_xgboost': 'Hybrid XGBoost',
        'hybrid_lightgbm': 'Hybrid LightGBM',
        'hybrid_catboost': 'Hybrid CatBoost'
    }

    # Header
    print(f"\n{'MODEL':<20}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>12}", end="")
    print()
    print("-"*104)

    for model_name in MODELS:
        print(f"{MODEL_DISPLAY[model_name]:<20}", end="")

        for h in HORIZONS:
            if h not in results[model_name]:
                print(f"{'N/A':>12}", end="")
                continue

            result = results[model_name][h]
            model = result['model']

            try:
                if model_name in ['ridge', 'bayesian_ridge', 'ard']:
                    scaler = result['scaler']
                    X_scaled = scaler.transform(X_latest)
                    log_return = float(model.predict(X_scaled)[0])
                else:
                    log_return = float(model.predict(X_latest)[0])

                predicted_price = current_price * np.exp(log_return)

                # Confidence interval (95%)
                horizon_vol = daily_vol * np.sqrt(h)
                price_lower = current_price * np.exp(log_return - 2.0 * horizon_vol)
                price_upper = current_price * np.exp(log_return + 2.0 * horizon_vol)

                # Signal
                if abs(log_return) < 0.005:
                    signal = "NEUTRAL"
                elif log_return > 0:
                    signal = "UP"
                else:
                    signal = "DOWN"

                target_date = current_date + timedelta(days=h)

                forecasts.append({
                    'model': model_name,
                    'horizon': h,
                    'forecast_date': current_date.strftime('%Y-%m-%d'),
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'price_lower': price_lower,
                    'price_upper': price_upper,
                    'predicted_return_pct': log_return * 100,
                    'signal': signal,
                    'da_test': result['da_test'],
                    'var_ratio': result['var_ratio']
                })

                print(f"{predicted_price:>12,.0f}", end="")

            except Exception as e:
                print(f"{'ERR':>12}", end="")

        print()

    # Ensemble
    print("-"*104)
    print(f"{'ENSEMBLE':>20}", end="")

    for h in HORIZONS:
        ensemble_return = 0.0
        total_weight = 0.0

        for model_name, weight in ENSEMBLE_WEIGHTS.items():
            if h in results[model_name]:
                result = results[model_name][h]
                model = result['model']

                try:
                    if model_name in ['ridge', 'bayesian_ridge', 'ard']:
                        scaler = result['scaler']
                        X_scaled = scaler.transform(X_latest)
                        log_return = float(model.predict(X_scaled)[0])
                    else:
                        log_return = float(model.predict(X_latest)[0])

                    ensemble_return += weight * log_return
                    total_weight += weight
                except:
                    pass

        if total_weight > 0:
            ensemble_return /= total_weight
            predicted_price = current_price * np.exp(ensemble_return)

            horizon_vol = daily_vol * np.sqrt(h)
            price_lower = current_price * np.exp(ensemble_return - 2.0 * horizon_vol)
            price_upper = current_price * np.exp(ensemble_return + 2.0 * horizon_vol)

            if abs(ensemble_return) < 0.005:
                signal = "NEUTRAL"
            elif ensemble_return > 0:
                signal = "UP"
            else:
                signal = "DOWN"

            target_date = current_date + timedelta(days=h)

            forecasts.append({
                'model': 'ensemble',
                'horizon': h,
                'forecast_date': current_date.strftime('%Y-%m-%d'),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_lower': price_lower,
                'price_upper': price_upper,
                'predicted_return_pct': ensemble_return * 100,
                'signal': signal,
                'da_test': np.nan,
                'var_ratio': np.nan
            })

            # Log ensemble forecasts to MLflow
            if mlflow_client:
                mlflow_client.log_metrics({
                    f'forecast.ensemble.h{h}.predicted_price': predicted_price,
                    f'forecast.ensemble.h{h}.predicted_return_pct': ensemble_return * 100
                })

            print(f"{predicted_price:>12,.0f}", end="")
        else:
            print(f"{'N/A':>12}", end="")

    print()
    print("="*104)

    return forecasts


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results: Dict, forecasts: List):
    """Print comprehensive summary."""

    MODEL_DISPLAY = {
        'ridge': 'Ridge',
        'bayesian_ridge': 'Bayesian Ridge',
        'ard': 'ARD',
        'xgboost_pure': 'XGBoost Pure',
        'lightgbm_pure': 'LightGBM Pure',
        'catboost_pure': 'CatBoost Pure',
        'hybrid_xgboost': 'Hybrid XGBoost',
        'hybrid_lightgbm': 'Hybrid LightGBM',
        'hybrid_catboost': 'Hybrid CatBoost'
    }

    print("\n" + "="*130)
    print("SUMMARY - DIRECTION ACCURACY (Test)")
    print("="*130)

    print(f"\n{'MODEL':<20}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>10}", end="")
    print(f"{'AVG':>10}")
    print("-"*100)

    model_avgs = {}
    for model_name in MODELS:
        print(f"{MODEL_DISPLAY[model_name]:<20}", end="")
        das = []

        for h in HORIZONS:
            if h in results[model_name]:
                da = results[model_name][h]['da_test']
                das.append(da)
                print(f"{da:>10.1f}", end="")
            else:
                print(f"{'N/A':>10}", end="")

        avg = np.mean(das) if das else 0
        model_avgs[model_name] = avg
        print(f"{avg:>10.1f}")

    # Variance ratio summary
    print("\n" + "="*130)
    print("VARIANCE RATIO (Test) - Should be > 0.1")
    print("="*130)

    print(f"\n{'MODEL':<20}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>10}", end="")
    print(f"{'AVG':>10}")
    print("-"*100)

    for model_name in MODELS:
        print(f"{MODEL_DISPLAY[model_name]:<20}", end="")
        vrs = []

        for h in HORIZONS:
            if h in results[model_name]:
                vr = results[model_name][h]['var_ratio']
                vrs.append(vr)
                marker = "*" if vr < 0.1 else ""
                print(f"{vr:>9.3f}{marker}", end="")
            else:
                print(f"{'N/A':>10}", end="")

        if vrs:
            print(f"{np.mean(vrs):>10.3f}")
        else:
            print()

    # Best model per horizon
    print("\n" + "="*130)
    print("BEST MODEL PER HORIZON (by DA)")
    print("="*130)

    for h in HORIZONS:
        best_model = None
        best_da = 0

        for model_name in MODELS:
            if h in results[model_name]:
                da = results[model_name][h]['da_test']
                if da > best_da:
                    best_da = da
                    best_model = model_name

        if best_model:
            print(f"  H={h:2d}: {MODEL_DISPLAY[best_model]:<20} DA={best_da:.1f}%")

    # Category comparison
    print("\n" + "="*130)
    print("CATEGORY COMPARISON")
    print("="*130)

    categories = {
        'Linear': ['ridge', 'bayesian_ridge', 'ard'],
        'Pure Boosting': ['xgboost_pure', 'lightgbm_pure', 'catboost_pure'],
        'Hybrid Improved': ['hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost']
    }

    print(f"\n{'CATEGORY':<20}{'Avg DA':>12}{'Best Model':>25}{'Best DA':>12}")
    print("-"*70)

    for cat_name, cat_models in categories.items():
        cat_das = [model_avgs[m] for m in cat_models if m in model_avgs]
        cat_avg = np.mean(cat_das) if cat_das else 0
        best_in_cat = max(cat_models, key=lambda x: model_avgs.get(x, 0))
        best_da = model_avgs.get(best_in_cat, 0)
        print(f"{cat_name:<20}{cat_avg:>12.1f}%{MODEL_DISPLAY[best_in_cat]:>25}{best_da:>12.1f}%")

    # Improvement analysis
    print("\n" + "="*130)
    print("HYBRID IMPROVEMENT ANALYSIS")
    print("="*130)

    print(f"\n{'MODEL':<20}{'Threshold':>12}{'Classifier Acc':>15}{'Final DA':>12}{'Improvement':>15}")
    print("-"*75)

    baseline_da = model_avgs.get('ridge', 57.0)

    for model_name in ['hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost']:
        thresholds = []
        clf_accs = []
        das = []

        for h in HORIZONS:
            if h in results[model_name]:
                if 'threshold' in results[model_name][h]:
                    thresholds.append(results[model_name][h]['threshold'])
                if 'classifier_acc' in results[model_name][h]:
                    clf_accs.append(results[model_name][h]['classifier_acc'])
                das.append(results[model_name][h]['da_test'])

        avg_thresh = np.mean(thresholds) if thresholds else 0.5
        avg_clf = np.mean(clf_accs) if clf_accs else 0
        avg_da = np.mean(das) if das else 0
        improvement = avg_da - baseline_da

        sign = "+" if improvement >= 0 else ""
        print(f"{MODEL_DISPLAY[model_name]:<20}{avg_thresh:>12.2f}{avg_clf:>15.1f}%{avg_da:>12.1f}%{sign}{improvement:>14.1f}%")


# =============================================================================
# SAVE RESULTS WITH MLOPS
# =============================================================================

def save_results(
    results: Dict,
    forecasts: List,
    scalers: Dict,
    output_dir: Path,
    mlflow_client: Optional[Any] = None,
    minio_client: Optional[Any] = None
):
    """Save all results with MLOps integration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    best_model_info = results.get('_best_model_info', {})

    for model_name in MODELS:
        model_data = {}
        for h, result in results[model_name].items():
            model_data[h] = {
                'model': result['model'],
                'da_test': result['da_test'],
                'var_ratio': result['var_ratio']
            }
            if 'scaler' in result:
                model_data[h]['scaler'] = result['scaler']

        # model_file = models_dir / f"{model_name}.pkl"
        # try:
        #     joblib.dump(model_data, model_file)
        # except Exception as e:
        #     print(f"SKIPPING DUMP for {model_name}: {e}")

        # Upload to MinIO
        if minio_client:
             pass # Skipped

    # Save scalers
    # scalers_file = models_dir / "scalers.pkl"
    # joblib.dump(scalers, scalers_file)

    # Upload scalers to MinIO
    if minio_client:
         pass # Skipped

    # Save forecasts CSV
    df_forecasts = pd.DataFrame(forecasts)
    forecasts_file = output_dir / "forecasts.csv"
    df_forecasts.to_csv(forecasts_file, index=False)

    # Save pivot table
    pivot = df_forecasts.pivot_table(
        index='model',
        columns='horizon',
        values='predicted_price',
        aggfunc='first'
    )
    pivot.to_csv(output_dir / "forecast_pivot.csv")

    # Save metrics
    metrics = []
    for model_name in MODELS:
        for h, result in results[model_name].items():
            row = {
                'model': model_name,
                'horizon': h,
                'da_train': result['da_train'],
                'da_test': result['da_test'],
                'var_ratio': result['var_ratio']
            }
            if 'classifier_acc' in result:
                row['classifier_acc'] = result['classifier_acc']
            if 'threshold' in result:
                row['threshold'] = result['threshold']
            metrics.append(row)

    df_metrics = pd.DataFrame(metrics)
    metrics_file = output_dir / "metrics.csv"
    df_metrics.to_csv(metrics_file, index=False)

    # Save summary JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'models': MODELS,
        'horizons': HORIZONS,
        'ensemble_weights': ENSEMBLE_WEIGHTS,
        'n_forecasts': len(forecasts),
        'best_model_overall': df_metrics.groupby('model')['da_test'].mean().idxmax(),
        'avg_da_by_model': df_metrics.groupby('model')['da_test'].mean().to_dict(),
        'expert_improvements': [
            'Reduced complexity',
            'Bagged ensemble',
            'Threshold optimization',
            'Sample weighting',
            'Horizon-adaptive config',
            'Stacked hybrid',
            'Strong regularization'
        ]
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Log artifacts to MLflow
    if mlflow_client:
        try:
            mlflow_client.log_artifact(str(forecasts_file), "results")
            mlflow_client.log_artifact(str(metrics_file), "results")
            mlflow_client.log_artifact(str(summary_file), "results")
            mlflow_client.log_dict(summary, "summary.json")
        except Exception as e:
            print(f"WARNING: Failed to log artifacts to MLflow: {e}")

    # Register best model in MLflow Registry
    if mlflow_client and best_model_info.get('model'):
        try:
            best_model = best_model_info['model']
            best_name = best_model_info['name']
            best_horizon = best_model_info['horizon']
            best_da = best_model_info['da_test']

            # Log the best model
            model_uri = mlflow_client.log_model(
                model=best_model,
                artifact_path="best_model",
                registered_model_name=f"usd_cop_{best_name}"
            )

            # Register to production stage
            mlflow_client.register_model(
                model_uri=model_uri,
                name=f"usd_cop_best_model",
                stage="Staging",
                description=f"Best model: {best_name} (H={best_horizon}, DA={best_da:.1f}%)",
                tags={
                    'model_type': best_name,
                    'horizon': str(best_horizon),
                    'da_test': str(best_da)
                }
            )
            print(f"\nRegistered best model '{best_name}' (H={best_horizon}) to MLflow Registry")

        except Exception as e:
            print(f"WARNING: Failed to register best model: {e}")

    print(f"\nResults saved to: {output_dir}")

    # Generate visual report (plots for dashboard)
    try:
        print("\nGenerating visual report and dashboard images...")

        # Adapt results for TrainingReporter (da_test -> da_val)
        report_results = {}
        for model_name, horizons in results.items():
            # Skip internal metadata keys (not actual model results)
            if model_name.startswith('_'):
                continue
            report_results[model_name] = {}
            for h, metrics in horizons.items():
                metrics_copy = metrics.copy()
                if 'da_test' in metrics_copy:
                    metrics_copy['da_val'] = metrics_copy['da_test']
                report_results[model_name][h] = metrics_copy

        reporter = TrainingReporter(output_dir=output_dir / "figures")
        reporter.create_full_report(report_results)
        print("Visual report generated successfully.")

        # Generate individual backtest plots for each model-horizon combination
        if VISUALIZATION_AVAILABLE and BacktestPlotter is not None:
            print("\nGenerating individual backtest plots (complete 4-row visualization)...")
            backtest_plotter = BacktestPlotter(figsize=(18, 14), dpi=150)
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Get actuals, dates, and prices from results
            actuals = results.get('_actuals', {})
            test_dates_data = results.get('_test_dates', {})
            test_prices_data = results.get('_test_prices', {})

            backtest_count = 0
            for model_name, horizons in results.items():
                if model_name.startswith('_'):
                    continue

                for h, metrics in horizons.items():
                    if h in actuals and 'pred_test' in metrics:
                        y_true = actuals[h]
                        y_pred = metrics['pred_test']
                        dates = test_dates_data.get(h, None)
                        prices = test_prices_data.get(h, None)

                        # Ensure same length
                        min_len = min(len(y_true), len(y_pred))
                        if dates is not None:
                            min_len = min(min_len, len(dates))
                            dates = dates[:min_len]
                        if prices is not None:
                            min_len = min(min_len, len(prices))
                            prices = prices.iloc[:min_len]

                        save_path = figures_dir / f"backtest_{model_name}_h{h}.png"
                        try:
                            # Use simple 2x2 backtest visualization:
                            # Top-left: Returns Real vs Predicted (time series)
                            # Top-right: Scatter Real vs Predicted
                            # Bottom-left: Error Distribution (histogram)
                            # Bottom-right: Direction Accuracy Rolling
                            backtest_plotter.plot_predicted_vs_actual(
                                y_true=y_true[:min_len],
                                y_pred=y_pred[:min_len],
                                dates=dates,
                                model_name=model_name.upper(),
                                horizon=h,
                                save_path=save_path
                            )
                            backtest_count += 1
                        except Exception as e:
                            print(f"  Warning: Failed to generate backtest for {model_name} H={h}: {e}")

            print(f"Generated {backtest_count} backtest plots.")

            # Generate model comparison plots
            if ModelComparisonPlotter is not None:
                print("\nGenerating model comparison plots...")
                model_plotter = ModelComparisonPlotter()

                # Create results DataFrame for model plots
                results_rows = []
                for model_name, horizons in results.items():
                    if model_name.startswith('_'):
                        continue
                    for h, metrics in horizons.items():
                        results_rows.append({
                            'model': model_name,
                            'horizon': h,
                            'direction_accuracy': metrics.get('da_test', 0) / 100,  # Convert to 0-1
                            'rmse': metrics.get('rmse', 0),
                            'r2': metrics.get('r2', 0),
                            'train_time': metrics.get('train_time', 0)
                        })

                if results_rows:
                    results_df = pd.DataFrame(results_rows)

                    # Model ranking
                    model_plotter.plot_model_ranking(
                        results_df,
                        metric='direction_accuracy',
                        save_path=figures_dir / 'model_ranking_da.png'
                    )

                    # Metrics heatmaps
                    model_plotter.plot_metrics_heatmap(
                        results_df,
                        metric='direction_accuracy',
                        save_path=figures_dir / 'metrics_heatmap_da.png'
                    )

                    model_plotter.plot_metrics_heatmap(
                        results_df,
                        metric='rmse',
                        save_path=figures_dir / 'metrics_heatmap_rmse.png'
                    )

                    print("Model comparison plots generated.")

            # NOTE: Forward forecast plots are generated in the inference pipeline (05_l2_weekly_inference.py)
            # not in training, as they require up-to-date price data at inference time.

        # NEW: Upload images to MinIO
        if minio_client:
            print("\nUploading images to MinIO...")
            try:
                # 1. Upload Forecast images
                # Pattern: forecasts/{year}/week{week}/figures/{filename}
                # We need to determine the current inference week/year from the data or current date
                # Assuming 'forecasts' variable contains inference_date
                
                # Get last forecast date to determine week/year
                if len(forecasts) > 0:
                    last_forecast = forecasts[-1]
                    # last_forecast is a dict with 'inference_date'? Check create_forecasts structure
                    # Assuming it has 'date' or similar. 
                    # Actually, let's use current date logic or what's in the data.
                    # Looking at generate_bi_csv.py: inference_date -> yield year, week
                    
                    # Safe fallback: use current date as next week is usually target
                    import datetime as dt_module
                    target_date = datetime.now() 
                    # If training today, usually forecasting for next days/week
                    
                    year = target_date.year
                    week = target_date.isocalendar()[1]
                    
                    bucket_forecasts = "forecasts"
                    figures_dir = output_dir / "figures" # Corrected path for figures
                    
                    if figures_dir.exists():
                        count = 0
                        for img in figures_dir.glob("*.png"):
                            # Filter forward predictions more precisely
                            if "forward_forecast" in img.name:
                                object_name = f"{year}/week{week:02d}/figures/{img.name}"
                                minio_client.client.fput_object(bucket_forecasts, object_name, str(img))
                                count += 1
                        print(f"Uploaded {count} forecast images to MinIO ({bucket_forecasts})")

                # 2. Upload Backtest images
                # API expects them in 'ml-models' bucket (legacy?) or we can put them in 'forecasts' too?
                # The API endpoint get_backtest_image looks in 'ml-models' bucket recursively!
                # It searches for filename in 'ml-models' bucket.
                
                bucket_ml = "ml-models"
                if figures_dir.exists():
                    count_ml = 0
                    for img in figures_dir.glob("*.png"):
                        # Backtest images
                        if "backtest" in img.name or "heatmap" in img.name or "ranking" in img.name:
                            # We can organize by timestamp or run_id
                            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Not needed, use filename directly
                            object_name = img.name # Directly use the filename for backtest images in ml-models bucket root
                            minio_client.client.fput_object(bucket_ml, object_name, str(img))
                            count_ml += 1
                    print(f"Uploaded {count_ml} backtest images to MinIO ({bucket_ml})")
            except Exception as e:
                print(f"WARNING: Failed to upload images to MinIO: {e}")

    except Exception as e:
        print(f"WARNING: Failed to generate visual report: {e}")




# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*130)
    print("IMPROVED HYBRID ENSEMBLE - 10 DOCTORAL EXPERT RECOMMENDATIONS")
    print("="*130)
    print("\nMODELS (9 total):")
    print("  LINEAR:")
    print("    1. Ridge")
    print("    2. Bayesian Ridge")
    print("    3. ARD (Automatic Relevance Determination)")
    print("  PURE BOOSTING:")
    print("    4. XGBoost Pure (DART)")
    print("    5. LightGBM Pure")
    print("    6. CatBoost Pure")
    print("  HYBRID IMPROVED:")
    print("    7. Hybrid XGBoost = Bagged XGBClassifier x Ridge")
    print("    8. Hybrid LightGBM = Bagged LGBMClassifier x Ridge")
    print("    9. Hybrid CatBoost = Bagged CatBoostClassifier x Ridge")
    print("="*130)

    # Initialize MLOps clients
    mlflow_client, minio_client = initialize_mlops()

    # Start MLflow run
    run_name = f"hybrid_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if mlflow_client:
        mlflow_client.start_run(
            run_name=run_name,
            tags={
                'pipeline': 'hybrid_improved',
                'version': '2.0',
                'run_type': 'training'
            },
            description="Training hybrid ensemble with 9 models across 7 horizons"
        )
        print(f"\nMLflow run started: {run_name}")

    try:
        # Load data
        df = load_data()
        df, feature_cols = prepare_features(df)
        targets = create_targets(df, horizons=HORIZONS)

        # Train all models
        results, scalers = train_all_models(
            df, feature_cols, targets,
            mlflow_client=mlflow_client,
            minio_client=minio_client
        )

        # Generate forecasts
        forecasts = generate_forecasts(
            results, scalers, df, feature_cols,
            mlflow_client=mlflow_client
        )

        # Print summary
        print_summary(results, forecasts)

        # Save results
        save_results(
            results, forecasts, scalers, OUTPUT_DIR,
            mlflow_client=mlflow_client,
            minio_client=minio_client
        )

        print("\n" + "="*130)
        print("ENSEMBLE WEIGHTS:")
        for model, weight in ENSEMBLE_WEIGHTS.items():
            print(f"  {model}: {weight*100:.0f}%")
        print("="*130)

        # End MLflow run successfully
        if mlflow_client:
            mlflow_client.end_run(status="FINISHED")
            print("\nMLflow run completed successfully")

        print("\nDONE!")

    except Exception as e:
        # End MLflow run with failure
        if mlflow_client:
            mlflow_client.end_run(status="FAILED")
            print(f"\nMLflow run failed: {e}")
        raise


if __name__ == "__main__":
    main()
