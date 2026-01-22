# pipeline_limpio_regresion/run_hybrid_ensemble.py
"""
HYBRID ENSEMBLE PIPELINE - 7 Models:

1. RIDGE: Linear regression with L2 regularization (baseline)
2. BAYESIAN RIDGE: Probabilistic linear regression
3. ARD: Automatic Relevance Determination - prior individual por feature
4. DART XGBOOST: Dropout trees to prevent overfitting
5. HYBRID XGBOOST: Classification (direction) × Ridge (magnitude)
6. HYBRID LIGHTGBM: Classification (direction) × Ridge (magnitude)
7. HYBRID CATBOOST: Classification (direction) × Ridge (magnitude)

All models generate NUMERICAL price predictions for ALL horizons.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.features.common import prepare_features, create_targets

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [1, 5, 10, 15, 20, 25, 30]
# 10 MODELOS COMPLETOS:
# - 3 Lineales: ridge, bayesian_ridge, ard
# - 3 Boosting Puros: xgboost_pure, lightgbm_pure, catboost_pure
# - 1 DART: dart_xgboost
# - 3 Híbridos: hybrid_xgboost, hybrid_lightgbm, hybrid_catboost
MODELS = [
    'ridge', 'bayesian_ridge', 'ard',                    # Lineales
    'xgboost_pure', 'lightgbm_pure', 'catboost_pure',    # Boosting Puros
    'dart_xgboost',                                       # DART
    'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'  # Híbridos
]
RANDOM_STATE = 42

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(r"C:\Users\pedro\OneDrive\Documents\data\RL_COMBINED_ML_FEATURES_FIXED.csv")
OUTPUT_DIR = BASE_DIR / "results" / "hybrid_ensemble" / datetime.now().strftime('%Y%m%d_%H%M%S')

# Ensemble weights (will be adjusted based on performance)
# Boosting puros tienen peso bajo porque tienden a colapsar (DA ~46-48%)
# Híbridos tienen peso alto porque resuelven el problema de colapso
ENSEMBLE_WEIGHTS = {
    # Lineales (45% total) - Estables y robustos
    'ridge': 0.20,
    'bayesian_ridge': 0.15,
    'ard': 0.10,
    # Boosting Puros (10% total) - Propensos a colapso, peso reducido
    'xgboost_pure': 0.03,
    'lightgbm_pure': 0.03,
    'catboost_pure': 0.04,
    # DART (5%) - Mejor que puros pero no tanto como híbridos
    'dart_xgboost': 0.05,
    # Híbridos (40% total) - Los mejores para FX
    'hybrid_xgboost': 0.15,
    'hybrid_lightgbm': 0.12,
    'hybrid_catboost': 0.13
}


# =============================================================================
# MODEL CLASSES
# =============================================================================

class HybridDirectionMagnitude:
    """
    HYBRID APPROACH: Classification (direction) × Ridge (magnitude)

    How it works:
    1. XGBoost/LightGBM/CatBoost CLASSIFIER predicts P(UP)
    2. Ridge REGRESSOR predicts magnitude |return|
    3. Final prediction = sign(P(UP) - 0.5) × |Ridge prediction|

    Why it prevents collapse:
    - Classification loss (log-loss) PENALIZES predicting P=0.5
    - Model is FORCED to commit to a direction (UP or DOWN)
    - Ridge provides stable magnitude that never collapses
    """

    def __init__(self, classifier_type: str = 'xgboost'):
        self.classifier_type = classifier_type
        self.classifier = None
        self.ridge = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train both classifier (direction) and Ridge (magnitude).
        """
        # Create binary target: 1 = UP, 0 = DOWN
        y_binary = (y > 0).astype(int)

        # Scaler for Ridge
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 1. Train classifier for DIRECTION
        if self.classifier_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.classifier = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_weight=10,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=RANDOM_STATE,
                    verbosity=0
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.classifier = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=RANDOM_STATE
                )

        elif self.classifier_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                self.classifier = LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_samples=20,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    force_col_wise=True
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.classifier = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    random_state=RANDOM_STATE
                )

        elif self.classifier_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                self.classifier = CatBoostClassifier(
                    iterations=100,
                    depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    l2_leaf_reg=0.5,
                    random_strength=1.0,
                    random_state=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.classifier = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    random_state=RANDOM_STATE
                )

        # Train classifier
        self.classifier.fit(X, y_binary)

        # 2. Train Ridge for MAGNITUDE
        self.ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
        self.ridge.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict return using: direction (classifier) × magnitude (Ridge)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Step 1: Get direction probability from classifier
        proba = self.classifier.predict_proba(X)
        prob_up = proba[:, 1]  # P(UP)

        # Convert to direction: +1 if P(UP) > 0.5, else -1
        direction = np.where(prob_up > 0.5, 1, -1)

        # Step 2: Get magnitude from Ridge (absolute value)
        X_scaled = self.scaler.transform(X)
        magnitude = np.abs(self.ridge.predict(X_scaled))

        # Step 3: Combine
        predicted_return = direction * magnitude

        return predicted_return

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of UP direction."""
        return self.classifier.predict_proba(X)[:, 1]

    def get_direction_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get classifier direction accuracy."""
        y_binary = (y > 0).astype(int)
        pred_binary = self.classifier.predict(X)
        return np.mean(pred_binary == y_binary) * 100

    def get_components(self, X: np.ndarray) -> Dict:
        """Get individual components for analysis."""
        proba = self.classifier.predict_proba(X)
        prob_up = proba[:, 1]
        direction = np.where(prob_up > 0.5, 1, -1)

        X_scaled = self.scaler.transform(X)
        ridge_pred = self.ridge.predict(X_scaled)
        magnitude = np.abs(ridge_pred)

        return {
            'prob_up': prob_up,
            'direction': direction,
            'ridge_prediction': ridge_pred,
            'magnitude': magnitude,
            'final_prediction': direction * magnitude
        }


class DARTXGBoost:
    """
    DART XGBoost: Dropout Additive Regression Trees
    Randomly drops trees during training to prevent co-adaptation.
    """

    def __init__(self, drop_rate: float = 0.15):
        self.drop_rate = drop_rate
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                booster='dart',
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                rate_drop=self.drop_rate,
                skip_drop=0.5,
                sample_type='uniform',
                normalize_type='tree',
                random_state=RANDOM_STATE,
                verbosity=0
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# =============================================================================
# BOOSTING PURE MODELS (Con variance scaling para evitar colapso)
# =============================================================================

class XGBoostPure:
    """
    XGBoost Pure Regressor con variance scaling.

    Problema conocido: En mercados de bajo SNR, tiende a predecir ~0.
    Solución: Variance scaling con cap para preservar señal direccional.
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_weight=10,
                random_state=RANDOM_STATE,
                verbosity=0
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)

        # Variance scaling para evitar colapso
        pred_std = np.std(preds)
        pred_mean = np.mean(preds)
        min_pred_std = 0.005
        max_scale_factor = 10.0

        if pred_std < min_pred_std and pred_std > 1e-8:
            scale_factor = min(min_pred_std / pred_std, max_scale_factor)
            preds = pred_mean + (preds - pred_mean) * scale_factor

        return preds


class LightGBMPure:
    """
    LightGBM Pure Regressor con variance scaling.

    Características: Histogram-based splitting, leaf-wise growth.
    Mismo problema de colapso que XGBoost, misma solución.
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=RANDOM_STATE,
                verbose=-1,
                force_col_wise=True
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)

        # Variance scaling
        pred_std = np.std(preds)
        pred_mean = np.mean(preds)
        min_pred_std = 0.005
        max_scale_factor = 10.0

        if pred_std < min_pred_std and pred_std > 1e-8:
            scale_factor = min(min_pred_std / pred_std, max_scale_factor)
            preds = pred_mean + (preds - pred_mean) * scale_factor

        return preds


class CatBoostPure:
    """
    CatBoost Pure Regressor con variance scaling.

    Características: Ordered boosting, symmetric trees.
    Más robusto que XGBoost/LightGBM pero aún puede colapsar.
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(
                iterations=150,
                depth=4,
                learning_rate=0.05,
                subsample=0.8,
                l2_leaf_reg=1.0,
                random_strength=1.0,
                random_state=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)

        # Variance scaling
        pred_std = np.std(preds)
        pred_mean = np.mean(preds)
        min_pred_std = 0.005
        max_scale_factor = 10.0

        if pred_std < min_pred_std and pred_std > 1e-8:
            scale_factor = min(min_pred_std / pred_std, max_scale_factor)
            preds = pred_mean + (preds - pred_mean) * scale_factor

        return preds


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare data."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# TRAINING
# =============================================================================

def train_all_models(df: pd.DataFrame, feature_cols: List[str], targets: Dict[int, pd.Series]):
    """Train all 10 model types for all horizons."""

    print("\n" + "="*120)
    print("TRAINING COMPLETE ENSEMBLE - 10 MODELS x 7 HORIZONS")
    print("="*120)

    results = {model: {} for model in MODELS}
    scalers = {}

    for h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON {h} DAYS")
        print(f"{'='*60}")

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

        # Scaler for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scalers[h] = scaler

        print(f"Train: {len(X_train)}, Test: {len(X_test)}, Gap: {gap}")
        print("-"*60)

        # =====================================================================
        # MODEL 1: RIDGE
        # =====================================================================
        ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
        ridge.fit(X_train_scaled, y_train)

        pred_train = ridge.predict(X_train_scaled)
        pred_test = ridge.predict(X_test_scaled)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['ridge'][h] = {
            'model': ridge,
            'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"Ridge:            DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 2: BAYESIAN RIDGE
        # =====================================================================
        bayesian = BayesianRidge(max_iter=300)
        bayesian.fit(X_train_scaled, y_train)

        pred_train = bayesian.predict(X_train_scaled)
        pred_test = bayesian.predict(X_test_scaled)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['bayesian_ridge'][h] = {
            'model': bayesian,
            'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"Bayesian Ridge:   DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 3: ARD (Automatic Relevance Determination)
        # Prior individual por feature - seleccion automatica bayesiana
        # =====================================================================
        ard = ARDRegression(max_iter=500, tol=1e-4)
        ard.fit(X_train_scaled, y_train)

        pred_train = ard.predict(X_train_scaled)
        pred_test = ard.predict(X_test_scaled)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        # Count relevant features (lambda < threshold means relevant)
        lambda_threshold = 1e6  # Features with lambda > this are pruned
        n_relevant = np.sum(ard.lambda_ < lambda_threshold)

        results['ard'][h] = {
            'model': ard,
            'scaler': scaler,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test,
            'n_relevant_features': n_relevant,
            'total_features': len(feature_cols)
        }

        print(f"ARD:              DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (features={n_relevant}/{len(feature_cols)})")

        # =====================================================================
        # MODEL 4: XGBOOST PURE
        # =====================================================================
        xgb_pure = XGBoostPure()
        xgb_pure.fit(X_train, y_train)

        pred_train = xgb_pure.predict(X_train)
        pred_test = xgb_pure.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['xgboost_pure'][h] = {
            'model': xgb_pure,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"XGBoost Pure:     DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 5: LIGHTGBM PURE
        # =====================================================================
        lgb_pure = LightGBMPure()
        lgb_pure.fit(X_train, y_train)

        pred_train = lgb_pure.predict(X_train)
        pred_test = lgb_pure.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['lightgbm_pure'][h] = {
            'model': lgb_pure,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"LightGBM Pure:    DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 6: CATBOOST PURE
        # =====================================================================
        cat_pure = CatBoostPure()
        cat_pure.fit(X_train, y_train)

        pred_train = cat_pure.predict(X_train)
        pred_test = cat_pure.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['catboost_pure'][h] = {
            'model': cat_pure,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"CatBoost Pure:    DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 7: DART XGBOOST
        # =====================================================================
        dart = DARTXGBoost(drop_rate=0.15)
        dart.fit(X_train, y_train)

        pred_train = dart.predict(X_train)
        pred_test = dart.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)

        results['dart_xgboost'][h] = {
            'model': dart,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'pred_test': pred_test
        }

        print(f"DART XGBoost:     DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}")

        # =====================================================================
        # MODEL 4: HYBRID XGBOOST
        # =====================================================================
        hybrid_xgb = HybridDirectionMagnitude(classifier_type='xgboost')
        hybrid_xgb.fit(X_train, y_train)

        pred_train = hybrid_xgb.predict(X_train)
        pred_test = hybrid_xgb.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_xgb.get_direction_accuracy(X_test, y_test)

        results['hybrid_xgboost'][h] = {
            'model': hybrid_xgb,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'classifier_acc': classifier_acc,
            'pred_test': pred_test
        }

        print(f"Hybrid XGBoost:   DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (classifier={classifier_acc:.1f}%)")

        # =====================================================================
        # MODEL 5: HYBRID LIGHTGBM
        # =====================================================================
        hybrid_lgb = HybridDirectionMagnitude(classifier_type='lightgbm')
        hybrid_lgb.fit(X_train, y_train)

        pred_train = hybrid_lgb.predict(X_train)
        pred_test = hybrid_lgb.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_lgb.get_direction_accuracy(X_test, y_test)

        results['hybrid_lightgbm'][h] = {
            'model': hybrid_lgb,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'classifier_acc': classifier_acc,
            'pred_test': pred_test
        }

        print(f"Hybrid LightGBM:  DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (classifier={classifier_acc:.1f}%)")

        # =====================================================================
        # MODEL 6: HYBRID CATBOOST
        # =====================================================================
        hybrid_cat = HybridDirectionMagnitude(classifier_type='catboost')
        hybrid_cat.fit(X_train, y_train)

        pred_train = hybrid_cat.predict(X_train)
        pred_test = hybrid_cat.predict(X_test)

        da_train = np.mean(np.sign(pred_train) == np.sign(y_train)) * 100
        da_test = np.mean(np.sign(pred_test) == np.sign(y_test)) * 100
        var_ratio = np.var(pred_test) / (np.var(y_test) + 1e-8)
        classifier_acc = hybrid_cat.get_direction_accuracy(X_test, y_test)

        results['hybrid_catboost'][h] = {
            'model': hybrid_cat,
            'da_train': da_train,
            'da_test': da_test,
            'var_ratio': var_ratio,
            'classifier_acc': classifier_acc,
            'pred_test': pred_test
        }

        print(f"Hybrid CatBoost:  DA_train={da_train:.1f}%  DA_test={da_test:.1f}%  var_ratio={var_ratio:.3f}  (classifier={classifier_acc:.1f}%)")

    return results, scalers


# =============================================================================
# FORWARD FORECAST
# =============================================================================

def generate_forecasts(results: Dict, scalers: Dict, df: pd.DataFrame, feature_cols: List[str]):
    """Generate forward forecasts for all models and horizons."""

    print("\n" + "="*120)
    print("FORWARD FORECAST - ALL MODELS x ALL HORIZONS")
    print("="*120)

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
        'dart_xgboost': 'DART XGBoost',
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
                elif model_name == 'dart_xgboost':
                    log_return = float(model.predict(X_latest)[0])
                else:  # Hybrid models
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

                # Target date
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

    # =========================================================================
    # ENSEMBLE FORECAST
    # =========================================================================
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

            print(f"{predicted_price:>12,.0f}", end="")
        else:
            print(f"{'N/A':>12}", end="")

    print()
    print("="*104)

    return forecasts


# =============================================================================
# SUMMARY AND COMPARISON
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
        'dart_xgboost': 'DART XGBoost',
        'hybrid_xgboost': 'Hybrid XGBoost',
        'hybrid_lightgbm': 'Hybrid LightGBM',
        'hybrid_catboost': 'Hybrid CatBoost'
    }

    print("\n" + "="*120)
    print("SUMMARY - DIRECTION ACCURACY (Test)")
    print("="*120)

    print(f"\n{'MODEL':<20}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>10}", end="")
    print(f"{'AVG':>10}")
    print("-"*100)

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

        if das:
            print(f"{np.mean(das):>10.1f}")
        else:
            print()

    # Variance ratio summary
    print("\n" + "="*120)
    print("VARIANCE RATIO (Test) - Should be > 0.1 to avoid collapse")
    print("="*120)

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
                if vr < 0.1:
                    print(f"{vr:>10.3f}*", end="")  # Mark collapsed
                else:
                    print(f"{vr:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")

        if vrs:
            print(f"{np.mean(vrs):>10.3f}")
        else:
            print()

    # Best model per horizon
    print("\n" + "="*120)
    print("BEST MODEL PER HORIZON (by DA)")
    print("="*120)

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

    # Hybrid comparison
    print("\n" + "="*120)
    print("HYBRID MODELS COMPARISON")
    print("="*120)

    hybrid_models = ['hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost']

    print(f"\n{'MODEL':<20}{'Avg DA':>12}{'Avg VarRatio':>15}{'Avg ClassAcc':>15}")
    print("-"*62)

    for model_name in hybrid_models:
        das = []
        vrs = []
        cas = []

        for h in HORIZONS:
            if h in results[model_name]:
                das.append(results[model_name][h]['da_test'])
                vrs.append(results[model_name][h]['var_ratio'])
                if 'classifier_acc' in results[model_name][h]:
                    cas.append(results[model_name][h]['classifier_acc'])

        avg_da = np.mean(das) if das else 0
        avg_vr = np.mean(vrs) if vrs else 0
        avg_ca = np.mean(cas) if cas else 0

        print(f"{MODEL_DISPLAY[model_name]:<20}{avg_da:>12.1f}%{avg_vr:>15.3f}{avg_ca:>15.1f}%")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results: Dict, forecasts: List, scalers: Dict, output_dir: Path):
    """Save all results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

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

        joblib.dump(model_data, models_dir / f"{model_name}.pkl")

    # Save scalers
    joblib.dump(scalers, models_dir / "scalers.pkl")

    # Save forecasts CSV
    df_forecasts = pd.DataFrame(forecasts)
    df_forecasts.to_csv(output_dir / "forecasts.csv", index=False)

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
            metrics.append(row)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(output_dir / "metrics.csv", index=False)

    # Save summary JSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'models': MODELS,
        'horizons': HORIZONS,
        'ensemble_weights': ENSEMBLE_WEIGHTS,
        'n_forecasts': len(forecasts),
        'best_model_overall': df_metrics.groupby('model')['da_test'].mean().idxmax(),
        'avg_da_by_model': df_metrics.groupby('model')['da_test'].mean().to_dict()
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*120)
    print("COMPLETE ENSEMBLE PIPELINE - 10 MODELS")
    print("="*120)
    print("\nLinear Models (3):")
    print("  1. Ridge (baseline)")
    print("  2. Bayesian Ridge")
    print("  3. ARD = Automatic Relevance Determination")
    print("\nBoosting Pure Models (3) - Con variance scaling:")
    print("  4. XGBoost Pure")
    print("  5. LightGBM Pure")
    print("  6. CatBoost Pure")
    print("\nDART Model (1):")
    print("  7. DART XGBoost = Dropout trees para prevenir overfitting")
    print("\nHybrid Models (3) - Clasificador(direccion) x Ridge(magnitud):")
    print("  8. Hybrid XGBoost")
    print("  9. Hybrid LightGBM")
    print(" 10. Hybrid CatBoost")
    print("="*120)

    # Load data
    df = load_data()
    df, feature_cols = prepare_features(df)
    targets = create_targets(df)

    # Train all models
    results, scalers = train_all_models(df, feature_cols, targets)

    # Generate forecasts
    forecasts = generate_forecasts(results, scalers, df, feature_cols)

    # Print summary
    print_summary(results, forecasts)

    # Save results
    save_results(results, forecasts, scalers, OUTPUT_DIR)

    print("\n" + "="*120)
    print("ENSEMBLE WEIGHTS USED:")
    for model, weight in ENSEMBLE_WEIGHTS.items():
        print(f"  {model}: {weight*100:.0f}%")
    print("="*120)
    print("\nDONE!")


if __name__ == "__main__":
    main()
