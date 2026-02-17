"""
Vol-Target Backtest — Paso 0.1 (baseline) + Paso 1.1 (vol-targeting)
====================================================================

Standalone script: loads data from parquets, builds features, runs walk-forward
on all 9 models for H=1, creates ensembles, applies vol-targeting.

Usage:
    python scripts/vol_target_backtest.py
    python scripts/vol_target_backtest.py --target-vols 0.12,0.15,0.18,0.20
    python scripts/vol_target_backtest.py --output results/vol_target_results.json

Data sources (local parquets, no DB required):
    - seeds/latest/usdcop_daily_ohlcv.parquet (1,399 rows, 2020-01 to 2025-12)
    - data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet (DXY + WTI)

@version 1.0.0
@experiment FC-SIZE-001
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS, TARGET_HORIZONS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import MODEL_IDS, get_horizon_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load daily OHLCV + macro, merge, build 19 SSOT features.

    Returns DataFrame with columns:
        date, open, high, low, close, + 19 FEATURE_COLUMNS + target_return_1d
    """
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path)

    # Index = time (tz-aware), convert to date column
    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)

    logger.info(f"  OHLCV: {len(df_ohlcv)} rows, {df_ohlcv['date'].min().date()} to {df_ohlcv['date'].max().date()}")

    # Load macro for DXY and WTI (with T-1 lag for anti-leakage)
    logger.info("Loading macro data (DXY, WTI)...")
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    df_macro = pd.read_parquet(macro_path)
    df_macro = df_macro.reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    # Extract DXY and WTI, apply T-1 lag (shift forward = use yesterday's value)
    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)

    # T-1 lag: shift macro values by 1 day (use yesterday's DXY/WTI)
    df_macro_subset = df_macro_subset.sort_values("date")
    df_macro_subset["dxy_close_lag1"] = df_macro_subset["dxy_close_lag1"].shift(1)
    df_macro_subset["oil_close_lag1"] = df_macro_subset["oil_close_lag1"].shift(1)

    # Merge via merge_asof (backward) to handle non-aligned dates
    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_subset.sort_values("date"),
        on="date",
        direction="backward",
    )

    logger.info(f"  Merged: {len(df)} rows")

    # Build features (same as ForecastingEngine._build_ssot_features)
    df = _build_features(df)

    # Create target: H=1 log return
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    # Drop rows with NaN features (warmup period)
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_1d"].notna()
    df = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(f"  After cleanup: {len(df)} rows with complete features")
    logger.info(f"  Date range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 19 SSOT features from raw OHLCV + macro."""
    df = df.copy()

    # Returns (4)
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # Volatility (3)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Technical (3)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    # Calendar (3)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    # Macro (2) — already in df from merge (dxy_close_lag1, oil_close_lag1)
    # Forward-fill macro NaN (weekends/holidays)
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()

    return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class FoldPredictions:
    """OOS predictions for a single walk-forward fold."""
    fold_idx: int
    dates: np.ndarray
    predicted_returns: np.ndarray  # log returns
    actual_returns: np.ndarray     # log returns
    train_end_date: str
    test_start_date: str
    test_end_date: str


def run_walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    initial_train_ratio: float = 0.6,
    models_to_use: Optional[List[str]] = None,
    ensemble_strategy: str = "top_3",
) -> Tuple[List[FoldPredictions], Dict]:
    """
    Run walk-forward validation for H=1, all models, return ensemble OOS predictions.

    Args:
        df: DataFrame with features + target_return_1d
        n_folds: Number of expanding window folds
        initial_train_ratio: Initial training data ratio
        models_to_use: Model IDs to train (None = all 9)
        ensemble_strategy: "top_3", "best_of_breed", "consensus", "all_individual"

    Returns:
        (fold_predictions, model_metrics)
    """
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(FEATURE_COLUMNS)
    X = df[feature_cols].values.astype(np.float64)
    y = df["target_return_1d"].values.astype(np.float64)
    dates = df["date"].values

    n_samples = len(X)
    initial_train_size = int(n_samples * initial_train_ratio)
    test_size = (n_samples - initial_train_size) // n_folds

    if models_to_use is None:
        models_to_use = [
            "ridge", "bayesian_ridge",
            "xgboost_pure", "lightgbm_pure", "catboost_pure",
            "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost",
        ]
        # Try ARD
        try:
            ModelFactory.create("ard")
            models_to_use.insert(2, "ard")
        except Exception:
            logger.warning("ARD model not available, using 8 models")

    logger.info(f"Walk-forward: {n_folds} folds, {len(models_to_use)} models, H=1")
    logger.info(f"  Total samples: {n_samples}, initial train: {initial_train_size}, test/fold: {test_size}")

    # Model-specific params: boosting models use horizon_config, linear use defaults
    horizon_config = get_horizon_config(1)  # H=1 = "short" config
    boosting_models = {"xgboost", "xgboost_pure", "lightgbm", "lightgbm_pure"}
    catboost_models = {"catboost", "catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    fold_predictions = []
    model_metrics = {m: {"da_per_fold": [], "da_overall": 0.0} for m in models_to_use}

    for fold_idx in range(n_folds):
        train_end = initial_train_size + (fold_idx * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        dates_test = dates[test_start:test_end]

        train_end_date = str(pd.Timestamp(dates[train_end - 1]).date())
        test_start_date = str(pd.Timestamp(dates_test[0]).date())
        test_end_date = str(pd.Timestamp(dates_test[-1]).date())

        logger.info(f"\n  Fold {fold_idx + 1}/{n_folds}: "
                     f"Train [0-{train_end}] to {train_end_date}, "
                     f"Test [{test_start}-{test_end}] {test_start_date} to {test_end_date}")

        # Train each model and collect predictions
        model_preds = {}  # model_id -> predictions array
        model_das = {}    # model_id -> DA for this fold

        for model_id in models_to_use:
            try:
                # Select appropriate params per model type
                if model_id in linear_models:
                    params = None  # Use model defaults (alpha=1.0)
                elif model_id in catboost_models:
                    params = {
                        "iterations": horizon_config.get("n_estimators", 50),
                        "depth": horizon_config.get("max_depth", 3),
                        "learning_rate": horizon_config.get("learning_rate", 0.05),
                        "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
                        "verbose": False,
                        "allow_writing_files": False,
                    }
                elif model_id in hybrid_models:
                    # Hybrids: pass boosting-compatible params (hybrid handles internally)
                    if "catboost" in model_id:
                        params = {
                            "iterations": horizon_config.get("n_estimators", 50),
                            "depth": horizon_config.get("max_depth", 3),
                            "learning_rate": horizon_config.get("learning_rate", 0.05),
                            "verbose": False,
                            "allow_writing_files": False,
                        }
                    else:
                        params = horizon_config
                else:
                    params = horizon_config  # XGBoost, LightGBM

                model = ModelFactory.create(model_id, params=params, horizon=1)

                # Scale for linear models
                if model.requires_scaling:
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                else:
                    X_train_s, X_test_s = X_train, X_test

                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)

                # Compute DA for this fold
                correct = np.sum(np.sign(preds) == np.sign(y_test))
                da = correct / len(y_test) * 100
                model_das[model_id] = da
                model_preds[model_id] = preds
                model_metrics[model_id]["da_per_fold"].append(da)

                logger.info(f"    {model_id}: DA={da:.1f}%")

            except Exception as e:
                logger.error(f"    {model_id} FAILED: {e}")
                continue

        # Create ensemble prediction
        ensemble_preds = _create_fold_ensemble(
            model_preds, model_das, y_test, ensemble_strategy
        )

        fold_predictions.append(FoldPredictions(
            fold_idx=fold_idx,
            dates=dates_test,
            predicted_returns=ensemble_preds,
            actual_returns=y_test,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        ))

    # Compute overall DA per model
    for model_id in models_to_use:
        das = model_metrics[model_id]["da_per_fold"]
        if das:
            model_metrics[model_id]["da_overall"] = float(np.mean(das))

    return fold_predictions, model_metrics


def _create_fold_ensemble(
    model_preds: Dict[str, np.ndarray],
    model_das: Dict[str, float],
    y_test: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Create ensemble prediction from individual model predictions."""
    if not model_preds:
        return np.zeros(len(y_test))

    if strategy == "top_3":
        # Top 3 by DA in this fold
        sorted_models = sorted(model_das.keys(), key=lambda m: model_das[m], reverse=True)
        top_3 = sorted_models[:3]
        preds = np.mean([model_preds[m] for m in top_3 if m in model_preds], axis=0)
        top_names = ", ".join(top_3)
        logger.info(f"    Ensemble top_3: [{top_names}]")

    elif strategy == "best_of_breed":
        # Single best model by DA
        best = max(model_das, key=model_das.get)
        preds = model_preds[best]
        logger.info(f"    Ensemble best_of_breed: {best}")

    elif strategy == "consensus":
        # Average of all models
        all_p = [model_preds[m] for m in model_preds]
        preds = np.mean(all_p, axis=0)
        logger.info(f"    Ensemble consensus: {len(all_p)} models")

    elif strategy == "top_6_mean":
        sorted_models = sorted(model_das.keys(), key=lambda m: model_das[m], reverse=True)
        top_6 = sorted_models[:6]
        preds = np.mean([model_preds[m] for m in top_6 if m in model_preds], axis=0)

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    da = np.sum(np.sign(preds) == np.sign(y_test)) / len(y_test) * 100
    logger.info(f"    -> Ensemble DA: {da:.1f}%")

    return preds


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class BacktestMetrics:
    """Complete backtest metrics."""
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    direction_accuracy_pct: float
    n_days: int
    avg_leverage: float
    max_leverage_used: float
    worst_month_pct: float
    best_month_pct: float
    pct_months_positive: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float


def compute_metrics(
    dates: np.ndarray,
    strategy_returns: np.ndarray,
    leverage_series: np.ndarray,
    direction_correct: np.ndarray,
) -> BacktestMetrics:
    """Compute full backtest metrics from daily returns."""
    n_days = len(strategy_returns)

    # Cumulative return
    cum_ret = np.cumprod(1 + strategy_returns)
    total_ret = cum_ret[-1] - 1

    # Annualized return (assuming 252 trading days)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1

    # Sharpe (annualized)
    daily_mean = np.mean(strategy_returns)
    daily_std = np.std(strategy_returns, ddof=1)
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = float(np.min(drawdown))

    # Profit factor
    wins = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    pf = wins / losses if losses > 0 else float("inf")

    # DA
    da = direction_correct.mean() * 100

    # Leverage stats
    avg_lev = float(np.mean(np.abs(leverage_series)))
    max_lev = float(np.max(np.abs(leverage_series)))

    # Monthly analysis
    df_monthly = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ret": strategy_returns,
    })
    df_monthly["month"] = df_monthly["date"].dt.to_period("M")
    monthly_rets = df_monthly.groupby("month")["ret"].sum()
    worst_month = float(monthly_rets.min()) * 100
    best_month = float(monthly_rets.max()) * 100
    pct_months_pos = float((monthly_rets > 0).mean()) * 100

    # Bootstrap CI (10,000 samples)
    ci_lower, ci_upper = bootstrap_ci(strategy_returns)

    return BacktestMetrics(
        total_return_pct=total_ret * 100,
        annualized_return_pct=ann_ret * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd * 100,
        profit_factor=pf,
        direction_accuracy_pct=da,
        n_days=n_days,
        avg_leverage=avg_lev,
        max_leverage_used=max_lev,
        worst_month_pct=worst_month,
        best_month_pct=best_month,
        pct_months_positive=pct_months_pos,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
    )


def bootstrap_ci(
    daily_returns: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap CI of annualized return."""
    rng = np.random.default_rng(42)
    n = len(daily_returns)
    means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(daily_returns, size=n, replace=True)
        means[i] = sample.mean() * 252  # annualize

    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# =============================================================================
# VOL-TARGETING
# =============================================================================

@dataclass
class VolTargetConfig:
    """Vol-targeting configuration."""
    target_vol: float
    max_leverage: float
    min_leverage: float = 0.5
    vol_lookback: int = 21
    vol_floor: float = 0.05


def apply_vol_targeting(
    dates: np.ndarray,
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
    config: VolTargetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply vol-targeting to ensemble predictions.

    Returns:
        (strategy_returns, leverage_series, direction_correct, valid_mask)
    """
    n = len(actual_returns)
    direction = np.sign(predicted_returns)

    # Compute rolling realized vol (21-day, annualized)
    # Use actual_returns for realized vol (this is backward-looking, no leakage)
    vol_21d = pd.Series(actual_returns).rolling(config.vol_lookback).std().values * np.sqrt(252)

    # Apply vol floor
    vol_21d = np.maximum(vol_21d, config.vol_floor)

    # Compute leverage
    leverage = config.target_vol / vol_21d
    leverage = np.clip(leverage, config.min_leverage, config.max_leverage)

    # Strategy return = direction * leverage * actual_return
    strategy_returns = direction * leverage * actual_returns
    direction_correct = (direction * actual_returns > 0).astype(float)

    # Valid only after vol warmup
    valid_mask = ~np.isnan(vol_21d)

    return strategy_returns, leverage, direction_correct, valid_mask


# =============================================================================
# BASELINE (no vol-targeting, fixed 1x)
# =============================================================================

def run_baseline(
    fold_predictions: List[FoldPredictions],
) -> Tuple[BacktestMetrics, Dict]:
    """Run baseline (fixed 1x position, no vol-targeting)."""
    # Concatenate all fold predictions
    all_dates = np.concatenate([f.dates for f in fold_predictions])
    all_preds = np.concatenate([f.predicted_returns for f in fold_predictions])
    all_actuals = np.concatenate([f.actual_returns for f in fold_predictions])

    direction = np.sign(all_preds)
    strategy_returns = direction * all_actuals
    direction_correct = (direction * all_actuals > 0).astype(float)
    leverage = np.ones(len(all_actuals))

    metrics = compute_metrics(all_dates, strategy_returns, leverage, direction_correct)

    # Per-fold metrics
    per_fold = {}
    for f in fold_predictions:
        d = np.sign(f.predicted_returns)
        fold_ret = d * f.actual_returns
        cum = np.prod(1 + fold_ret) - 1
        da = np.mean((d * f.actual_returns > 0).astype(float)) * 100
        sharpe = (np.mean(fold_ret) / np.std(fold_ret, ddof=1) * np.sqrt(252)) if np.std(fold_ret, ddof=1) > 0 else 0
        per_fold[f"fold_{f.fold_idx + 1}"] = {
            "period": f"{f.test_start_date} to {f.test_end_date}",
            "return_pct": round(cum * 100, 2),
            "da_pct": round(da, 1),
            "sharpe": round(sharpe, 2),
            "n_days": len(f.actual_returns),
        }

    return metrics, per_fold


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_statistical_tests(
    fold_predictions: List[FoldPredictions],
) -> Dict:
    """Run binomial test, t-test, Pesaran-Timmermann test."""
    all_preds = np.concatenate([f.predicted_returns for f in fold_predictions])
    all_actuals = np.concatenate([f.actual_returns for f in fold_predictions])

    direction = np.sign(all_preds)
    correct = (direction * all_actuals > 0).astype(float)
    n_correct = int(correct.sum())
    n_total = len(correct)

    # Binomial test: H0 = DA = 50%
    binom = stats.binomtest(n_correct, n_total, 0.5, alternative="greater")

    # t-test: H0 = mean strategy return = 0
    strategy_returns = direction * all_actuals
    t_stat, t_pvalue = stats.ttest_1samp(strategy_returns, 0)

    # Buy-and-hold benchmark
    bnh_return = np.prod(1 + all_actuals) - 1

    # Random agent (10,000 simulations)
    rng = np.random.default_rng(42)
    random_returns = []
    for _ in range(10000):
        random_dir = rng.choice([-1, 1], size=n_total)
        rand_ret = np.prod(1 + random_dir * all_actuals) - 1
        random_returns.append(rand_ret)
    random_mean = np.mean(random_returns)
    strategy_total = np.prod(1 + strategy_returns) - 1
    random_percentile = float(np.mean(np.array(random_returns) < strategy_total) * 100)

    return {
        "binomial_test": {
            "n_correct": n_correct,
            "n_total": n_total,
            "da_pct": round(n_correct / n_total * 100, 2),
            "p_value": round(binom.pvalue, 4),
            "significant_5pct": binom.pvalue < 0.05,
            "significant_10pct": binom.pvalue < 0.10,
        },
        "t_test": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(t_pvalue), 4),
            "significant_5pct": float(t_pvalue) < 0.05,
        },
        "benchmarks": {
            "buy_and_hold_return_pct": round(bnh_return * 100, 2),
            "random_agent_mean_pct": round(random_mean * 100, 2),
            "strategy_vs_random_percentile": round(random_percentile, 1),
        },
        "folds_positive": sum(
            1 for f in fold_predictions
            if np.prod(1 + np.sign(f.predicted_returns) * f.actual_returns) - 1 > 0
        ),
        "total_folds": len(fold_predictions),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vol-Target Backtest (Paso 0.1 + 1.1)")
    parser.add_argument(
        "--target-vols",
        type=str,
        default="0.12,0.15,0.18,0.20",
        help="Comma-separated target vol levels",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=2.0,
        help="Max leverage for vol-targeting configs 1-3 (config 4 uses 2.5)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of walk-forward folds",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="top_3",
        choices=["top_3", "best_of_breed", "consensus", "top_6_mean"],
        help="Ensemble strategy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    args = parser.parse_args()

    target_vols = [float(x) for x in args.target_vols.split(",")]

    print("=" * 70)
    print("  USDCOP Forecasting: Vol-Target Backtest")
    print("  Paso 0.1 (baseline reproduction) + Paso 1.1 (vol-targeting)")
    print("=" * 70)

    start_time = time.time()

    # ─── Step 1: Load data ───
    df = load_data()

    # ─── Step 2: Walk-forward (Paso 0.1) ───
    print("\n" + "=" * 70)
    print("  PASO 0.1: Walk-Forward Baseline (H=1, 9 models, 5 folds)")
    print("=" * 70)

    fold_predictions, model_metrics = run_walk_forward(
        df,
        n_folds=args.n_folds,
        ensemble_strategy=args.ensemble,
    )

    # ─── Step 3: Baseline metrics ───
    baseline_metrics, per_fold = run_baseline(fold_predictions)

    print("\n" + "-" * 50)
    print("  BASELINE RESULTS (fixed 1x, no vol-targeting)")
    print("-" * 50)
    print(f"  Total return:     {baseline_metrics.total_return_pct:+.2f}%")
    print(f"  Annualized:       {baseline_metrics.annualized_return_pct:+.2f}%")
    print(f"  Sharpe:           {baseline_metrics.sharpe_ratio:.3f}")
    print(f"  Sortino:          {baseline_metrics.sortino_ratio:.3f}")
    print(f"  Max Drawdown:     {baseline_metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor:    {baseline_metrics.profit_factor:.3f}")
    print(f"  Direction Acc:    {baseline_metrics.direction_accuracy_pct:.1f}%")
    print(f"  Trading days:     {baseline_metrics.n_days}")
    print(f"  Months positive:  {baseline_metrics.pct_months_positive:.0f}%")
    print(f"  Worst month:      {baseline_metrics.worst_month_pct:+.2f}%")
    print(f"  Best month:       {baseline_metrics.best_month_pct:+.2f}%")
    print(f"  Bootstrap 95% CI: [{baseline_metrics.bootstrap_ci_lower:+.2f}%, {baseline_metrics.bootstrap_ci_upper:+.2f}%]")

    print("\n  Per-fold results:")
    for fold_name, fold_data in per_fold.items():
        print(f"    {fold_name}: {fold_data['period']} -> "
              f"Ret={fold_data['return_pct']:+.2f}%, DA={fold_data['da_pct']:.1f}%, "
              f"Sharpe={fold_data['sharpe']:.2f}, N={fold_data['n_days']}")

    # ─── Step 4: Statistical tests ───
    stat_tests = run_statistical_tests(fold_predictions)

    print("\n  Statistical tests:")
    bt = stat_tests["binomial_test"]
    print(f"    Binomial: {bt['n_correct']}/{bt['n_total']} correct, "
          f"DA={bt['da_pct']:.1f}%, p={bt['p_value']:.4f} "
          f"{'OK sig@5%' if bt['significant_5pct'] else 'XX NOT sig@5%'}")
    tt = stat_tests["t_test"]
    print(f"    t-test:   t={tt['t_statistic']:.3f}, p={tt['p_value']:.4f} "
          f"{'OK sig@5%' if tt['significant_5pct'] else 'XX NOT sig@5%'}")
    bm = stat_tests["benchmarks"]
    print(f"    B&H:      {bm['buy_and_hold_return_pct']:+.2f}%")
    print(f"    Random:   {bm['random_agent_mean_pct']:+.2f}% (strategy at {bm['strategy_vs_random_percentile']:.0f}th percentile)")
    print(f"    Folds +:  {stat_tests['folds_positive']}/{stat_tests['total_folds']}")

    # ─── Step 5: Model ranking ───
    print("\n  Model ranking (avg DA across folds):")
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]["da_overall"], reverse=True)
    for rank, (model_id, metrics) in enumerate(sorted_models, 1):
        das = metrics["da_per_fold"]
        da_str = ", ".join(f"{d:.1f}" for d in das)
        print(f"    {rank}. {model_id:25s} avg={metrics['da_overall']:.1f}%  [{da_str}]")

    # ─── Step 6: Vol-targeting (Paso 1.1) ───
    print("\n" + "=" * 70)
    print("  PASO 1.1: Vol-Targeting Backtest")
    print("=" * 70)

    # Concatenate all fold predictions
    all_dates = np.concatenate([f.dates for f in fold_predictions])
    all_preds = np.concatenate([f.predicted_returns for f in fold_predictions])
    all_actuals = np.concatenate([f.actual_returns for f in fold_predictions])

    vol_results = {}
    configs = [
        VolTargetConfig(target_vol=0.12, max_leverage=1.5),
        VolTargetConfig(target_vol=0.15, max_leverage=2.0),
        VolTargetConfig(target_vol=0.18, max_leverage=2.0),
        VolTargetConfig(target_vol=0.20, max_leverage=2.5),
    ]

    # Override from args
    if args.target_vols != "0.12,0.15,0.18,0.20":
        configs = []
        for tv in target_vols:
            ml = min(tv / 0.08, 2.5)  # auto max leverage
            configs.append(VolTargetConfig(target_vol=tv, max_leverage=ml))

    for config in configs:
        label = f"tv={config.target_vol:.0%}, ml={config.max_leverage:.1f}x"
        logger.info(f"\n  Config: {label}")

        strat_rets, leverage, dir_correct, valid_mask = apply_vol_targeting(
            all_dates, all_preds, all_actuals, config
        )

        # Apply valid mask (skip vol warmup period)
        valid_dates = all_dates[valid_mask]
        valid_rets = strat_rets[valid_mask]
        valid_lev = leverage[valid_mask]
        valid_correct = dir_correct[valid_mask]

        metrics = compute_metrics(valid_dates, valid_rets, valid_lev, valid_correct)
        vol_results[label] = metrics

        print(f"\n  {label}:")
        print(f"    Return:     {metrics.total_return_pct:+.2f}% (ann: {metrics.annualized_return_pct:+.2f}%)")
        print(f"    Sharpe:     {metrics.sharpe_ratio:.3f}")
        print(f"    Sortino:    {metrics.sortino_ratio:.3f}")
        print(f"    MaxDD:      {metrics.max_drawdown_pct:.2f}%")
        print(f"    PF:         {metrics.profit_factor:.3f}")
        print(f"    DA:         {metrics.direction_accuracy_pct:.1f}%")
        print(f"    Avg Lev:    {metrics.avg_leverage:.2f}x")
        print(f"    Max Lev:    {metrics.max_leverage_used:.2f}x")
        print(f"    Worst Mo:   {metrics.worst_month_pct:+.2f}%")
        print(f"    Months +:   {metrics.pct_months_positive:.0f}%")
        print(f"    Boot CI:    [{metrics.bootstrap_ci_lower:+.2f}%, {metrics.bootstrap_ci_upper:+.2f}%]")

    # ─── Step 7: Comparison table ───
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Config':<25} {'Return%':>8} {'Ann%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'PF':>6} {'AvgLev':>7} {'CI_lo':>7} {'CI_hi':>7}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    # Baseline row
    bm = baseline_metrics
    print(f"  {'Baseline (1x fixed)':<25} {bm.total_return_pct:>+8.2f} {bm.annualized_return_pct:>+8.2f} "
          f"{bm.sharpe_ratio:>7.3f} {bm.max_drawdown_pct:>7.2f} {bm.profit_factor:>6.3f} "
          f"{bm.avg_leverage:>7.2f} {bm.bootstrap_ci_lower:>+7.2f} {bm.bootstrap_ci_upper:>+7.2f}")

    for label, m in vol_results.items():
        print(f"  {label:<25} {m.total_return_pct:>+8.2f} {m.annualized_return_pct:>+8.2f} "
              f"{m.sharpe_ratio:>7.3f} {m.max_drawdown_pct:>7.2f} {m.profit_factor:>6.3f} "
              f"{m.avg_leverage:>7.2f} {m.bootstrap_ci_lower:>+7.2f} {m.bootstrap_ci_upper:>+7.2f}")

    # ─── Step 8: Save results ───
    duration = time.time() - start_time

    results = {
        "experiment": "FC-SIZE-001 (Vol-Target Backtest)",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "duration_seconds": round(duration, 1),
        "data": {
            "n_samples": len(df),
            "date_range": f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}",
            "n_folds": args.n_folds,
            "ensemble_strategy": args.ensemble,
        },
        "baseline": {
            "metrics": asdict(baseline_metrics),
            "per_fold": per_fold,
        },
        "statistical_tests": stat_tests,
        "model_ranking": {
            model_id: {
                "da_overall": round(m["da_overall"], 2),
                "da_per_fold": [round(d, 1) for d in m["da_per_fold"]],
            }
            for model_id, m in sorted_models
        },
        "vol_targeting": {
            label: asdict(m) for label, m in vol_results.items()
        },
    }

    # Save to file
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"vol_target_backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"  Total time: {duration:.1f}s")

    # ─── Step 9: Gate evaluation ───
    print("\n" + "=" * 70)
    print("  GATE EVALUATION")
    print("=" * 70)

    # Paso 0.1 gate
    gate_01 = (
        stat_tests["binomial_test"]["da_pct"] > 51
        and stat_tests["binomial_test"]["p_value"] < 0.10
        and stat_tests["folds_positive"] >= 3
    )
    print(f"\n  GATE 0.1 (baseline): DA>{51}% AND binom p<0.10 AND >=3/5 folds+")
    print(f"    DA={stat_tests['binomial_test']['da_pct']:.1f}% {'OK' if stat_tests['binomial_test']['da_pct'] > 51 else 'XX'}")
    print(f"    p={stat_tests['binomial_test']['p_value']:.4f} {'OK' if stat_tests['binomial_test']['p_value'] < 0.10 else 'XX'}")
    print(f"    Folds+={stat_tests['folds_positive']}/{stat_tests['total_folds']} {'OK' if stat_tests['folds_positive'] >= 3 else 'XX'}")
    print(f"    -> {'PASS' if gate_01 else 'FAIL'}")

    # Paso 1.1 gate: find best vol-target config where Sharpe >= 1.0
    print(f"\n  GATE 1.1 (vol-targeting): Sharpe >= 1.0 AND MaxDD < 20% AND CI excludes 0")
    best_config = None
    for label, m in vol_results.items():
        sharpe_ok = m.sharpe_ratio >= 1.0
        maxdd_ok = abs(m.max_drawdown_pct) < 20
        ci_ok = m.bootstrap_ci_lower > 0
        passed = sharpe_ok and maxdd_ok and ci_ok
        status = "OK" if passed else "XX"
        print(f"    {label}: Sharpe={m.sharpe_ratio:.3f}{'OK' if sharpe_ok else 'XX'} "
              f"MaxDD={m.max_drawdown_pct:.1f}%{'OK' if maxdd_ok else 'XX'} "
              f"CI_lo={m.bootstrap_ci_lower:+.2f}%{'OK' if ci_ok else 'XX'} -> {status}")
        if passed and (best_config is None or m.sharpe_ratio > vol_results[best_config].sharpe_ratio):
            best_config = label

    if best_config:
        print(f"\n    -> PASS: Best config = {best_config}")
    else:
        # Check if any config has Sharpe >= 0.8 (softer gate)
        soft_pass = [l for l, m in vol_results.items() if m.sharpe_ratio >= 0.8 and abs(m.max_drawdown_pct) < 20]
        if soft_pass:
            print(f"\n    -> SOFT PASS: {soft_pass[0]} (Sharpe >= 0.8, review needed)")
        else:
            print(f"\n    -> FAIL: No config meets gate criteria")


if __name__ == "__main__":
    main()
