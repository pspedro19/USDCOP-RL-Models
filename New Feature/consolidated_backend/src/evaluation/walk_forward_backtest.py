# backend/src/evaluation/walk_forward_backtest.py
"""
Walk-Forward Backtest con expanding window.

Simula producción real:
1. Entrena solo con datos pasados
2. Escala datos frescos en cada ventana (fit solo en train de la ventana)
3. Reentrena modelo en cada ventana
4. Calcula métricas por ventana y agregadas

Métricas calculadas:
- Direction Accuracy (DA)
- Sharpe Ratio
- Maximum Drawdown (MaxDD)
- Profit Factor
- Equity Curve

Usage:
    from src.evaluation.walk_forward_backtest import walk_forward_backtest

    results = walk_forward_backtest(
        model_factory=lambda: ModelFactory.create('ridge'),
        X=X,
        y=y,
        horizon=15,
        n_windows=5,
        min_train_pct=0.4,
        scaler_class=StandardScaler
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List, Tuple
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Resultado de una ventana de walk-forward."""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int

    # Métricas
    da: float
    mse: float
    rmse: float
    mae: float

    # Trading metrics
    sharpe: float
    max_drawdown: float
    profit_factor: float
    total_return: float

    # Predictions
    predictions: np.ndarray
    actuals: np.ndarray
    returns: np.ndarray


@dataclass
class BacktestResult:
    """Resultado completo del walk-forward backtest."""
    model_name: str
    horizon: int
    n_windows: int

    # Aggregate metrics
    da_mean: float
    da_std: float
    da_by_window: List[float]

    sharpe_mean: float
    sharpe_std: float
    sharpe_by_window: List[float]

    max_drawdown: float
    profit_factor: float
    total_return: float

    # Equity curve
    equity_curve: np.ndarray

    # Window details
    window_results: List[WindowResult]

    # Variance monitoring
    variance_ratios: List[float]
    avg_variance_ratio: float


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: 252 for daily, 52 for weekly, 12 for monthly

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / annualization_factor
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return < 1e-10:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return float(sharpe)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Cumulative returns (1 = starting value)

    Returns:
        Maximum drawdown as positive percentage (0.15 = 15% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0

    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max

    return float(np.max(drawdowns))


def calculate_profit_factor(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Uses directional accuracy: if prediction direction matches actual,
    it's a win weighted by actual magnitude.

    Args:
        predictions: Model predictions
        actuals: Actual values

    Returns:
        Profit factor (>1 is profitable)
    """
    # Directional returns: positive if direction correct
    correct_direction = np.sign(predictions) == np.sign(actuals)

    # Weight by actual magnitude
    wins = np.abs(actuals[correct_direction]).sum()
    losses = np.abs(actuals[~correct_direction]).sum()

    if losses < 1e-10:
        return float('inf') if wins > 0 else 1.0

    return float(wins / losses)


def walk_forward_backtest(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    horizon: int = 15,
    n_windows: int = 5,
    min_train_pct: float = 0.4,
    gap: int = None,
    scaler_class: type = StandardScaler,
    requires_scaling: bool = True,
    model_name: str = "model"
) -> BacktestResult:
    """
    Execute walk-forward backtest with expanding window.

    Simulates real production:
    1. Train only on past data
    2. Fresh scaler fitted on each window's training data
    3. Retrain model on each window
    4. Calculate metrics per window and aggregated

    Args:
        model_factory: Callable that returns a new model instance
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        horizon: Prediction horizon (used for gap if not specified)
        n_windows: Number of test windows
        min_train_pct: Minimum percentage of data for first training set
        gap: Gap between train and test (default: horizon)
        scaler_class: Scaler class to use (must have fit_transform/transform)
        requires_scaling: Whether to scale features
        model_name: Name for logging

    Returns:
        BacktestResult with all metrics and equity curve
    """
    n_samples = len(y)
    gap = gap or horizon

    # Calculate window boundaries
    min_train_samples = int(n_samples * min_train_pct)
    remaining = n_samples - min_train_samples - gap

    if remaining < n_windows * 20:
        logger.warning(
            f"Insufficient data for {n_windows} windows. "
            f"Reducing to {max(2, remaining // 50)} windows."
        )
        n_windows = max(2, remaining // 50)

    window_size = remaining // n_windows

    logger.info(
        f"Walk-Forward Backtest: {model_name} H={horizon}, "
        f"{n_windows} windows, window_size={window_size}"
    )

    # Storage for results
    window_results: List[WindowResult] = []
    all_predictions = []
    all_actuals = []
    equity_curve = [1.0]  # Start at 1
    variance_ratios = []

    for w in range(n_windows):
        # Define boundaries
        train_end = min_train_samples + w * window_size
        test_start = train_end + gap
        test_end = test_start + window_size if w < n_windows - 1 else n_samples

        # Split data
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        # Skip if insufficient data
        if len(X_train) < 100 or len(X_test) < 10:
            logger.warning(f"Window {w+1} skipped: insufficient data")
            continue

        # ===== FRESH SCALER PER WINDOW =====
        if requires_scaling:
            scaler = scaler_class()
            X_train_scaled = scaler.fit_transform(X_train)  # FIT on train
            X_test_scaled = scaler.transform(X_test)        # ONLY transform test
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # ===== TRAIN MODEL =====
        model = model_factory()
        model.fit(X_train_scaled, y_train)

        # ===== PREDICT =====
        predictions = model.predict(X_test_scaled)

        # ===== CALCULATE METRICS =====
        # Direction Accuracy
        da = float(np.mean(np.sign(predictions) == np.sign(y_test)))

        # MSE, RMSE, MAE
        mse = float(np.mean((predictions - y_test) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(predictions - y_test)))

        # Variance ratio (for collapse detection)
        pred_std = np.std(predictions)
        true_std = np.std(y_test)
        variance_ratio = pred_std / (true_std + 1e-8)
        variance_ratios.append(variance_ratio)

        # Trading returns: use sign of prediction * actual return
        # This is the return we get if we follow the model's direction
        trading_returns = np.sign(predictions) * y_test

        # Sharpe ratio for this window
        sharpe = calculate_sharpe_ratio(trading_returns)

        # Profit factor
        profit_factor = calculate_profit_factor(predictions, y_test)

        # Equity curve update
        for ret in trading_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))

        # Total return for window
        total_return = float(np.sum(trading_returns))

        # Max drawdown for window
        window_equity = [1.0]
        for ret in trading_returns:
            window_equity.append(window_equity[-1] * (1 + ret))
        max_dd = calculate_max_drawdown(np.array(window_equity))

        # Store window result
        result = WindowResult(
            window_idx=w,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train=len(X_train),
            n_test=len(X_test),
            da=da,
            mse=mse,
            rmse=rmse,
            mae=mae,
            sharpe=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            total_return=total_return,
            predictions=predictions,
            actuals=y_test,
            returns=trading_returns
        )
        window_results.append(result)

        all_predictions.extend(predictions)
        all_actuals.extend(y_test)

        logger.debug(
            f"Window {w+1}: train={len(X_train)}, test={len(X_test)}, "
            f"DA={da:.2%}, Sharpe={sharpe:.2f}, VR={variance_ratio:.3f}"
        )

    # ===== AGGREGATE METRICS =====
    if not window_results:
        raise ValueError("No valid windows generated")

    da_by_window = [r.da for r in window_results]
    sharpe_by_window = [r.sharpe for r in window_results]

    # Overall metrics
    equity_array = np.array(equity_curve)
    overall_max_dd = calculate_max_drawdown(equity_array)
    overall_profit_factor = calculate_profit_factor(
        np.array(all_predictions),
        np.array(all_actuals)
    )
    overall_return = float(equity_array[-1] - 1)

    backtest_result = BacktestResult(
        model_name=model_name,
        horizon=horizon,
        n_windows=len(window_results),
        da_mean=float(np.mean(da_by_window)),
        da_std=float(np.std(da_by_window)),
        da_by_window=da_by_window,
        sharpe_mean=float(np.mean(sharpe_by_window)),
        sharpe_std=float(np.std(sharpe_by_window)),
        sharpe_by_window=sharpe_by_window,
        max_drawdown=overall_max_dd,
        profit_factor=overall_profit_factor,
        total_return=overall_return,
        equity_curve=equity_array,
        window_results=window_results,
        variance_ratios=variance_ratios,
        avg_variance_ratio=float(np.mean(variance_ratios))
    )

    logger.info(
        f"Walk-Forward Complete: {model_name} H={horizon} - "
        f"DA={backtest_result.da_mean:.2%} +/- {backtest_result.da_std:.2%}, "
        f"Sharpe={backtest_result.sharpe_mean:.2f}, "
        f"MaxDD={overall_max_dd:.2%}, "
        f"VR={backtest_result.avg_variance_ratio:.3f}"
    )

    return backtest_result


def backtest_all_models(
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    horizons: List[int],
    model_factory_fn: Callable,
    n_windows: int = 5,
    **kwargs
) -> Dict[str, Dict[int, BacktestResult]]:
    """
    Run walk-forward backtest for multiple models and horizons.

    Args:
        model_names: List of model names
        X: Features array
        y: Target array
        horizons: List of horizons to test
        model_factory_fn: Function(model_name, horizon) -> model instance
        n_windows: Number of test windows
        **kwargs: Additional args passed to walk_forward_backtest

    Returns:
        Nested dict: {model_name: {horizon: BacktestResult}}
    """
    results = {}

    for model_name in model_names:
        results[model_name] = {}

        for horizon in horizons:
            logger.info(f"Backtesting {model_name} H={horizon}...")

            try:
                # Create factory for this model
                factory = lambda: model_factory_fn(model_name, horizon)

                result = walk_forward_backtest(
                    model_factory=factory,
                    X=X,
                    y=y,
                    horizon=horizon,
                    n_windows=n_windows,
                    model_name=model_name,
                    **kwargs
                )

                results[model_name][horizon] = result

            except Exception as e:
                logger.error(f"Backtest failed for {model_name} H={horizon}: {e}")
                results[model_name][horizon] = None

    return results


def create_backtest_summary_df(
    results: Dict[str, Dict[int, BacktestResult]]
) -> pd.DataFrame:
    """
    Create summary DataFrame from backtest results.

    Args:
        results: Nested dict from backtest_all_models

    Returns:
        DataFrame with columns: model, horizon, da_mean, da_std, sharpe, maxdd, pf, return
    """
    rows = []

    for model_name, horizon_results in results.items():
        for horizon, result in horizon_results.items():
            if result is None:
                continue

            rows.append({
                'model': model_name,
                'horizon': horizon,
                'da_mean': result.da_mean,
                'da_std': result.da_std,
                'sharpe_mean': result.sharpe_mean,
                'sharpe_std': result.sharpe_std,
                'max_drawdown': result.max_drawdown,
                'profit_factor': result.profit_factor,
                'total_return': result.total_return,
                'variance_ratio': result.avg_variance_ratio,
                'n_windows': result.n_windows
            })

    return pd.DataFrame(rows)
