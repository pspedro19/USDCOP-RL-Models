"""
Regime Detection Features for USDCOP Forecasting
=================================================

Pre-registered features for EXP-REGIME-001.
Pure functions: no I/O, no side effects, no state.

Features:
  F1: trend_slope_60d     — Normalized OLS slope (direction)
  F2: range_ratio_20d     — Normalized price range (magnitude)
  F3: return_20d_sign     — Binary regime indicator
  F4: vol_regime_ratio    — Short/long vol ratio (vol expansion/contraction)
  F5: hmm_regime_features — 3-state HMM posterior probabilities

All features use ONLY past data (no lookahead).
All parameters are fixed (no optimization after pre-registration).

@experiment EXP-REGIME-001
@created 2026-02-16
@status PRE-REGISTERED
"""

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# F1: Trend Slope (60-day normalized OLS)
# =============================================================================

def trend_slope_60d(close: pd.Series, window: int = 60) -> pd.Series:
    """Normalized slope of 60-day linear regression on close prices.

    Positive = COP depreciating (USD/COP trending up).
    Negative = COP appreciating (USD/COP trending down).
    Near zero = range-bound.

    Args:
        close: Daily close prices.
        window: Lookback window in trading days.

    Returns:
        Series of normalized slope values.
    """
    def _slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
        mean_price = np.mean(arr)
        return slope / mean_price if mean_price != 0 else 0.0

    return close.rolling(window, min_periods=window).apply(_slope, raw=True)


# =============================================================================
# F2: Range Ratio (20-day normalized range)
# =============================================================================

def range_ratio_20d(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Normalized price range: high = trending, low = range-bound.

    Args:
        high: Daily high prices.
        low: Daily low prices.
        close: Daily close prices.
        window: Lookback window in trading days.

    Returns:
        Series of (max-min)/mean values.
    """
    rolling_range = high.rolling(window).max() - low.rolling(window).min()
    rolling_mean = close.rolling(window).mean()
    return rolling_range / rolling_mean


# =============================================================================
# F3: Return 20-day Sign (binary regime)
# =============================================================================

def return_20d_sign(close: pd.Series) -> pd.Series:
    """Binary regime indicator: +1 = recent depreciation, -1 = appreciation.

    Intentionally redundant with return_20d (continuous). Tests whether
    the sign alone carries information for tree model splits.

    Args:
        close: Daily close prices.

    Returns:
        Series of +1/-1 values.
    """
    return np.sign(close.pct_change(20))


# =============================================================================
# F4: Vol Regime Ratio (short/long vol)
# =============================================================================

def vol_regime_ratio(
    close: pd.Series, short: int = 5, long: int = 60
) -> pd.Series:
    """Vol regime: >1 = vol expanding (crisis), <1 = vol contracting (calm).

    Args:
        close: Daily close prices.
        short: Short-term vol window.
        long: Long-term vol window.

    Returns:
        Series of vol_short / vol_long ratios.
    """
    log_returns = np.log(close / close.shift(1))
    vol_short = log_returns.rolling(short).std()
    vol_long = log_returns.rolling(long).std()
    return vol_short / vol_long


# =============================================================================
# F5: HMM Regime Probabilities (3-state Gaussian HMM)
# =============================================================================

def hmm_regime_features(
    close: pd.Series,
    n_states: int = 3,
    train_window: int = 504,
    refit_every: int = 21,
) -> pd.DataFrame:
    """Rolling HMM regime probabilities.

    Uses ROLLING window (not expanding) to avoid lookahead.
    Re-fits every 21 days to limit computation.
    Outputs probabilities (not hard states) to avoid label switching.
    States sorted by volatility: 0=calm, 1=normal, 2=crisis.

    Args:
        close: Daily close prices (DatetimeIndex).
        n_states: Number of HMM states (fixed at 3).
        train_window: Rolling training window in days.
        refit_every: Re-fit interval in trading days.

    Returns:
        DataFrame with columns:
          hmm_prob_calm:   P(state=low_vol | data up to T)
          hmm_prob_crisis: P(state=high_vol | data up to T)
          hmm_entropy:     -sum(p * log(p)) -- regime uncertainty
    """
    import warnings
    from hmmlearn.hmm import GaussianHMM

    log_returns = np.log(close / close.shift(1)).dropna()

    result = pd.DataFrame(
        index=log_returns.index,
        columns=["hmm_prob_calm", "hmm_prob_crisis", "hmm_entropy"],
        dtype=float,
    )

    refit_indices = set(range(train_window, len(log_returns), refit_every))
    current_model: Optional[GaussianHMM] = None
    current_vol_order: Optional[np.ndarray] = None

    for i in range(train_window, len(log_returns)):
        # Re-fit on schedule
        if i in refit_indices or current_model is None:
            train_data = log_returns.iloc[i - train_window : i].values.reshape(-1, 1)

            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_data)

                if (np.any(np.isnan(model.means_))
                        or np.any(np.isnan(model.covars_))):
                    continue  # keep current_model

                # DON'T reorder model internals (breaks hmmlearn validation).
                # Instead, store the volatility sort order for output mapping.
                current_vol_order = np.argsort(model.covars_.flatten()[:n_states])
                current_model = model

            except (ValueError, np.linalg.LinAlgError):
                continue  # keep current_model

        if current_model is None:
            continue

        # Predict using last 60 days of context
        context_start = max(0, i - 60)
        context = log_returns.iloc[context_start : i + 1].values.reshape(-1, 1)

        try:
            raw_probs = current_model.predict_proba(context)[-1]
        except (ValueError, np.linalg.LinAlgError):
            continue

        # Map raw probs to sorted states (0=calm, -1=crisis)
        sorted_probs = raw_probs[current_vol_order]

        date = log_returns.index[i]
        result.loc[date, "hmm_prob_calm"] = sorted_probs[0]
        result.loc[date, "hmm_prob_crisis"] = sorted_probs[-1]

        entropy = float(-np.sum(raw_probs * np.log(raw_probs + 1e-10)))
        result.loc[date, "hmm_entropy"] = entropy

    return result


# =============================================================================
# TREATMENT GROUP BUILDER
# =============================================================================

TREATMENT_GROUPS = {
    "CONTROL": [],
    "TREAT-A": ["F1"],
    "TREAT-B": ["F1", "F2"],
    "TREAT-C": ["F1", "F2", "F3"],
    "TREAT-D": ["F1", "F2", "F3", "F4"],
    "TREAT-E": ["F1", "F4"],
    "TREAT-F": ["F5"],
    "TREAT-G": ["F1", "F4", "F5"],
}

# Column names produced by each feature
FEATURE_COLUMNS = {
    "F1": ["trend_slope_60d"],
    "F2": ["range_ratio_20d"],
    "F3": ["return_20d_sign"],
    "F4": ["vol_regime_ratio"],
    "F5": ["hmm_prob_calm", "hmm_prob_crisis", "hmm_entropy"],
}


def build_regime_features(
    df: pd.DataFrame, group: str, hmm_cache: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Add regime features to a DataFrame based on treatment group.

    Args:
        df: DataFrame with at least 'close', 'high', 'low' columns.
        group: Treatment group name (e.g., "TREAT-A", "CONTROL").
        hmm_cache: Pre-computed HMM features (to avoid redundant fits).

    Returns:
        DataFrame with additional regime feature columns.
    """
    if group == "CONTROL":
        return df

    features = TREATMENT_GROUPS.get(group, [])
    df = df.copy()

    for feat in features:
        if feat == "F1":
            df["trend_slope_60d"] = trend_slope_60d(df["close"])
        elif feat == "F2":
            df["range_ratio_20d"] = range_ratio_20d(df["high"], df["low"], df["close"])
        elif feat == "F3":
            df["return_20d_sign"] = return_20d_sign(df["close"])
        elif feat == "F4":
            df["vol_regime_ratio"] = vol_regime_ratio(df["close"])
        elif feat == "F5":
            if hmm_cache is not None:
                for col in FEATURE_COLUMNS["F5"]:
                    df[col] = hmm_cache[col]
            else:
                hmm_df = hmm_regime_features(df["close"])
                for col in FEATURE_COLUMNS["F5"]:
                    df[col] = hmm_df[col]

    return df


def get_feature_columns(group: str) -> list:
    """Get the list of additional feature column names for a treatment group."""
    features = TREATMENT_GROUPS.get(group, [])
    cols = []
    for feat in features:
        cols.extend(FEATURE_COLUMNS[feat])
    return cols
