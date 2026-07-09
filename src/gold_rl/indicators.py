"""SPEC-03 features + SPEC-04 regime classifier for Gold (daily).

All features are CAUSAL (no look-ahead): every value at row t uses only data <= t. Wilder's EMA
for ATR/ADX (never pandas ewm on the raw). The regime classifier is rule-based v1 with hysteresis
(min dwell) so labels are stable — the classic HMM upgrade is deferred (STRATEGY §2.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 4 regimes (STRATEGY §2.1)
COMPRESSION = "compression"
TREND = "trend"
STRETCHED = "stretched"
EVENT = "event"
REGIMES = (COMPRESSION, TREND, STRETCHED, EVENT)

# risk multiplier per regime (deterministic sizing input, STRATEGY table)
REGIME_RISK_MULT = {TREND: 1.0, COMPRESSION: 1.0, STRETCHED: 0.6, EVENT: 0.35}


# --------------------------------------------------------------------------- Wilder indicators
def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    # Wilder's smoothing = EMA with alpha = 1/period
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def wilder_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = wilder_atr(high, low, close, period)
    a = 1 / period
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean()


def hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst exponent for a 1-D window. Noisy on short windows — treat as a slow,
    smoothed feature, never a binary switch (STRATEGY caveat)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20 or np.allclose(x.std(), 0):
        return np.nan
    lags = range(2, min(20, n // 2))
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.std(diff)) if diff.size else np.nan)
    tau = np.array(tau)
    lags_arr = np.array(list(lags))
    ok = np.isfinite(tau) & (tau > 0)
    if ok.sum() < 3:
        return np.nan
    # slope of log(tau) vs log(lag) ~ Hurst (this variance-of-diffs estimator ~ 2*H convention)
    poly = np.polyfit(np.log(lags_arr[ok]), np.log(tau[ok]), 1)
    return float(poly[0] * 2.0)


# --------------------------------------------------------------------------- feature build
def build_daily_features(df: pd.DataFrame, *, hurst_window: int = 100) -> pd.DataFrame:
    """Add causal daily features. Input must have [time, open, high, low, close] sorted by time."""
    d = df.sort_values("time").reset_index(drop=True).copy()
    c = d["close"]
    d["log_ret"] = np.log(c / c.shift(1))
    for w in (20, 50, 100, 200):
        d[f"sma_{w}"] = c.rolling(w, min_periods=w).mean()
    d["atr_14"] = wilder_atr(d["high"], d["low"], c, 14)
    d["atr_pct"] = d["atr_14"] / c
    d["adx_14"] = wilder_adx(d["high"], d["low"], c, 14)
    d["realized_vol_20"] = d["log_ret"].rolling(20, min_periods=20).std() * np.sqrt(252)
    d["z_sma50"] = (c - d["sma_50"]) / c.rolling(50, min_periods=50).std()
    # rolling Hurst (slow feature). Compute on log-price.
    logp = np.log(c)
    d["hurst"] = logp.rolling(hurst_window, min_periods=hurst_window).apply(
        lambda w: hurst_rs(w.values), raw=False
    )
    d["hurst_smooth"] = d["hurst"].rolling(10, min_periods=3).mean()
    return d


# --------------------------------------------------------------------------- regime classifier
def classify_regime(
    df: pd.DataFrame,
    *,
    dwell: int = 4,
    event_flags: pd.Series | None = None,
    hurst_trending: float | None = None,
    hurst_mean_rev: float | None = None,
) -> pd.DataFrame:
    """Rule-based 4-regime classifier with hysteresis (min dwell days). Causal.

    - EVENT: macro high-impact flag active (if provided).
    - TREND: ADX high AND Hurst >= hurst_trending (persistent).
    - STRETCHED: |z_sma50| extreme AND Hurst < hurst_mean_rev (mean-reverting).
    - COMPRESSION: low vol / low ADX (range) — the default/breakout-watch state.

    Hurst pivots (audit A10-01): pass the ASSET's fitted thresholds from its
    AssetProfile (`regime_gate.hurst_trending` / `hurst_mean_rev`). When None
    (not yet fitted — e.g. xauusd.yaml ships nulls) an EXPLICIT 0.5 pivot is
    used and logged: an honest neutral prior, NOT the COP values (0.52/0.42).
    Fitting per-asset thresholds is a registered trial (test D1), never done
    silently here.
    """
    if hurst_trending is None or hurst_mean_rev is None:
        import logging
        logging.getLogger(__name__).info(
            "classify_regime: hurst thresholds not fitted for this asset — "
            "using explicit neutral 0.5 pivot (A10-01/02; fit via test D1 to override)")
    _h_trend = 0.5 if hurst_trending is None else float(hurst_trending)
    _h_mrev = 0.5 if hurst_mean_rev is None else float(hurst_mean_rev)
    d = df.copy()
    vol = d["realized_vol_20"]
    vol_lo = vol.rolling(252, min_periods=60).quantile(0.35)
    adx = d["adx_14"]
    hurst = d["hurst_smooth"]
    z = d["z_sma50"].abs()

    raw = pd.Series(index=d.index, dtype=object)
    for i in range(len(d)):
        if event_flags is not None and bool(event_flags.iloc[i]):
            raw.iloc[i] = EVENT
            continue
        a, h, zz, v, vlo = adx.iloc[i], hurst.iloc[i], z.iloc[i], vol.iloc[i], vol_lo.iloc[i]
        if pd.isna(a) or pd.isna(h):
            raw.iloc[i] = COMPRESSION
        elif a >= 25 and h >= _h_trend:
            raw.iloc[i] = TREND
        elif zz >= 2.0 and h < _h_mrev:
            raw.iloc[i] = STRETCHED
        elif not pd.isna(vlo) and v <= vlo and a < 20:
            raw.iloc[i] = COMPRESSION
        else:
            raw.iloc[i] = COMPRESSION
    # hysteresis: require `dwell` consecutive days of a new label before switching
    stable = raw.copy()
    cur = raw.iloc[0]
    run = 0
    for i in range(len(raw)):
        if raw.iloc[i] == cur:
            run = 0
        else:
            run += 1
            if run >= dwell:
                cur = raw.iloc[i]
                run = 0
            else:
                stable.iloc[i] = cur
        stable.iloc[i] = cur if stable.iloc[i] != cur and run < dwell else stable.iloc[i]
        stable.iloc[i] = cur
    d["regime"] = stable
    d["regime_risk_mult"] = d["regime"].map(REGIME_RISK_MULT).astype(float)
    return d


def regime_transitions_per_year(d: pd.DataFrame) -> float:
    """Label stability metric — Gold regimes last weeks, so transitions/year should be low."""
    reg = d["regime"].dropna()
    if reg.empty:
        return 0.0
    changes = int((reg != reg.shift(1)).sum())
    years = max((d["time"].max() - d["time"].min()).days / 365.25, 1e-9)
    return round(changes / years, 2)
