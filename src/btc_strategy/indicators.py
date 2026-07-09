"""Causal daily features + 4-regime classifier for BTC (design/SPEC-01, adapted).

All features are CAUSAL (row t uses only data <= t). Wilder's EMA for ATR/ADX. Realized vol is
annualized with **√365** because BTC trades 24/7 (PRE-REGISTRATION §2). The rule-based regime
classifier with hysteresis stands in for the frozen-fit HMM of design/SPEC-01 until on-chain data
is ingested (migration 052) — same 4 economic states, stable labels, low churn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANN = np.sqrt(365.0)  # crypto 24/7 annualization

# 4 regimes (design/SPEC-01 cycle states, mapped to price-only proxies until on-chain lands)
ACCUMULATION = "accumulation"   # low vol / basing (COMPRESSION analogue)
MARKUP = "markup"               # trending up (TREND analogue)
DISTRIBUTION = "distribution"   # stretched / euphoric (STRETCHED analogue)
MARKDOWN = "markdown"           # trending down / capitulation (EVENT/bear analogue)
REGIMES = (ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN)

# risk multiplier per regime (deterministic sizing input). Spot-only: never > 1.0.
REGIME_RISK_MULT = {MARKUP: 1.0, ACCUMULATION: 0.8, DISTRIBUTION: 0.5, MARKDOWN: 0.35}


# --------------------------------------------------------------------------- Wilder indicators
def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
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
    """Rescaled-range Hurst for a 1-D window. Noisy on short windows — a slow feature, not a switch."""
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
    d["realized_vol_20"] = d["log_ret"].rolling(20, min_periods=20).std() * ANN
    d["z_sma50"] = (c - d["sma_50"]) / c.rolling(50, min_periods=50).std()
    logp = np.log(c)
    d["hurst"] = logp.rolling(hurst_window, min_periods=hurst_window).apply(
        lambda w: hurst_rs(w.values), raw=False)
    d["hurst_smooth"] = d["hurst"].rolling(10, min_periods=3).mean()
    return d


# --------------------------------------------------------------------------- regime classifier
def classify_regime(df: pd.DataFrame, *, dwell: int = 5) -> pd.DataFrame:
    """Rule-based 4-regime classifier with hysteresis (min dwell days). Causal.

    - MARKUP: strong uptrend (ADX high, Hurst persistent, price>SMA200).
    - DISTRIBUTION: overextended above trend (|z_sma50| high) while Hurst rolling over.
    - MARKDOWN: price below SMA200 in a persistent downtrend (bear/capitulation).
    - ACCUMULATION: low-vol basing (the default / breakout-watch state).
    Dwell (min consecutive days) prevents label churn — BTC cycle regimes last weeks/months.
    """
    d = df.copy()
    adx = d["adx_14"]
    hurst = d["hurst_smooth"]
    z = d["z_sma50"]
    above200 = d["close"] > d["sma_200"]

    raw = pd.Series(index=d.index, dtype=object)
    for i in range(len(d)):
        a, h, zz, up = adx.iloc[i], hurst.iloc[i], z.iloc[i], bool(above200.iloc[i])
        if pd.isna(a) or pd.isna(h):
            raw.iloc[i] = ACCUMULATION
        elif not up and a >= 20:
            raw.iloc[i] = MARKDOWN
        elif up and a >= 25 and h >= 0.5:
            raw.iloc[i] = MARKUP
        elif up and abs(zz) >= 2.0 and h < 0.5:
            raw.iloc[i] = DISTRIBUTION
        else:
            raw.iloc[i] = ACCUMULATION
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
        stable.iloc[i] = cur
    d["regime"] = stable
    d["regime_risk_mult"] = d["regime"].map(REGIME_RISK_MULT).astype(float)
    return d


def regime_transitions_per_year(d: pd.DataFrame) -> float:
    reg = d["regime"].dropna()
    if reg.empty:
        return 0.0
    changes = int((reg != reg.shift(1)).sum())
    years = max((d["time"].max() - d["time"].min()).days / 365.25, 1e-9)
    return round(changes / years, 2)


def merge_funding_features(df: pd.DataFrame, *, z_window: int = 30) -> pd.DataFrame:
    """Merge Binance perp funding (crypto_derivatives_daily seed) as CAUSAL features (OLA 5 B1).

    Anti-leakage: funding for day D settles during D -> shift(1) so day t only sees funding
    through t-1 (macro T-1 discipline). Graceful: missing seed => z_funding = 0.0 (strategy
    degrades to price-only, same behavior as today).
    """
    seed = Path(__file__).resolve().parents[2] / "seeds/latest/btcusdt_derivatives_daily.parquet"
    out = df.copy()
    try:
        der = pd.read_parquet(seed)[["date", "funding_rate"]]
        der["date"] = pd.to_datetime(der["date"])
        der = der.sort_values("date").reset_index(drop=True)
        # recompute z on the merged calendar (don't trust stored z blindly), then shift(1)
        mu = der["funding_rate"].rolling(z_window, min_periods=10).mean()
        sd = der["funding_rate"].rolling(z_window, min_periods=10).std(ddof=0)
        der["z_funding"] = ((der["funding_rate"] - mu) / sd.replace(0, np.nan)).fillna(0.0)
        left = out.copy()
        left["_d"] = pd.to_datetime(left["time"]).dt.tz_localize(None).dt.normalize()
        der["_d"] = der["date"].dt.tz_localize(None).dt.normalize() if der["date"].dt.tz is not None else der["date"].dt.normalize()
        merged = pd.merge_asof(left.sort_values("_d"), der[["_d", "funding_rate", "z_funding"]],
                               on="_d", direction="backward")
        out["funding_rate"] = merged["funding_rate"].shift(1).fillna(0.0).values
        out["z_funding"] = merged["z_funding"].shift(1).fillna(0.0).values
    except Exception:
        out["funding_rate"] = 0.0
        out["z_funding"] = 0.0
    return out
