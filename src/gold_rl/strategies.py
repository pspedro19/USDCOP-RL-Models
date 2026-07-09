"""SPEC-06 risk layer (vol-targeting) + SPEC-07 baselines + regime-gated strategy.

Separation of concerns (STRATEGY §1): a strategy decides DIRECTION {-1,0,+1}; the deterministic
risk layer decides SIZE (vol-targeting + regime multiplier + leverage cap). The effective daily
position = direction * size, always evaluated CAUSALLY (decided on info up to t-1, applied to t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import EVENT


# --------------------------------------------------------------------------- risk layer (SPEC-06)
def vol_target_size(df: pd.DataFrame, *, target_vol: float = 0.10, max_leverage: float = 1.5,
                    min_leverage: float = 0.0) -> pd.Series:
    """Deterministic vol-targeting: size = target_vol / realized_vol, scaled by regime, capped.

    When gold gets violent (realized_vol up), size shrinks automatically — this is where the
    alpha lives (STRATEGY §4). The vol FLOOR (6%) and a conservative 1.5x leverage cap stop the
    sizer from taking 3x in ultra-calm periods (which produced a 52% drawdown when tuned loosely).
    """
    rv = df["realized_vol_20"].clip(lower=0.06)  # floor 6% annualized -> caps leverage sanely
    base = (target_vol / rv)
    size = base * df.get("regime_risk_mult", 1.0)
    return size.clip(lower=min_leverage, upper=max_leverage)


# --------------------------------------------------------------------------- direction signals
def direction_long_only(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def direction_trend_follower(df: pd.DataFrame) -> pd.Series:
    """B2: long when close>SMA100 AND ADX>25, else flat (honest simple system)."""
    long = (df["close"] > df["sma_100"]) & (df["adx_14"] > 25)
    return long.astype(float)  # 1.0 / 0.0


def direction_regime_gated(df: pd.DataFrame) -> pd.Series:
    """Regime-gated SECULAR trend rider (the candidate 'strategy'):

    Rides the long-term trend (close vs SMA200 — flips far less than SMA100, so low churn) and
    sits out the regimes where the trend edge dies:
      - long when close > SMA200 (secular uptrend), flat otherwise (no shorts: gold's edge is up),
      - flat entirely in EVENT (blackout) and STRETCHED (mean-reverting chop kills trend-riding).
    The risk layer scales size down in COMPRESSION/high-vol; direction stays low-turnover.
    """
    trend_up = (df["close"] > df["sma_200"]).astype(float)  # 1.0 / 0.0, low churn
    reg = df["regime"]
    d = trend_up.where(~reg.isin([EVENT, "stretched"]), 0.0)
    return d.fillna(0.0).astype(float)


STRATEGIES = {
    "gold_long_only_b1": ("Gold · Long-only vol-targeted (B1)", direction_long_only, "rule_based"),
    "gold_trend_b2": ("Gold · Trend-follower Daily (B2)", direction_trend_follower, "rule_based"),
    "gold_regime_gated_v1": ("Gold · Regime-gated vol-targeted", direction_regime_gated, "hybrid"),
}


def build_positions(df: pd.DataFrame, direction_fn, *, target_vol: float = 0.10,
                    max_leverage: float = 1.5) -> pd.DataFrame:
    """Return df with columns direction, size, position (all causal: shift(1) before applying)."""
    out = df.copy()
    out["direction"] = direction_fn(out).fillna(0.0)
    out["size"] = vol_target_size(out, target_vol=target_vol, max_leverage=max_leverage).fillna(0.0)
    raw_pos = (out["direction"] * out["size"]).fillna(0.0)
    # CAUSAL: the position held on day t was decided using info available at t-1.
    out["position"] = raw_pos.shift(1).fillna(0.0)
    return out


# ── XAU-TREND-ENS (OLA 6 X1 / H-XAU-TREND-01): multi-lookback ensemble, ex-ante ──
# Prior CTA estándar declarado sin optimizar: votos de 3/6/12 meses (63/126/252d),
# long-flat (drift secular + carry negativo del short salvo unanimidad — no implementado
# el short: long-flat puro, la variante preferida a priori).
def direction_trend_ensemble(df: pd.DataFrame) -> pd.Series:
    votes = sum((df["close"] > df["close"].rolling(w).mean()).astype(float)
                for w in (63, 126, 252)) / 3.0
    return votes.clip(lower=0.0)  # ∈ {0, 1/3, 2/3, 1} — posición en tercios


STRATEGIES["gold_trend_ens"] = (
    "Gold · Trend ensemble 3/6/12m (ENS)", direction_trend_ensemble, "rule_based")

# ── H-XAU-DXY-01 (pre-registrado 2026-07-07, prior ex-ante congelado) ──
# Tilt de EXPOSICIÓN sobre B1 (no timing): cuando el DÓLAR sube fuerte (z del retorno
# 20d del DXY > +1σ sobre su propia ventana 252d), recorta exposición al CUT; si no, 1.0.
# Asimétrico: solo frena en dólar fuerte (el denominador encarece el oro), nunca amplifica.
# Prior: CUT=0.6. Sensibilidades pre-registradas {0.5, 0.7}: se reportan TODAS las celdas
# (cada una = 1 trial); elegir la mejor está prohibido (constitución §1).
# Bar de adopción: Calmar > 0.223 (B1 vol-targeted = B1' de Oro) en la ventana con DXY.

def _dxy_tilt(df, cut: float):
    base = pd.Series(1.0, index=df.index)
    z = df.get("dxy_z20")
    if z is None:
        return base  # sin DXY (pre-2020 o seed sin merge): idéntico a B1
    return base.where(~(z > 1.0), cut).fillna(1.0)


def direction_dxy_tilt(df: pd.DataFrame) -> pd.Series:
    return _dxy_tilt(df, 0.6)


def direction_dxy_tilt_s05(df: pd.DataFrame) -> pd.Series:
    return _dxy_tilt(df, 0.5)


def direction_dxy_tilt_s07(df: pd.DataFrame) -> pd.Series:
    return _dxy_tilt(df, 0.7)


STRATEGIES["gold_dxy_tilt"] = ("Gold · B1 + tilt DXY (prior 0.6)", direction_dxy_tilt, "rule_based")
STRATEGIES["gold_dxy_tilt_s05"] = ("Gold · tilt DXY sens 0.5", direction_dxy_tilt_s05, "rule_based")
STRATEGIES["gold_dxy_tilt_s07"] = ("Gold · tilt DXY sens 0.7", direction_dxy_tilt_s07, "rule_based")
