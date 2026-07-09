"""Spot-exposure risk layer + baselines (design/SPEC-06 engine S3 + SPEC-10 baselines B1/B2).

Separation of concerns: a strategy decides an INTENT ∈ {0,1} (be in the market or not); the
deterministic risk layer decides the SPOT EXPOSURE SIZE ∈ [0,1] via vol-targeting × regime
multiplier. Effective exposure = intent × size, always CAUSAL (decided on info up to t-1).

Spot-only invariant (design §8): exposure ∈ [0, 1] on EVERY bar — never short, never levered.
"""
from __future__ import annotations

import pandas as pd

from .indicators import DISTRIBUTION, MARKDOWN

# PRE-REGISTRATION §2: σ_target 30% annualized; realized-vol floor keeps the sizer sane when BTC
# is unusually calm (avoids taking full size into a low-vol top); spot cap = 1.0 (no leverage).
TARGET_VOL = 0.30
VOL_FLOOR = 0.30       # BTC realized vol is typically 40-100%; floor caps exposure sanely
MAX_EXPOSURE = 1.0     # spot-only
MIN_EXPOSURE = 0.0


# --------------------------------------------------------------------------- risk layer (SPEC-06)
def vol_target_size(df: pd.DataFrame, *, target_vol: float = TARGET_VOL,
                    max_exposure: float = MAX_EXPOSURE) -> pd.Series:
    """Deterministic vol-targeting: exposure = target_vol / realized_vol, × regime, clipped [0,1].

    When BTC gets violent (realized_vol up), exposure shrinks automatically — the whole edge of a
    spot overlay (design §6). The vol FLOOR stops division blowups in calm regimes; the 1.0 cap is
    the spot-only ceiling (you cannot hold more than 100% of NAV in spot).
    """
    rv = df["realized_vol_20"].clip(lower=VOL_FLOOR)
    base = target_vol / rv
    size = base * df.get("regime_risk_mult", 1.0)
    return size.clip(lower=MIN_EXPOSURE, upper=max_exposure)


# --------------------------------------------------------------------------- intent signals {0,1}
def intent_hodl(df: pd.DataFrame) -> pd.Series:
    """B1: always in the market (HODL). The size layer still vol-targets it."""
    return pd.Series(1.0, index=df.index)


def intent_trend(df: pd.DataFrame) -> pd.Series:
    """B2: in when close>SMA100 AND ADX>25 (trend-follower), else flat."""
    long = (df["close"] > df["sma_100"]) & (df["adx_14"] > 25)
    return long.astype(float)


def intent_regime_gated(df: pd.DataFrame) -> pd.Series:
    """Regime-gated secular trend rider (the candidate strategy S3-lite):

    Rides the secular uptrend (close vs SMA200 — low churn) and sits out the regimes where the
    long edge dies: flat in MARKDOWN (bear/capitulation) and DISTRIBUTION (euphoric top). The size
    layer scales exposure down further via the regime multiplier. Spot-only: no shorts — BTC's
    long-run edge is up, and the win is AVOIDING the drawdowns, not fading them.
    """
    trend_up = (df["close"] > df["sma_200"]).astype(float)
    reg = df["regime"]
    out = trend_up.where(~reg.isin([MARKDOWN, DISTRIBUTION]), 0.0)
    return out.fillna(0.0).astype(float)


STRATEGIES = {
    "btc_hodl_b1": ("Bitcoin · HODL vol-targeted (B1)", intent_hodl, "rule_based"),
    "btc_trend_b2": ("Bitcoin · Trend-follower Daily (B2)", intent_trend, "rule_based"),
    "btc_exposure_s3": ("Bitcoin · Regime-gated exposure (S3)", intent_regime_gated, "hybrid"),
}


def build_positions(df: pd.DataFrame, intent_fn, *, target_vol: float = TARGET_VOL,
                    max_exposure: float = MAX_EXPOSURE) -> pd.DataFrame:
    """Return df with columns intent, size, position (spot exposure ∈ [0,1], causal shift(1))."""
    out = df.copy()
    out["intent"] = intent_fn(out).fillna(0.0)
    out["size"] = vol_target_size(out, target_vol=target_vol, max_exposure=max_exposure).fillna(0.0)
    raw = (out["intent"] * out["size"]).clip(lower=MIN_EXPOSURE, upper=max_exposure).fillna(0.0)
    # CAUSAL: the exposure held on day t was decided using info available at t-1.
    out["position"] = raw.shift(1).fillna(0.0)
    return out


# ── S4: trend × funding brake (OLA 5 B2 / H-POS-01) ─────────────────────────────
# Prior declarado ex-ante (PRE-REGISTRATION §5 style): el freno SOLO reduce exposición
# cuando los longs están crowded (z_funding alto); nunca amplifica. k y floor son priors
# económicos fijos — NO optimizados sobre el test (quant-constitution §1).
FUNDING_BRAKE_K = 0.25      # 25% de recorte por sigma de crowding por encima de 0
FUNDING_BRAKE_FLOOR = 0.4   # nunca recortar por debajo del 40% del size base


def intent_trend_funding(df: pd.DataFrame) -> pd.Series:
    """S4 = B2 trend intent × funding positioning brake (asymmetric, only reduces)."""
    base = intent_trend(df)
    z = df.get("z_funding")
    if z is None:
        return base
    brake = (1.0 - FUNDING_BRAKE_K * z.clip(lower=0.0)).clip(lower=FUNDING_BRAKE_FLOOR, upper=1.0)
    return (base * brake).astype(float)


STRATEGIES["btc_trend_funding_s4"] = (
    "Bitcoin · Trend × funding brake (S4)", intent_trend_funding, "hybrid")

# ── H-BTC-VOLBRK-01 (paso 2 del orden de construcción, pre-registrado 2026-07-07) ──
# PRIOR EX-ANTE (declarado antes de correr, nunca ajustado sobre el OOS): un spike de
# vol realizada (z>2.0 vs su propia media/std 252d) marca régimen de crash/liquidación
# donde la continuación de tendencia muere → el breaker CORTA exposición a la mitad.
# Asimétrico: solo frena, nunca amplifica. Sensibilidades pre-registradas z∈{1.5,2.5}
# se REPORTAN (cada celda = 1 trial) — elegir la mejor está prohibido (constitución §1).
VOLBRK_Z = 2.0        # prior económico: 2σ = evento de cola
VOLBRK_CUT = 0.5      # corta a la mitad (no flat: whipsaw cost)


def intent_trend_volbrk(df: pd.DataFrame) -> pd.Series:
    """B2 × vol-spike breaker: intención B2, recortada 0.5× cuando rvol z-score>2 (causal)."""
    base = intent_trend(df)
    rv = df["realized_vol_20"]
    mu = rv.rolling(252, min_periods=60).mean()
    sd = rv.rolling(252, min_periods=60).std()
    z = ((rv - mu) / sd).shift(1)  # causal: la decisión usa la vol de AYER
    return base.where(~(z > VOLBRK_Z), base * VOLBRK_CUT).fillna(0.0)


STRATEGIES["btc_trend_volbrk_s5"] = (
    "Bitcoin · Trend + vol-spike breaker (S5)", intent_trend_volbrk, "rule_based")
