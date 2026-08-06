"""Las tres políticas de exposición al SP500 — el patrón B1 / B2 / S3.

Se publican las TRES. La regla del repo (y del quant-constitution) es honesta:
si la gated (S3) no bate a B2 en Sharpe y Calmar bajo costos ×2 + DSR/PBO, entonces
B2 ES la estrategia. Pasó con BTC (S3 no supera a B2 y se documenta como tal).

    spx_hodl_b1        HODL con vol-targeting          H1  (baseline honesto)
    spx_trend_b2       Trend follower (MA200 + TSMOM)  H3  (el que suele ganar)
    spx_regime_gated_v1  Tendencia × techo de régimen  H2+H3 (la hipótesis)

Cada política devuelve PESOS en fecha de decisión t (info hasta el cierre de t).
El motor aplica el shift(1) → ejecución en la apertura de t+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__:
    from .benchmarks import MAX_LEV, TARGET_VOL, vol_target_weights
    from .regime import RegimeConfig, exposure_ceiling
else:  # soporte del runner standalone documentado
    from benchmarks import MAX_LEV, TARGET_VOL, vol_target_weights
    from regime import RegimeConfig, exposure_ceiling

__all__ = ["POLICIES", "STRATEGY_IDS", "spx_hodl_b1", "spx_trend_b2", "spx_regime_gated_v1"]

STRATEGY_IDS = ("spx_hodl_b1", "spx_trend_b2", "spx_regime_gated_v1")


def _tsmom_12_1(close: pd.Series) -> pd.Series:
    """Time-series momentum 12-1: signo del retorno de los últimos 252d saltando
    el último mes (Moskowitz-Ooi-Pedersen 2012). +1 alcista, 0 bajista."""
    mom = close.shift(21) / close.shift(252) - 1.0
    return (mom > 0).astype(float)


def _ma200_filter(close: pd.Series) -> pd.Series:
    ma = close.rolling(200, min_periods=200).mean()
    return (close > ma).astype(float)


# ----------------------------------------------------------------- B1 (H1)
def spx_hodl_b1(df: pd.DataFrame) -> pd.Series:
    """HODL con vol-targeting: la exposición honesta mínima. Es el listón real."""
    return vol_target_weights(df["close"].astype(float)).rename("spx_hodl_b1")


# ----------------------------------------------------------------- B2 (H3)
def spx_trend_b2(df: pd.DataFrame) -> pd.Series:
    """Trend follower: exposición vol-target SÓLO cuando MA200 y TSMOM 12-1 alinean.

    Neely et al. (2014): las reglas técnicas simples capturan la prima de tendencia
    del equity index. El vol-targeting evita que la vol dicte el tamaño de la apuesta.
    """
    close = df["close"].astype(float)
    trend_on = _ma200_filter(close) * _tsmom_12_1(close)     # AND de las dos reglas
    base = vol_target_weights(close)
    return (trend_on * base).clip(upper=MAX_LEV).rename("spx_trend_b2")


# ----------------------------------------------------------------- S3 (H2+H3)
@dataclass(frozen=True, slots=True)
class GatedConfig:
    regime: RegimeConfig = RegimeConfig()


def spx_regime_gated_v1(df: pd.DataFrame, cfg: GatedConfig = GatedConfig()) -> pd.Series:
    """La hipótesis: tendencia (B2) MODULADA por el techo de exposición de régimen.

    El overlay macro/VIX/breadth (analogía de exposure-coach) sólo RECORTA la
    exposición de la tendencia en estados de riesgo — nunca inventa dirección ni
    apalanca. Es una prueba de H2: ¿añadir el overlay macro-régimen mejora sobre H3?
    """
    trend = spx_trend_b2(df)
    ceiling = exposure_ceiling(df, cfg.regime)
    gated = (trend * ceiling).clip(lower=0.0, upper=MAX_LEV)
    return gated.rename("spx_regime_gated_v1")


POLICIES: dict[str, callable] = {
    "spx_hodl_b1": spx_hodl_b1,
    "spx_trend_b2": spx_trend_b2,
    "spx_regime_gated_v1": spx_regime_gated_v1,
}
