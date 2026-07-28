"""Clasificador de régimen point-in-time (SDD-000 §6) + techo de exposición.

Diseño específico de SP500 (decisión registrada, no heredada de COP)
-------------------------------------------------------------------
El VIX de SPX es su PROPIA volatilidad implícita. Usarlo como feature *predictivo*
de la dirección del SPX roza la circularidad (McLean-Pontiff). Aquí el VIX entra
sólo como ETIQUETA DE ESTADO contemporáneo (high_vol / bull), nunca como alfa.
El sesgo direccional lo aportan la tendencia (MA200, TSMOM) y el driver macro
exógeno — coherente con la premisa del charter: gestionar exposición, no predecir.

Umbrales de régimen: NO se copian los de COP (0.52/0.42). Como en Gold, se parte
de un pivote neutral explícito y logueado; ajustarlos sería un trial registrado.

Todas las estadísticas son EXPANDING (usan sólo información hasta la fecha t).
El techo de exposición replica en pequeño la agregación de `exposure-coach`:
breadth/tendencia/vol → un único ceiling de capital en [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RegimeConfig", "label_regimes", "exposure_ceiling", "REGIMES"]

REGIMES = ("bull", "bear", "high_vol", "sideways")


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    trend_window: int = 252          # ventana de retorno trailing
    drawdown_bear: float = 0.20      # -20% desde máximo trailing = bear
    high_vol_pct: float = 0.80       # percentil expandido de VIX = high_vol
    sideways_band: float = 0.05      # |ret_252| < 5% = lateral
    ceiling_floor: float = 0.0       # exposición mínima en estrés
    warmup: int = 252                # días sin señal (NaN) al inicio


def _expanding_percentile_rank(s: pd.Series) -> pd.Series:
    """Rango percentil expandido de cada punto dentro de su propia historia (PIT)."""
    # rank(t) = fracción de observaciones hasta t que son <= valor de t.
    return s.expanding().apply(
        lambda w: float((w <= w[-1]).mean()), raw=True
    )


def label_regimes(df: pd.DataFrame, cfg: RegimeConfig = RegimeConfig()) -> pd.DataFrame:
    """Devuelve un DataFrame booleano con una columna por régimen (SDD-000 §6).

    Un mismo día puede pertenecer a más de un régimen (bull y high_vol coexisten).
    """
    close = df["close"].astype(float)
    vix = df["vix"].astype(float)

    ret_252 = close.pct_change(cfg.trend_window)
    roll_max = close.rolling(cfg.trend_window, min_periods=cfg.trend_window).max()
    drawdown = close / roll_max - 1.0

    vix_median = vix.expanding().median()
    vix_rank = _expanding_percentile_rank(vix)

    out = pd.DataFrame(index=df.index)
    out["bull"] = (ret_252 > 0) & (vix < vix_median)
    out["bear"] = drawdown < -cfg.drawdown_bear
    out["high_vol"] = vix_rank > cfg.high_vol_pct
    out["sideways"] = ret_252.abs() < cfg.sideways_band

    # Warm-up: sin 252 días de historia no hay etiqueta honesta.
    out.iloc[: cfg.warmup] = False
    return out


def exposure_ceiling(df: pd.DataFrame, cfg: RegimeConfig = RegimeConfig()) -> pd.Series:
    """Techo de exposición en [ceiling_floor, 1] (analogía de `exposure-coach`).

    Combina tres señales de estado, todas PIT:
      - tendencia   : precio sobre/bajo su MA200 (participación estructural)
      - riesgo VIX  : rango percentil expandido del VIX (más alto → menos techo)
      - macro       : driver exógeno de estrés (más alto → menos techo)
    El resultado es un ceiling, NO una señal direccional: recorta, nunca apalanca.
    """
    close = df["close"].astype(float)
    ma200 = close.rolling(200, min_periods=200).mean()
    trend_ok = (close > ma200).astype(float)                 # 1 arriba, 0 abajo

    vix_rank = _expanding_percentile_rank(df["vix"].astype(float))
    risk_haircut = np.clip(1.0 - (vix_rank - 0.5).clip(lower=0) * 2.0, 0.0, 1.0)

    macro = df["macro_stress"].astype(float).clip(0.0, 1.0)
    macro_haircut = 1.0 - macro                               # 1 sin estrés, 0 pánico

    ceiling = trend_ok * risk_haircut * macro_haircut
    ceiling = ceiling.clip(lower=cfg.ceiling_floor, upper=1.0)
    ceiling.iloc[: cfg.warmup] = 0.0
    return ceiling.rename("exposure_ceiling")
