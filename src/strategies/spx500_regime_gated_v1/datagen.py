"""Generador de mercado SINTÉTICO tipo SPY (demo de cableado, NO evidencia de alfa).

⚠️  ESTO NO ES DATA REAL. Es una serie determinista con estructura de régimen
(Markov de 2 estados: calmo/estresado) para que el clasificador de régimen y las
políticas de exposición tengan algo con *forma* sobre lo que morder. Cualquier
métrica que salga de aquí mide el CABLEADO, no el mercado. Para evidencia real:

    df = load_real()   # ver README §"Enchufar datos reales"

Contrato de columnas (idéntico al que espera el motor de backtest, SDD-006):
    open_to_open_return : retorno realizado de open_t a open_{t+1}  (para PnL)
    close               : nivel de precio total-return (para señales)
    vix                 : volatilidad implícita sintética (estado de riesgo)
    macro_stress        : driver macro exógeno sintético (proxy NFCI/HY-OAS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate", "SYNTHETIC_WARNING"]

SYNTHETIC_WARNING = (
    "DATA SINTÉTICA — las métricas miden el cableado del pipeline, no alfa real."
)

# Estados latentes: 0 = calmo (drift +, vol baja), 1 = estrés (drift -, vol alta).
_MU = np.array([0.00045, -0.00060])      # drift diario por estado
_SD = np.array([0.0068, 0.0180])         # vol diaria por estado
# Matriz de transición persistente (los regímenes duran meses, no días).
_TRANS = np.array([[0.992, 0.008],
                   [0.030, 0.970]])


def generate(
    start: str = "1990-01-02",
    end: str = "2024-12-31",
    seed: int = 20260709,
) -> pd.DataFrame:
    """Serie diaria SPY-like con regímenes. Determinista dado `seed`."""
    idx = pd.bdate_range(start, end, tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(seed)

    # --- cadena de Markov de estados (PIT: el estado de t no mira el futuro) ---
    states = np.empty(n, dtype=int)
    states[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        # P(estado_t = 0 | estado_{t-1}) = _TRANS[estado_{t-1}, 0]
        states[t] = int(u[t] > _TRANS[states[t - 1], 0])

    # --- retornos diarios condicionados al estado ---
    daily = rng.normal(_MU[states], _SD[states])
    close = 100.0 * np.exp(np.cumsum(daily))

    # open_to_open ~ retorno diario con un pequeño gap de apertura correlacionado
    gap = rng.normal(0.0, 0.0015, n)
    o2o = daily + gap - np.r_[0.0, gap[:-1]]  # el gap se revierte al día siguiente

    # --- VIX sintético: vol realizada 22d reescalada + prima por estado + ruido ---
    rv22 = pd.Series(daily, index=idx).rolling(22).std().to_numpy() * np.sqrt(252)
    rv22 = np.nan_to_num(rv22, nan=np.nanmean(rv22[~np.isnan(rv22)]))
    vix = 100.0 * rv22 + 6.0 * states + rng.normal(0.0, 1.2, n)
    vix = np.clip(vix, 9.0, 85.0)

    # --- driver macro exógeno: estado suavizado que ADELANTA levemente al estrés ---
    stress_raw = pd.Series(states, dtype=float).rolling(20, min_periods=1).mean()
    macro = stress_raw.to_numpy() + rng.normal(0.0, 0.08, n)
    macro = np.clip(macro, 0.0, 1.0)

    return pd.DataFrame(
        {
            "open_to_open_return": o2o,
            "close": close,
            "vix": vix,
            "macro_stress": macro,
            "_state": states,  # solo para inspección; las políticas NO lo ven
        },
        index=idx,
    )
