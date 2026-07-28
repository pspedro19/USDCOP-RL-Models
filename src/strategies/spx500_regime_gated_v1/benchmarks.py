"""Los cuatro benchmarks obligatorios (SDD-006 §3).

Toda tabla los reporta SIEMPRE. Reportar sólo contra SPY_TR está prohibido: es el
más fácil de batir en Sharpe. Cada benchmark devuelve pesos en fecha de decisión.

Nota de honestidad: SIXTY_FORTY debería ser 60% SPY / 40% IEF con rebalanceo
mensual. La demo sintética no tiene serie de bonos, así que aquí es una exposición
constante de 0.6 (proxy) — documentado, no escondido. Con datos reales se enchufa IEF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["BENCHMARKS", "build_benchmark", "realized_vol", "vol_target_weights"]

TARGET_VOL = 0.10
MAX_LEV = 1.5


def realized_vol(close: pd.Series, window: int = 22) -> pd.Series:
    """Vol anualizada de los retornos diarios (PIT: sólo pasado)."""
    daily = close.pct_change()
    return daily.rolling(window, min_periods=window).std() * np.sqrt(252)


def vol_target_weights(
    close: pd.Series, target: float = TARGET_VOL, cap: float = MAX_LEV
) -> pd.Series:
    rv = realized_vol(close)
    w = (target / rv).clip(upper=cap)
    return w.fillna(0.0).rename("VOL_TARGET_10")


def _spy_tr(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, name="SPY_TR")


def _sixty_forty(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.6, index=df.index, name="SIXTY_FORTY")


def _ma200(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    ma = close.rolling(200, min_periods=200).mean()
    return (close > ma).astype(float).rename("MA200")


def _vol_target(df: pd.DataFrame) -> pd.Series:
    return vol_target_weights(df["close"].astype(float))


BENCHMARKS: dict[str, callable] = {
    "SPY_TR": _spy_tr,
    "SIXTY_FORTY": _sixty_forty,
    "VOL_TARGET_10": _vol_target,
    "MA200": _ma200,
}


def build_benchmark(name: str, df: pd.DataFrame) -> pd.Series:
    if name not in BENCHMARKS:
        raise KeyError(f"Benchmark desconocido: {name}. Válidos: {list(BENCHMARKS)}")
    return BENCHMARKS[name](df)
