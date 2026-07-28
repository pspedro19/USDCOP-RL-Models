"""Motor de backtest con contrato de ejecución next-open (SDD-006 §1 y §6).

El ÚNICO punto donde puede entrar look-ahead de ejecución es el `shift(1)` de los
pesos: la señal calculada con info hasta el cierre de t se ejecuta en la apertura
de t+1. Ese shift está aislado en UNA línea (`_lag_weights`) y testeado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from costs import CostModel

__all__ = ["BacktestConfig", "BacktestResult", "BacktestEngine"]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    execution: Literal["next_open"] = "next_open"   # único valor permitido
    cost_bps_roundtrip: float = 2.0
    max_leverage: float = 1.5
    allow_short: bool = False


@dataclass(frozen=True, slots=True)
class BacktestResult:
    returns_net: pd.Series      # retorno de la estrategia neto de costos
    returns_gross: pd.Series
    turnover: pd.Series
    weights_exec: pd.Series     # peso efectivamente en riesgo cada día (ya rezagado)
    cost: pd.Series


def _lag_weights(weights: pd.Series) -> pd.Series:
    """El shift crítico. Rezagado EXACTAMENTE una vez (SDD-006 §6)."""
    return weights.shift(1).fillna(0.0)


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig = BacktestConfig()):
        self.cfg = cfg
        self.costs = CostModel(cfg.cost_bps_roundtrip)

    def run(self, weights: pd.Series, market: pd.DataFrame) -> BacktestResult:
        """`weights` indexado por t = fecha de DECISIÓN (cierre de t)."""
        lo = 0.0 if not self.cfg.allow_short else -self.cfg.max_leverage
        w = weights.clip(lower=lo, upper=self.cfg.max_leverage)

        # Turnover sobre los pesos de DECISIÓN: peso constante ⇒ diff 0 ⇒ costo 0
        # (invariante SDD-006). El primer día no cobra establecimiento.
        turnover_dec = w.diff().abs().fillna(0.0)

        w_exec = _lag_weights(w)                     # ← el único shift, aislado y testeado
        ret = market["open_to_open_return"].astype(float)
        w_exec, ret = w_exec.align(ret, join="inner")

        # El costo se realiza en la ejecución (mismo rezago que los pesos).
        turnover = turnover_dec.shift(1).reindex(ret.index).fillna(0.0)
        cost = self.costs.apply(turnover)
        gross = w_exec * ret
        net = gross - cost

        return BacktestResult(
            returns_net=net.rename("strategy"),
            returns_gross=gross.rename("gross"),
            turnover=turnover.rename("turnover"),
            weights_exec=w_exec.rename("w_exec"),
            cost=cost.rename("cost"),
        )
