"""Dependency-light value objects shared by profitability analysis scripts.

Keep this module free of database, API, metrics-engine, and service imports.  Asset
adapters and small diagnostics must be able to construct a sleeve without importing
the entire production service graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Sleeve:
    """One strategy's realized series, on its own clock."""

    def __init__(
        self,
        asset: str,
        strategy_id: str,
        index,
        position,
        asset_ret,
        cost,
        swap,
        n_trades: int,
        clock: int,
        clock_label: str,
        dumb_name: str,
        dumb_position,
        invalid_baselines: tuple = (),
    ):
        self.asset = asset
        self.strategy_id = strategy_id
        self.index = pd.Index(index)
        self.position = np.asarray(position, dtype=float)
        self.asset_ret = np.asarray(asset_ret, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.swap = (
            np.asarray(swap, dtype=float)
            if swap is not None
            else np.zeros_like(self.cost)
        )
        self.n_trades = int(n_trades)
        self.clock = int(clock)
        self.clock_label = clock_label
        self.dumb_name = dumb_name
        self.dumb_position = np.asarray(dumb_position, dtype=float)
        # A comparison the data source cannot support is never a comparison it passed.
        self.invalid_baselines = tuple(invalid_baselines)

    @property
    def strat_ret(self) -> np.ndarray:
        """Net strategy return implied by the normalized sleeve fields."""
        return self.position * self.asset_ret - self.cost - self.swap


__all__ = ["Sleeve"]
