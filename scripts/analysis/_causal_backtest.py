"""Small shared split primitives for research backtests.

The functions here operate on positional daily rows.  A label at row ``i`` for
horizon ``h`` matures at row ``i + h``.  For feature selection frozen before an
OOS cutoff, that maturity row must be strictly earlier than the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def matured_label_indices_before(
    dates: pd.Series | pd.Index | np.ndarray,
    *,
    horizon: int,
    cutoff: str | pd.Timestamp,
) -> np.ndarray:
    """Return rows whose full forward label matures strictly before ``cutoff``."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive row count")
    ordered = pd.DatetimeIndex(pd.to_datetime(dates))
    if not ordered.is_monotonic_increasing:
        raise ValueError("dates must be sorted before constructing a causal split")
    cutoff_idx = int(ordered.searchsorted(pd.Timestamp(cutoff), side="left"))
    stop = max(0, cutoff_idx - horizon)
    return np.arange(stop, dtype=int)


__all__ = ["matured_label_indices_before"]
