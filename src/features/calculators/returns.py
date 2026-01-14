"""
Calculator para log returns - Backward Compatibility Layer
==========================================================
Delegates to feature_store.core for the actual implementation.

DEPRECATED: Import directly from feature_store instead:
    from feature_store import LogReturnCalculator

Contrato: CTR-005a
"""

import numpy as np
import pandas as pd

# Import from SSOT
from feature_store.core import LogReturnCalculator, CalculatorRegistry


def log_return(
    close: pd.Series,
    periods: int,
    bar_idx: int
) -> float:
    """
    Calcula log return para un periodo dado.

    DEPRECATED: Use LogReturnCalculator from feature_store instead.

    Args:
        close: Serie de precios de cierre
        periods: Numero de barras hacia atras (1=5min, 12=1h, 48=4h)
        bar_idx: Indice de la barra actual

    Returns:
        Log return como float. Retorna 0.0 si no hay suficientes datos.
    """
    if bar_idx < periods:
        return 0.0

    current = close.iloc[bar_idx]
    previous = close.iloc[bar_idx - periods]

    if previous <= 0 or current <= 0:
        return 0.0

    result = np.log(current / previous)

    if np.isnan(result) or np.isinf(result):
        return 0.0

    return float(result)
