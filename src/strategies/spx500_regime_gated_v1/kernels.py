"""Puente a los kernels de ciencia ya implementados (raíz de SP500/).

Los módulos `economic_metrics`, `deflated_sharpe`, `pbo` y `gates` viven en la
raíz del repo como implementaciones de referencia. Este shim los pone en el path
para que la capa de estrategia los consuma SIN duplicar código ni tocar los
originales. La regla del repo es que la estrategia usa los gates, no los reescribe.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deflated_sharpe import deflated_sharpe, expected_max_sharpe  # noqa: E402
from economic_metrics import PERIODS_PER_YEAR, cer, cer_gain_bps, sharpe  # noqa: E402
from gates import (  # noqa: E402
    REQUIRED_BENCHMARKS,
    gate_g4,
    gate_g6,
    validate_report_benchmarks,
)
from pbo import cscv  # noqa: E402

__all__ = [
    "PERIODS_PER_YEAR",
    "REQUIRED_BENCHMARKS",
    "cer",
    "cer_gain_bps",
    "cscv",
    "deflated_sharpe",
    "expected_max_sharpe",
    "gate_g4",
    "gate_g6",
    "sharpe",
    "validate_report_benchmarks",
]
