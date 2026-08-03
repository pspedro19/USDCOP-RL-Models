from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from src.metrics.formulas import sharpe_ratio as governed_sharpe_ratio


REPO_ROOT = Path(__file__).resolve().parents[2]


def _economic_metrics():
    path = (
        REPO_ROOT
        / "src"
        / "strategies"
        / "spx500_regime_gated_v1"
        / "economic_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("spx_economic_metrics_bl18", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spx_sharpe_delegates_to_governed_formula_without_numeric_drift() -> None:
    module = _economic_metrics()
    returns = np.array([0.01, -0.004, 0.006, 0.002, -0.001], dtype=float)

    expected = governed_sharpe_ratio(returns, periods_per_year=52)
    assert expected is not None
    assert module.sharpe(returns, periods_per_year=52) == expected


def test_spx_sharpe_preserves_legacy_zero_for_degenerate_samples() -> None:
    module = _economic_metrics()
    assert module.sharpe(np.array([], dtype=float)) == 0.0
    assert module.sharpe(np.array([0.01], dtype=float)) == 0.0
    assert module.sharpe(np.array([0.01, 0.01], dtype=float)) == 0.0
