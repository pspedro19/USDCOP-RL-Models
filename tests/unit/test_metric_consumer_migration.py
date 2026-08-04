from __future__ import annotations

import importlib.util
import ast
from pathlib import Path

import numpy as np

from src.metrics.formulas import sharpe_ratio as governed_sharpe_ratio


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pipeline_governed_ratio():
    path = REPO_ROOT / "services" / "pipeline_data_api.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_governed_risk_adjusted_ratio"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "governed_sharpe_ratio": governed_sharpe_ratio}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_governed_risk_adjusted_ratio"]


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


def test_spx_distribution_public_api_remains_available() -> None:
    module = _economic_metrics()
    distribution = module.sharpe_distribution(n_paths=3, seed=42, t=20)
    assert distribution.name == "sharpe_oos"
    assert len(distribution) == 3
    assert np.isfinite(distribution.to_numpy()).all()


def test_pipeline_api_sharpe_delegates_without_numeric_drift() -> None:
    calculate = _pipeline_governed_ratio()
    returns = np.array([0.01, -0.004, 0.006, 0.002, -0.001], dtype=float)

    expected = governed_sharpe_ratio(returns, periods_per_year=252)
    assert expected is not None
    assert calculate(returns) == expected


def test_pipeline_api_sharpe_preserves_legacy_zero_for_degenerate_input() -> None:
    calculate = _pipeline_governed_ratio()

    assert calculate(np.array([], dtype=float)) == 0.0
    assert calculate(np.array([0.01], dtype=float)) == 0.0
    assert calculate(np.array([0.01, 0.01], dtype=float)) == 0.0
