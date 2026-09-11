import numpy as np
import pandas as pd
import pytest

from src.gold_rl.backtest import block_bootstrap_pvalue


def test_bootstrap_pvalue_reports_finite_monte_carlo_resolution():
    returns = pd.Series(np.full(60, 0.01))
    result = block_bootstrap_pvalue(returns, block=5, n_boot=99, seed=7)
    assert result["n_exceedances"] == 0
    assert result["p_value"] ==  pytest.approx(1 / 100)
    assert result["p_value"] > 0.0
