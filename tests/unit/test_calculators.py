"""
Tests para Feature Calculators.
CLAUDE-T2 | Plan Item: P1-13
Contrato: CTR-005

Valida:
- returns.log_return sin NaN
- rsi.calculate en [0, 100]
- atr.calculate_pct >= 0
- adx.calculate en [0, 100]
- macro z_score, change_1d
- Todos deterministas
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.calculators import returns, rsi, atr, adx, macro


class TestReturnsCalculator:
    """Tests para returns.py (CTR-005a)."""

    @pytest.fixture
    def close_series(self):
        """100 barras de datos sinteticos reproducibles."""
        np.random.seed(42)
        return pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))

    def test_log_return_5min_correct(self, close_series):
        """Log return 5min (periods=1) calculado correctamente."""
        ret = returns.log_return(close_series, periods=1, bar_idx=50)
        expected = np.log(close_series.iloc[50] / close_series.iloc[49])
        assert np.isclose(ret, expected, rtol=1e-10)

    def test_log_return_1h_correct(self, close_series):
        """Log return 1h (periods=12) calculado correctamente."""
        ret = returns.log_return(close_series, periods=12, bar_idx=50)
        expected = np.log(close_series.iloc[50] / close_series.iloc[38])
        assert np.isclose(ret, expected, rtol=1e-10)

    def test_log_return_4h_correct(self, close_series):
        """Log return 4h (periods=48) calculado correctamente."""
        ret = returns.log_return(close_series, periods=48, bar_idx=60)
        expected = np.log(close_series.iloc[60] / close_series.iloc[12])
        assert np.isclose(ret, expected, rtol=1e-10)

    def test_log_return_insufficient_data_returns_zero(self, close_series):
        """Sin suficientes datos retorna 0.0."""
        ret = returns.log_return(close_series, periods=12, bar_idx=5)
        assert ret == 0.0

    def test_log_return_no_nan(self, close_series):
        """Nunca retorna NaN."""
        for bar_idx in range(100):
            ret = returns.log_return(close_series, periods=1, bar_idx=bar_idx)
            assert not np.isnan(ret), f"NaN en bar_idx={bar_idx}"

    def test_log_return_deterministic(self, close_series):
        """Mismo input produce mismo output."""
        r1 = returns.log_return(close_series, periods=1, bar_idx=50)
        r2 = returns.log_return(close_series, periods=1, bar_idx=50)
        r3 = returns.log_return(close_series, periods=1, bar_idx=50)
        assert r1 == r2 == r3


class TestRSICalculator:
    """Tests para rsi.py (CTR-005b)."""

    @pytest.fixture
    def close_series(self):
        np.random.seed(42)
        return pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))

    def test_rsi_in_valid_range(self, close_series):
        """RSI siempre en [0, 100]."""
        for bar_idx in range(9, 100):
            rsi_val = rsi.calculate(close_series, period=9, bar_idx=bar_idx)
            assert 0.0 <= rsi_val <= 100.0, f"RSI={rsi_val} fuera de rango en bar {bar_idx}"

    def test_rsi_neutral_on_insufficient_data(self, close_series):
        """RSI retorna 50 (neutral) sin suficientes datos."""
        rsi_val = rsi.calculate(close_series, period=9, bar_idx=5)
        assert rsi_val == 50.0

    def test_rsi_deterministic(self, close_series):
        """Mismo input produce mismo RSI."""
        r1 = rsi.calculate(close_series, period=9, bar_idx=50)
        r2 = rsi.calculate(close_series, period=9, bar_idx=50)
        assert r1 == r2

    def test_rsi_no_nan(self, close_series):
        """Nunca retorna NaN."""
        for bar_idx in range(100):
            rsi_val = rsi.calculate(close_series, period=9, bar_idx=bar_idx)
            assert not np.isnan(rsi_val), f"NaN en bar_idx={bar_idx}"

    def test_rsi_responds_to_price_movement(self):
        """RSI sube con precios subiendo, baja con precios bajando."""
        # Precios subiendo
        rising = pd.Series([100 + i for i in range(50)])
        rsi_rising = rsi.calculate(rising, period=9, bar_idx=40)

        # Precios bajando
        falling = pd.Series([100 - i for i in range(50)])
        rsi_falling = rsi.calculate(falling, period=9, bar_idx=40)

        assert rsi_rising > 50, f"RSI deberia ser > 50 para precios subiendo: {rsi_rising}"
        assert rsi_falling < 50, f"RSI deberia ser < 50 para precios bajando: {rsi_falling}"


class TestATRCalculator:
    """Tests para atr.py (CTR-005c)."""

    @pytest.fixture
    def ohlcv(self):
        np.random.seed(42)
        close = pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))
        return pd.DataFrame({
            "open": close + np.random.randn(100),
            "high": close + np.abs(np.random.randn(100) * 3),
            "low": close - np.abs(np.random.randn(100) * 3),
            "close": close,
        })

    def test_atr_pct_non_negative(self, ohlcv):
        """ATR % siempre >= 0."""
        for bar_idx in range(10, 100):
            atr_val = atr.calculate_pct(ohlcv, period=10, bar_idx=bar_idx)
            assert atr_val >= 0.0, f"ATR={atr_val} negativo en bar {bar_idx}"

    def test_atr_pct_zero_on_insufficient_data(self, ohlcv):
        """ATR % retorna 0 sin suficientes datos."""
        atr_val = atr.calculate_pct(ohlcv, period=10, bar_idx=5)
        assert atr_val == 0.0

    def test_atr_pct_deterministic(self, ohlcv):
        """Mismo input produce mismo ATR."""
        a1 = atr.calculate_pct(ohlcv, period=10, bar_idx=50)
        a2 = atr.calculate_pct(ohlcv, period=10, bar_idx=50)
        assert a1 == a2

    def test_atr_pct_no_nan(self, ohlcv):
        """Nunca retorna NaN."""
        for bar_idx in range(100):
            atr_val = atr.calculate_pct(ohlcv, period=10, bar_idx=bar_idx)
            assert not np.isnan(atr_val), f"NaN en bar_idx={bar_idx}"

    def test_atr_pct_increases_with_volatility(self):
        """ATR aumenta con mayor volatilidad."""
        # Baja volatilidad
        low_vol = pd.DataFrame({
            "open": [100] * 50,
            "high": [101] * 50,
            "low": [99] * 50,
            "close": [100] * 50,
        })

        # Alta volatilidad
        high_vol = pd.DataFrame({
            "open": [100] * 50,
            "high": [110] * 50,
            "low": [90] * 50,
            "close": [100] * 50,
        })

        atr_low = atr.calculate_pct(low_vol, period=10, bar_idx=40)
        atr_high = atr.calculate_pct(high_vol, period=10, bar_idx=40)

        assert atr_high > atr_low, f"ATR alta vol ({atr_high}) deberia ser > ATR baja vol ({atr_low})"


class TestADXCalculator:
    """Tests para adx.py (CTR-005d)."""

    @pytest.fixture
    def ohlcv(self):
        np.random.seed(42)
        close = pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))
        return pd.DataFrame({
            "open": close + np.random.randn(100),
            "high": close + np.abs(np.random.randn(100) * 3),
            "low": close - np.abs(np.random.randn(100) * 3),
            "close": close,
        })

    def test_adx_in_valid_range(self, ohlcv):
        """ADX siempre en [0, 100]."""
        for bar_idx in range(30, 100):
            adx_val = adx.calculate(ohlcv, period=14, bar_idx=bar_idx)
            assert 0.0 <= adx_val <= 100.0, f"ADX={adx_val} fuera de rango en bar {bar_idx}"

    def test_adx_zero_on_insufficient_data(self, ohlcv):
        """ADX retorna 0 sin suficientes datos."""
        adx_val = adx.calculate(ohlcv, period=14, bar_idx=10)
        assert adx_val == 0.0

    def test_adx_deterministic(self, ohlcv):
        """Mismo input produce mismo ADX."""
        a1 = adx.calculate(ohlcv, period=14, bar_idx=50)
        a2 = adx.calculate(ohlcv, period=14, bar_idx=50)
        assert a1 == a2

    def test_adx_no_nan(self, ohlcv):
        """Nunca retorna NaN."""
        for bar_idx in range(100):
            adx_val = adx.calculate(ohlcv, period=14, bar_idx=bar_idx)
            assert not np.isnan(adx_val), f"NaN en bar_idx={bar_idx}"


class TestMacroCalculator:
    """Tests para macro.py (CTR-005e)."""

    @pytest.fixture
    def dxy_series(self):
        np.random.seed(42)
        return pd.Series(104 + np.cumsum(np.random.randn(300) * 0.1))

    def test_z_score_reasonable_range(self, dxy_series):
        """Z-score en rango razonable [-4, 4]."""
        for bar_idx in range(60, 300):
            z = macro.z_score(dxy_series, bar_idx=bar_idx)
            assert -4.0 <= z <= 4.0, f"z_score={z} fuera de rango en bar {bar_idx}"

    def test_z_score_zero_on_insufficient_data(self, dxy_series):
        """Z-score retorna 0 sin suficientes datos."""
        z = macro.z_score(dxy_series, bar_idx=0)
        assert z == 0.0

    def test_change_1d_reasonable_range(self, dxy_series):
        """Cambio diario en rango razonable [-0.10, 0.10]."""
        for bar_idx in range(290, 300):
            change = macro.change_1d(dxy_series, bar_idx=bar_idx)
            assert -0.10 <= change <= 0.10, f"change_1d={change} fuera de rango en bar {bar_idx}"

    def test_change_1d_zero_on_insufficient_data(self, dxy_series):
        """Cambio diario retorna 0 sin suficientes datos."""
        change = macro.change_1d(dxy_series, bar_idx=0)
        assert change == 0.0

    def test_get_value_returns_correct_value(self, dxy_series):
        """get_value retorna el valor correcto."""
        val = macro.get_value(dxy_series, bar_idx=50)
        assert val == dxy_series.iloc[50]

    def test_get_value_zero_on_nan(self):
        """get_value retorna 0 para NaN."""
        series = pd.Series([1.0, np.nan, 3.0])
        val = macro.get_value(series, bar_idx=1)
        assert val == 0.0

    def test_macro_deterministic(self, dxy_series):
        """Funciones macro son deterministicas."""
        z1 = macro.z_score(dxy_series, bar_idx=100)
        z2 = macro.z_score(dxy_series, bar_idx=100)
        assert z1 == z2

        c1 = macro.change_1d(dxy_series, bar_idx=290)
        c2 = macro.change_1d(dxy_series, bar_idx=290)
        assert c1 == c2


class TestCalculatorsIntegration:
    """Tests de integracion para todos los calculators."""

    @pytest.fixture
    def full_data(self):
        """Dataset completo para tests de integracion."""
        np.random.seed(42)
        n = 500
        close = pd.Series(4250 + np.cumsum(np.random.randn(n) * 5))
        ohlcv = pd.DataFrame({
            "open": close + np.random.randn(n),
            "high": close + np.abs(np.random.randn(n) * 3),
            "low": close - np.abs(np.random.randn(n) * 3),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, n),
        })
        macro_df = pd.DataFrame({
            "dxy": np.random.uniform(103, 105, n),
            "vix": np.random.uniform(15, 25, n),
            "embi": np.random.uniform(300, 400, n),
            "brent": np.random.uniform(70, 80, n),
            "usdmxn": np.random.uniform(17, 18, n),
            "rate_spread": np.random.uniform(5, 7, n),
        })
        return ohlcv, macro_df

    def test_all_calculators_no_exceptions(self, full_data):
        """Todos los calculators funcionan sin excepciones."""
        ohlcv, macro_df = full_data

        for bar_idx in range(50, 500):
            # Returns
            returns.log_return(ohlcv["close"], periods=1, bar_idx=bar_idx)
            returns.log_return(ohlcv["close"], periods=12, bar_idx=bar_idx)
            returns.log_return(ohlcv["close"], periods=48, bar_idx=bar_idx)

            # Technical
            rsi.calculate(ohlcv["close"], period=9, bar_idx=bar_idx)
            atr.calculate_pct(ohlcv, period=10, bar_idx=bar_idx)
            adx.calculate(ohlcv, period=14, bar_idx=bar_idx)

            # Macro
            macro.z_score(macro_df["dxy"], bar_idx=bar_idx)
            macro.change_1d(macro_df["dxy"], bar_idx=bar_idx)
            macro.z_score(macro_df["vix"], bar_idx=bar_idx)

    def test_all_calculators_no_nan(self, full_data):
        """Ningun calculator produce NaN."""
        ohlcv, macro_df = full_data

        for bar_idx in range(50, 500):
            vals = [
                returns.log_return(ohlcv["close"], periods=1, bar_idx=bar_idx),
                rsi.calculate(ohlcv["close"], period=9, bar_idx=bar_idx),
                atr.calculate_pct(ohlcv, period=10, bar_idx=bar_idx),
                adx.calculate(ohlcv, period=14, bar_idx=bar_idx),
                macro.z_score(macro_df["dxy"], bar_idx=bar_idx),
            ]
            for val in vals:
                assert not np.isnan(val), f"NaN en bar_idx={bar_idx}"
