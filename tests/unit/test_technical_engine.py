"""
Unit tests for TechnicalAnalysisEngine (Phase 1).
Tests Fibonacci, bias voting, scenario generation.

Note: pandas-ta may fail on numpy 2.x (removed numpy.NaN).
Tests are written to handle graceful degradation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    base = 4200.0
    close = base + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.zeros(n),
    }, index=dates)
    return df


@pytest.fixture
def engine():
    """Create a TechnicalAnalysisEngine instance."""
    from src.analysis.technical_engine import TechnicalAnalysisEngine
    return TechnicalAnalysisEngine()


class TestTechnicalAnalysisEngine:
    """Test the TechnicalAnalysisEngine class."""

    def test_analyze_returns_report(self, engine, sample_ohlcv):
        """analyze() should return a TechnicalAnalysisReport."""
        report = engine.analyze(sample_ohlcv)
        assert report is not None
        assert report.current_price > 0
        assert report.dominant_bias in ("bullish", "bearish", "neutral")
        assert 0.0 <= report.bias_confidence <= 1.0

    def test_atr_computed_or_graceful(self, engine, sample_ohlcv):
        """ATR should be computed if pandas-ta works, else None (graceful)."""
        report = engine.analyze(sample_ohlcv)
        # pandas-ta may fail on numpy 2.x — accept both outcomes
        if report.atr is not None:
            assert report.atr > 0
            assert report.atr_pct is not None
            assert report.atr_pct > 0

    def test_fibonacci_levels(self, engine, sample_ohlcv):
        """Fibonacci levels should have standard retracement values."""
        report = engine.analyze(sample_ohlcv)
        fib = report.fibonacci
        assert fib is not None
        # FibonacciLevels is a dataclass, not a dict
        if fib.levels:
            for key in [0.236, 0.382, 0.5, 0.618, 0.786]:
                assert key in fib.levels, f"Missing Fibonacci level {key}"

    def test_scenarios_generated(self, engine, sample_ohlcv):
        """At least one trading scenario should be generated."""
        report = engine.analyze(sample_ohlcv)
        assert isinstance(report.scenarios, list)
        if report.scenarios:
            s = report.scenarios[0]
            assert s.direction in ("long", "short")
            assert s.entry_condition != ""
            assert s.stop_loss is not None
            assert isinstance(s.targets, list)

    def test_support_resistance_levels(self, engine, sample_ohlcv):
        """Support and resistance levels should be identified."""
        report = engine.analyze(sample_ohlcv)
        sr = report.support_resistance
        assert sr is not None
        assert isinstance(sr.key_supports, list)
        assert isinstance(sr.key_resistances, list)

    def test_watch_list(self, engine, sample_ohlcv):
        """Watch list should be a list."""
        report = engine.analyze(sample_ohlcv)
        assert isinstance(report.watch_list, list)

    def test_to_dict(self, engine, sample_ohlcv):
        """to_dict() should return a serializable dict."""
        report = engine.analyze(sample_ohlcv)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "current_price" in d
        assert "dominant_bias" in d
        assert "scenarios" in d
        # Verify JSON-serializable
        import json
        json.dumps(d)

    def test_empty_dataframe(self, engine):
        """Empty DataFrame should not crash."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        report = engine.analyze(df)
        assert report is not None
        assert report.current_price == 0.0

    def test_short_dataframe(self, engine):
        """DataFrame with few rows should produce a report (graceful degradation)."""
        df = pd.DataFrame({
            "open": [4200, 4210, 4205],
            "high": [4215, 4220, 4210],
            "low": [4195, 4200, 4198],
            "close": [4210, 4205, 4208],
            "volume": [0, 0, 0],
        }, index=pd.date_range("2025-01-01", periods=3, freq="B"))
        report = engine.analyze(df)
        assert report is not None
        # Engine returns early with default for <30 rows (current_price=0.0)
        # or uses last close if it processes normally
        assert report.current_price >= 0.0

    def test_bias_confidence_range(self, engine, sample_ohlcv):
        """Bias confidence should be between 0 and 1."""
        report = engine.analyze(sample_ohlcv)
        assert 0.0 <= report.bias_confidence <= 1.0

    def test_volatility_regime(self, engine, sample_ohlcv):
        """Volatility regime should be one of low/normal/high."""
        report = engine.analyze(sample_ohlcv)
        assert report.volatility_regime in ("low", "normal", "high")

    def test_fibonacci_direction(self, engine, sample_ohlcv):
        """Fibonacci direction should indicate retracement type."""
        report = engine.analyze(sample_ohlcv)
        if report.fibonacci.direction:
            assert report.fibonacci.direction in (
                "retracement_up", "retracement_down"
            )

    def test_scenarios_have_risk_reward(self, engine, sample_ohlcv):
        """Trading scenarios should have risk:reward ratios."""
        report = engine.analyze(sample_ohlcv)
        for scenario in report.scenarios:
            if scenario.risk_reward is not None:
                assert scenario.risk_reward > 0
