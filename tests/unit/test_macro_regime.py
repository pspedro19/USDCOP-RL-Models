"""
Unit tests for MacroRegimeEngine (Phase 1).
Tests HMM regime detection, Granger causality, changepoints, correlations, z-score alerts.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_macro_df():
    """Create a sample macro DataFrame with multiple variables."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # Generate correlated macro variables
    dxy = 104.0 + np.cumsum(np.random.randn(n) * 0.1)
    vix = 18.0 + np.cumsum(np.random.randn(n) * 0.3)
    vix = np.clip(vix, 10, 60)
    wti = 75.0 + np.cumsum(np.random.randn(n) * 0.5)
    embi = 350.0 + np.cumsum(np.random.randn(n) * 2)
    ust10y = 4.3 + np.cumsum(np.random.randn(n) * 0.01)
    ibr = 9.5 + np.cumsum(np.random.randn(n) * 0.005)
    gold = 2050 + np.cumsum(np.random.randn(n) * 3)
    brent = 78 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        "dxy_close": dxy,
        "vix_close": vix,
        "wti_close": wti,
        "embi_close": embi,
        "ust10y_close": ust10y,
        "ibr_close": ibr,
        "gold_close": gold,
        "brent_close": brent,
    }, index=dates)
    return df


@pytest.fixture
def sample_cop_series():
    """Create a sample USDCOP close series."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    base = 4200.0
    prices = base + np.cumsum(np.random.randn(n) * 10)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def engine():
    """Create a MacroRegimeEngine instance."""
    from src.analysis.macro_regime import MacroRegimeEngine
    return MacroRegimeEngine()


class TestMacroRegimeEngine:
    """Test the MacroRegimeEngine class."""

    def test_analyze_returns_report(self, engine, sample_macro_df, sample_cop_series):
        """analyze() should return a MacroRegimeReport."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert report is not None

    def test_hmm_regime_detected(self, engine, sample_macro_df, sample_cop_series):
        """HMM should detect a regime state."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert report.regime is not None
        assert report.regime.label in ("risk_on", "transition", "risk_off")
        assert 0.0 <= report.regime.confidence <= 1.0
        assert report.regime.since is not None

    def test_hmm_transition_probabilities(self, engine, sample_macro_df, sample_cop_series):
        """HMM should produce transition probabilities."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        tp = report.regime.transition_probabilities
        if tp:
            # Keys should be regime labels
            for key in tp:
                assert key in ("risk_on", "transition", "risk_off")
            # Probabilities should sum to ~1
            assert abs(sum(tp.values()) - 1.0) < 0.01

    def test_granger_leaders(self, engine, sample_macro_df, sample_cop_series):
        """Granger causality should identify leading variables."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        # Granger leaders may or may not be found with random data
        assert isinstance(report.granger_leaders, list)
        for leader in report.granger_leaders:
            assert leader.variable != ""
            assert 1 <= leader.optimal_lag <= 5
            assert leader.p_value <= 0.1  # Threshold for inclusion

    def test_changepoints(self, engine, sample_macro_df, sample_cop_series):
        """Changepoint detection should return valid changepoints."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert isinstance(report.changepoints, list)
        for cp in report.changepoints:
            assert cp.date != ""
            assert cp.variable != ""
            assert cp.direction in ("up", "down")
            assert cp.magnitude >= 0

    def test_correlations(self, engine, sample_macro_df, sample_cop_series):
        """Rolling correlations should be computed for macro variables."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert isinstance(report.correlations, dict)
        for key, data in report.correlations.items():
            assert "current" in data
            assert "avg_60d" in data
            assert "expected_direction" in data
            if data["current"] is not None:
                assert -1.0 <= data["current"] <= 1.0

    def test_zscore_alerts(self, engine, sample_macro_df, sample_cop_series):
        """Z-score alerts should flag extreme readings."""
        # Inject an extreme value to trigger alert
        sample_macro_df.iloc[-1, sample_macro_df.columns.get_loc("vix_close")] = 55.0

        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert isinstance(report.zscore_alerts, list)
        for alert in report.zscore_alerts:
            assert alert.variable != ""
            assert abs(alert.z_score) > 2.0
            assert alert.direction in ("extreme_high", "extreme_low")

    def test_insights_generated(self, engine, sample_macro_df, sample_cop_series):
        """Insights should be generated in Spanish."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        assert isinstance(report.insights, list)
        assert len(report.insights) > 0
        # First insight should describe the regime
        assert "regimen" in report.insights[0].lower() or "regime" in report.insights[0].lower()

    def test_empty_data(self, engine):
        """Empty DataFrames should not crash."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        report = engine.analyze(empty_df, empty_series, "2026-01-01", "2026-01-05")
        assert report is not None
        assert report.regime.label == "transition"

    def test_short_data(self, engine):
        """Short data should produce partial results without crashing."""
        np.random.seed(42)
        dates = pd.date_range("2025-12-01", periods=30, freq="B")
        macro_df = pd.DataFrame({
            "dxy_close": 104 + np.random.randn(30) * 0.1,
        }, index=dates)
        cop = pd.Series(4200 + np.random.randn(30) * 10, index=dates)

        report = engine.analyze(macro_df, cop, "2026-01-01", "2026-01-05")
        assert report is not None

    def test_to_dict(self, engine, sample_macro_df, sample_cop_series):
        """to_dict() should return a serializable dict."""
        report = engine.analyze(
            sample_macro_df, sample_cop_series,
            week_start="2026-01-05", week_end="2026-02-20",
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "regime" in d
        assert "granger_leaders" in d
        assert "correlations" in d
        assert "zscore_alerts" in d
        assert "insights" in d
        # Verify serializable
        import json
        json.dumps(d)


class TestMacroRegimeDataClasses:
    """Test dataclass construction and serialization."""

    def test_regime_state_defaults(self):
        from src.analysis.macro_regime import RegimeState
        rs = RegimeState()
        assert rs.label == "transition"
        assert rs.confidence == 0.0

    def test_granger_leader_to_dict(self):
        from src.analysis.macro_regime import GrangerLeader
        gl = GrangerLeader(variable="dxy", optimal_lag=2, f_statistic=5.3, p_value=0.02, direction="positive")
        d = gl.to_dict()
        assert d["variable"] == "dxy"
        assert d["optimal_lag"] == 2

    def test_zscore_alert_to_dict(self):
        from src.analysis.macro_regime import ZScoreAlert
        alert = ZScoreAlert(variable="vix", variable_name="VIX", z_score=2.5, direction="extreme_high")
        d = alert.to_dict()
        assert d["z_score"] == 2.5

    def test_find_column(self):
        from src.analysis.macro_regime import _find_column
        df = pd.DataFrame({"dxy_close": [1, 2], "vix_close": [3, 4]})
        assert _find_column(df, "dxy") == "dxy_close"
        assert _find_column(df, "vix") == "vix_close"
        assert _find_column(df, "nonexistent") is None
