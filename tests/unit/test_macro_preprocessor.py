"""
Tests for MacroDataPreprocessor (Phase 0)
==========================================
"""

import math
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def macro_df():
    """Create a synthetic macro DataFrame for testing."""
    dates = pd.date_range("2025-06-01", periods=120, freq="B")
    np.random.seed(42)
    data = {
        "dxy_close": 104 + np.cumsum(np.random.randn(120) * 0.2),
        "vix_close": 18 + np.cumsum(np.random.randn(120) * 0.5),
        "wti_close": 75 + np.cumsum(np.random.randn(120) * 0.8),
        "embi_col": 350 + np.cumsum(np.random.randn(120) * 2),
        "ust10y_close": 4.3 + np.cumsum(np.random.randn(120) * 0.02),
        "ust2y_close": 4.1 + np.cumsum(np.random.randn(120) * 0.02),
        "ibr_overnight": 9.5 + np.cumsum(np.random.randn(120) * 0.01),
        "tpm_banrep": np.full(120, 9.25),
        "gold_close": 2100 + np.cumsum(np.random.randn(120) * 5),
        "brent_close": 78 + np.cumsum(np.random.randn(120) * 0.7),
        "fedfunds_rate": np.full(120, 5.25),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "fecha"
    return df


@pytest.fixture
def cop_series(macro_df):
    """Create a synthetic COP close series."""
    np.random.seed(123)
    dates = macro_df.index
    cop = 4200 + np.cumsum(np.random.randn(len(dates)) * 15)
    return pd.Series(cop, index=dates, name="close")


class TestMacroDataPreprocessor:
    def test_compute_digest_returns_all_groups(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        # Should have groups populated
        assert digest.groups, "groups should not be empty"
        assert len(digest.groups) >= 3, f"Expected >= 3 groups, got {len(digest.groups)}"

    def test_top_movers_returns_top_5(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        assert len(digest.top_movers) <= 5
        if digest.top_movers:
            # Should be sorted by |z_score_20d| descending
            z_scores = [abs(m.get("z_score_20d", 0) or 0) for m in digest.top_movers]
            assert z_scores == sorted(z_scores, reverse=True)

    def test_anomalies_have_high_z_score(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        for anomaly in digest.anomalies:
            z = abs(anomaly.get("z_score_20d", 0) or 0)
            assert z >= 2.0, f"Anomaly z-score should be >= 2.0, got {z}"

    def test_digest_to_dict(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        d = digest.to_dict()
        assert isinstance(d, dict)
        assert "groups" in d
        assert "top_movers" in d
        assert "anomalies" in d
        assert "as_of_date" in d

    def test_digest_to_prompt_text(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        text = digest.to_prompt_text()
        assert isinstance(text, str)
        assert len(text) > 50, "Prompt text should be substantial"

    def test_correlations_computed(self, macro_df, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(macro_df, cop_series, date(2025, 11, 28))

        # At least some variables should have correlations
        d = digest.to_dict()
        has_correlation = False
        for group_vars in d["groups"].values():
            for var in group_vars:
                if var.get("correlation_20d") is not None:
                    has_correlation = True
                    break
        assert has_correlation, "Should have at least one variable with correlation"

    def test_empty_macro_df(self, cop_series):
        from src.analysis.macro_preprocessor import MacroDataPreprocessor

        preprocessor = MacroDataPreprocessor()
        digest = preprocessor.compute_digest(pd.DataFrame(), cop_series, date(2025, 11, 28))

        assert digest.groups == {}
        assert digest.top_movers == []
        assert digest.anomalies == []


class TestVariableGroups:
    def test_all_groups_defined(self):
        from src.analysis.macro_preprocessor import VARIABLE_GROUPS

        expected = {"commodities", "usd_strength", "colombia_rates", "risk_sentiment", "inflation", "fed_policy"}
        assert set(VARIABLE_GROUPS.keys()) == expected

    def test_impact_chains_exist(self):
        from src.analysis.macro_preprocessor import IMPACT_CHAINS

        assert len(IMPACT_CHAINS) > 0
        for group, text in IMPACT_CHAINS.items():
            assert isinstance(text, str)
            assert len(text) > 10
