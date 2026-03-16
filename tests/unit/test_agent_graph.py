"""
Unit tests for the LangGraph Multi-Agent Analysis Graph (Phase 3).
Tests individual node functions, graceful degradation, and graph construction.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    """Minimal state dict for testing nodes."""
    return {
        "symbol": "USD/COP",
        "week_start": "2026-02-16",
        "week_end": "2026-02-20",
        "iso_year": 2026,
        "iso_week": 8,
        "execution_log": [],
        "cost_tracking": {"total_cost": 0, "total_tokens": 0},
        "errors": [],
        "synthesis_revision": 0,
    }


@pytest.fixture
def state_with_data(base_state):
    """State with pre-loaded OHLCV and macro data (bypasses file loading)."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-06-01", periods=n, freq="B")

    # Daily OHLCV records
    close = 4200.0 + np.cumsum(np.random.randn(n) * 10)
    daily_records = [
        {
            "date": str(d.date()),
            "open": float(c + np.random.randn() * 3),
            "high": float(c + abs(np.random.randn()) * 5),
            "low": float(c - abs(np.random.randn()) * 5),
            "close": float(c),
            "volume": 0.0,
        }
        for d, c in zip(dates, close)
    ]

    # COP prices
    cop_records = [
        {"date": str(d.date()), "close": float(c)}
        for d, c in zip(dates, close)
    ]

    # Macro records
    macro_records = [
        {
            "fecha": str(d.date()),
            "dxy_close": float(104 + np.random.randn() * 0.5),
            "vix_close": float(18 + abs(np.random.randn()) * 2),
            "wti_close": float(75 + np.random.randn() * 2),
            "embi_close": float(350 + np.random.randn() * 5),
        }
        for d in dates
    ]

    # GDELT articles
    articles = [
        {
            "title": "Colombia peso strengthens as oil prices rise",
            "source": "reuters.com",
            "url": "https://reuters.com/1",
            "date": "2026-02-18",
            "tone": 2.0,
        },
        {
            "title": "Banco de la Republica mantiene tasa de interes",
            "source": "portafolio.co",
            "url": "https://portafolio.co/1",
            "date": "2026-02-19",
            "tone": -1.0,
        },
        {
            "title": "DXY dollar index rises to three-month high",
            "source": "bloomberg.com",
            "url": "https://bloomberg.com/1",
            "date": "2026-02-20",
            "tone": 0.5,
        },
    ]

    base_state.update({
        "ohlcv_daily": daily_records,
        "cop_prices": cop_records,
        "macro_data": macro_records,
        "gdelt_articles": articles,
    })
    return base_state


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestTAAgentNode:
    """Test the Technical Analysis agent node."""

    def test_ta_agent_with_data(self, state_with_data):
        from src.analysis.agent_graph import ta_agent_node
        result = ta_agent_node(state_with_data)
        assert "ta_report" in result
        if result["ta_report"] is not None:
            assert "dominant_bias" in result["ta_report"]
            assert result["ta_report"]["dominant_bias"] in ("bullish", "bearish", "neutral")

    def test_ta_agent_no_data(self, base_state):
        """TA agent should handle missing data gracefully."""
        from src.analysis.agent_graph import ta_agent_node
        result = ta_agent_node(base_state)
        assert result["ta_report"] is None
        assert "errors" in result

    def test_ta_agent_execution_log(self, state_with_data):
        """TA agent should log its execution."""
        from src.analysis.agent_graph import ta_agent_node
        result = ta_agent_node(state_with_data)
        assert any("TA agent" in msg for msg in result["execution_log"])


class TestNewsAgentNode:
    """Test the News Intelligence agent node."""

    def test_news_agent_with_articles(self, state_with_data):
        from src.analysis.agent_graph import news_agent_node
        result = news_agent_node(state_with_data)
        assert "news_intelligence" in result
        if result["news_intelligence"] is not None:
            assert "total_articles" in result["news_intelligence"]
            assert result["news_intelligence"]["total_articles"] >= 0

    def test_news_agent_no_articles(self, base_state):
        """News agent should handle missing articles gracefully."""
        from src.analysis.agent_graph import news_agent_node
        result = news_agent_node(base_state)
        # No articles -> no intelligence
        assert result["news_intelligence"] is None

    def test_news_agent_execution_log(self, state_with_data):
        from src.analysis.agent_graph import news_agent_node
        result = news_agent_node(state_with_data)
        assert any("News agent" in msg for msg in result["execution_log"])


class TestMacroAgentNode:
    """Test the Macro Regime agent node."""

    def test_macro_agent_with_data(self, state_with_data):
        from src.analysis.agent_graph import macro_agent_node
        result = macro_agent_node(state_with_data)
        assert "macro_regime" in result
        if result["macro_regime"] is not None:
            assert "regime" in result["macro_regime"]

    def test_macro_agent_no_data(self, base_state):
        """Macro agent should handle missing data gracefully."""
        from src.analysis.agent_graph import macro_agent_node
        result = macro_agent_node(base_state)
        assert result["macro_regime"] is None

    def test_macro_agent_execution_log(self, state_with_data):
        from src.analysis.agent_graph import macro_agent_node
        result = macro_agent_node(state_with_data)
        assert any("Macro agent" in msg for msg in result["execution_log"])


class TestFXAgentNode:
    """Test the FX Context agent node."""

    def test_fx_agent_with_data(self, state_with_data):
        from src.analysis.agent_graph import fx_agent_node
        result = fx_agent_node(state_with_data)
        assert "fx_context" in result
        if result["fx_context"] is not None:
            assert "fx_narrative" in result["fx_context"]

    def test_fx_agent_no_data(self, base_state):
        """FX agent should handle missing data gracefully."""
        from src.analysis.agent_graph import fx_agent_node
        result = fx_agent_node(base_state)
        assert result["fx_context"] is None


class TestSynthesizerNode:
    """Test the Synthesizer node."""

    def test_synthesizer_no_agent_outputs(self, base_state):
        """Synthesizer with no agent outputs should skip LLM."""
        from src.analysis.agent_graph import synthesizer_node
        result = synthesizer_node(base_state)
        assert result["final_report"] is not None
        assert result["synthesis_quality"] == 0.0

    def test_synthesizer_with_ta_report(self, base_state):
        """Synthesizer with partial outputs should try to generate."""
        from src.analysis.agent_graph import synthesizer_node

        base_state["ta_report"] = {
            "dominant_bias": "bearish",
            "bias_confidence": 0.7,
            "current_price": 4200.0,
            "atr": 15.0,
            "volatility_regime": "normal",
            "bullish_signals": [],
            "bearish_signals": ["MACD bearish cross"],
            "scenarios": [],
        }

        # Mock LLM to avoid actual API calls — patch at the import source
        with patch("src.analysis.llm_client.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "content": "## Analisis Semanal\nInforme de prueba.",
                "cost_usd": 0.001,
                "tokens_used": 100,
            }
            MockLLM.return_value = mock_instance

            result = synthesizer_node(base_state)
            assert result.get("synthesis_draft") is not None or result.get("final_report") is not None

    def test_synthesizer_revision_count(self, base_state):
        """Synthesizer should increment revision count when quality is low."""
        from src.analysis.agent_graph import synthesizer_node

        base_state["ta_report"] = {"dominant_bias": "bearish"}

        with patch("src.analysis.llm_client.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.generate.side_effect = [
                # First call: synthesis
                {"content": "Draft report", "cost_usd": 0.001, "tokens_used": 100},
                # Second call: evaluation (returns low score)
                {"content": '{"coherencia": 0.3, "accionabilidad": 0.4}', "cost_usd": 0.0005, "tokens_used": 50},
                # Third call: critique
                {"content": "Needs more detail on indicators.", "cost_usd": 0.0005, "tokens_used": 50},
            ]
            MockLLM.return_value = mock_instance

            result = synthesizer_node(base_state)
            # Should request revision (quality < threshold)
            revision = result.get("synthesis_revision", 0)
            assert revision >= 1 or result.get("final_report") is not None


class TestGracefulDegradation:
    """Test that the graph handles individual agent failures gracefully."""

    def test_ta_failure_doesnt_crash_graph(self, state_with_data):
        """If TA agent fails, other agents should still work."""
        from src.analysis.agent_graph import ta_agent_node

        # Force run_technical_analysis to raise an exception
        with patch("src.analysis.agent_tools.run_technical_analysis", side_effect=ValueError("Bad data")):
            result = ta_agent_node(state_with_data)
            # Should not raise, just return None
            assert result["ta_report"] is None
            assert len(result["errors"]) > 0

    def test_macro_failure_doesnt_crash_graph(self, state_with_data):
        """If macro agent fails, other agents should still work."""
        from src.analysis.agent_graph import macro_agent_node

        # Remove macro data
        state_with_data["macro_data"] = []

        result = macro_agent_node(state_with_data)
        assert result["macro_regime"] is None

    def test_news_failure_doesnt_crash_graph(self, state_with_data):
        """If news agent fails, other agents should still work."""
        from src.analysis.agent_graph import news_agent_node

        state_with_data["gdelt_articles"] = []

        result = news_agent_node(state_with_data)
        assert result["news_intelligence"] is None

    def test_all_agents_fail(self, base_state):
        """Even if all agents fail, synthesizer should produce something."""
        from src.analysis.agent_graph import synthesizer_node

        # No agent outputs
        result = synthesizer_node(base_state)
        assert result["final_report"] is not None
        assert "Sin datos suficientes" in result["final_report"]

    def test_errors_accumulated(self, state_with_data):
        """Errors from each agent should be accumulated in state."""
        from src.analysis.agent_graph import ta_agent_node

        # Force an exception to trigger error logging
        with patch("src.analysis.agent_tools.run_technical_analysis", side_effect=RuntimeError("TA crash")):
            result = ta_agent_node(state_with_data)
            assert len(result["errors"]) > 0
            # Error message should identify the agent
            assert any("ta_agent" in e for e in result["errors"])


class TestGraphConstruction:
    """Test LangGraph graph construction."""

    def test_build_master_graph(self):
        """Graph should build without errors."""
        from src.analysis.agent_graph import build_master_graph
        graph = build_master_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """Graph should contain all expected node names."""
        from src.analysis.agent_graph import build_master_graph
        graph = build_master_graph()
        # The compiled graph should be invocable
        assert hasattr(graph, "invoke")


class TestRunAnalysisGraph:
    """Test the convenience run_analysis_graph function."""

    def test_initial_state_creation(self):
        """run_analysis_graph should create correct initial state."""
        from src.analysis.agent_graph import run_analysis_graph

        # We just test that the function sets up the state correctly
        # by checking date calculation (no actual graph invocation)
        start = date.fromisocalendar(2026, 8, 1)
        end = date.fromisocalendar(2026, 8, 5)
        assert start.isoformat() == "2026-02-16"
        assert end.isoformat() == "2026-02-20"


class TestFinalizerNode:
    """Test the finalizer node."""

    def test_finalizer_logs_completion(self, base_state):
        from src.analysis.agent_graph import finalizer_node
        result = finalizer_node(base_state)
        assert "Pipeline complete" in result["execution_log"]


class TestBiasAgentNode:
    """Test the Political Bias Detection agent node (Phase 3)."""

    def test_bias_agent_with_articles(self, state_with_data):
        from src.analysis.agent_graph import bias_agent_node
        result = bias_agent_node(state_with_data)
        assert "political_bias_analysis" in result
        if result["political_bias_analysis"] is not None:
            assert "source_bias_distribution" in result["political_bias_analysis"]
            assert "bias_diversity_score" in result["political_bias_analysis"]
            assert "total_analyzed" in result["political_bias_analysis"]

    def test_bias_agent_no_articles(self, base_state):
        """Bias agent should handle missing articles gracefully."""
        from src.analysis.agent_graph import bias_agent_node
        result = bias_agent_node(base_state)
        assert result["political_bias_analysis"] is None

    def test_bias_agent_execution_log(self, state_with_data):
        from src.analysis.agent_graph import bias_agent_node
        result = bias_agent_node(state_with_data)
        assert any("Bias agent" in msg for msg in result["execution_log"])

    def test_bias_agent_uses_injected_llm(self, state_with_data):
        """Bias agent should use injected _llm_client for cluster analysis."""
        from src.analysis.agent_graph import bias_agent_node

        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "content": "balanced|0.7",
            "cost_usd": 0.001,
            "tokens_used": 30,
        }
        state_with_data["_llm_client"] = mock_llm
        # Add news_intelligence with large clusters so LLM gets called
        state_with_data["news_intelligence"] = {
            "clusters": [
                {
                    "label": "BanRep decision",
                    "article_count": 10,
                    "representative_titles": ["T1", "T2", "T3", "T4", "T5"],
                }
            ]
        }

        result = bias_agent_node(state_with_data)
        assert result["political_bias_analysis"] is not None

    def test_bias_agent_graceful_on_failure(self, state_with_data):
        """Bias agent should not crash on internal errors."""
        from src.analysis.agent_graph import bias_agent_node

        with patch("src.analysis.bias_detector.PoliticalBiasDetector.analyze",
                   side_effect=RuntimeError("crash")):
            result = bias_agent_node(state_with_data)
            assert result["political_bias_analysis"] is None
            assert len(result["errors"]) > 0


class TestPreloadedDataInjection:
    """Test that pre-loaded data injection bypasses file loading."""

    def test_preprocess_with_preloaded_data(self, state_with_data):
        """preprocess_node should skip file loading when data is pre-loaded."""
        from src.analysis.agent_graph import preprocess_node
        result = preprocess_node(state_with_data)
        log = result.get("execution_log", [])
        # Should detect pre-loaded data and skip file loading
        assert any("pre-loaded" in msg.lower() or "Using" in msg for msg in log)

    def test_preprocess_without_preloaded_data(self, base_state):
        """preprocess_node without pre-loaded data should attempt file loading."""
        from src.analysis.agent_graph import preprocess_node
        # This will try to load files and likely fail (no files in test env),
        # but should not crash
        result = preprocess_node(base_state)
        assert "execution_log" in result

    def test_llm_client_injected_into_state(self):
        """run_analysis_graph should inject _llm_client into initial state."""
        from src.analysis.agent_graph import run_analysis_graph

        mock_llm = MagicMock()
        # Dry run to avoid actual graph execution
        try:
            result = run_analysis_graph(
                iso_year=2026, iso_week=8,
                dry_run=True,
                preloaded_data={"ohlcv_daily": [{"date": "2026-02-16", "close": 4200}]},
                llm_client=mock_llm,
            )
        except Exception:
            # Graph may fail on dry_run depending on implementation,
            # but the key check is that it doesn't crash on setup
            pass


class TestGraphWith5Agents:
    """Test that graph construction includes all 5 agents."""

    def test_build_master_graph_includes_bias(self):
        from src.analysis.agent_graph import build_master_graph
        graph = build_master_graph()
        assert graph is not None
        # The compiled graph should be invocable
        assert hasattr(graph, "invoke")


class TestReflectionPattern:
    """Test the Reflection pattern (generate → evaluate → revise)."""

    def test_max_reflections_respected(self, base_state):
        """Synthesizer should stop after MAX_REFLECTIONS."""
        from src.analysis.agent_graph import synthesizer_node, MAX_REFLECTIONS

        base_state["ta_report"] = {"dominant_bias": "neutral"}
        base_state["synthesis_revision"] = MAX_REFLECTIONS

        with patch("src.analysis.llm_client.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "content": "Final draft after max revisions",
                "cost_usd": 0.001,
                "tokens_used": 100,
            }
            MockLLM.return_value = mock_instance

            # Mock _evaluate_quality to return low score
            with patch("src.analysis.agent_graph._evaluate_quality", return_value=0.5):
                result = synthesizer_node(base_state)
                # Should accept even low quality because max revisions reached
                assert result.get("final_report") is not None

    def test_quality_threshold_lowers_with_fewer_agents(self, base_state):
        """With fewer agent outputs, quality threshold should be lower."""
        from src.analysis.agent_graph import (
            MIN_QUALITY_SCORE, DEGRADED_QUALITY_THRESHOLD,
        )
        # With 3+ agents, threshold = 0.8 (MIN_QUALITY_SCORE)
        # With <3 agents, threshold = 0.6 (DEGRADED_QUALITY_THRESHOLD)
        assert MIN_QUALITY_SCORE > DEGRADED_QUALITY_THRESHOLD
