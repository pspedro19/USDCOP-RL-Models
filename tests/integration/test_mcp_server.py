"""
Integration tests for MCP News Server (Phase 4)
================================================
Tests the data layer CSV fallback, relevance scoring, and tool formatting.
Does NOT require a running PostgreSQL instance or the mcp package.
"""

import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal GDELT-like CSV for testing."""
    data = {
        "title": [
            "Colombia peso strengthens as oil prices rise",
            "Fed holds interest rates steady amid inflation concerns",
            "Dollar index DXY hits three-month high",
            "BanRep mantiene tasa de interés en Colombia",
            "Oil prices drop as OPEC signals output increase",
            "VIX volatility index spikes on trade war fears",
            "EMBI spread widens for emerging markets",
            "Random unrelated article about sports",
            "Petróleo WTI sube por tensiones geopolíticas",
            "Tasa de cambio dólar peso colombiano hoy",
        ],
        "source": [
            "reuters.com", "bloomberg.com", "cnbc.com", "portafolio.co",
            "ft.com", "cnn.com", "bloomberg.com", "espn.com",
            "larepublica.co", "eltiempo.com",
        ],
        "domain": [
            "reuters.com", "bloomberg.com", "cnbc.com", "portafolio.co",
            "ft.com", "cnn.com", "bloomberg.com", "espn.com",
            "larepublica.co", "eltiempo.com",
        ],
        "date": [
            "2026-03-01T10:00:00+00:00", "2026-03-01T12:00:00+00:00",
            "2026-03-02T09:00:00+00:00", "2026-03-02T14:00:00+00:00",
            "2026-03-03T08:00:00+00:00", "2026-03-03T15:00:00+00:00",
            "2026-03-04T11:00:00+00:00", "2026-03-04T16:00:00+00:00",
            "2026-03-05T07:00:00+00:00", "2026-03-05T13:00:00+00:00",
        ],
        "tone": [2.0, -0.5, 1.0, -1.5, -2.0, 3.0, -1.0, 0.0, 0.5, -0.3],
        "language": ["en", "en", "en", "es", "en", "en", "en", "en", "es", "es"],
        "category": [
            "commodities", "monetary_policy", "fx_market", "monetary_policy",
            "commodities", "risk_premium", "risk_premium", "general",
            "commodities", "fx_market",
        ],
    }
    csv_path = tmp_path / "gdelt_articles_historical.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def csv_layer(sample_csv):
    """Create a NewsDataLayer loaded with CSV data (no PG, no async connect)."""
    from src.news_engine.mcp_data_layer import NewsDataLayer

    layer = NewsDataLayer()
    layer._csv_df = pd.read_csv(sample_csv, low_memory=False)
    for col in ("date", "published_at", "seendate"):
        if col in layer._csv_df.columns:
            layer._csv_df["_date"] = pd.to_datetime(
                layer._csv_df[col], errors="coerce", utc=True
            )
            break
    if "title" not in layer._csv_df.columns:
        layer._csv_df["title"] = ""
    return layer


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True, scope="module")
def _ensure_event_loop():
    """Ensure an event loop exists for the test module."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# ---------------------------------------------------------------------------
# Relevance scoring tests
# ---------------------------------------------------------------------------

class TestRelevanceScoring:
    def test_tier_3_keywords(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        score = _compute_relevance("Tasa de cambio dólar peso colombiano USDCOP", "es")
        assert score >= 9.0

    def test_tier_2_keywords(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        score = _compute_relevance("Fed raises tasas de interés amid inflación concerns")
        assert score >= 4.0

    def test_tier_1_keywords(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        score = _compute_relevance("Gold commodities rally as DXY weakens")
        assert score >= 2.0

    def test_spanish_boost(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        en_score = _compute_relevance("Oil prices rise", "en")
        es_score = _compute_relevance("Oil prices rise", "es")
        assert es_score == en_score + 0.5

    def test_empty_title(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        assert _compute_relevance("") == 0.0
        assert _compute_relevance("", "es") == 0.0

    def test_no_keywords(self):
        from src.news_engine.mcp_data_layer import _compute_relevance

        score = _compute_relevance("Local sports team wins championship")
        assert score == 0.0

    def test_3_tier_weights_defined(self):
        from src.news_engine.mcp_data_layer import RELEVANCE_KEYWORDS

        assert 3.0 in RELEVANCE_KEYWORDS
        assert 2.0 in RELEVANCE_KEYWORDS
        assert 1.0 in RELEVANCE_KEYWORDS


class TestValidCategories:
    def test_has_9_categories(self):
        from src.news_engine.mcp_data_layer import VALID_CATEGORIES

        assert len(VALID_CATEGORIES) == 9

    def test_expected_categories(self):
        from src.news_engine.mcp_data_layer import VALID_CATEGORIES

        expected = {
            "monetary_policy", "fx_market", "commodities", "inflation",
            "fiscal_policy", "risk_premium", "capital_flows", "political", "general",
        }
        assert set(VALID_CATEGORIES) == expected


# ---------------------------------------------------------------------------
# CSV Data Layer tests
# ---------------------------------------------------------------------------

class TestNewsDataLayerCSV:
    """Test the CSV fallback of NewsDataLayer."""

    def test_not_pg(self, csv_layer):
        assert not csv_layer.is_pg

    def test_search_keyword(self, csv_layer):
        results = _run(csv_layer.search("oil"))
        assert len(results) >= 1
        assert any("oil" in str(r.get("title", "")).lower() for r in results)

    def test_search_no_results(self, csv_layer):
        results = _run(csv_layer.search("cryptocurrency bitcoin"))
        assert len(results) == 0

    def test_search_with_date_filter(self, csv_layer):
        results = _run(
            csv_layer.search("", date_from="2026-03-03", date_to="2026-03-05")
        )
        assert len(results) >= 1

    def test_search_limit(self, csv_layer):
        results = _run(csv_layer.search("", limit=3))
        assert len(results) <= 3

    def test_top_headlines(self, csv_layer):
        results = _run(csv_layer.top_headlines(limit=5, min_relevance=0.0))
        assert len(results) <= 5

    def test_top_headlines_relevance_filter(self, csv_layer):
        results = _run(csv_layer.top_headlines(min_relevance=5.0))
        for r in results:
            rel = r.get("relevance", r.get("_relevance", 0)) or 0
            assert rel >= 5.0

    def test_by_category(self, csv_layer):
        results = _run(csv_layer.by_category("commodities"))
        assert len(results) >= 1
        for r in results:
            assert str(r.get("category", "")).lower() == "commodities"

    def test_source_stats(self, csv_layer):
        stats = _run(csv_layer.source_stats())
        assert stats["total_articles"] == 10
        assert isinstance(stats["by_source"], list)
        assert isinstance(stats["by_category"], list)

    def test_daily_briefing(self, csv_layer):
        briefing = _run(csv_layer.daily_briefing("2026-03-01"))
        assert briefing["date"] == "2026-03-01"
        assert "top_headlines" in briefing
        assert "by_category" in briefing

    def test_weekly_summary(self, csv_layer):
        summary = _run(
            csv_layer.weekly_summary("2026-03-01", "2026-03-05", headlines_per_day=3)
        )
        assert summary["week_start"] == "2026-03-01"
        assert summary["week_end"] == "2026-03-05"
        assert "prompt_injection_text" in summary
        assert len(summary["prompt_injection_text"]) > 0
        assert "daily_summaries" in summary

    def test_raw_query_raises_on_csv(self, csv_layer):
        with pytest.raises(RuntimeError, match="PostgreSQL"):
            _run(csv_layer.raw_query("SELECT 1"))


# ---------------------------------------------------------------------------
# Tool formatting tests (skip if mcp not installed)
# ---------------------------------------------------------------------------

try:
    from src.news_engine.mcp_server import _format_articles, _format_stats, _safe_val
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


@pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
class TestToolFormatting:
    def test_format_articles_empty(self):
        result = _format_articles([], "Test Header")
        assert "No articles found" in result
        assert "Test Header" in result

    def test_format_articles_with_data(self):
        articles = [
            {"title": "Test Article", "source": "reuters.com", "date": "2026-03-01", "tone": 1.5},
        ]
        result = _format_articles(articles, "Results")
        assert "Test Article" in result
        assert "reuters.com" in result
        assert "1 article(s)" in result

    def test_format_stats(self):
        stats = {
            "total_articles": 100,
            "avg_tone": 0.5,
            "date_range": {"earliest": "2026-01-01", "latest": "2026-03-01"},
            "by_source": [{"source": "reuters", "article_count": 50}],
            "by_category": [{"category": "fx_market", "article_count": 30}],
        }
        result = _format_stats(stats)
        assert "100" in result
        assert "reuters" in result
        assert "fx_market" in result

    def test_safe_val(self):
        assert _safe_val(None) is None
        assert _safe_val(42) == 42
        assert _safe_val(3.14) == 3.14
        assert _safe_val("text") == "text"
        assert _safe_val(True) is True

        from datetime import datetime
        assert isinstance(_safe_val(datetime.now()), str)


# ---------------------------------------------------------------------------
# SQL safety tests
# ---------------------------------------------------------------------------

class TestRawQuerySafety:
    def test_blocks_insert(self):
        from src.news_engine.mcp_data_layer import NewsDataLayer

        layer = NewsDataLayer()
        layer._pg_pool = MagicMock()

        with pytest.raises(ValueError, match="INSERT"):
            _run(layer.raw_query("INSERT INTO news_articles VALUES (1)"))

    def test_blocks_delete(self):
        from src.news_engine.mcp_data_layer import NewsDataLayer

        layer = NewsDataLayer()
        layer._pg_pool = MagicMock()

        with pytest.raises(ValueError, match="DELETE"):
            _run(layer.raw_query("DELETE FROM news_articles"))

    def test_blocks_drop(self):
        from src.news_engine.mcp_data_layer import NewsDataLayer

        layer = NewsDataLayer()
        layer._pg_pool = MagicMock()

        with pytest.raises(ValueError, match="DROP"):
            _run(layer.raw_query("DROP TABLE news_articles"))

    def test_blocks_update(self):
        from src.news_engine.mcp_data_layer import NewsDataLayer

        layer = NewsDataLayer()
        layer._pg_pool = MagicMock()

        with pytest.raises(ValueError, match="UPDATE"):
            _run(layer.raw_query("UPDATE news_articles SET title='x'"))

    def test_blocks_truncate(self):
        from src.news_engine.mcp_data_layer import NewsDataLayer

        layer = NewsDataLayer()
        layer._pg_pool = MagicMock()

        with pytest.raises(ValueError, match="TRUNCATE"):
            _run(layer.raw_query("TRUNCATE news_articles"))
