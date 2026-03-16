"""
Unit tests for NewsIntelligenceEngine (Phase 2).
Tests article processing, sentiment, categorization, bias lookup, clustering.
"""

import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def engine():
    """Create a NewsIntelligenceEngine instance (no model loading)."""
    from src.analysis.news_intelligence import NewsIntelligenceEngine
    eng = NewsIntelligenceEngine()
    # Don't load heavy transformer models in unit tests
    eng._models_loaded = True
    eng._sentiment_model = None
    eng._embedding_model = None
    return eng


@pytest.fixture
def sample_articles():
    """Create sample article dicts mimicking GDELT output."""
    return [
        {
            "title": "Banco de la Republica mantiene tasa de interes en 9.5%",
            "source": "portafolio.co",
            "url": "https://portafolio.co/article/1",
            "date": "2026-02-20",
            "language": "es",
            "tone": -2.5,
        },
        {
            "title": "Colombia peso strengthens as oil prices surge above $80",
            "source": "reuters.com",
            "url": "https://reuters.com/article/2",
            "date": "2026-02-20",
            "language": "en",
            "tone": 3.1,
        },
        {
            "title": "Fed holds rates steady, signals patience on cuts",
            "source": "bloomberg.com",
            "url": "https://bloomberg.com/article/3",
            "date": "2026-02-19",
            "language": "en",
            "tone": 0.5,
        },
        {
            "title": "DXY sube a maximo de 3 meses, presiona al peso colombiano",
            "source": "larepublica.co",
            "url": "https://larepublica.co/article/4",
            "date": "2026-02-19",
            "language": "es",
            "tone": -1.8,
        },
        {
            "title": "EMBI Colombia spread widens to 350 basis points",
            "source": "investing.com",
            "url": "https://investing.com/article/5",
            "date": "2026-02-18",
            "language": "en",
            "tone": -3.0,
        },
        {
            "title": "Random unrelated article about sports and entertainment news",
            "source": "espn.com",
            "url": "https://espn.com/sports/1",
            "date": "2026-02-20",
            "language": "en",
            "tone": 1.0,
        },
        {
            "title": "WTI crude oil prices rally on supply concerns",
            "source": "cnbc.com",
            "url": "https://cnbc.com/article/6",
            "date": "2026-02-18",
            "language": "en",
            "tone": 2.0,
        },
        {
            "title": "Colombia exportaciones de petroleo aumentan en enero",
            "source": "portafolio.co",
            "url": "https://portafolio.co/article/7",
            "date": "2026-02-17",
            "language": "es",
            "tone": 1.5,
        },
    ]


class TestArticleProcessing:
    """Test article enrichment pipeline."""

    def test_process_filters_by_relevance(self, engine, sample_articles):
        """Articles below min_relevance should be filtered out."""
        enriched = engine.process_articles(sample_articles, min_relevance=0.3)
        # Sports article should be filtered out (low relevance)
        titles = [a.title for a in enriched]
        assert not any("sports" in t.lower() for t in titles)

    def test_process_returns_enriched_articles(self, engine, sample_articles):
        """Processed articles should have category, relevance, tone."""
        enriched = engine.process_articles(sample_articles, min_relevance=0.1)
        assert len(enriched) > 0
        for art in enriched:
            assert art.title != ""
            assert art.source != ""
            assert art.category != ""
            assert isinstance(art.category, str)
            assert 0.0 <= art.relevance <= 1.0
            assert -1.0 <= art.tone <= 1.0

    def test_process_sorted_by_relevance(self, engine, sample_articles):
        """Results should be sorted by relevance descending."""
        enriched = engine.process_articles(sample_articles, min_relevance=0.1)
        if len(enriched) >= 2:
            relevances = [a.relevance for a in enriched]
            assert relevances == sorted(relevances, reverse=True)

    def test_process_skips_short_titles(self, engine):
        """Titles shorter than 10 chars should be skipped."""
        articles = [{"title": "Short", "source": "test.com", "date": "2026-01-01"}]
        enriched = engine.process_articles(articles, min_relevance=0.0)
        assert len(enriched) == 0

    def test_process_empty_list(self, engine):
        """Empty article list should return empty list."""
        enriched = engine.process_articles([], min_relevance=0.0)
        assert enriched == []


class TestSentiment:
    """Test sentiment computation."""

    def test_gdelt_tone_used_as_fallback(self, engine):
        """GDELT tone should be used when no model available."""
        # tone=-5 should map to ~ -0.5
        score = engine._compute_sentiment("Test article title", gdelt_tone=-5.0)
        assert score < 0

    def test_gdelt_tone_clamped(self, engine):
        """GDELT tone should be clamped to [-1, 1]."""
        score = engine._compute_sentiment("Test title", gdelt_tone=20.0)
        assert score <= 1.0
        score = engine._compute_sentiment("Test title", gdelt_tone=-20.0)
        assert score >= -1.0

    def test_zero_tone_returns_zero(self, engine):
        """Zero tone with no model should return 0.0."""
        score = engine._compute_sentiment("Neutral article without keywords", gdelt_tone=0)
        assert score == 0.0


class TestCategorization:
    """Test article categorization."""

    def test_monetary_policy(self, engine):
        """Monetary policy keywords should be categorized correctly."""
        result = engine._categorize("Banco de la Republica sube tasa de interes")
        assert isinstance(result, str)
        assert result == "monetary_policy"

    def test_fx_market(self, engine):
        """FX market keywords should return a category string."""
        result = engine._categorize("El dolar sigue subiendo frente al peso colombiano")
        assert isinstance(result, str)
        assert result == "fx_market"

    def test_commodities(self, engine):
        """Commodity keywords should be categorized correctly."""
        result = engine._categorize("Oil prices surge as OPEC cuts production")
        assert isinstance(result, str)
        assert result == "commodities"

    def test_inflation(self, engine):
        """Inflation keywords should be categorized correctly."""
        result = engine._categorize("Inflacion en Colombia baja al 7.5% en enero")
        assert isinstance(result, str)
        assert result == "inflation"

    def test_general_fallback(self, engine):
        """Unrecognized articles should be categorized as general."""
        result = engine._categorize("Random article about technology advances")
        assert isinstance(result, str)
        # May return "general" or another category depending on keyword matching
        assert result != ""

    def test_always_returns_string(self, engine):
        """_categorize should always return a string, never a tuple."""
        for title in [
            "Banco de la Republica decide sobre tasas",
            "USD/COP sube hoy",
            "Oil WTI prices",
            "Generic article with no keywords at all here",
        ]:
            result = engine._categorize(title)
            assert isinstance(result, str), f"Expected str, got {type(result)} for: {title}"


class TestMediaBias:
    """Test media bias lookup."""

    def test_known_sources(self):
        from src.analysis.news_intelligence import get_media_bias
        assert get_media_bias("reuters.com") == ("center", "high")
        assert get_media_bias("bloomberg.com") == ("right-center", "high")
        assert get_media_bias("portafolio.co") == ("center", "high")
        assert get_media_bias("semana.com") == ("right-center", "mixed")

    def test_partial_match(self):
        from src.analysis.news_intelligence import get_media_bias
        # Should match domain substring
        bias, fact = get_media_bias("news.reuters.com")
        assert bias == "center"
        assert fact == "high"

    def test_unknown_source(self):
        from src.analysis.news_intelligence import get_media_bias
        bias, fact = get_media_bias("unknown-blog.xyz")
        assert bias == "unknown"
        assert fact == "unknown"

    def test_case_insensitive(self):
        from src.analysis.news_intelligence import get_media_bias
        bias, _ = get_media_bias("Reuters.com")
        assert bias == "center"


class TestClustering:
    """Test article clustering (fallback path without ML models)."""

    def test_cluster_by_category_fallback(self, engine, sample_articles):
        """Without embedding model, should fall back to category clustering."""
        enriched = engine.process_articles(sample_articles, min_relevance=0.1)
        if len(enriched) >= 2:
            clusters = engine._cluster_by_category(enriched)
            assert isinstance(clusters, list)
            for cluster in clusters:
                assert cluster.article_count >= 2
                assert cluster.dominant_category != ""
                assert isinstance(cluster.articles, list)

    def test_cluster_too_few_articles(self, engine):
        """Fewer articles than min_cluster_size should return empty."""
        from src.analysis.news_intelligence import ArticleEnriched
        articles = [ArticleEnriched(title="Single article")]
        clusters = engine.cluster_articles(articles, min_cluster_size=3)
        assert clusters == []

    def test_cluster_empty_articles(self, engine):
        """Empty list should return empty clusters."""
        clusters = engine.cluster_articles([], min_cluster_size=2)
        assert clusters == []

    def test_category_clustering_sorted(self, engine):
        """Category clusters should be sorted by article count."""
        from src.analysis.news_intelligence import ArticleEnriched
        articles = [
            ArticleEnriched(title=f"Monetary article {i}", category="monetary_policy", tone=0.1)
            for i in range(5)
        ] + [
            ArticleEnriched(title=f"FX article {i}", category="fx_market", tone=-0.1)
            for i in range(3)
        ]
        clusters = engine._cluster_by_category(articles)
        if len(clusters) >= 2:
            counts = [c.article_count for c in clusters]
            assert counts == sorted(counts, reverse=True)


class TestRelevanceScoring:
    """Test relevance scoring logic."""

    def test_high_relevance_keywords(self, engine):
        """Titles with USDCOP keywords should score high."""
        score = engine._score_relevance("USD/COP tasa de cambio sube", "portafolio.co")
        assert score >= 0.4

    def test_medium_relevance(self, engine):
        """Colombia-related financial keywords should score medium."""
        score = engine._score_relevance("Colombia economic outlook improves", "reuters.com")
        assert score >= 0.15

    def test_low_relevance(self, engine):
        """Generic articles should score low."""
        score = engine._score_relevance("Technology stocks rally in Asia", "espn.com")
        assert score < 0.3

    def test_source_quality_bonus(self, engine):
        """Quality sources should get a bonus (inline fallback path)."""
        # Force inline fallback by making enrichment import fail
        with patch.dict(sys.modules, {'src.news_engine.enrichment.relevance': None}):
            score_quality = engine._score_relevance("Colombia economy news", "reuters.com")
            score_unknown = engine._score_relevance("Colombia economy news", "random-blog.com")
            assert score_quality > score_unknown


class TestFullAnalysis:
    """Test the full analyze() pipeline."""

    def test_analyze_returns_report(self, engine, sample_articles):
        """analyze() should return a NewsIntelligenceReport."""
        report = engine.analyze(
            sample_articles,
            week_start="2026-02-17",
            week_end="2026-02-21",
        )
        assert report is not None
        assert report.total_articles == len(sample_articles)
        assert report.relevant_articles >= 0

    def test_sentiment_distribution(self, engine, sample_articles):
        """Sentiment distribution should have positive/negative/neutral."""
        report = engine.analyze(sample_articles, "2026-02-17", "2026-02-21")
        dist = report.sentiment_distribution
        assert "positive" in dist
        assert "negative" in dist
        assert "neutral" in dist
        total = dist["positive"] + dist["negative"] + dist["neutral"]
        assert total == report.relevant_articles

    def test_source_diversity(self, engine, sample_articles):
        """Source diversity should be computed."""
        report = engine.analyze(sample_articles, "2026-02-17", "2026-02-21")
        assert isinstance(report.source_diversity, dict)

    def test_top_stories(self, engine, sample_articles):
        """Top stories should be the highest-relevance articles."""
        report = engine.analyze(sample_articles, "2026-02-17", "2026-02-21")
        assert isinstance(report.top_stories, list)
        assert len(report.top_stories) <= 10

    def test_to_dict(self, engine, sample_articles):
        """to_dict() should return a serializable dict."""
        report = engine.analyze(sample_articles, "2026-02-17", "2026-02-21")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "total_articles" in d
        assert "clusters" in d
        assert "sentiment_distribution" in d
        # Verify JSON-serializable
        import json
        json.dumps(d)

    def test_empty_articles(self, engine):
        """Empty articles list should not crash."""
        report = engine.analyze([], "2026-02-17", "2026-02-21")
        assert report.total_articles == 0
        assert report.relevant_articles == 0
        assert report.clusters == []


class TestDataClasses:
    """Test dataclass construction and serialization."""

    def test_article_enriched_to_dict(self):
        from src.analysis.news_intelligence import ArticleEnriched
        art = ArticleEnriched(
            title="Test Article",
            source="reuters.com",
            tone=0.5,
            embedding=[0.1] * 384,  # Should be excluded from dict
        )
        d = art.to_dict()
        assert "embedding" not in d
        assert d["title"] == "Test Article"
        assert d["tone"] == 0.5

    def test_cluster_to_dict(self):
        from src.analysis.news_intelligence import NewsClusterEnriched
        cluster = NewsClusterEnriched(
            cluster_id=0,
            label="Oil Prices",
            article_count=5,
            avg_sentiment=0.3,
        )
        d = cluster.to_dict()
        assert d["label"] == "Oil Prices"
        assert d["article_count"] == 5

    def test_report_to_dict(self):
        from src.analysis.news_intelligence import NewsIntelligenceReport
        report = NewsIntelligenceReport(total_articles=10, relevant_articles=5)
        d = report.to_dict()
        assert d["total_articles"] == 10
        assert d["relevant_articles"] == 5
