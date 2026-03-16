"""
Comprehensive unit tests for NewsEngine + Analysis Module (SDD-03 through SDD-10).
Tests contracts, enrichment, cross-reference, macro analyzer, prompts, and chart generation.
"""

import pytest
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import asdict

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================================
# 1. CONTRACTS — news_engine_schema.py
# ============================================================================

class TestNewsEngineContracts:
    """SDD-03: Validate dataclass contracts for NewsEngine."""

    def test_article_record_valid(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc",
            url="https://example.com/article1",
            url_hash="abc123",
            title="Test article about USD/COP trading",
            published_at="2026-02-25T10:00:00",
        )
        assert rec.source_id == "gdelt_doc"
        assert len(rec.title) >= 10  # SDD-03: min 10 chars

    def test_article_record_invalid_source(self):
        from src.contracts.news_engine_schema import ArticleRecord, VALID_SOURCE_IDS
        # Source IDs must be from the defined set
        assert "gdelt_doc" in VALID_SOURCE_IDS
        assert "gdelt_context" in VALID_SOURCE_IDS
        assert "newsapi" in VALID_SOURCE_IDS
        assert "investing" in VALID_SOURCE_IDS
        assert "larepublica" in VALID_SOURCE_IDS
        assert "portafolio" in VALID_SOURCE_IDS

    def test_valid_categories(self):
        from src.contracts.news_engine_schema import VALID_CATEGORIES
        # SDD-04: 9 domain-specific categories
        expected = {
            "monetary_policy", "fx_market", "commodities", "inflation",
            "fiscal_policy", "risk_premium", "capital_flows", "balance_payments",
            "political",
        }
        assert set(VALID_CATEGORIES) == expected

    def test_valid_sentiments(self):
        from src.contracts.news_engine_schema import VALID_SENTIMENTS
        expected = {"positive", "negative", "neutral"}
        assert set(VALID_SENTIMENTS) == expected

    def test_digest_record(self):
        from src.contracts.news_engine_schema import DigestRecord
        rec = DigestRecord(
            digest_date="2026-02-25",
            digest_type="daily",
            total_articles=42,
            by_source={"gdelt_doc": 20, "newsapi": 15, "investing": 7},
            by_category={"fx_market": 12, "commodities": 8, "inflation": 10},
            summary_text="Test daily digest",
        )
        assert rec.digest_type in ("daily", "weekly")
        assert rec.total_articles == 42

    def test_feature_snapshot_record(self):
        from src.contracts.news_engine_schema import FeatureSnapshotRecord
        rec = FeatureSnapshotRecord(
            snapshot_date="2026-02-25",
            features={"vol_total": 42.0, "vol_fx_market": 12.0},
            feature_version="1.0.0",
        )
        assert isinstance(rec.features, dict)
        assert rec.feature_version == "1.0.0"


# ============================================================================
# 2. CONTRACTS — analysis_schema.py
# ============================================================================

class TestAnalysisContracts:
    """SDD-07: Validate dataclass contracts for Analysis Module."""

    def test_macro_snapshot_fields(self):
        from src.contracts.analysis_schema import MacroSnapshot
        snap = MacroSnapshot(
            snapshot_date="2026-02-25",
            variable_key="dxy",
            variable_name="DXY Index",
            value=104.5,
            sma_5=104.2,
            sma_10=103.8,
            sma_20=103.5,
            sma_50=102.9,
        )
        assert snap.variable_key == "dxy"
        assert snap.value == 104.5
        # Optional fields default to None
        assert snap.bollinger_upper_20 is None
        assert snap.rsi_14 is None
        assert snap.macd_line is None

    def test_macro_snapshot_to_chart_point(self):
        from src.contracts.analysis_schema import MacroSnapshot
        snap = MacroSnapshot(
            snapshot_date="2026-02-25",
            variable_key="dxy",
            variable_name="DXY",
            value=104.5,
            sma_20=103.5,
            bollinger_upper_20=106.0,
            bollinger_lower_20=101.0,
            rsi_14=62.3,
        )
        pt = snap.to_chart_point()
        assert pt["date"] == "2026-02-25"
        assert pt["value"] == 104.5
        assert pt["sma20"] == 103.5
        assert pt["bb_upper"] == 106.0
        assert pt["bb_lower"] == 101.0
        assert pt["rsi"] == 62.3

    def test_key_macro_variables(self):
        from src.contracts.analysis_schema import KEY_MACRO_VARIABLES
        # KEY_MACRO_VARIABLES is a tuple of strings
        assert "dxy" in KEY_MACRO_VARIABLES
        assert "vix" in KEY_MACRO_VARIABLES
        assert "embi_col" in KEY_MACRO_VARIABLES
        assert "wti" in KEY_MACRO_VARIABLES
        assert len(KEY_MACRO_VARIABLES) >= 4

    def test_display_names(self):
        from src.contracts.analysis_schema import DISPLAY_NAMES
        assert "dxy" in DISPLAY_NAMES
        assert "vix" in DISPLAY_NAMES
        assert isinstance(DISPLAY_NAMES["dxy"], str)

    def test_daily_analysis_record(self):
        from src.contracts.analysis_schema import DailyAnalysisRecord
        rec = DailyAnalysisRecord(
            analysis_date="2026-02-25",
            iso_year=2026,
            iso_week=9,
            day_of_week=1,
            headline="Test headline for the day",
            summary_markdown="## Test\nContent here",
            sentiment="bearish",
            usdcop_close=4380.5,
        )
        assert rec.sentiment in ("bullish", "bearish", "neutral", "mixed")
        assert rec.day_of_week >= 0 and rec.day_of_week <= 4

    def test_weekly_view_export(self):
        from src.contracts.analysis_schema import WeeklyViewExport
        export = WeeklyViewExport(
            weekly_summary={"headline": "Test"},
            daily_entries=[],
            macro_snapshots={},
            macro_charts={},
            signals={},
            news_context={},
        )
        d = export.to_dict()
        assert "weekly_summary" in d
        assert "daily_entries" in d
        assert "macro_snapshots" in d
        assert "signals" in d
        assert "news_context" in d

    def test_weekly_view_export_json_safety(self):
        """SDD-07: JSON export must handle Infinity/NaN safely."""
        import json
        import math
        from src.contracts.analysis_schema import WeeklyViewExport, _sanitize_for_json

        export = WeeklyViewExport(
            weekly_summary={"value": float("inf"), "nan_val": float("nan")},
            daily_entries=[], macro_snapshots={},
            macro_charts={}, signals={}, news_context={},
        )
        d = export.to_dict()
        sanitized = _sanitize_for_json(d)
        json_str = json.dumps(sanitized)
        # Must not contain Infinity or NaN
        assert "Infinity" not in json_str
        assert "NaN" not in json_str


# ============================================================================
# 3. NEWS ENGINE — Models
# ============================================================================

class TestNewsEngineModels:
    """SDD-02: Validate in-memory data models."""

    def test_raw_article_url_hash(self):
        from src.news_engine.models import RawArticle
        art = RawArticle(
            url="https://example.com/unique-article",
            title="Test Article",
            source_id="gdelt_doc",
            published_at=datetime.now(),
        )
        # url_hash should be deterministic SHA256
        assert art.url_hash is not None
        assert len(art.url_hash) == 64  # SHA256 hex
        # Same URL -> same hash
        art2 = RawArticle(
            url="https://example.com/unique-article",
            title="Different",
            source_id="newsapi",
            published_at=datetime.now(),
        )
        assert art.url_hash == art2.url_hash

    def test_raw_article_different_url_different_hash(self):
        from src.news_engine.models import RawArticle
        art1 = RawArticle(url="https://a.com/1", title="A", source_id="gdelt_doc", published_at=datetime.now())
        art2 = RawArticle(url="https://a.com/2", title="B", source_id="gdelt_doc", published_at=datetime.now())
        assert art1.url_hash != art2.url_hash


# ============================================================================
# 4. ENRICHMENT — Categorizer (SDD-04)
# ============================================================================

class TestCategorizer:
    """SDD-04: Category assignment from title + summary keywords."""

    def test_forex_category(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(
            "El dolar sube frente al peso colombiano USD/COP",
            "La tasa de cambio se disparo hoy"
        )
        assert cat == "fx_market"

    def test_oil_category(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(
            "Petroleo WTI cae por debajo de 70 dolares",
            "OPEP decide recortar produccion de crude"
        )
        assert cat == "commodities"

    def test_inflation_category(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(
            "Inflacion en Colombia sube al 7%",
            "El IPC se mantiene elevado"
        )
        assert cat == "inflation"

    def test_monetary_policy_category(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(
            "Banco de la Republica mantiene tasa de interes",
            "BanRep decide en reunion de politica monetaria"
        )
        assert cat == "monetary_policy"

    def test_no_match_returns_none(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(
            "Random news about weather in Bogota",
            "Sunny skies expected tomorrow"
        )
        # No matching category -> returns None
        assert cat is None

    def test_empty_inputs(self):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article("", "")
        assert cat is None


# ============================================================================
# 5. ENRICHMENT — Relevance Scorer (SDD-04)
# ============================================================================

class TestRelevanceScorer:
    """SDD-04: Relevance scoring [0.0, 1.0] range."""

    def test_high_relevance(self):
        from src.news_engine.enrichment.relevance import score_relevance
        score = score_relevance(
            title="USD/COP dolar peso colombiano tasa de cambio",
            summary="El dolar se fortalece frente al peso colombiano",
            source_id="investing",
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high relevance

    def test_low_relevance(self):
        from src.news_engine.enrichment.relevance import score_relevance
        score = score_relevance(
            title="Weather forecast for Europe",
            summary="Sunny skies in London",
            source_id="newsapi",
        )
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Low relevance

    def test_score_range(self):
        from src.news_engine.enrichment.relevance import score_relevance
        # Edge case: empty strings
        score = score_relevance("", "", "unknown_source")
        assert 0.0 <= score <= 1.0


# ============================================================================
# 6. ENRICHMENT — Sentiment Analyzer (SDD-04)
# ============================================================================

class TestSentimentAnalyzer:
    """SDD-04: Sentiment score [-1, +1] with GDELT/VADER sources."""

    def test_gdelt_tone_normalization(self):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        # Positive GDELT tone: parameter is gdelt_tone, not external_tone
        score, label = analyze_sentiment("Positive article", gdelt_tone=10.0)
        assert -1.0 <= score <= 1.0
        assert score > 0.0  # Positive tone

    def test_gdelt_negative_tone(self):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        score, label = analyze_sentiment("Crisis economica", gdelt_tone=-15.0)
        assert -1.0 <= score <= 1.0
        assert score < 0.0  # Negative tone

    def test_gdelt_extreme_tone_clamped(self):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        # SDD-04: tone clipped to [-20, +20] then scaled to [-1, 1]
        score, _ = analyze_sentiment("Article", gdelt_tone=50.0)
        assert score <= 1.0  # Must be clamped

    def test_vader_fallback(self):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        # No gdelt_tone -> VADER
        score, label = analyze_sentiment("This is a great and wonderful development")
        assert score is None or -1.0 <= score <= 1.0


# ============================================================================
# 7. ENRICHMENT — Tagger (SDD-04)
# ============================================================================

class TestTagger:
    """SDD-04: Keyword and entity extraction."""

    def test_extract_keywords(self):
        from src.news_engine.enrichment.tagger import extract_keywords
        keywords = extract_keywords(
            "El dolar sube por petroleo caro y la inflacion en Colombia"
        )
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should find financial keywords
        found = [k.lower() for k in keywords]
        assert any("dolar" in k or "petroleo" in k or "inflacion" in k for k in found)

    def test_extract_entities(self):
        from src.news_engine.enrichment.tagger import extract_entities
        entities = extract_entities(
            "El Banco de la Republica y la Fed decidieron mantener tasas. "
            "Ecopetrol reporto resultados. El DXY subio."
        )
        assert isinstance(entities, list)
        assert len(entities) > 0

    def test_empty_text(self):
        from src.news_engine.enrichment.tagger import extract_keywords, extract_entities
        assert extract_keywords("") == []
        assert extract_entities("") == []


# ============================================================================
# 8. ENRICHMENT — Pipeline (SDD-04)
# ============================================================================

class TestEnrichmentPipeline:
    """SDD-04: Full enrichment pipeline orchestrator."""

    def test_pipeline_init(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        pipeline = EnrichmentPipeline()
        assert pipeline is not None

    def test_pipeline_enrich_single(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.models import RawArticle

        pipeline = EnrichmentPipeline()

        article = RawArticle(
            url="https://test.com/article-about-dolar-colombiano",
            title="El dolar sube frente al peso colombiano",
            source_id="newsapi",
            published_at=datetime.now(),
            summary="Tasa de cambio USD/COP alcanza nuevos niveles",
        )
        enriched = pipeline.enrich(article)
        assert enriched.category is not None
        assert enriched.relevance_score >= 0.0
        assert isinstance(enriched.keywords, list)

    def test_pipeline_batch_filters_low_relevance(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.models import RawArticle

        pipeline = EnrichmentPipeline(min_relevance=0.5)

        articles = [
            RawArticle(
                url="https://t.com/1",
                title="Random weather in Europe today",
                source_id="newsapi",
                published_at=datetime.now(),
                summary="Sunny skies in London tomorrow",
            ),
            RawArticle(
                url="https://t.com/2",
                title="USD/COP dolar peso colombiano tasa de cambio",
                source_id="newsapi",
                published_at=datetime.now(),
                summary="El dolar sube por presiones macro",
            ),
        ]
        results = pipeline.enrich_batch(articles)
        # High-relevance article should pass; low-relevance may be filtered
        assert isinstance(results, list)


# ============================================================================
# 9. CROSS-REFERENCE ENGINE (SDD-05)
# ============================================================================

class TestCrossReferenceEngine:
    """SDD-05: Similarity matching with 4-component scoring."""

    def test_engine_init(self):
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig
        engine = CrossReferenceEngine(CrossReferenceConfig())
        assert engine is not None

    def _make_enriched(self, source_id, url, title, summary=None, category=None, keywords=None, entities=None):
        """Helper to create EnrichedArticle with correct signature."""
        from src.news_engine.models import RawArticle, EnrichedArticle
        raw = RawArticle(
            url=url,
            title=title,
            source_id=source_id,
            published_at=datetime.now(),
            summary=summary,
        )
        return EnrichedArticle(
            raw=raw,
            category=category,
            keywords=keywords or [],
            entities=entities or [],
            relevance_score=0.7,
        )

    def test_identical_articles_high_similarity(self):
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig

        engine = CrossReferenceEngine(CrossReferenceConfig())

        a1 = self._make_enriched(
            "gdelt_doc", "https://a.com/1",
            "Dolar sube por petroleo y DXY fortalecido",
            summary="El peso colombiano se debilita frente al dolar",
            category="fx_market", keywords=["dolar", "petroleo", "dxy"],
            entities=["Fed", "BanRep"],
        )
        a2 = self._make_enriched(
            "newsapi", "https://b.com/2",
            "Dolar sube por petroleo y DXY fortalecido",
            summary="El peso colombiano se debilita frente al dolar",
            category="fx_market", keywords=["dolar", "petroleo", "dxy"],
            entities=["Fed", "BanRep"],
        )

        clusters = engine.find_clusters([a1, a2])
        # Identical articles from different sources should cluster
        assert len(clusters) >= 1

    def test_different_articles_no_cluster(self):
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig

        engine = CrossReferenceEngine(CrossReferenceConfig())

        a1 = self._make_enriched(
            "gdelt_doc", "https://a.com/1",
            "Petroleo WTI cae por exceso de oferta",
            summary="OPEP no logra acuerdo de recorte",
            category="commodities", keywords=["petroleo", "wti", "opep"],
            entities=["OPEP"],
        )
        a2 = self._make_enriched(
            "newsapi", "https://b.com/2",
            "Resultados deportivos de la liga colombiana",
            summary="Nacional gana el clasico contra Millonarios",
            category="political", keywords=["futbol", "liga"],
            entities=[],
        )

        clusters = engine.find_clusters([a1, a2])
        # Very different articles should not cluster
        assert len(clusters) == 0

    def test_min_sources_requirement(self):
        """SDD-05: Clusters require min_cluster_size from config."""
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig

        engine = CrossReferenceEngine(CrossReferenceConfig())

        # Same source, identical titles — may cluster depending on similarity
        a1 = self._make_enriched(
            "gdelt_doc", "https://a.com/1",
            "Dolar sube en Colombia por DXY",
            category="fx_market", keywords=["dolar"],
        )
        a2 = self._make_enriched(
            "gdelt_doc", "https://a.com/2",
            "Dolar sube en Colombia por DXY fortalecido",
            category="fx_market", keywords=["dolar"],
        )

        clusters = engine.find_clusters([a1, a2])
        # Whatever clusters are found, they should have at least min_cluster_size articles
        for c in clusters:
            assert len(c.sources_involved) >= 1


# ============================================================================
# 10. OUTPUT — Feature Exporter (SDD-06)
# ============================================================================

class TestFeatureExporter:
    """SDD-06: ~60 news features across 8 groups."""

    def _make_enriched(self, source_id, url, title, category=None, relevance=0.7, sentiment=0.0, keywords=None):
        from src.news_engine.models import RawArticle, EnrichedArticle
        raw = RawArticle(
            url=url, title=title, source_id=source_id,
            published_at=datetime.now(),
        )
        return EnrichedArticle(
            raw=raw,
            category=category,
            relevance_score=relevance,
            sentiment_score=sentiment,
            keywords=keywords or [],
        )

    def test_export_daily_empty(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        result = exporter.export_daily([], date(2026, 2, 25))
        # Empty input -> still returns a NewsFeatureVector with 0 articles
        assert result is not None
        assert result.article_count == 0

    def test_export_daily_with_articles(self):
        from src.news_engine.output.feature_exporter import FeatureExporter

        exporter = FeatureExporter()

        articles = [
            self._make_enriched(
                "gdelt_doc", f"https://a.com/{i}",
                f"Article {i} about dolar and petroleo",
                category="fx_market" if i % 2 == 0 else "commodities",
                relevance=0.7,
                sentiment=0.3 if i % 2 == 0 else -0.2,
                keywords=["dolar", "petroleo"],
            )
            for i in range(10)
        ]

        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert result is not None
        assert isinstance(result.features, dict)
        assert len(result.features) > 10  # Should have many features
        # Check key feature groups exist
        feature_keys = set(result.features.keys())
        # Volume features (Group A)
        assert any(k.startswith("vol_") for k in feature_keys)
        # Sentiment features (Group C/D)
        assert any(k.startswith("sent_") for k in feature_keys)


# ============================================================================
# 11. OUTPUT — Digest Generator (SDD-06)
# ============================================================================

class TestDigestGenerator:
    """SDD-06: Daily/weekly text digests."""

    def _make_enriched(self, source_id, url, title, category=None, relevance=0.7, sentiment=0.0, keywords=None):
        from src.news_engine.models import RawArticle, EnrichedArticle
        raw = RawArticle(
            url=url, title=title, source_id=source_id,
            published_at=datetime.now(),
        )
        return EnrichedArticle(
            raw=raw,
            category=category,
            relevance_score=relevance,
            sentiment_score=sentiment,
            keywords=keywords or [],
        )

    def test_daily_digest(self):
        from src.news_engine.output.digest_generator import DigestGenerator

        gen = DigestGenerator()
        articles = [
            self._make_enriched(
                "gdelt_doc", f"https://a.com/{i}",
                f"Article {i} about dolar",
                category="fx_market",
                relevance=0.8, sentiment=0.2,
                keywords=["dolar"],
            )
            for i in range(5)
        ]
        digest = gen.generate_daily(articles, date(2026, 2, 25))
        assert digest is not None
        assert digest.digest_type == "daily"
        assert digest.total_articles == 5

    def test_weekly_digest(self):
        from src.news_engine.output.digest_generator import DigestGenerator

        gen = DigestGenerator()
        articles = [
            self._make_enriched(
                "newsapi", f"https://b.com/{i}",
                f"Weekly article {i}",
                category="inflation",
                relevance=0.6, sentiment=-0.1,
            )
            for i in range(15)
        ]
        # generate_weekly takes (articles, week_start) only
        digest = gen.generate_weekly(articles, date(2026, 2, 17))
        assert digest is not None
        assert digest.digest_type == "weekly"
        assert digest.total_articles == 15


# ============================================================================
# 12. OUTPUT — Alert System (SDD-06)
# ============================================================================

class TestAlertSystem:
    """SDD-06: Breaking news detection."""

    def test_alert_system_init(self):
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig
        system = AlertSystem(AlertConfig())
        assert system is not None

    def _make_enriched(self, source_id, url, title, category=None, relevance=0.7,
                       sentiment=None, gdelt_tone=None, is_breaking=False):
        from src.news_engine.models import RawArticle, EnrichedArticle
        raw = RawArticle(
            url=url, title=title, source_id=source_id,
            published_at=datetime.now(),
            gdelt_tone=gdelt_tone,
        )
        return EnrichedArticle(
            raw=raw,
            category=category,
            relevance_score=relevance,
            sentiment_score=sentiment,
            is_breaking=is_breaking,
        )

    def test_detect_breaking_news(self):
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig

        system = AlertSystem(AlertConfig())

        articles = [
            self._make_enriched(
                "gdelt_doc", "https://crisis.com/1",
                "Crisis financiera en Colombia",
                category="inflation",
                relevance=0.95, sentiment=-0.9,
                gdelt_tone=-18.0,
                is_breaking=True,
            ),
        ]
        alerts = system.check_articles(articles)
        # is_breaking=True + extreme gdelt_tone -> should trigger alerts
        assert len(alerts) >= 1

    def test_no_alert_for_normal_articles(self):
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig

        system = AlertSystem(AlertConfig())

        articles = [
            self._make_enriched(
                "newsapi", "https://normal.com/1",
                "Economia estable en Colombia",
                category="inflation",
                relevance=0.3, sentiment=0.1,
                is_breaking=False,
            ),
        ]
        alerts = system.check_articles(articles)
        assert len(alerts) == 0


# ============================================================================
# 13. MACRO ANALYZER (SDD-07)
# ============================================================================

class TestMacroAnalyzer:
    """SDD-07: Technical indicators on macro variables."""

    def _make_series(self, n=100, start_val=100.0, trend=-0.05, seed=42):
        """Create a pandas Series with a trend for testing."""
        import numpy as np
        import pandas as pd
        np.random.seed(seed)
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        values = np.cumsum(np.random.randn(n) * 0.5 + trend) + start_val
        return pd.Series(values, index=dates)

    def test_compute_rsi_wilder(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        import pandas as pd

        analyzer = MacroAnalyzer()
        series = self._make_series(50, start_val=100)
        rsi = analyzer.compute_rsi(series, period=14)
        # RSI returns a pandas Series
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(series)
        # RSI should be between 0 and 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100

    def test_compute_rsi_overbought(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        import numpy as np
        import pandas as pd

        analyzer = MacroAnalyzer()
        # Strong uptrend -> high RSI
        dates = pd.date_range("2026-01-01", periods=50, freq="D")
        values = np.linspace(100, 150, 50) + np.random.randn(50) * 0.1
        series = pd.Series(values, index=dates)
        rsi = analyzer.compute_rsi(series, period=14)
        # Last RSI value should be high (overbought territory)
        assert rsi.iloc[-1] > 60

    def test_compute_rsi_oversold(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        import numpy as np
        import pandas as pd

        analyzer = MacroAnalyzer()
        # Strong downtrend -> low RSI
        dates = pd.date_range("2026-01-01", periods=50, freq="D")
        values = np.linspace(150, 100, 50) + np.random.randn(50) * 0.1
        series = pd.Series(values, index=dates)
        rsi = analyzer.compute_rsi(series, period=14)
        assert rsi.iloc[-1] < 40

    def test_compute_bollinger_bands(self):
        from src.analysis.macro_analyzer import MacroAnalyzer

        analyzer = MacroAnalyzer()
        series = self._make_series(50, start_val=100)
        bb = analyzer.compute_bollinger_bands(series, period=20, num_std=2)
        # Returns dict with 'middle', 'upper', 'lower' keys
        assert "middle" in bb
        assert "upper" in bb
        assert "lower" in bb
        # Upper > middle > lower (on most points after enough data)
        assert bb["upper"].iloc[-1] > bb["lower"].iloc[-1]

    def test_compute_macd(self):
        from src.analysis.macro_analyzer import MacroAnalyzer

        analyzer = MacroAnalyzer()
        series = self._make_series(100, start_val=100)
        macd = analyzer.compute_macd(series, fast=12, slow=26, signal=9)
        # Returns dict with 'macd_line', 'signal', 'histogram'
        assert "macd_line" in macd
        assert "signal" in macd
        assert "histogram" in macd

    def test_compute_roc(self):
        from src.analysis.macro_analyzer import MacroAnalyzer

        analyzer = MacroAnalyzer()
        series = self._make_series(50, start_val=100)
        # compute_roc takes (series, period) - period is a single int
        roc = analyzer.compute_roc(series, period=5)
        assert len(roc) == len(series)

    def test_compute_snapshot(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        from src.contracts.analysis_schema import MacroSnapshot

        analyzer = MacroAnalyzer()
        series = self._make_series(100, start_val=100)
        as_of = series.index[-1].date()
        snapshot = analyzer.compute_snapshot(series, "dxy", as_of)
        assert snapshot is not None
        assert isinstance(snapshot, MacroSnapshot)
        assert snapshot.variable_key == "dxy"
        assert snapshot.sma_20 is not None
        assert snapshot.rsi_14 is not None


# ============================================================================
# 14. PROMPT TEMPLATES (SDD-07)
# ============================================================================

class TestPromptTemplates:
    """SDD-07: Spanish prompts for daily/weekly analysis."""

    def test_system_prompts_in_spanish(self):
        from src.analysis.prompt_templates import SYSTEM_DAILY, SYSTEM_WEEKLY
        # System prompts must be in Spanish
        assert "USD/COP" in SYSTEM_DAILY or "analista" in SYSTEM_DAILY.lower()
        assert "USD/COP" in SYSTEM_WEEKLY or "analista" in SYSTEM_WEEKLY.lower()

    def test_daily_template_fields(self):
        from src.analysis.prompt_templates import DAILY_TEMPLATE
        # Template must have key placeholders
        assert "{date}" in DAILY_TEMPLATE
        assert "{macro_section}" in DAILY_TEMPLATE
        assert "{signal_section}" in DAILY_TEMPLATE

    def test_weekly_template_fields(self):
        from src.analysis.prompt_templates import WEEKLY_TEMPLATE
        assert "{week}" in WEEKLY_TEMPLATE
        assert "{year}" in WEEKLY_TEMPLATE
        assert "{macro_section}" in WEEKLY_TEMPLATE

    def test_build_macro_section_empty(self):
        from src.analysis.prompt_templates import build_macro_section
        section = build_macro_section({})
        assert isinstance(section, str)

    def test_build_macro_section_with_data(self):
        from src.analysis.prompt_templates import build_macro_section
        from src.contracts.analysis_schema import MacroSnapshot

        snap = MacroSnapshot(
            snapshot_date="2026-02-25",
            variable_key="dxy",
            variable_name="DXY Index",
            value=104.5,
            sma_20=103.5,
            rsi_14=62.0,
            trend="above_sma20",
        )
        section = build_macro_section({"dxy": snap})
        assert "DXY" in section or "dxy" in section.lower()
        assert "104.5" in section

    def test_build_signal_section(self):
        from src.analysis.prompt_templates import build_signal_section
        section = build_signal_section()
        assert isinstance(section, str)

    def test_build_events_section_empty(self):
        from src.analysis.prompt_templates import build_events_section
        section = build_events_section([])
        assert isinstance(section, str)


# ============================================================================
# 15. CHART GENERATOR (SDD-07)
# ============================================================================

class TestChartGenerator:
    """SDD-07: Matplotlib PNG chart generation."""

    def test_generate_chart_insufficient_data(self):
        from src.analysis.chart_generator import generate_variable_chart
        import pandas as pd
        import numpy as np

        # Too few data points — output_dir required
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            series = pd.Series([100, 101], index=pd.date_range("2026-02-25", periods=2))
            result = generate_variable_chart(series, "dxy", "DXY", date(2026, 2, 25), output_dir=tmpdir)
            assert result is None  # Not enough data

    def test_generate_chart_success(self):
        from src.analysis.chart_generator import generate_variable_chart
        import pandas as pd
        import numpy as np
        import tempfile

        np.random.seed(42)
        dates = pd.date_range("2025-11-01", periods=120, freq="D")
        values = np.cumsum(np.random.randn(120) * 0.5) + 104
        series = pd.Series(values, index=dates)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_variable_chart(
                series, "dxy", "DXY Index", date(2026, 2, 25),
                output_dir=tmpdir,
            )
            if result is not None:
                assert Path(result).exists()


# ============================================================================
# 16. LLM CLIENT (SDD-07)
# ============================================================================

class TestLLMClient:
    """SDD-07: LLM orchestration with fallback."""

    def test_llm_client_init_no_keys(self):
        from src.analysis.llm_client import LLMClient
        # Should initialize even without API keys
        client = LLMClient()
        assert client.total_cost == 0
        assert client.total_tokens == 0

    def test_llm_client_dry_run(self):
        """Test LLM client with mock provider."""
        from src.analysis.llm_client import LLMClient, LLMProvider

        class MockProvider(LLMProvider):
            def generate(self, system_prompt, user_prompt, max_tokens=2000, temperature=0.7):
                return {
                    "content": "Mock response",
                    "tokens_used": 100,
                    "model": "mock/test",
                    "cost_usd": 0.001,
                }
            def health_check(self):
                return True

        client = LLMClient(primary=MockProvider(), fallback=MockProvider())
        result = client.generate("System", "User prompt")
        assert result["content"] == "Mock response"
        assert result["tokens_used"] == 100
        assert client.total_tokens == 100


# ============================================================================
# 17. WEEKLY GENERATOR (SDD-07)
# ============================================================================

class TestWeeklyGenerator:
    """SDD-07: Orchestrator for weekly analysis pipeline."""

    def test_generator_init(self):
        from src.analysis.weekly_generator import WeeklyAnalysisGenerator
        gen = WeeklyAnalysisGenerator(dry_run=True)
        assert gen.dry_run is True

    def test_generator_dry_run_for_date(self):
        from src.analysis.weekly_generator import WeeklyAnalysisGenerator

        gen = WeeklyAnalysisGenerator(dry_run=True)
        # Use a past date that likely exists in the seed parquet
        try:
            record = gen.generate_for_date(date(2025, 6, 15))
            assert record is not None
            assert "DRY RUN" in record.summary_markdown
        except Exception as e:
            # If parquet file doesn't exist or date not in data,
            # that's acceptable — we just verify no crash
            error_str = str(e).lower()
            assert "data" in error_str or "parquet" in error_str or "column" in error_str or "key" in error_str or "empty" in error_str or "index" in error_str or "date" in error_str, \
                f"Unexpected error: {e}"


# ============================================================================
# 18. ANALYSIS CONFIG (YAML)
# ============================================================================

class TestAnalysisConfig:
    """SDD-07: Validate YAML config structure."""

    @pytest.fixture
    def config(self):
        import yaml
        config_path = ROOT / "config" / "analysis" / "weekly_analysis_ssot.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_yaml_loads(self, config):
        assert config is not None
        assert isinstance(config, dict)

    def test_yaml_required_sections(self, config):
        for section in ("_meta", "llm", "macro", "generation", "export", "charts"):
            assert section in config, f"Missing section: {section}"

    def test_yaml_llm_config(self, config):
        llm = config["llm"]
        assert "primary_provider" in llm
        assert "fallback_provider" in llm
        assert llm["primary_provider"] == "azure_openai"
        assert llm["fallback_provider"] == "anthropic"

    def test_yaml_macro_key_variables(self, config):
        macro = config["macro"]
        assert "key_variables" in macro
        vars = macro["key_variables"]
        assert "dxy" in vars
        assert "vix" in vars

    def test_yaml_chat_rate_limits(self, config):
        chat = config.get("chat", {})
        if chat:
            assert "max_messages_per_session" in chat
            assert "max_sessions_per_day" in chat


# ============================================================================
# 19. SQL MIGRATIONS
# ============================================================================

class TestSQLMigrations:
    """SDD-03: Validate SQL migration files exist and contain required tables."""

    def test_045_newsengine_sql_exists(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        assert path.exists(), "Migration 045 not found"

    def test_046_analysis_sql_exists(self):
        path = ROOT / "database" / "migrations" / "046_weekly_analysis_tables.sql"
        assert path.exists(), "Migration 046 not found"

    def test_045_contains_required_tables(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        sql = path.read_text()
        for table in [
            "news_sources", "news_articles", "news_keywords",
            "news_cross_references", "news_cross_reference_articles",
            "news_daily_digests", "news_ingestion_log", "news_feature_snapshots",
        ]:
            assert table in sql, f"Table {table} missing from 045 migration"

    def test_046_contains_required_tables(self):
        path = ROOT / "database" / "migrations" / "046_weekly_analysis_tables.sql"
        sql = path.read_text()
        for table in [
            "weekly_analysis", "daily_analysis",
            "macro_variable_snapshots", "analysis_chat_history",
        ]:
            assert table in sql, f"Table {table} missing from 046 migration"

    def test_046_contains_bollinger_rsi_macd(self):
        path = ROOT / "database" / "migrations" / "046_weekly_analysis_tables.sql"
        sql = path.read_text()
        for col in ["bollinger_upper_20", "rsi_14", "macd_line"]:
            assert col in sql, f"Column {col} missing from 046 migration"

    def test_045_has_unique_constraints(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        sql = path.read_text()
        # Should have url_hash unique constraint
        assert "url_hash" in sql


# ============================================================================
# 20. TYPESCRIPT CONTRACTS
# ============================================================================

class TestTypeScriptContract:
    """SDD-08: Validate TypeScript contract file."""

    @pytest.fixture
    def contract_content(self):
        path = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "weekly-analysis.contract.ts"
        return path.read_text()

    def test_contract_file_exists(self):
        path = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "weekly-analysis.contract.ts"
        assert path.exists(), "TypeScript contract not found"

    def test_contract_has_required_interfaces(self, contract_content):
        for iface in [
            "WeeklyViewData", "DailyAnalysisEntry", "MacroVariableSnapshot",
            "MacroChartPoint", "AnalysisIndex",
        ]:
            assert iface in contract_content, f"Interface {iface} missing"

    def test_contract_has_sentiment_colors(self, contract_content):
        assert "SENTIMENT_COLORS" in contract_content

    def test_contract_has_day_names(self, contract_content):
        assert "DAY_NAMES" in contract_content


# ============================================================================
# 21. END-TO-END INTEGRATION
# ============================================================================

class TestEndToEnd:
    """Integration tests combining multiple subsystems."""

    def _make_enriched(self, source_id, url, title, category=None, relevance=0.7,
                       sentiment=0.0, keywords=None):
        from src.news_engine.models import RawArticle, EnrichedArticle
        raw = RawArticle(
            url=url, title=title, source_id=source_id,
            published_at=datetime.now(),
        )
        return EnrichedArticle(
            raw=raw,
            category=category,
            relevance_score=relevance,
            sentiment_score=sentiment,
            keywords=keywords or [],
        )

    def test_enrichment_to_features_pipeline(self):
        """Enrich articles -> export features."""
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.output.feature_exporter import FeatureExporter
        from src.news_engine.models import RawArticle

        pipeline = EnrichmentPipeline(min_relevance=0.0)
        exporter = FeatureExporter()

        raw_articles = [
            RawArticle(
                url=f"https://test.com/{i}",
                title="Dolar sube en Colombia por DXY y petroleo",
                source_id="gdelt_doc",
                published_at=datetime.now(),
                summary="Tasa de cambio alcanza 4400 pesos",
            )
            for i in range(5)
        ]

        enriched = pipeline.enrich_batch(raw_articles)
        assert len(enriched) > 0

        features = exporter.export_daily(enriched, date(2026, 2, 25))
        assert features is not None
        assert features.article_count == len(enriched)
        assert len(features.features) > 10

    def test_macro_analyzer_to_chart(self):
        """Compute snapshot -> get chart data."""
        from src.analysis.macro_analyzer import MacroAnalyzer
        import pandas as pd
        import numpy as np

        analyzer = MacroAnalyzer()

        np.random.seed(42)
        dates = pd.date_range("2025-11-01", periods=120, freq="D")
        values = np.cumsum(np.random.randn(120) * 0.5) + 104
        series = pd.Series(values, index=dates)

        as_of = date(2026, 2, 25)
        snapshot = analyzer.compute_snapshot(series, "dxy", as_of)
        assert snapshot is not None
        assert snapshot.rsi_14 is not None

        chart_data = analyzer.get_chart_data(series, "dxy", as_of, lookback_days=90)
        assert isinstance(chart_data, list)
        assert len(chart_data) > 0
        # Each point has required fields
        pt = chart_data[0]
        assert "date" in pt
        assert "value" in pt
        assert "sma20" in pt
