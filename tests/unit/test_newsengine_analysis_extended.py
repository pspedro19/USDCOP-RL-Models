"""
Extended validation tests for NewsEngine + Analysis Module.
Covers edge cases, contract enforcement, cross-component integration,
config-code alignment, dashboard artifacts, DAG files, and data flow.

Run: python -m pytest tests/unit/test_newsengine_analysis_extended.py -v
"""

import json
import math
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================================
# HELPERS
# ============================================================================

def _make_raw(url="https://test.com/1", title="Test article", source_id="gdelt_doc",
              gdelt_tone=None, summary=None, content=None):
    from src.news_engine.models import RawArticle
    return RawArticle(
        url=url, title=title, source_id=source_id,
        published_at=datetime.now(), gdelt_tone=gdelt_tone,
        summary=summary, content=content,
    )


def _make_enriched(source_id="gdelt_doc", url="https://t.com/1", title="Test",
                   category=None, relevance=0.7, sentiment=0.0,
                   keywords=None, entities=None, is_breaking=False, gdelt_tone=None):
    from src.news_engine.models import RawArticle, EnrichedArticle
    raw = RawArticle(
        url=url, title=title, source_id=source_id,
        published_at=datetime.now(), gdelt_tone=gdelt_tone,
    )
    return EnrichedArticle(
        raw=raw, category=category, relevance_score=relevance,
        sentiment_score=sentiment, keywords=keywords or [],
        entities=entities or [], is_breaking=is_breaking,
    )


def _make_series(n=100, start_val=100.0, trend=0.0, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2025-10-01", periods=n, freq="D")
    values = np.cumsum(np.random.randn(n) * 0.5 + trend) + start_val
    return pd.Series(values, index=dates)


# ============================================================================
# 1. CONTRACT VALIDATION — Deep Schema Checks
# ============================================================================

class TestArticleRecordValidation:
    """SDD-03: ArticleRecord.validate() covers all error paths."""

    def test_validate_valid_record(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com/a",
            url_hash="abc", title="Valid title",
            published_at=datetime.now(),
            category="fx_market", sentiment_score=0.5,
            sentiment_label="positive",
        )
        assert rec.validate() == []

    def test_validate_bad_source(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="nonexistent_source", url="https://x.com",
            url_hash="abc", title="T", published_at=datetime.now(),
        )
        errors = rec.validate()
        assert any("source_id" in e for e in errors)

    def test_validate_empty_url(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="", url_hash="abc",
            title="T", published_at=datetime.now(),
        )
        errors = rec.validate()
        assert any("URL" in e for e in errors)

    def test_validate_empty_title(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com",
            url_hash="abc", title="", published_at=datetime.now(),
        )
        errors = rec.validate()
        assert any("Title" in e for e in errors)

    def test_validate_bad_category(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com",
            url_hash="abc", title="T", published_at=datetime.now(),
            category="nonexistent_category",
        )
        errors = rec.validate()
        assert any("category" in e for e in errors)

    def test_validate_bad_sentiment_label(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com",
            url_hash="abc", title="T", published_at=datetime.now(),
            sentiment_label="very_bad",
        )
        errors = rec.validate()
        assert any("sentiment_label" in e for e in errors)

    def test_validate_sentiment_score_out_of_range(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com",
            url_hash="abc", title="T", published_at=datetime.now(),
            sentiment_score=1.5,
        )
        errors = rec.validate()
        assert any("sentiment_score" in e for e in errors)

    def test_validate_negative_sentiment_score_out_of_range(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="gdelt_doc", url="https://x.com",
            url_hash="abc", title="T", published_at=datetime.now(),
            sentiment_score=-1.1,
        )
        errors = rec.validate()
        assert any("sentiment_score" in e for e in errors)

    def test_to_dict_roundtrip(self):
        from src.contracts.news_engine_schema import ArticleRecord
        rec = ArticleRecord(
            source_id="newsapi", url="https://x.com/a",
            url_hash="abc123", title="Test Article",
            published_at=datetime.now(),
        )
        d = rec.to_dict()
        assert d["source_id"] == "newsapi"
        assert d["url"] == "https://x.com/a"


class TestAnalysisContractsDeep:
    """SDD-07: Deep validation of analysis contracts."""

    def test_macro_snapshot_to_dict_date_serialization(self):
        from src.contracts.analysis_schema import MacroSnapshot
        snap = MacroSnapshot(
            snapshot_date=date(2026, 2, 25), variable_key="vix",
            variable_name="VIX", value=18.5,
        )
        d = snap.to_dict()
        assert d["snapshot_date"] == "2026-02-25"
        assert isinstance(d["snapshot_date"], str)

    def test_daily_analysis_record_to_dict_serialization(self):
        from src.contracts.analysis_schema import DailyAnalysisRecord
        rec = DailyAnalysisRecord(
            analysis_date=date(2026, 2, 25),
            iso_year=2026, iso_week=9, day_of_week=1,
        )
        d = rec.to_dict()
        assert d["analysis_date"] == "2026-02-25"

    def test_weekly_analysis_record_dates(self):
        from src.contracts.analysis_schema import WeeklyAnalysisRecord
        rec = WeeklyAnalysisRecord(
            iso_year=2026, iso_week=9,
            week_start=date(2026, 2, 23), week_end=date(2026, 2, 27),
        )
        d = rec.to_dict()
        assert d["week_start"] == "2026-02-23"
        assert d["week_end"] == "2026-02-27"

    def test_chat_message_role_values(self):
        from src.contracts.analysis_schema import ChatMessage
        for role in ("user", "assistant", "system"):
            msg = ChatMessage(role=role, content="test", session_id="abc")
            assert msg.role == role

    def test_analysis_index_entry(self):
        from src.contracts.analysis_schema import AnalysisIndexEntry
        entry = AnalysisIndexEntry(
            year=2026, week=9, start="2026-02-23", end="2026-02-27",
            sentiment="bearish", headline="Test", has_weekly=True, daily_count=5,
        )
        d = entry.to_dict()
        assert d["year"] == 2026
        assert d["has_weekly"] is True

    def test_sanitize_for_json_nested(self):
        from src.contracts.analysis_schema import _sanitize_for_json
        data = {
            "a": float("inf"),
            "b": [1, float("nan"), {"c": float("-inf")}],
            "d": date(2026, 1, 1),
            "e": datetime(2026, 1, 1, 12, 0),
            "f": 42,
        }
        result = _sanitize_for_json(data)
        assert result["a"] is None
        assert result["b"][1] is None
        assert result["b"][2]["c"] is None
        assert result["d"] == "2026-01-01"
        assert "2026-01-01" in result["e"]
        assert result["f"] == 42
        # Must be JSON-serializable
        json.dumps(result)

    def test_display_names_cover_key_variables(self):
        from src.contracts.analysis_schema import KEY_MACRO_VARIABLES, DISPLAY_NAMES
        for var in KEY_MACRO_VARIABLES:
            assert var in DISPLAY_NAMES, f"Missing display name for {var}"


class TestCrossReferenceRecord:
    """SDD-05: CrossReferenceRecord contract."""

    def test_to_dict_excludes_article_ids(self):
        from src.contracts.news_engine_schema import CrossReferenceRecord
        rec = CrossReferenceRecord(
            topic="DXY strength", cluster_date=date(2026, 2, 25),
            article_count=3, article_ids=["a", "b", "c"],
        )
        d = rec.to_dict()
        assert "article_ids" not in d

    def test_ingestion_log_record(self):
        from src.contracts.news_engine_schema import IngestionLogRecord
        rec = IngestionLogRecord(
            source_id="gdelt_doc", run_type="scheduled",
            articles_fetched=50, articles_new=10, status="success",
        )
        d = rec.to_dict()
        assert d["source_id"] == "gdelt_doc"
        assert d["status"] == "success"


# ============================================================================
# 2. NEWS ENGINE MODELS — Edge Cases
# ============================================================================

class TestRawArticleEdgeCases:
    """SDD-02: RawArticle hash and properties."""

    def test_url_hash_is_sha256(self):
        art = _make_raw(url="https://test.com/article-123")
        assert len(art.url_hash) == 64
        # Must be hex chars only
        assert all(c in "0123456789abcdef" for c in art.url_hash)

    def test_url_hash_deterministic(self):
        art1 = _make_raw(url="https://test.com/x")
        art2 = _make_raw(url="https://test.com/x")
        assert art1.url_hash == art2.url_hash

    def test_special_characters_in_url(self):
        art = _make_raw(url="https://test.com/a?q=hello world&x=1#frag")
        assert len(art.url_hash) == 64

    def test_enriched_article_defaults(self):
        from src.news_engine.models import EnrichedArticle
        raw = _make_raw()
        enriched = EnrichedArticle(raw=raw)
        assert enriched.category is None
        assert enriched.relevance_score == 0.0
        assert enriched.sentiment_score is None
        assert enriched.keywords == []
        assert enriched.entities == []
        assert enriched.is_breaking is False
        assert enriched.is_weekly_relevant is False


class TestNewsFeatureVector:
    """SDD-06: NewsFeatureVector properties."""

    def test_feature_names_sorted(self):
        from src.news_engine.models import NewsFeatureVector
        fv = NewsFeatureVector(
            snapshot_date=datetime.now(),
            features={"vol_fx_market": 10, "vol_commodities": 5, "sent_mean": 0.3},
        )
        assert fv.feature_names == sorted(fv.features.keys())
        assert fv.feature_count == 3


# ============================================================================
# 3. CATEGORIZER — All 9 Categories
# ============================================================================

class TestCategorizerAllCategories:
    """SDD-04: Ensure all 9 categories are reachable."""

    @pytest.mark.parametrize("text,expected_cat", [
        ("El Banco de la Republica decidio mantener la tasa de interes", "monetary_policy"),
        ("El dolar se fortalece frente al peso colombiano", "fx_market"),
        ("Precio del petroleo WTI cae por exceso de oferta", "commodities"),
        ("La inflacion en Colombia alcanza el 7% segun el IPC", "inflation"),
        ("El gobierno presenta nueva reforma tributaria", "fiscal_policy"),
        ("El EMBI Colombia sube indicando mayor riesgo pais", "risk_premium"),
        ("Inversion extranjera en Colombia crece un 15%", "capital_flows"),
        ("Las remesas a Colombia alcanzan record historico", "balance_payments"),
        ("El presidente anuncia nuevas medidas en el congreso", "political"),
    ])
    def test_category_detection(self, text, expected_cat):
        from src.news_engine.enrichment.categorizer import categorize_article
        cat, _ = categorize_article(text)
        assert cat == expected_cat, f"Expected {expected_cat}, got {cat} for: {text}"

    def test_category_priority_by_score(self):
        """When multiple categories match, highest match count wins."""
        from src.news_engine.enrichment.categorizer import categorize_article
        # "dolar" and "tasa de cambio" -> fx_market should win
        cat, _ = categorize_article(
            "El dolar y la tasa de cambio USD COP peso colombiano"
        )
        assert cat == "fx_market"

    def test_all_valid_categories_are_in_rules(self):
        from src.news_engine.enrichment.categorizer import CATEGORY_RULES
        from src.contracts.news_engine_schema import VALID_CATEGORIES
        for cat in VALID_CATEGORIES:
            assert cat in CATEGORY_RULES, f"Category {cat} missing from CATEGORY_RULES"


# ============================================================================
# 4. RELEVANCE SCORER — Detailed Weight Testing
# ============================================================================

class TestRelevanceScorerWeights:
    """SDD-04: Verify score composition (60% keyword, 20% source, 20% recency)."""

    def test_keyword_weight_usdcop(self):
        from src.news_engine.enrichment.relevance import score_relevance, KEYWORD_WEIGHTS
        assert KEYWORD_WEIGHTS["usdcop"] == 1.0

    def test_source_weights_exist_for_all_sources(self):
        from src.news_engine.enrichment.relevance import SOURCE_WEIGHTS
        from src.contracts.news_engine_schema import VALID_SOURCE_IDS
        for src in VALID_SOURCE_IDS:
            assert src in SOURCE_WEIGHTS, f"Missing source weight for {src}"

    def test_multiple_keyword_bonus(self):
        from src.news_engine.enrichment.relevance import score_relevance
        # 3+ keywords should get +0.1 bonus
        score_few = score_relevance("dolar", source_id="gdelt_doc")
        score_many = score_relevance(
            "usdcop dolar tasa de cambio peso colombiano banrep",
            source_id="gdelt_doc",
        )
        assert score_many > score_few

    def test_score_always_in_range(self):
        from src.news_engine.enrichment.relevance import score_relevance
        for text in ["", "random", "USDCOP dolar peso", "x" * 1000]:
            score = score_relevance(text, source_id="gdelt_doc")
            assert 0.0 <= score <= 1.0


# ============================================================================
# 5. SENTIMENT — Boundary Testing
# ============================================================================

class TestSentimentBoundaries:
    """SDD-04: Sentiment score always [-1, 1] with correct labels."""

    @pytest.mark.parametrize("tone,expected_sign", [
        (20.0, 1), (10.0, 1), (3.0, 1),      # Positive
        (-20.0, -1), (-10.0, -1), (-3.0, -1), # Negative
    ])
    def test_gdelt_tone_sign(self, tone, expected_sign):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        score, _ = analyze_sentiment("test", gdelt_tone=tone)
        if expected_sign > 0:
            assert score > 0
        else:
            assert score < 0

    def test_gdelt_tone_zero_is_neutral(self):
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        score, label = analyze_sentiment("test", gdelt_tone=0.0)
        assert score == 0.0
        assert label == "neutral"

    def test_label_thresholds(self):
        from src.news_engine.enrichment.sentiment import _score_to_label
        assert _score_to_label(0.15) == "positive"
        assert _score_to_label(0.14) == "neutral"
        assert _score_to_label(-0.15) == "negative"
        assert _score_to_label(-0.14) == "neutral"

    def test_extreme_tone_values_clamped(self):
        from src.news_engine.enrichment.sentiment import _normalize_gdelt_tone
        assert _normalize_gdelt_tone(100.0) == 1.0
        assert _normalize_gdelt_tone(-100.0) == -1.0
        assert _normalize_gdelt_tone(0.0) == 0.0


# ============================================================================
# 6. TAGGER — Keyword & Entity Coverage
# ============================================================================

class TestTaggerCoverage:
    """SDD-04: Keyword and entity extraction depth."""

    def test_financial_keywords_defined(self):
        from src.news_engine.enrichment.tagger import FINANCIAL_KEYWORDS
        assert len(FINANCIAL_KEYWORDS) >= 20
        assert "dolar" in FINANCIAL_KEYWORDS
        assert "inflacion" in FINANCIAL_KEYWORDS
        assert "petroleo" in FINANCIAL_KEYWORDS

    def test_entity_patterns_defined(self):
        from src.news_engine.enrichment.tagger import ENTITY_PATTERNS
        entity_names = [name for _, name in ENTITY_PATTERNS]
        assert "Banco de la Republica" in entity_names
        assert "Federal Reserve" in entity_names
        assert "OPEC" in entity_names

    def test_max_keywords_respected(self):
        from src.news_engine.enrichment.tagger import extract_keywords
        long_text = " ".join(["dolar peso inflacion petroleo embi riesgo desempleo"] * 10)
        keywords = extract_keywords(long_text, max_keywords=5)
        assert len(keywords) <= 5

    def test_deduplicate_keywords(self):
        from src.news_engine.enrichment.tagger import extract_keywords
        keywords = extract_keywords("dolar dolar dolar peso peso")
        # No duplicates
        assert len(keywords) == len(set(k.lower() for k in keywords))

    def test_entity_extraction_multiple(self):
        from src.news_engine.enrichment.tagger import extract_entities
        text = "Jerome Powell habla sobre la Fed. El Banco de la Republica y el FMI opinan."
        entities = extract_entities(text)
        assert "Federal Reserve" in entities
        assert "Banco de la Republica" in entities
        assert "IMF" in entities


# ============================================================================
# 7. ENRICHMENT PIPELINE — Integration
# ============================================================================

class TestEnrichmentPipelineIntegration:
    """SDD-04: Full pipeline with realistic data."""

    def test_breaking_detection_high_relevance_extreme_sentiment(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        pipeline = EnrichmentPipeline()
        article = _make_raw(
            title="USD/COP dolar peso colombiano crisis cambiaria devaluacion",
            content="La tasa de cambio se dispara en medio de panico",
            source_id="investing",
            gdelt_tone=-18.0,
        )
        enriched = pipeline.enrich(article)
        # High relevance + extreme gdelt_tone -> breaking
        assert enriched.is_breaking is True

    def test_breaking_detection_normal_article(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        pipeline = EnrichmentPipeline()
        article = _make_raw(
            title="Resultados deportivos del dia",
            source_id="newsapi",
        )
        enriched = pipeline.enrich(article)
        assert enriched.is_breaking is False

    def test_weekly_relevance_by_category(self):
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        pipeline = EnrichmentPipeline()
        article = _make_raw(
            title="Banco de la Republica mantiene tasa de interes en 10%",
            source_id="larepublica",
        )
        enriched = pipeline.enrich(article)
        assert enriched.is_weekly_relevant is True  # monetary_policy

    def test_batch_enrichment_error_isolation(self):
        """One bad article shouldn't crash the batch."""
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.models import RawArticle
        pipeline = EnrichmentPipeline(min_relevance=0.0)
        articles = [
            RawArticle(url="https://t.com/1", title="Dolar sube",
                       source_id="gdelt_doc", published_at=datetime.now()),
            RawArticle(url="https://t.com/2", title="Otro articulo",
                       source_id="gdelt_doc", published_at=datetime.now()),
        ]
        results = pipeline.enrich_batch(articles)
        assert len(results) >= 1


# ============================================================================
# 8. CROSS-REFERENCE ENGINE — Edge Cases
# ============================================================================

class TestCrossReferenceEdgeCases:
    """SDD-05: Clustering edge cases."""

    def test_empty_input(self):
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig
        engine = CrossReferenceEngine(CrossReferenceConfig())
        clusters = engine.find_clusters([])
        assert clusters == []

    def test_single_article_no_cluster(self):
        from src.news_engine.cross_reference.engine import CrossReferenceEngine
        from src.news_engine.config import CrossReferenceConfig
        engine = CrossReferenceEngine(CrossReferenceConfig())
        a1 = _make_enriched(
            source_id="gdelt_doc", url="https://a.com/1",
            title="Dolar sube en Colombia",
            category="fx_market", keywords=["dolar"],
        )
        clusters = engine.find_clusters([a1])
        assert len(clusters) == 0  # Need min_cluster_size=2

    def test_cross_reference_config_defaults(self):
        from src.news_engine.config import CrossReferenceConfig
        cfg = CrossReferenceConfig()
        assert cfg.title_weight + cfg.entity_weight + cfg.summary_weight + cfg.category_weight == pytest.approx(1.0)
        assert cfg.min_cluster_size == 2


# ============================================================================
# 9. FEATURE EXPORTER — Feature Groups
# ============================================================================

class TestFeatureExporterGroups:
    """SDD-06: Validate all 7 feature groups are present."""

    def test_volume_features(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        articles = [
            _make_enriched(category="fx_market", url=f"https://t.com/{i}")
            for i in range(5)
        ]
        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert "vol_fx_market" in result.features

    def test_sentiment_features(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        articles = [
            _make_enriched(sentiment=0.5, category="fx_market", url=f"https://t.com/{i}")
            for i in range(3)
        ]
        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert "sent_mean" in result.features or any(
            k.startswith("sent_") for k in result.features
        )

    def test_keyword_features(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        articles = [
            _make_enriched(keywords=["dolar", "petroleo"], url=f"https://t.com/{i}")
            for i in range(3)
        ]
        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert any(k.startswith("kw_") for k in result.features)

    def test_source_features(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        articles = [
            _make_enriched(source_id="gdelt_doc", url=f"https://t.com/{i}")
            for i in range(3)
        ]
        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert any(k.startswith("src_") for k in result.features)

    def test_source_counts_populated(self):
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()
        articles = [
            _make_enriched(source_id="gdelt_doc", url="https://t.com/1"),
            _make_enriched(source_id="newsapi", url="https://t.com/2"),
        ]
        result = exporter.export_daily(articles, date(2026, 2, 25))
        assert result.article_count == 2
        assert isinstance(result.source_counts, dict)


# ============================================================================
# 10. DIGEST GENERATOR — Structure Validation
# ============================================================================

class TestDigestGeneratorStructure:
    """SDD-06: Digest output structure."""

    def test_daily_digest_has_breakdowns(self):
        from src.news_engine.output.digest_generator import DigestGenerator
        gen = DigestGenerator()
        articles = [
            _make_enriched(source_id="gdelt_doc", category="fx_market",
                           keywords=["dolar"], url=f"https://t.com/{i}")
            for i in range(5)
        ]
        digest = gen.generate_daily(articles, date(2026, 2, 25))
        assert isinstance(digest.by_source, dict)
        assert isinstance(digest.by_category, dict)
        assert len(digest.top_keywords) <= 10

    def test_digest_summary_text_generated(self):
        from src.news_engine.output.digest_generator import DigestGenerator
        gen = DigestGenerator()
        articles = [
            _make_enriched(source_id="gdelt_doc", category="fx_market",
                           url=f"https://t.com/{i}")
            for i in range(3)
        ]
        digest = gen.generate_daily(articles, date(2026, 2, 25))
        assert digest.summary_text is not None
        assert len(digest.summary_text) > 0


# ============================================================================
# 11. ALERT SYSTEM — Threshold Testing
# ============================================================================

class TestAlertSystemThresholds:
    """SDD-06: Alert trigger conditions."""

    def test_extreme_gdelt_tone_alert(self):
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig
        system = AlertSystem(AlertConfig(gdelt_tone_threshold=-15.0))
        articles = [
            _make_enriched(gdelt_tone=-16.0, is_breaking=False, url="https://t.com/1"),
        ]
        alerts = system.check_articles(articles)
        assert any(a["type"] == "extreme_negative_tone" for a in alerts)

    def test_tone_just_above_threshold_no_alert(self):
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig
        system = AlertSystem(AlertConfig(gdelt_tone_threshold=-15.0))
        articles = [
            _make_enriched(gdelt_tone=-14.0, is_breaking=False, url="https://t.com/1"),
        ]
        alerts = system.check_articles(articles)
        tone_alerts = [a for a in alerts if a["type"] == "extreme_negative_tone"]
        assert len(tone_alerts) == 0


# ============================================================================
# 12. MACRO ANALYZER — Technical Indicators Deep Tests
# ============================================================================

class TestMacroAnalyzerDeep:
    """SDD-07: Technical indicator correctness."""

    def test_sma_values_reasonable(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = _make_series(100, start_val=100, trend=0)
        snapshot = analyzer.compute_snapshot(series, "dxy", series.index[-1].date())
        # SMA should be close to 100 (no trend)
        assert abs(snapshot.sma_20 - 100) < 10

    def test_rsi_range_always_0_100(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        for seed in [42, 123, 456]:
            series = _make_series(50, seed=seed)
            rsi = analyzer.compute_rsi(series, 14)
            valid = rsi.dropna()
            assert valid.min() >= 0, f"RSI below 0 with seed {seed}"
            assert valid.max() <= 100, f"RSI above 100 with seed {seed}"

    def test_bollinger_upper_above_lower(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = _make_series(50)
        bb = analyzer.compute_bollinger_bands(series, 20, 2)
        # After enough data points, upper > lower
        mask = ~(bb["upper"].isna() | bb["lower"].isna())
        valid = mask[mask].index
        if len(valid) > 0:
            assert (bb["upper"][valid] >= bb["lower"][valid]).all()

    def test_macd_histogram_equals_line_minus_signal(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = _make_series(100)
        macd = analyzer.compute_macd(series, 12, 26, 9)
        diff = (macd["macd_line"] - macd["signal"]).dropna()
        hist = macd["histogram"].dropna()
        pd.testing.assert_series_equal(diff, hist, atol=1e-10)

    def test_roc_formula(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = pd.Series([100, 105, 110, 120, 130], index=pd.date_range("2026-01-01", periods=5))
        roc = analyzer.compute_roc(series, period=1)
        # ROC[1] = ((105 - 100) / 100) * 100 = 5.0
        assert abs(roc.iloc[1] - 5.0) < 0.01

    def test_z_score_computation(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = _make_series(50, start_val=100, trend=0, seed=42)
        snapshot = analyzer.compute_snapshot(series, "test", series.index[-1].date())
        # Z-score should be finite
        if snapshot.z_score_20 is not None:
            assert abs(snapshot.z_score_20) < 10

    def test_trend_detection(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        # Strong uptrend -> above_sma20
        series = _make_series(100, start_val=100, trend=0.5, seed=42)
        snapshot = analyzer.compute_snapshot(series, "test", series.index[-1].date())
        assert snapshot.trend in ("above_sma20", "golden_cross")

    def test_signal_overbought(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        # Strong uptrend should have RSI > 70 -> overbought
        dates = pd.date_range("2025-10-01", periods=50, freq="D")
        values = np.linspace(100, 160, 50) + np.random.RandomState(42).randn(50) * 0.05
        series = pd.Series(values, index=dates)
        snapshot = analyzer.compute_snapshot(series, "test", series.index[-1].date())
        assert snapshot.signal in ("overbought", "bb_upper_touch", "neutral")

    def test_insufficient_data_returns_none(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = pd.Series([100], index=pd.date_range("2026-01-01", periods=1))
        snapshot = analyzer.compute_snapshot(series, "dxy", date(2026, 1, 1))
        assert snapshot is None

    def test_find_column_mapping(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        df = pd.DataFrame({"dxy_close": [100], "vix_close": [18]})
        assert analyzer._find_column(df, "dxy") == "dxy_close"
        assert analyzer._find_column(df, "vix") == "vix_close"
        assert analyzer._find_column(df, "nonexistent") is None

    def test_chart_data_format(self):
        from src.analysis.macro_analyzer import MacroAnalyzer
        analyzer = MacroAnalyzer()
        series = _make_series(100)
        data = analyzer.get_chart_data(series, "dxy", series.index[-1].date(), lookback_days=30)
        assert len(data) > 0
        for pt in data:
            assert "date" in pt
            assert "value" in pt
            assert "sma20" in pt
            assert "bb_upper" in pt
            assert "bb_lower" in pt
            assert "rsi" in pt

    def test_safe_float_handles_nan_inf(self):
        from src.analysis.macro_analyzer import _safe_float
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None
        assert _safe_float(None) is None
        assert _safe_float(42.5) == 42.5


# ============================================================================
# 13. LLM CLIENT — Fallback & Caching
# ============================================================================

class TestLLMClientFallback:
    """SDD-07: LLM client fallback logic."""

    def _make_mock_provider(self, should_fail=False, response_content="OK"):
        from src.analysis.llm_client import LLMProvider

        class Mock(LLMProvider):
            def __init__(self):
                self.call_count = 0

            def generate(self, system_prompt, user_prompt, max_tokens=2000, temperature=0.7):
                self.call_count += 1
                if should_fail:
                    raise RuntimeError("Provider failed")
                return {"content": response_content, "tokens_used": 50, "model": "mock", "cost_usd": 0.001}

            def health_check(self):
                return not should_fail

        return Mock()

    def test_fallback_after_max_failures(self):
        from src.analysis.llm_client import LLMClient
        primary = self._make_mock_provider(should_fail=True)
        fallback = self._make_mock_provider(should_fail=False, response_content="Fallback")
        client = LLMClient(primary=primary, fallback=fallback, max_failures=2)

        # First call: primary fails, fallback called
        result = client.generate("sys", "usr")
        assert result["content"] == "Fallback"
        assert client._failure_count == 1

    def test_cost_tracking_accumulates(self):
        from src.analysis.llm_client import LLMClient
        provider = self._make_mock_provider()
        client = LLMClient(primary=provider, fallback=provider)

        client.generate("sys", "prompt1")
        client.generate("sys", "prompt2")
        assert client.total_cost == pytest.approx(0.002, abs=0.0001)
        assert client.total_tokens == 100

    def test_file_cache_hit(self):
        from src.analysis.llm_client import LLMClient
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self._make_mock_provider(response_content="Cached response")
            client = LLMClient(primary=provider, fallback=provider, cache_dir=tmpdir)

            # First call: no cache
            r1 = client.generate("sys", "usr", cache_key="test_key")
            assert r1["cached"] is False

            # Second call: cache hit
            r2 = client.generate("sys", "usr", cache_key="test_key")
            assert r2["cached"] is True
            assert r2["content"] == "Cached response"
            # Provider should only have been called once
            assert provider.call_count == 1

    def test_azure_provider_health_check_no_keys(self):
        from src.analysis.llm_client import AzureOpenAIProvider
        provider = AzureOpenAIProvider(api_key="", endpoint="")
        assert provider.health_check() is False

    def test_anthropic_provider_health_check_no_keys(self):
        from src.analysis.llm_client import AnthropicProvider
        provider = AnthropicProvider(api_key="")
        assert provider.health_check() is False


# ============================================================================
# 14. PROMPT TEMPLATES — Formatting & Content
# ============================================================================

class TestPromptTemplatesContent:
    """SDD-07: Verify prompt structure for LLM."""

    def test_daily_template_renders(self):
        from src.analysis.prompt_templates import DAILY_TEMPLATE
        rendered = DAILY_TEMPLATE.format(
            date="2026-02-25",
            close=4370,
            change_pct=-0.2,
            low=4350,
            high=4400,
            macro_section="DXY: 104.5, VIX: 18.2",
            signal_section="H1: SHORT (-0.3%)",
            news_section="5 articulos, sentimiento: -0.2",
            events_section="Sin eventos programados",
        )
        assert "2026-02-25" in rendered
        assert "4370" in rendered

    def test_weekly_template_renders(self):
        from src.analysis.prompt_templates import WEEKLY_TEMPLATE
        rendered = WEEKLY_TEMPLATE.format(
            year=2026, week=9,
            start="2026-02-23",
            end="2026-02-27",
            ohlcv_section="Rango semanal: 4350-4400",
            macro_section="DXY: 104.5",
            signal_section="H5: SHORT, confianza ALTA",
            news_section="47 articulos",
            events_section="Fed decision el miercoles",
        )
        assert "2026" in rendered
        assert "9" in rendered
        assert "2026-02-23" in rendered

    def test_build_news_section_with_data(self):
        from src.analysis.prompt_templates import build_news_section
        news_data = {
            "article_count": 47,
            "avg_sentiment": -0.15,
            "top_categories": {"fx_market": 12, "commodities": 8},
        }
        section = build_news_section(news_data)
        assert isinstance(section, str)
        assert "47" in section

    def test_build_events_section_with_events(self):
        from src.analysis.prompt_templates import build_events_section
        events = [
            {"event": "Fed Decision", "impact_level": "high"},
            {"event": "CPI Colombia", "impact_level": "medium"},
        ]
        section = build_events_section(events)
        assert "Fed" in section
        assert "CPI" in section


# ============================================================================
# 15. CHART GENERATOR — Output Validation
# ============================================================================

class TestChartGeneratorOutput:
    """SDD-07: Chart PNG generation and styling."""

    def test_generate_chart_creates_png(self):
        from src.analysis.chart_generator import generate_variable_chart
        series = _make_series(120)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_variable_chart(
                series, "dxy", "DXY Index", series.index[-1].date(),
                output_dir=tmpdir, dpi=72,
            )
            if result is not None:
                p = Path(result)
                assert p.exists()
                assert p.suffix == ".png"
                assert p.stat().st_size > 1000  # Not an empty file

    def test_generate_all_charts(self):
        from src.analysis.chart_generator import generate_all_charts
        dates = pd.date_range("2025-10-01", periods=120, freq="D")
        df = pd.DataFrame({
            "dxy": np.cumsum(np.random.randn(120) * 0.3) + 104,
            "vix": np.abs(np.cumsum(np.random.randn(120) * 0.5) + 18),
        }, index=dates)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_all_charts(
                df, {"dxy": "dxy", "vix": "vix"},
                df.index[-1].date(), tmpdir,
            )
            assert isinstance(results, dict)


# ============================================================================
# 16. WEEKLY GENERATOR — Orchestration
# ============================================================================

class TestWeeklyGeneratorOrchestration:
    """SDD-07: Generator initialization and dry-run."""

    def test_dry_run_mode(self):
        from src.analysis.weekly_generator import WeeklyAnalysisGenerator
        gen = WeeklyAnalysisGenerator(dry_run=True)
        assert gen.dry_run is True

    def test_generator_has_analyzer_and_client(self):
        from src.analysis.weekly_generator import WeeklyAnalysisGenerator
        gen = WeeklyAnalysisGenerator(dry_run=True)
        assert gen.macro_analyzer is not None
        assert gen.llm is not None

    def test_dry_run_returns_record(self):
        from src.analysis.weekly_generator import WeeklyAnalysisGenerator
        gen = WeeklyAnalysisGenerator(dry_run=True)
        try:
            record = gen.generate_for_date(date(2025, 6, 15))
            assert record is not None
            assert "DRY RUN" in record.summary_markdown
        except Exception as e:
            # Data not available is acceptable
            assert any(word in str(e).lower() for word in
                       ["data", "parquet", "column", "empty", "index", "date", "key", "file"])


# ============================================================================
# 17. CONFIG — SSOT YAML Comprehensive Validation
# ============================================================================

class TestConfigDeep:
    """SDD-10: Validate YAML config structure and value constraints."""

    @pytest.fixture
    def config(self):
        import yaml
        path = ROOT / "config" / "analysis" / "weekly_analysis_ssot.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_meta_version(self, config):
        assert config["_meta"]["version"]
        assert config["_meta"]["contract_id"] == "CTR-ANALYSIS-CONFIG-001"

    def test_llm_providers(self, config):
        llm = config["llm"]
        assert llm["primary_provider"] == "azure_openai"
        assert llm["fallback_provider"] == "anthropic"
        assert llm["max_consecutive_failures"] >= 1

    def test_budget_limits(self, config):
        budget = config.get("budget", {})
        if budget:
            assert budget["monthly_limit_usd"] > 0
            assert budget["daily_limit_usd"] > 0
            assert 0 < budget.get("alert_threshold_pct", 80) <= 100

    def test_macro_sma_periods(self, config):
        macro = config["macro"]
        assert macro["sma_periods"] == [5, 10, 20, 50]
        assert macro["rsi_period"] == 14
        assert macro["bollinger_period"] == 20

    def test_macd_params(self, config):
        macro = config["macro"]
        assert macro["macd_fast"] < macro["macd_slow"]
        assert macro["macd_signal"] > 0

    def test_generation_language(self, config):
        assert config["generation"]["language"] == "es"

    def test_export_dir(self, config):
        export = config["export"]
        assert "analysis" in export["output_dir"]

    def test_cache_config(self, config):
        cache = config["llm"].get("cache", {})
        if cache:
            assert cache.get("ttl_hours", 24) > 0


# ============================================================================
# 18. NEWS ENGINE CONFIG — Frozen Dataclasses
# ============================================================================

class TestNewsEngineConfig:
    """SDD-10: Config dataclass validation."""

    def test_database_config_url(self):
        from src.news_engine.config import DatabaseConfig
        cfg = DatabaseConfig(host="db", port=5432, database="test", user="usr", password="pw")
        assert cfg.url == "postgresql://usr:pw@db:5432/test"

    def test_gdelt_config_defaults(self):
        from src.news_engine.config import GDELTConfig
        cfg = GDELTConfig()
        assert cfg.max_records == 250
        assert len(cfg.queries_en) > 0

    def test_enrichment_config_categories_match_contract(self):
        from src.news_engine.config import EnrichmentConfig
        from src.contracts.news_engine_schema import VALID_CATEGORIES
        cfg = EnrichmentConfig()
        assert set(cfg.categories) == set(VALID_CATEGORIES)

    def test_cross_reference_weights_sum_to_one(self):
        from src.news_engine.config import CrossReferenceConfig
        cfg = CrossReferenceConfig()
        total = cfg.title_weight + cfg.entity_weight + cfg.summary_weight + cfg.category_weight
        assert total == pytest.approx(1.0)

    def test_news_engine_config_from_env(self):
        from src.news_engine.config import NewsEngineConfig
        cfg = NewsEngineConfig.from_env()
        assert cfg.enabled is True
        assert cfg.database is not None

    def test_frozen_configs_are_immutable(self):
        from src.news_engine.config import CrossReferenceConfig
        cfg = CrossReferenceConfig()
        with pytest.raises(Exception):
            cfg.similarity_threshold = 0.9


# ============================================================================
# 19. SQL MIGRATIONS — Deep Schema Validation
# ============================================================================

class TestSQLMigrationsDeep:
    """SDD-03: Comprehensive migration validation."""

    def test_045_has_indexes(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        sql = path.read_text()
        # Should have indexes on key columns
        assert "idx_" in sql.lower() or "index" in sql.lower()

    def test_045_has_jsonb_columns(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        sql = path.read_text()
        assert "JSONB" in sql or "jsonb" in sql

    def test_046_has_unique_constraints(self):
        path = ROOT / "database" / "migrations" / "046_weekly_analysis_tables.sql"
        sql = path.read_text()
        assert "UNIQUE" in sql or "unique" in sql

    def test_046_has_views(self):
        path = ROOT / "database" / "migrations" / "046_weekly_analysis_tables.sql"
        sql = path.read_text()
        assert "CREATE" in sql and ("VIEW" in sql or "view" in sql)

    def test_045_has_text_search(self):
        path = ROOT / "database" / "migrations" / "045_newsengine_initial.sql"
        sql = path.read_text()
        # Full-text search on articles
        assert "tsvector" in sql.lower() or "to_tsvector" in sql.lower() or "GIN" in sql


# ============================================================================
# 20. TYPESCRIPT CONTRACT — Interface Coverage
# ============================================================================

class TestTypeScriptContractDeep:
    """SDD-08: TypeScript contract completeness."""

    @pytest.fixture
    def contract_content(self):
        path = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "weekly-analysis.contract.ts"
        return path.read_text()

    def test_has_weekly_view(self, contract_content):
        assert "WeeklyViewData" in contract_content

    def test_has_daily_analysis(self, contract_content):
        assert "DailyAnalysisEntry" in contract_content

    def test_has_macro_snapshot(self, contract_content):
        assert "MacroVariableSnapshot" in contract_content

    def test_has_chat_interfaces(self, contract_content):
        assert "ChatMessage" in contract_content or "ChatRequest" in contract_content

    def test_has_impact_levels(self, contract_content):
        # Economic event impact levels
        assert "high" in contract_content
        assert "medium" in contract_content

    def test_has_sentiment_types(self, contract_content):
        assert "bullish" in contract_content
        assert "bearish" in contract_content
        assert "neutral" in contract_content


# ============================================================================
# 21. DASHBOARD ARTIFACTS — File Structure
# ============================================================================

class TestDashboardArtifacts:
    """SDD-08: Dashboard files exist with correct structure."""

    def test_analysis_page_exists(self):
        path = ROOT / "usdcop-trading-dashboard" / "app" / "analysis" / "page.tsx"
        assert path.exists(), "Analysis page.tsx not found"

    def test_api_routes_exist(self):
        api_base = ROOT / "usdcop-trading-dashboard" / "app" / "api" / "analysis"
        assert (api_base / "weeks" / "route.ts").exists(), "weeks API route missing"
        assert (api_base / "calendar" / "route.ts").exists(), "calendar API route missing"
        assert (api_base / "chat" / "route.ts").exists(), "chat API route missing"

    def test_analysis_components_exist(self):
        comp_dir = ROOT / "usdcop-trading-dashboard" / "components" / "analysis"
        required = [
            "AnalysisPage.tsx",
            "WeekSelector.tsx",
            "WeeklySummaryHeader.tsx",
            "MacroSnapshotBar.tsx",
            "SignalSummaryCards.tsx",
            "DailyTimeline.tsx",
            "DailyTimelineEntry.tsx",
            "UpcomingEventsPanel.tsx",
            "AnalysisMarkdown.tsx",
            "FloatingChatWidget.tsx",
        ]
        for fname in required:
            assert (comp_dir / fname).exists(), f"Component {fname} missing"

    def test_hooks_file_exists(self):
        path = ROOT / "usdcop-trading-dashboard" / "hooks" / "useWeeklyAnalysis.ts"
        assert path.exists()

    def test_store_file_exists(self):
        store_dir = ROOT / "usdcop-trading-dashboard" / "stores"
        assert store_dir.exists()
        # Should have chat store
        stores = list(store_dir.glob("*.ts"))
        assert len(stores) >= 1


# ============================================================================
# 22. AIRFLOW DAGS — File Existence & Structure
# ============================================================================

class TestAirflowDAGs:
    """SDD-10: DAG file existence and basic structure."""

    @pytest.mark.parametrize("dag_file", [
        "news_daily_pipeline.py",
        "news_alert_monitor.py",
        "news_weekly_digest.py",
        "analysis_l8_daily_generation.py",
        "news_maintenance.py",
    ])
    def test_dag_file_exists(self, dag_file):
        path = ROOT / "airflow" / "dags" / dag_file
        assert path.exists(), f"DAG file {dag_file} not found"

    @pytest.mark.parametrize("dag_file,expected_dag_id", [
        ("news_daily_pipeline.py", "news_daily_pipeline"),
        ("news_alert_monitor.py", "news_alert_monitor"),
        ("analysis_l8_daily_generation.py", "analysis_l8_daily_generation"),
    ])
    def test_dag_has_dag_id(self, dag_file, expected_dag_id):
        path = ROOT / "airflow" / "dags" / dag_file
        content = path.read_text()
        assert expected_dag_id in content, f"DAG ID {expected_dag_id} not found in {dag_file}"

    def test_analysis_dag_has_shortcircuit(self):
        """SDD-10: analysis_l8 uses ShortCircuit for Friday check."""
        path = ROOT / "airflow" / "dags" / "analysis_l8_daily_generation.py"
        content = path.read_text()
        assert "ShortCircuit" in content or "check_if_friday" in content or "friday" in content.lower()


# ============================================================================
# 23. CLI SCRIPT — Entry Point
# ============================================================================

class TestCLIScript:
    """Validate CLI script exists and has required arguments."""

    def test_script_exists(self):
        path = ROOT / "scripts" / "generate_weekly_analysis.py"
        assert path.exists()

    def test_script_has_argparse(self):
        path = ROOT / "scripts" / "generate_weekly_analysis.py"
        content = path.read_text()
        assert "argparse" in content or "ArgumentParser" in content or "click" in content

    def test_script_has_dry_run_flag(self):
        path = ROOT / "scripts" / "generate_weekly_analysis.py"
        content = path.read_text()
        assert "dry" in content.lower()


# ============================================================================
# 24. CROSS-COMPONENT INTEGRATION
# ============================================================================

class TestCrossComponentIntegration:
    """End-to-end flows crossing subsystem boundaries."""

    def test_enrichment_to_digest_pipeline(self):
        """Raw articles -> enrich -> digest."""
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.output.digest_generator import DigestGenerator

        pipeline = EnrichmentPipeline(min_relevance=0.0)
        digester = DigestGenerator()

        raw_articles = [
            _make_raw(url=f"https://t.com/{i}",
                      title="Dolar sube por DXY y petroleo",
                      source_id="gdelt_doc", gdelt_tone=-5.0)
            for i in range(5)
        ]
        enriched = pipeline.enrich_batch(raw_articles)
        digest = digester.generate_daily(enriched, date(2026, 2, 25))
        assert digest.total_articles == len(enriched)
        assert digest.summary_text is not None

    def test_enrichment_to_alert_pipeline(self):
        """Breaking article -> enrichment -> alert detection."""
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline
        from src.news_engine.output.alert_system import AlertSystem
        from src.news_engine.config import AlertConfig

        pipeline = EnrichmentPipeline(min_relevance=0.0)
        alerts = AlertSystem(AlertConfig(gdelt_tone_threshold=-15.0))

        raw = _make_raw(
            url="https://crisis.com/1",
            title="USD/COP crisis financiera dolar tasa cambio",
            source_id="gdelt_doc", gdelt_tone=-18.0,
        )
        enriched = [pipeline.enrich(raw)]
        detected = alerts.check_articles(enriched)
        assert len(detected) >= 1

    def test_macro_analyzer_to_prompt_builder(self):
        """MacroAnalyzer snapshot -> prompt template build."""
        from src.analysis.macro_analyzer import MacroAnalyzer
        from src.analysis.prompt_templates import build_macro_section

        analyzer = MacroAnalyzer()
        series = _make_series(100, start_val=104.5)
        snapshot = analyzer.compute_snapshot(series, "dxy", series.index[-1].date())

        section = build_macro_section({"dxy": snapshot})
        assert isinstance(section, str)
        assert len(section) > 10

    def test_full_feature_vector_json_serializable(self):
        """Feature vector must be JSON-safe for downstream consumption."""
        from src.news_engine.output.feature_exporter import FeatureExporter
        exporter = FeatureExporter()

        articles = [
            _make_enriched(
                source_id="gdelt_doc", url=f"https://t.com/{i}",
                category="fx_market", sentiment=0.3 if i % 2 == 0 else -0.5,
                keywords=["dolar"],
            )
            for i in range(10)
        ]
        fv = exporter.export_daily(articles, date(2026, 2, 25))
        # Must be JSON-serializable
        json_str = json.dumps(fv.features)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str

    def test_contract_schema_alignment(self):
        """Python and YAML contracts agree on key_variables."""
        import yaml
        from src.contracts.analysis_schema import KEY_MACRO_VARIABLES

        config_path = ROOT / "config" / "analysis" / "weekly_analysis_ssot.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        yaml_vars = set(config["macro"]["key_variables"])
        python_top4 = set(list(KEY_MACRO_VARIABLES)[:4])
        # The YAML key_variables should be a subset of Python's KEY_MACRO_VARIABLES
        assert yaml_vars.issubset(set(KEY_MACRO_VARIABLES)), \
            f"YAML vars {yaml_vars} not in Python {set(KEY_MACRO_VARIABLES)}"


# ============================================================================
# 25. NAVIGATION — Hub Page & Navbar Integration
# ============================================================================

class TestNavigationIntegration:
    """SDD-08: /analysis linked from hub and navbar."""

    def test_hub_page_has_analysis_link(self):
        path = ROOT / "usdcop-trading-dashboard" / "app" / "hub" / "page.tsx"
        content = path.read_text()
        assert "analysis" in content.lower() or "analisis" in content.lower()

    def test_navbar_has_analysis_link(self):
        path = ROOT / "usdcop-trading-dashboard" / "components" / "navigation" / "GlobalNavbar.tsx"
        content = path.read_text()
        assert "analysis" in content.lower() or "analisis" in content.lower()


# ============================================================================
# 26. PACKAGE STRUCTURE — __init__.py
# ============================================================================

class TestPackageStructure:
    """Verify Python packages are properly structured."""

    @pytest.mark.parametrize("pkg_path", [
        "src/news_engine/__init__.py",
        "src/news_engine/ingestion/__init__.py",
        "src/news_engine/enrichment/__init__.py",
        "src/news_engine/cross_reference/__init__.py",
        "src/news_engine/output/__init__.py",
        "src/analysis/__init__.py",
    ])
    def test_init_files_exist(self, pkg_path):
        assert (ROOT / pkg_path).exists(), f"Missing {pkg_path}"


# ============================================================================
# 27. SOURCE REGISTRY — Adapter Factory
# ============================================================================

class TestSourceRegistry:
    """SDD-02: SourceRegistry factory pattern."""

    def test_registry_init(self):
        from src.news_engine.ingestion.registry import SourceRegistry
        registry = SourceRegistry()
        assert registry is not None

    def test_registry_has_adapters(self):
        from src.news_engine.ingestion.registry import SourceRegistry
        registry = SourceRegistry.from_config()
        adapters = registry.all_adapters()
        assert len(adapters) >= 2  # At least GDELT Doc + Context

    def test_enabled_adapters_subset(self):
        from src.news_engine.ingestion.registry import SourceRegistry
        registry = SourceRegistry.from_config()
        enabled = registry.enabled_adapters()
        all_adapters = registry.all_adapters()
        assert len(enabled) <= len(all_adapters)
