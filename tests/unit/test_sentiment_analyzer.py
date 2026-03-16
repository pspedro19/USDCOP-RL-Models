"""
Unit tests for the Hybrid Sentiment Analyzer.

Tests cover:
- FX impact rules (keyword-based adjustments)
- Language detection
- Score blending with weight redistribution
- Label thresholding
- LLM score parsing
- Batch analysis
- Legacy fallback (sentiment.py)
- SentimentResult dataclass
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.analysis.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    FXImpactRules,
    detect_language,
    _title_hash,
    _score_to_label,
    get_analyzer,
    reset_analyzer,
)


# ---------------------------------------------------------------------------
# FXImpactRules
# ---------------------------------------------------------------------------

class TestFXImpactRules:
    """Test USDCOP-specific FX impact rules."""

    def test_oil_up_positive_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("Oil prices surge above $80")
        assert adj > 0, "Oil up should be positive for COP"
        assert "oil_up" in rules

    def test_oil_down_negative_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("Crude oil prices fall sharply")
        assert adj < 0, "Oil down should be negative for COP"
        assert "oil_down" in rules

    def test_fed_hike_negative_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("Fed raises interest rates by 25bps")
        assert adj < 0, "Fed hike should be negative for COP"
        assert "fed_hike" in rules

    def test_fed_cut_positive_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("Fed cuts rates amid slowdown")
        assert adj > 0, "Fed cut should be positive for COP"
        assert "fed_cut" in rules

    def test_banrep_cut_negative_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("BanRep recorta tasas de interés")
        assert adj < 0, "BanRep cut should be negative for COP"
        assert "banrep_cut" in rules

    def test_banrep_hike_positive_for_cop(self):
        adj, rules = FXImpactRules.compute_adjustment("BanRep sube tasa de referencia")
        assert adj > 0, "BanRep hike should be positive for COP"

    def test_dollar_strength_negative(self):
        adj, rules = FXImpactRules.compute_adjustment("Dollar strength pushes DXY to highs")
        assert adj < 0
        assert "usd_strength" in rules

    def test_em_crisis_negative(self):
        adj, rules = FXImpactRules.compute_adjustment("Capital flight from emerging markets")
        assert adj < 0

    def test_no_match_zero_adjustment(self):
        adj, rules = FXImpactRules.compute_adjustment("Local weather forecast sunny skies")
        assert adj == 0
        assert rules == []

    def test_max_clamp(self):
        # Multiple positive rules should not exceed MAX_ADJUSTMENT
        title = "Oil prices surge as Fed cuts rates and BanRep sube tasas"
        adj, rules = FXImpactRules.compute_adjustment(title)
        assert adj <= FXImpactRules.MAX_ADJUSTMENT
        assert adj >= -FXImpactRules.MAX_ADJUSTMENT

    def test_content_used_for_matching(self):
        adj, rules = FXImpactRules.compute_adjustment(
            "Financial news update",
            content="The oil prices surge above $80 per barrel"
        )
        assert adj > 0


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_spanish_detected(self):
        assert detect_language("El dólar se debilitó frente al peso colombiano") == "es"

    def test_english_detected(self):
        assert detect_language("Dollar weakens against emerging market currencies") == "en"

    def test_mixed_defaults_to_english(self):
        assert detect_language("Hello world") == "en"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_spanish_financial(self):
        assert detect_language("Las tasas de interés del Banco de la República suben") == "es"


# ---------------------------------------------------------------------------
# Score Blending
# ---------------------------------------------------------------------------

class TestScoreBlending:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(config={
            "multilingual_model": None,  # Disable model loading
            "finbert_model": None,
        })

    def test_blend_all_signals_en(self):
        score, components, conf = self.analyzer._blend_scores(
            "en",
            multilingual_score=0.5,
            finbert_score=0.3,
            llm_score=0.4,
            gdelt_score=0.6,
        )
        assert -1 <= score <= 1
        assert len(components) == 4
        assert conf > 0

    def test_blend_all_signals_es(self):
        score, components, conf = self.analyzer._blend_scores(
            "es",
            multilingual_score=0.5,
            finbert_score=0.3,  # Should be ignored (weight=0 for ES)
            llm_score=0.4,
            gdelt_score=0.6,
        )
        # FinBERT has weight 0 for ES, so only 3 signals used
        assert "finbert" not in components or components.get("finbert") is None
        assert -1 <= score <= 1

    def test_blend_missing_signals_redistributes(self):
        # Only multilingual available for EN
        score, components, conf = self.analyzer._blend_scores(
            "en",
            multilingual_score=0.5,
        )
        # Should use the full weight on multilingual
        assert score == 0.5
        assert "multilingual" in components

    def test_blend_no_signals_returns_zero(self):
        score, components, conf = self.analyzer._blend_scores("en")
        assert score == 0.0
        assert components == {}

    def test_blend_opposing_signals(self):
        score, components, conf = self.analyzer._blend_scores(
            "en",
            multilingual_score=0.8,
            finbert_score=-0.6,
        )
        # Should be somewhere between
        assert -0.6 < score < 0.8
        # Low confidence due to disagreement
        assert conf < 0.5

    def test_blend_agreeing_signals_high_confidence(self):
        score, components, conf = self.analyzer._blend_scores(
            "en",
            multilingual_score=0.5,
            finbert_score=0.5,
            llm_score=0.5,
        )
        assert conf > 0.9  # All agree


# ---------------------------------------------------------------------------
# Label Thresholding
# ---------------------------------------------------------------------------

class TestLabelThresholding:
    def test_positive(self):
        assert _score_to_label(0.3) == "positive"

    def test_negative(self):
        assert _score_to_label(-0.5) == "negative"

    def test_neutral(self):
        assert _score_to_label(0.1) == "neutral"

    def test_at_threshold_positive(self):
        assert _score_to_label(0.15) == "positive"

    def test_at_threshold_negative(self):
        assert _score_to_label(-0.15) == "negative"

    def test_just_below_threshold(self):
        assert _score_to_label(0.14) == "neutral"


# ---------------------------------------------------------------------------
# LLM Score Parsing
# ---------------------------------------------------------------------------

class TestLLMScoreParsing:
    def test_valid_json(self):
        content = '{"scores": [{"id": 1, "score": 0.5}, {"id": 2, "score": -0.3}]}'
        titles = ["Oil prices rise", "Fed hikes rates"]
        result = SentimentAnalyzer._parse_llm_scores(content, titles)
        assert len(result) == 2
        assert result[_title_hash("Oil prices rise")] == 0.5
        assert result[_title_hash("Fed hikes rates")] == -0.3

    def test_markdown_wrapped_json(self):
        content = '```json\n{"scores": [{"id": 1, "score": 0.7}]}\n```'
        titles = ["Dollar falls"]
        result = SentimentAnalyzer._parse_llm_scores(content, titles)
        assert len(result) == 1
        assert result[_title_hash("Dollar falls")] == 0.7

    def test_score_clamping(self):
        content = '{"scores": [{"id": 1, "score": 2.5}]}'
        titles = ["Test"]
        result = SentimentAnalyzer._parse_llm_scores(content, titles)
        assert result[_title_hash("Test")] == 1.0  # Clamped

    def test_invalid_json_returns_empty(self):
        result = SentimentAnalyzer._parse_llm_scores("not json", ["Test"])
        assert result == {}

    def test_out_of_range_id_ignored(self):
        content = '{"scores": [{"id": 999, "score": 0.5}]}'
        titles = ["Only one"]
        result = SentimentAnalyzer._parse_llm_scores(content, titles)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# analyze_single (without models)
# ---------------------------------------------------------------------------

class TestAnalyzeSingle:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(config={
            "multilingual_model": None,
            "finbert_model": None,
        })

    def test_with_gdelt_tone_only(self):
        result = self.analyzer.analyze_single(
            title="Some article",
            gdelt_tone=10.0,
        )
        assert isinstance(result, SentimentResult)
        assert result.score > 0  # GDELT tone 10 -> positive
        assert result.label == "positive"

    def test_with_llm_score_only(self):
        result = self.analyzer.analyze_single(
            title="Some article",
            llm_score=-0.6,
        )
        assert result.score < 0
        assert result.label == "negative"

    def test_fx_rules_applied(self):
        result = self.analyzer.analyze_single(
            title="Oil prices surge above $80",
            llm_score=0.0,  # Neutral LLM score
        )
        # FX rules should push it positive
        assert result.fx_adjusted_score > result.score
        assert "fx_rules" in result.components

    def test_empty_title_returns_default(self):
        result = self.analyzer.analyze_single(title="")
        assert result.score == 0.0
        assert result.label == "neutral"

    def test_spanish_article_no_finbert(self):
        result = self.analyzer.analyze_single(
            title="El dólar se debilitó frente al peso colombiano",
            language="es",
            llm_score=0.5,
        )
        # FinBERT should not be used for Spanish
        assert "finbert" not in result.components


# ---------------------------------------------------------------------------
# analyze_batch
# ---------------------------------------------------------------------------

class TestAnalyzeBatch:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(config={
            "multilingual_model": None,
            "finbert_model": None,
        })

    def test_batch_returns_correct_count(self):
        articles = [
            {"title": "Oil rises", "gdelt_tone": 5.0},
            {"title": "Fed hikes", "gdelt_tone": -3.0},
            {"title": "Market neutral"},
        ]
        results = self.analyzer.analyze_batch(articles)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_batch_with_llm_scores(self):
        articles = [
            {"title": "Oil rises"},
            {"title": "Fed hikes"},
        ]
        llm_scores = {
            _title_hash("Oil rises"): 0.5,
            _title_hash("Fed hikes"): -0.7,
        }
        results = self.analyzer.analyze_batch(articles, llm_scores=llm_scores)
        assert results[0].score > 0
        assert results[1].score < 0


# ---------------------------------------------------------------------------
# score_batch_with_llm
# ---------------------------------------------------------------------------

class TestLLMBatch:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer(config={
            "multilingual_model": None,
            "finbert_model": None,
        })

    def test_llm_batch_scoring(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "content": '{"scores": [{"id": 1, "score": 0.5}, {"id": 2, "score": -0.3}]}',
            "cost_usd": 0.001,
            "tokens_used": 100,
        }

        articles = [
            {"title": "Oil prices surge above $80"},
            {"title": "Fed mantiene tasas sin cambios"},
        ]

        scores = self.analyzer.score_batch_with_llm(articles, mock_llm, cache_key="test")
        assert len(scores) == 2
        mock_llm.generate.assert_called_once()

    def test_llm_batch_empty_articles(self):
        mock_llm = MagicMock()
        scores = self.analyzer.score_batch_with_llm([], mock_llm)
        assert scores == {}
        mock_llm.generate.assert_not_called()

    def test_llm_batch_failure_returns_empty(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        articles = [{"title": "Some article title here"}]
        scores = self.analyzer.score_batch_with_llm(articles, mock_llm)
        assert scores == {}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def teardown_method(self):
        reset_analyzer()

    def test_get_analyzer_returns_same_instance(self):
        a1 = get_analyzer(config={"multilingual_model": None, "finbert_model": None})
        a2 = get_analyzer()
        assert a1 is a2

    def test_reset_clears_singleton(self):
        a1 = get_analyzer(config={"multilingual_model": None, "finbert_model": None})
        reset_analyzer()
        a2 = get_analyzer(config={"multilingual_model": None, "finbert_model": None})
        assert a1 is not a2


# ---------------------------------------------------------------------------
# Legacy fallback (sentiment.py)
# ---------------------------------------------------------------------------

class TestLegacyFallback:
    def test_legacy_with_gdelt_tone(self):
        from src.news_engine.enrichment.sentiment import _legacy_analyze_sentiment
        score, label = _legacy_analyze_sentiment("Test", gdelt_tone=10.0)
        assert score is not None
        assert score > 0
        assert label == "positive"

    def test_legacy_with_negative_gdelt(self):
        from src.news_engine.enrichment.sentiment import _legacy_analyze_sentiment
        score, label = _legacy_analyze_sentiment("Test", gdelt_tone=-15.0)
        assert score < 0
        assert label == "negative"


# ---------------------------------------------------------------------------
# Integration: sentiment.py delegates to hybrid analyzer
# ---------------------------------------------------------------------------

class TestSentimentDelegation:
    def teardown_method(self):
        reset_analyzer()

    def test_analyze_sentiment_delegates(self):
        """Verify analyze_sentiment() from enrichment module produces non-None results."""
        from src.news_engine.enrichment.sentiment import analyze_sentiment
        score, label = analyze_sentiment("Oil prices surge", gdelt_tone=5.0)
        assert score is not None
        assert label is not None


# ---------------------------------------------------------------------------
# Title hash
# ---------------------------------------------------------------------------

class TestTitleHash:
    def test_consistent(self):
        assert _title_hash("Hello World") == _title_hash("hello world")

    def test_strips_whitespace(self):
        assert _title_hash("  hello  ") == _title_hash("hello")

    def test_truncates_long_titles(self):
        long = "a" * 200
        h = _title_hash(long)
        assert len(h) == 100
