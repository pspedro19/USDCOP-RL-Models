"""
Tests for PoliticalBiasDetector (Phase 3)
==========================================
"""

from unittest.mock import MagicMock

import pytest


class TestMediaBiasExpanded:
    def test_has_60_plus_outlets(self):
        from src.analysis.bias_detector import MEDIA_BIAS_EXPANDED

        assert len(MEDIA_BIAS_EXPANDED) >= 60, (
            f"Expected >= 60 outlets, got {len(MEDIA_BIAS_EXPANDED)}"
        )

    def test_all_entries_have_valid_bias(self):
        from src.analysis.bias_detector import MEDIA_BIAS_EXPANDED

        valid_labels = {"left", "center-left", "center", "center-right", "right"}
        for domain, (bias, _) in MEDIA_BIAS_EXPANDED.items():
            assert bias in valid_labels, f"{domain} has invalid bias: {bias}"

    def test_all_entries_have_valid_factuality(self):
        from src.analysis.bias_detector import MEDIA_BIAS_EXPANDED

        valid = {"high", "mixed", "low"}
        for domain, (_, fact) in MEDIA_BIAS_EXPANDED.items():
            assert fact in valid, f"{domain} has invalid factuality: {fact}"

    def test_has_colombian_sources(self):
        from src.analysis.bias_detector import MEDIA_BIAS_EXPANDED

        colombian = [d for d in MEDIA_BIAS_EXPANDED if ".co" in d]
        assert len(colombian) >= 10, f"Expected >= 10 Colombian sources, got {len(colombian)}"

    def test_has_international_sources(self):
        from src.analysis.bias_detector import MEDIA_BIAS_EXPANDED

        international = {"reuters.com", "bloomberg.com", "ft.com", "bbc.com", "cnn.com"}
        for src in international:
            assert src in MEDIA_BIAS_EXPANDED, f"Missing international source: {src}"


class TestGetMediaBiasExpanded:
    def test_exact_match(self):
        from src.analysis.bias_detector import get_media_bias_expanded

        bias, fact = get_media_bias_expanded("reuters.com")
        assert bias == "center"
        assert fact == "high"

    def test_substring_match(self):
        from src.analysis.bias_detector import get_media_bias_expanded

        bias, fact = get_media_bias_expanded("https://www.reuters.com/article/123")
        assert bias == "center"

    def test_case_insensitive(self):
        from src.analysis.bias_detector import get_media_bias_expanded

        bias, _ = get_media_bias_expanded("Reuters.COM")
        assert bias == "center"

    def test_unknown_source(self):
        from src.analysis.bias_detector import get_media_bias_expanded

        bias, fact = get_media_bias_expanded("randomsite.xyz")
        assert bias == "unknown"
        assert fact == "unknown"

    def test_empty_source(self):
        from src.analysis.bias_detector import get_media_bias_expanded

        bias, fact = get_media_bias_expanded("")
        assert bias == "unknown"


class TestPoliticalBiasDetector:
    @pytest.fixture
    def detector(self):
        from src.analysis.bias_detector import PoliticalBiasDetector

        return PoliticalBiasDetector()

    @pytest.fixture
    def sample_articles(self):
        return [
            {"title": "Dollar rises", "source": "reuters.com"},
            {"title": "Fed holds rates", "source": "bloomberg.com"},
            {"title": "Peso weakens", "source": "eltiempo.com"},
            {"title": "BanRep decision", "source": "portafolio.co"},
            {"title": "Oil drops", "source": "cnn.com"},
            {"title": "VIX spikes", "source": "foxnews.com"},
            {"title": "Inflation data", "source": "larepublica.co"},
            {"title": "Trade war", "source": "nytimes.com"},
        ]

    def test_analyze_returns_all_fields(self, detector, sample_articles):
        result = detector.analyze(sample_articles)

        assert "source_bias_distribution" in result
        assert "bias_diversity_score" in result
        assert "factuality_distribution" in result
        assert "cluster_bias_assessments" in result
        assert "flagged_articles" in result
        assert "bias_narrative" in result
        assert "total_analyzed" in result

    def test_total_analyzed_matches_input(self, detector, sample_articles):
        result = detector.analyze(sample_articles)
        assert result["total_analyzed"] == len(sample_articles)

    def test_bias_distribution_sums_to_total(self, detector, sample_articles):
        result = detector.analyze(sample_articles)
        dist = result["source_bias_distribution"]
        total = sum(dist.values())
        assert total == len(sample_articles)

    def test_factuality_distribution_sums_to_total(self, detector, sample_articles):
        result = detector.analyze(sample_articles)
        dist = result["factuality_distribution"]
        total = sum(dist.values())
        assert total == len(sample_articles)

    def test_diversity_score_in_range(self, detector, sample_articles):
        result = detector.analyze(sample_articles)
        score = result["bias_diversity_score"]
        assert 0.0 <= score <= 1.0

    def test_diversity_score_higher_for_mixed_sources(self, detector):
        """Diverse sources should yield higher diversity score than single-source."""
        diverse = [
            {"title": "A", "source": "nytimes.com"},         # center-left
            {"title": "B", "source": "foxnews.com"},         # right
            {"title": "C", "source": "reuters.com"},         # center
            {"title": "D", "source": "bloomberg.com"},       # center-right
            {"title": "E", "source": "elespectador.com"},    # center-left
        ]
        uniform = [
            {"title": "A", "source": "reuters.com"},
            {"title": "B", "source": "portafolio.co"},
            {"title": "C", "source": "bbc.com"},
            {"title": "D", "source": "ft.com"},
            {"title": "E", "source": "economist.com"},
        ]

        diverse_result = detector.analyze(diverse)
        uniform_result = detector.analyze(uniform)

        assert diverse_result["bias_diversity_score"] > uniform_result["bias_diversity_score"]

    def test_flagged_articles_excludes_center_and_unknown(self, detector):
        articles = [
            {"title": "A", "source": "reuters.com"},      # center -> not flagged
            {"title": "B", "source": "foxnews.com"},      # right -> flagged
            {"title": "C", "source": "randomsite.xyz"},   # unknown -> not flagged
        ]
        result = detector.analyze(articles)
        assert result["flagged_articles"] == 1

    def test_empty_articles(self, detector):
        result = detector.analyze([])
        assert result["total_analyzed"] == 0
        assert result["bias_diversity_score"] == 0.0
        assert result["flagged_articles"] == 0

    def test_no_llm_calls_without_clusters(self, detector, sample_articles):
        """Without clusters, no LLM calls should be made."""
        mock_llm = MagicMock()
        result = detector.analyze(sample_articles, clusters=None, llm_client=mock_llm)
        mock_llm.generate.assert_not_called()
        assert result["cluster_bias_assessments"] == []
        assert result["bias_narrative"] == ""

    def test_cluster_bias_requires_min_articles(self, detector, sample_articles):
        """Clusters below CLUSTER_MIN_ARTICLES should be skipped."""
        mock_llm = MagicMock()
        small_clusters = [
            {"label": "Small cluster", "article_count": 2, "representative_titles": ["T1", "T2"]},
        ]
        result = detector.analyze(sample_articles, clusters=small_clusters, llm_client=mock_llm)
        mock_llm.generate.assert_not_called()
        assert result["cluster_bias_assessments"] == []

    def test_cluster_bias_called_for_large_clusters(self, detector, sample_articles):
        """Clusters >= CLUSTER_MIN_ARTICLES should trigger LLM call."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "content": "balanced|0.8",
            "cost_usd": 0.001,
            "tokens_used": 30,
        }

        large_clusters = [
            {
                "label": "BanRep decision",
                "article_count": 7,
                "representative_titles": ["T1", "T2", "T3", "T4", "T5"],
            },
        ]
        result = detector.analyze(sample_articles, clusters=large_clusters, llm_client=mock_llm)
        assert mock_llm.generate.call_count >= 1
        assert len(result["cluster_bias_assessments"]) == 1
        assert result["cluster_bias_assessments"][0]["bias_label"] == "balanced"
        assert result["cluster_bias_assessments"][0]["confidence"] == 0.8

    def test_max_5_clusters_analyzed(self, detector, sample_articles):
        """Only first 5 large clusters should be analyzed."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "content": "balanced|0.7",
            "cost_usd": 0.001,
            "tokens_used": 30,
        }

        clusters = [
            {
                "label": f"Cluster {i}",
                "article_count": 10,
                "representative_titles": [f"Title {j}" for j in range(5)],
            }
            for i in range(8)
        ]

        result = detector.analyze(sample_articles, clusters=clusters, llm_client=mock_llm)
        assert len(result["cluster_bias_assessments"]) <= 5

    def test_invalid_llm_response_defaults_to_balanced(self, detector, sample_articles):
        """Invalid LLM response should default to balanced."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "content": "garbage response",
            "cost_usd": 0.001,
            "tokens_used": 30,
        }

        clusters = [
            {
                "label": "Test",
                "article_count": 6,
                "representative_titles": ["T1", "T2", "T3"],
            },
        ]
        result = detector.analyze(sample_articles, clusters=clusters, llm_client=mock_llm)
        assert len(result["cluster_bias_assessments"]) == 1
        assert result["cluster_bias_assessments"][0]["bias_label"] == "balanced"

    def test_llm_failure_handled_gracefully(self, detector, sample_articles):
        """LLM failure should not crash the analysis."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API error")

        clusters = [
            {
                "label": "Crash test",
                "article_count": 10,
                "representative_titles": ["T1", "T2", "T3"],
            },
        ]
        result = detector.analyze(sample_articles, clusters=clusters, llm_client=mock_llm)
        # Should still return valid output, just without cluster assessments
        assert result["total_analyzed"] == len(sample_articles)
        assert result["cluster_bias_assessments"] == []
