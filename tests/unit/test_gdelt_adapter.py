"""
Unit tests for GDELTDocAdapter.

Tests query building, date formatting, JSON error handling, date windowing,
article parsing, rate limiting, deduplication, and ABC compliance.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.news_engine.config import GDELTConfig
from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter, GDELTContextAdapter
from src.news_engine.ingestion.base_adapter import SourceAdapter
from src.news_engine.models import RawArticle


# ============================================================================
# Fixtures: sample GDELT articles
# ============================================================================

SAMPLE_GDELT_ARTICLES = {
    "articles": [
        {
            "url": "https://www.reuters.com/colombia-peso-rally-2026",
            "title": "Colombian peso rallies on strong exports",
            "seendate": "20260225T143000Z",
            "language": "English",
            "sourcecountry": "Colombia",
            "tone": "-2.5,3.1,5.6,1.2,8.3,45.2",
            "socialimage": "https://img.reuters.com/peso.jpg",
            "articlecount": 15,
        },
        {
            "url": "https://www.bloomberg.com/colombia-rates-2026",
            "title": "Banco Republica holds rates steady",
            "seendate": "20260225T120000Z",
            "language": "English",
            "sourcecountry": "United States",
            "tone": "1.2,2.3,3.4,0.5,4.5,30.1",
            "socialimage": "",
            "articlecount": 8,
        },
        {
            "url": "https://www.eltiempo.com/colombia-inflacion-2026",
            "title": "Inflacion en Colombia baja al 5.2%",
            "seendate": "20260224T180000Z",
            "language": "Spanish",
            "sourcecountry": "Colombia",
            "tone": "3.5,1.2,4.7,0.8,6.2,22.0",
            "socialimage": "",
            "articlecount": 5,
        },
    ]
}


# ============================================================================
# Test: ABC compliance
# ============================================================================

class TestSourceAdapterInterface:
    def test_doc_is_subclass(self):
        assert issubclass(GDELTDocAdapter, SourceAdapter)

    def test_context_is_subclass(self):
        assert issubclass(GDELTContextAdapter, SourceAdapter)

    def test_doc_has_source_id(self):
        adapter = GDELTDocAdapter()
        assert adapter.source_id == "gdelt_doc"

    def test_doc_has_source_name(self):
        adapter = GDELTDocAdapter()
        assert adapter.source_name == "GDELT DOC 2.0"

    def test_context_has_source_id(self):
        adapter = GDELTContextAdapter()
        assert adapter.source_id == "gdelt_context"

    def test_implements_fetch_latest(self):
        assert callable(GDELTDocAdapter.fetch_latest)

    def test_implements_fetch_historical(self):
        assert callable(GDELTDocAdapter.fetch_historical)

    def test_implements_health_check(self):
        assert callable(GDELTDocAdapter.health_check)


# ============================================================================
# Test: Query building
# ============================================================================

class TestQueryBuilding:
    def test_default_queries_include_en_and_es(self):
        adapter = GDELTDocAdapter()
        queries = adapter._build_query_list()
        langs = {lang for _, lang in queries}
        assert "en" in langs
        assert "es" in langs

    def test_default_queries_count(self):
        adapter = GDELTDocAdapter()
        queries = adapter._build_query_list()
        # 6 EN + 5 ES = 11 total
        assert len(queries) == 11

    def test_es_disabled(self):
        cfg = GDELTConfig(use_es_queries=False)
        adapter = GDELTDocAdapter(config=cfg)
        queries = adapter._build_query_list()
        for _, lang in queries:
            assert lang == "en"

    def test_custom_queries(self):
        cfg = GDELTConfig(
            queries_en=("TestQuery sourcelang:english",),
            queries_es=(),
            use_es_queries=False,
        )
        adapter = GDELTDocAdapter(config=cfg)
        queries = adapter._build_query_list()
        assert len(queries) == 1
        assert queries[0] == ("TestQuery sourcelang:english", "en")

    def test_queries_have_sourcelang(self):
        """All default queries should include sourcelang filter."""
        adapter = GDELTDocAdapter()
        queries = adapter._build_query_list()
        for q, lang in queries:
            assert f"sourcelang:{lang.replace('en', 'english').replace('es', 'spanish')}" in q

    def test_or_queries_have_parentheses(self):
        """OR queries must be wrapped in parentheses per GDELT syntax."""
        adapter = GDELTDocAdapter()
        queries = adapter._build_query_list()
        for q, _ in queries:
            if " OR " in q:
                # The OR part should be in parens
                # Extract the part before sourcelang
                parts = q.split(" sourcelang:")
                query_part = parts[0].strip()
                assert query_part.startswith("(") and ")" in query_part


# ============================================================================
# Test: Article parsing
# ============================================================================

class TestParseArticle:
    def setup_method(self):
        self.adapter = GDELTDocAdapter()

    def test_parses_standard_article(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art is not None
        assert "peso" in art.title.lower()
        assert art.source_id == "gdelt_doc"
        assert art.url == "https://www.reuters.com/colombia-peso-rally-2026"

    def test_parses_published_at(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.published_at is not None
        assert art.published_at.year == 2026
        assert art.published_at.month == 2
        assert art.published_at.day == 25

    def test_published_at_has_utc_timezone(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.published_at.tzinfo is not None

    def test_extracts_country_colombia(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.country_focus == "CO"

    def test_extracts_country_default_us(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][1], "en"
        )
        assert art.country_focus == "US"

    def test_extracts_tone(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.gdelt_tone == pytest.approx(-2.5)

    def test_extracts_volume(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.gdelt_volume == 15

    def test_extracts_image_url(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.image_url == "https://img.reuters.com/peso.jpg"

    def test_stores_raw_json(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][0], "en"
        )
        assert art.raw_json is not None
        assert art.raw_json["url"] == "https://www.reuters.com/colombia-peso-rally-2026"

    def test_uses_gdelt_language(self):
        art = self.adapter._parse_article(
            SAMPLE_GDELT_ARTICLES["articles"][2], "en"
        )
        assert art.language == "Spanish"

    def test_missing_url_returns_none(self):
        art = self.adapter._parse_article(
            {"title": "Test", "seendate": "20260225T120000Z"}, "en"
        )
        assert art is None

    def test_missing_title_returns_none(self):
        art = self.adapter._parse_article(
            {"url": "https://test.com", "seendate": "20260225T120000Z"}, "en"
        )
        assert art is None

    def test_missing_seendate_returns_none(self):
        art = self.adapter._parse_article(
            {"url": "https://test.com", "title": "Test"}, "en"
        )
        assert art is None

    def test_invalid_seendate_returns_none(self):
        art = self.adapter._parse_article(
            {"url": "https://test.com", "title": "Test", "seendate": "not-a-date"}, "en"
        )
        assert art is None


# ============================================================================
# Test: JSON error handling
# ============================================================================

class TestSafeParseJson:
    def test_valid_json(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.text = '{"articles": []}'
        resp.json.return_value = {"articles": []}
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result == {"articles": []}

    def test_rate_limit_text_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/plain"}
        resp.text = "Please limit requests to one every 5 seconds."
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result is None

    def test_html_error_page(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body>Error</body></html>"
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result is None

    def test_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.text = ""
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result is None

    def test_500_status(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.headers = {}
        resp.text = "Internal Server Error"
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result is None

    def test_malformed_json(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.text = '{invalid json'
        resp.json.side_effect = json.JSONDecodeError("", "", 0)
        result = GDELTDocAdapter._safe_parse_json(resp)
        assert result is None


# ============================================================================
# Test: Tone parsing
# ============================================================================

class TestParseTone:
    def test_standard_tone(self):
        assert GDELTDocAdapter._parse_tone("-2.5,3.1,5.6") == pytest.approx(-2.5)

    def test_positive_tone(self):
        assert GDELTDocAdapter._parse_tone("3.5,1.2,4.7") == pytest.approx(3.5)

    def test_single_value(self):
        assert GDELTDocAdapter._parse_tone("1.5") == pytest.approx(1.5)

    def test_empty_string(self):
        assert GDELTDocAdapter._parse_tone("") is None

    def test_none_input(self):
        assert GDELTDocAdapter._parse_tone(None) is None

    def test_garbage(self):
        assert GDELTDocAdapter._parse_tone("not,a,number") is None


# ============================================================================
# Test: Date windowing
# ============================================================================

class TestGenerateWindows:
    def test_single_window(self):
        start = datetime(2026, 2, 1)
        end = datetime(2026, 2, 5)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        assert len(windows) == 1
        assert windows[0] == (start, end)

    def test_multiple_windows(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 2, 28)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        # ~59 days / 7 days = ~8-9 windows
        assert 8 <= len(windows) <= 9

    def test_exact_window_boundary(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 7)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        assert len(windows) == 1

    def test_windows_cover_full_range(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 3, 1)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        assert windows[0][0] == start
        assert windows[-1][1] == end

    def test_empty_range(self):
        start = datetime(2026, 3, 1)
        end = datetime(2026, 2, 1)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        assert len(windows) == 0

    def test_same_date(self):
        start = datetime(2026, 2, 1)
        end = datetime(2026, 2, 1)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        assert len(windows) == 0

    def test_three_month_range(self):
        start = datetime(2025, 12, 1)
        end = datetime(2026, 2, 28)
        windows = GDELTDocAdapter._generate_windows(start, end, 7)
        # ~90 days / 7 = ~13 windows
        assert 12 <= len(windows) <= 14


# ============================================================================
# Test: Country extraction
# ============================================================================

class TestExtractCountry:
    def test_colombia_lowercase(self):
        assert GDELTDocAdapter._extract_country(
            {"sourcecountry": "colombia"}
        ) == "CO"

    def test_colombia_mixed_case(self):
        assert GDELTDocAdapter._extract_country(
            {"sourcecountry": "Colombia"}
        ) == "CO"

    def test_co_code(self):
        assert GDELTDocAdapter._extract_country(
            {"sourcecountry": "CO"}
        ) == "CO"

    def test_us_default(self):
        assert GDELTDocAdapter._extract_country(
            {"sourcecountry": "United States"}
        ) == "US"

    def test_empty_defaults_to_us(self):
        assert GDELTDocAdapter._extract_country({}) == "US"


# ============================================================================
# Test: Deduplication
# ============================================================================

class TestDeduplication:
    def setup_method(self):
        self.adapter = GDELTDocAdapter()

    def test_dedup_by_url(self):
        articles = [
            RawArticle(
                url="https://test.com/article-1",
                title="Article 1",
                source_id="gdelt_doc",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://test.com/article-1",
                title="Article 1 (dup)",
                source_id="gdelt_doc",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://test.com/article-2",
                title="Article 2",
                source_id="gdelt_doc",
                published_at=datetime.now(timezone.utc),
            ),
        ]
        result = self.adapter._deduplicate(articles)
        assert len(result) == 2

    def test_dedup_preserves_first(self):
        articles = [
            RawArticle(
                url="https://test.com/same",
                title="First",
                source_id="gdelt_doc",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://test.com/same",
                title="Second",
                source_id="gdelt_doc",
                published_at=datetime.now(timezone.utc),
            ),
        ]
        result = self.adapter._deduplicate(articles)
        assert result[0].title == "First"


# ============================================================================
# Test: Config defaults
# ============================================================================

class TestGDELTConfig:
    def test_rate_limit_is_8_seconds(self):
        cfg = GDELTConfig()
        assert cfg.rate_limit_seconds == 8.0

    def test_historical_window_is_7_days(self):
        cfg = GDELTConfig()
        assert cfg.historical_window_days == 7

    def test_max_records_250(self):
        cfg = GDELTConfig()
        assert cfg.max_records == 250

    def test_request_timeout_30(self):
        cfg = GDELTConfig()
        assert cfg.request_timeout == 30

    def test_use_es_queries_default_true(self):
        cfg = GDELTConfig()
        assert cfg.use_es_queries is True

    def test_en_queries_have_or_syntax(self):
        cfg = GDELTConfig()
        for q in cfg.queries_en:
            if " OR " in q:
                parts = q.split(" sourcelang:")
                assert parts[0].strip().startswith("(")

    def test_es_queries_have_or_syntax(self):
        cfg = GDELTConfig()
        for q in cfg.queries_es:
            if " OR " in q:
                parts = q.split(" sourcelang:")
                assert parts[0].strip().startswith("(")


# ============================================================================
# Test: Context adapter
# ============================================================================

class TestContextAdapter:
    def test_fetch_latest_returns_empty(self):
        adapter = GDELTContextAdapter()
        assert adapter.fetch_latest() == []

    def test_fetch_historical_returns_empty(self):
        adapter = GDELTContextAdapter()
        result = adapter.fetch_historical(
            datetime(2026, 1, 1), datetime(2026, 2, 1)
        )
        assert result == []

    def test_source_id(self):
        adapter = GDELTContextAdapter()
        assert adapter.source_id == "gdelt_context"
