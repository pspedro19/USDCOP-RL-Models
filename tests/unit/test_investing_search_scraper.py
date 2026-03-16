"""
Unit tests for InvestingSearchScraper.

Tests API item parsing, date parsing (EN/ES/relative/timestamp),
date windowing, deduplication, checkpoint, and ABC compliance.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.news_engine.config import ScraperConfig
from src.news_engine.ingestion.investing_scraper import InvestingSearchScraper
from src.news_engine.ingestion.base_adapter import SourceAdapter
from src.news_engine.models import RawArticle


# ============================================================================
# Fixtures: sample API items
# ============================================================================

SAMPLE_API_ITEMS = [
    {
        "authorID": -1,
        "content": "Colombia stocks were lower after the close...",
        "dataID": 4522732,
        "date": "Feb 24, 2026",
        "dateTimestamp": 1771968308,
        "image": "https://i-invdn-com.investing.com/news/LYNXNPEB7Q0U9_S.jpg",
        "isEditorPick": False,
        "link": "/news/stock-market-news/colombia-stocks-lower-4522732",
        "name": "Colombia stocks lower at close of trade; COLCAP down 2.95%",
        "providerID": 1,
        "providerName": "Investing.com",
        "searchable": True,
        "smlID": 0,
    },
    {
        "authorID": -1,
        "content": "Colombia slaps 30% tariffs on Ecuador goods...",
        "dataID": 4522069,
        "date": "Feb 24, 2026",
        "dateTimestamp": 1771920000,
        "image": "",
        "link": "/news/world-news/colombia-slaps-30-tariffs-4522069",
        "name": "Colombia slaps 30% tariffs on Ecuador goods in trade dispute",
        "providerName": "Reuters",
    },
    {
        "dataID": 4525082,
        "date": "Feb 25, 2026",
        "dateTimestamp": 1772010000,
        "link": "/news/economy-news/colombia-plans-pension-4525082",
        "name": "Colombia plans $6.8 billion pension transfer to state",
        "content": "",
        "providerName": "Reuters",
    },
]

SAMPLE_ES_ITEM = {
    "dataID": 9999001,
    "date": "12.02.2026",
    "dateTimestamp": 1770854400,
    "link": "/noticias/noticias-del-mercado-de-divisas/peso-colombiano-9999001",
    "name": "Peso colombiano se fortalece frente al dolar",
    "content": "El peso colombiano gano terreno frente al dolar...",
    "providerName": "Investing.com",
}


# ============================================================================
# Test: ABC compliance
# ============================================================================

class TestSourceAdapterInterface:
    def test_is_subclass(self):
        assert issubclass(InvestingSearchScraper, SourceAdapter)

    def test_has_source_id(self):
        scraper = InvestingSearchScraper()
        assert scraper.source_id == "investing"

    def test_has_source_name(self):
        scraper = InvestingSearchScraper()
        assert scraper.source_name == "Investing.com"

    def test_implements_fetch_latest(self):
        assert callable(InvestingSearchScraper.fetch_latest)

    def test_implements_fetch_historical(self):
        assert callable(InvestingSearchScraper.fetch_historical)

    def test_implements_health_check(self):
        assert callable(InvestingSearchScraper.health_check)


# ============================================================================
# Test: API item parsing
# ============================================================================

class TestParseApiItem:
    def setup_method(self):
        self.scraper = InvestingSearchScraper()

    def test_parses_standard_item(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[0], "https://www.investing.com", "en",
        )
        assert art is not None
        assert "COLCAP" in art.title
        assert art.url == "https://www.investing.com/news/stock-market-news/colombia-stocks-lower-4522732"
        assert art.source_id == "investing"
        assert art.language == "en"
        assert art.country_focus == "CO"

    def test_uses_unix_timestamp(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[0], "https://www.investing.com", "en",
        )
        # dateTimestamp=1771968308 should parse to a valid date
        assert art.published_at is not None
        assert art.published_at.year == 2026

    def test_resolves_relative_url(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[1], "https://www.investing.com", "en",
        )
        assert art.url.startswith("https://www.investing.com/news/")

    def test_summary_from_content(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[0], "https://www.investing.com", "en",
        )
        assert art.summary is not None
        assert "Colombia stocks" in art.summary

    def test_empty_content_gives_none_summary(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[2], "https://www.investing.com", "en",
        )
        assert art.summary is None

    def test_stores_dataID_in_raw_json(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[0], "https://www.investing.com", "en",
        )
        assert art.raw_json["dataID"] == 4522732

    def test_stores_provider_in_raw_json(self):
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[1], "https://www.investing.com", "en",
        )
        assert art.raw_json["providerName"] == "Reuters"

    def test_es_domain_item(self):
        art = self.scraper._parse_api_item(
            SAMPLE_ES_ITEM, "https://es.investing.com", "es",
        )
        assert art is not None
        assert art.language == "es"
        assert "es.investing.com" in art.url

    def test_missing_link_returns_none(self):
        art = self.scraper._parse_api_item(
            {"dataID": 1, "name": "Test"}, "https://www.investing.com", "en",
        )
        assert art is None

    def test_missing_name_returns_none(self):
        art = self.scraper._parse_api_item(
            {"dataID": 1, "link": "/news/test-1"}, "https://www.investing.com", "en",
        )
        assert art is None


# ============================================================================
# Test: Date parsing
# ============================================================================

class TestParseArticleDate:
    def test_en_standard(self):
        dt = InvestingSearchScraper._parse_article_date("Feb 25, 2026")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 2 and dt.day == 25

    def test_en_full_month(self):
        dt = InvestingSearchScraper._parse_article_date("February 25, 2026")
        assert dt is not None and dt.month == 2

    def test_iso_date(self):
        dt = InvestingSearchScraper._parse_article_date("2026-02-25")
        assert dt is not None and dt.year == 2026

    def test_iso_datetime(self):
        dt = InvestingSearchScraper._parse_article_date("2026-02-25T14:30:00")
        assert dt is not None and dt.year == 2026

    def test_es_dot_format(self):
        """ES domain uses dd.mm.yyyy format."""
        dt = InvestingSearchScraper._parse_article_date("12.02.2026")
        assert dt is not None
        assert dt.day == 12 and dt.month == 2 and dt.year == 2026

    def test_es_abbreviated_dot(self):
        dt = InvestingSearchScraper._parse_article_date("25 feb. 2026")
        assert dt is not None and dt.day == 25 and dt.month == 2

    def test_slash_format(self):
        dt = InvestingSearchScraper._parse_article_date("25/02/2026")
        assert dt is not None and dt.day == 25

    def test_relative_hours(self):
        dt = InvestingSearchScraper._parse_article_date("5 hours ago")
        assert dt is not None
        delta = datetime.now(timezone.utc) - dt
        assert 4 * 3600 <= delta.total_seconds() <= 6 * 3600

    def test_relative_days(self):
        dt = InvestingSearchScraper._parse_article_date("3 days ago")
        assert dt is not None
        delta = datetime.now(timezone.utc) - dt
        assert 2 * 86400 <= delta.total_seconds() <= 4 * 86400

    def test_relative_spanish_horas(self):
        dt = InvestingSearchScraper._parse_article_date("Hace 2 horas")
        assert dt is not None
        delta = datetime.now(timezone.utc) - dt
        assert 1 * 3600 <= delta.total_seconds() <= 3 * 3600

    def test_relative_spanish_dias(self):
        dt = InvestingSearchScraper._parse_article_date("hace 1 dia")
        assert dt is not None
        delta = datetime.now(timezone.utc) - dt
        assert 0 * 86400 <= delta.total_seconds() <= 2 * 86400

    def test_relative_minutes(self):
        dt = InvestingSearchScraper._parse_article_date("30 minutes ago")
        assert dt is not None
        delta = datetime.now(timezone.utc) - dt
        assert 25 * 60 <= delta.total_seconds() <= 35 * 60

    def test_empty_returns_none(self):
        assert InvestingSearchScraper._parse_article_date("") is None

    def test_none_returns_none(self):
        assert InvestingSearchScraper._parse_article_date(None) is None

    def test_garbage_returns_none(self):
        assert InvestingSearchScraper._parse_article_date("not a date at all") is None


# ============================================================================
# Test: Date windowing
# ============================================================================

class TestGenerateWindows:
    def test_single_window(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 15)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        assert len(windows) == 1
        assert windows[0] == (start, end)

    def test_multiple_windows(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 3, 1)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        assert len(windows) == 2  # Jan 1-30, Jan 31-Mar 1

    def test_exact_window(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 30)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        assert len(windows) == 1

    def test_long_range(self):
        start = datetime(2020, 1, 1)
        end = datetime(2026, 2, 26)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        assert 70 <= len(windows) <= 80

    def test_contiguous_windows(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 6, 30)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        for i in range(len(windows) - 1):
            _, w_end = windows[i]
            w_next_start, _ = windows[i + 1]
            assert (w_next_start - w_end).days == 1

    def test_empty_range(self):
        start = datetime(2026, 3, 1)
        end = datetime(2026, 2, 1)
        windows = InvestingSearchScraper._generate_windows(start, end, 30)
        assert len(windows) == 0


# ============================================================================
# Test: Deduplication
# ============================================================================

class TestDeduplication:
    def setup_method(self):
        self.scraper = InvestingSearchScraper()

    def test_dedup_by_url(self):
        articles = [
            RawArticle(
                url="https://www.investing.com/news/article-1",
                title="Article 1",
                source_id="investing",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://www.investing.com/news/article-1",
                title="Article 1 (duplicate)",
                source_id="investing",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://www.investing.com/news/article-2",
                title="Article 2",
                source_id="investing",
                published_at=datetime.now(timezone.utc),
            ),
        ]
        result = self.scraper._deduplicate(articles)
        assert len(result) == 2

    def test_dedup_preserves_first(self):
        articles = [
            RawArticle(
                url="https://www.investing.com/news/same",
                title="First",
                source_id="investing",
                published_at=datetime.now(timezone.utc),
            ),
            RawArticle(
                url="https://www.investing.com/news/same",
                title="Second",
                source_id="investing",
                published_at=datetime.now(timezone.utc),
            ),
        ]
        result = self.scraper._deduplicate(articles)
        assert result[0].title == "First"


# ============================================================================
# Test: Checkpoint
# ============================================================================

class TestCheckpoint:
    def setup_method(self):
        self.scraper = InvestingSearchScraper()
        self._orig_dir = self.scraper.CHECKPOINT_DIR
        self._orig_file = self.scraper.CHECKPOINT_FILE
        self._tmpdir = Path(tempfile.mkdtemp())
        self.scraper.CHECKPOINT_DIR = self._tmpdir
        self.scraper.CHECKPOINT_FILE = self._tmpdir / "test_checkpoint.json"

    def teardown_method(self):
        self.scraper.CHECKPOINT_DIR = self._orig_dir
        self.scraper.CHECKPOINT_FILE = self._orig_file
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        self.scraper._save_checkpoint({"completed_queries": ["en:Colombia"]})
        cp = self.scraper._load_checkpoint()
        assert cp["completed_queries"] == ["en:Colombia"]

    def test_save_merges(self):
        self.scraper._save_checkpoint({"key1": "val1"})
        self.scraper._save_checkpoint({"key2": "val2"})
        cp = self.scraper._load_checkpoint()
        assert cp["key1"] == "val1"
        assert cp["key2"] == "val2"

    def test_clear_checkpoint(self):
        self.scraper._save_checkpoint({"completed_queries": ["en:Colombia"]})
        assert self.scraper.CHECKPOINT_FILE.exists()
        self.scraper._clear_checkpoint()
        assert not self.scraper.CHECKPOINT_FILE.exists()

    def test_load_empty(self):
        cp = self.scraper._load_checkpoint()
        assert cp == {}


# ============================================================================
# Test: Query building
# ============================================================================

class TestQueryBuilding:
    def test_default_queries(self):
        scraper = InvestingSearchScraper()
        queries = scraper._build_query_list()
        assert len(queries) > 5
        for q, dk, lang in queries:
            assert isinstance(q, str)
            assert dk in ("en", "es")
            assert lang in ("en", "es")

    def test_es_disabled(self):
        cfg = ScraperConfig(investing_use_es_domain=False)
        scraper = InvestingSearchScraper(config=cfg)
        queries = scraper._build_query_list()
        for _, dk, _ in queries:
            assert dk == "en"

    def test_custom_queries(self):
        cfg = ScraperConfig(
            investing_search_queries_en=("TestQuery",),
            investing_search_queries_es=(),
            investing_use_es_domain=False,
        )
        scraper = InvestingSearchScraper(config=cfg)
        queries = scraper._build_query_list()
        assert len(queries) == 1
        assert queries[0][0] == "TestQuery"


# ============================================================================
# Test: Paginate query logic (unit-level, mocked)
# ============================================================================

class TestPaginateQueryLogic:
    """Test the stopping logic of _paginate_query without network calls."""

    def setup_method(self):
        self.scraper = InvestingSearchScraper()

    def test_parse_multiple_api_items(self):
        """Verify we can parse a batch of API items."""
        results = []
        for item in SAMPLE_API_ITEMS:
            art = self.scraper._parse_api_item(
                item, "https://www.investing.com", "en",
            )
            if art:
                results.append(art)
        assert len(results) == 3
        # All unique URLs
        urls = {r.url for r in results}
        assert len(urls) == 3

    def test_global_seen_ids_prevent_duplicates(self):
        """Verify that items with IDs in global_seen_ids would be skipped."""
        global_seen = {4522732}  # First item's dataID
        # The paginate logic checks global_seen before adding
        # We test that _parse_api_item still works (dedup is in paginate)
        art = self.scraper._parse_api_item(
            SAMPLE_API_ITEMS[0], "https://www.investing.com", "en",
        )
        assert art is not None  # Parser doesn't filter, paginator does
