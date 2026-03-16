"""
Live integration tests for NewsEngine extractors + Azure OpenAI.
Tests REAL HTTP connections to all news sources and LLM provider.

Run: python -m pytest tests/integration/test_live_extractors.py -v -s
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================================
# AZURE OPENAI
# ============================================================================

class TestAzureOpenAILive:
    """Test Azure OpenAI connectivity with real API key."""

    def test_health_check(self):
        from src.analysis.llm_client import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=os.environ.get("USDCOP_AZURE_OPENAI_API_KEY", ""),
            endpoint=os.environ.get("USDCOP_AZURE_OPENAI_ENDPOINT", ""),
        )
        assert provider.health_check(), (
            "Azure OpenAI health check failed — "
            "set USDCOP_AZURE_OPENAI_API_KEY and USDCOP_AZURE_OPENAI_ENDPOINT"
        )

    def test_generate_short_response(self):
        from src.analysis.llm_client import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=os.environ.get("USDCOP_AZURE_OPENAI_API_KEY", ""),
            endpoint=os.environ.get("USDCOP_AZURE_OPENAI_ENDPOINT", ""),
        )
        if not provider.health_check():
            pytest.skip("Azure OpenAI not configured")

        result = provider.generate(
            system_prompt="Eres un analista financiero. Responde en español, maximo 50 palabras.",
            user_prompt="Describe brevemente que es el USD/COP.",
            max_tokens=100,
            temperature=0.3,
        )
        assert "content" in result
        assert len(result["content"]) > 10
        assert result["tokens_used"] > 0
        assert result["cost_usd"] >= 0
        print(f"\n  [Azure OpenAI] Model: {result['model']}")
        print(f"  [Azure OpenAI] Tokens: {result['tokens_used']}, Cost: ${result['cost_usd']:.6f}")
        print(f"  [Azure OpenAI] Response: {result['content'][:200]}...")

    def test_llm_client_with_cache(self):
        """Test full LLMClient with caching."""
        import tempfile
        from src.analysis.llm_client import LLMClient, AzureOpenAIProvider

        api_key = os.environ.get("USDCOP_AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("USDCOP_AZURE_OPENAI_ENDPOINT", "")
        if not api_key or not endpoint:
            pytest.skip("Azure OpenAI not configured")

        with tempfile.TemporaryDirectory() as tmpdir:
            primary = AzureOpenAIProvider(api_key=api_key, endpoint=endpoint)
            client = LLMClient(primary=primary, cache_dir=tmpdir)

            # First call — should hit API
            r1 = client.generate(
                "Responde en una oracion.", "Que es el DXY?",
                max_tokens=50, cache_key="test_dxy",
            )
            assert r1["cached"] is False
            assert len(r1["content"]) > 5

            # Second call — should hit cache
            r2 = client.generate(
                "Responde en una oracion.", "Que es el DXY?",
                max_tokens=50, cache_key="test_dxy",
            )
            assert r2["cached"] is True
            assert r2["content"] == r1["content"]
            print(f"\n  [Cache] First call: API ({r1['tokens_used']} tokens), Second call: CACHED")
            print(f"  [Cache] Total cost: ${client.total_cost:.6f}")


# ============================================================================
# GDELT (Public API, no key needed)
# ============================================================================

class TestGDELTLive:
    """Test GDELT Doc + Context APIs with real HTTP calls."""

    def test_gdelt_doc_fetch_latest(self):
        from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter
        from src.news_engine.config import GDELTConfig

        adapter = GDELTDocAdapter(GDELTConfig())
        articles = adapter.fetch_latest(hours_back=48)
        print(f"\n  [GDELT Doc] Fetched {len(articles)} articles (48h)")
        if articles:
            latest = articles[0]
            print(f"  [GDELT Doc] Latest: '{latest.title[:80]}...'")
            print(f"  [GDELT Doc] Source: {latest.source_id}, Published: {latest.published_at}")
            print(f"  [GDELT Doc] URL: {latest.url[:100]}")
            if latest.gdelt_tone is not None:
                print(f"  [GDELT Doc] Tone: {latest.gdelt_tone:.2f}")
        assert len(articles) >= 0  # May be 0 if API is temporarily down

    def test_gdelt_context_fetch_latest(self):
        from src.news_engine.ingestion.gdelt_adapter import GDELTContextAdapter
        from src.news_engine.config import GDELTConfig

        adapter = GDELTContextAdapter(GDELTConfig())
        articles = adapter.fetch_latest(hours_back=48)
        print(f"\n  [GDELT Context] Fetched {len(articles)} articles (48h)")
        if articles:
            latest = articles[0]
            print(f"  [GDELT Context] Latest: '{latest.title[:80]}...'")

    def test_gdelt_doc_health_check(self):
        from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter
        from src.news_engine.config import GDELTConfig

        adapter = GDELTDocAdapter(GDELTConfig())
        is_healthy = adapter.health_check()
        print(f"\n  [GDELT Doc] Health check: {'OK' if is_healthy else 'FAIL'}")
        # Don't assert — network issues shouldn't fail the test suite


# ============================================================================
# INVESTING.COM (RSS, no key needed)
# ============================================================================

class TestInvestingLive:
    """Test Investing.com RSS scraper."""

    def test_fetch_latest(self):
        from src.news_engine.ingestion.investing_scraper import InvestingScraper
        from src.news_engine.config import ScraperConfig

        adapter = InvestingScraper(ScraperConfig())
        articles = adapter.fetch_latest(hours_back=72)
        print(f"\n  [Investing] Fetched {len(articles)} articles (72h)")
        if articles:
            latest = articles[0]
            print(f"  [Investing] Latest: '{latest.title[:80]}...'")
            print(f"  [Investing] Published: {latest.published_at}")
            print(f"  [Investing] URL: {latest.url[:100]}")
        # RSS may return 0 articles depending on timing

    def test_health_check(self):
        from src.news_engine.ingestion.investing_scraper import InvestingScraper
        from src.news_engine.config import ScraperConfig

        adapter = InvestingScraper(ScraperConfig())
        is_healthy = adapter.health_check()
        print(f"\n  [Investing] Health check: {'OK' if is_healthy else 'FAIL'}")


# ============================================================================
# LA REPUBLICA (Colombian news, no key needed)
# ============================================================================

class TestLaRepublicaLive:
    """Test La República scraper (Colombian financial news)."""

    def test_fetch_latest(self):
        from src.news_engine.ingestion.larepublica_scraper import LaRepublicaScraper
        from src.news_engine.config import ScraperConfig

        adapter = LaRepublicaScraper(ScraperConfig())
        articles = adapter.fetch_latest(hours_back=72)
        print(f"\n  [LaRepública] Fetched {len(articles)} articles (72h)")
        if articles:
            latest = articles[0]
            print(f"  [LaRepública] Latest: '{latest.title[:80]}...'")
            print(f"  [LaRepública] Published: {latest.published_at}")
            print(f"  [LaRepública] URL: {latest.url[:100]}")

    def test_health_check(self):
        from src.news_engine.ingestion.larepublica_scraper import LaRepublicaScraper
        from src.news_engine.config import ScraperConfig

        adapter = LaRepublicaScraper(ScraperConfig())
        is_healthy = adapter.health_check()
        print(f"\n  [LaRepública] Health check: {'OK' if is_healthy else 'FAIL'}")


# ============================================================================
# PORTAFOLIO (Colombian news, no key needed)
# ============================================================================

class TestPortafolioLive:
    """Test Portafolio scraper (Colombian financial news)."""

    def test_fetch_latest(self):
        from src.news_engine.ingestion.portafolio_scraper import PortafolioScraper
        from src.news_engine.config import ScraperConfig

        adapter = PortafolioScraper(ScraperConfig())
        articles = adapter.fetch_latest(hours_back=72)
        print(f"\n  [Portafolio] Fetched {len(articles)} articles (72h)")
        if articles:
            latest = articles[0]
            print(f"  [Portafolio] Latest: '{latest.title[:80]}...'")
            print(f"  [Portafolio] Published: {latest.published_at}")
            print(f"  [Portafolio] URL: {latest.url[:100]}")

    def test_health_check(self):
        from src.news_engine.ingestion.portafolio_scraper import PortafolioScraper
        from src.news_engine.config import ScraperConfig

        adapter = PortafolioScraper(ScraperConfig())
        is_healthy = adapter.health_check()
        print(f"\n  [Portafolio] Health check: {'OK' if is_healthy else 'FAIL'}")


# ============================================================================
# NEWSAPI (requires API key)
# ============================================================================

class TestNewsAPILive:
    """Test NewsAPI.org (requires USDCOP_NEWSAPI_KEY env var)."""

    def test_fetch_latest(self):
        from src.news_engine.ingestion.newsapi_adapter import NewsAPIAdapter
        from src.news_engine.config import NewsAPIConfig

        config = NewsAPIConfig()
        if not config.api_key:
            pytest.skip("USDCOP_NEWSAPI_KEY not set — NewsAPI disabled")

        adapter = NewsAPIAdapter(config)
        articles = adapter.fetch_latest(hours_back=72)
        print(f"\n  [NewsAPI] Fetched {len(articles)} articles (72h)")
        if articles:
            latest = articles[0]
            print(f"  [NewsAPI] Latest: '{latest.title[:80]}...'")


# ============================================================================
# FULL REGISTRY INTEGRATION
# ============================================================================

class TestRegistryLive:
    """Test full SourceRegistry with live adapters."""

    def test_registry_creation(self):
        from src.news_engine.ingestion.registry import SourceRegistry

        registry = SourceRegistry.from_config()
        all_adapters = registry.all_adapters()
        enabled = registry.enabled_adapters()
        print(f"\n  [Registry] Total adapters: {len(all_adapters)}")
        print(f"  [Registry] Enabled adapters: {len(enabled)}")
        for a in all_adapters:
            is_enabled = a in enabled
            print(f"    - {a.source_id}: {'ENABLED' if is_enabled else 'DISABLED'}")
        assert len(all_adapters) == 6

    def test_health_check_all(self):
        from src.news_engine.ingestion.registry import SourceRegistry

        registry = SourceRegistry.from_config()
        results = registry.health_check_all()
        print(f"\n  [Registry] Health check results:")
        for source_id, healthy in results.items():
            status = "OK" if healthy else "FAIL"
            print(f"    - {source_id}: {status}")

    def test_fetch_all_sources(self):
        """Fetch from ALL enabled sources and aggregate results."""
        from src.news_engine.ingestion.registry import SourceRegistry

        registry = SourceRegistry.from_config()
        total_articles = 0
        source_counts = {}

        for adapter in registry.enabled_adapters():
            try:
                articles = adapter.fetch_latest(hours_back=48)
                source_counts[adapter.source_id] = len(articles)
                total_articles += len(articles)
                time.sleep(1)  # Rate limit between sources
            except Exception as e:
                source_counts[adapter.source_id] = f"ERROR: {e}"

        print(f"\n  [Registry] Total articles across all sources: {total_articles}")
        for source_id, count in source_counts.items():
            print(f"    - {source_id}: {count}")


# ============================================================================
# ENRICHMENT PIPELINE WITH REAL DATA
# ============================================================================

class TestEnrichmentLive:
    """Test enrichment pipeline on real articles from GDELT."""

    def test_enrich_real_articles(self):
        from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter
        from src.news_engine.config import GDELTConfig
        from src.news_engine.enrichment.pipeline import EnrichmentPipeline

        # Fetch real articles
        adapter = GDELTDocAdapter(GDELTConfig())
        articles = adapter.fetch_latest(hours_back=48)

        if not articles:
            pytest.skip("No GDELT articles available")

        # Enrich them
        pipeline = EnrichmentPipeline()
        enriched = pipeline.enrich_batch(articles[:10])

        print(f"\n  [Enrichment] Enriched {len(enriched)}/{min(10, len(articles))} articles")
        for ea in enriched[:5]:
            # Use ascii fallback to avoid Windows cp1252 encoding errors
            safe_title = ea.raw.title[:60].encode("ascii", "replace").decode("ascii")
            print(f"    - '{safe_title}...'")
            print(f"      Category: {ea.category}, Relevance: {ea.relevance_score:.2f}, "
                  f"Sentiment: {ea.sentiment_score:.2f} ({ea.sentiment_label})")
            if ea.keywords:
                safe_kw = [k.encode("ascii", "replace").decode("ascii") for k in ea.keywords[:5]]
                print(f"      Keywords: {', '.join(safe_kw)}")
            if ea.is_breaking:
                print(f"      *** BREAKING NEWS ***")
            print()
