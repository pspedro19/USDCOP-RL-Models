"""
SourceRegistry (SDD-02 §4)
============================
Central registry of all news source adapters.
Follows the ExtractorRegistry pattern from L0 macro DAGs.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.news_engine.config import NewsEngineConfig
from src.news_engine.ingestion.base_adapter import SourceAdapter
from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter, GDELTContextAdapter
from src.news_engine.ingestion.newsapi_adapter import NewsAPIAdapter
from src.news_engine.ingestion.investing_scraper import InvestingSearchScraper
from src.news_engine.ingestion.larepublica_scraper import LaRepublicaScraper
from src.news_engine.ingestion.portafolio_scraper import PortafolioScraper

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Central registry of all news source adapters.

    Usage:
        registry = SourceRegistry.from_config(config)
        for adapter in registry.enabled_adapters():
            articles = adapter.fetch_latest()
    """

    def __init__(self):
        self._adapters: dict[str, SourceAdapter] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, adapter: SourceAdapter, enabled: bool = True) -> None:
        """Register a source adapter."""
        self._adapters[adapter.source_id] = adapter
        self._enabled[adapter.source_id] = enabled

    def get(self, source_id: str) -> Optional[SourceAdapter]:
        """Get adapter by source ID."""
        return self._adapters.get(source_id)

    def enabled_adapters(self) -> list[SourceAdapter]:
        """Return all enabled adapters in registration order."""
        return [
            adapter for sid, adapter in self._adapters.items()
            if self._enabled.get(sid, True)
        ]

    def all_adapters(self) -> list[SourceAdapter]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all enabled adapters."""
        results = {}
        for adapter in self.enabled_adapters():
            try:
                results[adapter.source_id] = adapter.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {adapter.source_id}: {e}")
                results[adapter.source_id] = False
        return results

    @classmethod
    def from_config(cls, config: Optional[NewsEngineConfig] = None) -> "SourceRegistry":
        """Create registry with all adapters from config."""
        config = config or NewsEngineConfig()
        registry = cls()

        # News API adapters
        registry.register(GDELTDocAdapter(config.gdelt))
        registry.register(GDELTContextAdapter(config.gdelt))
        registry.register(
            NewsAPIAdapter(config.newsapi),
            enabled=bool(config.newsapi.api_key),
        )

        # Web scrapers
        registry.register(InvestingSearchScraper(config.scraper))
        registry.register(LaRepublicaScraper(config.scraper))
        registry.register(PortafolioScraper(config.scraper))

        logger.info(
            f"SourceRegistry initialized: "
            f"{len(registry.enabled_adapters())}/{len(registry.all_adapters())} enabled"
        )
        return registry

    def __len__(self) -> int:
        return len(self._adapters)

    def __repr__(self) -> str:
        enabled = len(self.enabled_adapters())
        total = len(self._adapters)
        return f"<SourceRegistry {enabled}/{total} enabled>"
