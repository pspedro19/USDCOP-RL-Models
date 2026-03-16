"""
NewsEngine Configuration (SDD-10)
==================================
Pydantic settings for all NewsEngine components.
Reads from environment variables with USDCOP_ prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "usdcop"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class GDELTConfig:
    """GDELT DOC + Context API settings."""
    doc_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    context_base_url: str = "https://api.gdeltproject.org/api/v2/context/context"
    max_records: int = 250
    rate_limit_seconds: float = 8.0
    request_timeout: int = 30
    # Window size for historical extraction (days per chunk)
    historical_window_days: int = 7
    # Queries: OR terms MUST be wrapped in parentheses per GDELT syntax
    # Use sourcelang: inline filter to separate EN/ES
    queries_en: tuple = (
        'Colombia sourcelang:english',
        '("Colombian peso" OR USDCOP OR "USD COP") sourcelang:english',
        '(Ecopetrol OR "Colombia oil" OR "petroleo Colombia") sourcelang:english',
        '("Banco Republica" OR "Colombia central bank") sourcelang:english',
        '("Colombia economy" OR "Colombia inflation" OR "Colombia GDP") sourcelang:english',
        '("Colombia bonds" OR "EMBI Colombia" OR "Colombia risk") sourcelang:english',
    )
    queries_es: tuple = (
        'Colombia sourcelang:spanish',
        '("peso colombiano" OR "dolar Colombia" OR "tasa de cambio") sourcelang:spanish',
        '(Ecopetrol OR "petroleo Colombia") sourcelang:spanish',
        '("Banco Republica" OR "tasa de interes Colombia") sourcelang:spanish',
        '("inflacion Colombia" OR "reforma tributaria Colombia") sourcelang:spanish',
    )
    use_es_queries: bool = True


@dataclass(frozen=True)
class NewsAPIConfig:
    """NewsAPI.org settings."""
    base_url: str = "https://newsapi.org/v2"
    api_key: str = ""
    max_daily_requests: int = 100
    page_size: int = 100
    queries: tuple = (
        "Colombia peso exchange rate",
        "USD COP forex",
        "Banco Republica interest rate",
    )

    def __post_init__(self):
        if not self.api_key:
            key = os.environ.get("USDCOP_NEWSAPI_KEY", "")
            object.__setattr__(self, "api_key", key)


@dataclass(frozen=True)
class ScraperConfig:
    """Settings for web scrapers (Investing, La Republica, Portafolio)."""
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 2.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # Investing.com search scraper (replaces RSS)
    investing_search_queries_en: tuple = (
        "Colombia",
        "Colombian peso",
        "USDCOP",
        "Banco Republica Colombia",
        "Ecopetrol",
    )
    investing_search_queries_es: tuple = (
        "Colombia economia",
        "peso colombiano dolar",
        "Banco Republica tasa",
    )
    investing_search_delay_min: float = 3.0
    investing_search_delay_max: float = 6.0
    investing_search_max_pages: int = 10
    investing_search_window_days: int = 30
    investing_use_es_domain: bool = True
    investing_session_rotation_every: int = 50
    # Legacy RSS (kept for reference, no longer used by default)
    investing_rss_urls: tuple = (
        "https://www.investing.com/rss/news_14.rss",   # Forex
        "https://www.investing.com/rss/news_25.rss",   # Commodities
        "https://www.investing.com/rss/news_1.rss",    # Economy
    )
    larepublica_rss_url: str = "https://www.larepublica.co/rss/finanzas.xml"
    larepublica_sitemap: str = "https://www.larepublica.co/sitemap-news.xml"
    portafolio_rss_url: str = "https://www.portafolio.co/rss/economia.xml"
    portafolio_sitemap: str = "https://www.portafolio.co/sitemap-news.xml"


@dataclass(frozen=True)
class EnrichmentConfig:
    """Enrichment pipeline settings."""
    min_relevance_score: float = 0.3
    sentiment_default: str = "neutral"
    categories: tuple = (
        "monetary_policy",
        "fx_market",
        "commodities",
        "inflation",
        "fiscal_policy",
        "risk_premium",
        "capital_flows",
        "balance_payments",
        "political",
    )
    high_priority_keywords: tuple = (
        "dolar", "tasa de cambio", "banco de la republica",
        "tasa de interes", "fed rate", "petroleo", "inflacion",
        "EMBI", "devaluacion",
    )


@dataclass(frozen=True)
class CrossReferenceConfig:
    """Cross-reference engine settings."""
    similarity_threshold: float = 0.6
    min_cluster_size: int = 2
    max_cluster_size: int = 20
    time_window_hours: int = 48
    # Similarity weights
    title_weight: float = 0.40
    entity_weight: float = 0.30
    summary_weight: float = 0.20
    category_weight: float = 0.10


@dataclass(frozen=True)
class AlertConfig:
    """Breaking news alert settings."""
    enabled: bool = True
    check_interval_minutes: int = 30
    gdelt_tone_threshold: float = -15.0
    volume_spike_threshold: float = 3.0   # 3x std above mean
    slack_webhook: str = ""

    def __post_init__(self):
        if not self.slack_webhook:
            webhook = os.environ.get("USDCOP_ALERT_SLACK_WEBHOOK", "")
            object.__setattr__(self, "slack_webhook", webhook)


@dataclass(frozen=True)
class FeatureExportConfig:
    """Feature export settings."""
    version: str = "v1.0"
    total_features: int = 60
    output_format: str = "json"   # json, parquet, csv


@dataclass
class NewsEngineConfig:
    """Top-level NewsEngine configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    gdelt: GDELTConfig = field(default_factory=GDELTConfig)
    newsapi: NewsAPIConfig = field(default_factory=NewsAPIConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    cross_reference: CrossReferenceConfig = field(default_factory=CrossReferenceConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    feature_export: FeatureExportConfig = field(default_factory=FeatureExportConfig)
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "NewsEngineConfig":
        """Create config from environment variables."""
        return cls(
            database=DatabaseConfig(
                host=os.environ.get("USDCOP_DB_HOST", "localhost"),
                port=int(os.environ.get("USDCOP_DB_PORT", "5432")),
                database=os.environ.get("USDCOP_DB_NAME", "usdcop"),
                user=os.environ.get("USDCOP_DB_USER", "postgres"),
                password=os.environ.get("USDCOP_DB_PASSWORD", "postgres"),
            ),
            enabled=os.environ.get("USDCOP_NEWS_ENGINE_ENABLED", "true").lower() == "true",
        )
