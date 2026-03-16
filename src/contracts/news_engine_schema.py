"""
NewsEngine Contract Schema (SDD-02/03/06)
==========================================
Pydantic-style validation models for NewsEngine data exchange.
Mirrors the database schema from migration 045.

Contract: CTR-NEWS-SCHEMA-001
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

VALID_SOURCE_IDS = (
    "gdelt_doc", "gdelt_context", "newsapi",
    "investing", "larepublica", "portafolio",
)

VALID_CATEGORIES = (
    "monetary_policy", "fx_market", "commodities", "inflation",
    "fiscal_policy", "risk_premium", "capital_flows", "balance_payments",
    "political",
)

VALID_SENTIMENTS = ("positive", "negative", "neutral")

VALID_DIGEST_TYPES = ("daily", "weekly")


# ---------------------------------------------------------------------------
# Article schema
# ---------------------------------------------------------------------------

@dataclass
class ArticleRecord:
    """Validated article for DB insertion."""
    source_id: str
    url: str
    url_hash: str
    title: str
    published_at: datetime
    content: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    relevance_score: float = 0.0
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    gdelt_tone: Optional[float] = None
    keywords: Optional[list] = None
    entities: Optional[list] = None
    language: str = "es"
    country_focus: str = "CO"
    is_breaking: bool = False
    is_weekly_relevant: bool = False
    image_url: Optional[str] = None
    author: Optional[str] = None
    raw_json: Optional[dict] = None

    def validate(self) -> list:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.source_id not in VALID_SOURCE_IDS:
            errors.append(f"Invalid source_id: {self.source_id}")
        if not self.url:
            errors.append("URL is required")
        if not self.title:
            errors.append("Title is required")
        if self.category and self.category not in VALID_CATEGORIES:
            errors.append(f"Invalid category: {self.category}")
        if self.sentiment_label and self.sentiment_label not in VALID_SENTIMENTS:
            errors.append(f"Invalid sentiment_label: {self.sentiment_label}")
        if self.sentiment_score is not None and not (-1.0 <= self.sentiment_score <= 1.0):
            errors.append(f"sentiment_score must be [-1, 1], got {self.sentiment_score}")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Digest schema
# ---------------------------------------------------------------------------

@dataclass
class DigestRecord:
    """Validated daily/weekly digest for DB insertion."""
    digest_date: date
    digest_type: str = "daily"
    total_articles: int = 0
    by_source: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    avg_sentiment: Optional[float] = None
    top_keywords: list = field(default_factory=list)
    top_articles: list = field(default_factory=list)
    cross_ref_count: int = 0
    summary_text: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature snapshot schema
# ---------------------------------------------------------------------------

@dataclass
class FeatureSnapshotRecord:
    """Validated feature snapshot for DB insertion."""
    snapshot_date: date
    features: dict = field(default_factory=dict)
    feature_version: str = "v1.0"
    article_count: int = 0
    source_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cross-reference schema
# ---------------------------------------------------------------------------

@dataclass
class CrossReferenceRecord:
    """Validated cross-reference cluster for DB insertion."""
    topic: str
    cluster_date: date
    article_count: int = 0
    avg_sentiment: Optional[float] = None
    dominant_category: Optional[str] = None
    sources_involved: list = field(default_factory=list)
    summary: Optional[str] = None
    article_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("article_ids", None)
        return d


# ---------------------------------------------------------------------------
# Ingestion log schema
# ---------------------------------------------------------------------------

@dataclass
class IngestionLogRecord:
    """Ingestion run log entry."""
    source_id: str
    run_type: str = "scheduled"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    articles_fetched: int = 0
    articles_new: int = 0
    articles_updated: int = 0
    errors: int = 0
    error_details: Optional[str] = None
    status: str = "running"

    def to_dict(self) -> dict:
        return asdict(self)
