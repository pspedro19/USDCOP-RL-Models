"""
NewsEngine Data Models (SDD-02)
================================
Dataclasses for raw articles, macro data points, and enrichment results.
These are the in-memory representations used between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawArticle:
    """Normalized article from any source (SDD-02 §3).

    Every adapter transforms its source-specific format into RawArticle
    before passing to the enrichment pipeline.
    """
    url: str
    title: str
    source_id: str
    published_at: datetime

    content: Optional[str] = None
    summary: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None
    language: str = "es"
    country_focus: str = "CO"

    # GDELT-specific
    gdelt_tone: Optional[float] = None          # -100 to 100 (practical: -20 to 20)
    gdelt_volume: Optional[int] = None

    # Raw response for debugging
    raw_json: Optional[dict] = None

    @property
    def url_hash(self) -> str:
        """SHA256 hash of URL for deduplication."""
        import hashlib
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()


@dataclass
class EnrichedArticle:
    """Article after enrichment pipeline (SDD-04)."""
    raw: RawArticle

    category: Optional[str] = None
    subcategory: Optional[str] = None
    relevance_score: float = 0.0
    sentiment_score: Optional[float] = None     # -1.0 to 1.0
    sentiment_label: Optional[str] = None       # positive, negative, neutral
    keywords: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    is_breaking: bool = False
    is_weekly_relevant: bool = False


@dataclass
class GDELTTimelinePoint:
    """Single data point from GDELT timeline API."""
    date: datetime
    value: float                                 # Tone or volume
    series: str = "tone"                         # "tone" or "volume"


@dataclass
class CrossReference:
    """A cluster of related articles across sources (SDD-05)."""
    topic: str
    cluster_date: datetime
    articles: list = field(default_factory=list)  # List of article IDs
    avg_sentiment: Optional[float] = None
    dominant_category: Optional[str] = None
    sources_involved: list = field(default_factory=list)
    summary: Optional[str] = None


@dataclass
class NewsDigest:
    """Daily or weekly digest summary (SDD-06)."""
    digest_date: datetime
    digest_type: str = "daily"                    # "daily" or "weekly"
    total_articles: int = 0
    by_source: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    avg_sentiment: Optional[float] = None
    top_keywords: list = field(default_factory=list)
    top_articles: list = field(default_factory=list)
    cross_ref_count: int = 0
    summary_text: Optional[str] = None


@dataclass
class NewsFeatureVector:
    """Daily feature vector extracted from news data (SDD-06 §5).

    ~60 features across 7 groups (no macro — macro comes from existing L0 tables).
    """
    snapshot_date: datetime
    features: dict = field(default_factory=dict)  # {feature_name: value}
    article_count: int = 0
    source_counts: dict = field(default_factory=dict)
    version: str = "v1.0"

    @property
    def feature_names(self) -> list:
        return sorted(self.features.keys())

    @property
    def feature_count(self) -> int:
        return len(self.features)
