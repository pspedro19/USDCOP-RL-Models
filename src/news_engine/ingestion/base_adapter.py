"""
SourceAdapter ABC (SDD-02 §2)
==============================
Every news source implements this interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from src.news_engine.models import RawArticle

logger = logging.getLogger(__name__)


def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Parse dates from RSS feeds with multiple format support.

    Handles:
    - RFC 2822: "Tue, 25 Feb 2026 19:21:28 -0500"
    - ISO 8601: "2026-02-25T19:21:28-05:00"
    - Simple:   "2026-02-25 19:21:28"
    """
    if not date_str:
        return None

    # Try RFC 2822 first (standard RSS format)
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass

    # Try ISO 8601 (common in modern feeds)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        pass

    # Try common simple formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


class SourceAdapter(ABC):
    """Base class for all news source adapters.

    Every adapter must implement:
    - fetch_latest(): Get recent articles (last N hours)
    - fetch_historical(): Get articles for a date range (backfill)
    - health_check(): Verify source is reachable

    Adapters normalize all outputs to RawArticle dataclass before returning.
    """

    source_id: str = ""
    source_name: str = ""

    def __init__(self, config=None):
        self.config = config
        self._last_fetch_time: Optional[datetime] = None
        self._error_count: int = 0
        self._max_retries: int = 3
        self._retry_delay: float = 2.0

    @abstractmethod
    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        """Fetch recent articles from the last N hours.

        Args:
            hours_back: How far back to look (default: 24 hours)

        Returns:
            List of RawArticle instances, deduplicated by URL.
        """
        ...

    @abstractmethod
    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        """Fetch articles for a historical date range (backfill).

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of RawArticle instances.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the source is reachable and responding.

        Returns:
            True if healthy, False otherwise.
        """
        ...

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        for attempt in range(self._max_retries):
            try:
                result = func(*args, **kwargs)
                self._error_count = 0
                return result
            except Exception as e:
                self._error_count += 1
                wait = self._retry_delay * (2 ** attempt)
                logger.warning(
                    f"[{self.source_id}] Attempt {attempt + 1}/{self._max_retries} "
                    f"failed: {e}. Retrying in {wait:.1f}s..."
                )
                if attempt < self._max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"[{self.source_id}] All retries exhausted: {e}")
                    raise

    def _deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """Remove duplicate articles by URL hash."""
        seen = set()
        unique = []
        for article in articles:
            h = article.url_hash
            if h not in seen:
                seen.add(h)
                unique.append(article)
        return unique

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} source_id={self.source_id!r}>"
