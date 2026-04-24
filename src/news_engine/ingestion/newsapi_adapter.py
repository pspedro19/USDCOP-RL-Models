"""
NewsAPI.org Adapter (SDD-02 §2.3)
===================================
Fetches articles from NewsAPI.org (100 daily requests, API key required).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

from src.news_engine.config import NewsAPIConfig
from src.news_engine.ingestion.base_adapter import SourceAdapter
from src.news_engine.models import RawArticle

logger = logging.getLogger(__name__)


class NewsAPIAdapter(SourceAdapter):
    """NewsAPI.org adapter."""

    source_id = "newsapi"
    source_name = "NewsAPI.org"

    def __init__(self, config: NewsAPIConfig | None = None):
        super().__init__(config)
        self.cfg = config or NewsAPIConfig()
        if not self.cfg.api_key:
            logger.warning("[newsapi] No API key configured — adapter disabled")

    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        if not self.cfg.api_key:
            return []

        from_date = datetime.utcnow() - timedelta(hours=hours_back)
        all_articles = []
        for query in self.cfg.queries:
            try:
                articles = self._fetch_query(query, from_date)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"[newsapi] Query '{query}' failed: {e}")
        return self._deduplicate(all_articles)

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        if not self.cfg.api_key:
            return []

        all_articles = []
        for query in self.cfg.queries:
            try:
                articles = self._fetch_query(query, start_date, end_date)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"[newsapi] Historical query '{query}' failed: {e}")
        return self._deduplicate(all_articles)

    def health_check(self) -> bool:
        if not self.cfg.api_key:
            return False
        try:
            resp = requests.get(
                f"{self.cfg.base_url}/everything",
                params={"q": "test", "pageSize": "1", "apiKey": self.cfg.api_key},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _fetch_query(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime | None = None,
    ) -> list[RawArticle]:
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "publishedAt",
            "pageSize": str(self.cfg.page_size),
            "language": "en",
            "apiKey": self.cfg.api_key,
        }
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%S")

        resp = self._retry_with_backoff(
            requests.get,
            f"{self.cfg.base_url}/everything",
            params=params,
            timeout=30,
        )
        data = resp.json()

        if data.get("status") != "ok":
            logger.error(f"[newsapi] API error: {data.get('message', 'Unknown')}")
            return []

        articles = []
        for art in data.get("articles", []):
            try:
                published = datetime.fromisoformat(
                    art["publishedAt"].replace("Z", "+00:00")
                )
            except (ValueError, KeyError):
                continue

            articles.append(RawArticle(
                url=art.get("url", ""),
                title=art.get("title", ""),
                source_id=self.source_id,
                published_at=published,
                content=art.get("content"),
                summary=art.get("description"),
                image_url=art.get("urlToImage"),
                author=art.get("author"),
                language="en",
                raw_json=art,
            ))
        logger.info(f"[newsapi] Query '{query}': {len(articles)} articles")
        return articles
