"""
La Republica Scraper (SDD-02 §2.6)
=====================================
Colombian financial newspaper. RSS + HTML article scraping.
Focuses on: Finanzas, Economia, Globoeconomia sections.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

from src.news_engine.config import ScraperConfig
from src.news_engine.ingestion.base_adapter import SourceAdapter, parse_date_flexible
from src.news_engine.models import RawArticle

logger = logging.getLogger(__name__)


class LaRepublicaScraper(SourceAdapter):
    """La Republica RSS + HTML scraper."""

    source_id = "larepublica"
    source_name = "La Republica"

    def __init__(self, config: ScraperConfig | None = None):
        super().__init__(config)
        self.cfg = config or ScraperConfig()

    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        articles = []

        # Try RSS first
        try:
            rss_articles = self._parse_rss(cutoff)
            articles.extend(rss_articles)
        except Exception as e:
            logger.warning(f"[larepublica] RSS failed: {e}")

        # Try sitemap fallback
        if not articles:
            try:
                sitemap_articles = self._parse_sitemap(cutoff)
                articles.extend(sitemap_articles)
            except Exception as e:
                logger.warning(f"[larepublica] Sitemap failed: {e}")

        return self._deduplicate(articles)

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        return self.fetch_latest(hours_back=72)

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                self.cfg.larepublica_rss_url,
                headers={"User-Agent": self.cfg.user_agent},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _parse_rss(self, cutoff: datetime) -> list[RawArticle]:
        """Parse La Republica RSS feed."""
        try:
            import feedparser
        except ImportError:
            logger.error("[larepublica] feedparser not installed")
            return []

        resp = requests.get(
            self.cfg.larepublica_rss_url,
            headers={"User-Agent": self.cfg.user_agent},
            timeout=self.cfg.request_timeout,
        )
        feed = feedparser.parse(resp.content)

        articles = []
        for entry in feed.entries:
            published = parse_date_flexible(entry.get("published", ""))
            if published is None:
                continue

            if published.replace(tzinfo=None) < cutoff:
                continue

            articles.append(RawArticle(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                source_id=self.source_id,
                published_at=published,
                summary=entry.get("summary", ""),
                language="es",
                country_focus="CO",
            ))

        logger.info(f"[larepublica] RSS: {len(articles)} articles")
        return articles

    def _parse_sitemap(self, cutoff: datetime) -> list[RawArticle]:
        """Parse sitemap-news.xml as fallback."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("[larepublica] beautifulsoup4 not installed")
            return []

        resp = requests.get(
            self.cfg.larepublica_sitemap,
            headers={"User-Agent": self.cfg.user_agent},
            timeout=self.cfg.request_timeout,
        )
        soup = BeautifulSoup(resp.content, "xml")

        articles = []
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            lastmod = url_tag.find("lastmod")
            news_title = url_tag.find("news:title")

            if not loc or not news_title:
                continue

            try:
                if lastmod:
                    published = datetime.fromisoformat(lastmod.text.replace("Z", "+00:00"))
                else:
                    continue
            except ValueError:
                continue

            if published.replace(tzinfo=None) < cutoff:
                continue

            articles.append(RawArticle(
                url=loc.text,
                title=news_title.text,
                source_id=self.source_id,
                published_at=published,
                language="es",
                country_focus="CO",
            ))

        logger.info(f"[larepublica] Sitemap: {len(articles)} articles")
        return articles

    def scrape_article_content(self, url: str) -> str | None:
        """Scrape full article content from URL."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.cfg.user_agent},
                timeout=self.cfg.request_timeout,
            )
            soup = BeautifulSoup(resp.content, "html.parser")

            # La Republica article body selector
            body = soup.find("div", class_="article-body") or soup.find("article")
            if body:
                paragraphs = body.find_all("p")
                return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return None
        except Exception as e:
            logger.warning(f"[larepublica] Scrape failed for {url}: {e}")
            return None
