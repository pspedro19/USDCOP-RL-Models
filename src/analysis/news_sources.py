"""
Pluggable news-source module for multi-asset analysis (ports & adapters).
========================================================================

Adding a NEW real news source = write one adapter class implementing the
`NewsSource` protocol + register it in `SOURCE_REGISTRY`. Adding a NEW asset
(index, future, FX pair…) = one entry in config/analysis/analysis_assets.yaml —
no code change here.

Design
------
  NewsQuery      — what to fetch (asset id + per-source query strings + week)
  NewsArticle    — normalised article record (source-agnostic)
  NewsSource     — port: .fetch(NewsQuery) -> list[NewsArticle]
  *Source        — adapters: GoogleNewsSource (primary), GDELTSource (fallback)
  build_news_sources(cfg) — factory: reads the `news.sources` order from the SSOT
  AssetNewsFetcher        — facade: runs sources by strategy (first_nonempty | aggregate)

All sources are public (no API key). Every adapter is best-effort: any failure
returns [] so analysis generation is never blocked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- DTOs
@dataclass
class NewsArticle:
    title: str | None
    url: str | None
    source: str | None
    published_at: str | None = None
    language: str | None = None
    tone: float | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "language": self.language,
            "tone": self.tone,
        }

    def key(self) -> str:
        return (self.url or self.title or "").strip().lower()


@dataclass
class NewsQuery:
    asset_id: str
    google_query: str
    gdelt_query: str
    w_start: date
    w_end: date


# --------------------------------------------------------------------------- port
@runtime_checkable
class NewsSource(Protocol):
    name: str

    def fetch(self, q: NewsQuery) -> list[NewsArticle]:
        ...


# --------------------------------------------------------------------- adapters
class GoogleNewsSource:
    """Google News RSS (aggregates Investing.com, CNBC, CoinDesk, Reuters…).

    Reliable, no key, historical via after:/before: date operators.
    """

    name = "google_news"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def fetch(self, q: NewsQuery) -> list[NewsArticle]:
        try:
            import feedparser
        except Exception:  # noqa: BLE001
            logger.info("[news:google] feedparser unavailable — skipping")
            return []
        from urllib.parse import quote

        after = q.w_start.strftime("%Y-%m-%d")
        before = (q.w_end + timedelta(days=1)).strftime("%Y-%m-%d")  # before: is exclusive
        query = f"{q.google_query} after:{after} before:{before}"
        base = self.cfg.get("base_url", "https://news.google.com/rss/search")
        url = (f"{base}?q={quote(query)}"
               f"&hl={self.cfg.get('hl', 'en-US')}"
               f"&gl={self.cfg.get('gl', 'US')}"
               f"&ceid={self.cfg.get('ceid', 'US:en')}")
        try:
            feed = feedparser.parse(url)
        except Exception as e:  # noqa: BLE001
            logger.warning("[news:google] parse failed %s %s: %s", q.asset_id, q.w_start, e)
            return []

        cap = int(self.cfg.get("max_articles", 40))
        out: list[NewsArticle] = []
        for e in feed.entries:
            pub = None
            if getattr(e, "published_parsed", None):
                try:
                    pub = date(e.published_parsed.tm_year, e.published_parsed.tm_mon,
                               e.published_parsed.tm_mday)
                except Exception:  # noqa: BLE001
                    pub = None
            if pub is not None and not (q.w_start <= pub <= q.w_end):
                continue  # strict week filter (Google before:/after: can bleed a day)
            src = None
            if hasattr(e, "source") and isinstance(getattr(e, "source"), dict):
                src = e.source.get("title")
            title = getattr(e, "title", None)
            if title and " - " in title and not src:
                title, _, src = title.rpartition(" - ")  # strip " - Source" suffix
            out.append(NewsArticle(
                title=title,
                url=getattr(e, "link", None),
                source=src or "Google News",
                published_at=getattr(e, "published", None),
                language="en",
            ))
            if len(out) >= cap:
                break
        if out:
            logger.info("[news:google] %d for %s %s", len(out), q.asset_id, q.w_start)
        return out


class GDELTSource:
    """GDELT DOC 2.0 fallback — public, historical, but rate-limited (≤1 req/5s)."""

    name = "gdelt"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self._last_call = 0.0

    def fetch(self, q: NewsQuery) -> list[NewsArticle]:
        from urllib.parse import urlencode
        try:
            import requests
        except Exception:  # noqa: BLE001
            return []

        # Global pacing to honour GDELT's "≤ 1 request / 5s".
        min_interval = float(self.cfg.get("min_interval_s", 6))
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        start_dt = datetime(q.w_start.year, q.w_start.month, q.w_start.day, tzinfo=timezone.utc)
        end_dt = datetime(q.w_end.year, q.w_end.month, q.w_end.day, 23, 59, 59, tzinfo=timezone.utc)
        params = {
            "query": q.gdelt_query,
            "mode": "artlist",
            "maxrecords": str(self.cfg.get("max_records", 60)),
            "format": "json",
            "sort": "datedesc",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        }
        url = f"{self.cfg.get('gdelt_base_url', 'https://api.gdeltproject.org/api/v2/doc/doc')}?{urlencode(params)}"
        for attempt, wait in enumerate(self.cfg.get("backoff_seconds", [0, 8, 16, 24])):
            if wait:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = requests.get(url, timeout=self.cfg.get("request_timeout_s", 20),
                                    headers={"User-Agent": "usdcop-analysis/1.0"})
            except Exception as e:  # noqa: BLE001
                logger.warning("[news:gdelt] request error %s %s: %s", q.asset_id, q.w_start, e)
                continue
            if resp.status_code == 429:
                logger.info("[news:gdelt] 429 (try %d) %s %s", attempt + 1, q.asset_id, q.w_start)
                continue
            if resp.status_code != 200:
                logger.warning("[news:gdelt] HTTP %s %s %s", resp.status_code, q.asset_id, q.w_start)
                return []
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                return []
            return [
                NewsArticle(
                    title=a.get("title"), url=a.get("url"), source=a.get("domain"),
                    published_at=a.get("seendate"), language=a.get("language"),
                )
                for a in (data.get("articles") or [])
            ]
        logger.warning("[news:gdelt] gave up after backoff %s %s", q.asset_id, q.w_start)
        return []


# ---------------------------------------------------------------- registry/factory
# name -> adapter class. Register a new source here to make it available in the SSOT.
SOURCE_REGISTRY: dict[str, type] = {
    "google_news": GoogleNewsSource,
    "gdelt": GDELTSource,
}


def build_news_sources(news_cfg: dict) -> list[NewsSource]:
    """Instantiate the ordered source list declared in the SSOT `news.sources`.

    Each adapter receives its own config sub-block (news[<name>]) merged with the
    shared news config, so per-source knobs live next to the source.
    """
    order = news_cfg.get("sources") or ["google_news", "gdelt"]
    sources: list[NewsSource] = []
    for name in order:
        cls = SOURCE_REGISTRY.get(name)
        if cls is None:
            logger.warning("[news] unknown source '%s' in config — skipping", name)
            continue
        sub = dict(news_cfg)          # shared knobs (gdelt_base_url, timeouts…)
        sub.update(news_cfg.get(name, {}))  # per-source overrides (news.google_news.*)
        sources.append(cls(sub))
    return sources


@dataclass
class AssetNewsFetcher:
    """Facade: run configured sources for a NewsQuery per the chosen strategy.

    strategy = 'first_nonempty' (default): return the first source that yields
    articles (primary→fallback). 'aggregate': merge + de-dup across all sources.
    """

    sources: list[NewsSource]
    strategy: str = "first_nonempty"

    def fetch(self, q: NewsQuery) -> list[NewsArticle]:
        if self.strategy == "aggregate":
            seen: set[str] = set()
            merged: list[NewsArticle] = []
            for s in self.sources:
                for a in _safe(s, q):
                    k = a.key()
                    if k and k not in seen:
                        seen.add(k)
                        merged.append(a)
            return merged
        # first_nonempty
        for s in self.sources:
            arts = _safe(s, q)
            if arts:
                return arts
        return []


def _safe(source: NewsSource, q: NewsQuery) -> list[NewsArticle]:
    try:
        return source.fetch(q)
    except Exception as e:  # noqa: BLE001
        logger.warning("[news] source %s failed for %s %s: %s",
                       getattr(source, "name", "?"), q.asset_id, q.w_start, e)
        return []
