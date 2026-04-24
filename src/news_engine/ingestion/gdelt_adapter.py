"""
GDELT DOC + Context Adapter (SDD-02 SS2.1, SS2.2)
==================================================
Fetches articles and sentiment timelines from GDELT APIs.
Two sub-adapters: DOC (articles + tone) and Context (entity timelines).

- GDELT DOC 2.0: No auth, rate limit 1 req per 5 seconds, ~3 months retention
- Tone: -100 to +100 (practical: -20 to +20)
- Volume: article count per 15-min bucket
- OR terms in queries MUST be wrapped in parentheses
- sourcelang:english/spanish as inline query filter
- startdatetime/enddatetime format: YYYYMMDDHHmmss
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import requests

from src.news_engine.config import GDELTConfig
from src.news_engine.ingestion.base_adapter import SourceAdapter
from src.news_engine.models import GDELTTimelinePoint, RawArticle

logger = logging.getLogger(__name__)


class GDELTDocAdapter(SourceAdapter):
    """GDELT DOC 2.0 adapter -- articles + tone scores.

    Key fixes over original:
    - OR queries wrapped in parentheses (GDELT requirement)
    - Rate limit 6s (GDELT enforces 5s minimum)
    - Handles non-JSON error responses gracefully
    - Proper startdatetime/enddatetime for date range queries
    - 7-day windowing for historical extraction
    - Inline sourcelang: filter for language separation
    """

    source_id = "gdelt_doc"
    source_name = "GDELT DOC 2.0"

    def __init__(self, config: GDELTConfig | None = None):
        super().__init__(config)
        self.cfg = config or GDELTConfig()
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        """Fetch recent articles across all configured queries."""
        all_articles = []
        queries = self._build_query_list()
        for query, lang in queries:
            try:
                articles = self._fetch_query(
                    query, lang=lang, timespan_hours=hours_back,
                )
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"[gdelt_doc] Query '{query}' failed: {e}")
        return self._deduplicate(all_articles)

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        """Fetch articles for a date range using startdatetime/enddatetime.

        GDELT retains ~3 months of data. Requests for older data return empty.
        Uses windowing (default 7 days per window) to avoid the 250-record cap.
        """
        all_articles = []
        windows = self._generate_windows(
            start_date, end_date, self.cfg.historical_window_days,
        )
        queries = self._build_query_list()
        total_windows = len(windows) * len(queries)
        completed = 0

        for w_start, w_end in windows:
            for query, lang in queries:
                completed += 1
                try:
                    articles = self._fetch_query(
                        query,
                        lang=lang,
                        start_dt=w_start,
                        end_dt=w_end,
                    )
                    all_articles.extend(articles)
                    if completed % 10 == 0:
                        logger.info(
                            f"[gdelt_doc] Progress: {completed}/{total_windows} "
                            f"windows, {len(all_articles)} articles so far"
                        )
                except Exception as e:
                    logger.warning(
                        f"[gdelt_doc] Window {w_start.date()}-{w_end.date()} "
                        f"query '{query}' failed: {e}"
                    )

        result = self._deduplicate(all_articles)
        logger.info(
            f"[gdelt_doc] Historical: {len(result)} unique articles "
            f"({start_date.date()} to {end_date.date()})"
        )
        return result

    def health_check(self) -> bool:
        """Verify GDELT API is responding with a simple query."""
        try:
            self._rate_limit_wait()
            params = {
                "query": "Colombia",
                "mode": "artlist",
                "maxrecords": "1",
                "format": "json",
                "timespan": "72h",
            }
            url = f"{self.cfg.doc_base_url}?{urlencode(params)}"
            resp = requests.get(url, timeout=self.cfg.request_timeout)
            self._last_request_time = time.monotonic()
            if resp.status_code != 200:
                return False
            data = self._safe_parse_json(resp)
            return data is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_query_list(self) -> list[tuple[str, str]]:
        """Build list of (query, language) tuples from config."""
        queries = []
        for q in self.cfg.queries_en:
            queries.append((q, "en"))
        if self.cfg.use_es_queries:
            for q in self.cfg.queries_es:
                queries.append((q, "es"))
        return queries

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def _fetch_query(
        self,
        query: str,
        lang: str = "en",
        timespan_hours: int | None = None,
        start_dt: datetime | None = None,
        end_dt: datetime | None = None,
    ) -> list[RawArticle]:
        """Fetch articles for a single query.

        Either use timespan_hours (for fetch_latest) or start_dt/end_dt
        (for fetch_historical). If both provided, start_dt/end_dt takes precedence.
        """
        self._rate_limit_wait()

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(self.cfg.max_records),
            "format": "json",
            "sort": "datedesc",
        }

        if start_dt and end_dt:
            params["startdatetime"] = start_dt.strftime("%Y%m%d%H%M%S")
            params["enddatetime"] = end_dt.strftime("%Y%m%d%H%M%S")
        elif timespan_hours:
            params["timespan"] = f"{timespan_hours}h"
        else:
            params["timespan"] = "24h"

        url = f"{self.cfg.doc_base_url}?{urlencode(params)}"

        try:
            resp = requests.get(url, timeout=self.cfg.request_timeout)
            self._last_request_time = time.monotonic()
        except requests.exceptions.Timeout:
            logger.warning(f"[gdelt_doc] Timeout for query '{query}'")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"[gdelt_doc] Request error for query '{query}': {e}")
            return []

        # Handle non-JSON responses (rate limit errors, HTML error pages)
        data = self._safe_parse_json(resp)
        if data is None:
            # Check if rate limited
            if "limit requests" in resp.text.lower():
                logger.warning(
                    "[gdelt_doc] Rate limited. Waiting 15s before retry..."
                )
                time.sleep(15)
                # Single retry
                try:
                    resp = requests.get(url, timeout=self.cfg.request_timeout)
                    self._last_request_time = time.monotonic()
                    data = self._safe_parse_json(resp)
                    if data is None:
                        return []
                except Exception:
                    return []
            else:
                logger.warning(
                    f"[gdelt_doc] Non-JSON response for '{query}': "
                    f"{resp.text[:100]}"
                )
                return []

        articles_raw = data.get("articles", [])
        articles = []
        for art in articles_raw:
            parsed = self._parse_article(art, lang)
            if parsed:
                articles.append(parsed)

        logger.debug(f"[gdelt_doc] Query '{query}': {len(articles)} articles")
        return articles

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_article(self, art: dict, default_lang: str) -> RawArticle | None:
        """Parse a single GDELT article dict into RawArticle."""
        url = art.get("url", "")
        title = art.get("title", "")
        if not url or not title:
            return None

        try:
            published = datetime.strptime(
                art.get("seendate", ""),
                "%Y%m%dT%H%M%SZ",
            ).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            published = None

        if published is None:
            return None

        return RawArticle(
            url=url,
            title=title,
            source_id=self.source_id,
            published_at=published,
            summary=title,
            language=art.get("language", default_lang),
            country_focus=self._extract_country(art),
            gdelt_tone=self._parse_tone(art.get("tone", "")),
            gdelt_volume=art.get("articlecount"),
            image_url=art.get("socialimage"),
            raw_json=art,
        )

    # ------------------------------------------------------------------
    # Date windowing
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_windows(
        start: datetime,
        end: datetime,
        window_days: int,
    ) -> list[tuple[datetime, datetime]]:
        """Generate date windows for historical extraction.

        Returns list of (window_start, window_end) tuples.
        Uses 7-day windows by default to stay under maxrecords=250 cap.
        """
        if start >= end:
            return []

        windows = []
        current = start
        while current < end:
            w_end = min(current + timedelta(days=window_days), end)
            windows.append((current, w_end))
            current = w_end + timedelta(seconds=1)

        return windows

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit_wait(self):
        """Enforce minimum delay between GDELT requests."""
        if self._last_request_time > 0:
            elapsed = time.monotonic() - self._last_request_time
            wait = self.cfg.rate_limit_seconds - elapsed
            if wait > 0:
                time.sleep(wait)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json(resp: requests.Response) -> dict | None:
        """Safely parse JSON from response, returning None on failure.

        GDELT returns plain text error messages (not JSON) when:
        - Rate limited: "Please limit requests to one every 5 seconds."
        - Invalid query: HTML error pages
        - Server errors: Various text responses
        """
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "")
        # Quick check: if it doesn't look like JSON, skip parsing
        text = resp.text.strip()
        if not text or (not text.startswith("{") and not text.startswith("[")):
            return None
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _parse_tone(tone_str: str) -> float | None:
        """Parse GDELT tone string (comma-separated values, first is avg tone)."""
        if not tone_str:
            return None
        try:
            return float(tone_str.split(",")[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _extract_country(article: dict) -> str:
        """Extract primary country from GDELT article metadata."""
        source_country = article.get("sourcecountry", "")
        if "colombia" in source_country.lower() or "CO" in source_country:
            return "CO"
        return "US"  # Default for English-language financial news

    # ------------------------------------------------------------------
    # Tone + Volume timelines
    # ------------------------------------------------------------------

    def fetch_tone_timeline(
        self,
        query: str = "Colombia",
        hours_back: int = 72,
    ) -> list[GDELTTimelinePoint]:
        """Fetch tone timeline for sentiment tracking."""
        return self._fetch_timeline(query, "timelinetone", "tone", hours_back)

    def fetch_volume_timeline(
        self,
        query: str = "Colombia",
        hours_back: int = 72,
    ) -> list[GDELTTimelinePoint]:
        """Fetch article volume timeline."""
        return self._fetch_timeline(query, "timelinevol", "volume", hours_back)

    def _fetch_timeline(
        self,
        query: str,
        mode: str,
        series_name: str,
        hours_back: int,
    ) -> list[GDELTTimelinePoint]:
        """Generic timeline fetcher for tone/volume modes."""
        self._rate_limit_wait()
        params = {
            "query": query,
            "mode": mode,
            "format": "json",
            "timespan": f"{hours_back}h",
        }
        url = f"{self.cfg.doc_base_url}?{urlencode(params)}"
        try:
            resp = requests.get(url, timeout=self.cfg.request_timeout)
            self._last_request_time = time.monotonic()
            data = self._safe_parse_json(resp)
            if data is None:
                return []
            points = []
            for timeline in data.get("timeline", []):
                for point in timeline.get("data", []):
                    try:
                        dt = datetime.strptime(
                            point["date"], "%Y%m%dT%H%M%SZ"
                        ).replace(tzinfo=UTC)
                        points.append(GDELTTimelinePoint(
                            date=dt,
                            value=float(point.get("value", 0)),
                            series=series_name,
                        ))
                    except (ValueError, KeyError):
                        continue
            return points
        except Exception as e:
            logger.warning(f"[gdelt_doc] {mode} timeline failed: {e}")
            return []


class GDELTContextAdapter(SourceAdapter):
    """GDELT Context 2.0 adapter -- entity/theme context timelines.

    Context API returns sentiment timelines, NOT articles.
    fetch_latest() and fetch_historical() return empty lists by design.
    Use fetch_context_timeline() for sentiment data.
    """

    source_id = "gdelt_context"
    source_name = "GDELT Context 2.0"

    def __init__(self, config: GDELTConfig | None = None):
        super().__init__(config)
        self.cfg = config or GDELTConfig()

    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        """Context API returns timelines, not articles. Returns empty list."""
        return []

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        return []

    def health_check(self) -> bool:
        try:
            params = {
                "query": "Colombia",
                "mode": "timelinetone",
                "format": "json",
                "timespan": "72h",
            }
            url = f"{self.cfg.context_base_url}?{urlencode(params)}"
            resp = requests.get(url, timeout=15)
            return resp.status_code == 200
        except Exception:
            return False

    def fetch_context_timeline(
        self,
        entity: str = "Colombia",
        hours_back: int = 72,
    ) -> list[GDELTTimelinePoint]:
        """Fetch entity context timeline from GDELT Context API."""
        params = {
            "query": entity,
            "mode": "timelinetone",
            "format": "json",
            "timespan": f"{hours_back}h",
        }
        url = f"{self.cfg.context_base_url}?{urlencode(params)}"
        try:
            resp = requests.get(url, timeout=15)
            data = GDELTDocAdapter._safe_parse_json(resp)
            if data is None:
                return []
            points = []
            for timeline in data.get("timeline", []):
                for point in timeline.get("data", []):
                    try:
                        dt = datetime.strptime(
                            point["date"], "%Y%m%dT%H%M%SZ"
                        ).replace(tzinfo=UTC)
                        points.append(GDELTTimelinePoint(
                            date=dt,
                            value=float(point.get("value", 0)),
                            series="context_tone",
                        ))
                    except (ValueError, KeyError):
                        continue
            return points
        except Exception as e:
            logger.warning(f"[gdelt_context] Context timeline failed: {e}")
            return []
