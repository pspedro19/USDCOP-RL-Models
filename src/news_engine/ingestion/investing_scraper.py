"""
Investing.com Search Scraper (SDD-02 §2.5)
============================================
Fetches Colombia-focused news via the internal search API discovered at:

    POST /search/service/SearchInnerPage
    Body: search_text=<query>&tab=news&offset=<N>&limit=<N>

This is the same endpoint the website calls on infinite-scroll.  Returns JSON
with ``news[]`` array containing dataID, date, dateTimestamp, link, content, etc.

Searches both www.investing.com (EN) and es.investing.com (ES).
Supports full pagination (100 items/batch), checkpoint/resume, and dedup by dataID.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.news_engine.config import ScraperConfig
from src.news_engine.ingestion.base_adapter import SourceAdapter, parse_date_flexible
from src.news_engine.models import RawArticle

logger = logging.getLogger(__name__)

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False
    logger.warning("cloudscraper not installed — pip install cloudscraper")


# ---------------------------------------------------------------------------
# Relative-date patterns (EN + ES)
# ---------------------------------------------------------------------------

_RELATIVE_PATTERNS = [
    (re.compile(r"(\d+)\s*hour", re.IGNORECASE), "hours"),
    (re.compile(r"(\d+)\s*hora", re.IGNORECASE), "hours"),
    (re.compile(r"(\d+)\s*minute", re.IGNORECASE), "minutes"),
    (re.compile(r"(\d+)\s*minuto", re.IGNORECASE), "minutes"),
    (re.compile(r"(\d+)\s*day", re.IGNORECASE), "days"),
    (re.compile(r"(\d+)\s*d[ií]a", re.IGNORECASE), "days"),
    (re.compile(r"(\d+)\s*week", re.IGNORECASE), "weeks"),
    (re.compile(r"(\d+)\s*semana", re.IGNORECASE), "weeks"),
    (re.compile(r"(\d+)\s*month", re.IGNORECASE), "months"),
    (re.compile(r"(\d+)\s*mes", re.IGNORECASE), "months"),
]

# Absolute date formats (EN and ES)
_DATE_FORMATS = [
    "%b %d, %Y",          # Feb 25, 2026
    "%B %d, %Y",          # February 25, 2026
    "%d %b %Y",           # 25 Feb 2026
    "%d %b. %Y",          # 25 feb. 2026
    "%d.%m.%Y",           # 25.02.2026  (ES domain format)
    "%d/%m/%Y",           # 25/02/2026
    "%Y-%m-%d",           # 2026-02-25
]

# Spanish month abbreviations -> English for strptime
_ES_MONTH_MAP = {
    "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec",
    "enero": "January", "febrero": "February", "marzo": "March",
    "abril": "April", "mayo": "May", "junio": "June",
    "julio": "July", "agosto": "August", "septiembre": "September",
    "octubre": "October", "noviembre": "November", "diciembre": "December",
}


class InvestingSearchScraper(SourceAdapter):
    """Investing.com search scraper using the internal SearchInnerPage API.

    Primary method: POST to ``/search/service/SearchInnerPage`` with offset/limit
    pagination (same endpoint the website calls on infinite scroll).

    Searches EN (www.investing.com) and ES (es.investing.com) domains.
    """

    source_id = "investing"
    source_name = "Investing.com"

    DOMAINS = {
        "en": "https://www.investing.com",
        "es": "https://es.investing.com",
    }
    API_PATH = "/search/service/SearchInnerPage"

    # Pagination
    BATCH_SIZE = 100
    # Safety cap: stop paginating after this many articles per query
    MAX_ARTICLES_PER_QUERY = 5000

    CHECKPOINT_DIR = Path("data/news")
    CHECKPOINT_FILE = Path("data/news/investing_search_checkpoint.json")

    def __init__(self, config: ScraperConfig | None = None):
        super().__init__(config)
        self.cfg = config or ScraperConfig()
        self._sessions: dict[str, object] = {}  # domain_key -> session
        self._request_count = 0

    # ------------------------------------------------------------------
    # SourceAdapter interface
    # ------------------------------------------------------------------

    def fetch_latest(self, hours_back: int = 24) -> list[RawArticle]:
        """Fetch recent articles from all configured queries."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        all_articles: list[RawArticle] = []
        seen_ids: set[int] = set()

        queries = self._build_query_list()
        for query, domain_key, lang in queries:
            try:
                articles, new_ids = self._paginate_query(
                    query, domain_key, lang, seen_ids,
                    max_offset=300,  # ~300 articles max for latest
                    stop_before=cutoff,
                )
                all_articles.extend(articles)
                seen_ids.update(new_ids)
            except Exception as e:
                logger.warning("[investing] Query '%s' (%s) failed: %s", query, domain_key, e)
            self._delay_between_queries()

        # Filter by cutoff (redundant safety — _paginate_query already stops early)
        filtered = [a for a in all_articles if a.published_at >= cutoff]
        result = self._deduplicate(filtered)
        logger.info("[investing] fetch_latest(%dh): %d articles", hours_back, len(result))
        return result

    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawArticle]:
        """Fetch ALL articles from the search API, then filter to [start, end].

        The API doesn't support date filters, so we paginate through all results
        for each query and keep only those within range.  Checkpoint saves progress
        per query to allow resume on interruption.
        """
        all_articles: list[RawArticle] = []
        seen_ids: set[int] = set()

        # Ensure tz-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        queries = self._build_query_list()
        checkpoint = self._load_checkpoint()
        completed_queries = set(checkpoint.get("completed_queries", []))

        for i, (query, domain_key, lang) in enumerate(queries):
            qkey = f"{domain_key}:{query}"
            if qkey in completed_queries:
                logger.info("[investing] Skipping completed query '%s' (%s)", query, domain_key)
                continue

            logger.info(
                "[investing] Query %d/%d: '%s' (%s)",
                i + 1, len(queries), query, domain_key,
            )

            try:
                articles, new_ids = self._paginate_query(
                    query, domain_key, lang, seen_ids,
                    max_offset=self.MAX_ARTICLES_PER_QUERY,
                )
                # Filter to date range
                in_range = []
                for art in articles:
                    pub = art.published_at
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=UTC)
                    if start_date <= pub <= end_date + timedelta(days=1):
                        in_range.append(art)

                all_articles.extend(in_range)
                seen_ids.update(new_ids)
                logger.info(
                    "[investing]   -> %d total, %d in range [%s, %s] (cumulative: %d)",
                    len(articles), len(in_range),
                    start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                    len(all_articles),
                )
            except Exception as e:
                logger.warning("[investing] Query '%s' (%s) failed: %s", query, domain_key, e)

            completed_queries.add(qkey)
            self._save_checkpoint({"completed_queries": list(completed_queries)})
            self._delay_between_queries()

        self._clear_checkpoint()
        result = self._deduplicate(all_articles)
        logger.info("[investing] fetch_historical: %d total articles", len(result))
        return result

    def health_check(self) -> bool:
        """Verify search API is reachable."""
        try:
            session = self._get_session("en")
            resp = session.post(
                f"{self.DOMAINS['en']}{self.API_PATH}",
                data={"search_text": "Colombia", "tab": "news", "limit": "1"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return "news" in data
            return False
        except Exception as e:
            logger.warning("[investing] Health check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal API pagination
    # ------------------------------------------------------------------

    def _paginate_query(
        self,
        query: str,
        domain_key: str,
        lang: str,
        global_seen_ids: set[int],
        max_offset: int = 5000,
        stop_before: datetime | None = None,
    ) -> tuple[list[RawArticle], set[int]]:
        """Paginate through SearchInnerPage API for a single query.

        Args:
            query: Search text
            domain_key: "en" or "es"
            lang: Language tag for RawArticle
            global_seen_ids: IDs already collected (skip duplicates across queries)
            max_offset: Stop after this many items
            stop_before: If set, stop when articles are older than this datetime

        Returns:
            (articles, local_seen_ids) — articles collected and their dataIDs
        """
        articles: list[RawArticle] = []
        local_seen: set[int] = set()
        consecutive_empty = 0
        consecutive_all_dupes = 0

        base_url = self.DOMAINS[domain_key]
        api_url = f"{base_url}{self.API_PATH}"

        offset = 0
        while offset < max_offset:
            batch = self._fetch_batch(api_url, query, domain_key, offset, self.BATCH_SIZE)
            if batch is None or len(batch) == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                offset += self.BATCH_SIZE
                continue

            consecutive_empty = 0
            new_in_batch = 0
            oldest_in_batch = None

            for item in batch:
                data_id = item.get("dataID")
                if not data_id:
                    continue

                # Skip globally-seen IDs (across queries)
                if data_id in global_seen_ids or data_id in local_seen:
                    continue

                local_seen.add(data_id)
                new_in_batch += 1

                article = self._parse_api_item(item, base_url, lang)
                if article:
                    articles.append(article)
                    if oldest_in_batch is None or article.published_at < oldest_in_batch:
                        oldest_in_batch = article.published_at

            # Early stop: if all items in batch were already seen, we're looping
            if new_in_batch == 0:
                consecutive_all_dupes += 1
                if consecutive_all_dupes >= 3:
                    logger.debug("[investing] 3 consecutive all-dupe batches, stopping")
                    break
            else:
                consecutive_all_dupes = 0

            # Early stop: if oldest article in batch is before cutoff
            if stop_before and oldest_in_batch:
                if oldest_in_batch.tzinfo is None:
                    oldest_in_batch = oldest_in_batch.replace(tzinfo=UTC)
                if oldest_in_batch < stop_before:
                    break

            offset += self.BATCH_SIZE
            time.sleep(random.uniform(1.0, 2.0))

        logger.debug(
            "[investing] Query '%s' (%s): %d articles, %d unique IDs",
            query[:30], domain_key, len(articles), len(local_seen),
        )
        return articles, local_seen

    def _fetch_batch(
        self,
        api_url: str,
        query: str,
        domain_key: str,
        offset: int,
        limit: int,
    ) -> list[dict] | None:
        """Fetch a single batch from SearchInnerPage API."""
        self._rotate_session_if_needed()
        session = self._get_session(domain_key)

        payload = {
            "search_text": query,
            "tab": "news",
            "offset": str(offset),
            "limit": str(limit),
        }

        for attempt in range(self._max_retries):
            try:
                resp = session.post(api_url, data=payload, timeout=self.cfg.request_timeout)
                self._request_count += 1

                if resp.status_code == 429:
                    wait = self._retry_delay * (2 ** attempt) + random.uniform(5, 15)
                    logger.warning("[investing] Rate limited (429). Waiting %.1fs...", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 403:
                    logger.warning("[investing] Blocked (403). Rotating session...")
                    self._sessions.pop(domain_key, None)
                    self._request_count = 0
                    time.sleep(self._retry_delay * (2 ** attempt) + random.uniform(3, 8))
                    session = self._get_session(domain_key)
                    continue

                if resp.status_code != 200:
                    logger.debug("[investing] HTTP %d for offset=%d", resp.status_code, offset)
                    return None

                data = resp.json()
                return data.get("news", [])

            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait = self._retry_delay * (2 ** attempt)
                    logger.debug("[investing] Attempt %d failed: %s. Retry in %.1fs", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.warning("[investing] All retries exhausted at offset=%d: %s", offset, e)
                    return None

        return None

    # ------------------------------------------------------------------
    # Parse a single API item -> RawArticle
    # ------------------------------------------------------------------

    def _parse_api_item(
        self, item: dict, base_url: str, lang: str,
    ) -> RawArticle | None:
        """Convert a SearchInnerPage JSON item to RawArticle."""
        link = item.get("link", "")
        title = item.get("name", "") or item.get("description", "")
        if not link or not title:
            return None

        # Resolve relative URLs
        if link.startswith("/"):
            link = base_url + link

        # Parse date — prefer Unix timestamp, fall back to date string
        published_at = None
        ts = item.get("dateTimestamp")
        if ts:
            try:
                published_at = datetime.fromtimestamp(int(ts), tz=UTC)
            except (ValueError, OSError, OverflowError):
                pass

        if not published_at:
            date_str = item.get("date", "")
            published_at = self._parse_article_date(date_str)

        if not published_at:
            published_at = datetime.now(UTC)

        content = item.get("content", "") or ""
        summary = content[:300] if content else None

        return RawArticle(
            url=link,
            title=title.strip(),
            source_id=self.source_id,
            published_at=published_at,
            summary=summary,
            language=lang,
            country_focus="CO",
            raw_json={
                "dataID": item.get("dataID"),
                "providerName": item.get("providerName", ""),
                "domain": base_url,
            },
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self, domain_key: str = "en"):
        """Create or return cached cloudscraper session per domain."""
        if not HAS_CLOUDSCRAPER:
            raise ImportError("cloudscraper required: pip install cloudscraper")

        if domain_key not in self._sessions:
            self._sessions[domain_key] = self._create_session(domain_key)
        return self._sessions[domain_key]

    def _create_session(self, domain_key: str):
        """Create a fresh cloudscraper session."""
        session = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "windows",
                "desktop": True,
            },
        )
        base_url = self.DOMAINS[domain_key]
        session.headers.update({
            "User-Agent": self.cfg.user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{base_url}/search/?q=Colombia&tab=news",
            "Origin": base_url,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        return session

    def _rotate_session_if_needed(self):
        """Rotate sessions every N requests."""
        if self._request_count >= self.cfg.investing_session_rotation_every:
            logger.debug("[investing] Rotating sessions after %d requests", self._request_count)
            self._sessions.clear()
            self._request_count = 0

    # ------------------------------------------------------------------
    # Date parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_article_date(text: str) -> datetime | None:
        """Parse article date from various EN/ES formats.

        Handles:
        - Absolute: "Feb 25, 2026", "25.02.2026", "25/02/2026"
        - Relative: "5 hours ago", "Hace 3 horas", "2 days ago"
        - ISO 8601: "2026-02-25T19:21:28"
        """
        if not text:
            return None

        text = text.strip()

        # Try parse_date_flexible first (RFC 2822, ISO 8601)
        result = parse_date_flexible(text)
        if result:
            return result

        # Relative dates ("5 hours ago", "Hace 3 dias")
        for pattern, unit in _RELATIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                n = int(match.group(1))
                now = datetime.now(UTC)
                if unit == "minutes":
                    return now - timedelta(minutes=n)
                elif unit == "hours":
                    return now - timedelta(hours=n)
                elif unit == "days":
                    return now - timedelta(days=n)
                elif unit == "weeks":
                    return now - timedelta(weeks=n)
                elif unit == "months":
                    return now - timedelta(days=n * 30)

        # Normalize Spanish month names
        normalized = text.lower()
        for es_month, en_month in _ES_MONTH_MAP.items():
            normalized = normalized.replace(es_month, en_month.lower())

        normalized = normalized.replace(" de ", " ")
        normalized = re.sub(r"(\w{3})\.", r"\1", normalized)

        # Try absolute formats
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(normalized, fmt.lower()).replace(tzinfo=UTC)
            except ValueError:
                continue

        # Try case-sensitive formats
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue

        return None

    # ------------------------------------------------------------------
    # Date windowing (kept for interface compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_windows(
        start: datetime, end: datetime, window_days: int = 30,
    ) -> list[tuple[datetime, datetime]]:
        """Split a date range into windows of window_days."""
        windows = []
        current = start
        while current < end:
            window_end = min(current + timedelta(days=window_days - 1), end)
            windows.append((current, window_end))
            current = window_end + timedelta(days=1)
        return windows

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        if self.CHECKPOINT_FILE.exists():
            try:
                return json.loads(self.CHECKPOINT_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checkpoint(self, update: dict):
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cp = self._load_checkpoint()
        cp.update(update)
        self.CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))

    def _clear_checkpoint(self):
        if self.CHECKPOINT_FILE.exists():
            self.CHECKPOINT_FILE.unlink()

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_query_list(self) -> list[tuple[str, str, str]]:
        """Build list of (query, domain_key, language) tuples."""
        queries = []
        for q in self.cfg.investing_search_queries_en:
            queries.append((q, "en", "en"))
        if self.cfg.investing_use_es_domain:
            for q in self.cfg.investing_search_queries_es:
                queries.append((q, "es", "es"))
        return queries

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def _delay_between_queries(self):
        time.sleep(random.uniform(
            self.cfg.investing_search_delay_min,
            self.cfg.investing_search_delay_max,
        ))

    def _delay_between_windows(self):
        time.sleep(random.uniform(5.0, 10.0))
