#!/usr/bin/env python3
"""
Historical Colombia News Scraper (2020-2026)
=============================================
Fetches Colombia-focused financial news from multiple sources,
deduplicates, and exports to CSV + JSON.

Sources:
  1. Google News RSS  — Free, reliable, date-filtered, monthly windows (PRIMARY)
  2. GDELT Doc API    — Free, no key, monthly windows, global coverage
  3. NewsAPI          — Last 30 days (free tier), EN/ES
  4. Portafolio       — RSS, ES/Colombia (recent only)
  5. Investing.com    — Search scraper, EN+ES, Colombia-focused (cloudscraper)

Usage:
  python scripts/scrape_colombia_news_historical.py
  python scripts/scrape_colombia_news_historical.py --start 2024-01-01 --end 2026-02-25
  python scripts/scrape_colombia_news_historical.py --resume
  python scripts/scrape_colombia_news_historical.py --source google
  python scripts/scrape_colombia_news_historical.py --source gdelt
  python scripts/scrape_colombia_news_historical.py --source investing

Output:
  data/news/colombia_news_historical.csv
  data/news/colombia_news_historical.json
  data/news/scrape_checkpoint.json  (resume state)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "data" / "news"
CHECKPOINT_FILE = OUTPUT_DIR / "scrape_checkpoint.json"
CSV_FILE = OUTPUT_DIR / "colombia_news_historical.csv"
JSON_FILE = OUTPUT_DIR / "colombia_news_historical.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

COLUMNS = [
    "date", "title", "url", "source", "category", "summary",
    "sentiment_tone", "language", "country_focus", "url_hash",
]


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ============================================================================
# Google News RSS (PRIMARY — reliable, fast, date-filtered)
# ============================================================================

GOOGLE_NEWS_QUERIES_EN = [
    "Colombia peso exchange rate",
    "USDCOP currency",
    "Banco Republica interest rate Colombia",
    "Colombia economy inflation",
    "Colombia oil petroleum exports",
    "Colombia foreign investment trade",
]

GOOGLE_NEWS_QUERIES_ES = [
    "dolar peso colombiano",
    "Banco Republica tasa interes",
    "Colombia inflacion economia",
    "petroleo Colombia exportaciones",
    "USDCOP devaluacion",
    "remesas Colombia inversion extranjera",
]


def scrape_google_news_historical(
    start: date,
    end: date,
    window_days: int = 30,
    checkpoint: dict = None,
) -> list[dict]:
    """Scrape Google News RSS in monthly windows from start to end."""
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser required: pip install feedparser")
        return []

    all_articles = []
    seen_urls = set()
    current = start

    # Resume from checkpoint
    if checkpoint and checkpoint.get("google_last_end"):
        resume_date = date.fromisoformat(checkpoint["google_last_end"])
        if resume_date > start:
            current = resume_date + timedelta(days=1)
            logger.info(f"Google News: Resuming from {current}")

    total_windows = ((end - current).days // window_days) + 1
    window_num = 0

    all_queries = (
        [(q, "en-US", "US:en") for q in GOOGLE_NEWS_QUERIES_EN]
        + [(q, "es-419", "CO:es") for q in GOOGLE_NEWS_QUERIES_ES]
    )

    while current < end:
        window_end = min(current + timedelta(days=window_days - 1), end)
        window_num += 1

        window_count = 0
        for query, hl, ceid in all_queries:
            date_filter = f"after:{current.isoformat()} before:{(window_end + timedelta(days=1)).isoformat()}"
            full_query = f"{query} {date_filter}"

            try:
                resp = requests.get(
                    "https://news.google.com/rss/search",
                    params={"q": full_query, "hl": hl, "gl": hl.split("-")[0], "ceid": ceid},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.debug(f"Google News HTTP {resp.status_code} for '{query[:30]}'")
                    time.sleep(2)
                    continue

                feed = feedparser.parse(resp.content)
                for entry in feed.entries:
                    u = entry.get("link", "")
                    h = url_hash(u)
                    if h in seen_urls:
                        continue
                    seen_urls.add(h)

                    # Parse date from published field
                    art_date = ""
                    pub = entry.get("published", "")
                    if pub:
                        try:
                            from src.news_engine.ingestion.base_adapter import parse_date_flexible
                            dt = parse_date_flexible(pub)
                            if dt:
                                art_date = dt.strftime("%Y-%m-%d")
                        except Exception:
                            pass

                    if not art_date:
                        art_date = current.isoformat()

                    lang = "es" if hl.startswith("es") else "en"

                    all_articles.append({
                        "date": art_date,
                        "title": entry.get("title", "").strip(),
                        "url": u,
                        "source": "google_news",
                        "category": entry.get("source", {}).get("title", "") if isinstance(entry.get("source"), dict) else "",
                        "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                        "sentiment_tone": 0,
                        "language": lang,
                        "country_focus": "CO",
                        "url_hash": h,
                    })
                    window_count += 1

            except requests.exceptions.Timeout:
                logger.debug(f"Google News timeout for '{query[:30]}'")
            except Exception as e:
                logger.debug(f"Google News error: {e}")

            time.sleep(1.5)  # Be polite to Google

        logger.info(f"Google News window {window_num}/{total_windows}: {current} -> {window_end} = {window_count} articles (total: {len(all_articles)})")

        save_checkpoint({"google_last_end": window_end.isoformat()})
        current = window_end + timedelta(days=1)
        time.sleep(1)

    return all_articles


# ============================================================================
# GDELT Historical Fetcher (via GDELTDocAdapter)
# ============================================================================

def scrape_gdelt_historical(
    start: date,
    end: date,
    window_days: int = 7,
    checkpoint: dict = None,
) -> list[dict]:
    """Scrape GDELT using the GDELTDocAdapter.

    Delegates to the adapter for proper rate limiting, OR query syntax,
    startdatetime/enddatetime, and error handling.

    NOTE: GDELT artlist mode retains only ~2-3 weeks of data.
    Historical backfill for older dates will return 0 articles.
    """
    from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter

    adapter = GDELTDocAdapter()

    # Handle checkpoint resume
    if checkpoint and checkpoint.get("gdelt_last_end"):
        resume_date = date.fromisoformat(checkpoint["gdelt_last_end"])
        if resume_date > start:
            start = resume_date + timedelta(days=1)
            logger.info(f"GDELT: Resuming from {start}")

    if start >= end:
        logger.info("GDELT: Nothing to scrape (start >= end)")
        return []

    start_dt = datetime(start.year, start.month, start.day)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)

    raw_articles = adapter.fetch_historical(start_dt, end_dt)

    # Convert RawArticle -> dict format matching CSV schema
    articles = []
    seen_hashes = set()
    for art in raw_articles:
        h = url_hash(art.url)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        art_date = ""
        if art.published_at:
            art_date = art.published_at.strftime("%Y-%m-%d")

        # Parse tone from raw_json or gdelt_tone
        tone = 0
        if art.gdelt_tone is not None:
            tone = art.gdelt_tone

        articles.append({
            "date": art_date,
            "title": art.title,
            "url": art.url,
            "source": "gdelt",
            "category": art.raw_json.get("domain", "") if art.raw_json else "",
            "summary": "",
            "sentiment_tone": tone,
            "language": art.language or "en",
            "country_focus": art.country_focus or "CO",
            "url_hash": h,
        })

    save_checkpoint({"gdelt_last_end": end.isoformat()})
    logger.info(f"GDELT: {len(articles)} articles")
    return articles


# ============================================================================
# NewsAPI (last 30 days)
# ============================================================================

def scrape_newsapi(api_key: str) -> list[dict]:
    """Fetch Colombia news from NewsAPI (last 30 days, free tier limit)."""
    if not api_key:
        logger.info("NewsAPI: Skipped (no API key)")
        return []

    base_url = "https://newsapi.org/v2/everything"
    queries = [
        "Colombia peso exchange rate",
        "USD COP forex",
        "Banco Republica interest rate",
        "Colombia economy",
        "Colombia inflation petrol",
    ]

    all_articles = []
    seen_urls = set()
    from_date = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")

    for query in queries:
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
            "language": "en",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"NewsAPI HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            for art in data.get("articles", []):
                u = art.get("url", "")
                h = url_hash(u)
                if h in seen_urls:
                    continue
                seen_urls.add(h)

                pub = art.get("publishedAt", "")[:10]

                all_articles.append({
                    "date": pub,
                    "title": art.get("title", ""),
                    "url": u,
                    "source": "newsapi",
                    "category": art.get("source", {}).get("name", ""),
                    "summary": art.get("description", "") or "",
                    "sentiment_tone": 0,
                    "language": "en",
                    "country_focus": "CO",
                    "url_hash": h,
                })

            time.sleep(1)
        except Exception as e:
            logger.warning(f"NewsAPI query '{query}' failed: {e}")

    logger.info(f"NewsAPI: {len(all_articles)} articles")
    return all_articles


# ============================================================================
# Portafolio RSS (recent only)
# ============================================================================

def scrape_portafolio_rss() -> list[dict]:
    """Fetch recent articles from Portafolio RSS."""
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed")
        return []

    from src.news_engine.ingestion.base_adapter import parse_date_flexible

    rss_url = "https://www.portafolio.co/rss/economia.xml"
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        feed = feedparser.parse(resp.content)
    except Exception as e:
        logger.warning(f"Portafolio RSS failed: {e}")
        return []

    articles = []
    for entry in feed.entries:
        published = parse_date_flexible(entry.get("published", ""))
        if not published:
            continue

        art_date = published.strftime("%Y-%m-%d") if hasattr(published, "strftime") else ""
        u = entry.get("link", "")

        articles.append({
            "date": art_date,
            "title": entry.get("title", ""),
            "url": u,
            "source": "portafolio",
            "category": "economia",
            "summary": entry.get("summary", ""),
            "sentiment_tone": 0,
            "language": "es",
            "country_focus": "CO",
            "url_hash": url_hash(u),
        })

    logger.info(f"Portafolio RSS: {len(articles)} articles")
    return articles


# ============================================================================
# Investing.com Search Scraper (cloudscraper + BeautifulSoup)
# ============================================================================

def scrape_investing_historical(
    start: date,
    end: date,
    checkpoint: dict = None,
) -> list[dict]:
    """Scrape Investing.com search pages for Colombia news in date windows.

    Delegates to InvestingSearchScraper.fetch_historical() and converts
    RawArticle objects to the flat dict format matching the CSV schema.
    """
    from src.news_engine.ingestion.investing_scraper import InvestingSearchScraper

    scraper = InvestingSearchScraper()

    # Handle checkpoint resume: if we have a saved window idx,
    # the scraper's own checkpoint will pick it up
    if checkpoint and checkpoint.get("investing_last_end"):
        resume_date = date.fromisoformat(checkpoint["investing_last_end"])
        if resume_date > start:
            start = resume_date + timedelta(days=1)
            logger.info(f"Investing.com: Resuming from {start}")

    start_dt = datetime(start.year, start.month, start.day)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)

    raw_articles = scraper.fetch_historical(start_dt, end_dt)

    # Convert RawArticle -> dict format matching CSV schema
    articles = []
    seen_hashes = set()
    for art in raw_articles:
        h = url_hash(art.url)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        art_date = ""
        if art.published_at:
            art_date = art.published_at.strftime("%Y-%m-%d")

        articles.append({
            "date": art_date,
            "title": art.title,
            "url": art.url,
            "source": "investing",
            "category": "",
            "summary": art.summary or "",
            "sentiment_tone": 0,
            "language": art.language or "en",
            "country_focus": "CO",
            "url_hash": h,
        })

    save_checkpoint({"investing_last_end": end.isoformat()})
    logger.info(f"Investing.com: {len(articles)} articles")
    return articles


# ============================================================================
# Checkpoint & Export
# ============================================================================

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(update: dict):
    cp = load_checkpoint()
    cp.update(update)
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))


def load_existing_articles() -> list[dict]:
    """Load existing articles from CSV for deduplication."""
    if not CSV_FILE.exists():
        return []
    articles = []
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append(row)
    return articles


def export_csv(articles: list[dict]):
    """Export articles to CSV."""
    articles.sort(key=lambda a: a.get("date", ""), reverse=True)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(articles)

    logger.info(f"Exported {len(articles)} articles to {CSV_FILE}")


def export_json(articles: list[dict]):
    """Export articles to JSON with metadata."""
    articles.sort(key=lambda a: a.get("date", ""), reverse=True)

    dates = [a["date"] for a in articles if a.get("date")]
    sources = {}
    for a in articles:
        src = a.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    # Count articles per year-month
    monthly = {}
    for a in articles:
        ym = a.get("date", "")[:7]
        if ym:
            monthly[ym] = monthly.get(ym, 0) + 1

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_articles": len(articles),
            "date_range": {
                "start": min(dates) if dates else None,
                "end": max(dates) if dates else None,
            },
            "sources": sources,
            "monthly_counts": dict(sorted(monthly.items())),
        },
        "articles": articles,
    }

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(articles)} articles to {JSON_FILE}")


def deduplicate(articles: list[dict]) -> list[dict]:
    """Deduplicate articles by url_hash."""
    seen = set()
    unique = []
    for a in articles:
        h = a.get("url_hash", url_hash(a.get("url", "")))
        if h not in seen:
            seen.add(h)
            unique.append(a)
    return unique


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Historical Colombia News Scraper")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--source",
        choices=["google", "gdelt", "newsapi", "portafolio", "investing", "all"],
        default="all",
        help="Source to scrape",
    )
    parser.add_argument("--newsapi-key", default=os.environ.get("USDCOP_NEWSAPI_KEY", ""),
                        help="NewsAPI key")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Colombia News Historical Scraper")
    logger.info(f"Date range: {start} -> {end}")
    logger.info(f"Sources: {args.source}")

    checkpoint = load_checkpoint() if args.resume else {}
    all_articles = load_existing_articles() if args.resume else []

    # --- Google News RSS (PRIMARY) ---
    if args.source in ("google", "all"):
        logger.info("=" * 60)
        logger.info("Google News RSS (primary)")
        logger.info("=" * 60)
        google_articles = scrape_google_news_historical(start, end, window_days=30, checkpoint=checkpoint)
        all_articles.extend(google_articles)

    # --- GDELT ---
    if args.source in ("gdelt", "all"):
        logger.info("=" * 60)
        logger.info("GDELT Doc API (supplementary)")
        logger.info("=" * 60)
        gdelt_articles = scrape_gdelt_historical(start, end, window_days=60, checkpoint=checkpoint)
        all_articles.extend(gdelt_articles)

    # --- NewsAPI ---
    if args.source in ("newsapi", "all"):
        logger.info("=" * 60)
        logger.info("NewsAPI (last 30 days)")
        logger.info("=" * 60)
        newsapi_key = args.newsapi_key
        newsapi_articles = scrape_newsapi(newsapi_key)
        all_articles.extend(newsapi_articles)

    # --- Portafolio ---
    if args.source in ("portafolio", "all"):
        logger.info("=" * 60)
        logger.info("Portafolio RSS")
        logger.info("=" * 60)
        port_articles = scrape_portafolio_rss()
        all_articles.extend(port_articles)

    # --- Investing.com ---
    if args.source in ("investing", "all"):
        logger.info("=" * 60)
        logger.info("Investing.com Search Scraper")
        logger.info("=" * 60)
        investing_articles = scrape_investing_historical(start, end, checkpoint=checkpoint)
        all_articles.extend(investing_articles)

    # Deduplicate
    unique_articles = deduplicate(all_articles)
    logger.info(f"Total after dedup: {len(unique_articles)} (was {len(all_articles)})")

    # Export
    export_csv(unique_articles)
    export_json(unique_articles)

    # Summary
    sources = {}
    for a in unique_articles:
        src = a.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    dates = [a["date"] for a in unique_articles if a.get("date")]

    logger.info("")
    logger.info("=" * 60)
    logger.info("SCRAPE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total articles: {len(unique_articles)}")
    logger.info(f"Date range: {min(dates) if dates else 'N/A'} -> {max(dates) if dates else 'N/A'}")
    for src, count in sorted(sources.items()):
        logger.info(f"  {src:15s}: {count:5d}")
    logger.info(f"Output: {CSV_FILE}")
    logger.info(f"Output: {JSON_FILE}")


if __name__ == "__main__":
    main()
