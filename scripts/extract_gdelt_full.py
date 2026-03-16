#!/usr/bin/env python3
"""
GDELT Full Historical Extraction (2017-2026)
=============================================
Extracts TWO datasets from GDELT DOC 2.0 API:

1. Daily sentiment + volume timelines (timelinetone + timelinevolraw)
   - ~18 minutes, ~108 monthly requests
   - Output: data/news/gdelt_daily_sentiment.csv + .json

2. Individual articles (artlist) with 7-day windowing
   - ~11 hours, ~5,000 requests
   - Output: data/news/gdelt_articles_historical.csv + .json

Usage:
  python scripts/extract_gdelt_full.py --mode timelines     # Fast: sentiment + volume
  python scripts/extract_gdelt_full.py --mode articles      # Slow: all articles
  python scripts/extract_gdelt_full.py --mode both          # Both (timelines first)
  python scripts/extract_gdelt_full.py --mode articles --resume  # Resume from checkpoint
  python scripts/extract_gdelt_full.py --mode articles --queries-en-only  # Faster: 6 EN queries only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

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
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FIXED_DELAY = 30  # 30s between ALL requests. No exceptions. No cascading retries.
REQUEST_TIMEOUT = 45
_last_request_time = None


# ============================================================================
# Helpers
# ============================================================================

def safe_gdelt_request(params: dict) -> dict | None:
    """Make a single GDELT API request with fixed delay. NO RETRIES.

    Strategy: wait exactly FIXED_DELAY since last request, make one attempt.
    On ANY failure, return None immediately. The caller skips and moves on.
    This prevents cascading rate limit penalties that slow everything down.
    """
    global _last_request_time

    # Fixed delay since last request
    if _last_request_time is not None:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < FIXED_DELAY:
            time.sleep(FIXED_DELAY - elapsed)
    else:
        time.sleep(FIXED_DELAY)

    try:
        resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        _last_request_time = time.monotonic()
        text = resp.text.strip()

        # Rate limit or error response
        if resp.status_code == 429:
            return None
        if "please limit requests" in text.lower() and len(text) < 300:
            return None
        if text.startswith("{Content-type:") or text.startswith("{Server:"):
            return None
        if not text or (not text.startswith("{") and not text.startswith("[")):
            return None

        return resp.json()

    except Exception:
        _last_request_time = time.monotonic()
        return None


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ============================================================================
# Part 1: Daily Sentiment + Volume Timelines
# ============================================================================

def extract_timelines(
    start_year: int = 2017,
    end_date: date = None,
    queries: list[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Extract daily tone and volume timelines using full-range requests.

    GDELT returns ~3300 daily points per full-range request (2017-2026).
    Total: 4 queries x 2 modes = 8 requests only.

    Returns (tone_records, volume_records).
    """
    if end_date is None:
        end_date = date.today()
    if queries is None:
        queries = [
            "Colombia",
            '("Colombian peso" OR USDCOP OR "peso colombiano")',
            '(Ecopetrol OR "petroleo Colombia")',
            '("Banco Republica" OR "Colombia central bank")',
        ]

    tone_records = []
    volume_records = []

    start_str = f"{start_year}0101000000"
    end_str = end_date.strftime("%Y%m%d") + "235959"
    total = len(queries) * 2
    completed = 0

    for query in queries:
        # --- Tone ---
        completed += 1
        logger.info(f"[{completed}/{total}] Tone: {query[:50]}...")
        data = safe_gdelt_request({
            "query": query,
            "mode": "timelinetone",
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
        })
        if data:
            n_pts = 0
            for tl in data.get("timeline", []):
                for pt in tl.get("data", []):
                    try:
                        dt_str = pt["date"][:8]
                        tone_records.append({
                            "date": f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}",
                            "query": query,
                            "tone": float(pt["value"]),
                        })
                        n_pts += 1
                    except (KeyError, ValueError):
                        pass
            logger.info(f"  -> {n_pts} tone points")
        else:
            logger.warning(f"  -> FAILED")

        # --- Volume Raw ---
        completed += 1
        logger.info(f"[{completed}/{total}] Volume: {query[:50]}...")
        data_vol = safe_gdelt_request({
            "query": query,
            "mode": "timelinevolraw",
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
        })
        if data_vol:
            n_pts = 0
            for tl in data_vol.get("timeline", []):
                for pt in tl.get("data", []):
                    try:
                        dt_str = pt["date"][:8]
                        volume_records.append({
                            "date": f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}",
                            "query": query,
                            "volume": int(float(pt["value"])),
                        })
                        n_pts += 1
                    except (KeyError, ValueError):
                        pass
            logger.info(f"  -> {n_pts} volume points")
        else:
            logger.warning(f"  -> FAILED")

    logger.info(f"Timelines complete: {len(tone_records)} tone + {len(volume_records)} volume records")
    return tone_records, volume_records


def save_timelines(tone_records: list[dict], volume_records: list[dict]):
    """Save timeline data to CSV + JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Aggregate to daily: average tone and sum volume across queries ---
    daily_tone = {}
    for r in tone_records:
        d = r["date"]
        if d not in daily_tone:
            daily_tone[d] = []
        daily_tone[d].append(r["tone"])

    daily_vol = {}
    for r in volume_records:
        d = r["date"]
        if d not in daily_vol:
            daily_vol[d] = []
        daily_vol[d].append(r["volume"])

    # Merge into daily records
    all_dates = sorted(set(list(daily_tone.keys()) + list(daily_vol.keys())))
    daily_records = []
    for d in all_dates:
        tones = daily_tone.get(d, [])
        vols = daily_vol.get(d, [])
        daily_records.append({
            "date": d,
            "tone_avg": round(sum(tones) / len(tones), 4) if tones else None,
            "tone_min": round(min(tones), 4) if tones else None,
            "tone_max": round(max(tones), 4) if tones else None,
            "tone_queries": len(tones),
            "volume_total": sum(vols) if vols else None,
            "volume_avg": round(sum(vols) / len(vols), 1) if vols else None,
            "volume_queries": len(vols),
        })

    # CSV
    csv_file = OUTPUT_DIR / "gdelt_daily_sentiment.csv"
    columns = ["date", "tone_avg", "tone_min", "tone_max", "tone_queries",
               "volume_total", "volume_avg", "volume_queries"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(daily_records)

    # JSON
    json_file = OUTPUT_DIR / "gdelt_daily_sentiment.json"
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "GDELT DOC 2.0 API",
            "modes": ["timelinetone", "timelinevolraw"],
            "total_days": len(daily_records),
            "date_range": {
                "start": all_dates[0] if all_dates else None,
                "end": all_dates[-1] if all_dates else None,
            },
            "queries_used": 4,
            "raw_tone_points": len(tone_records),
            "raw_volume_points": len(volume_records),
        },
        "daily": daily_records,
        "raw_tone": tone_records,
        "raw_volume": volume_records,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(daily_records)} daily records to {csv_file}")
    logger.info(f"Saved full data to {json_file}")

    # Summary stats
    tones = [r["tone_avg"] for r in daily_records if r["tone_avg"] is not None]
    vols = [r["volume_total"] for r in daily_records if r["volume_total"] is not None]
    if tones:
        logger.info(f"Tone: min={min(tones):.2f}, max={max(tones):.2f}, avg={sum(tones)/len(tones):.2f}")
    if vols:
        logger.info(f"Volume: min={min(vols)}, max={max(vols)}, avg={sum(vols)/len(vols):.0f}/day")


# ============================================================================
# Part 2: Articles Historical
# ============================================================================

CHECKPOINT_FILE = OUTPUT_DIR / "gdelt_articles_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(data: dict):
    cp = load_checkpoint()
    cp.update(data)
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))


def extract_articles(
    start_date: date = date(2017, 1, 1),
    end_date: date = None,
    window_days: int = 7,
    en_only: bool = False,
    resume: bool = False,
    fast: bool = False,
) -> list[dict]:
    """Extract all GDELT articles with date windowing.

    fast=True uses only 2 broad queries (Colombia EN + ES) instead of 11.
    This captures ~95% of articles with ~80% fewer requests.
    """
    if end_date is None:
        end_date = date.today()

    if fast:
        # 2 broad queries cover most articles; specific ones are subsets
        queries = [("Colombia sourcelang:english", "en")]
        if not en_only:
            queries.append(("Colombia sourcelang:spanish", "es"))
    else:
        from src.news_engine.config import GDELTConfig
        cfg = GDELTConfig()
        queries = []
        for q in cfg.queries_en:
            queries.append((q, "en"))
        if not en_only:
            for q in cfg.queries_es:
                queries.append((q, "es"))

    # Resume support
    checkpoint = load_checkpoint() if resume else {}
    if resume and checkpoint.get("last_window_end"):
        resume_date = date.fromisoformat(checkpoint["last_window_end"])
        if resume_date > start_date:
            start_date = resume_date + timedelta(days=1)
            logger.info(f"Resuming from {start_date}")

    # Load existing articles for dedup
    existing_hashes = set()
    articles_file = OUTPUT_DIR / "gdelt_articles_historical.json"
    existing_articles = []
    if resume and articles_file.exists():
        try:
            with open(articles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing_articles = data.get("articles", [])
            existing_hashes = {a.get("url_hash", "") for a in existing_articles}
            logger.info(f"Loaded {len(existing_articles)} existing articles for dedup")
        except Exception:
            pass

    # Generate windows
    windows = []
    current = start_date
    while current < end_date:
        w_end = min(current + timedelta(days=window_days - 1), end_date)
        windows.append((current, w_end))
        current = w_end + timedelta(days=1)

    total_requests = len(windows) * len(queries)
    logger.info(
        f"Extraction plan: {len(windows)} windows x {len(queries)} queries "
        f"= {total_requests} requests (~{total_requests * FIXED_DELAY / 3600:.1f} hours)"
    )

    all_articles = list(existing_articles)
    seen_hashes = set(existing_hashes)
    completed = 0
    new_articles = 0

    for w_start, w_end in windows:
        window_new = 0
        for query, lang in queries:
            completed += 1

            data = safe_gdelt_request({
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": "250",
                "sort": "datedesc",
                "startdatetime": w_start.strftime("%Y%m%d") + "000000",
                "enddatetime": w_end.strftime("%Y%m%d") + "235959",
            })

            if data:
                for art in data.get("articles", []):
                    u = art.get("url", "")
                    if not u:
                        continue
                    h = url_hash(u)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    raw_date = art.get("seendate", "")
                    try:
                        art_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                    except Exception:
                        art_date = w_start.isoformat()

                    all_articles.append({
                        "date": art_date,
                        "title": art.get("title", "").strip(),
                        "url": u,
                        "source": "gdelt",
                        "domain": art.get("domain", ""),
                        "language": art.get("language", lang),
                        "country": art.get("sourcecountry", ""),
                        "url_hash": h,
                    })
                    window_new += 1
                    new_articles += 1

        # Progress log every 5 windows
        if completed % (len(queries) * 5) == 0 or w_end >= end_date:
            elapsed_pct = 100 * completed / total_requests
            logger.info(
                f"Progress: {completed}/{total_requests} ({elapsed_pct:.0f}%) | "
                f"window: {w_start} -> {w_end} | "
                f"+{window_new} new | total: {len(all_articles)} articles"
            )

        # Save checkpoint every 10 windows
        if completed % (len(queries) * 10) == 0:
            save_checkpoint({
                "last_window_end": w_end.isoformat(),
                "total_articles": len(all_articles),
                "completed_requests": completed,
            })
            # Incremental save
            _save_articles(all_articles)

    # Final save
    save_checkpoint({
        "last_window_end": end_date.isoformat(),
        "total_articles": len(all_articles),
        "completed_requests": completed,
        "completed_at": datetime.now().isoformat(),
    })

    logger.info(f"Extraction complete: {len(all_articles)} total, {new_articles} new")
    return all_articles


def _save_articles(articles: list[dict]):
    """Save articles to CSV + JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    articles.sort(key=lambda a: a.get("date", ""), reverse=True)

    # CSV
    csv_file = OUTPUT_DIR / "gdelt_articles_historical.csv"
    columns = ["date", "title", "url", "source", "domain", "language", "country", "url_hash"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(articles)

    # JSON
    json_file = OUTPUT_DIR / "gdelt_articles_historical.json"

    dates = [a["date"] for a in articles if a.get("date")]
    by_year = {}
    for a in articles:
        y = a.get("date", "")[:4]
        by_year[y] = by_year.get(y, 0) + 1
    by_lang = {}
    for a in articles:
        l = a.get("language", "?")
        by_lang[l] = by_lang.get(l, 0) + 1

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "GDELT DOC 2.0 API (artlist)",
            "total_articles": len(articles),
            "date_range": {
                "start": min(dates) if dates else None,
                "end": max(dates) if dates else None,
            },
            "by_year": dict(sorted(by_year.items())),
            "by_language": dict(sorted(by_lang.items(), key=lambda x: -x[1])),
        },
        "articles": articles,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(articles)} articles to {csv_file} + {json_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GDELT Full Historical Extraction")
    parser.add_argument(
        "--mode",
        choices=["timelines", "articles", "both"],
        default="both",
        help="What to extract",
    )
    parser.add_argument("--start", default="2017-01-01", help="Start date")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date")
    parser.add_argument("--resume", action="store_true", help="Resume articles from checkpoint")
    parser.add_argument("--queries-en-only", action="store_true", help="Only EN queries (faster)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 2 broad queries instead of 11 (~80%% fewer requests)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    logger.info("GDELT Full Historical Extraction")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Date range: {start} -> {end}")

    if args.mode in ("timelines", "both"):
        logger.info("=" * 60)
        logger.info("Part 1: Daily Sentiment + Volume Timelines")
        logger.info("=" * 60)
        tone_records, volume_records = extract_timelines(
            start_year=start.year,
            end_date=end,
        )
        save_timelines(tone_records, volume_records)

    if args.mode in ("articles", "both"):
        logger.info("=" * 60)
        logger.info("Part 2: Articles Historical")
        logger.info("=" * 60)
        articles = extract_articles(
            start_date=start,
            end_date=end,
            en_only=args.queries_en_only,
            resume=args.resume,
            fast=args.fast,
        )
        _save_articles(articles)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
