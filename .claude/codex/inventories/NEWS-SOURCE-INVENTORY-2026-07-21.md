---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# News acquisition inventory — 2026-07-21

## Implemented source paths

### NewsEngine adapters (`src/news_engine/ingestion`)

- `investing_scraper.py` — Investing.com scraper (cloudscraper/HTML or RSS path).
- `portafolio_scraper.py` — Portafolio scraper/RSS.
- `larepublica_scraper.py` — La República scraper/RSS.
- `gdelt_adapter.py` — GDELT DOC API, including tone metadata.
- `newsapi_adapter.py` — NewsAPI.org, disabled when `USDCOP_NEWSAPI_KEY` is absent.
- `registry.py` — adapter registry and configuration-based enablement.

### Analysis adapters (`src/analysis/news_sources.py`)

- Google News RSS is primary and aggregates publisher feeds (including Investing,
  CNBC, CoinDesk and Reuters when available).
- GDELT DOC is fallback with pacing, timeout and retry/backoff.

### Pipelines and persistence

- Airflow DAGs: `news_daily_pipeline.py`, `news_alert_monitor.py`,
  `news_weekly_digest.py`, `news_maintenance.py`.
- Storage: `news_sources`, `news_articles`, `news_ingestion_log`, cross-reference,
  digest and feature-snapshot tables from migration 045.
- Historical files: `data/news/colombia_news_historical.{csv,json}` and
  `data/news/scrape_checkpoint.json`.

## Health/evidence status

- Unit contract tests: 89/89 passed for source adapters and news analysis.
- Existing historical inventory reports Google News, GDELT, Investing.com and
  Portafolio records.
- Current local news backups are readable, but freshness and provider-run manifests
  are not equivalent to a live health check.
- Latest GDELT live probe timed out; it is not marked healthy.
- NewsAPI requires a configured secret and was not executed without one.
- Google News, Investing and publisher scrapers are implemented, but each needs a
  successful run record containing request window, response status, row count,
  latency, checksum and checkpoint before promotion.

## Required operational proof

For every enabled source, persist a manifest under the acquisition evidence path and
record `source_id`, `retrieved_at`, `published_at` range, HTTP status, article count,
deduplication count, checksum, latency and error/backoff details. Keep source data
as untrusted content, enforce URL safety and require `published_at <= bar_timestamp`
before any feature use.
