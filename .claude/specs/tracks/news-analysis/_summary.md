# Rule: News Engine & Analysis Module

> Governs the News Engine (ingestion + enrichment) and Analysis Module (LLM-generated narratives).
> These are two coupled pipelines sharing one PostgreSQL database.
> Created: 2026-02-27

---

## Architecture Overview

```
Pipeline A: NewsEngine (Ingestion → Enrichment → Storage → Features)
    |
    ├── 5 Source Adapters (GDELT, NewsAPI, Investing.com, La Republica, Portafolio)
    ├── 5-Stage Enrichment (categorize, relevance, sentiment, NER, breaking detection)
    ├── Cross-Reference Engine (topic clustering across sources)
    ├── Storage Layer (PostgreSQL: 8 tables)
    └── Feature Exporter (~60 daily features for ML integration)

Pipeline B: Analysis Module (Macro Analysis → LLM Narratives → Dashboard Export)
    |
    ├── MacroAnalyzer (SMA, Bollinger, RSI, MACD, ROC, z-score for 13 variables)
    ├── LLMClient (Azure OpenAI primary + Anthropic Claude fallback)
    ├── PromptTemplates (Spanish, contextual, with macro + signals + news)
    ├── ChartGenerator (matplotlib dark-theme PNGs)
    └── WeeklyAnalysisGenerator (orchestrator: daily + weekly + JSON export)

Dashboard: /analysis (14 React components, 5 API routes, file-based JSON)
    └── Dynamic asset selector: USD/COP + Gold (xauusd) + Bitcoin (btcusdt) — see "Multi-Asset Analysis"
```

---

## News Engine: Source Adapters

5 adapters implemented (no Google News adapter — adapter file does not exist):

| Adapter | Source | File | Type | Rate Limit |
|---------|--------|------|------|------------|
| `GDELTDocAdapter` + `GDELTContextAdapter` | GDELT DOC/Context 2.0 | `gdelt_adapter.py` | News + Sentiment API | 6s min delay |
| `NewsAPIAdapter` | NewsAPI.org | `newsapi_adapter.py` | News API | 100 req/day (API key gated) |
| `InvestingSearchScraper` | Investing.com | `investing_scraper.py` | Web Search (cloudscraper) | 3-6s + session rotation |
| `LaRepublicaScraper` | La Republica | `larepublica_scraper.py` | RSS + HTML (feedparser) | Polite crawl |
| `PortafolioScraper` | Portafolio.co | `portafolio_scraper.py` | RSS + HTML (feedparser) | Polite crawl |

**Base class**: `SourceAdapter` (ABC) in `src/news_engine/ingestion/base_adapter.py`
- Required: `fetch_latest(hours_back)`, `fetch_historical(start, end)`, `health_check()`
- Built-in: retry logic, deduplication by URL hash, flexible date parsing

**Registry**: `SourceRegistry` in `src/news_engine/ingestion/registry.py`
- `from_config()` loads all adapters, auto-disables NewsAPI if no key
- `google_news_adapter.py` referenced in CLAUDE.md but never implemented — removed from docs

---

## News Engine: Enrichment Pipeline

**File**: `src/news_engine/enrichment/pipeline.py` → `EnrichmentPipeline`

| Stage | Module | Output |
|-------|--------|--------|
| 1. Categorize | `categorizer.py` | 9 categories: monetary_policy, fx_market, commodities, inflation, fiscal_policy, risk_premium, capital_flows, balance_payments, political |
| 2. Relevance | `relevance.py` | Score 0.0-1.0 (keyword 60% + source quality 20% + recency 20%) |
| 3. Sentiment | `sentiment.py` | Score -1.0 to +1.0 (GDELT tone primary, VADER fallback) |
| 4. NER/Tags | `tagger.py` | Keywords (up to 10) + entities (BanRep, Fed, DANE, etc.) |
| 5. Flags | `pipeline.py` | is_breaking (relevance≥0.8 + extreme sentiment), is_weekly_relevant |

**Cross-Reference**: `src/news_engine/cross_reference/engine.py`
- Token-based similarity (Jaccard), no ML dependencies
- Clusters related articles across sources within 48h window
- Config: similarity_threshold=0.6, min_cluster=2, max_cluster=20

---

## News Engine: Data Models

**File**: `src/news_engine/models.py`

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `RawArticle` | Normalized from any source | url, title, source_id, published_at, gdelt_tone, language |
| `EnrichedArticle` | After enrichment | raw + category, relevance_score, sentiment_score, keywords[], entities[], is_breaking |

**Contract**: `src/contracts/news_engine_schema.py` (CTR-NEWS-SCHEMA-001)

---

## News Engine: Feature Export

**File**: `src/news_engine/output/feature_exporter.py`

~60 daily features across 7 groups:

| Group | Examples | Count |
|-------|----------|-------|
| Volume/category | vol_monetary_policy, vol_fx_market, ... | 9 |
| Keyword mentions | kw_dolar, kw_tasa_de_cambio, kw_embi, ... | 14 |
| Sentiment/category | sent_monetary_policy, sent_fx_market, ... | 9 |
| Overall sentiment | sent_avg, sent_positive_pct, sent_negative_pct | 4 |
| Per-source volume | vol_gdelt_doc, vol_investing, vol_portafolio, ... | 5 |
| Cross-reference | num_clusters, avg_cluster_size, consensus_sentiment | 4 |
| Relevance flags | high_relevance_count, breaking_news_count, avg_relevance | 4 |

---

## News Engine: Database Tables

**Migration**: `database/migrations/045_newsengine_initial.sql`

| Table | Purpose | PK |
|-------|---------|------|
| `news_sources` | Source registry (seeded with 6 sources) | source_id |
| `news_articles` | Enriched articles | auto id, UNIQUE(source_id, url_hash) |
| `news_keywords` | Keyword tracking with priorities | auto id, UNIQUE(keyword) |
| `news_cross_references` | Topic clusters | auto id |
| `news_cross_reference_articles` | M:N join (cluster ↔ article) | (cross_reference_id, article_id) |
| `news_daily_digests` | Daily/weekly summaries | auto id, UNIQUE(digest_date) |
| `news_ingestion_log` | Fetch audit trail | auto id |
| `news_feature_snapshots` | Daily feature vectors | auto id, UNIQUE(snapshot_date) |

---

## News Engine: Historical Data (Extracted 2026-02-27)

| Source | Articles | Date Range | File |
|--------|----------|------------|------|
| GDELT (articles) | 128,141 | 2017-01-07 to 2026-02-27 | `data/news/gdelt_articles_historical.csv` |
| GDELT (sentiment) | 3,323 daily | 2017-01-01 to 2026-02-26 | `data/news/gdelt_daily_sentiment.csv` |
| Google News | 13,220 | Various | `data/news/google_news_*.csv` |
| Investing.com | 1,718 | Various | `data/news/investing_articles_historical.csv` |

**Full extraction script**: `scripts/data/extract_gdelt_full.py`
- Modes: `--mode timelines|articles|both`
- Strategies: `--fast` (2 broad queries), default (11 specific queries)
- Checkpoint/resume: `data/news/gdelt_articles_checkpoint.json`

---

## Analysis Module: Components

### Python Backend (6 files in `src/analysis/`)

| File | Class | Purpose |
|------|-------|---------|
| `macro_analyzer.py` | `MacroAnalyzer` | SMA-5/10/20/50, Bollinger, RSI (Wilder's), MACD, ROC, z-score for 13 variables |
| `llm_client.py` | `LLMClient` | Azure OpenAI primary + Anthropic fallback, file-based caching, token/cost tracking |
| `prompt_templates.py` | (constants) | `SYSTEM_DAILY`, `SYSTEM_WEEKLY`, `DAILY_TEMPLATE`, `WEEKLY_TEMPLATE` + builders |
| `chart_generator.py` | `generate_all_charts()` | Matplotlib dark-theme PNGs for macro overlays |
| `weekly_generator.py` | `WeeklyAnalysisGenerator` | Orchestrator: daily + weekly analysis + JSON export |
| `__init__.py` | Package | Exports |

### Key Macro Variables (13)

| Variable | Display Name | Impact on USDCOP |
|----------|-------------|------------------|
| dxy | Dollar Index | Positive (stronger USD = COP weakens) |
| vix | VIX Volatility | Positive (risk-off = COP weakens) |
| wti | Crudo WTI | Negative (higher oil = COP strengthens) |
| embi_col | EMBI Colombia | Positive (higher spread = COP weakens) |
| ust10y | Treasury 10Y | Mixed |
| ust2y | Treasury 2Y | Mixed |
| ibr | IBR Colombia | Mixed |
| tpm | TPM BanRep | Mixed |
| fedfunds | Fed Funds Rate | Positive |
| gold | Oro | Negative |
| brent | Brent Crude | Negative |
| cpi_us | CPI USA | Indirect |
| cpi_col | CPI Colombia | Indirect |

### LLM Configuration

**SSOT**: `config/analysis/weekly_analysis_ssot.yaml` (CTR-ANALYSIS-CONFIG-001)

| Setting | Value |
|---------|-------|
| Primary provider | Azure OpenAI (GPT-4o) |
| Fallback provider | Anthropic (Claude Sonnet) |
| Cache | File-based, TTL 24h, in `data/cache/analysis/` |
| Budget | $15/month, $1/day |
| Language | Spanish (es) |
| Daily max tokens | 1,500 |
| Weekly max tokens | 2,000 |

**API Keys** (env vars; all in `.env.example`, consumed by `src/analysis/llm_client.py` + `src/news_engine/config.py`):
- `USDCOP_AZURE_OPENAI_API_KEY` (primary LLM)
- `USDCOP_AZURE_OPENAI_ENDPOINT`
- `USDCOP_ANTHROPIC_API_KEY` (fallback LLM)
- `USDCOP_NEWSAPI_KEY` (optional — NewsAPI adapter; disabled when unset)

---

## Analysis Module: Database Tables

**Migration**: `database/migrations/046_weekly_analysis_tables.sql`

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `weekly_analysis` | Weekly AI reports | iso_year, iso_week, summary_markdown, sentiment, themes, h5_signal, h1_signals |
| `daily_analysis` | Daily AI entries | analysis_date, headline, summary_markdown, sentiment, macro_publications, news_highlights |
| `macro_variable_snapshots` | SMA/BB/RSI/MACD per var per day | variable_key, snapshot_date, sma_5/10/20/50, z_score, trend |
| `analysis_chat_history` | Chat messages | session_id, role, content, context_year, context_week, tokens_used |

---

## Analysis Module: CLI

```bash
# Generate current week
python scripts/pipeline/generate_weekly_analysis.py

# Generate specific week
python scripts/pipeline/generate_weekly_analysis.py --week 2026-W09

# Generate daily only
python scripts/pipeline/generate_weekly_analysis.py --date 2026-02-25

# Backfill range
python scripts/pipeline/generate_weekly_analysis.py --from 2026-W06 --to 2026-W09

# Dry run (no LLM calls)
python scripts/pipeline/generate_weekly_analysis.py --dry-run

# Export only (cached LLM results)
python scripts/pipeline/generate_weekly_analysis.py --export-only
```

---

## Analysis Module: Contracts

### Python (`src/contracts/analysis_schema.py`, CTR-ANALYSIS-SCHEMA-001)

| Class | Purpose |
|-------|---------|
| `MacroSnapshot` | Technical indicators for 1 variable on 1 date |
| `DailyAnalysisRecord` | Daily headline, sentiment, OHLCV, signals, cost |
| `WeeklyAnalysisRecord` | Weekly summary, themes, signals aggregation |
| `ChatMessage` | Chat history entry |
| `WeeklyViewExport` | Complete JSON export for dashboard |
| `AnalysisIndexEntry` | Entry in analysis_index.json |

### TypeScript (`lib/contracts/weekly-analysis.contract.ts`, CTR-ANALYSIS-FRONTEND-001)

| Interface | Purpose |
|-----------|---------|
| `WeeklyViewData` | Full week export (weekly + daily + macro + signals + charts) |
| `WeeklySummary` | Headline, markdown, sentiment, themes, OHLCV |
| `DailyAnalysisEntry` | Daily entry with macro + events + signals |
| `MacroVariableSnapshot` | Full technical snapshot + chart data |
| `SignalSummaries` | H5 + H1 signal objects |
| `ChatSession` | Messages + context |
| `AnalysisIndex` | Available weeks list |

---

## Dashboard: /analysis Page

### Components (14 in `components/analysis/`)

| Component | Purpose |
|-----------|---------|
| `AnalysisPage` | Main container: week selector + all sub-components |
| `WeekSelector` | `[< W08 | SEMANA 9 | W10 >]` navigation |
| `WeeklySummaryHeader` | Sentiment badge + themes + executive summary markdown |
| `MacroSnapshotBar` | 4 key vars (DXY, VIX, Oil, EMBI) with SMA-20 + trend |
| `MacroChartGrid` | Grid of macro variable charts |
| `MacroVariableChart` | Individual Recharts line chart with SMA/BB overlays |
| `MacroDetailModal` | Full-screen macro chart with all indicators |
| `MacroEventChip` | Colored chip for macro events |
| `SignalSummaryCards` | H5 signal card + H1 daily summary card |
| `DailyTimeline` | Vertical timeline with 5 daily entries |
| `DailyTimelineEntry` | Day card with headline, sentiment, analysis, events |
| `UpcomingEventsPanel` | Next week's economic calendar |
| `AnalysisMarkdown` | react-markdown + remark-gfm renderer |
| `FloatingChatWidget` | Bottom-right floating AI chat popup |

### API Routes (5 in `app/api/analysis/`)

| Route | Method | Source | Asset param |
|-------|--------|--------|-------------|
| `/api/analysis/weeks` | GET | `analysis_index.json` | `?asset=` |
| `/api/analysis/week/[year]/[week]` | GET | `weekly_YYYY_WXX.json` | `?asset=` |
| `/api/analysis/calendar` | GET | `upcoming_events.json` | `?asset=` |
| `/api/analysis/chat` | POST | LLM via backend | `body.asset` |
| `/api/analysis/assets` | GET | SSOT `analysis-assets.ts` (drives selector) | — |

> All asset params are backward compatible: no param ⇒ `usdcop`. See "Multi-Asset Analysis".

### React Hooks (`hooks/useWeeklyAnalysis.ts`)

| Hook | Purpose | Stale Time |
|------|---------|------------|
| `useAnalysisIndex()` | Available weeks list | 5 min |
| `useWeeklyView(year, week)` | Full weekly data | 10 min |
| `useUpcomingEvents()` | Economic calendar | 30 min |

### Data Flow

```
Python generate_weekly_analysis.py
    → public/data/analysis/analysis_index.json
    → public/data/analysis/weekly_YYYY_WXX.json
    → public/data/analysis/charts/*.png

Dashboard /analysis reads JSON files via API routes (no DB hit)
```

---

## Multi-Asset Analysis (Added 2026-07-05)

The `/analysis` page now has a **dynamic asset selector** filtering USD/COP, Gold (`xauusd`),
and Bitcoin (`btcusdt`). Each asset shows per-week weekly + daily analysis exactly like USD/COP.

### SSOT & Scalability

- **SSOT**: `usdcop-trading-dashboard/lib/contracts/analysis-assets.ts` (frontend + `/api/analysis/assets`)
  mirrored by `config/analysis/analysis_assets.yaml` (Python generation profiles).
- `asset_id ∈ {usdcop, xauusd, btcusdt}` — matches `registry.json` + the on-disk data namespace.
- Adding an analysed asset (index/future) = **ONE entry each side** (scalable / additive).

### Data Namespacing

```
public/data/analysis/<asset>/weekly_YYYY_WXX.json
public/data/analysis/<asset>/analysis_index.json
public/data/analysis/<asset>/upcoming_events.json
```

- USD/COP stays at the **legacy root** (`public/data/analysis/*.json`) — unchanged.
- Resolver: `usdcop-trading-dashboard/lib/analysis-paths.ts` (`readAnalysisJson`) falls back to root
  for the default asset. **Path-safe**: unknown / traversal asset ids collapse to `usdcop`.
- Per-asset files live on the host bind-mount (survive `docker compose down -v`) and are CLI-regenerable.

### API Routes (backward compatible)

- Existing routes (`weeks`, `week/[year]/[week]`, `calendar`) take `?asset=`; `chat` takes `body.asset`.
- **NEW** `/api/analysis/assets` returns the SSOT asset list and drives the selector.
- No param ⇒ `usdcop` (fully backward compatible).

### Frontend

- `components/analysis/AssetSelector.tsx` + asset-scoped hooks in `hooks/useWeeklyAnalysis.ts`.
- Chat-context asset in `stores/useAnalysisChatStore.ts`; `AnalysisPage.tsx` resets the week on asset switch.

### Gold / BTC Generation (REAL data only — no synthetic)

- `src/analysis/asset_analysis_generator.py` + CLI `scripts/pipeline/generate_asset_analysis.py`.
- Inputs: real daily OHLCV seeds (`seeds/latest/{xauusd,btcusdt}_daily_ohlcv.parquet`) →
  technicals (RSI-Wilder / SMA / MACD / ATR / support-resistance) + real strategy positioning from the
  published `gold_trend_b2` / `btc_trend_b2` backtest trade bundles + per-day `daily_entries` + real news.
- **USD/COP is UNTOUCHED** — it keeps its richer macro LangGraph pipeline (`generate_weekly_analysis.py`).

### Pluggable News Module

**File**: `src/analysis/news_sources.py` (ports & adapters)

- `NewsSource` protocol; adapters:
  - `GoogleNewsSource` (**PRIMARY** — Google News RSS, date-scoped via `after:`/`before:`, no key, no
    rate limit, aggregates Investing.com / CNBC / CoinDesk / Reuters)
  - `GDELTSource` (fallback, paced ≤1 req/5s + backoff)
- `SOURCE_REGISTRY` + `build_news_sources()` factory + `AssetNewsFetcher` facade
  (strategy `first_nonempty` | `aggregate`).
- Adding a news source = one adapter class + one registry line; SSOT for source order/config is
  `config/analysis/analysis_assets.yaml` `news.sources`.

### Status

- **27 weeks × Gold + BTC** generated with real news (~40 articles/week, 1,080/asset).
- Tests: `tests/unit/test_news_sources.py` (7) + `tests/unit/test_asset_analysis_generator.py` (7).

### CLI

```bash
# Generate a Gold/BTC week (real data)
python scripts/pipeline/generate_asset_analysis.py --asset xauusd --week 2026-W09
python scripts/pipeline/generate_asset_analysis.py --asset btcusdt --week 2026-W09
```

---

## Airflow DAGs (6 total)

### NewsEngine DAGs (5)

| DAG | File | Schedule | Purpose |
|-----|------|----------|---------|
| `news_daily_pipeline` | `news_daily_pipeline.py` | 3x/day (7,12,18 UTC) Mon-Fri | Ingest → Enrich → Cross-ref → Features + Digest |
| `news_alert_monitor` | `news_alert_monitor.py` | Every 30min (7-22 UTC) Mon-Fri | GDELT crisis keyword scan |
| `news_weekly_digest` | `news_weekly_digest.py` | Mon 8:00 UTC | Previous week text digest |
| `news_maintenance` | `news_maintenance.py` | Sun 3:00 UTC | Cleanup logs, vacuum DB, backup |
| `analysis_l8_daily_generation` | `analysis_l8_daily_generation.py` | Mon-Fri 19:00 UTC (14:00 COT) | Daily analysis + Friday weekly + JSON export |

### Analysis L8 DAG Tasks

```
check_trading_day (ShortCircuit)
    → compute_macro_snapshots
    → generate_daily_analysis
    → check_if_friday (ShortCircuit)
    → generate_weekly_summary (Fri only)
    → export_dashboard_json
```

---

## Integration with Existing System

### What Analysis Reads (FROM existing system)

| Data | Source | Used For |
|------|--------|----------|
| USDCOP OHLCV | `usdcop_daily_ohlcv.parquet` | Day price summary in LLM context |
| Macro Indicators | `MACRO_DAILY_CLEAN.parquet` | SMA/BB/RSI/MACD computation |
| H1 Forecast Signals | `forecast_h1_signals` table | Daily analysis context (TODO: wire) |
| H5 Forecast Signals | `forecast_h5_signals` table | Weekly analysis context (TODO: wire) |

### What Analysis Produces (NEW)

| Output | Consumer |
|--------|----------|
| `analysis_index.json` | Dashboard `/analysis` page |
| `weekly_YYYY_WXX.json` | Dashboard `/analysis` page |
| `charts/*.png` | Dashboard macro chart grid |
| ~60 news features/day | Future ML integration |
| Daily/weekly DB records | Airflow monitoring |

---

## Operational Status (Updated 2026-04-06)

### News Pipeline: OPERATIONAL
- `news_daily_pipeline` runs 3x/day (02:00, 07:00, 13:00 COT), all tasks complete
- Active sources: Investing.com (78 articles in DB), Portafolio (283 articles in DB)
- Total articles in DB: 361 (as of 2026-04-06)
- Inactive: GDELT (rate limited), LaRepublica (low match rate), NewsAPI (no API key)
- LLM weekly analysis: W01-W15 generated (Azure OpenAI GPT-4o-mini, ~$0.01/week)
- DB gap: `weekly_analysis` and `daily_analysis` tables have 0 rows (L8 writes to JSON only)
- Container deps: `feedparser` + `vaderSentiment` must be installed after restart
- Historical data: ~13K Google News + ~3K GDELT + ~1.5K Investing articles in CSV

### Bugs Fixed (2026-03-12/13)
1. `registry.enabled_adapters()` returns list, not tuples — fixed tuple unpacking
2. DB connection: `_get_news_config()` now uses `POSTGRES_*` env vars (not `USDCOP_DB_*`)
3. `upsert_raw_articles_batch()` added for `RawArticle` ingestion (separate from `EnrichedArticle` upsert)
4. `_rows_to_enriched()` helper converts DB dicts to `EnrichedArticle` objects
5. Parameter names fixed: `articles_fetched=`, `error_details=` (not `articles_count=`, `error_message=`)
6. Status value: `"failed"` not `"error"` (DB CHECK constraint)
7. `FeatureExporter()` takes no constructor args (removed `config.feature_export`)
8. `snapshot_date` datetime→date conversion for DB upsert
9. `digest.total_articles` (not `article_count`) in log line
10. **`vaderSentiment` package** (not `nltk.sentiment`) — correct pip package for VADER
11. **UTC→COT date mismatch** — `date.today()` in UTC didn't match `DATE(published_at AT TIME ZONE 'America/Bogota')`, causing enrichment to find 0 articles. Fixed with `_today_cot()` helper using `ZoneInfo('America/Bogota')`

---

## Known TODOs (Non-Blocking)

| Location | TODO | Impact |
|----------|------|--------|
| `weekly_generator.py:96` | `build_signal_section()` returns empty | Signals show empty in LLM prompt |
| `weekly_generator.py:235-236` | H5/H1 signals dict empty | SignalSummaryCards show no data |
| `weekly_generator.py:238` | Economic calendar not wired | UpcomingEventsPanel empty |
| `weekly_generator.py:246` | NewsEngine not wired | News context empty in LLM prompt |

These TODOs mean the analysis generates correctly but with empty signals/news/calendar sections.
The macro analysis and OHLCV context work fully.

---

## SDD Documents (Detailed Specs)

Full specifications are in `.claude/specs/tracks/news-analysis/` (14 documents, v2.0.0):

| Doc | Title | Focus |
|-----|-------|-------|
| 00 | INDEX | Document map and dependencies |
| 01 | SYSTEM_OVERVIEW | Architecture, data flow, integration points |
| 02 | NEWS_ENGINE_ARCHITECTURE | Adapters, enrichment, storage, features |
| 03 | ANALYSIS_MODULE_SPEC | LLM client, macro analyzer, generators |
| 04 | DATA_CONTRACTS | Python + TypeScript schemas |
| 05 | DATABASE_SCHEMA | All tables, indexes, migrations |
| 06 | DASHBOARD_ANALYSIS_PAGE | React components, API routes, visual design |
| 07 | AIRFLOW_DAGS | DAG specs, task flows, schedules |
| 08 | CONFIG_AND_SSOT | YAML config, env vars, secrets |
| 09 | TESTING_STRATEGY | Unit, integration, E2E test plan |
| 10 | DEPLOYMENT_GUIDE | Docker, Airflow, monitoring |
| 11 | SECURITY_AND_COST | Rate limits, LLM budget, API key management |
| 12 | DESIGN_DECISIONS | 12 ADRs (Architecture Decision Records) |

---

## DO NOT

- Do NOT call LLM without checking budget limits (`$1/day`, `$15/month`)
- Do NOT skip caching — LLM calls are expensive; always check cache first
- Do NOT store raw HTML in `news_articles` — only store cleaned text
- Do NOT use VADER as primary sentiment for GDELT articles — GDELT tone is more reliable
- Do NOT modify enrichment categories without updating both Python + TypeScript contracts
- Do NOT hardcode LLM provider — use the strategy pattern (primary + fallback)
- Do NOT call `/api/analysis/chat` without rate limiting (max 100/day per user)
- Do NOT skip `_sanitize_for_json()` on analysis exports — same JSON safety as strategy exports
- Do NOT run `analysis_l8_daily_generation` before news ingestion DAGs complete (19:00 UTC = 2h after last ingestion)

---

## Reconciliation (audit 2026-07)

> Corrects drift found by the 10-agent audit (see `../../audit/AUDIT-2026-07-remediation.md` §A10).

- **The "Known TODOs" table is obsolete** — signals/news/calendar integration is wired and generating features. Do not treat those rows as open work.
- **Real remaining gap**: some migration-046 analysis tables are created but currently unused (A10-04/05).
