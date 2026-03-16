# SDD-10: Unified Orchestration & Operations

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-10 |
| **Título** | Unified Orchestration & Operations |
| **Versión** | 2.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🔄 MERGED from NewsEngine SDD-07 + Analysis Module Airflow spec |

---

## 1. DAG Registry

| # | DAG ID | Schedule | Subsystem | Purpose |
|---|--------|----------|-----------|---------|
| 1 | `news_daily_pipeline` | `0 7,12,18 * * 1-5` | NewsEngine | Ingest + enrich + cross-ref + features + digest |
| 2 | `news_alert_monitor` | `*/30 7-22 * * 1-5` | NewsEngine | GDELT crisis keywords scan |
| 3 | `news_weekly_digest` | `0 8 * * 1` | NewsEngine | Monday weekly text digest |
| 4 | `analysis_l8_daily_generation` | `0 19 * * 1-5` | Analysis | AI daily analysis + Friday weekly + JSON export |
| 5 | `news_maintenance` | `0 3 * * 0` | Both | Cleanup logs, vacuum DB, backup |

---

## 2. Schedule Overview (Typical Week)

```
MONDAY – FRIDAY
  07:00 UTC  news_daily_pipeline ──── Ingest → Enrich → CrossRef → Features → Digest
  07:30 UTC  news_alert_monitor ───── Every 30 min until 22:00 (crisis scan)
  12:00 UTC  news_daily_pipeline ──── Second run
  18:00 UTC  news_daily_pipeline ──── Third run
  19:00 UTC  analysis_l8_daily ────── Macro SMA → LLM daily analysis → JSON export
                                       (depends on 18:00 NewsEngine data being fresh)
  21:00 UTC  (Fridays only) ───────── Weekly analysis generation + export

MONDAY (additional)
  08:00 UTC  news_weekly_digest ───── Text digest of previous week

SUNDAY
  03:00 UTC  news_maintenance ─────── Cleanup + vacuum + backup
```

**Key dependency:** `analysis_l8_daily_generation` runs at 19:00 UTC, **2 hours after** the last NewsEngine pipeline run (18:00 UTC), ensuring all article data is fresh.

---

## 3. DAG 1: `news_daily_pipeline` (NewsEngine)

```
health_check
    │
    ├── ingest_apis (GDELT, NewsAPI)
    ├── ingest_scrapers (Investing, LaRepública, Portafolio)
    └── ingest_macro (FRED, BanRep)
         │
         ▼
    enrich_all
         │
         ▼
    cross_ref
         │
         ├── export_features
         ├── daily_digest
         └── alert_check
```

| Task | Timeout | Retries | Trigger Rule |
|------|---------|---------|-------------|
| `health_check` | 60s | 0 | — |
| `ingest_apis` | 300s | 2 | `all_done` |
| `ingest_scrapers` | 600s | 1 | `all_done` |
| `ingest_macro` | 120s | 2 | `all_done` |
| `enrich_all` | 300s | 1 | `one_success` |
| `cross_ref` | 120s | 1 | `all_success` |
| `export_features` | 60s | 1 | `all_success` |
| `daily_digest` | 60s | 1 | `all_success` |
| `alert_check` | 60s | 1 | `all_success` |

---

## 4. DAG 4: `analysis_l8_daily_generation` (Analysis Module)

```
check_trading_day (ShortCircuit)
         │
    compute_macro_snapshots
         │
    generate_daily_analysis
         │
    check_if_friday (ShortCircuit)
         │                    │
    generate_weekly_summary   │
         │                    │
         └────────┬───────────┘
                  │
         export_dashboard_json
```

| Task | Timeout | Retries | Notes |
|------|---------|---------|-------|
| `check_trading_day` | 10s | 0 | ShortCircuit: skip holidays |
| `compute_macro_snapshots` | 120s | 2 | Reads macro_data + macro_indicators_daily |
| `generate_daily_analysis` | 300s | 2 | LLM call + NewsEngine data query |
| `check_if_friday` | 5s | 0 | ShortCircuit: weekly only on Fridays |
| `generate_weekly_summary` | 300s | 2 | LLM call with all 5 daily summaries |
| `export_dashboard_json` | 60s | 1 | `trigger_rule='none_failed'` |

**NewsEngine integration in `generate_daily_analysis`:**
```python
# Inside the task, query NewsEngine data for the LLM prompt:
article_stats = query_articles_by_date(ds)       # From articles table
crossref_topics = query_crossrefs_by_date(ds)     # From cross_references table
news_sentiment = query_avg_sentiment_by_date(ds)  # From articles.sentiment_score
```

---

## 5. Unified CLI

```
usdcop-intel — USDCOP Trading Intelligence Platform

NewsEngine Commands:
  ingest        Ingest from all or specific sources
  enrich        Run enrichment pipeline
  crossref      Run cross-reference engine
  digest        Generate daily or weekly text digest
  export        Export feature vector CSV/Parquet
  alert         Check for breaking news alerts
  backfill      Historical backfill (NewsEngine)

Analysis Commands:
  analyze       Generate AI daily/weekly analysis
  analyze-export  Export analysis JSON to dashboard
  chat-test     Test chat endpoint with sample question

Shared Commands:
  health        Health check all sources + LLM providers
  stats         Database statistics (both subsystems)
  migrate       Run database migrations
  scheduler     Start automated scheduler daemon
```

---

## 6. Configuration

```python
class UnifiedConfig(BaseSettings):
    """Unified config for both subsystems."""
    
    # ═══ Database ═══
    database_url: str = "postgresql://pedro:xxx@localhost:5432/usdcop_intel"
    
    # ═══ NewsEngine Sources ═══
    gdelt_enabled: bool = True
    newsapi_enabled: bool = True
    newsapi_key: SecretStr
    investing_enabled: bool = True
    larepublica_enabled: bool = True
    portafolio_enabled: bool = True
    fred_enabled: bool = True
    fred_api_key: SecretStr = ""
    banrep_enabled: bool = True
    
    # ═══ Scraping ═══
    scraper_min_delay: float = 2.0
    scraper_max_delay: float = 5.0
    
    # ═══ NewsEngine Enrichment ═══
    crossref_threshold: float = 0.35
    crossref_time_window_hours: int = 48
    
    # ═══ Analysis Module ═══
    analysis_module_enabled: bool = True
    analysis_chat_enabled: bool = True
    
    # ═══ LLM ═══
    azure_openai_api_key: SecretStr
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str = "2024-10-21"
    anthropic_api_key: SecretStr = ""
    
    # ═══ Alerts ═══
    alert_tone_threshold: float = -5.0
    alert_slack_webhook: str = ""
    
    # ═══ Export ═══
    export_dir: str = "./data/exports"
    analysis_export_dir: str = "usdcop-trading-dashboard/public/data/analysis"
    
    class Config:
        env_prefix = "USDCOP_"
        env_file = ".env"
```

---

## 7. Monitoring (Grafana)

```
Panel: Platform Health
├── NewsEngine
│   ├── Articles ingested today: 342
│   ├── Sources healthy: 7/7
│   ├── Enrichment status: complete ✅
│   ├── Cross-references today: 8
│   └── Features exported: v81
│
├── Analysis Module
│   ├── Daily analysis status: generated ✅
│   ├── Weekly report status: complete (Fri only)
│   ├── LLM tokens today: 1,450
│   ├── LLM cost MTD: $8.50
│   ├── Chat sessions active: 2
│   └── Macro snapshots: 13/13 complete
│
└── System
    ├── DB size: 3.2 GB
    ├── Last backup: Sun 03:00
    └── Errors this week: 0
```

---

## 8. Docker Compose (Production)

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: usdcop_intel
      POSTGRES_USER: pedro
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  newsengine:
    build:
      context: .
      dockerfile: Dockerfile.newsengine
    depends_on: [db]
    env_file: .env
    volumes:
      - ./data:/app/data

  airflow-webserver:
    image: apache/airflow:2.8.0
    depends_on: [db]
    volumes:
      - ./airflow/dags:/opt/airflow/dags
    ports:
      - "8080:8080"

  airflow-scheduler:
    image: apache/airflow:2.8.0
    depends_on: [airflow-webserver]
    volumes:
      - ./airflow/dags:/opt/airflow/dags

  # Existing trading system containers (already running):
  # usdcop-trading-api, usdcop-dashboard, etc.
  # Analysis Module endpoints are added to usdcop-trading-api

volumes:
  pgdata:
```
