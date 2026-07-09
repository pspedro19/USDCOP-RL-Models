# SDD-00: Unified System Architecture

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-00 |
| **Título** | System Architecture — USDCOP Trading Intelligence Platform |
| **Versión** | 2.0.0 |
| **Autor** | Pedro Sánchez Briceño — Finaipro / Lean Tech |
| **Fecha** | 2026-02-25 |
| **Estado** | Draft |
| **Subsistemas** | NewsEngine (data) + Analysis Module (presentation) |

---

## 1. Problem Statement

The USDCOP trading system operates two forecasting pipelines (H1 daily, H5 weekly) tracking 40+ macro variables. Two critical capabilities are missing:

**Gap 1 — No unified news intelligence.** News/sentiment data comes from scattered scripts with no shared storage, duplicated logic, and no cross-referencing between sources.

**Gap 2 — No narrative analysis layer.** Operators must manually correlate model signals, macro movements, and economic events. There is no AI-generated interpretation of what happened and why.

## 2. Solution: Two Pipelines, One Platform

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     USDCOP TRADING INTELLIGENCE PLATFORM                                 │
│                                                                                          │
│  ┌─── PIPELINE A: NewsEngine (Data) ──────────────────────────────────────────────────┐ │
│  │                                                                                     │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │ │
│  │  │ SOURCES  │──▶│ INGEST   │──▶│ ENRICH   │──▶│ CROSS-REF│──▶│ FEATURE EXPORT   │ │ │
│  │  │ SDD-01   │   │ SDD-02   │   │ SDD-04   │   │ SDD-05   │   │ SDD-06           │ │ │
│  │  │          │   │          │   │          │   │          │   │ ~81 features/day  │ │ │
│  │  │ 9 sources│   │ Adapters │   │ Category │   │ Cluster  │   │ → RL model input  │ │ │
│  │  │ API+scrp │   │ Normalize│   │ Sentimnt │   │ Topic    │   │ → Digest/Alerts   │ │ │
│  │  └──────────┘   └──────────┘   │ Relevnce │   │ Score    │   └──────────────────┘ │ │
│  │                                 └──────────┘   └──────────┘                         │ │
│  └─────────────────────────────────────┬───────────────────────────────────────────────┘ │
│                                        │                                                 │
│                                        ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐│
│  │                        UNIFIED POSTGRESQL STORAGE  (SDD-03)                          ││
│  │  NewsEngine: sources, articles, macro_data, keywords, cross_references,              ││
│  │              daily_digests, ingestion_log, feature_snapshots                          ││
│  │  Analysis:   weekly_analysis, daily_analysis, macro_variable_snapshots,               ││
│  │              analysis_chat_history                                                    ││
│  └──────────────────────────────────────────────────────────────────────────────────────┘│
│                                        │                                                 │
│  ┌─── PIPELINE B: Analysis Module (Presentation) ────────────────────────────────────┐  │
│  │                                     │                                              │  │
│  │  ┌──────────────────┐   ┌───────────▼────────┐   ┌──────────────────────────────┐ │  │
│  │  │ ANALYSIS ENGINE  │   │ DASHBOARD FRONTEND │   │ CHAT WIDGET                 │ │  │
│  │  │ SDD-07           │   │ SDD-08             │   │ SDD-09                      │ │  │
│  │  │                  │   │                    │   │                             │ │  │
│  │  │ • Macro SMA calc │   │ • /analysis page   │   │ • Floating popup            │ │  │
│  │  │ • LLM prompts    │──▶│ • Week selector    │   │ • Context-aware LLM         │ │  │
│  │  │ • Daily/weekly   │   │ • Daily timeline   │   │ • Streaming responses       │ │  │
│  │  │   generation     │   │ • Macro snapshots  │   │ • Session persistence       │ │  │
│  │  │ • JSON export    │   │ • Signal cards     │   │                             │ │  │
│  │  └──────────────────┘   └────────────────────┘   └──────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│  ┌─── ORCHESTRATION (SDD-10) ────────────────────────────────────────────────────────┐  │
│  │  Airflow DAGs: news_daily_pipeline, news_alert_monitor, news_weekly_digest,        │  │
│  │                analysis_l8_daily_generation, news_maintenance                      │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow — End to End

```
PHASE 1: ACQUISITION (NewsEngine — SDD-01, 02)
  GDELT, NewsAPI, Scrapers, FRED, BanRep
        │
        ▼ RawArticle / MacroDataPoint
PHASE 2: STORAGE (SDD-03)
  PostgreSQL: articles, macro_data
        │
        ▼
PHASE 3: ENRICHMENT (SDD-04)
  Categorize → Tag → Score relevance → Sentiment → Weekly detect
        │
        ▼
PHASE 4: CROSS-REFERENCE (SDD-05)
  Cluster articles by topic → Match score → Multi-source detection
        │
        ▼
PHASE 5: OUTPUT (SDD-06)
  Feature vector (81 features) → CSV/Parquet → RL Model (PPO/SAC)
  Daily digest → Text/JSON → Console
  Breaking alerts → Slack/Telegram
        │
        ▼ (enriched articles + macro_data + model signals feed into...)
PHASE 6: ANALYSIS GENERATION (SDD-07)
  Macro SMA computation → LLM prompt building → AI generation
  → daily_analysis, weekly_analysis rows + JSON export
        │
        ▼
PHASE 7: PRESENTATION (SDD-08, 09)
  Next.js dashboard: /analysis page with timeline, macro cards, calendar
  Floating chat widget with contextual LLM assistant
```

---

## 4. Integration Points Between Subsystems

### 4.1 NewsEngine → Analysis Module

| Data | From (NewsEngine) | To (Analysis Module) | How |
|------|--------------------|----------------------|-----|
| Enriched articles | `articles` table (categories, sentiment, relevance) | `daily_analysis.key_events`, prompt context | SQL query by date |
| Cross-references | `cross_references` table | Prompt context for "multi-source topics" | SQL join |
| Macro indicators | `macro_data` table | `macro_variable_snapshots` SMA computation | Direct read |
| Feature snapshots | `feature_snapshots` table | Optional: enrich analysis with model features | JSON read |
| News volume counts | `articles` COUNT by category/date | Analysis prompt: "X articles about oil today" | SQL aggregation |

### 4.2 Analysis Module → Existing Trading System

| Data | From (Analysis Module) | To (Trading Dashboard) | How |
|------|--------------------|----------------------|-----|
| Weekly analysis JSON | `weekly_analysis` table → JSON export | `/analysis` page | Static JSON files |
| Macro snapshots | `macro_variable_snapshots` | Macro cards in dashboard | JSON export |
| Chat responses | Azure OpenAI / Anthropic | Chat widget | HTTP/WebSocket API |
| Model signals context | `forecast_h1`, `forecast_h5` tables (existing) | Analysis prompts + signal cards | SQL read |

### 4.3 Existing Trading System → Analysis Module

| Data | From (Existing) | Used By (Analysis Module) | How |
|------|--------------------|----------------------|-----|
| USDCOP OHLCV | Parquet files / DB | Daily/weekly price summaries | Pandas read |
| H1 forecast signals | `forecast_h1` DB tables | Daily analysis prompt + signal cards | SQL query |
| H5 forecast signals | `forecast_h5` DB tables | Weekly analysis prompt + signal cards | SQL query |
| Macro indicators (40+) | `macro_indicators_daily` | SMA computation, prompt context | `UnifiedMacroLoader` |
| Economic calendar | `EconomicCalendar` class | Publication detection for analysis | Class method call |

---

## 5. Unified Tech Stack

| Layer | Technology | Used By |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Both subsystems |
| **Storage** | PostgreSQL 16 (prod), SQLite (dev) | Both subsystems |
| **ORM** | SQLAlchemy 2.0 + Alembic | NewsEngine |
| **Orchestration** | Apache Airflow 2.8+ | Both subsystems |
| **ML Tracking** | MLflow | Trading system (existing) |
| **LLM Primary** | Azure OpenAI (GPT-4o) | Analysis Module |
| **LLM Fallback** | Anthropic (Claude Sonnet 4.5) | Analysis Module |
| **HTTP Client** | `requests` + `httpx` (async) | Both subsystems |
| **HTML Parsing** | `beautifulsoup4`, `feedparser` | NewsEngine scrapers |
| **NLP** | `vaderSentiment` (V1), `pysentimiento` (V2) | NewsEngine enrichment |
| **Data Processing** | `pandas`, `numpy` | Both subsystems |
| **Config** | `pydantic-settings` | Both subsystems |
| **Frontend** | Next.js 14 + React 18 + TypeScript | Dashboard |
| **UI Framework** | Tailwind CSS + Radix UI + Framer Motion | Dashboard |
| **State Management** | Zustand + @tanstack/react-query | Dashboard |
| **Charting** | Recharts | Dashboard |
| **Markdown** | `react-markdown` + `remark-gfm` | Dashboard |
| **Cache** | Redis | LLM response caching |
| **Testing** | `pytest` (Python), Playwright (E2E) | Both subsystems |

---

## 6. Unified Directory Structure

```
usdcop-trading-platform/
│
├── specs/                                    # This SDD suite
│   ├── 00_SYSTEM_ARCHITECTURE.md
│   ├── 01_DATA_SOURCES.md
│   ├── 02_INGESTION_LAYER.md
│   ├── 03_STORAGE_SCHEMA.md
│   ├── 04_ENRICHMENT_PIPELINE.md
│   ├── 05_CROSS_REFERENCE_ENGINE.md
│   ├── 06_OUTPUT_LAYER.md
│   ├── 07_ANALYSIS_ENGINE.md
│   ├── 08_DASHBOARD_FRONTEND.md
│   ├── 09_CHAT_WIDGET.md
│   ├── 10_ORCHESTRATION.md
│   ├── 11_IMPLEMENTATION_ROADMAP.md
│   └── 12_DESIGN_DECISIONS.md
│
├── src/
│   ├── news_engine/                          # SUBSYSTEM A: NewsEngine
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── ingestion/                        # SDD-02
│   │   │   ├── base_adapter.py
│   │   │   ├── gdelt_adapter.py
│   │   │   ├── newsapi_adapter.py
│   │   │   ├── investing_scraper.py
│   │   │   ├── larepublica_scraper.py
│   │   │   ├── portafolio_scraper.py
│   │   │   ├── fred_adapter.py
│   │   │   ├── banrep_adapter.py
│   │   │   └── registry.py
│   │   ├── storage/                          # SDD-03
│   │   │   ├── database.py
│   │   │   ├── models_db.py
│   │   │   └── migrations/
│   │   ├── enrichment/                       # SDD-04
│   │   │   ├── pipeline.py
│   │   │   ├── categorizer.py
│   │   │   ├── relevance.py
│   │   │   ├── sentiment.py
│   │   │   └── tagger.py
│   │   ├── cross_reference/                  # SDD-05
│   │   │   └── engine.py
│   │   └── output/                           # SDD-06
│   │       ├── feature_exporter.py
│   │       ├── digest_generator.py
│   │       └── alert_system.py
│   │
│   ├── analysis/                             # SUBSYSTEM B: Analysis Module
│   │   ├── __init__.py
│   │   ├── llm_client.py                     # SDD-07 — LLM provider abstraction
│   │   ├── macro_analyzer.py                 # SDD-07 — SMA + trend computation
│   │   ├── prompt_templates.py               # SDD-07 — Spanish prompt templates
│   │   └── weekly_generator.py               # SDD-07 — Orchestrator
│   │
│   ├── contracts/
│   │   ├── news_engine_schema.py             # NewsEngine Pydantic models
│   │   └── analysis_schema.py                # Analysis Module dataclasses
│   │
│   └── cli.py                                # Unified CLI (Typer)
│
├── config/
│   └── analysis/
│       └── weekly_analysis_ssot.yaml         # Analysis SSOT config
│
├── scripts/
│   └── generate_weekly_analysis.py           # Analysis CLI tool
│
├── airflow/
│   └── dags/
│       ├── news_daily_pipeline.py            # NewsEngine DAGs (SDD-10)
│       ├── news_alert_monitor.py
│       ├── news_weekly_digest.py
│       ├── news_maintenance.py
│       └── analysis_l8_daily_generation.py   # Analysis DAG (SDD-10)
│
├── database/
│   └── migrations/
│       ├── 001_newsengine_initial.sql        # NewsEngine tables
│       └── 045_weekly_analysis_tables.sql    # Analysis tables
│
├── usdcop-trading-dashboard/                 # Next.js Dashboard
│   ├── app/
│   │   ├── analysis/page.tsx                 # SDD-08
│   │   └── api/analysis/                     # SDD-08 API routes
│   ├── components/analysis/                  # SDD-08 + SDD-09
│   ├── hooks/useWeeklyAnalysis.ts            # SDD-08
│   ├── lib/contracts/
│   │   └── weekly-analysis.contract.ts       # TS types
│   └── public/data/analysis/                 # JSON export target
│
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 7. Design Principles

| # | Principle | Application |
|---|-----------|-------------|
| 1 | **Source-Agnostic Processing** | All articles (API or scraper) go through identical enrichment |
| 2 | **Idempotency** | Every pipeline step produces same result on re-run |
| 3 | **Fail-Safe by Source** | One source failing doesn't stop the pipeline |
| 4 | **Feature Reproducibility** | Feature vectors are 100% reproducible from stored data |
| 5 | **Separation of Concerns** | Ingestion → Storage → Enrichment → Output → Analysis are independent |
| 6 | **Zero New Containers** | Analysis module reuses existing PostgreSQL, Redis, Airflow, FastAPI, Next.js |
| 7 | **SOLID/DRY** | Abstract interfaces, single responsibility, dependency inversion throughout |
| 8 | **Spec-Driven** | Every component has a written spec before implementation |
| 9 | **Feature-Flagged** | `NEWS_ENGINE_ENABLED`, `ANALYSIS_MODULE_ENABLED` control activation |
| 10 | **Cost-Conscious** | LLM budget ~$8-15/month with aggressive caching |

---

## 8. Retained Documents (No Changes Needed)

The following SDD documents from the original NewsEngine spec suite are **retained as-is** because the Analysis Module does not modify their scope:

| SDD | Why Unchanged |
|-----|---------------|
| **SDD-01** (Data Sources) | Analysis Module consumes existing sources, doesn't add new ingestion sources |
| **SDD-02** (Ingestion Layer) | SourceAdapter pattern and all adapters remain identical |
| **SDD-04** (Enrichment Pipeline) | Categorizer, tagger, relevance, sentiment logic unchanged |
| **SDD-05** (Cross-Reference Engine) | Clustering algorithm unchanged; output feeds into analysis prompts |
| **SDD-06** (Output Layer) | Feature vector spec unchanged; daily/weekly digests coexist with analysis reports |

These documents should be placed alongside the new/merged documents in the `specs/` directory.
