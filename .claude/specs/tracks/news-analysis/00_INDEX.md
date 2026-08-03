---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/analysis/asset_analysis_generator.py
  - scripts/pipeline/generate_asset_analysis.py
  - src/analysis/news_sources.py
  - config/analysis/analysis_assets.yaml
  - usdcop-trading-dashboard/lib/contracts/analysis-assets.ts
---
# USDCOP Trading Intelligence Platform — Unified SDD Suite

**Author:** Pedro Sánchez Briceño  
**Version:** 2.0.0  
**Date:** February 2026  
**Status:** DRAFT  

---

## System Identity

**Project Name:** USDCOP Trading Intelligence Platform  
**Subsystems:**
- **NewsEngine** — Data acquisition, enrichment, cross-referencing, feature generation
- **Analysis Module** — AI-generated narrative analysis, dashboard presentation, conversational assistant

---

## Document Map

| SDD | Title | Status | Scope |
|-----|-------|--------|-------|
| **00** | [System Architecture](00_SYSTEM_ARCHITECTURE.md) | 🔄 REWRITTEN | Full platform architecture, data flow, integration |
| **01** | [Data Sources](01_DATA_SOURCES.md) | ✅ RETAINED | 9 news/macro sources + new LLM/dashboard sources |
| **02** | [Ingestion Layer](02_INGESTION_LAYER.md) | ✅ RETAINED | SourceAdapter ABC, GDELT/NewsAPI/Scrapers/FRED/BanRep |
| **03** | [Storage Schema](03_STORAGE_SCHEMA.md) | 🔄 MERGED | NewsEngine tables + Analysis tables (unified schema) |
| **04** | [Enrichment Pipeline](04_ENRICHMENT_PIPELINE.md) | ✅ RETAINED | Categorizer, tagger, relevance, sentiment, weekly detector |
| **05** | [Cross-Reference Engine](05_CROSS_REFERENCE_ENGINE.md) | ✅ RETAINED | Similarity, clustering, topic extraction |
| **06** | [Feature & Output Layer](06_OUTPUT_LAYER.md) | ✅ RETAINED | ~81 feature vector, daily/weekly digests, alerts |
| **07** | [Analysis Engine](07_ANALYSIS_ENGINE.md) | 🆕 NEW | LLM integration, macro SMA analyzer, prompt templates, generator |
| **08** | [Dashboard & Frontend](08_DASHBOARD_FRONTEND.md) | 🆕 NEW | /analysis page, components, design system, data hooks |
| **09** | [Chat Widget](09_CHAT_WIDGET.md) | 🆕 NEW | Floating assistant, context injection, WebSocket, UX |
| **10** | [Orchestration & Ops](10_ORCHESTRATION.md) | 🔄 MERGED | All Airflow DAGs, CLI, deployment, monitoring |
| **11** | [Implementation Roadmap](../../archive/2026-07/11_IMPLEMENTATION_ROADMAP.md) | 🆕 NEW | 8-phase plan, timeline, risks, verification |
| **12** | [Design Decisions](12_DESIGN_DECISIONS.md) | 🆕 NEW | Unified ADRs, SOLID/DRY compliance, trade-offs |

---

## Multi-Asset Analysis & Pluggable News (shipped 2026-07-05)

The `/analysis` page now has a **dynamic asset selector** — USD/COP · Gold (`xauusd`) · Bitcoin
(`btcusdt`) — each showing its own per-week weekly + daily analysis. USD/COP is **unchanged**
(macro LangGraph pipeline `generate_weekly_analysis.py`, legacy root data path); Gold/BTC use a new
real-data generator.

| Concern | Where |
|---------|-------|
| Multi-asset generator (`src/analysis/asset_analysis_generator.py`) + CLI (`scripts/pipeline/generate_asset_analysis.py`) | [07 — Analysis Engine](07_ANALYSIS_ENGINE.md) |
| Pluggable news module (`src/analysis/news_sources.py` — Google News primary, GDELT fallback) | [01 — Data Sources](01_DATA_SOURCES.md) |
| Asset selector UI + asset-aware API routes | [08 — Dashboard & Frontend](08_DASHBOARD_FRONTEND.md) |
| SSOT: `config/analysis/analysis_assets.yaml` (Python) + `usdcop-trading-dashboard/lib/contracts/analysis-assets.ts` (TS) | `_summary.md` |

---

## How the Systems Connect

```
    SDD-01 → SDD-02 → SDD-03 ← SDD-04 ← SDD-05
   (Sources)  (Ingest) (Store)  (Enrich)  (XRef)
                         │
                         ├── SDD-06 (Features → RL Model)
                         │
                         └── SDD-07 (Analysis Engine)
                              │
                              ├── SDD-08 (Dashboard UI)
                              │
                              └── SDD-09 (Chat Widget)

    SDD-10 orchestrates ALL of the above
    SDD-11 sequences the implementation
    SDD-12 documents the decisions
```

## Key Principle: One Storage, Two Pipelines

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED POSTGRESQL                           │
│                                                                  │
│  NewsEngine Tables          Analysis Tables                      │
│  ├── sources                ├── weekly_analysis                  │
│  ├── articles               ├── daily_analysis                   │
│  ├── macro_data             ├── macro_variable_snapshots         │
│  ├── keywords               └── analysis_chat_history            │
│  ├── cross_references                                            │
│  ├── daily_digests          Shared:                               │
│  ├── ingestion_log          ├── macro_data (SDD-03 + SDD-07)    │
│  └── feature_snapshots      └── articles (enriched → analysis)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Documentos de este directorio

<!-- idx:auto -->

| Documento | Estado |
|---|---|
| [SDD-00: Unified System Architecture](00_SYSTEM_ARCHITECTURE.md) | IMPLEMENTED |
| [SDD-01: Data Sources Specification](01_DATA_SOURCES.md) | PARTIAL |
| [SDD-02: Ingestion Layer](02_INGESTION_LAYER.md) | IMPLEMENTED |
| [SDD-03: Unified Storage Schema](03_STORAGE_SCHEMA.md) | IMPLEMENTED |
| [SDD-04: Enrichment Pipeline](04_ENRICHMENT_PIPELINE.md) | PARTIAL |
| [SDD-05: Cross-Reference Engine](05_CROSS_REFERENCE_ENGINE.md) | IMPLEMENTED |
| [SDD-06: Output Layer](06_OUTPUT_LAYER.md) | IMPLEMENTED |
| [SDD-07: Analysis Engine](07_ANALYSIS_ENGINE.md) | PARTIAL |
| [SDD-08: Dashboard & Frontend](08_DASHBOARD_FRONTEND.md) | PARTIAL |
| [SDD-09: Chat Widget](09_CHAT_WIDGET.md) | PARTIAL |
| [SDD-10: Unified Orchestration & Operations](10_ORCHESTRATION.md) | IMPLEMENTED |
| [SDD-12: Design Decisions & ADRs](12_DESIGN_DECISIONS.md) | IMPLEMENTED |
| [Rule: News Engine & Analysis Module](_summary.md) | PARTIAL |

<!-- /idx -->
