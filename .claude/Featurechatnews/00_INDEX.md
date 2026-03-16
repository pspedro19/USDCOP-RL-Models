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
| **11** | [Implementation Roadmap](11_IMPLEMENTATION_ROADMAP.md) | 🆕 NEW | 8-phase plan, timeline, risks, verification |
| **12** | [Design Decisions](12_DESIGN_DECISIONS.md) | 🆕 NEW | Unified ADRs, SOLID/DRY compliance, trade-offs |

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
