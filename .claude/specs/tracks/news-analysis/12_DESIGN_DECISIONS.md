# SDD-12: Design Decisions & ADRs

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-12 |
| **Título** | Architectural Decision Records |
| **Versión** | 2.0.0 |
| **Fecha** | 2026-02-25 |

---

## ADR-001: Two Pipelines, One Storage

**Context:** We have two subsystems — NewsEngine (data acquisition) and Analysis Module (AI narrative). They could use separate databases.

**Decision:** Single PostgreSQL database with two table groups (A: NewsEngine, B: Analysis). Analysis reads from NewsEngine tables for prompt context.

**Consequences:** Simpler deployment. Cross-system queries are efficient (same DB). Must manage table naming carefully (`daily_digests` vs `daily_analysis`).

---

## ADR-002: Unified SourceAdapter for APIs + Scrapers

**Context:** APIs and scrapers have different error profiles and data formats.

**Decision:** Single `SourceAdapter` ABC with implementations for both. All normalize to `RawArticle` schema.

**Consequences:** Downstream pipeline (enrichment, cross-ref, features) is source-agnostic. Adapter complexity absorbed at ingestion layer.

---

## ADR-003: GDELT as Primary Sentiment Source

**Context:** GDELT includes tone scores. Scraped articles don't.

**Decision:** GDELT tone as primary. VADER as fallback for scraper-only articles. `pysentimiento` planned for V2.

**Consequences:** Sentiment quality varies by source. The `sentiment_source` column tracks provenance.

---

## ADR-004: Feature Vector at Day Level, Not Article Level

**Context:** RL model operates on daily timeframes, not per-article.

**Decision:** Aggregate all article metrics to daily vectors (~81 features). Individual articles retained in DB for analysis prompts.

**Consequences:** Clean RL input. Article-level detail available for Analysis Module prompts.

---

## ADR-005: Enrichment Separate from Ingestion

**Context:** Could enrich during ingestion for simplicity.

**Decision:** Two-phase: ingest raw, then enrich. Allows re-enrichment without re-ingestion.

**Consequences:** Extra pipeline step, but historical data can be re-processed when keywords/categories change.

---

## ADR-006: Dual Persistence (DB + Static JSON) for Analysis

**Context:** Analysis data needs fast dashboard reads AND rich context for chat.

**Decision:** Write to PostgreSQL (source of truth) AND export pre-computed JSON files to `public/data/analysis/`. Dashboard reads JSON. Chat reads DB.

**Consequences:** Slight data duplication. Dashboard works without DB connection. Follows existing parquet→JSON export pattern.

---

## ADR-007: Vertical Timeline Over Horizontal

**Context:** Original design had horizontal dots (`●──●──●──●──●`).

**Decision:** Vertical timeline with expandable cards per day.

**Rationale:** Horizontal is sparse for 5 items. Vertical accommodates variable-length markdown. Mobile-friendly (native scroll). Matches "blog post" metaphor.

---

## ADR-008: Azure OpenAI Primary, Anthropic Fallback

**Context:** Need reliable LLM provider for daily generation.

**Decision:** Azure OpenAI primary (existing deployment). Anthropic as automatic fallback after 3 consecutive failures.

**Rationale:** Strategy pattern allows hot-swap. Cost comparable at this volume (~$8-15/month).

---

## ADR-009: Single Airflow DAG for Analysis (Not 3)

**Context:** Could split into daily, weekly, and calendar sync DAGs.

**Decision:** One DAG (`analysis_l8_daily_generation`) with ShortCircuit branching.

**Rationale:** Simpler dependency management. Calendar handled by existing `EconomicCalendar` class. Friday-only weekly generation via ShortCircuit operator.

---

## ADR-010: NewsEngine Data in LLM Prompts

**Context:** Analysis Engine could generate narratives from macro data alone, ignoring NewsEngine articles.

**Decision:** Explicitly query NewsEngine tables (articles, cross_references) and include summaries in LLM prompts.

**Rationale:** This is the key integration point. Analysis mentioning "3 sources reported DXY strength today" is far more valuable than pure macro analysis. Cross-references (SDD-05) are a strong signal of importance.

---

## ADR-011: HTTP Chat First, WebSocket Later

**Context:** Chat could use WebSocket streaming from day one.

**Decision:** MVP uses HTTP POST (full response). WebSocket streaming added in Phase 7 polish.

**Rationale:** For ~200-word responses, latency difference is minimal (2-4s). HTTP is simpler to implement, test, and debug. WebSocket adds reconnection complexity.

---

## ADR-012: Chat Scoped to /analysis Only

**Context:** Chat could be available globally.

**Decision:** FloatingChatWidget only renders on `/analysis` route.

**Rationale:** Context is only meaningful with analysis data. Avoids confusing users on other pages. Cost control (fewer accidental sessions).

---

## SOLID Compliance

| Principle | Application |
|-----------|------------|
| **S** — Single Responsibility | `MacroAnalyzer` only computes SMAs. `LLMClient` only calls APIs. `Categorizer` only assigns categories. Each React component has one visual job. |
| **O** — Open/Closed | `SourceAdapter` ABC extensible for new sources. `LLMClient` ABC extensible for new providers. SSOT YAML allows adding variables without code changes. |
| **L** — Liskov Substitution | `AzureOpenAIClient` and `AnthropicClient` fully interchangeable. Any `SourceAdapter` impl works in `SourceRegistry`. |
| **I** — Interface Segregation | Separate hooks per data type. Separate repos per table. No mega-interfaces. |
| **D** — Dependency Inversion | `WeeklyGenerator` depends on `LLMClient` abstraction. Enrichment pipeline depends on scorer interfaces, not implementations. |

---

## DRY Compliance

| Reused Asset | From | By |
|-------------|------|-----|
| `EconomicCalendar` class | Existing trading system | Analysis Engine (publication detection) |
| `UnifiedMacroLoader` | Existing trading system | Analysis Engine (SMA computation) |
| `MACRO_DB_TO_FRIENDLY` | `src/data/contracts.py` | Analysis Engine (display names) |
| `safe_json_dump()` | `src/contracts/strategy_schema.py` | Analysis Engine (JSON export) |
| PostgreSQL instance | Existing infra | Both subsystems |
| Redis instance | Existing infra | LLM response caching |
| Airflow cluster | Existing infra | All DAGs |
| Card/Badge components | Dashboard UI library | Analysis page |
| Tailwind design tokens | Dashboard theme | Analysis page |
| React Query | Dashboard state | Analysis hooks |
| Framer Motion | Dashboard animations | Timeline + chat |
