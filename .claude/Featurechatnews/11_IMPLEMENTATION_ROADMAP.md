# SDD-11: Unified Implementation Roadmap

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-11 |
| **Título** | Implementation Roadmap |
| **Versión** | 2.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🆕 NEW |

---

## 1. Phase Overview

```
Phase 1 (Week 1-2):  NewsEngine MVP — GDELT + BanRep + storage + basic enrichment
Phase 2 (Week 2-3):  NewsEngine Core — All scrapers + sentiment + cross-ref + features
Phase 3 (Week 3-4):  NewsEngine Production — PostgreSQL + FRED + alerts + Airflow
Phase 4 (Week 4-5):  Analysis Foundation — DB migration + contracts + LLM client + MacroAnalyzer
Phase 5 (Week 5-6):  Analysis Engine — Generator + prompts + CLI + DAG
Phase 6 (Week 6-7):  Dashboard Frontend — All React components + API routes
Phase 7 (Week 7-8):  Chat Widget — Floating chat + context injection
Phase 8 (Week 8-9):  Integration & Polish — Cross-system links + monitoring + backfill
```

---

## 2. Phase Details

### Phase 1 — NewsEngine MVP (Week 1-2) 🔴 CRITICAL

| Task | SDD | Files |
|------|-----|-------|
| Pydantic config + models | 02, 03 | `config.py`, `models.py` |
| GDELT DOC adapter | 02 | `gdelt_adapter.py` |
| BanRep adapter (TRM) | 02 | `banrep_adapter.py` |
| SQLite storage + schema | 03 | `database.py`, `models_db.py`, migration 001 |
| Basic enrichment (categorizer + relevance) | 04 | `categorizer.py`, `relevance.py` |
| Feature exporter (CSV) | 06 | `feature_exporter.py` |
| CLI basics | 10 | `cli.py` |

**Verification:** `newsengine ingest --source gdelt_doc` fetches articles → stored in SQLite → `newsengine export` produces CSV.

### Phase 2 — NewsEngine Core (Week 2-3) 🔴 CRITICAL

| Task | SDD | Files |
|------|-----|-------|
| NewsAPI adapter | 02 | `newsapi_adapter.py` |
| La República scraper | 02 | `larepublica_scraper.py` |
| Portafolio scraper | 02 | `portafolio_scraper.py` |
| Sentiment analyzer (GDELT + VADER) | 04 | `sentiment.py` |
| Tagger | 04 | `tagger.py` |
| Cross-reference engine | 05 | `engine.py` |
| Daily digest generator | 06 | `digest_generator.py` |
| Source registry | 02 | `registry.py` |

**Verification:** Full daily pipeline runs: 7 sources → enrichment → cross-ref → digest output.

### Phase 3 — NewsEngine Production (Week 3-4) 🟡 HIGH

| Task | SDD | Files |
|------|-----|-------|
| PostgreSQL migration | 03 | Migration 001 (Postgres version) |
| FRED adapter | 02 | `fred_adapter.py` |
| Investing.com scraper | 02 | `investing_scraper.py` |
| Breaking alert system | 06 | `alert_system.py` |
| Weekly digest | 06 | `digest_generator.py` (weekly mode) |
| Airflow DAGs (NewsEngine) | 10 | `news_daily_pipeline.py`, `news_alert_monitor.py`, etc. |
| Docker compose | 10 | `docker-compose.yml` |

**Verification:** Airflow web UI shows healthy DAGs. Full week of data collected.

### Phase 4 — Analysis Foundation (Week 4-5) 🔴 CRITICAL

| Task | SDD | Files |
|------|-----|-------|
| SSOT YAML config | 07 | `weekly_analysis_ssot.yaml` |
| Python contracts (dataclasses) | 07 | `analysis_schema.py` |
| TypeScript contracts | 08 | `weekly-analysis.contract.ts` |
| DB migration (4 analysis tables) | 03 | Migration 045 |
| LLM client (Azure OpenAI + Anthropic) | 07 | `llm_client.py` |
| MacroAnalyzer (SMA + trends) | 07 | `macro_analyzer.py` |
| Install npm deps | 08 | `react-markdown`, `remark-gfm`, `openai` |

**Verification:** Migration runs clean. `MacroAnalyzer.get_all_snapshots()` returns 13+ variables with SMAs.

### Phase 5 — Analysis Engine (Week 5-6) 🔴 CRITICAL

| Task | SDD | Files |
|------|-----|-------|
| Prompt templates (daily + weekly + chat) | 07 | `prompt_templates.py` |
| WeeklyAnalysisGenerator orchestrator | 07 | `weekly_generator.py` |
| NewsEngine data integration in prompts | 07 | `_build_news_context()` |
| CLI script | 07 | `generate_weekly_analysis.py` |
| Airflow DAG (analysis) | 10 | `analysis_l8_daily_generation.py` |
| Backfill 4 weeks of analysis | — | CLI command |

**Verification:** `python scripts/generate_weekly_analysis.py --week 2026-W09` generates daily + weekly analysis. JSON exported to dashboard directory.

### Phase 6 — Dashboard Frontend (Week 6-7) 🟡 HIGH

| Task | SDD | Files |
|------|-----|-------|
| AnalysisPage + WeekSelector | 08 | 2 components |
| WeeklySummaryHeader + MacroSnapshotBar | 08 | 2 components |
| SignalSummaryCards | 08 | 1 component |
| DailyTimeline + DailyTimelineEntry + MacroEventChip | 08 | 3 components |
| UpcomingEventsPanel + AnalysisMarkdown | 08 | 2 components |
| API routes (4 endpoints) | 08 | 4 route files |
| useWeeklyAnalysis hook | 08 | 1 hook |
| Navigation integration | 08 | 2 modified files |

**Verification:** Navigate to `localhost:3000/analysis`. Week selector works. Timeline shows 5 days with real data. Mobile responsive.

### Phase 7 — Chat Widget (Week 7-8) 🟢 MEDIUM

| Task | SDD | Files |
|------|-----|-------|
| FloatingChatWidget + ChatPanel | 09 | 2 components |
| Zustand chat store | 09 | 1 store |
| POST /api/analysis/chat | 08 | 1 API route |
| Context injection (week + NewsEngine data) | 09 | In chat route |
| Quick actions, typing indicator, auto-scroll | 09 | In ChatPanel |

**Verification:** Click chat → panel opens → send question → get contextual LLM response → rate limiting works.

### Phase 8 — Integration & Polish (Week 8-9) 🟢 MEDIUM

| Task | SDD | Files |
|------|-----|-------|
| Cross-system data links in analysis prompts | 07 | Verified queries |
| Grafana monitoring (both subsystems) | 10 | Dashboard config |
| LLM cost tracking | 07 | `llm_tokens_used` aggregation |
| Redis caching for LLM responses | 07 | Cache layer |
| Dark/light mode consistency | 08 | CSS pass |
| Error boundaries | 08 | React error boundaries |
| E2E tests (Playwright) | — | Test files |
| Documentation (README) | — | README.md |

---

## 3. Timeline Summary

```
Week 1-2:  [████████████████████] NewsEngine MVP
Week 2-3:  [████████████████████] NewsEngine Core
Week 3-4:  [████████████████████] NewsEngine Production
Week 4-5:  [████████████████████] Analysis Foundation
Week 5-6:  [████████████████████] Analysis Engine
Week 6-7:  [████████████████████] Dashboard Frontend
Week 7-8:  [████████████████████] Chat Widget
Week 8-9:  [████████████████████] Integration & Polish
```

**Total: ~9 weeks, ~180 hours**

---

## 4. Risk Matrix

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|------------|
| Investing.com anti-scraping blocks | High | Medium | RSS-first strategy, Investing is lowest priority source |
| LLM rate limits | Medium | High | Dual provider (Azure + Anthropic), Redis cache |
| GDELT data gaps | Medium | Medium | Forward-fill + quality flags |
| Macro data incomplete | Medium | Medium | Forward-fill, `is_complete` flag |
| Frontend performance (large markdown) | Low | Low | Lazy-load collapsed entries |
| Prompt quality drift | Low | Medium | Version prompts, log all inputs/outputs |
| Cost overrun from chat | Low | Medium | Hard rate limits |

---

## 5. New Files Summary

| Subsystem | New Files | Modified Files |
|-----------|-----------|----------------|
| NewsEngine | ~25 Python files | — |
| Analysis Engine | 6 Python files + 1 YAML + 1 SQL | — |
| Dashboard | 19 TS/TSX files | 2 (Navbar, Hub) |
| Airflow | 5 DAG files | — |
| **Total** | **~57 new files** | **2 modified** |
