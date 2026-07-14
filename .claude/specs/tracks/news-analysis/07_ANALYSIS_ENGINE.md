# SDD-07: Analysis Engine

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-07 |
| **Título** | Analysis Engine — LLM Generation & Macro Analyzer |
| **Versión** | 1.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🆕 NEW |
| **Depende de** | SDD-03, SDD-06 (feature data), existing trading system |
| **Requerido por** | SDD-08 (Dashboard), SDD-09 (Chat), SDD-10 (Orchestration) |

---

## 1. Responsibility

The Analysis Engine takes enriched data from the NewsEngine (articles, macro, cross-references) plus model signals from the existing trading system, and produces **AI-generated narrative analysis** in Spanish. It is the bridge between raw data (Pipeline A) and human-readable insights (Pipeline B).

```
INPUTS                              PROCESSING                    OUTPUTS
──────                              ──────────                    ───────
articles table (enriched)     ──┐
macro_data + macro_indicators ──┤   MacroAnalyzer (SMA)    ──┐
EconomicCalendar class        ──┤──▶PromptBuilder          ──┤──▶ daily_analysis rows
forecast_h1/h5 signals        ──┤   LLMClient (generate)   ──┤   weekly_analysis rows
cross_references              ──┘   ResponseParser          ──┘   JSON exports (dashboard)
                                                                   macro_variable_snapshots
```

---

## 2. Module Structure

```
src/analysis/
├── __init__.py
├── llm_client.py              # LLM provider abstraction (Strategy pattern)
├── macro_analyzer.py          # SMA computation + trend detection
├── prompt_templates.py        # All prompts (Spanish)
└── weekly_generator.py        # Orchestrator: collect → prompt → LLM → persist → export
```

---

## 3. LLM Client — Strategy Pattern

```python
class LLMClient(ABC):
    """Open/Closed: extensible for new providers without modifying consumers."""
    
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Returns (response_text, tokens_used)."""
    
    @abstractmethod
    async def chat(self, system_prompt: str, messages: list[dict]) -> tuple[str, int]:
        """Returns (response_text, tokens_used)."""


class AzureOpenAIClient(LLMClient):
    """Primary provider. Uses openai SDK with Azure config."""
    # Retry: 3 attempts, exponential backoff on 429/500
    # Config from SSOT YAML: deployment, endpoint, api_version


class AnthropicClient(LLMClient):
    """Fallback provider. Uses anthropic SDK."""
    # Model: claude-sonnet-4-5-20250929
    # Activated after 3 consecutive primary failures


def create_llm_client(config: dict) -> LLMClient:
    """Factory. Reads primary_provider from SSOT config."""
```

---

## 4. Macro Analyzer

```python
class MacroAnalyzer:
    """Computes SMA, trends, and publications. 
    
    REUSES existing code:
    - EconomicCalendar from src/data/economic_calendar.py
    - UnifiedMacroLoader from src/data/macro_loader.py
    - MACRO_DB_TO_FRIENDLY from src/data/contracts.py
    
    READS from NewsEngine:
    - macro_data table (FRED/BanRep series) for supplementary data
    """
    
    def get_variable_snapshot(self, variable_key: str, as_of_date: date) -> MacroSnapshot:
        # 1. Load last 60 days via UnifiedMacroLoader
        # 2. Compute SMA-5, 10, 20, 50 (pandas rolling)
        # 3. Detect crosses: golden/death (SMA-5 vs SMA-20 crossover)
        # 4. Compute z-score = (value - mean_20d) / std_20d
        # 5. Determine impact_on_usdcop from SSOT direction config
    
    def get_all_snapshots(self, as_of_date: date) -> list[MacroSnapshot]:
        # Iterate all variables from SSOT config
    
    def get_publications_on_date(self, target_date: date) -> list[MacroPublication]:
        # Uses EconomicCalendar.create_publication_schedule_df()
    
    def get_upcoming_events(self, from_date: date, days_ahead: int = 7) -> list[dict]:
        # Future publications for UpcomingEventsPanel
    
    def persist_snapshots(self, snapshots: list, target_date: date):
        # UPSERT into macro_variable_snapshots table
```

### SMA Signal Logic

```
golden_cross  → SMA_5 crosses ABOVE SMA_20
death_cross   → SMA_5 crosses BELOW SMA_20  
crossing_up   → Value crosses above SMA_20
crossing_down → Value crosses below SMA_20
above         → Value > SMA_20 (steady)
below         → Value < SMA_20 (steady)
```

### Variables Tracked (from SSOT)

| Category | Variables | Impact |
|----------|-----------|--------|
| Key Features | DXY, VIX, WTI, EMBI Colombia | High |
| Rates | Tasa BanRep, IBR, US 10Y, US 2Y | Medium |
| Commodities | Brent, Gold, Coffee | Medium-Low |
| FX Peers | USD/MXN, USD/CLP | Medium-Low |

---

## 5. Prompt Templates (Spanish)

### 5.1 Daily Digest Prompt

```python
SYSTEM_PROMPT_DAILY = """
Eres un analista FX senior especializado en el par USD/COP y mercados emergentes 
latinoamericanos. Generas análisis diarios concisos, técnicos y fundamentados en datos.

REGLAS:
- Responde SOLO en JSON válido con el schema especificado
- Usa datos específicos del contexto proporcionado
- Máximo 250 palabras en el análisis
- Headline máximo 120 caracteres
- Sentimiento basado en impacto neto sobre el COP
"""

def build_daily_user_prompt(date, usdcop_ohlcv, h1_signal, 
                            macro_snapshots, publications,
                            news_summary) -> str:
    """
    Includes:
    - USDCOP OHLCV for the day
    - H1 model signal (direction, confidence, prediction)
    - Macro variable table with SMAs and trends
    - Economic publications that day
    - NewsEngine summary: article count, top categories, sentiment avg
    - Cross-reference topics (from NewsEngine SDD-05)
    
    Output JSON schema:
    {
      "headline": "string (max 120 chars)",
      "analysis": "string (markdown, 150-250 words)",
      "key_events": [{"event": str, "impact": str, "detail": str}],
      "sentiment": "bullish_cop|bearish_cop|neutral|volatile",
      "sentiment_score": float  // -1.0 to +1.0
    }
    """
```

### 5.2 Weekly Report Prompt

```python
SYSTEM_PROMPT_WEEKLY = """
Eres un estratega FX senior escribiendo la revisión semanal del USD/COP para 
una mesa de trading. Sintetizas los análisis diarios en una narrativa semanal 
coherente con perspectiva forward.
"""

def build_weekly_user_prompt(week_stats, daily_summaries, h5_signal,
                             macro_snapshots, upcoming_events,
                             weekly_news_stats) -> str:
    """
    Includes everything from daily PLUS:
    - All 5 daily headlines + summaries
    - H5 weekly signal
    - Week OHLCV
    - Upcoming economic events for next week
    - Weekly news stats from NewsEngine (total articles, top cross-refs)
    
    Output JSON schema:
    {
      "summary": "string (markdown, 300-500 words)",
      "sentiment": "bullish_cop|bearish_cop|neutral|mixed",
      "sentiment_score": float,
      "key_themes": ["string", ...]  // 3-5 themes
    }
    """
```

### 5.3 Chat System Prompt

```python
SYSTEM_PROMPT_CHAT = """
Eres un asistente analista de IA para una mesa de trading USD/COP.
Tienes acceso al contexto completo de la semana actual incluyendo:
- Análisis diarios generados
- Variables macro con SMAs y tendencias  
- Señales del modelo H1 (diario) y H5 (semanal)
- Noticias relevantes del NewsEngine

REGLAS:
- Responde siempre en español
- Referencia datos específicos del contexto
- Sé conciso (máximo 200 palabras)
- No inventes datos que no estén en el contexto
"""
```

### 5.4 NewsEngine Data in Prompts

The Analysis Engine enriches prompts with NewsEngine data:

```python
def _build_news_context(date: date) -> str:
    """Query NewsEngine tables to add news intelligence to prompts."""
    
    # From articles table (SDD-03 A2)
    article_stats = db.query("""
        SELECT category, COUNT(*) as cnt, AVG(sentiment_score) as avg_sent
        FROM articles WHERE DATE(published_at) = :date
        GROUP BY category ORDER BY cnt DESC
    """, date=date)
    
    # From cross_references table (SDD-03 A5)  
    crossrefs = db.query("""
        SELECT topic, match_score, sources_count
        FROM cross_references WHERE reference_date = :date
        ORDER BY match_score DESC LIMIT 5
    """, date=date)
    
    # Format for prompt
    return f"""
NOTICIAS DEL DÍA (NewsEngine):
- Total artículos: {sum(r.cnt for r in article_stats)}
- Categorías: {', '.join(f'{r.category}({r.cnt})' for r in article_stats[:5])}
- Sentiment promedio noticias: {avg_sentiment:.2f}

TEMAS MULTI-FUENTE (Cross-References):
{chr(10).join(f'- "{r.topic}" ({r.sources_count} fuentes, score: {r.match_score:.2f})' 
              for r in crossrefs)}
"""
```

---

## 6. Weekly Generator (Orchestrator)

```python
class WeeklyAnalysisGenerator:
    """Orchestrator: data → prompt → LLM → persist → export.
    
    Dependencies injected via __init__ (Dependency Inversion).
    """
    
    def __init__(self, llm_client: LLMClient, macro_analyzer: MacroAnalyzer, 
                 db_connection, config: dict):
        ...
    
    def generate_daily(self, target_date: date) -> DailyAnalysisRecord:
        # 1. Load USDCOP OHLCV (parquet/DB)
        # 2. Load H1 signal for date
        # 3. Get macro publications (MacroAnalyzer)
        # 4. Get macro snapshots (MacroAnalyzer)
        # 5. Query NewsEngine: article counts, sentiment, cross-refs
        # 6. Build prompt (build_daily_user_prompt)
        # 7. Call LLM (llm_client.generate)
        # 8. Parse JSON response (with retry on invalid JSON)
        # 9. UPSERT daily_analysis
        # 10. Persist macro_variable_snapshots
        # 11. Return record
    
    def generate_weekly(self, iso_year: int, iso_week: int) -> WeeklyAnalysisRecord:
        # 1. Ensure weekly_analysis skeleton exists
        # 2. Load all daily entries for this week
        # 3. Load H5 signal
        # 4. Compute week OHLCV
        # 5. Get macro snapshots (key variables for denormalization)
        # 6. Get upcoming events
        # 7. Query NewsEngine: weekly article stats, top cross-refs
        # 8. Build weekly prompt
        # 9. Call LLM → Parse → UPSERT weekly_analysis
    
    def export_to_dashboard(self, iso_year: int, iso_week: int):
        # Build WeekView JSON → Write to public/data/analysis/
        # Update analysis_index.json
        # Uses safe_json_dump()
    
    @classmethod
    def from_config(cls) -> 'WeeklyAnalysisGenerator':
        # Factory: reads SSOT YAML, creates LLM client, MacroAnalyzer, DB connection
```

---

## 7. CLI Script

```bash
# Generate single day
python scripts/pipeline/generate_weekly_analysis.py --date 2026-02-25

# Generate full week
python scripts/pipeline/generate_weekly_analysis.py --week 2026-W09

# Backfill range
python scripts/pipeline/generate_weekly_analysis.py --from 2026-01-06 --to 2026-02-25

# Export only (no LLM calls)
python scripts/pipeline/generate_weekly_analysis.py --export-only --week 2026-W09

# Dry run (print prompts, no LLM call)
python scripts/pipeline/generate_weekly_analysis.py --dry-run --date 2026-02-25
```

---

## 8. SSOT Configuration

**File:** `config/analysis/weekly_analysis_ssot.yaml`

Single YAML file drives all mappings, LLM config, and schedule:

```yaml
version: "1.0.0"
generation:
  daily_schedule_utc: "19:00"
  weekly_schedule_utc: "21:00"
llm:
  primary_provider: "azure_openai"
  fallback_provider: "anthropic"
  temperature: 0.3
  max_tokens_daily: 1500
  max_tokens_weekly: 3000
  max_tokens_chat: 2000
macro_variables:
  key_features:
    - { db: "fxrt_index_dxy_usa_d_dxy", friendly: "dxy", display: "DXY Index", impact: "high", direction: "direct" }
    - { db: "volt_vix_usa_d_vix", friendly: "vix", display: "VIX", impact: "high", direction: "direct" }
    - { db: "comm_oil_wti_glb_d_wti", friendly: "wti", display: "WTI Oil", impact: "high", direction: "inverse" }
    - { db: "crsk_spread_embi_col_d_embi", friendly: "embi", display: "EMBI Colombia", impact: "high", direction: "direct" }
  rates: [...]
  commodities: [...]
  fx_peers: [...]
sma_periods: [5, 10, 20, 50]
export:
  dashboard_dir: "usdcop-trading-dashboard/public/data/analysis"
chat:
  max_messages_per_session: 50
  max_sessions_per_day: 10
```

---

## 9. Cost Model

| Operation | Frequency | Input Tokens | Output Tokens | Cost/Call |
|-----------|-----------|-------------|---------------|-----------|
| Daily analysis | 5×/week | ~2,500 (incl. news context) | ~1,000 | ~$0.02 |
| Weekly report | 1×/week | ~5,500 | ~2,000 | ~$0.04 |
| Chat message | ~20/day | ~2,000 | ~500 | ~$0.01 |

**Monthly estimate:** $8–15 USD with caching.

---

## 10. Error Handling

| Failure | Handling |
|---------|----------|
| LLM timeout (429/500) | 3 retries, exponential backoff. Switch to fallback provider after 3 consecutive failures. |
| Invalid JSON response | Retry with stricter prompt (1x). Fallback: store raw text as markdown. |
| Missing macro data | Forward-fill from previous day. Flag `is_complete=False`. |
| Missing model signal | Generate without signal section. Note "Señal no disponible". |
| NewsEngine data unavailable | Generate with available data only. Note reduced context. |

---

## 11. Multi-Asset Analysis Generator (Gold / BTC)

> **Added 2026-07-05.** A **second, deliberately separate** generator runs *alongside* the USD/COP
> engine above to serve the science-stack assets (Gold, BTC) on the same `/analysis` page.

### 11.1 Why separate (SOLID)

USD/COP keeps its richer macro-driven LangGraph pipeline (`weekly_generator.py`, sections 1–10).
The science-stack assets (`gold_trend_b2`, `btc_trend_b2`) do **not** need the macro/LLM machinery —
they need a leaner **price / technical / strategy / news** engine. Rather than overload the COP
`WeeklyAnalysisGenerator`, a dedicated `AssetAnalysisGenerator` produces the **same `WeeklyViewData`
JSON contract** the dashboard already consumes, per asset.

- New module: `src/analysis/asset_analysis_generator.py` (class `AssetAnalysisGenerator`)
- New CLI: `scripts/pipeline/generate_asset_analysis.py`
- Assets: Gold (`xauusd`) and BTC (`btcusdt`)

### 11.2 Inputs (REAL data only — no synthetic)

| Input | Source |
|-------|--------|
| Daily OHLCV | `seeds/latest/{xauusd,btcusdt}_daily_ohlcv.parquet` |
| Technicals | Computed from that price series: RSI-Wilder, SMA20/50, MACD, ATR, 20-day support/resistance, trend label, 2 concrete scenarios |
| Strategy positioning | Published backtest trade bundles → mapped into the signal card: `usdcop-trading-dashboard/public/data/strategies/<sid>/backtests/*/trades_*.json` (`gold_trend_b2` / `btc_trend_b2`) |
| Per-day entries | `daily_entries` derived from the price series |
| News | Real news via the pluggable news module `src/analysis/news_sources.py` (Google News primary → GDELT fallback) |

### 11.3 Config-driven (SSOT)

Fully driven by `config/analysis/analysis_assets.yaml` — one per-asset profile
(`symbol`, `chart_symbol`, `ohlcv_seed`, `strategy_id`, `annualization_days`, `google_news_query`).
**No per-asset code** — adding an asset = one YAML entry (+ TS mirror `lib/contracts/analysis-assets.ts`).

### 11.4 Output (per-asset namespace)

```
usdcop-trading-dashboard/public/data/analysis/<asset>/
├── weekly_YYYY_WXX.json      # same WeeklyViewData contract as COP
├── analysis_index.json       # newest-first
└── upcoming_events.json
```

JSON-safe: `NaN`/`Inf` → `null`.

### 11.5 CLI

```bash
python scripts/pipeline/generate_asset_analysis.py --asset xauusd --week 2026-W05
python scripts/pipeline/generate_asset_analysis.py --asset btcusdt --from 2026-W01 --to 2026-W27
python scripts/pipeline/generate_asset_analysis.py --all-assets --year 2026 [--no-news]
```

### 11.6 Status & tests

- **Generated:** 27 weeks × Gold + BTC with real news (~40 articles/week).
- **Tests:** `tests/unit/test_asset_analysis_generator.py` (7) + `tests/unit/test_news_sources.py` (7).

---

## 12. News Intelligence & Political Bias (both tracks — added 2026-07-11)

> **Wiring fix (CTR-NEWS-ENRICH-001).** `NewsClusterCard` + `BiasDistributionCard` on `/analysis`
> render from `news_intelligence.clusters` and `political_bias_analysis`. These were **empty**: for
> USD/COP both fields are produced only by the LangGraph multi-agent pipeline, which silently no-ops
> when its heavy deps are absent at generation time (Airflow/Docker) — the import guard at
> `weekly_generator.py::_enrich_with_langgraph` swallows the `ImportError` and returns. Gold/BTC never
> emitted bias and emitted lean neutral clusters. Now both tracks fill the two blocks
> **deterministically, independent of LangGraph** (mirrors `_compute_regime_direct` for `macro_regime`).

### 12.1 Shared helper — `src/analysis/news_enrichment.py::enrich_news`

Single source of truth for turning a `list[dict]` of articles → `(news_intelligence, political_bias_analysis)`:

1. `NewsIntelligenceEngine.process_articles` (one sentiment pass) → enriched articles with tone /
   category / relevance / source-bias. **Adaptive relevance floor**: prefer articles ≥ `min_relevance`,
   but if fewer than `min_kept` (5) clear it, fall back to the full set so a genuinely-covered week
   never renders an empty panel.
2. `NewsIntelligenceEngine.cluster_articles` → HDBSCAN semantic clusters, **falling back to category
   grouping when embeddings are unavailable *or* HDBSCAN labels everything as noise** (the empty-`[]`
   case that previously returned zero clusters). Every cluster gets a human label (dominant category,
   Title-Cased) as a deterministic floor.
3. `PoliticalBiasDetector.analyze` → source-based bias distribution + factuality + diversity (Layer 1,
   free) — matches `PoliticalBiasOutput` exactly, so **no frontend/contract change**.

**LLM polish is opt-in + budget-gated** (`allow_llm`): cluster topic labels + bias narrative fire only
when an LLM client is passed *and* keys/budget exist; the deterministic path always runs at zero cost.
News only ever classifies/summarizes — it never touches a trading decision (quant-constitution §7).

### 12.2 Track wiring

| Track | Call site | Relevance floor | LLM |
|-------|-----------|-----------------|-----|
| USD/COP | `weekly_generator.py::_compute_news_intelligence` (called in `generate_for_week` step 7c, **only fills what LangGraph left empty**) | `0.3` (adaptive) | `allow_llm = not dry_run` |
| Gold/BTC | `asset_analysis_generator.py::generate_week` (replaces lean `_cluster_news` output) | `0.0` (feeds pre-queried) | off (DAG env) |

### 12.3 Robustness fixes applied to the shared engine (benefit all callers)

- **HDBSCAN-noise → category fallback** (`news_intelligence.py::cluster_articles`): empty HDBSCAN result
  now falls back instead of returning `[]`.
- **Source→domain bias match** (`news_intelligence.py::get_media_bias` + `bias_detector.py::get_media_bias_expanded`):
  bare `source_id`s (`portafolio`, `investing`) now match domain keys (`portafolio.co`, `investing.com`)
  via `domain.split(".")[0] == source`. Previously all DB-sourced articles resolved to `unknown`.
- **Day-inclusive week window** (`_compute_news_intelligence`): upper bound is exclusive of the next day
  so same-day intraday timestamps (08:45 COT) are kept.
- **Bootstrap article fallback** (`_get_all_articles` source #6): when CSVs + DB are unavailable (fresh
  clone), read the git-tracked `data/backups/features/news_articles.parquet` so the cards still render.

### 12.4 Article source of truth

`_get_all_articles` merges (in priority): GDELT/Google/Investing/Colombia CSVs (`data/news/*`, gitignored)
→ live `news_articles` DB → **tracked backup parquet** (fresh-clone fallback). A fresh clone with no CSVs
and an empty DB still renders news from the backup snapshot; forward pipelines/DAGs then supersede it.

### 12.5 Chat context (`app/api/analysis/chat/route.ts`)

The analysis chat now injects the richer week context it previously ignored: technical (trend/RSI/levels),
macro regime, top news-cluster themes, source-bias landscape, and upcoming events — in addition to
headline/sentiment/OHLCV/signals. The LLM call itself is unchanged (Azure→Anthropic when keys present;
config-gated placeholder otherwise, now showing the richer context).

### 12.6 Backfill CLI

```bash
# Inject news_intelligence + political_bias into published USD/COP weekly JSONs
python -m scripts.ops.patch_news_intelligence [--year 2026] [--force] [--dry-run]
# Gold/BTC: regenerate via the asset generator (writes the enriched blocks natively)
python scripts/pipeline/generate_asset_analysis.py --all-assets --year 2026
```
