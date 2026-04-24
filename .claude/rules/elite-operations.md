# Rule: Elite Operations Rulebook

> Comprehensive operational rulebook for the USDCOP trading system.
> Covers data freshness, automated recovery, schedule coordination, monitoring,
> and dashboard data completeness.
>
> Contract: CTR-OPS-001
> Version: 2.0.0
> Date: 2026-04-06 (updated for Smart Simple v2.0 with regime gate)

---

## Operational Priorities

```
1. DATA INTEGRITY    — Never train on stale data, never trade on stale signals
2. REGIME AWARENESS  — Regime gate (Hurst) must run before every trade decision
3. NO FAKE DATA      — Dashboard must NEVER show fallback/simulated prices
4. TRADING CONTINUITY — Signals fire Mon-Fri; gate decides whether to trade
5. DASHBOARD FRESHNESS — Core 4 pages current. Also: /hub, /execution, /login
6. OBSERVABILITY     — Know when something fails before it impacts trading
```

### v2.0 Architecture (2026-04-06)

```
Strategy: Smart Simple v2.0 (Ridge+BR+XGB + Regime Gate + Effective HS + DL + CB)
Production: +0.61% YTD (1 trade, gate blocked 13/14 mean-reverting weeks)
Backtest:   +25.63% 2025 (Sharpe 3.35, p=0.006, 34 trades)
Key files:  src/forecasting/regime_gate.py, dynamic_leverage.py, momentum_signal.py
DAG utils:  airflow/dags/utils/regime_gate_live.py, dynamic_leverage_live.py
Config:     config/execution/smart_simple_v1.yaml (regime_gate + dynamic_leverage sections)
Migration:  048_regime_gate_columns.sql
```

---

## Data Freshness Enforcement

### Thresholds (Hard)

| Data | Max Staleness | Gate Type | Consequence |
|------|---------------|-----------|-------------|
| OHLCV 5-min (`usdcop_m5_ohlcv`) | 3 days | BLOCKING | Training halted |
| Macro (`macro_indicators_daily`) | 7 days | BLOCKING | Training halted |
| Model .pkl artifacts | 10 days | WARNING | Inference continues with warning |
| Daily OHLCV seed parquet | 3 days | NONE | Stale features in training |
| MACRO_DAILY_CLEAN.parquet | 7 days | NONE | Stale macro in training |
| News articles | 24 hours | NONE | Analysis lacks fresh context |

### Gate Implementations

| DAG | Gate | Utility | Location |
|-----|------|---------|----------|
| H1-L3, H5-L3 | Pre-training data freshness | `validate_training_data_freshness()` | `airflow/dags/utils/data_quality.py` |
| H1-L5 | Model freshness warning | `check_model_freshness()` | Same |
| H5-L5 | Training completion sensor | `ExternalTaskSensor` | DAG file |
| Analysis L8 | News pipeline sensor | `ExternalTaskSensor` | DAG file |

---

## Schedule Coordination (UTC)

### Collision-Free Timeline

**Sunday (Maintenance Day)**:
```
04:00  Macro Backfill (moved from 06:00 to avoid H5-L3 collision)
       → Also regenerates MACRO_DAILY_CLEAN.parquet
05:00  Macro backfill finishes (~60 min)
06:00  H1-L3 Training (data freshness gate → 9 models)
06:30  H5-L3 Training (data freshness gate → Ridge + BR)
07:30  Training finishes
20:00  Seed Backup (moved from 18:00 to avoid macro update collision)
       → Also refreshes daily OHLCV seed
```

**Monday-Friday**:
```
13:00-17:00  OHLCV realtime (every 5 min) + Macro update (hourly)
13:00-18:00  core_watchdog hourly (8-13 COT): auto-heals stale data, forecasting, analysis
07:00,12:00,18:00  News daily pipeline (3x UTC = 02:00,07:00,13:00 COT)
13:15 Mon     H5-L5 Signal (ExternalTaskSensor on H5-L3)
14:00 Mon     forecast_weekly_generation: regenerates /forecasting CSV + 76 PNGs (~7 min)
18:00         H1-L5 Inference (model freshness check)
18:30         H1-L5 Vol-Targeting
18:35-22:00   H1-L7 Executor (narrowed from 13:00-19:00)
19:00         Analysis L8 (ExternalTaskSensor on News)
20:00         Seed Backup
```

### Why Narrowed H1-L7 Schedule
Previously: `*/5 13-19 * * 1-5` (8:00-14:00 COT, ~72 runs/day)
Now: `*/5 18-22 * * 1-5` (13:00-17:00 COT, ~48 runs/day)
Rationale: H1 entry happens at 13:30+ COT. Running from 08:00 wastes 60+ scheduler cycles.

---

## Weekend Handling

- **Saturday**: No OHLCV, no macro updates, no trading. News alert monitor may run.
- **Sunday**: Maintenance day. Macro backfill covers Fri-Sun gap. Training uses Friday's data.
- **Friday 12:50 COT**: H5 weekly position closes (market order). H1 trailing stop may still be active.
- **Colombian holidays**: `TradingCalendar` in OHLCV realtime DAG skips non-trading days.
- **US holidays**: Macro sources may be empty — 7-day macro threshold accommodates this.

---

## Automated Recovery Procedures

### When Training Fails (Data Gate)

```
SYMPTOM: H1-L3 or H5-L3 fails with "OHLCV freshness: age=Xd, threshold=3d"

FIX:
1. airflow dags trigger core_l0_01_ohlcv_backfill
2. Wait for completion
3. airflow dags trigger forecast_h1_l3_weekly_training
4. airflow dags trigger forecast_h5_l3_weekly_training
```

### When Seed Restore Fails on Startup

```
SYMPTOM: docker-compose up → DB has 0 rows in usdcop_m5_ohlcv

FIX:
1. Check seed files exist: ls seeds/latest/*.parquet
2. If missing: git lfs pull
3. Restart: docker-compose restart usdcop-postgres-timescale
4. If still empty: manual backfill
   airflow dags trigger core_l0_01_ohlcv_backfill
   airflow dags trigger core_l0_03_macro_backfill
```

### When Models Go Stale (>10 Days)

```
SYMPTOM: H1-L5 logs "MODEL FRESHNESS WARNING: models are X days old"

FIX:
1. Check L3 training logs in Airflow UI
2. If data gate blocked training → fix data first (see above)
3. Re-trigger: airflow dags trigger forecast_h1_l3_weekly_training
```

### When News Pipeline Fails

```
SYMPTOM: Analysis L8 sensor times out waiting for news_daily_pipeline

FIX:
1. Check adapter health in news_daily_pipeline logs
2. Common issues: GDELT rate limit, Investing.com CloudFlare block, feedparser not installed
3. If feedparser missing: docker exec usdcop-airflow-scheduler python -m pip install feedparser
4. Re-trigger: airflow dags trigger news_daily_pipeline
5. Analysis will run on next schedule (soft_fail=True on sensor)

ACTIVE SOURCES (as of 2026-03-13):
- Investing.com: Working (search API)
- Portafolio: Working (RSS + feedparser)
- LaRepublica: Working but may return 0 articles if no recent matches
- GDELT: Rate limited (0 articles typical, needs 8s+ between requests)
- NewsAPI: Disabled (no API key configured)
- Google News: Adapter file does not exist (never implemented)
```

---

## Dashboard Data Completeness

### Required Files Per Page

| Page | Files Needed | Generated By | Check |
|------|-------------|--------------|-------|
| `/forecasting` | `bi_dashboard_unified.csv` + 63+ PNGs | `generate_weekly_forecasts.py` | CSV exists with >500 rows |
| `/dashboard` | `summary_2025.json`, `approval_state.json`, `trades/*_2025.json` | `--phase backtest` | approval_state has 5 gates |
| `/production` | `summary.json`, `trades/*.json` | `--phase production` (after approval) | summary.json has strategy_id |
| `/analysis` | `analysis_index.json`, `weekly_YYYY_WXX.json` | `generate_weekly_analysis.py` | index has >=1 week |

### Regeneration Commands

```bash
# /forecasting page
python scripts/generate_weekly_forecasts.py

# /dashboard + /production pages
python scripts/train_and_export_smart_simple.py --phase both

# /analysis page
python scripts/generate_weekly_analysis.py
```

---

## DB Migration Checklist

Before first production use, ALL migrations must be applied:

```sql
-- Check applied migrations (no formal tracking — check tables exist)
SELECT tablename FROM pg_tables WHERE schemaname='public'
AND tablename IN (
    'forecast_h5_predictions', 'forecast_h5_signals',
    'news_articles', 'news_sources',
    'weekly_analysis', 'daily_analysis', 'macro_variable_snapshots'
) ORDER BY tablename;
```

If any are missing, apply from `database/migrations/`:
```bash
for m in 043 044 045 046; do
  docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
    < database/migrations/${m}_*.sql
done
```

---

## Budget & Rate Limits

| Service | Limit | Enforcement |
|---------|-------|-------------|
| Azure OpenAI (Analysis) | $1/day, $15/month | `weekly_analysis_ssot.yaml` |
| Anthropic (Fallback) | Same budget | LLMClient strategy pattern |
| GDELT API | 6s min delay between requests | `time.sleep(6)` in adapter |
| NewsAPI | 100 requests/day (free tier) | API key gated |
| TwelveData | 8 API keys, rotation | `api_key_rotation` in DAG |
| Investing.com | 3-6s delay + session rotation | CloudScraper in adapter |

---

## Verification Queries

Run after any recovery, migration, or infrastructure change:

```sql
-- 1. OHLCV freshness
SELECT symbol, MAX(time) as latest, COUNT(*) FROM usdcop_m5_ohlcv
GROUP BY symbol ORDER BY symbol;

-- 2. Macro freshness
SELECT MAX(fecha) as latest, COUNT(*) FROM macro_indicators_daily;

-- 3. H5 pipeline state
SELECT COUNT(*) as predictions FROM forecast_h5_predictions;
SELECT COUNT(*) as signals FROM forecast_h5_signals;
SELECT COUNT(*) as executions FROM forecast_h5_executions;

-- 4. News pipeline state
SELECT source_id, COUNT(*) FROM news_articles GROUP BY source_id;
SELECT COUNT(*) FROM news_feature_snapshots;

-- 5. Analysis state
SELECT iso_year, iso_week FROM weekly_analysis ORDER BY iso_year DESC, iso_week DESC LIMIT 5;
SELECT COUNT(*) FROM daily_analysis;
```

---

## Container Dependencies (Baked into Image — fixed 2026-04-16)

All required packages now live in `docker/Dockerfile.airflow-ml` and persist across rebuilds:

```dockerfile
RUN pip install --no-cache-dir \
    openai==1.54.0 \
    anthropic==0.39.0 \
    langgraph==0.2.50 \
    feedparser==6.0.11 \
    vaderSentiment==3.3.2
```

Verified imports baked into the Dockerfile verification step. `pip install` at runtime no longer
needed. To update, edit the Dockerfile and rebuild:

```bash
docker compose -f docker-compose.compact.yml build airflow-scheduler
docker compose -f docker-compose.compact.yml up -d --force-recreate airflow-scheduler airflow-webserver
```

## Dashboard Bind-Mount (required for DAG JSON output to reach Next.js frontend)

The `analysis_l8_daily_generation` DAG and `forecast_weekly_generation` DAG write to
`usdcop-trading-dashboard/public/...`. Scheduler and webserver must bind-mount the host path
so the Next.js frontend (also on host) sees the generated files:

```yaml
# docker-compose.compact.yml (airflow-scheduler + airflow-webserver)
volumes:
  - ./usdcop-trading-dashboard/public:/opt/airflow/usdcop-trading-dashboard/public:rw
  - ./scripts:/opt/airflow/scripts:ro
```

The `scripts:/opt/airflow/scripts` mount is required for `core_watchdog` auto-heal subprocess
calls to resolve `/opt/airflow/scripts/generate_weekly_{forecasts,analysis}.py`.

**Permissions**: The container runs as UID 50000 (airflow). Host host-side paths must be
writable by group `root` (GID 0) OR `chmod -R 777` on `usdcop-trading-dashboard/public/data/analysis/`
and `usdcop-trading-dashboard/public/forecasting/` for writes to succeed.

---

## DO NOT

- Do NOT ignore data freshness gates — they exist to prevent garbage-in-garbage-out
- Do NOT run training DAGs manually without checking data freshness first
- Do NOT bypass ExternalTaskSensors by removing them — they enforce pipeline ordering
- Do NOT schedule new DAGs without checking the collision-free timeline above
- Do NOT delete seed parquets — they are the last-resort restore mechanism
- Do NOT modify applied migration SQL files — create new numbered migrations
- Do NOT exceed LLM budget ($1/day) — check `data/cache/analysis/` for cached results first
- Do NOT trade on signals older than 24 hours — staleness invalidates directional predictions
