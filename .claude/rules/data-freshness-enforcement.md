# Rule: Data Freshness Enforcement & Automated Recovery

> Governs data freshness thresholds, automated recovery procedures, and monitoring rules.
> This rule ensures the trading system never trains on stale data or misses trading signals.
>
> Contract: CTR-DQ-OPS-001
> Version: 1.0.0
> Date: 2026-03-12

---

## Data Freshness Thresholds

| Data Source | Table | Max Staleness | Check Method | Consequence if Stale |
|-------------|-------|---------------|--------------|---------------------|
| OHLCV 5-min | `usdcop_m5_ohlcv` | 3 days | `validate_training_data_freshness()` | Training BLOCKED |
| Macro daily | `macro_indicators_daily` | 7 days | `validate_training_data_freshness()` | Training BLOCKED |
| H1 model .pkl | `outputs/forecasting/h1_daily_models/latest/` | 10 days | `check_model_freshness()` | WARNING (not blocked) |
| H5 model .pkl | `outputs/forecasting/h5_weekly_models/latest/` | 10 days | `check_model_freshness()` | WARNING (not blocked) |
| Daily OHLCV seed | `seeds/latest/usdcop_daily_ohlcv.parquet` | 3 days | File mtime check | Training uses stale features |
| MACRO_DAILY_CLEAN | `data/pipeline/04_cleaning/output/` | 7 days | File mtime check | Training uses stale macro |
| News articles | `news_articles` | 24 hours | `MAX(published_at)` | Analysis lacks fresh context |
| News features | `news_feature_snapshots` | 24 hours | `MAX(snapshot_date)` | ML news features stale |

### Why These Thresholds

- **OHLCV 3 days**: Market closes Friday 12:55 COT. Sunday training (01:00-01:30 COT) means latest data is Friday = 2 days old. 3-day threshold allows for this gap. If >3 days, realtime DAG or backfill failed.
- **Macro 7 days**: Some macro variables update weekly (FEDFUNDS, CPI). 7-day threshold prevents blocking on legitimately slow-updating variables.
- **Models 10 days**: Models retrain every Sunday. If >10 days old, two consecutive Sunday trainings failed. Soft warning lets inference continue with slightly stale models rather than halting trading.

---

## Automated Freshness Gates

### Pre-Training Gate (BLOCKING)

Implemented in `airflow/dags/utils/data_quality.py` as `validate_training_data_freshness()`.

Both H1-L3 and H5-L3 weekly training DAGs call this as the FIRST task. If it fails, training does NOT proceed.

```
validate_data_freshness >> load_and_build_features >> train_models >> ...
```

Recovery when gate fails:
1. Check realtime DAG logs: `airflow dags list-runs -d core_l0_02_ohlcv_realtime --limit 5`
2. If realtime hasn't run: check trading calendar, circuit breaker
3. If realtime ran but data is still stale: trigger backfill
4. After backfill: re-trigger training DAG

### Pre-Inference Check (WARNING ONLY)

Implemented as `check_model_freshness()` in the same utility module.

H1-L5 daily inference DAG calls this before loading models. Logs a WARNING but does NOT block inference.

Recovery when warning fires:
1. Check L3 training DAG logs
2. Re-trigger training: `airflow dags trigger forecast_h1_l3_weekly_training`
3. After training completes, next inference will use fresh models

---

## Recovery Procedures

### OHLCV Stale (>3 days)

```bash
# 1. Check what happened
airflow dags list-runs -d core_l0_02_ohlcv_realtime --limit 10

# 2. Trigger backfill (fetches gap from last DB row to today)
airflow dags trigger core_l0_01_ohlcv_backfill

# 3. Verify
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT symbol, MAX(time) as latest FROM usdcop_m5_ohlcv GROUP BY symbol;"
```

### Macro Stale (>7 days)

```bash
# 1. Trigger full macro backfill (7 sources)
airflow dags trigger core_l0_03_macro_backfill

# 2. Verify
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT MAX(fecha) as latest FROM macro_indicators_daily;"

# 3. The backfill also regenerates MACRO_DAILY_CLEAN.parquet (Change 8)
```

### Models Stale (>10 days)

```bash
# 1. Re-trigger training
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training

# 2. Verify models exist and are fresh
ls -la /opt/airflow/outputs/forecasting/h1_daily_models/latest/*.pkl
ls -la /opt/airflow/outputs/forecasting/h5_weekly_models/latest/*.pkl
```

### News Pipeline Stale (>24 hours)

```bash
# 1. Check if pipeline ran
docker exec usdcop-airflow-scheduler airflow tasks states-for-dag-run news_daily_pipeline "$(date -u +%Y-%m-%dT12:00:00+00:00 -d yesterday)"

# 2. Check if feedparser is installed (needed for Portafolio/LaRepublica)
docker exec usdcop-airflow-scheduler python -c "import feedparser; print('OK')" || \
  docker exec usdcop-airflow-scheduler python -m pip install feedparser

# 3. Re-trigger pipeline
airflow dags trigger news_daily_pipeline

# 4. Verify
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT source_id, COUNT(*) FROM news_articles GROUP BY source_id;"
```

### Seed Backup Failed

```bash
# 1. Re-trigger seed backup
airflow dags trigger core_l0_05_seed_backup

# 2. Verify backup manifest
cat /opt/airflow/data/backups/seeds/backup_manifest.json

# 3. Verify daily OHLCV seed was refreshed
python -c "import pandas as pd; df=pd.read_parquet('seeds/latest/usdcop_daily_ohlcv.parquet'); print(f'Rows: {len(df)}, Latest: {df.time.max()}')"
```

### Full System Recovery (DB empty after restart)

```bash
# 1. Docker init-scripts should auto-seed from parquets
docker-compose restart usdcop-postgres-timescale

# 2. If seeding fails (0 rows), manually restore
airflow dags trigger core_l0_01_ohlcv_backfill
airflow dags trigger core_l0_03_macro_backfill

# 3. Run pending migrations
for m in 043 044 045 046; do
  docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
    < database/migrations/${m}_*.sql
done

# 4. Retrain models
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training

# 5. Regenerate dashboard data
python scripts/generate_weekly_forecasts.py
python scripts/train_and_export_smart_simple.py --phase both
```

---

## Schedule Coordination (Updated 2026-03-12)

### Sunday (Maintenance Day)

| Time (UTC) | Time (COT) | DAG | Action |
|------------|------------|-----|--------|
| 03:00 | 22:00 Sat | `news_maintenance` | Cleanup logs, vacuum DB |
| 04:00 | 23:00 Sat | `core_l0_03_macro_backfill` | Full macro extraction + MACRO_DAILY_CLEAN regen |
| ~05:00 | ~00:00 | (finish) | Macro backfill complete |
| 06:00 | 01:00 | `forecast_h1_l3_weekly_training` | Data freshness gate, then train 9 models |
| 06:30 | 01:30 | `forecast_h5_l3_weekly_training` | Data freshness gate, then train Ridge+BR |
| ~07:30 | ~02:30 | (finish) | Both trainings complete |
| 20:00 | 15:00 | `core_l0_05_seed_backup` | OHLCV + macro + daily OHLCV seed backup |

### Monday

| Time (UTC) | Time (COT) | DAG | Action |
|------------|------------|-----|--------|
| 07:00 | 02:00 | `forecast_l0_daily_data` | Daily OHLCV from Investing.com |
| 08:00 | 03:00 | `news_weekly_digest` | Previous week text digest |
| 07:00 | 02:00 | `news_daily_pipeline` | 1st news ingestion run |
| 12:00 | 07:00 | `news_daily_pipeline` | 2nd news ingestion run |
| 13:15 | 08:15 | `forecast_h5_l5_weekly_signal` | H5 signal (waits for H5-L3 via sensor) |
| 13:45 | 08:45 | `forecast_h5_l5_vol_targeting` | H5 vol-targeting |
| 14:00 | 09:00 | `forecast_h5_l7_multiday_executor` | H5 entry |
| 18:00 | 13:00 | `forecast_h1_l5_daily_inference` | H1 signal (model freshness check) |
| 18:00 | 13:00 | `news_daily_pipeline` | 3rd news ingestion run |
| 18:30 | 13:30 | `forecast_h1_l5_vol_targeting` | H1 vol-targeting |
| 18:35-22:00 | 13:35-17:00 | `forecast_h1_l7_smart_executor` | H1 trailing stop (narrowed window) |

### Tuesday-Thursday

| Time (UTC) | Time (COT) | DAG | Action |
|------------|------------|-----|--------|
| 13:00-17:00 | 08:00-12:00 | `core_l0_02_ohlcv_realtime` | 5-min bars every 5 min |
| 13:00-17:00 | 08:00-12:00 | `core_l0_04_macro_update` | Macro hourly |
| 07:00,12:00,18:00 | 02:00,07:00,13:00 | `news_daily_pipeline` | 3x daily news |
| 18:00 | 13:00 | `forecast_h1_l5_daily_inference` | H1 daily signal |
| 18:30 | 13:30 | `forecast_h1_l5_vol_targeting` | H1 vol-targeting |
| 18:35-22:00 | 13:35-17:00 | `forecast_h1_l7_smart_executor` | H1 trailing stop |
| 19:00 | 14:00 | `analysis_l8_daily_generation` | Daily analysis (waits for news) |
| 20:00 | 15:00 | `core_l0_05_seed_backup` | Daily seed backup |

### Friday (Same as Tue-Thu plus)

| Time (UTC) | Time (COT) | DAG | Action |
|------------|------------|-----|--------|
| 17:50 | 12:50 | `forecast_h5_l7_multiday_executor` | H5 Friday close |
| 19:00 | 14:00 | `analysis_l8_daily_generation` | Daily + weekly summary |
| 19:30 | 14:30 | `forecast_h5_l6_weekly_monitor` | H5 weekly evaluation |
| 00:00 Sat | 19:00 | `forecast_h1_l6_paper_monitor` | H1 paper trading log |

### Weekend

- **Saturday**: No trading, no data. News alert monitor may run.
- **Sunday**: Maintenance day (see Sunday schedule above). No OHLCV or macro data expected.

---

## Monitoring Checklist

### Daily Checks (Mon-Fri, after market close ~13:00 COT)

- [ ] OHLCV realtime ran: `SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'` = today
- [ ] Macro update ran: `SELECT MAX(fecha) FROM macro_indicators_daily` = today or yesterday
- [ ] H1 inference produced signal (Mon-Fri only)
- [ ] H1 executor entered/trailed (if signal was SHORT)
- [ ] News pipeline completed 3x

### Weekly Checks (Sunday evening / Monday morning)

- [ ] H1-L3 training completed (check Airflow UI)
- [ ] H5-L3 training completed
- [ ] H5-L5 signal generated on Monday
- [ ] Seed backup is fresh: check `backup_manifest.json` timestamp
- [ ] MACRO_DAILY_CLEAN.parquet regenerated (check file mtime)
- [ ] Daily OHLCV seed refreshed

---

## DB Migration Tracking

| Migration | Tables Created | Status | Required For |
|-----------|---------------|--------|-------------|
| 025_forecast_experiments | forecast experiments | Applied | Forecasting |
| 038_nrt_tables | inference_ready_nrt | Applied | RL inference |
| 040_multi_pair | OHLCV indexes | Applied | Multi-pair |
| 041_forecast_vol_targeting | forecast_vol_targeting_signals | Applied | H1 pipeline |
| 042_forecast_executions | forecast_paper_trading | Applied | H1 paper trading |
| **043_forecast_h5_tables** | 5 H5 tables + 2 views | **Applied 2026-03-12** | H5 pipeline |
| **044_smart_simple_columns** | H5 confidence + stops cols | **Applied 2026-03-12** | Smart Simple v1.1 |
| **045_newsengine_initial** | 8 News Engine tables | **Applied 2026-03-12** | News pipeline |
| **046_weekly_analysis_tables** | 4 Analysis tables | **Applied 2026-03-12** | Analysis module |
| 047_pgvector_embeddings | pgvector extension | Not applied | Future (embeddings) |

---

## DO NOT

- Do NOT ignore data freshness warnings — they indicate L0 pipeline failures
- Do NOT train models on stale data (>3 days OHLCV or >7 days macro) — predictions will be outdated
- Do NOT skip seed backup — it's the primary restore mechanism on container restart
- Do NOT delete MACRO_DAILY_CLEAN.parquet — it's used by both H1 and H5 training as fallback
- Do NOT run L5 inference if L3 training hasn't completed this week — use ExternalTaskSensor
- Do NOT assume data is fresh after a weekend — always verify before Monday trading
- Do NOT modify migration files after they've been applied — create new migrations instead
