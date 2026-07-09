# Rule: Backup, Recovery & Disaster Protocol

> Governs all backup, restore, and disaster recovery procedures for the USDCOP trading system.
> Priority: Data integrity > Trading continuity > Dashboard freshness
>
> Contract: CTR-BACKUP-001
> Version: 1.0.0
> Date: 2026-03-12

---

## Backup Architecture (3 Tiers)

| Tier | What | Where | Frequency | Retention |
|------|------|-------|-----------|-----------|
| T1: Live DB | OHLCV, macro, forecasts, signals, trades | PostgreSQL (TimescaleDB) | Realtime | Until disk full |
| T2: Seed Parquets | OHLCV daily + 5-min, macro, daily OHLCV | `seeds/latest/` + `data/backups/seeds/` | Daily (`l0_seed_backup` DAG, Mon-Fri 20:00 UTC) | Overwritten daily |
| T2b: Feature Parquets | 12 derived tables (news, analysis, H5, asset_daily, macro monthly/quarterly) | `data/backups/features/` | Daily (`l0_seed_backup`, best-effort task) | Overwritten daily |
| T3: Git LFS | Seed parquets committed to repo | Git LFS | On backfill completion | Full history |
| T4 (roadmap): MinIO buckets | Model artifacts, seed backups | `s3://99-common-trading-models`, `s3://99-common-trading-backups` | On model promotion | Bucket-managed |

> **MinIO status (2026-04-16)**: 11 buckets operational. Currently used by init-scripts for seed
> fallback reading (`s3://99-common-trading-models`, `s3://mlflow`). The backup DAG does NOT yet
> write model artifacts to MinIO — models persist on filesystem only. **Roadmap**: migrate model
> artifacts + weekly seed backups to MinIO for off-host durability. Mention as "bucket storage
> available for artifacts; currently models in filesystem, migration in roadmap."

### Backup Files

| File | Source DAG | Content |
|------|-----------|---------|
| `seeds/latest/usdcop_daily_ohlcv.parquet` | `l0_seed_backup` | Daily OHLCV (~3K rows, used by H1/H5 training) |
| `seeds/latest/usdcop_m5_ohlcv.parquet` | `l0_ohlcv_backfill` | 5-min bars (~90K rows, used by RL training) |
| `seeds/latest/fx_multi_m5_ohlcv.parquet` | `l0_ohlcv_backfill` | All 3 FX pairs 5-min (~266K rows) |
| `data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet` | `l0_macro_backfill` | 17 macro vars cleaned (~10K rows) |
| `data/backups/seeds/usdcop_m5_ohlcv_backup.parquet` | `l0_seed_backup` | Daily automated OHLCV backup |
| `data/backups/seeds/macro_indicators_daily_backup.parquet` | `l0_seed_backup` | Daily macro backup |
| `data/backups/seeds/backup_manifest.json` | `l0_seed_backup` | Hashes + timestamps for validation |
| `data/backups/features/*.parquet` (12 tables) | `l0_seed_backup` (`export_feature_data_backup` task) | **Derived-data backup** (added 2026-07-05): news_articles, news_feature_snapshots, weekly_analysis, daily_analysis, forecast_h5_{predictions,signals,executions,subtrades,paper_trading}, asset_daily_ohlcv, macro_indicators_{monthly,quarterly} |
| `data/backups/features/feature_backup_manifest.json` | `l0_seed_backup` | Rows / sha256 / latest-timestamp per table (CTR-L0-FEATURE-BACKUP-001) |

---

## Recovery Scenarios

### Scenario 1: Fresh Install (Empty DB)

```bash
# Step 1: Start containers — init scripts auto-seed from parquets
docker-compose up -d

# Step 2: Verify seeding worked
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT symbol, COUNT(*), MIN(time), MAX(time) FROM usdcop_m5_ohlcv GROUP BY symbol;"

# Step 3: If seeding failed (0 rows), manually backfill
airflow dags trigger core_l0_01_ohlcv_backfill
airflow dags trigger core_l0_03_macro_backfill

# Step 4: Run required migrations (cold boot applies these automatically via
#         init-scripts/26-restore-features.sh, which globs 04[3-9]/05[0-9]_*.sql)
for m in 043 044 045 046 048 049 051 054; do
  docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
    < database/migrations/${m}_*.sql
done

# Step 5: Regenerate dashboard data
python scripts/pipeline/generate_weekly_forecasts.py
python scripts/pipeline/train_and_export_smart_simple.py --phase both
```

### Scenario 2: Stale Data (Training Blocked)

```bash
# Symptom: "OHLCV freshness: latest=..., age=Xd, threshold=3d"
airflow dags trigger core_l0_01_ohlcv_backfill

# Symptom: "Macro freshness: latest=..., age=Xd, threshold=7d"
airflow dags trigger core_l0_03_macro_backfill

# Then re-trigger training
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training
```

### Scenario 3: Model Artifacts Stale (>10 Days)

```bash
# Symptom: H1-L5 logs "MODEL FRESHNESS WARNING: models are X days old"
# Check why L3 training hasn't run:
airflow dags list-runs -d forecast_h1_l3_weekly_training --limit 3

# Re-trigger training
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training
```

### Scenario 4: Seed Backup Corrupt / Missing

```bash
# Rebuild from DB
airflow dags trigger core_l0_05_seed_backup

# If DB is also empty, restore from Git LFS
git lfs pull
docker-compose restart usdcop-postgres-timescale
```

### Scenario 5: Full Disaster (DB Lost + Seeds Lost)

```bash
# 1. Restore Git LFS seeds
git lfs pull

# 2. Restart postgres (init scripts restore from seeds)
docker-compose restart usdcop-postgres-timescale

# 3. Backfill to current
airflow dags trigger core_l0_01_ohlcv_backfill
airflow dags trigger core_l0_03_macro_backfill

# 4. Run migrations (043-054; cold boot does this automatically)
for m in 043 044 045 046 048 049 051 054; do
  docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
    < database/migrations/${m}_*.sql
done

# 4b. Restore derived/feature data into empty tables (news/analysis/H5/asset_daily)
python -m scripts.ops.backup.feature_data_backup --mode restore --dir data/backups/features

# 5. Retrain models
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training

# 6. Regenerate dashboard data
python scripts/pipeline/generate_weekly_forecasts.py
python scripts/pipeline/train_and_export_smart_simple.py --phase both
python scripts/pipeline/generate_weekly_analysis.py
```

---

## Restore Priority

On container startup, `04-data-seeding.py` restores OHLCV + macro-daily in this order:

```
1. data/backups/seeds/*.parquet  (freshest — daily automated backup)
2. seeds/latest/*.parquet        (Git LFS — may be days/weeks old)
3. (manual backfill required)    (if both above are empty)
```

### Cold-boot restore = two planes (updated 2026-07-05)

A clean-slate `docker compose down -v` boot comes back with REAL data across two planes:

1. **DB plane** — the `data-seeder` container restores in stages:
   - schema: `init-scripts/26-restore-features.sh` applies feature migrations `043-054` (globs
     `04[3-9]/05[0-9]_*.sql`; pgvector 047 skipped) so every derived table EXISTS.
   - core data: OHLCV + macro-daily from `data/backups/seeds/` (priority list above).
   - **derived/feature data**: `feature_data_backup.py --mode restore` (baked into `Dockerfile.data-seeder`,
     runs after seeding) bulk-inserts the 12 feature tables from `data/backups/features/*.parquet`
     **only into EMPTY tables** — never clobbers live data. Forward pipelines (scrapers, backfill DAGs,
     H5 promotion) then fill the gap to today.
2. **Dashboard file plane** — `usdcop-trading-dashboard/public/data/**` (registry bundles, production
   JSON, analysis JSON) is host-persisted (bind-mounted, not in a wiped volume), so the web surface
   survives a volume wipe independently of the DB.

> This supersedes the old "News/Analysis data is re-obtained by scrapers, NOT restored from a dump"
> statement — derived data IS now restored from parquet on cold boot (then pipelines top it up).
> `26-restore-features.sh` itself only guarantees the schema; the DATA restore happens in the data-seeder.

---

## Verification Queries

After any recovery, run these checks:

```sql
-- OHLCV freshness (must be within 3 days for training)
SELECT symbol, MAX(time) as latest, COUNT(*) as rows
FROM usdcop_m5_ohlcv GROUP BY symbol ORDER BY symbol;

-- Macro freshness (must be within 7 days for training)
SELECT MAX(fecha) as latest, COUNT(*) as rows
FROM macro_indicators_daily;

-- H5 pipeline state
SELECT MAX(signal_date) as latest, COUNT(*) as signals FROM forecast_h5_signals;

-- H1 pipeline state
SELECT MAX(signal_date) as latest, COUNT(*) as signals FROM forecast_vol_targeting_signals;

-- News pipeline state
SELECT source_id, COUNT(*) FROM news_articles GROUP BY source_id;
SELECT MAX(snapshot_date) as latest, COUNT(*) FROM news_feature_snapshots;
SELECT MAX(digest_date) as latest, COUNT(*) FROM news_daily_digests;
-- Note: table is news_daily_digests (not news_digests), news_ingestion_log (singular, not plural)

-- Analysis state
SELECT iso_year, iso_week FROM weekly_analysis ORDER BY 1 DESC, 2 DESC LIMIT 3;
```

---

## Backup DAG Schedule

| DAG | Schedule | Action |
|-----|----------|--------|
| `core_l0_05_seed_backup` | Mon-Fri 20:00 UTC | OHLCV + macro + daily OHLCV to parquet + **12 feature tables** to `data/backups/features/` + best-effort MinIO upload |
| `core_l0_01_ohlcv_backfill` | Manual / on failure | Full OHLCV gap-fill from TwelveData |
| `core_l0_03_macro_backfill` | Sun 04:00 UTC | Full macro extraction + MACRO_DAILY_CLEAN regen |

---

## DO NOT

- Do NOT delete seed parquets without a backup — they are the last-resort restore
- Do NOT skip verification queries after recovery — silent data loss is worse than downtime
- Do NOT manually edit parquet files — always regenerate via scripts or DAGs
- Do NOT restore from Git LFS seeds if DB has newer data — LFS seeds may be weeks old
- Do NOT run training DAGs without checking data freshness first — use the gates
- Do NOT modify migration files after they've been applied — create new migrations
