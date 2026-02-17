# Rule: L0 Data Layer Governance

> Governs ALL Layer 0 data pipelines: OHLCV (3 FX pairs), Macro (40 variables).
> This is the complete L0 reference.

---

## L0 DAG Inventory (5 DAGs, 5 files)

| # | DAG ID | File | Type | Schedule | Scope |
|---|--------|------|------|----------|-------|
| 1 | `core_l0_01_ohlcv_backfill` | `l0_ohlcv_backfill.py` | OHLCV seed restore + gap-fill + seed export | Manual | COP + MXN + BRL |
| 2 | `core_l0_02_ohlcv_realtime` | `l0_ohlcv_realtime.py` | OHLCV realtime | `*/5 13-17 * * 1-5` | COP + MXN + BRL |
| 3 | `core_l0_03_macro_backfill` | `l0_macro_backfill.py` | Macro backfill + seed export | Weekly Sun 6:00 UTC / Manual | Full history |
| 4 | `core_l0_04_macro_update` | `l0_macro_update.py` | Macro realtime | `0 13-17 * * 1-5` (hourly) | 40 vars, 7 sources |
| 5 | `core_l0_05_seed_backup` | `l0_seed_backup.py` | DB → parquet backup for startup restore | Daily `0 18 * * *` (13:00 COT) | OHLCV + macro |

### Backup Strategy
Two-tier backup system:
- **Automated daily** (#5): `l0_seed_backup` dumps OHLCV + macro tables to `data/backups/seeds/*.parquet` every day after market close. These are the **freshest** backups, read first by init-scripts on `docker-compose up`.
- **Seed export on backfill**: OHLCV backfill (#1) exports updated seed parquets to `seeds/latest/`. Macro backfill (#3) exports 9 MASTER files. These are committed to Git as the baseline restore.
- **Restore priority**: Daily backup parquet (if exists) → Git LFS seed parquet → MinIO → legacy CSV.
- Restore is handled by init-scripts (`04-data-seeding.py`, `04-seed-from-minio.py`) on container startup.

---

## PART A: OHLCV Pipeline

### Golden Rule: ALL Timestamps = America/Bogota (COT)

**No exceptions.** Every timestamp in every seed file, parquet, and DB row must be in
`America/Bogota` timezone. The session window is **8:00-12:55 COT, Monday-Friday**.

If you receive data in UTC, convert it before storing:
```python
ts.dt.tz_localize('UTC').dt.tz_convert('America/Bogota')
```

### Supported Pairs

| Pair | Table Symbol | Seed File | Data Source | Price Range |
|------|-------------|-----------|-------------|-------------|
| USD/COP | `USD/COP` | `seeds/latest/usdcop_m5_ohlcv.parquet` | TwelveData | 3,000-6,000 |
| USD/MXN | `USD/MXN` | `seeds/latest/usdmxn_m5_ohlcv.parquet` | Dukascopy | 10-30 |
| USD/BRL | `USD/BRL` | `seeds/latest/usdbrl_m5_ohlcv.parquet` | TwelveData | 3-8 |
| ALL | -- | `seeds/latest/fx_multi_m5_ohlcv.parquet` | Unified | -- |

### Timezone Conversion Reference

| Source | Raw TZ | Conversion to COT |
|--------|--------|--------------------|
| TwelveData (COP/MXN) | `timezone=America/Bogota` | Already COT, use directly |
| TwelveData (BRL) | `timezone=UTC` | `tz_localize('UTC').tz_convert('America/Bogota')` |
| Dukascopy | UTC (naive, epoch ms) | `tz_localize('UTC').tz_convert('America/Bogota')` |
| PostgreSQL TIMESTAMPTZ | UTC internally | `AT TIME ZONE 'America/Bogota'` in SQL |

### BRL API Quirk (CRITICAL)
TwelveData returns **incomplete data** for USD/BRL when `timezone=America/Bogota` is used.
Always fetch BRL with `timezone=UTC`, then convert to COT before storing. This applies to:
- `l0_ohlcv_realtime.py` (realtime DAG)
- `l0_ohlcv_backfill.py` (backfill DAG)
- `build_unified_fx_seed.py` (seed builder)

### Historical Bug (Fixed 2026-02-12)
USDCOP seed had UTC timestamps mislabeled as `America/Bogota` (used `tz_localize` instead
of `tz_convert`). Bars showed hours 13-17 instead of 8-12. Fixed in `build_unified_fx_seed.py`
by stripping the wrong tz, then: `tz_localize('UTC').tz_convert('America/Bogota')`.

### OHLCV Database Contract

#### Table: `usdcop_m5_ohlcv`
- **Name kept as-is** (renaming = 50+ file changes across codebase)
- `PRIMARY KEY (time, symbol)` -- supports multi-pair natively
- Indexes: `idx_ohlcv_symbol`, `idx_ohlcv_symbol_time` (migration 040)
- Views: `fx_latest_bar`, `fx_daily_bar_counts` (monitoring)

#### Schema
```sql
time        TIMESTAMPTZ     -- America/Bogota timezone
symbol      VARCHAR(20)     -- 'USD/COP', 'USD/MXN', 'USD/BRL'
open        DOUBLE PRECISION
high        DOUBLE PRECISION
low         DOUBLE PRECISION
close       DOUBLE PRECISION
volume      DOUBLE PRECISION  -- 0 for COP/BRL (no volume data)
source      VARCHAR(50)       -- 'twelvedata', 'twelvedata_backfill', 'twelvedata_multi', 'dukascopy'
```

#### UPSERT Pattern (all OHLCV DAGs must use)
```sql
INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source)
VALUES %s
ON CONFLICT (time, symbol) DO UPDATE SET
    volume = EXCLUDED.volume,
    source = EXCLUDED.source,
    updated_at = NOW()
```

### OHLCV Seed File Contract

#### Standard Schema (all parquets)
```
[time, symbol, open, high, low, close, volume]
```
- `time`: `datetime64[ns, America/Bogota]` (tz-aware)
- `symbol`: String (`"USD/COP"`, `"USD/MXN"`, `"USD/BRL"`)
- Volume: 0 where unavailable

#### Validation Rules (enforced by `build_unified_fx_seed.py`)
1. All hours in `[8, 9, 10, 11, 12]` COT
2. No weekends (dayofweek < 5)
3. No duplicate `(time, symbol)` pairs
4. No NaN in OHLC columns
5. `high >= low`, `high >= open`, `high >= close`
6. Prices within expected range per pair
7. Median bars/day approximately 60 (session = 5h / 5min)

#### Regeneration
```bash
python scripts/build_unified_fx_seed.py
```
Reads raw seeds, fixes timezones, filters session, validates, saves 4 parquets.

### OHLCV DAG Details

#### DAG 1: `l0_ohlcv_backfill.py` (`core_l0_01_ohlcv_backfill`)
- All 3 pairs by default, overridable via `dag_run.conf: {"symbols": ["USD/MXN"]}`
- Seed restore when DB empty (per-symbol seed parquets)
- Comprehensive gap detection from MIN to MAX date in DB
- Per-symbol TwelveData timezone (BRL needs UTC, with UTC->COT conversion)
- API key rotation across 8 keys
- **Exports updated seed files** after backfill (replaces formal backup)
- Flow: `health_check -> [process_usd_cop -> process_usd_mxn -> process_usd_brl] -> export_seeds -> validate`

#### DAG 2: `l0_ohlcv_realtime.py` (`core_l0_02_ohlcv_realtime`)
- Parallel tasks: COP, MXN, BRL (failure isolation per symbol)
- `TradingCalendar` for holiday validation (skip non-trading days)
- `CircuitBreaker` per-symbol API protection
- Per-symbol TwelveData timezone: `{'USD/COP': 'America/Bogota', 'USD/MXN': 'America/Bogota', 'USD/BRL': 'UTC'}`
- API key rotation across 8 keys
- Flow: `check_trading_day -> [start -> [fetch_cop, fetch_mxn, fetch_brl] -> validate -> end | skip -> end]`

#### DAG 5: `l0_seed_backup.py` (`core_l0_05_seed_backup`)
- **Schedule**: `0 18 * * *` (daily 18:00 UTC = 13:00 COT, after market close)
- **Purpose**: Full table dump of OHLCV + macro to parquet for startup restore
- **Output**: `data/backups/seeds/{usdcop_m5_ohlcv_backup,macro_indicators_daily_backup}.parquet` + `backup_manifest.json`
- **Atomic writes**: Write to `.tmp` then rename to prevent partial reads
- **Manifest**: JSON with timestamp, row counts, file hashes for validation
- **Contract**: CTR-L0-SEED-BACKUP-001
- Flow: `health_check -> export_ohlcv_backup -> export_macro_backup -> write_manifest -> validate_backups`

### OHLCV Usage Commands
```bash
# Realtime runs automatically on schedule (every 5 min, market hours)
# core_l0_02_ohlcv_realtime: COP + MXN + BRL with holiday + circuit breaker

# Backfill all 3 pairs (default)
airflow dags trigger core_l0_01_ohlcv_backfill

# Backfill a specific pair
airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"symbols": ["USD/MXN"]}'

# Legacy single-symbol syntax also supported
airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"symbol": "USD/BRL"}'

# Force gap detection (skip seed restore)
airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"force_backfill": true}'
```

---

## PART B: Macro Pipeline

### Architecture: 4-Table Design (CTR-L0-4TABLE-001)

Macro data is split by frequency into separate tables:

| Table | Frequency | Variables | Date Column | Example Variables |
|-------|-----------|-----------|-------------|-------------------|
| `macro_indicators_daily` | Daily | 18 | `fecha` | DXY, VIX, UST10Y, UST2Y, IBR, TPM, EMBI |
| `macro_indicators_monthly` | Monthly | 8 | `fecha` | FEDFUNDS, CPI, UNEMPLOYMENT |
| `macro_indicators_quarterly` | Quarterly | 4 | `fecha` | GDP, BOP |

### Data Sources (7 extractors via `ExtractorRegistry`)

| Source | Variables | API/Method |
|--------|-----------|------------|
| FRED | FEDFUNDS, CPI, unemployment, treasury yields | FRED API |
| Investing.com | DXY, VIX, commodities | Scraping |
| BanRep | IBR, TPM, EMBI_COL | BanRep API |
| BCRP | Peru macro data | BCRP API |
| Fedesarrollo | Colombia consumer confidence | Scraping |
| DANE | Colombia trade data | DANE API |
| BanRep BOP | Balance of payments | BanRep API |

### Critical Variables (must have `is_complete = true`)
`dxy`, `vix`, `ust10y`, `ust2y`, `ibr`, `tpm`, `embi_col`

### Macro Seed Files (9 MASTER files)
Located in `data/pipeline/04_cleaning/output/`:
```
MACRO_DAILY_MASTER.csv      MACRO_DAILY_MASTER.parquet      MACRO_DAILY_MASTER.xlsx
MACRO_MONTHLY_MASTER.csv    MACRO_MONTHLY_MASTER.parquet    MACRO_MONTHLY_MASTER.xlsx
MACRO_QUARTERLY_MASTER.csv  MACRO_QUARTERLY_MASTER.parquet  MACRO_QUARTERLY_MASTER.xlsx
```
These are regenerated by `l0_macro_backfill` after each run and tracked in Git.

### Macro Config
- Variable definitions: `config/macro_variables_ssot.yaml` (40 variables)
- Extractor registry: `airflow/dags/extractors/registry.py`
- Upsert service: `airflow/dags/services/upsert_service.py` (`FrequencyRoutedUpsertService`)
- Validation pipeline: `airflow/dags/validators/`

### Macro DAG Details

#### DAG 3: `l0_macro_backfill.py` (`core_l0_03_macro_backfill`)
- **Schedule**: `0 6 * * 0` (Sunday 6:00 UTC) / Manual
- **Decision branch**: Seeds exist + DB empty -> restore; otherwise -> full extraction
- **Extraction**: 7 sources, all variables, start_date configurable (default: 2015-01-01)
- **Validation**: Schema, range, completeness, leakage checks
- **Export**: Generates 9 MASTER files (3 frequencies x 3 formats)
- Flow: `health_check -> decide -> [restore_from_seeds | extract -> validate -> upsert] -> merge -> export_files -> post_validation -> report`
- **Contract**: CTR-L0-BACKFILL-002

```bash
# Auto-decide: restore from seeds if DB empty, else full extraction
airflow dags trigger core_l0_03_macro_backfill

# Force full extraction (ignore seeds)
airflow dags trigger core_l0_03_macro_backfill --conf '{"force_extract": true}'

# Custom date range
airflow dags trigger core_l0_03_macro_backfill --conf '{"start_date": "2020-01-01"}'
```

#### DAG 4: `l0_macro_update.py` (`core_l0_04_macro_update`)
- **Schedule**: `0 13-17 * * 1-5` (hourly 8:00-12:00 COT, Mon-Fri = 5 runs/day)
- **Strategy**: Always rewrites last 15 records per variable (no change detection)
- **Circuit breakers**: Per-source failure protection
- **`is_complete` flag**: Updated for critical variables after each run
- **Daily summary**: Generated at last run of day (12:00 COT)
- Flow: `health_check -> check_market_hours -> extract_all_sources -> upsert_all -> update_is_complete -> log_metrics -> daily_summary`
- **Contract**: CTR-L0-UPDATE-002

```bash
# Normal run (respects market hours)
airflow dags trigger core_l0_04_macro_update

# Force run outside market hours
airflow dags trigger core_l0_04_macro_update --conf '{"force_run": true}'

# Skip a problematic source
airflow dags trigger core_l0_04_macro_update --conf '{"skip_sources": ["dane"]}'
```

---

## DO NOT

### OHLCV
- Do NOT store OHLCV timestamps in UTC -- always convert to America/Bogota before insert
- Do NOT fetch BRL with `timezone=America/Bogota` from TwelveData (incomplete data)
- Do NOT rename `usdcop_m5_ohlcv` table -- use `symbol` column for multi-pair
- Do NOT modify seed parquets manually -- always regenerate via `build_unified_fx_seed.py`
- Do NOT assume all pairs have the same trading calendar (BRL has different holidays)
- Do NOT use `tz_localize` on timestamps that already have a timezone -- use `tz_convert`

### Macro
- Do NOT bypass the `FrequencyRoutedUpsertService` -- it routes to the correct table by frequency
- Do NOT skip `is_complete` flag update -- downstream L2 uses it to filter complete rows
- Do NOT extract macro data outside market hours without `force_run: true`
- Do NOT delete the 9 MASTER files in `04_cleaning/output/` -- they are the macro seed backup
