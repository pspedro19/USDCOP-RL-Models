---
kind: as-built
status: IMPLEMENTED
contract: CTR-L0-4TABLE-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - airflow/dags/l0_ohlcv_backfill.py
  - airflow/dags/l0_ohlcv_realtime.py
  - airflow/dags/l0_macro_backfill.py
  - airflow/dags/l0_macro_update.py
  - airflow/dags/l0_seed_backup.py
  - scripts/data/build_unified_fx_seed.py
  - config/macro_variables_ssot.yaml
---
# SDD Spec: Capa L0 (referencia)

> **Responsibility**: inventario de DAGs L0, schemas de tabla, catálogo de extractores, contratos
> de archivo y comandos. Las **invariantes** (regla de oro de timezone, UPSERT, DO-NOTs) viven en
> `../../rules/data-governance.md`, que se auto-carga.

---

## 1. Inventario de DAGs L0

| # | DAG ID | Archivo | Schedule | Alcance |
|---|--------|---------|----------|---------|
| 1 | `core_l0_01_ohlcv_backfill` | `l0_ohlcv_backfill.py` | Manual | COP + MXN + BRL |
| 2 | `core_l0_02_ohlcv_realtime` | `l0_ohlcv_realtime.py` | `*/5 13-17 * * 1-5` | COP + MXN + BRL |
| 3 | `core_l0_03_macro_backfill` | `l0_macro_backfill.py` | Dom 6:00 UTC / manual | Historia completa |
| 4 | `core_l0_04_macro_update` | `l0_macro_update.py` | `0 13-17 * * 1-5` | 40 vars, 7 fuentes |
| 5 | `core_l0_05_seed_backup` | `l0_seed_backup.py` | `0 20 * * 1-5` | OHLCV + macro + features |

### Estrategia de backup (dos niveles)

- **Automático entre semana** (#5): vuelca OHLCV + macro a `data/backups/seeds/*.parquet` cada
  día hábil tras el cierre. Son los backups **más frescos**, los primeros que leen los init-scripts.
- **Export en backfill**: OHLCV (#1) exporta seeds a `seeds/latest/`; macro (#3) exporta 9 MASTER.
  Van a Git como baseline de restore.
- **Feature data** (#5, desde 2026-07-05): 12 tablas derivadas (news, analysis, `forecast_h5_*`,
  `asset_daily_ohlcv`, macro mensual/trimestral) → `data/backups/features/*.parquet`
  (CTR-L0-FEATURE-BACKUP-001). Best-effort: nunca bloquea la ruta crítica.
- **Prioridad de restore**: parquet diario → seed en Git LFS → MinIO → CSV legacy.

---

## 2. OHLCV

### Pares soportados

| Par | Símbolo | Seed | Fuente | Rango |
|-----|---------|------|--------|-------|
| USD/COP | `USD/COP` | `usdcop_m5_ohlcv.parquet` | TwelveData | 3.000-6.000 |
| USD/MXN | `USD/MXN` | `usdmxn_m5_ohlcv.parquet` | Dukascopy | 10-30 |
| USD/BRL | `USD/BRL` | `usdbrl_m5_ohlcv.parquet` | TwelveData | 3-8 |

### Conversión de timezone

| Fuente | TZ cruda | Conversión a COT |
|--------|----------|------------------|
| TwelveData (COP/MXN) | `America/Bogota` | ya está, usar directo |
| TwelveData (BRL) | `UTC` | `tz_localize('UTC').tz_convert('America/Bogota')` |
| Dukascopy | UTC naive (epoch ms) | igual que BRL |
| PostgreSQL TIMESTAMPTZ | UTC interno | `AT TIME ZONE 'America/Bogota'` |

### Schema de tabla

```sql
time TIMESTAMPTZ, symbol VARCHAR(20), open/high/low/close DOUBLE PRECISION,
volume DOUBLE PRECISION, source VARCHAR(50)
PRIMARY KEY (time, symbol)      -- multi-par nativo
```

### UPSERT (obligatorio en todo DAG OHLCV)

```sql
INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source)
VALUES %s
ON CONFLICT (time, symbol) DO UPDATE SET
    volume = EXCLUDED.volume, source = EXCLUDED.source, updated_at = NOW()
```

### Validación de seeds (`build_unified_fx_seed.py`)

Horas en `[8..12]` COT · sin fines de semana · sin `(time, symbol)` duplicados · sin NaN en OHLC ·
`high >= low/open/close` · precios en rango por par · mediana ≈ 60 barras/día.

Regenerar: `python scripts/data/build_unified_fx_seed.py`

---

## 3. Macro — diseño de 4 tablas (CTR-L0-4TABLE-001)

| Tabla | Frecuencia | Vars | Ejemplos |
|-------|------------|------|----------|
| `macro_indicators_daily` | Diaria | 18 | DXY, VIX, UST10Y, IBR, TPM, EMBI |
| `macro_indicators_monthly` | Mensual | 18 | FEDFUNDS, CPI, desempleo |
| `macro_indicators_quarterly` | Trimestral | 4 | GDP, BOP |

**Extractores (7)**: FRED · Investing.com · BanRep · BCRP · Fedesarrollo · DANE · BanRep BOP.
**Críticas** (deben tener `is_complete = true`): `dxy`, `vix`, `ust10y`, `ust2y`, `ibr`, `tpm`,
`embi_col`.

**Seeds macro**: 9 archivos MASTER en `data/pipeline/04_cleaning/output/` (3 frecuencias × csv/
parquet/xlsx), regenerados por `l0_macro_backfill` y trackeados en Git.

---

## 4. Comandos

```bash
# OHLCV
airflow dags trigger core_l0_01_ohlcv_backfill
airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"symbols": ["USD/MXN"]}'
airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"force_backfill": true}'

# Macro
airflow dags trigger core_l0_03_macro_backfill
airflow dags trigger core_l0_03_macro_backfill --conf '{"force_extract": true}'
airflow dags trigger core_l0_04_macro_update --conf '{"force_run": true}'
airflow dags trigger core_l0_04_macro_update --conf '{"skip_sources": ["dane"]}'
```

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Invariantes L0 (auto-cargadas) | `../../rules/data-governance.md` |
| Umbrales de frescura | `../../rules/data-freshness.md` |
| Runbooks de recuperación | `../operations/freshness-recovery.md` |
| Session/tz por activo | `../assets/_asbuilt-implementation.md` |
