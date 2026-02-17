# Plan de Regeneración Completa - Datos Macro

## Diagnóstico (28 Enero 2026)

### Estado Actual por Variable

| Variable | Última Fecha | Fuente | Problema |
|----------|--------------|--------|----------|
| POLR_PRIME_RATE_USA_D_PRIME | 2026-01-26 | FRED DPRIME | API tiene retraso T+1 |
| FINC_BOND_YIELD2Y_USA_D_DGS2 | 2026-01-27 | FRED DGS2 | API tiene retraso T+1 |
| FINC_BOND_YIELD5Y_COL_D_COL5Y | 2026-01-27 | Investing | Falta integrar HPC |
| FINC_BOND_YIELD10Y_COL_D_COL10Y | 2026-01-27 | Investing | Falta integrar HPC |
| EQTY_INDEX_COLCAP_COL_D_COLCAP | 2026-01-27 | Investing | Falta integrar HPC |
| FINC_RATE_IBR_OVERNIGHT_COL_D_IBR | **2026-01-28** | SUAMECA REST | ✅ OK |
| POLR_POLICY_RATE_COL_M_TPM | **2026-01-28** | SUAMECA REST | ✅ OK |

### Causas Identificadas

1. **SUAMECA PRIME (220001)**: REST API devuelve `[]` - serie no disponible via API
2. **FRED APIs (DGS2, DPRIME)**: Retraso normal T+1 en publicación
3. **Investing.com**: Datos disponibles pero no integrados al HPC

---

## Plan de Ejecución

### Fase 1: Scraping Actualizado de Todas las Fuentes

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCRAPING PARALELO                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  FRED    │  │ SUAMECA  │  │   BCRP   │  │INVESTING │        │
│  │ 10 vars  │  │ IBR+TPM  │  │   EMBI   │  │ 12 vars  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       ▼             ▼             ▼             ▼               │
│  01_sources/   01_sources/   01_sources/   02_scrapers/        │
│  (10 CSVs)     (2 CSVs)      (1 CSV)       storage/HPC         │
└─────────────────────────────────────────────────────────────────┘
```

### Fase 2: Actualización HPC con Datos de Hoy

```
┌─────────────────────────────────────────────────────────────────┐
│                 ACTUALIZACIÓN HPC                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Investing.com API ──► COL5Y, COL10Y, COLCAP, DXY, VIX, etc.   │
│                        (datos del 28 enero)                     │
│                              │                                   │
│                              ▼                                   │
│                    datos_diarios_hpc.csv                        │
│                              │                                   │
│                              ▼                                   │
│                    Merge con IBR, TPM, EMBI, PRIME              │
└─────────────────────────────────────────────────────────────────┘
```

### Fase 3: Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE MACRO                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. run_fusion.py                                               │
│     └─► DATASET_MACRO_DAILY.csv (1566 filas histórico)         │
│                                                                  │
│  2. run_clean.py                                                │
│     └─► Merge histórico + HPC reciente                         │
│     └─► MACRO_DAILY_CLEAN.csv (1587+ filas)                    │
│                                                                  │
│  3. generate_master_consolidated.py                             │
│     └─► 9 archivos MASTER (CSV + XLSX + Parquet)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Limitaciones Conocidas

| Variable | Máxima Fecha Posible | Razón |
|----------|---------------------|-------|
| PRIME | 2026-01-26 | FRED API T+2 delay |
| DGS2 | 2026-01-27 | FRED API T+1 delay |
| COL5Y, COL10Y, COLCAP | 2026-01-28 | Investing.com actualizado |
| IBR, TPM | 2026-01-28 | SUAMECA REST API funciona |
| EMBI | 2026-01-27 | BCRP actualizado ayer |

---

## Archivos a Generar

### 9 Archivos Master

1. `MACRO_DAILY_MASTER.csv` (~280 KB)
2. `MACRO_DAILY_MASTER.xlsx` (~190 KB)
3. `MACRO_DAILY_MASTER.parquet` (~140 KB)
4. `MACRO_MONTHLY_MASTER.csv` (~16 KB)
5. `MACRO_MONTHLY_MASTER.xlsx` (~14 KB)
6. `MACRO_MONTHLY_MASTER.parquet` (~19 KB)
7. `MACRO_QUARTERLY_MASTER.csv` (~3 KB)
8. `MACRO_QUARTERLY_MASTER.xlsx` (~7 KB)
9. `MACRO_QUARTERLY_MASTER.parquet` (~5 KB)

---

## Comandos de Ejecución

```bash
# Fase 1: Scraping
python scripts/full_macro_regeneration.py --start 2020-01-01
python scripts/scraper_suameca_api.py --start 2020-01-01

# Fase 2: Actualizar HPC con Investing.com
python scripts/update_hpc_investing.py  # NUEVO

# Fase 3: Pipeline
python data/pipeline/03_fusion/run_fusion.py
python data/pipeline/04_cleaning/run_clean.py
python scripts/generate_master_consolidated.py
```
